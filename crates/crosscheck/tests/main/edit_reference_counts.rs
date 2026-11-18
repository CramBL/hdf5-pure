#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::LocationToken;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileAccessProperties, FileLocking, MemoryStrategy};
use hdf5_pure_format::__private::{AccessMode, ObjectHeader, ObjectHeaderPrefix};
use rstest::rstest;
use test_util::bytes;
use test_util::object_header::{Message, MessageType, v2};
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

#[rstest]
#[case::increase(Edit::Increase)]
#[case::decrease(Edit::Decrease)]
#[case::balanced(Edit::Balanced)]
#[case::removed_dataset_user(Edit::RemovedUser)]
#[case::retained_name(Edit::RetainName)]
#[case::anonymous(Edit::Anonymous)]
fn public_copy_and_delete_commits_preserve_native_count_owners(
    #[case] edit: Edit,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("native-public-count-owners.h5");
    write_native(&path, edit);
    let original_count = match edit {
        Edit::Decrease => 3,
        Edit::Anonymous
        | Edit::Balanced
        | Edit::Increase
        | Edit::RemovedUser
        | Edit::RetainName => 2,
    };
    let identity = assert_native(
        &path,
        edit.original_datasets(),
        edit.original_names(),
        original_count,
    );
    let original = std::fs::read(&path).unwrap();
    let reader = File::open(&path).unwrap();
    let widths = (
        reader.superblock().offset_size.get(),
        reader.superblock().length_size.get(),
    );
    let mut types = reader.root().named_datatypes().unwrap();
    types.sort();
    assert_eq!(types, edit.original_names());
    reader.close().unwrap();
    let headers = bytes::signature_offsets(&original, v2::SIGNATURE)
        .into_iter()
        .filter(|&at| {
            let header =
                ObjectHeader::parse(&original, AccessMode::ReadOnly, at, widths.0, widths.1)
                    .unwrap();
            header
                .messages
                .iter()
                .any(|message| message.msg_type == hdf5_pure::MessageType::DATATYPE)
                && !header
                    .messages
                    .iter()
                    .any(|message| message.msg_type == hdf5_pure::MessageType::DATA_LAYOUT)
        })
        .collect::<Vec<_>>();
    let [address] = headers.as_slice() else {
        panic!("expected one committed type header: {headers:?}");
    };
    let address = *address;
    let prefix = ObjectHeaderPrefix::parse(&original[address..]).unwrap();
    let end = address + prefix.len + usize::try_from(prefix.chunk0_size).unwrap() + 4;
    let padded = end.next_multiple_of(ALIGNMENT);
    assert_eq!(address % ALIGNMENT, 0);
    assert!(end < padded);
    assert!(original[end..padded].iter().all(|&byte| byte == 0));
    let mut expected = original.clone();
    v2::replace_message(
        &mut expected,
        address,
        MessageType::OBJECT_REFERENCE_COUNT,
        &Message::reference_count(edit.count()).data,
    );
    let pure = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_memory_strategy(memory),
    )
    .unwrap();
    for index in 0..edit.copies() {
        pure.copy("first", &format!("copy{index}")).unwrap();
    }
    if !matches!(edit, Edit::Anonymous | Edit::RetainName) {
        pure.root().delete("type").unwrap();
    }
    if matches!(edit, Edit::Decrease) {
        pure.root().delete("alias").unwrap();
    }
    if matches!(edit, Edit::RemovedUser) {
        pure.root().delete("first").unwrap();
    }
    pure.commit().unwrap();
    assert!(!pure.has_staged_edits());
    let after = std::fs::read(&path).unwrap();
    assert_eq!(&after[address..padded], &expected[address..padded]);
    assert!(
        pure.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| {
                section.end_address().get() <= address as u64
                    || section.address().get() >= padded as u64
            })
    );
    pure.close().unwrap();
    let names = edit.datasets();
    assert_eq!(
        assert_native(
            &path,
            &names,
            edit.names(),
            usize::try_from(edit.count()).unwrap()
        ),
        identity
    );
    let reopened = File::open(&path).unwrap();
    let mut datasets = reopened.root().datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, names);
    assert_eq!(reopened.root().named_datatypes().unwrap(), edit.names());
    for name in names {
        assert_eq!(
            reopened.dataset(&name).unwrap().read_i32().unwrap(),
            [7, 11]
        );
    }
    reopened.close().unwrap();
}

#[rstest]
#[case::version_1(true)]
#[case::version_2(false)]
fn deleting_a_group_alias_updates_only_the_original_count_and_checksum(#[case] version_1: bool) {
    let path = test_util::temp::temp_path("native-group-alias-count.h5");
    let native = if version_1 {
        file::libhdf5_create_earliest(&path)
    } else {
        file::libhdf5_create_v110(&path)
    };
    let group = native.create_group("g").unwrap();
    group
        .new_attr::<i64>()
        .create("value")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    drop(group);
    native.link_hard("g", "alias").unwrap();
    let info = native.loc_info_by_name("g").unwrap();
    assert_eq!(info.num_links, 2);
    assert_eq!(native.loc_info_by_name("alias").unwrap().token, info.token);
    let address = usize::try_from(c_object::header_address(&native, "g")).unwrap();
    native.close().unwrap();

    let mut expected = std::fs::read(&path).unwrap();
    let end = if version_1 {
        const COUNT_OFFSET: usize = 4;
        const CHUNK_SIZE_OFFSET: usize = 8;
        const PREFIX_LEN: usize = 16;
        let count = address + COUNT_OFFSET;
        let size = address + CHUNK_SIZE_OFFSET;
        assert_eq!(expected[address], 1);
        assert_eq!(&expected[count..count + 4], &2_u32.to_le_bytes());
        let chunk_size = u32::from_le_bytes(expected[size..size + 4].try_into().unwrap());
        expected[count..count + 4].copy_from_slice(&1_u32.to_le_bytes());
        address + PREFIX_LEN + usize::try_from(chunk_size).unwrap()
    } else {
        let prefix = ObjectHeaderPrefix::parse(&expected[address..]).unwrap();
        let end = address + prefix.len + usize::try_from(prefix.chunk0_size).unwrap() + 4;
        v2::replace_message(
            &mut expected,
            address,
            MessageType::OBJECT_REFERENCE_COUNT,
            &Message::reference_count(1).data,
        );
        end
    };

    let pure = File::open_rw(&path).unwrap();
    pure.root().delete("alias").unwrap();
    pure.commit().unwrap();
    pure.close().unwrap();

    let stored = std::fs::read(&path).unwrap();
    assert_eq!(&stored[address..end], &expected[address..end]);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.member_names().unwrap(), ["g"]);
    let remaining = native.loc_info_by_name("g").unwrap();
    assert_eq!(remaining.num_links, 1);
    assert_eq!(remaining.token, info.token);
    assert_eq!(
        native
            .group("g")
            .unwrap()
            .attr("value")
            .unwrap()
            .read_scalar::<i64>()
            .unwrap(),
        7
    );
    native.close().unwrap();
}

#[rstest]
#[case::compact(false)]
#[case::dense(true)]
fn copying_group_attributes_preserves_their_native_committed_datatype(#[case] dense: bool) {
    let attrs = if dense { 9 } else { 1 };
    let path = test_util::temp::temp_path("native-copied-committed-attributes.h5");
    let native = file::libhdf5_create_v18(&path);
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("mytype", &datatype).unwrap();
    let group = native.create_group("source").unwrap();
    for index in 0..attrs {
        group
            .new_attr_builder()
            .empty_as(&datatype)
            .shape([1])
            .create(format!("attr{index}").as_str())
            .unwrap()
            .write(&[i32::try_from(index).unwrap()])
            .unwrap();
    }
    let info = native.loc_info_by_name("mytype").unwrap();
    assert_eq!(info.num_links, attrs + 1);
    drop(group);
    drop(datatype);
    native.close().unwrap();
    let original = std::fs::read(&path).unwrap();
    assert_eq!(test_util::fractal_heap::has_fractal_heap(&original), dense);

    let pure = File::open_rw(&path).unwrap();
    pure.copy("source", "copy").unwrap();
    pure.root().delete("mytype").unwrap();
    pure.commit().unwrap();
    pure.close().unwrap();

    let native = file::libhdf5_open(&path);
    let mut names = native.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["copy", "source"]);
    for name in ["source", "copy"] {
        let group = native.group(name).unwrap();
        assert_eq!(group.attr_names().unwrap().len(), attrs);
        for index in 0..attrs {
            let attr = group.attr(&format!("attr{index}")).unwrap();
            assert_eq!(
                attr.read_raw::<i32>().unwrap(),
                [i32::try_from(index).unwrap()]
            );
            let datatype = attr.dtype().unwrap();
            assert!(datatype.is_committed());
            let remaining = datatype.as_location().unwrap().loc_info().unwrap();
            assert_eq!(remaining.token, info.token);
            assert_eq!(remaining.num_links, 2 * attrs);
        }
    }
    native.close().unwrap();
}

#[derive(Clone, Copy)]
enum Edit {
    Anonymous,
    Balanced,
    Decrease,
    Increase,
    RemovedUser,
    RetainName,
}

impl Edit {
    fn copies(self) -> usize {
        if matches!(self, Self::Increase) { 2 } else { 1 }
    }

    fn count(self) -> u32 {
        match self {
            Self::Anonymous | Self::Increase | Self::RetainName => 3,
            Self::Balanced | Self::Decrease | Self::RemovedUser => 2,
        }
    }

    fn original_datasets(self) -> &'static [&'static str] {
        if matches!(self, Self::Anonymous) {
            &["first", "second"]
        } else {
            &["first"]
        }
    }

    fn original_names(self) -> &'static [&'static str] {
        match self {
            Self::Anonymous => &[],
            Self::Decrease => &["alias", "type"],
            Self::Balanced | Self::Increase | Self::RemovedUser | Self::RetainName => &["type"],
        }
    }

    fn names(self) -> &'static [&'static str] {
        if matches!(self, Self::RetainName) {
            &["type"]
        } else {
            &[]
        }
    }

    fn datasets(self) -> Vec<String> {
        let mut names = (0..self.copies())
            .map(|index| format!("copy{index}"))
            .collect::<Vec<_>>();
        if !matches!(self, Self::RemovedUser) {
            names.push("first".into());
        }
        if matches!(self, Self::Anonymous) {
            names.push("second".into());
        }
        names
    }
}

fn write_native(path: &Path, edit: Edit) {
    let native = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties
                .libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
                .alignment(1, ALIGNMENT as u64)
        })
        .create(path)
        .unwrap();
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("type", &datatype).unwrap();
    for name in edit.original_datasets() {
        let dataset = native
            .new_dataset_builder()
            .empty_as(&datatype)
            .shape([2])
            .create(*name)
            .unwrap();
        dataset.write(&[7_i32, 11]).unwrap();
    }
    if matches!(edit, Edit::Anonymous) {
        native.unlink("type").unwrap();
    }
    if matches!(edit, Edit::Decrease) {
        native.link_hard("type", "alias").unwrap();
    }
    drop(datatype);
    native.close().unwrap();
}

fn assert_native(
    path: &Path,
    datasets: &[impl AsRef<str>],
    names: &[&str],
    count: usize,
) -> LocationToken {
    let native = file::libhdf5_open(path);
    let mut members = native.member_names().unwrap();
    members.sort();
    let mut expected = names
        .iter()
        .map(|name| (*name).to_owned())
        .chain(datasets.iter().map(|name| name.as_ref().to_owned()))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(members, expected);
    let mut identity = None;
    for name in datasets {
        let dataset = native.dataset(name.as_ref()).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), [7, 11]);
        assert!(dataset.attr_names().unwrap().is_empty());
        let datatype = dataset.dtype().unwrap();
        assert!(datatype.is_committed());
        let info = datatype.as_location().unwrap().loc_info().unwrap();
        assert_eq!(info.num_links, count);
        assert_eq!(info.token, *identity.get_or_insert(info.token));
    }
    for name in names {
        let info = native.loc_info_by_name(name).unwrap();
        assert_eq!(Some(info.token), identity);
        assert_eq!(info.num_links, count);
    }
    native.close().unwrap();
    identity.unwrap()
}

const ALIGNMENT: usize = 64;
