use std::collections::HashMap;
use std::path::Path;

use rstest::rstest;
use test_util::attribute;
use test_util::bytes;
use test_util::disk_log;
use test_util::object_header::v1;
use test_util::object_header::{Message, MessageType};
use test_util::superblock::v0;
use test_util::symbol_table;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::creation_order::Indexing;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

use crate::AttrValue;
use crate::Datatype;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FixedPointType;
use crate::MemoryStrategy;
use crate::access_mode::AccessMode;
use crate::attribute_info::AttributeInfoMessage;
use crate::edit::WriteEngine;
use crate::edit::group_inspection;
use crate::edit::test_util as edit_test_util;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::direct_dense(Storage::Dense, Destination::Direct)]
#[case::ancestor_dense(Storage::Dense, Destination::Descendant)]
#[case::direct_empty(Storage::Empty, Destination::Direct)]
#[case::ancestor_empty(Storage::Empty, Destination::Descendant)]
fn conversion_rejection_preserves_bytes_values_and_staged_inputs(
    #[case] storage: Storage,
    #[case] destination: Destination,
    #[values(false, true)] source_only: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy-dense.h5");
    write_fixture(&path, storage);
    let before = std::fs::read(&path).unwrap();
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.group("g").unwrap().attrs().unwrap(), storage.attrs());
    let original_header = reader.group("g").unwrap().header_address().unwrap();
    let geometry = reader.superblock().clone();
    assert_eq!(reader.file_size(), u64::try_from(before.len()).unwrap());
    drop(reader);
    let mut engine = if source_only {
        WriteEngine::open_source_only(&path).unwrap()
    } else {
        edit_test_util::open_engine(&path)
    };
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("valid"), i32_dataset(&[23]))
        .unwrap();
    let target = match destination {
        Destination::Direct => "g/d",
        Destination::Descendant => "g/child/d",
    };
    engine
        .stage_created_dataset(&ObjectPathBuf::parse(target), i32_dataset(&[29]))
        .unwrap();
    engine
        .create_group(&ObjectPathBuf::parse("valid_group"))
        .unwrap();
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("original"), i32_dataset(&[53, 59]))
        .unwrap();
    let staged = engine.staged.snapshot();
    let datasets = engine.staged.datasets().as_ptr();
    let groups = engine.staged.groups().as_ptr();
    let writes = engine.staged.writes().as_ptr();
    let free = engine
        .space
        .admitted_snapshot()
        .sections()
        .collect::<Vec<_>>();
    let size = engine.image.len();
    let superblock = engine.context.superblock().clone();
    disk_log::start();
    let error = engine.commit().unwrap_err();
    let io = disk_log::finish();
    assert_conversion_error(error);
    assert!(io.events.is_empty(), "{io:?}");
    assert_eq!(engine.disk_bytes().unwrap(), before);
    assert_eq!(engine.image.len(), size);
    assert_eq!(engine.context.superblock(), &superblock);
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        free
    );
    assert!(engine.staged.snapshot() == staged);
    assert_eq!(engine.staged.datasets().as_ptr(), datasets);
    assert_eq!(engine.staged.groups().as_ptr(), groups);
    assert_eq!(engine.staged.writes().as_ptr(), writes);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        u64::try_from(before.len()).unwrap()
    );
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.superblock(), &geometry);
    assert_eq!(
        reader.group("g").unwrap().header_address().unwrap(),
        original_header
    );
    assert_eq!(reader.group("g").unwrap().attrs().unwrap(), storage.attrs());
    assert_eq!(reader.root().datasets().unwrap(), ["original"]);
    assert_eq!(reader.root().groups().unwrap(), ["g"]);
    assert_eq!(
        reader.group("g").unwrap().datasets().unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(reader.group("g").unwrap().groups().unwrap(), ["child"]);
    assert_eq!(
        reader.dataset("original").unwrap().read_i32().unwrap(),
        [11, 13]
    );
    reader.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
fn public_commit_rejection_retains_both_staged_datasets(#[values(false, true)] source_only: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("public-rejection.h5");
    write_fixture(&path, Storage::Dense);
    let before = std::fs::read(&path).unwrap();
    let session = group_inspection::open_group_inspection(&path, source_only).unwrap();
    session
        .group("g")
        .unwrap()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[19]);
        })
        .unwrap();
    session
        .root()
        .create_dataset("valid", |builder| {
            builder.with_i32_data(&[23]);
        })
        .unwrap();
    assert_conversion_error(session.commit().unwrap_err());
    assert!(session.has_staged_edits());
    let error = session.dataset("g/d").unwrap().read_i32().unwrap_err();
    let Error::NotCommitted(pending) = error else {
        panic!("expected NotCommitted, got {error:?}")
    };
    assert_eq!(pending, "g/d");
    let error = session.dataset("valid").unwrap().read_i32().unwrap_err();
    let Error::NotCommitted(pending) = error else {
        panic!("expected NotCommitted, got {error:?}")
    };
    assert_eq!(pending, "valid");
    assert_eq!(
        session.group("g").unwrap().attrs().unwrap(),
        Storage::Dense.attrs()
    );
    assert_eq!(session.disk_bytes().unwrap(), before);
    assert_conversion_error(session.commit().unwrap_err());
    assert_eq!(session.disk_bytes().unwrap(), before);
    session.group("g").unwrap().delete("d").unwrap();
    session.commit().unwrap();
    assert!(!session.has_staged_edits());
    drop(session);
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.dataset("valid").unwrap().read_i32().unwrap(), [23]);
    assert_eq!(
        reader.group("g").unwrap().attrs().unwrap(),
        Storage::Dense.attrs()
    );
    assert_eq!(
        reader.group("g").unwrap().datasets().unwrap(),
        Vec::<String>::new()
    );
    reader.close().unwrap();
    directory.close().unwrap();
}

test_util_hdf5::hdf5_bundled_tests! {
    #[test]
    fn native_and_pure_readers_report_different_version_1_dense_attributes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native-visibility.h5");
        write_fixture(&path, Storage::Dense);
        let reference = file::libhdf5_open(&path);
        assert_eq!(reference.group("g").unwrap().attr_names().unwrap(), Vec::<String>::new());
        reference.close().unwrap();
        let reader = File::open(&path).unwrap();
        assert_eq!(reader.group("g").unwrap().attrs().unwrap(), Storage::Dense.attrs());
        reader.close().unwrap();
        directory.close().unwrap();
    }
}

#[rstest]
#[case::inline(Storage::Inline)]
#[case::dense_version_2(Storage::Modern)]
fn supported_attribute_storage_remains_editable(
    #[case] storage: Storage,
    #[values(false, true)] source_only: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("supported-conversion.h5");
    write_fixture(&path, storage);
    let session = group_inspection::open_group_inspection(&path, source_only).unwrap();
    session
        .group("g")
        .unwrap()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[31]);
        })
        .unwrap();
    session.commit().unwrap();
    session
        .group("g")
        .unwrap()
        .set_attr("b", AttrValue::I32(37))
        .unwrap();
    session.commit().unwrap();
    drop(session);
    let expected = HashMap::from([
        ("a".to_owned(), AttrValue::I32(7)),
        ("b".to_owned(), AttrValue::I32(37)),
    ]);
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.group("g").unwrap().attrs().unwrap(), expected);
    assert_eq!(reader.dataset("g/d").unwrap().read_i32().unwrap(), [31]);
    reader.close().unwrap();
    let reference = file::libhdf5_open(&path);
    let group = reference.group("g").unwrap();
    let mut names = group.attr_names().unwrap();
    names.sort();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(group.attr("a").unwrap().read_scalar::<i32>().unwrap(), 7);
    assert_eq!(group.attr("b").unwrap().read_scalar::<i32>().unwrap(), 37);
    assert_eq!(
        reference.dataset("g/d").unwrap().read_raw::<i32>().unwrap(),
        [31]
    );
    drop(group);
    reference.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::read_only(Unrelated::Read)]
#[case::root_link(Unrelated::RootLink)]
#[case::in_place_values(Unrelated::Values)]
fn an_unreconstructed_legacy_group_retains_its_dense_attributes(
    #[case] operation: Unrelated,
    #[values(false, true)] source_only: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("unrelated-group.h5");
    write_fixture(&path, Storage::Dense);
    let before = std::fs::read(&path).unwrap();
    let reader = File::open(&path).unwrap();
    let header = reader.group("g").unwrap().header_address().unwrap();
    assert_eq!(
        reader.group("g").unwrap().attrs().unwrap(),
        Storage::Dense.attrs()
    );
    reader.close().unwrap();
    let session = group_inspection::open_group_inspection(&path, source_only).unwrap();
    match operation {
        Unrelated::Read => {}
        Unrelated::RootLink => {
            session
                .root()
                .create_dataset("valid", |builder| {
                    builder.with_i32_data(&[41]);
                })
                .unwrap();
        }
        Unrelated::Values => session
            .dataset("original")
            .unwrap()
            .write_staged(|builder| {
                builder.with_i32_data(&[43, 47]);
            })
            .unwrap(),
    }
    session.commit().unwrap();
    drop(session);
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.group("g").unwrap().header_address().unwrap(), header);
    assert_eq!(
        reader.group("g").unwrap().attrs().unwrap(),
        Storage::Dense.attrs()
    );
    let values = if matches!(operation, Unrelated::Values) {
        [43, 47]
    } else {
        [11, 13]
    };
    assert_eq!(
        reader.dataset("original").unwrap().read_i32().unwrap(),
        values
    );
    if matches!(operation, Unrelated::RootLink) {
        assert_eq!(reader.dataset("valid").unwrap().read_i32().unwrap(), [41]);
    }
    if matches!(operation, Unrelated::Read) {
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
    reader.close().unwrap();
    directory.close().unwrap();
}

#[test]
fn a_version_0_superblock_rejects_bounded_open() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounded-legacy.h5");
    write_fixture(&path, Storage::Dense);
    let error = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}")
    };
    assert_eq!(
        reason,
        "bounded read-write access requires a latest-format file (v2/v3 superblock); \
         leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to fall back to \
         the whole-file mirror here"
    );
    directory.close().unwrap();
}

#[derive(Clone, Copy)]
enum Storage {
    Dense,
    Empty,
    Inline,
    Modern,
}

impl Storage {
    fn attrs(self) -> HashMap<String, AttrValue> {
        match self {
            Self::Empty => HashMap::new(),
            _ => HashMap::from([("a".to_owned(), AttrValue::I32(7))]),
        }
    }
}

#[derive(Clone, Copy)]
enum Destination {
    Descendant,
    Direct,
}

#[derive(Clone, Copy)]
enum Unrelated {
    Read,
    RootLink,
    Values,
}

fn assert_conversion_error(error: Error) {
    let Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}");
    };
    assert_eq!(
        reason,
        "a version 1 group's Attribute Info message cannot be preserved by conversion"
    );
}

fn i32_dataset(values: &[i32]) -> DatasetBuilder {
    let mut builder = DatasetBuilder::new("");
    builder.with_i32_data(values);
    builder
}

/// Writes native groups and rewrites legacy dense variants as version 1 headers.
fn write_fixture(path: &Path, storage: Storage) {
    let native = if matches!(storage, Storage::Modern) {
        file::libhdf5_create_v18(path)
    } else {
        file::libhdf5_create_earliest(path)
    };
    let group = if matches!(storage, Storage::Inline) {
        native.create_group("g").unwrap()
    } else {
        native
            .create_group_builder()
            .with_gcpl(|properties| {
                properties
                    .attr_creation_order(Indexing::No.attr_order())
                    .attr_phase_change(0, 0)
            })
            .create("g")
            .unwrap()
    };
    group
        .new_attr::<i32>()
        .shape(())
        .create("a")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    native
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .with_dcpl(|properties| properties.attr_creation_order(Indexing::No.attr_order()))
        .shape([2])
        .create("original")
        .unwrap()
        .write_raw(&[11, 13])
        .unwrap();
    group.create_group("child").unwrap();
    let address = usize::try_from(c_object::header_address(&native, "g")).unwrap();
    drop(group);
    native.close().unwrap();
    let mut image = std::fs::read(path).unwrap();
    let reader = File::open(path).unwrap();
    assert_eq!(
        reader.dataset("original").unwrap().datatype().unwrap(),
        Datatype::FixedPoint(FixedPointType::i32())
    );
    let header =
        ObjectHeader::parse_with_widths(&image, AccessMode::ReadOnly, address, reader.widths())
            .unwrap();
    let widths = reader.widths();
    reader.close().unwrap();
    if matches!(storage, Storage::Inline) {
        assert_eq!(header.version, HEADER_VERSION_ONE);
        assert!(
            header
                .messages
                .iter()
                .any(|message| message.msg_type == crate::MessageType::SYMBOL_TABLE)
        );
        assert!(
            header
                .messages
                .iter()
                .any(|message| message.msg_type == crate::MessageType::ATTRIBUTE)
        );
        assert!(
            header
                .messages
                .iter()
                .all(|message| message.msg_type != crate::MessageType::ATTRIBUTE_INFO)
        );
        return;
    }
    assert_eq!(header.version, HEADER_VERSION_TWO);
    let info = header
        .messages
        .iter()
        .find(|message| message.msg_type == crate::MessageType::ATTRIBUTE_INFO)
        .unwrap();
    let info = AttributeInfoMessage::parse_with_offset_width(&info.data, widths.offsets).unwrap();
    assert_eq!(info.max_creation_index, Some(1));
    assert!(info.fractal_heap_address.is_some());
    assert!(info.btree_name_index_address.is_some());
    if matches!(storage, Storage::Modern) {
        return;
    }
    let superblock = v0::Fields::read(&image, 0);
    let widths = superblock.widths;
    let messages = [MessageType::SYMBOL_TABLE, MessageType::ATTRIBUTE_INFO].map(|msg_type| {
        let message = header
            .messages
            .iter()
            .find(|message| message.msg_type.to_u16() == msg_type.0)
            .unwrap();
        let mut body =
            if matches!(storage, Storage::Empty) && msg_type == MessageType::ATTRIBUTE_INFO {
                attribute::tracked_info(1, false, widths)
            } else {
                message.data.clone()
            };
        body.resize(body.len().next_multiple_of(v1::MESSAGE_ALIGNMENT), 0);
        Message::new(msg_type, &body)
    });
    let entries = bytes::signature_offsets(&image, symbol_table::SIGNATURE)
        .into_iter()
        .flat_map(|node| {
            let count = usize::from(bytes::u16_at(&image, node + symbol_table::SYMBOL_COUNT_AT));
            (0..count).map(move |index| {
                node + symbol_table::NODE_PREFIX
                    + index * symbol_table::Entry::encoded_len(widths)
                    + widths.length
            })
        })
        .filter(|at| bytes::uint_at(&image, *at, widths.offset) == u64::try_from(address).unwrap())
        .collect::<Vec<_>>();
    let [address_at] = entries.as_slice() else {
        panic!("expected one entry referring to g, got {entries:?}")
    };
    let address_at = *address_at;
    let at = image.len().next_multiple_of(v1::MESSAGE_ALIGNMENT);
    image.resize(at, 0);
    image.extend_from_slice(&v1::Header::new().messages(messages).build());
    bytes::set_uint_at(
        &mut image,
        address_at,
        widths.offset,
        u64::try_from(at).unwrap(),
    );
    let eof = u64::try_from(image.len()).unwrap();
    bytes::set_uint_at(&mut image, superblock.eof_address_at, widths.offset, eof);
    std::fs::write(path, image).unwrap();
}

/// The object-header version defined in "Version 1 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
const HEADER_VERSION_ONE: u8 = 1;

/// The object-header version defined in "Version 2 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
const HEADER_VERSION_TWO: u8 = 2;
