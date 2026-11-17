#![cfg(feature = "__hdf5-1.10")]

use std::collections::HashMap;
use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5::types::VarLenAscii;
use hdf5_pure::{
    AttrValue, EditBacking, File, FileAccessProperties, FileLocking, FileSpacePageSize,
    FileSpaceStrategy,
};
use hdf5_pure_core::{LengthWidth, OffsetWidth};
use hdf5_pure_format::__private::{AccessMode, ObjectHeader, ObjectHeaderPrefix};
use rstest::rstest;
use test_util::global_heap;
use test_util::object_header::{MessageType, v2};
use test_util::widths::Widths;
use test_util_hdf5::__private::AnonymousCountSource;
use test_util_hdf5::file;
use test_util_hdf5::file_space::persisted::PersistedStrategy;
use test_util_hdf5::global_heap::attributes;
use test_util_hdf5::global_heap_fixture::members::{
    Buffering, LayoutFixture, ParentImage, Terminal,
};

#[rstest]
#[case::explicit(Terminal::Explicit)]
#[case::full(Terminal::Full)]
#[case::implied(Terminal::Implied)]
fn native_readers_observe_grouped_public_member_removals(
    #[case] terminal: Terminal,
    #[values(1, 3)] parents: usize,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
) {
    let path = test_util::temp::temp_path("native-grouped-public-members.h5");
    let fixture = LayoutFixture::write(&path, terminal, parents);
    let pure = File::open_rw_with_options(&path, buffering.properties(backing)).unwrap();
    assert_eq!(pure.edit_backing(), Some(backing));
    fixture.remove(&pure);
    pure.commit().unwrap();
    pure.close().unwrap();
    fixture.assert_compacted(&path);
    let native = file::libhdf5_open(&path);
    assert_eq!(fixture.values.len(), parents);
    for (index, expected) in fixture.values.iter().enumerate() {
        let dataset = native.dataset(&format!("keep_{index}")).unwrap();
        let values = dataset.read_raw::<VarLenAscii>().unwrap();
        assert_eq!(
            values.iter().map(VarLenAscii::as_str).collect::<Vec<_>>(),
            *expected
        );
        drop(dataset);
        let dataset = native.dataset(&format!("value_{index}")).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), [17]);
        assert_eq!(dataset.attr_names().unwrap(), Vec::<String>::new());
        drop(dataset);
    }
    native.close().unwrap();
}

#[rstest]
#[case::fsm_aggr(PersistedStrategy::FsmAggr)]
#[case::page(PersistedStrategy::Page)]
fn native_source_members_compact_through_public_persistent_sessions(
    #[case] strategy: PersistedStrategy,
    #[values(Terminal::Explicit, Terminal::Full, Terminal::Implied)] terminal: Terminal,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
) {
    let path = test_util::temp::temp_path("native-source-public-members.h5");
    let retained_value = write_native(&path, strategy, terminal);
    let original = std::fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    let [range] = ranges.as_slice() else {
        panic!("expected one native shared parent: {ranges:?}");
    };
    let parent = ParentImage::read(&original, range.start);
    assert_eq!(parent.indices().len(), 3);
    let first = parent.index_for(b"left");
    let second = parent.index_for(b"middle");
    let retained = parent.index_for(retained_value.as_bytes());
    assert_eq!(parent.tail_bytes(), terminal.bytes());
    assert_ne!(first, second);
    assert_ne!(first, retained);
    assert_ne!(second, retained);
    let pure = File::open_rw_with_options(&path, buffering.properties(backing)).unwrap();
    assert_eq!(pure.edit_backing(), Some(backing));
    assert_eq!(pure.superblock().offset_size, OffsetWidth::Eight);
    assert_eq!(pure.superblock().length_size, LengthWidth::Eight);
    let settings = pure.file_space_settings().unwrap().unwrap();
    assert_eq!(
        settings.strategy(),
        match strategy {
            PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
            PersistedStrategy::Page => FileSpaceStrategy::Page,
        }
    );
    assert!(settings.persist());
    assert_eq!(
        pure.root().attrs().unwrap()["third"],
        AttrValue::VarLenAsciiString(retained_value.clone())
    );
    pure.root().remove_attr("first").unwrap();
    pure.root().remove_attr("second").unwrap();
    pure.commit().unwrap();
    pure.close().unwrap();
    let after = std::fs::read(&path).unwrap();
    parent.assert_compacted(&after, &[retained]);
    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.root().attrs().unwrap(),
        HashMap::from([(
            "third".into(),
            AttrValue::VarLenAsciiString(retained_value.clone())
        )])
    );
    assert_eq!(
        pure.dataset("data").unwrap().read_i32().unwrap(),
        [7, 11, 13, 17]
    );
    pure.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(native.attr_names().unwrap(), ["third"]);
    assert_eq!(
        attributes::ascii_values(&native.attr("third").unwrap()),
        [retained_value.as_str()]
    );
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13, 17]
    );
    native.close().unwrap();
}

/// Writes a persistent shared-parent source and returns the retained attribute value.
///
/// # Panics
///
/// Panics if the native file, dataset, or attributes cannot be created or closed.
fn write_native(path: &Path, strategy: PersistedStrategy, terminal: Terminal) -> String {
    let retained = "r".repeat(
        global_heap::MIN_COLLECTION_SIZE
            - 4 * global_heap::HEADER_SIZE
            - 2 * global_heap::ALIGNMENT
            - terminal.bytes(),
    );
    let native = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties
                .file_space_strategy(NativeStrategy::FreeSpaceManager {
                    paged: matches!(strategy, PersistedStrategy::Page),
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(FileSpacePageSize::DEFAULT.get())
        })
        .create(path)
        .unwrap();
    native
        .new_dataset_builder()
        .with_data(&[7_i32, 11, 13, 17])
        .create("data")
        .unwrap();
    attributes::write_ascii(&native, "first", "left");
    attributes::write_ascii(&native, "second", "middle");
    attributes::write_ascii(&native, "third", &retained);
    native.close().unwrap();
    retained
}

#[test]
fn native_anonymous_count_updates_preserve_surrounding_unknown_storage() {
    let path = test_util::temp::temp_path("public-anonymous-count-storage.h5");
    AnonymousCountSource::write(&path);
    let identity = AnonymousCountSource::assert_native(&path, &["first", "second"]);
    let original = std::fs::read(&path).unwrap();
    let pure = File::open(&path).unwrap();
    let widths = (
        pure.superblock().offset_size.get(),
        pure.superblock().length_size.get(),
    );
    pure.close().unwrap();
    let headers: Vec<_> = test_util::bytes::signature_offsets(&original, v2::SIGNATURE)
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
        .collect();
    let [address] = headers.as_slice() else {
        panic!("{headers:?}");
    };
    let address = *address;
    let prefix = ObjectHeaderPrefix::parse(&original[address..]).unwrap();
    let end = address + usize::try_from(prefix.chunk0_size).unwrap() + prefix.len + 4;
    assert_eq!(address % 64, 0);
    assert!(end < end.next_multiple_of(64));
    assert!(
        original[end..end.next_multiple_of(64)]
            .iter()
            .all(|byte| *byte == 0)
    );
    let mut expected = original.clone();
    v2::replace_message(
        &mut expected,
        address,
        MessageType::OBJECT_REFERENCE_COUNT,
        &test_util::object_header::Message::reference_count(3).data,
    );
    let pure = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    pure.copy("first", "copy").unwrap();
    pure.commit().unwrap();
    let after = std::fs::read(&path).unwrap();
    assert_eq!(
        &after[address..end.next_multiple_of(64)],
        &expected[address..end.next_multiple_of(64)]
    );
    assert!(
        pure.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| section.end_address().get() <= address as u64
                || section.address().get() >= end as u64)
    );
    pure.close().unwrap();
    assert_eq!(
        AnonymousCountSource::assert_native(&path, &["copy", "first", "second"]),
        identity
    );
    let pure = File::open(&path).unwrap();
    assert!(pure.root().named_datatypes().unwrap().is_empty());
    for name in ["copy", "first", "second"] {
        assert_eq!(pure.dataset(name).unwrap().read_i32().unwrap(), [7, 11]);
    }
    pure.close().unwrap();
}

#[rstest]
#[case::candidate_free(RootContent::Numeric)]
#[case::root_string(RootContent::RootString)]
#[case::dataset_string(RootContent::DatasetString)]
fn native_root_self_link_deletion_preserves_live_values(
    #[case] content: RootContent,
    #[values(false, true)] earliest: bool,
) {
    let path = test_util::temp::temp_path("native-public-root-self-link.h5");
    let native = if earliest {
        file::libhdf5_create_earliest(&path)
    } else {
        file::libhdf5_create_v18(&path)
    };
    native.link_hard("/", "self").unwrap();
    native
        .new_attr::<i64>()
        .create("value")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    let dataset = native
        .new_dataset::<i32>()
        .shape([2])
        .create("data")
        .unwrap();
    dataset.write_raw(&[17, 31]).unwrap();
    match content {
        RootContent::Numeric => {}
        RootContent::RootString => attributes::write_ascii(&native, "text", "retained"),
        RootContent::DatasetString => attributes::write_ascii(&dataset, "text", "retained"),
    }
    drop(dataset);
    assert_eq!(native.loc_info_by_name("/").unwrap().num_links, 2);
    native.close().unwrap();
    let original = std::fs::read(&path).unwrap();
    let parents = global_heap::collection_ranges(&original, Widths::EIGHT);
    assert_eq!(parents.is_empty(), matches!(content, RootContent::Numeric));
    let pure = File::open_rw(&path).unwrap();
    pure.root().delete("self").unwrap();
    pure.commit().unwrap();
    assert!(!pure.has_staged_edits());
    pure.close().unwrap();
    let after = std::fs::read(&path).unwrap();
    for parent in parents {
        ParentImage::read(&original, parent.start).assert_unchanged(&after);
    }
    let pure = File::open(&path).unwrap();
    assert_eq!(pure.root().groups().unwrap(), Vec::<String>::new());
    assert_eq!(pure.root().datasets().unwrap(), ["data"]);
    assert_eq!(pure.dataset("data").unwrap().read_i32().unwrap(), [17, 31]);
    assert_eq!(pure.root().attrs().unwrap()["value"], AttrValue::I64(7));
    let attribute = match content {
        RootContent::Numeric => None,
        RootContent::RootString => Some(pure.root().attrs().unwrap()["text"].clone()),
        RootContent::DatasetString => {
            Some(pure.dataset("data").unwrap().attrs().unwrap()["text"].clone())
        }
    };
    if let Some(attribute) = attribute {
        assert_eq!(attribute, AttrValue::VarLenAsciiString("retained".into()));
    }
    pure.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(native.loc_info_by_name("/").unwrap().num_links, 1);
    assert!(!native.link_exists("self"));
    assert_eq!(
        native.attr("value").unwrap().read_scalar::<i64>().unwrap(),
        7
    );
    let dataset = native.dataset("data").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [17, 31]);
    let attribute = match content {
        RootContent::Numeric => None,
        RootContent::RootString => Some(native.attr("text").unwrap()),
        RootContent::DatasetString => Some(dataset.attr("text").unwrap()),
    };
    if let Some(attribute) = attribute {
        assert_eq!(
            attribute.read_scalar::<VarLenAscii>().unwrap().as_str(),
            "retained"
        );
    }
    drop(dataset);
    native.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum RootContent {
    DatasetString,
    Numeric,
    RootString,
}
