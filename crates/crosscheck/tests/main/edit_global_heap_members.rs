#![cfg(feature = "__hdf5-1.10")]

use std::collections::HashMap;
use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5::types::VarLenAscii;
use hdf5_pure::{AttrValue, EditBacking, File, FileSpacePageSize, FileSpaceStrategy};
use hdf5_pure_core::{LengthWidth, OffsetWidth};
use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
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
