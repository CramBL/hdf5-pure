use std::collections::HashMap;
use std::fs;
use std::path::Path;

use rstest::rstest;
use test_util::disk_log;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::file;
use test_util_hdf5::file_space::persisted::PersistedStrategy;
use test_util_hdf5::global_heap::attributes;
use test_util_hdf5::global_heap_fixture::members::{Buffering, ParentImage};
use test_util_hdf5::numeric::ByteOrder;

use super::Fault;
use crate::page_publication_failure::PagePublicationFailure;
use crate::{
    AttrValue, EditBacking, File, FileSpaceSettings, FileSpaceStrategy, LengthWidth, OffsetWidth,
    SyncPolicy,
};

#[rstest]
#[case::content(Fault::Content)]
#[case::unwritten_publication(Fault::Publication(0))]
#[case::written_publication(Fault::Publication(usize::MAX))]
#[case::compaction(Fault::Move(0))]
fn persistent_public_heap_failures_preserve_protected_allocations(
    #[case] fault: Fault,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation)] buffering: Buffering,
) {
    fault.run_native(
        FileSpaceSettings::new(strategy_value(strategy)).with_persistence(true),
        backing,
        buffering,
    );
}

#[rstest]
fn persistent_public_heap_session_failures_restore_the_failed_parent(
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
) {
    Fault::Move(0).run_native(
        FileSpaceSettings::new(strategy_value(strategy)).with_persistence(true),
        backing,
        Buffering::Session,
    );
}

#[rstest]
fn nonpersistent_page_compaction_failure_keeps_confirmed_publication_editable(
    #[values(Buffering::Unbuffered, Buffering::Operation)] buffering: Buffering,
) {
    Fault::Move(0).run_native(
        FileSpaceSettings::new(FileSpaceStrategy::Page),
        EditBacking::Mirrored,
        buffering,
    );
}

#[rstest]
#[case::recoverable(PagePublicationFailure::AfterPublish)]
#[case::uncertain(PagePublicationFailure::RepeatedPublishedSync)]
fn nonpersistent_page_publication_failure_retains_member_candidates_without_replay(
    #[case] failure: PagePublicationFailure,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(false, true)] gathered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("page-recovery.h5");
    write_source(&path, FileSpaceSettings::new(FileSpaceStrategy::Page));
    let original = fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    let [range] = ranges.as_slice() else {
        panic!("expected one parent: {ranges:?}");
    };
    let parent = ParentImage::read(&original, range.start);
    let file = File::open_page_publication_failure(&path, failure, sync, gathered).unwrap();
    stage(&file);
    super::assert_io(
        file.commit().unwrap_err(),
        "injected PAGE publication failure",
    );
    assert!(!file.has_staged_edits());
    let published = file.disk_bytes().unwrap();
    parent.assert_unchanged(&published);
    let stored = File::from_bytes(published.clone()).unwrap();
    assert_values(&stored, true, 23, None);
    stored.close().unwrap();
    file.root().set_attr("after", AttrValue::I64(31)).unwrap();
    if matches!(failure, PagePublicationFailure::RepeatedPublishedSync) {
        disk_log::start();
        super::assert_io(
            file.commit().unwrap_err(),
            "injected PAGE publication failure",
        );
        super::assert_io(
            file.sync().unwrap_err(),
            "injected PAGE publication failure",
        );
        assert!(disk_log::finish().operations.is_empty());
        assert!(file.has_staged_edits());
        assert_eq!(file.disk_bytes().unwrap(), published);
        drop(file);
    } else {
        file.commit().unwrap();
        assert!(!file.has_staged_edits());
        assert_values(&file, true, 23, Some(31));
        parent.assert_unchanged(&file.disk_bytes().unwrap());
        file.close().unwrap();
    }
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native
            .dataset("numbers")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [23]
    );
    assert_eq!(
        attributes::ascii_values(&native.attr("third").unwrap()),
        ["retained"]
    );
    if matches!(failure, PagePublicationFailure::AfterPublish) {
        assert_eq!(
            native.attr("after").unwrap().read_scalar::<i64>().unwrap(),
            31
        );
    }
    let mut names = native.attr_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        if matches!(failure, PagePublicationFailure::AfterPublish) {
            vec!["after", "third"]
        } else {
            vec!["third"]
        }
    );
    native.close().unwrap();
    directory.close().unwrap();
}

impl Fault {
    /// Checks a selected public failure against a fresh native source and both reopened readers.
    ///
    /// Issued bytes are inspected before retry or close, including session-buffered failures.
    ///
    /// # Panics
    ///
    /// Panics if fixture operations fail or any expected error, bytes, or public values differ.
    fn run_native(self, settings: FileSpaceSettings, backing: EditBacking, buffering: Buffering) {
        let fault = self;
        let directory = tempfile::tempdir().unwrap();
        let control = directory.path().join("control.h5");
        let path = directory.path().join("failure.h5");
        write_source(&control, settings);
        let original = fs::read(&control).unwrap();
        fs::write(&path, &original).unwrap();
        let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
        let [range] = ranges.as_slice() else {
            panic!("expected one shared parent: {ranges:?}");
        };
        let parent = ParentImage::read(&original, range.start);
        assert_eq!(parent.indices().len(), 3);
        let retained = parent.index_for(b"retained");
        let control_file = super::open(&control, backing, buffering, SyncPolicy::OnClose);
        assert_strategy(&control_file, settings);
        stage(&control_file);
        disk_log::start();
        control_file.commit().unwrap();
        let recording = disk_log::finish();
        parent.assert_compacted(&control_file.disk_bytes().unwrap(), &[retained]);
        let schedule = fault.schedule(&recording, &[range.start]);
        control_file.close().unwrap();
        let file = super::open(&path, backing, buffering, SyncPolicy::OnClose);
        assert_strategy(&file, settings);
        stage(&file);
        let free = file.reusable_free_sections().unwrap();
        disk_log::start();
        schedule.inject();
        let error = file.commit().unwrap_err();
        disk_log::clear_failures();
        let failed = disk_log::finish();
        schedule.assert_failed(&failed);
        fault.assert_error(error);
        assert!(!file.has_staged_edits());
        let after = file.disk_bytes().unwrap();
        let stored = File::from_bytes(after.clone()).unwrap();
        assert_values(
            &stored,
            fault.confirmed() || matches!(fault, Fault::Publication(usize::MAX)),
            if fault.before() { 17 } else { 23 },
            None,
        );
        stored.close().unwrap();
        parent.assert_unchanged(&after);
        fault.assert_publication(&failed, &original, &after);
        assert_values(
            &file,
            fault.confirmed(),
            if fault.before() { 17 } else { 23 },
            None,
        );
        if !fault.uncertain() {
            assert!(
                file.reusable_free_sections()
                    .unwrap()
                    .sections()
                    .iter()
                    .all(|section| section.end_address().get() <= range.start
                        || section.address().get() >= range.end)
            );
        }
        if fault.uncertain() {
            disk_log::start();
            super::assert_uncertain(file.commit().unwrap_err());
            super::assert_uncertain(file.sync().unwrap_err());
            assert!(disk_log::finish().events.is_empty());
            assert_eq!(file.disk_bytes().unwrap(), after);
            drop(file);
            File::clear_status_flags(&path).unwrap();
        } else {
            if fault.before() {
                assert_eq!(file.reusable_free_sections().unwrap(), free);
            }
            disk_log::start();
            file.commit().unwrap();
            assert!(disk_log::finish().events.is_empty());
            file.root().set_attr("after", AttrValue::I64(31)).unwrap();
            file.commit().unwrap();
            parent.assert_unchanged(&file.disk_bytes().unwrap());
            assert_values(
                &file,
                fault.confirmed(),
                if fault.before() { 17 } else { 23 },
                Some(31),
            );
            file.close().unwrap();
        }
        let published = fault.confirmed() || matches!(fault, Fault::Publication(usize::MAX));
        let reader = File::open(&path).unwrap();
        let fresh = (!fault.uncertain()).then_some(31);
        assert_values(
            &reader,
            published,
            if fault.before() { 17 } else { 23 },
            fresh,
        );
        reader.close().unwrap();
        let native = file::libhdf5_open(&path);
        assert_eq!(
            native
                .dataset("numbers")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [if fault.before() { 17 } else { 23 }]
        );
        assert_eq!(
            attributes::ascii_values(&native.attr("third").unwrap()),
            ["retained"]
        );
        if let Some(expected) = fresh {
            assert_eq!(
                native.attr("after").unwrap().read_scalar::<i64>().unwrap(),
                expected
            );
        }
        let mut names = native.attr_names().unwrap();
        names.sort();
        assert_eq!(
            names,
            if published {
                if fault.uncertain() {
                    vec!["third"]
                } else {
                    vec!["after", "third"]
                }
            } else if fault.uncertain() {
                vec!["first", "second", "third"]
            } else {
                vec!["after", "first", "second", "third"]
            }
        );
        native.close().unwrap();
        directory.close().unwrap();
    }
}

/// Writes a native source with eight-byte widths, root attributes, and little-endian integers.
///
/// # Panics
///
/// Panics if the file, dataset, or attributes cannot be created or closed.
fn write_source(path: &Path, settings: FileSpaceSettings) {
    let native = file::libhdf5_create_with_widths_and_space(
        path,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        0,
        settings,
    );
    native
        .new_dataset_builder()
        .with_data_as(&[17_i32], ByteOrder::LittleEndian.i32_datatype())
        .create("numbers")
        .unwrap();
    attributes::write_ascii(&native, "first", "removed");
    attributes::write_ascii(&native, "second", "discard");
    attributes::write_ascii(&native, "third", "retained");
    native.close().unwrap();
}

/// Stages removal of the two obsolete attributes and replacement of numeric data with 23.
///
/// # Panics
///
/// Panics if a removal or numeric write cannot be staged.
fn stage(file: &File) {
    file.root().remove_attr("first").unwrap();
    file.root().remove_attr("second").unwrap();
    file.dataset("numbers").unwrap().write(&[23_i32]).unwrap();
}

fn assert_strategy(file: &File, expected: FileSpaceSettings) {
    let settings = file.file_space_settings().unwrap().unwrap();
    assert_eq!(settings, expected);
}

/// Checks numeric data and the complete root attribute map for the publication state.
///
/// `published` selects removal of the original attributes, and `after` is the optional fresh
/// scalar attribute value.
fn assert_values(file: &File, published: bool, number: i32, after: Option<i64>) {
    assert_eq!(
        file.dataset("numbers").unwrap().read_i32().unwrap(),
        [number]
    );
    let mut expected = HashMap::from([(
        "third".into(),
        AttrValue::VarLenAsciiString("retained".into()),
    )]);
    if !published {
        expected.insert(
            "first".into(),
            AttrValue::VarLenAsciiString("removed".into()),
        );
        expected.insert(
            "second".into(),
            AttrValue::VarLenAsciiString("discard".into()),
        );
    }
    if let Some(value) = after {
        expected.insert("after".into(), AttrValue::I64(value));
    }
    assert_eq!(file.root().attrs().unwrap(), expected);
}

fn strategy_value(strategy: PersistedStrategy) -> FileSpaceStrategy {
    match strategy {
        PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
        PersistedStrategy::Page => FileSpaceStrategy::Page,
    }
}
