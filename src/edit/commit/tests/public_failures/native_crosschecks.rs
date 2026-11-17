use std::fs;

use rstest::rstest;
use test_util::disk_log;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private;
use test_util_hdf5::file;
use test_util_hdf5::file_space::persisted::PersistedStrategy;
use test_util_hdf5::global_heap::attributes;
use test_util_hdf5::global_heap_fixture::members::{Buffering, ParentImage};

use crate::AttrValue;
use crate::EditBacking;
use crate::File;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::SyncPolicy;
use crate::edit::test_util::durability;
use crate::edit::test_util::durability::Fault;
use crate::edit::test_util::durability::native as durability_native;
use crate::page_publication_failure::PagePublicationFailure;

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
    __private::write_heap_failure_source(&path, FileSpaceSettings::new(FileSpaceStrategy::Page));
    let original = fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    let [range] = ranges.as_slice() else {
        panic!("expected one parent: {ranges:?}");
    };
    let parent = ParentImage::read(&original, range.start);
    let file = File::open_page_publication_failure(&path, failure, sync, gathered).unwrap();
    durability_native::stage(&file);
    durability::assert_io(
        file.commit().unwrap_err(),
        "injected PAGE publication failure",
    );
    assert!(!file.has_staged_edits());
    let published = file.disk_bytes().unwrap();
    parent.assert_unchanged(&published);
    let stored = File::from_bytes(published.clone()).unwrap();
    durability_native::assert_values(&stored, true, 23, None);
    stored.close().unwrap();
    file.root().set_attr("after", AttrValue::I64(31)).unwrap();
    if matches!(failure, PagePublicationFailure::RepeatedPublishedSync) {
        disk_log::start();
        durability::assert_io(
            file.commit().unwrap_err(),
            "injected PAGE publication failure",
        );
        durability::assert_io(
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
        durability_native::assert_values(&file, true, 23, Some(31));
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

fn strategy_value(strategy: PersistedStrategy) -> FileSpaceStrategy {
    match strategy {
        PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
        PersistedStrategy::Page => FileSpaceStrategy::Page,
    }
}
