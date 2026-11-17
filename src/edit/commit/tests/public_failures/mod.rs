use rstest::rstest;
use test_util::disk_log;
use test_util_hdf5::global_heap_fixture::members::Buffering;

use crate::EditBacking;
use crate::File;
use crate::SyncPolicy;
use crate::edit::test_util::durability;
use crate::edit::test_util::durability::Fault;
use crate::edit::test_util::durability::HeapScenario;
use crate::edit::test_util::durability::Scenario;

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
fn public_heap_preparation_rejection_keeps_the_complete_staged_batch(
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rejected.h5");
    let fixture = Scenario::write(&path, true, 1);
    let file = durability::open(&path, backing, buffering, SyncPolicy::OnClose);
    fixture.stage(&file);
    file.root().delete("alias").unwrap();
    let before = file.disk_bytes().unwrap();
    let free = file.reusable_free_sections().unwrap();
    disk_log::start();
    let error = file.commit().unwrap_err();
    durability::assert_contradiction(error, fixture.owner.unwrap());
    assert!(disk_log::finish().events.is_empty());
    assert!(file.has_staged_edits());
    assert_eq!(file.disk_bytes().unwrap(), before);
    assert_eq!(file.reusable_free_sections().unwrap(), free);
    fixture.assert_values(&file, false, 17, None);
    disk_log::start();
    durability::assert_contradiction(file.commit().unwrap_err(), fixture.owner.unwrap());
    assert!(disk_log::finish().events.is_empty());
    assert!(file.has_staged_edits());
    assert_eq!(file.disk_bytes().unwrap(), before);
    drop(file);
    File::clear_status_flags(&path).unwrap();
    let reader = File::open(&path).unwrap();
    fixture.assert_values(&reader, false, 17, None);
    reader.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::content(Fault::Content)]
#[case::unwritten_publication(Fault::Publication(0))]
#[case::written_publication(Fault::Publication(usize::MAX))]
#[case::first_move(Fault::Move(0))]
#[case::first_free_header(Fault::Free)]
#[case::later_move(Fault::Move(1))]
fn public_heap_write_failures_preserve_their_phase_and_completed_prefix(
    #[case] fault: Fault,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation)] buffering: Buffering,
) {
    fault.run(backing, buffering, SyncPolicy::OnClose);
}

#[rstest]
#[case::first_move(Fault::Move(0))]
#[case::first_free_header(Fault::Free)]
#[case::later_move(Fault::Move(1))]
fn public_heap_session_buffer_failures_keep_the_published_tree(
    #[case] fault: Fault,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
) {
    fault.run(backing, Buffering::Session, SyncPolicy::OnClose);
}

#[rstest]
#[case::content(Fault::ContentBarrier)]
#[case::publication(Fault::PublicationBarrier)]
#[case::collection(Fault::CollectionBarrier)]
#[case::restoration(Fault::RestorationBarrier)]
fn public_heap_barrier_failures_preserve_restoration_and_allocation_evidence(
    #[case] fault: Fault,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation)] buffering: Buffering,
) {
    fault.run(backing, buffering, SyncPolicy::Always);
}
