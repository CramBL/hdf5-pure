use std::fs;
use std::io::ErrorKind;

use rstest::rstest;
use test_util::disk_log::{self, FailurePoint, IoEvent, IoOutcome, Recording, Synchronization};

use crate::{Error, File, FormatError, HeapCompactionState};

use super::{
    HEADER, HeapWrite, Prefix, RECORD, assert_failed_write, assert_io, assert_issued_write,
    assert_original_values, assert_published_scalars, assert_published_values, heap_write,
    open_editor, parent_range, replay_prefix, stage, successful_commit, write_source,
};

#[derive(Clone, Copy)]
enum Fault {
    Write(Prefix),
    Barrier,
}

impl Fault {
    fn schedule(self, control: &Recording, parent: u64) -> usize {
        let free = heap_write(control, parent, HeapWrite::FreeHeader);
        let boundary = match self {
            Self::Write(_) => free + 1,
            Self::Barrier => free,
        };
        let writes = control.events[..boundary]
            .iter()
            .filter(|event| matches!(event, IoEvent::Write { .. }))
            .count();
        let syncs = control.events[..boundary]
            .iter()
            .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
            .count();
        disk_log::fail_sync_after(syncs, FailurePoint::Before);
        match self {
            Self::Write(prefix) => {
                assert_eq!(
                    control.events[boundary],
                    IoEvent::Synchronize {
                        kind: Synchronization::All,
                        outcome: IoOutcome::Succeeded,
                    }
                );
                // The failed collection barrier leaves two undo entries. Let the free header
                // restore complete before failing the move restoration.
                disk_log::fail_write_after(writes + 1, prefix.bytes(RECORD));
            }
            Self::Barrier => disk_log::fail_write_after(writes, HEADER),
        }
        boundary
    }

    fn restored_bytes(self) -> usize {
        match self {
            Self::Write(prefix) => prefix.bytes(RECORD),
            Self::Barrier => RECORD,
        }
    }

    fn assert_error(self, error: Error) {
        let Error::HeapCompactionFailed {
            completed,
            state,
            source,
        } = error
        else {
            panic!("expected a heap compaction failure, got {error:?}");
        };
        assert_eq!(completed, 0);
        assert_eq!(state, HeapCompactionState::Uncertain);
        let Error::CommitPartiallyApplied { refusal, restore } = *source else {
            panic!("expected an uncertain restoration, got {source:?}");
        };
        let (first, second) = match self {
            Self::Write(_) => (SYNC_FAILURE, WRITE_FAILURE),
            Self::Barrier => (WRITE_FAILURE, SYNC_FAILURE),
        };
        assert_io(*refusal, first);
        assert_io(*restore, second);
    }

    fn assert_issued(
        self,
        failed: &Recording,
        control: &Recording,
        boundary: usize,
        original: &[u8],
        parent: u64,
    ) {
        assert_eq!(failed.events[..boundary], control.events[..boundary]);
        assert_eq!(
            replay_prefix(failed, original, boundary),
            replay_prefix(control, original, boundary)
        );
        let move_at = HeapWrite::Move.offset(parent);
        let free_at = HeapWrite::FreeHeader.offset(parent);
        let movement = usize::try_from(move_at).unwrap();
        let free = usize::try_from(free_at).unwrap();
        let write_outcome = match self {
            Self::Write(_) => {
                assert_eq!(
                    failed.events[boundary],
                    IoEvent::Synchronize {
                        kind: Synchronization::All,
                        outcome: IoOutcome::Failed(ErrorKind::Other),
                    }
                );
                IoOutcome::Failed(ErrorKind::Other)
            }
            Self::Barrier => {
                assert_failed_write(
                    control,
                    failed,
                    boundary,
                    HEADER,
                    original,
                    parent_range(parent),
                );
                IoOutcome::Succeeded
            }
        };
        assert_issued_write(
            failed,
            boundary + 1,
            free_at,
            &original[free..free + HEADER],
            HEADER,
            IoOutcome::Succeeded,
        );
        assert_issued_write(
            failed,
            boundary + 2,
            move_at,
            &original[movement..movement + RECORD],
            self.restored_bytes(),
            write_outcome,
        );
        if matches!(self, Self::Barrier) {
            assert_eq!(
                failed.events[boundary + 3],
                IoEvent::Synchronize {
                    kind: Synchronization::Data,
                    outcome: IoOutcome::Failed(ErrorKind::Other),
                }
            );
            assert_eq!(failed.events.len(), boundary + 4);
        } else {
            assert_eq!(failed.events.len(), boundary + 3);
        }
    }

    fn expected_image(self, original: &[u8], mut published: Vec<u8>, parents: &[u64]) -> Vec<u8> {
        for &parent in parents {
            let range = parent_range(parent);
            published[range.clone()].copy_from_slice(&original[range]);
        }
        let start = parent_range(parents[0]).start + HEADER;
        let restored = self.restored_bytes();
        let suffix = start + restored..start + RECORD;
        let moved = &original[start + RECORD + restored..start + 2 * RECORD];
        if restored < RECORD {
            assert_ne!(&original[suffix.clone()], moved);
        }
        published[suffix].copy_from_slice(moved);
        published
    }
}

#[rstest]
#[case::restore_write_unwritten(Fault::Write(Prefix::Zero))]
#[case::restore_write_partial(Fault::Write(Prefix::Partial))]
#[case::restore_write_complete(Fault::Write(Prefix::Full))]
#[case::restore_barrier(Fault::Barrier)]
fn public_heap_restoration_failure_quarantines_allocation(#[case] fault: Fault) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let control_path = directory.path().join("control.h5");
    let parents = write_source(&path);
    fs::copy(&path, &control_path).unwrap();
    let (original, published, control) =
        successful_commit(&control_path, &parents, HeapWrite::Move);
    let file = open_editor(&path, HeapWrite::Move);
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_original_values(&file);
    stage(&file);
    assert!(file.has_staged_edits());
    disk_log::start();
    let boundary = fault.schedule(&control, parents[0]);
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    fault.assert_error(error);
    fault.assert_issued(&failed, &control, boundary, &original, parents[0]);
    assert!(!file.has_staged_edits());

    let expected = fault.expected_image(&original, published, &parents);
    let issued = replay_prefix(&failed, &original, failed.events.len());
    assert_eq!(fs::read(&path).unwrap(), expected);
    assert_eq!(issued, expected);
    assert_published_scalars(&file);
    assert_untouched_labels(&file);

    disk_log::start();
    assert_uncertain(file.commit());
    file.root().create_group("after").unwrap();
    assert!(file.has_staged_edits());
    assert_uncertain(file.commit());
    assert_uncertain(file.sync());
    assert!(file.has_staged_edits());
    assert_uncertain(file.close());
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(fs::read(&path).unwrap(), expected);

    let reopened = File::open(&path).unwrap();
    assert!(reopened.root().groups().unwrap().is_empty());
    if matches!(fault, Fault::Write(Prefix::Zero)) {
        assert_published_scalars(&reopened);
        assert_untouched_labels(&reopened);
        let error = reopened
            .dataset("keep")
            .unwrap()
            .read_string_region(&[0], &[1])
            .unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected duplicate heap member, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::DuplicateGlobalHeapObjectIndex {
                collection_address: parents[0],
                index: 91,
            }
        );
    } else {
        assert_published_values(&reopened);
    }
    reopened.close().unwrap();
}

fn assert_uncertain(result: Result<(), Error>) {
    assert!(matches!(
        result.unwrap_err(),
        Error::AllocationPublicationUncertain
    ));
}

fn assert_untouched_labels(file: &File) {
    assert_eq!(
        file.dataset("keep")
            .unwrap()
            .read_string_region(&[1], &[2])
            .unwrap(),
        ["retained"; 2]
    );
}

const WRITE_FAILURE: &str = "injected disk write failure";
const SYNC_FAILURE: &str = "injected disk synchronization failure";
