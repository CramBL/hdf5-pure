use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use rstest::rstest;
use test_util::disk_log::{self, FailurePoint, IoEvent, IoOutcome, Recording, Synchronization};

use crate::{Error, File, FormatError, HeapCompactionState};

use super::{
    HEADER, HeapWrite, Prefix, RECORD, assert_failed_write, assert_io, assert_issued_write,
    assert_original_values, assert_published_scalars, assert_published_values, heap_write,
    open_editor, parent_range, replay_prefix, stage, successful_commit, write_source,
};

#[test]
fn unwritten_heap_restoration_quarantines_the_session_and_leaves_a_duplicate_member() {
    let directory = tempfile::tempdir().unwrap();
    let source = RestorationSource::prepare(directory.path());
    let parent = source.parents[0];
    let failed = source.fail_restore_write(0);
    let reopened = source.assert_quarantined_and_reopen(&failed, 0);
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
            collection_address: parent,
            index: 91,
        }
    );
    reopened.close().unwrap();
}

#[rstest]
#[case::restore_write_partial(Prefix::Partial)]
#[case::restore_write_complete(Prefix::Full)]
fn failed_heap_restoration_write_quarantines_the_session_with_readable_values(
    #[case] prefix: Prefix,
) {
    let directory = tempfile::tempdir().unwrap();
    let source = RestorationSource::prepare(directory.path());
    let restored = prefix.bytes(RECORD);
    let failed = source.fail_restore_write(restored);
    let reopened = source.assert_quarantined_and_reopen(&failed, restored);
    assert_published_values(&reopened);
    reopened.close().unwrap();
}

#[test]
fn failed_heap_restoration_barrier_quarantines_the_session_after_undo_writes() {
    let directory = tempfile::tempdir().unwrap();
    let source = RestorationSource::prepare(directory.path());
    let parent = source.parents[0];
    let boundary = heap_write(&source.control, parent, HeapWrite::FreeHeader);
    disk_log::start();
    let writes = source.control.events[..boundary]
        .iter()
        .filter(|event| matches!(event, IoEvent::Write { .. }))
        .count();
    source.fail_next_sync(boundary);
    disk_log::fail_write_after(writes, HEADER);

    let failed = source.commit_failure(WRITE_FAILURE, SYNC_FAILURE);
    source.assert_control_prefix(&failed, boundary);
    assert_failed_write(
        &source.control,
        &failed,
        boundary,
        HEADER,
        &source.original,
        parent_range(parent),
    );
    source.assert_undo_writes(&failed, boundary + 1, RECORD, IoOutcome::Succeeded);
    assert_eq!(
        failed.events[boundary + 3],
        IoEvent::Synchronize {
            kind: Synchronization::Data,
            outcome: IoOutcome::Failed(ErrorKind::Other),
        }
    );
    assert_eq!(failed.events.len(), boundary + 4);

    let reopened = source.assert_quarantined_and_reopen(&failed, RECORD);
    assert_published_values(&reopened);
    reopened.close().unwrap();
}

struct RestorationSource {
    path: PathBuf,
    file: File,
    parents: [u64; 3],
    original: Vec<u8>,
    published: Vec<u8>,
    control: Recording,
}

impl RestorationSource {
    fn prepare(directory: &Path) -> Self {
        let path = directory.join("failure.h5");
        let control_path = directory.join("control.h5");
        let parents = write_source(&path);
        fs::copy(&path, &control_path).unwrap();
        let (original, published, control) =
            successful_commit(&control_path, &parents, HeapWrite::Move);
        let file = open_editor(&path, HeapWrite::Move);
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_original_values(&file);
        stage(&file);
        assert!(file.has_staged_edits());
        Self {
            path,
            file,
            parents,
            original,
            published,
            control,
        }
    }

    fn fail_restore_write(&self, restored: usize) -> Recording {
        let free = heap_write(&self.control, self.parents[0], HeapWrite::FreeHeader);
        let boundary = free + 1;
        assert_eq!(
            self.control.events[boundary],
            IoEvent::Synchronize {
                kind: Synchronization::All,
                outcome: IoOutcome::Succeeded,
            }
        );
        let writes = self.control.events[..boundary]
            .iter()
            .filter(|event| matches!(event, IoEvent::Write { .. }))
            .count();
        disk_log::start();
        self.fail_next_sync(boundary);
        disk_log::fail_write_after(writes + 1, restored);

        let failed = self.commit_failure(SYNC_FAILURE, WRITE_FAILURE);
        self.assert_control_prefix(&failed, boundary);
        assert_eq!(
            failed.events[boundary],
            IoEvent::Synchronize {
                kind: Synchronization::All,
                outcome: IoOutcome::Failed(ErrorKind::Other),
            }
        );
        self.assert_undo_writes(
            &failed,
            boundary + 1,
            restored,
            IoOutcome::Failed(ErrorKind::Other),
        );
        assert_eq!(failed.events.len(), boundary + 3);
        failed
    }

    fn fail_next_sync(&self, boundary: usize) {
        let syncs = self.control.events[..boundary]
            .iter()
            .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
            .count();
        disk_log::fail_sync_after(syncs, FailurePoint::Before);
    }

    fn commit_failure(&self, expected_refusal: &str, expected_restore: &str) -> Recording {
        let error = self.file.commit().unwrap_err();
        disk_log::clear_failures();
        let failed = disk_log::finish();
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
        assert_io(*refusal, expected_refusal);
        assert_io(*restore, expected_restore);
        assert!(!self.file.has_staged_edits());
        failed
    }

    fn assert_control_prefix(&self, failed: &Recording, boundary: usize) {
        assert_eq!(failed.events[..boundary], self.control.events[..boundary]);
        assert_eq!(
            replay_prefix(failed, &self.original, boundary),
            replay_prefix(&self.control, &self.original, boundary),
        );
    }

    fn assert_undo_writes(
        &self,
        failed: &Recording,
        event: usize,
        restored: usize,
        move_outcome: IoOutcome,
    ) {
        let parent = self.parents[0];
        let move_at = HeapWrite::Move.offset(parent);
        let free_at = HeapWrite::FreeHeader.offset(parent);
        let movement = usize::try_from(move_at).unwrap();
        let free = usize::try_from(free_at).unwrap();
        assert_issued_write(
            failed,
            event,
            free_at,
            &self.original[free..free + HEADER],
            HEADER,
            IoOutcome::Succeeded,
        );
        assert_issued_write(
            failed,
            event + 1,
            move_at,
            &self.original[movement..movement + RECORD],
            restored,
            move_outcome,
        );
    }

    fn assert_quarantined_and_reopen(self, failed: &Recording, restored: usize) -> File {
        let expected = self.expected_image(restored);
        let issued = replay_prefix(failed, &self.original, failed.events.len());
        assert_eq!(fs::read(&self.path).unwrap(), expected);
        assert_eq!(issued, expected);
        assert_published_scalars(&self.file);
        assert_untouched_labels(&self.file);

        disk_log::start();
        assert_uncertain(self.file.commit());
        self.file.root().create_group("after").unwrap();
        assert!(self.file.has_staged_edits());
        assert_uncertain(self.file.commit());
        assert_uncertain(self.file.sync());
        assert!(self.file.has_staged_edits());
        assert_uncertain(self.file.close());
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(fs::read(&self.path).unwrap(), expected);

        let reopened = File::open(&self.path).unwrap();
        assert!(reopened.root().groups().unwrap().is_empty());
        reopened
    }

    fn expected_image(&self, restored: usize) -> Vec<u8> {
        let mut published = self.published.clone();
        for &parent in &self.parents {
            let range = parent_range(parent);
            published[range.clone()].copy_from_slice(&self.original[range]);
        }
        let start = parent_range(self.parents[0]).start + HEADER;
        let suffix = start + restored..start + RECORD;
        let moved = &self.original[start + RECORD + restored..start + 2 * RECORD];
        if restored < RECORD {
            assert_ne!(&self.original[suffix.clone()], moved);
        }
        published[suffix].copy_from_slice(moved);
        published
    }
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
