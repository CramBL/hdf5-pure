use std::fs;
use std::io::ErrorKind;

use std::path::{Path, PathBuf};

use test_util::disk_log::{
    self, DiskOp, FailurePoint, IoEvent, IoOutcome, Recording, Synchronization,
};

use crate::{
    AttrValue, EditBacking, Error, File, FileAccessProperties, HeapCompactionState, Layout,
    SyncPolicy,
};

use super::{
    HeapWrite, assert_allocated, assert_failed_write, assert_io, assert_issued_write,
    assert_original_values, assert_parents, assert_published_values, editor_options, heap_write,
    parent_range, replay_prefix, stage, write_source_with_gap,
};

#[test]
fn session_buffered_heap_write_failure_drains_the_remaining_published_write() {
    let directory = tempfile::tempdir().unwrap();
    let backing = EditBacking::Bounded;
    let options = editor_options()
        .with_memory_strategy(backing.into())
        .with_sync_policy(SyncPolicy::OnClose)
        .with_page_buffer_size(64 * 1024);
    let commit = BufferedCommit::prepare(directory.path(), options, backing);
    let boundary = commit.move_event;
    let parent = commit.parents[0];
    disk_log::start();
    disk_log::fail_write_after(
        commit.recording.events[..boundary]
            .iter()
            .filter(|event| matches!(event, IoEvent::Write { .. }))
            .count(),
        HeapWrite::Gathered.length(),
    );
    let failed = commit.commit_failure("injected disk write failure");

    commit.assert_prefix(&failed, boundary);
    assert_failed_write(
        &commit.recording,
        &failed,
        boundary,
        HeapWrite::Gathered.length(),
        &commit.original,
        parent_range(parent),
    );
    commit.assert_restored_parent_write(&failed, boundary + 1);

    let Layout::Contiguous {
        address: Some(address),
        size: 4,
    } = commit.file.dataset("numbers").unwrap().layout().unwrap()
    else {
        panic!("expected allocated numeric storage");
    };
    // The restoration flush also drains the gathered run containing the numeric write.
    let operation = commit.recording.events[boundary + 1].operation().unwrap();
    let DiskOp::Write { offset, bytes } = &commit.recording.operations[operation] else {
        panic!("expected the published batch's remaining write");
    };
    let number = usize::try_from(address.checked_sub(*offset).unwrap()).unwrap();
    assert_eq!(&bytes[number..number + 4], &23_i32.to_le_bytes());
    assert_issued_write(
        &failed,
        boundary + 2,
        *offset,
        bytes,
        bytes.len(),
        IoOutcome::Succeeded,
    );
    assert_eq!(failed.events.len(), boundary + 3, "{:?}", failed.events);
    commit.assert_retained_values_and_retry(&failed);
}

#[test]
fn operation_buffered_heap_barrier_failure_syncs_the_restored_parent() {
    let directory = tempfile::tempdir().unwrap();
    let backing = EditBacking::Mirrored;
    let options = editor_options()
        .with_memory_strategy(backing.into())
        .with_sync_policy(SyncPolicy::Always)
        .with_page_buffer_size(0);
    let commit = BufferedCommit::prepare(directory.path(), options, backing);
    let boundary = commit.move_event + 1;
    assert_eq!(
        commit.recording.events[boundary],
        IoEvent::Synchronize {
            kind: Synchronization::All,
            outcome: IoOutcome::Succeeded,
        }
    );
    disk_log::start();
    disk_log::fail_sync_after(
        commit.recording.events[..boundary]
            .iter()
            .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
            .count(),
        FailurePoint::After,
    );
    let failed = commit.commit_failure("injected disk synchronization failure");

    commit.assert_prefix(&failed, boundary);
    assert_eq!(
        failed.events[boundary],
        IoEvent::Synchronize {
            kind: Synchronization::All,
            outcome: IoOutcome::Failed(ErrorKind::Other),
        }
    );
    commit.assert_restored_parent_write(&failed, boundary + 1);
    assert_eq!(
        failed.events[boundary + 2],
        IoEvent::Synchronize {
            kind: Synchronization::Data,
            outcome: IoOutcome::Succeeded,
        }
    );
    assert_eq!(failed.events.len(), boundary + 3);
    commit.assert_retained_values_and_retry(&failed);
}

struct BufferedCommit {
    path: PathBuf,
    file: File,
    parents: [u64; 3],
    original: Vec<u8>,
    published: Vec<u8>,
    recording: Recording,
    move_event: usize,
}

impl BufferedCommit {
    fn prepare(directory: &Path, options: FileAccessProperties, backing: EditBacking) -> Self {
        let path = directory.join("failure.h5");
        let control_path = directory.join("control.h5");
        // A page boundary separates the published superblock and the first heap write.
        let parents = write_source_with_gap(&path, 4096);
        fs::copy(&path, &control_path).unwrap();
        let control = File::open_rw_with_options(&control_path, options).unwrap();
        assert_eq!(control.edit_backing(), Some(backing));
        let original = control.disk_bytes().unwrap();
        stage(&control);
        disk_log::start();
        control.commit().unwrap();
        let recording = disk_log::finish();
        let published = control.disk_bytes().unwrap();
        assert_parents(&parents, &original, &published, parents.len());
        assert_published_values(&control);
        control.close().unwrap();

        let file = File::open_rw_with_options(&path, options).unwrap();
        assert_eq!(file.edit_backing(), Some(backing));
        assert_eq!(file.disk_bytes().unwrap(), original);
        assert_original_values(&file);
        stage(&file);
        let move_event = heap_write(&recording, parents[0], HeapWrite::Gathered);
        Self {
            path,
            file,
            parents,
            original,
            published,
            recording,
            move_event,
        }
    }

    fn commit_failure(&self, message: &str) -> Recording {
        let error = self.file.commit().unwrap_err();
        disk_log::clear_failures();
        let failed = disk_log::finish();
        assert_error_after_restore(error, message);
        failed
    }

    fn assert_prefix(&self, failed: &Recording, boundary: usize) {
        assert_eq!(failed.events[..boundary], self.recording.events[..boundary]);
        assert_eq!(
            replay_prefix(failed, &self.original, boundary),
            replay_prefix(&self.recording, &self.original, boundary),
        );
    }

    fn assert_restored_parent_write(&self, failed: &Recording, event: usize) {
        let address = HeapWrite::Gathered.offset(self.parents[0]);
        let start = usize::try_from(address).unwrap();
        let len = HeapWrite::Gathered.length();
        assert_issued_write(
            failed,
            event,
            address,
            &self.original[start..start + len],
            len,
            IoOutcome::Succeeded,
        );
    }

    fn assert_retained_values_and_retry(self, failed: &Recording) {
        let after = self.file.disk_bytes().unwrap();
        assert_eq!(
            after,
            replay_prefix(failed, &self.original, failed.events.len()),
        );
        assert_eq!(&after[..48], &self.published[..48]);
        assert_parents(&self.parents, &self.original, &after, 0);
        assert_allocated(&self.file, &self.parents);
        assert_published_values(&self.file);
        assert!(!self.file.has_staged_edits());

        disk_log::start();
        self.file.commit().unwrap();
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(self.file.disk_bytes().unwrap(), after);
        self.file
            .root()
            .set_attr("after", AttrValue::I64(31))
            .unwrap();
        self.file.commit().unwrap();
        self.file.close().unwrap();
        assert_parents(
            &self.parents,
            &self.original,
            &fs::read(&self.path).unwrap(),
            0,
        );
        let reopened = File::open(&self.path).unwrap();
        assert_published_values(&reopened);
        assert_eq!(
            reopened.root().attrs().unwrap()["after"],
            AttrValue::I64(31),
        );
        reopened.close().unwrap();
    }
}

fn assert_error_after_restore(error: Error, message: &str) {
    let Error::HeapCompactionFailed {
        completed,
        state,
        source,
    } = error
    else {
        panic!("expected a heap compaction failure, got {error:?}");
    };
    assert_eq!(completed, 0);
    assert_eq!(state, HeapCompactionState::Restored);
    assert_io(*source, message);
}
