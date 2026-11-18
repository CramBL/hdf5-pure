use std::fs;
use std::io::ErrorKind;

use rstest::rstest;
use test_util::disk_log::{self, DiskOp, FailurePoint, IoEvent, IoOutcome, Synchronization};

use crate::{
    AttrValue, EditBacking, Error, File, FileAccessProperties, HeapCompactionState, Layout,
    SyncPolicy,
};

use super::{
    HeapWrite, assert_allocated, assert_failed_write, assert_io, assert_issued_write,
    assert_original_values, assert_parents, assert_published_values, editor_options, heap_write,
    parent_range, replay_prefix, stage, write_source_with_gap,
};

#[derive(Clone, Copy)]
enum BufferFault {
    SessionIssue,
    OperationBarrier,
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

impl BufferFault {
    fn options(self) -> (FileAccessProperties, EditBacking) {
        let (backing, sync, buffer) = match self {
            Self::SessionIssue => (EditBacking::Bounded, SyncPolicy::OnClose, 64 * 1024),
            Self::OperationBarrier => (EditBacking::Mirrored, SyncPolicy::Always, 0),
        };
        (
            editor_options()
                .with_memory_strategy(backing.into())
                .with_sync_policy(sync)
                .with_page_buffer_size(buffer),
            backing,
        )
    }
}

#[rstest]
#[case::bounded_session_on_close(BufferFault::SessionIssue)]
#[case::mirrored_operation_always(BufferFault::OperationBarrier)]
fn public_heap_buffered_failures_issue_restoration_before_returning(#[case] fault: BufferFault) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let control_path = directory.path().join("control.h5");
    // A page boundary separates the published superblock and the first heap write.
    let parents = write_source_with_gap(&path, 4096);
    fs::copy(&path, &control_path).unwrap();
    let (options, backing) = fault.options();
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
    let parent = parents[0];
    let move_event = heap_write(&recording, parent, HeapWrite::Gathered);
    let boundary = match fault {
        BufferFault::SessionIssue => move_event,
        BufferFault::OperationBarrier => move_event + 1,
    };
    disk_log::start();
    match fault {
        BufferFault::SessionIssue => disk_log::fail_write_after(
            recording.events[..boundary]
                .iter()
                .filter(|event| matches!(event, IoEvent::Write { .. }))
                .count(),
            HeapWrite::Gathered.length(),
        ),
        BufferFault::OperationBarrier => {
            assert_eq!(
                recording.events[boundary],
                IoEvent::Synchronize {
                    kind: Synchronization::All,
                    outcome: IoOutcome::Succeeded,
                }
            );
            disk_log::fail_sync_after(
                recording.events[..boundary]
                    .iter()
                    .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                    .count(),
                FailurePoint::After,
            );
        }
    }
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    assert_eq!(failed.events[..boundary], recording.events[..boundary]);
    assert_eq!(
        replay_prefix(&failed, &original, boundary),
        replay_prefix(&recording, &original, boundary)
    );
    let message = match fault {
        BufferFault::SessionIssue => {
            assert_failed_write(
                &recording,
                &failed,
                boundary,
                HeapWrite::Gathered.length(),
                &original,
                parent_range(parent),
            );
            "injected disk write failure"
        }
        BufferFault::OperationBarrier => {
            assert_eq!(
                failed.events[boundary],
                IoEvent::Synchronize {
                    kind: Synchronization::All,
                    outcome: IoOutcome::Failed(ErrorKind::Other),
                }
            );
            "injected disk synchronization failure"
        }
    };
    assert_error_after_restore(error, message);
    let address = HeapWrite::Gathered.offset(parent);
    let start = usize::try_from(address).unwrap();
    let len = HeapWrite::Gathered.length();
    assert_issued_write(
        &failed,
        boundary + 1,
        address,
        &original[start..start + len],
        len,
        IoOutcome::Succeeded,
    );
    if matches!(fault, BufferFault::OperationBarrier) {
        assert_eq!(
            failed.events[boundary + 2],
            IoEvent::Synchronize {
                kind: Synchronization::Data,
                outcome: IoOutcome::Succeeded,
            }
        );
        assert_eq!(failed.events.len(), boundary + 3);
    } else {
        let Layout::Contiguous {
            address: Some(address),
            size: 4,
        } = file.dataset("numbers").unwrap().layout().unwrap()
        else {
            panic!("expected allocated numeric storage");
        };
        // The restoration flush also drains the gathered run containing the numeric write.
        let operation = recording.events[boundary + 1].operation().unwrap();
        let DiskOp::Write { offset, bytes } = &recording.operations[operation] else {
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
    }
    let after = file.disk_bytes().unwrap();
    assert_eq!(
        after,
        replay_prefix(&failed, &original, failed.events.len())
    );
    assert_eq!(&after[..48], &published[..48]);
    assert_parents(&parents, &original, &after, 0);
    assert_allocated(&file, &parents);
    assert_published_values(&file);
    assert!(!file.has_staged_edits());

    disk_log::start();
    file.commit().unwrap();
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(file.disk_bytes().unwrap(), after);
    file.root().set_attr("after", AttrValue::I64(31)).unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    assert_parents(&parents, &original, &fs::read(&path).unwrap(), 0);
    let reopened = File::open(&path).unwrap();
    assert_published_values(&reopened);
    assert_eq!(
        reopened.root().attrs().unwrap()["after"],
        AttrValue::I64(31)
    );
    reopened.close().unwrap();
}
