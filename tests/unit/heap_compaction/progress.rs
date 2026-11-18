use std::fs;

use test_util::disk_log::{self, IoEvent, IoOutcome, Recording, Synchronization};
use test_util::global_heap::{self, Object};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;

use crate::{AttrValue, Error, File, HeapCompactionState};

use super::{
    HEADER, HeapWrite, PARTIAL, RECORD, assert_allocated, assert_failed_write, assert_io,
    assert_issued_write, assert_original_values, assert_parents, assert_published_values,
    heap_write, open_editor, parent_range, replay_prefix, stage, successful_commit, write_source,
};

#[test]
fn public_heap_later_failure_allows_fresh_heap_reclamation_in_the_same_editor() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let control_path = directory.path().join("control.h5");
    let [completed, restored, untouched] = write_source(&path);
    let parents = [completed, restored, untouched];
    let capacity = u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap();
    fs::copy(&path, &control_path).unwrap();
    let (original, published, control) =
        successful_commit(&control_path, &parents, HeapWrite::Move);
    let file = open_editor(&path, HeapWrite::Move);
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_original_values(&file);
    let mut labels = file.dataset("keep").unwrap();
    let mut numbers = file.dataset("numbers").unwrap();
    stage(&file);

    let event = heap_write(&control, restored, HeapWrite::FreeHeader);
    disk_log::start();
    disk_log::fail_write_after(
        control.events[..event]
            .iter()
            .filter(|event| matches!(event, IoEvent::Write { .. }))
            .count(),
        PARTIAL,
    );
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    let Error::HeapCompactionFailed {
        completed: prefix,
        state,
        source,
    } = error
    else {
        panic!("expected a later collection failure, got {error:?}");
    };
    assert_eq!(prefix, 1);
    assert_eq!(state, HeapCompactionState::Restored);
    assert_io(*source, "injected disk write failure");
    assert_later_failure(&control, &failed, event, &original, completed, restored);

    let after_failure = fs::read(&path).unwrap();
    assert_eq!(
        replay_prefix(&failed, &original, failed.events.len()),
        after_failure
    );
    let mut expected = published;
    for parent in [restored, untouched] {
        let range = parent_range(parent);
        expected[range.clone()].copy_from_slice(&original[range]);
    }
    assert_eq!(after_failure, expected);
    assert_parents(&parents, &original, &after_failure, 1);
    assert_allocated(&file, &parents);
    assert_published_values(&file);
    assert_eq!(labels.read_string().unwrap(), ["retained"; 3]);
    assert_eq!(numbers.read_i32().unwrap(), [23]);
    assert!(!file.has_staged_edits());
    disk_log::start();
    file.commit().unwrap();
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(fs::read(&path).unwrap(), after_failure);

    let replacement = [
        global_heap_fixture::reference(8, completed, 91),
        vec![0; 16],
        global_heap_fixture::reference(8, untouched, 91),
    ]
    .concat();
    let datatype = labels.datatype().unwrap();
    labels
        .write_staged(|builder| {
            builder.with_raw_data(datatype, replacement.clone(), 3);
        })
        .unwrap();
    numbers
        .write_staged(|builder| {
            builder.with_i32_data(&[29]);
        })
        .unwrap();
    assert!(file.has_staged_edits());
    disk_log::start();
    file.commit().unwrap();
    let new_commit = disk_log::finish();
    assert!(!new_commit.operations.is_empty());
    assert!(!file.has_staged_edits());
    let after_followup = fs::read(&path).unwrap();
    assert_eq!(
        replay_prefix(&new_commit, &after_failure, new_commit.events.len()),
        after_followup
    );
    let followup_free = heap_write(&new_commit, restored, HeapWrite::FreeHeader);
    let heap_writes = new_commit
        .events
        .iter()
        .enumerate()
        .filter_map(|(event, operation)| {
            let IoEvent::Write {
                offset, written, ..
            } = operation
            else {
                return None;
            };
            parents
                .iter()
                .any(|parent| {
                    *offset < *parent + capacity
                        && *parent < *offset + u64::try_from(*written).unwrap()
                })
                .then_some(event)
        })
        .collect::<Vec<_>>();
    assert_eq!(heap_writes, [followup_free]);
    assert_eq!(
        new_commit.events[followup_free + 1],
        IoEvent::Synchronize {
            kind: Synchronization::All,
            outcome: IoOutcome::Succeeded,
        }
    );
    // The fresh edit removes member 91. Member 7 had no reference in that edit's source.
    let retained = global_heap::collection(&[Object::new(7, b"removed")], Widths::EIGHT);
    let header = HEADER + RECORD..2 * HEADER + RECORD;
    assert_issued_write(
        &new_commit,
        followup_free,
        HeapWrite::FreeHeader.offset(restored),
        &retained[header.clone()],
        HEADER,
        IoOutcome::Succeeded,
    );
    let range = parent_range(restored);
    let mut expected_parent = after_failure[range.clone()].to_vec();
    expected_parent[header.clone()].copy_from_slice(&retained[header]);
    assert_eq!(after_followup[range], expected_parent);
    assert_unchanged_parents(&after_failure, &after_followup, &[completed, untouched]);
    assert_allocated(&file, &parents);
    assert_eq!(labels.read_raw().unwrap(), replacement);
    assert_eq!(labels.read_string().unwrap(), ["retained", "", "retained"]);
    assert_eq!(numbers.read_i32().unwrap(), [29]);
    drop(labels);
    drop(numbers);
    file.close().unwrap();

    let reopened = File::open(&path).unwrap();
    assert_followup_values(&reopened, &replacement);
    reopened.close().unwrap();
    assert_unchanged_parents(&after_followup, &fs::read(&path).unwrap(), &parents);
}

fn assert_later_failure(
    control: &Recording,
    failed: &Recording,
    event: usize,
    original: &[u8],
    completed: u64,
    restored: u64,
) {
    let completed_free = heap_write(control, completed, HeapWrite::FreeHeader);
    let failed_move = heap_write(control, restored, HeapWrite::Move);
    assert!(completed_free + 1 < failed_move && failed_move < event);
    assert_eq!(
        failed.events[completed_free + 1],
        IoEvent::Synchronize {
            kind: Synchronization::All,
            outcome: IoOutcome::Succeeded,
        }
    );
    assert_eq!(failed.events[..event], control.events[..event]);
    assert_eq!(
        replay_prefix(failed, original, event),
        replay_prefix(control, original, event)
    );
    assert_failed_write(
        control,
        failed,
        event,
        PARTIAL,
        original,
        parent_range(restored),
    );
    let source = parent_range(restored).start;
    for (phase, target, bytes) in [
        (
            event + 1,
            HeapWrite::FreeHeader,
            &original[source + HEADER + RECORD..source + 2 * HEADER + RECORD],
        ),
        (
            event + 2,
            HeapWrite::Move,
            &original[source + HEADER..source + HEADER + RECORD],
        ),
    ] {
        assert_issued_write(
            failed,
            phase,
            target.offset(restored),
            bytes,
            bytes.len(),
            IoOutcome::Succeeded,
        );
    }
    assert_eq!(
        failed.events[event + 3],
        IoEvent::Synchronize {
            kind: Synchronization::Data,
            outcome: IoOutcome::Succeeded,
        }
    );
    assert_eq!(failed.events.len(), event + 4);
}

fn assert_followup_values(file: &File, expected: &[u8]) {
    let labels = file.dataset("keep").unwrap();
    assert_eq!(labels.read_raw().unwrap(), expected);
    assert_eq!(labels.read_string().unwrap(), ["retained", "", "retained"]);
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [29]);
    let value = file.dataset("value").unwrap();
    assert_eq!(value.read_i32().unwrap(), [17]);
    assert_eq!(value.attrs().unwrap().get("a"), Some(&AttrValue::I64(9)));
}

fn assert_unchanged_parents(before: &[u8], after: &[u8], parents: &[u64]) {
    for &parent in parents {
        let range = parent_range(parent);
        assert_eq!(before[range.clone()], after[range], "heap parent {parent}");
    }
}
