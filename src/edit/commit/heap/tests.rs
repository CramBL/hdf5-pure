use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::LengthWidth;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::global_heap;
use test_util::temp::TempPath;
use test_util::widths::Widths;

use crate::File;
use crate::Layout;
use crate::edit::SyncPolicy;
use crate::edit::retirement::RootRetirement;
use crate::image::WriteBuffering;
use crate::writer::FileBuilder;

use super::*;

#[rstest]
#[case::unchanged(&[])]
#[case::first(&[1])]
#[case::middle(&[7])]
#[case::all(&[1, 7, 23])]
fn copies_complete_overlapping_records_and_only_the_free_header(
    #[case] removed: &[u16],
    #[values(false, true)] session: bool,
    #[values(false, true)] mirror: bool,
) {
    let (_path, mut engine, start, original) = fixture(session, mirror);
    let plan = compaction(start, &original, removed);
    let mut expected = original.clone();
    for movement in plan.moves() {
        expected.copy_within(
            local(movement.object().range(), start),
            usize::try_from(movement.destination().start().get() - start.get()).unwrap(),
        );
    }
    if let Some(free) = plan.free_space() {
        expected[local(free.header_range(), start)].copy_from_slice(free.bytes());
    }
    let expected_undo = usize::try_from(
        plan.moves()
            .iter()
            .map(|movement| movement.destination().length())
            .sum::<u64>(),
    )
    .unwrap()
        + plan.free_space().map_or(0, |free| free.bytes().len());
    disk_log::start();
    rootless(&mut engine, &[plan]).unwrap();
    let recording = disk_log::finish();
    let writes = recording
        .events
        .iter()
        .filter(|event| matches!(event, IoEvent::Write { .. }))
        .count();
    assert_eq!(writes == 0, removed.is_empty());
    assert_eq!(
        engine
            .image()
            .read_exact_at(start.get(), original.len())
            .unwrap(),
        expected
    );
    assert_eq!(
        engine.image.disk_bytes().unwrap()[usize::try_from(start.get()).unwrap()..]
            [..original.len()],
        expected
    );
    let work = engine.image.buffering_work();
    assert_eq!((work.pending_runs, work.pending_bytes), (0, 0));
    assert_eq!(work.mirror_bytes > 0, mirror);
    let undo_bytes: usize = engine
        .inplace_undo
        .iter()
        .map(|(_, bytes)| bytes.len())
        .sum();
    assert_eq!(undo_bytes, expected_undo);
    assert!(
        engine
            .inplace_undo
            .iter()
            .all(|(_, bytes)| bytes.len() <= COPY_BYTES)
    );
    assert!(
        engine
            .inplace_undo
            .iter()
            .map(|(_, bytes)| bytes.capacity())
            .sum::<usize>()
            >= undo_bytes
    );
    engine.undo_inplace_writes().unwrap();
    engine.image.flush().unwrap();
    assert_eq!(
        engine
            .image()
            .read_exact_at(start.get(), original.len())
            .unwrap(),
        original
    );
}

#[rstest]
#[case::completion_sync(Failure::Barrier)]
#[case::write(Failure::Write)]
#[case::issue(Failure::Issue)]
#[case::restore(Failure::Restore)]
#[case::restore_issue(Failure::RestoreIssue)]
#[case::restore_sync(Failure::RestoreBarrier)]
fn failure_restores_parent_and_descriptors_or_seals_uncertainty(
    #[case] failure: Failure,
    #[values(false, true)] mirror: bool,
) {
    let session = matches!(failure, Failure::Issue | Failure::RestoreIssue);
    let (_path, mut engine, start, original) = fixture(session, mirror);
    if matches!(failure, Failure::Barrier | Failure::RestoreBarrier) {
        engine.set_sync_policy(SyncPolicy::Always);
    }
    let descriptor = start
        .checked_add(u64::try_from(original.len()).unwrap())
        .unwrap();
    engine.write_inplace_journaled(descriptor, &[9; 8]).unwrap();
    let plan = compaction(start, &original, &[1]);
    disk_log::start();
    match failure {
        Failure::Barrier => disk_log::fail_sync_after(0, FailurePoint::Before),
        Failure::Write | Failure::Issue | Failure::Restore | Failure::RestoreBarrier => {
            disk_log::fail_write_after(0, 2)
        }
        Failure::RestoreIssue => {}
    }
    let result = if matches!(failure, Failure::RestoreIssue) {
        JournalScope::run_rootless(&mut engine, |journal| {
            apply(journal, &plan)?;
            disk_log::fail_write_after(0, 2);
            Err(Error::EditUnsupported("heap mutation failed"))
        })
    } else if matches!(failure, Failure::Restore | Failure::RestoreBarrier) {
        JournalScope::run_rootless(&mut engine, |journal| {
            let error = apply(journal, &plan).unwrap_err();
            if matches!(failure, Failure::RestoreBarrier) {
                disk_log::fail_sync_after(0, FailurePoint::Before);
            } else {
                disk_log::fail_write_after(0, 0);
            }
            Err(error)
        })
    } else {
        rootless(&mut engine, &[plan])
    };
    disk_log::finish();
    let error = result.unwrap_err();
    if matches!(
        failure,
        Failure::Restore | Failure::RestoreBarrier | Failure::RestoreIssue
    ) {
        let Error::CommitPartiallyApplied { refusal, restore } = error else {
            panic!("{error:?}")
        };
        if matches!(failure, Failure::RestoreIssue) {
            assert!(matches!(
                *refusal,
                Error::EditUnsupported("heap mutation failed")
            ));
        } else {
            assert_io(*refusal, "injected disk write failure");
        }
        assert_io(
            *restore,
            if matches!(failure, Failure::RestoreBarrier) {
                "injected disk synchronization failure"
            } else {
                "injected disk write failure"
            },
        );
        assert!(matches!(
            engine.context.allocations.available().unwrap_err(),
            Error::AllocationPublicationUncertain
        ));
    } else {
        assert_io(
            error,
            if matches!(failure, Failure::Barrier) {
                "injected disk synchronization failure"
            } else {
                "injected disk write failure"
            },
        );
        assert_eq!(
            engine
                .image()
                .read_exact_at(start.get(), original.len())
                .unwrap(),
            original
        );
        assert_eq!(
            engine.image().read_exact_at(descriptor.get(), 8).unwrap(),
            [0; 8]
        );
        assert_eq!(engine.image.buffering_work().pending_bytes, 0);
        engine.context.allocations.available().unwrap();
    }
    assert!(engine.inplace_undo.is_empty());
}

#[test]
fn streamed_source_and_undo_reads_have_separate_byte_costs() {
    let (path, engine, start, original) = fixture(false, false);
    drop(engine);
    let reads = Arc::new(AtomicU64::new(0));
    let mut engine = WriteEngine::open_bounded_counting(&path, reads.clone()).unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let plan = compaction(start, &original, &[1]);
    let moved = plan
        .moves()
        .iter()
        .map(|movement| movement.object().range().length())
        .sum::<u64>();
    let header = u64::try_from(plan.free_space().unwrap().bytes().len()).unwrap();
    reads.store(0, Ordering::Relaxed);
    rootless(&mut engine, &[plan]).unwrap();
    assert_eq!(reads.load(Ordering::Relaxed), moved * 2 + header);
    assert_eq!(
        engine
            .inplace_undo
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .sum::<u64>(),
        moved + header
    );
    assert!(moved > COPY_BYTES as u64);
    assert!(moved + header < original.len() as u64);
    assert_eq!(engine.image.buffering_work().write_bytes, moved + header);
}

#[test]
fn source_read_failure_precedes_any_destination_write() {
    let (path, mut engine, start, original) = fixture(false, false);
    let plan = compaction(start, &original, &[1]);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(start.get() + 40)
        .unwrap();
    let error = rootless(&mut engine, &[plan]).unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::Source(_))),
        "{error:?}"
    );
    assert!(engine.inplace_undo.is_empty());
    assert_eq!(engine.image.buffering_work().write_bytes, 0);
}

#[test]
fn later_parent_failure_preserves_a_published_prefix_and_the_prior_journal() {
    let (_path, mut engine, start, original) = fixture(false, false);
    let plan = compaction(start, &original, &[1]);
    let invalid = compaction(
        start
            .checked_add(u64::try_from(original.len()).unwrap())
            .unwrap(),
        &original,
        &[1],
    );
    let unchanged_prefix =
        0..usize::try_from(plan.moves()[0].destination().start().get() - start.get()).unwrap();
    let descriptor = start
        .checked_add(u64::try_from(original.len()).unwrap())
        .unwrap();
    let invalid_end = descriptor
        .checked_add(u64::try_from(original.len()).unwrap())
        .unwrap()
        .get();
    engine.write_inplace_journaled(descriptor, &[9; 8]).unwrap();
    let prefix = engine.inplace_undo[0].1.as_ptr();
    let end = engine.image().len();
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let pending = RootRetirement::new(root, Vec::new()).into_pending_frees(Vec::new(), end);
    let failure = engine
        .publish_pending_tree_with_completion(pending, BTreeMap::new(), |published| {
            Ok(after_publication(published, &[plan, invalid]).unwrap_err())
        })
        .unwrap();
    assert_eq!(failure.completed, 1);
    assert_eq!(failure.state, HeapMutationState::Restored);
    let Error::Format(FormatError::UnexpectedSourceEof {
        expected,
        available,
    }) = failure.error
    else {
        panic!("{:?}", failure.error)
    };
    assert_eq!((expected, available), (invalid_end, end));
    assert_eq!(engine.inplace_undo.len(), 1);
    assert_eq!(engine.inplace_undo[0].1.as_ptr(), prefix);
    let current = engine
        .image()
        .read_exact_at(start.get(), original.len())
        .unwrap();
    assert_ne!(current, original);
    assert_eq!(
        &current[unchanged_prefix.clone()],
        &original[unchanged_prefix]
    );
}

#[test]
fn lost_postpublication_permission_preserves_bytes_and_records_uncertainty() {
    let (_path, mut engine, start, original) = fixture(false, false);
    let plan = compaction(start, &original, &[1]);
    let end = engine.image().len();
    engine.make_allocation_uncertain();
    let failure = published_batch(&mut engine, &[plan], end).unwrap_err();
    assert_eq!(failure.completed, 0);
    assert_eq!(failure.state, HeapMutationState::Uncertain);
    assert!(matches!(
        failure.error,
        Error::AllocationPublicationUncertain
    ));
    assert_eq!(
        engine
            .image()
            .read_exact_at(start.get(), original.len())
            .unwrap(),
        original
    );
    assert!(engine.inplace_undo.is_empty());
}

#[test]
fn later_rootless_failure_restores_every_parent_before_original_descriptors() {
    let (_path, mut engine, start, original) = fixture(true, false);
    let descriptor = start
        .checked_add(u64::try_from(original.len()).unwrap())
        .unwrap();
    engine.write_inplace_journaled(descriptor, &[9; 8]).unwrap();
    let plans = [
        compaction(start, &original, &[1]),
        compaction(descriptor, &original, &[1]),
    ];
    let invalid_end = descriptor
        .checked_add(u64::try_from(original.len()).unwrap())
        .unwrap()
        .get();
    let source_end = engine.image().len();
    let error = rootless(&mut engine, &plans).unwrap_err();
    let Error::Format(FormatError::UnexpectedSourceEof {
        expected,
        available,
    }) = error
    else {
        panic!("{error:?}")
    };
    assert_eq!((expected, available), (invalid_end, source_end));
    assert_eq!(
        engine
            .image()
            .read_exact_at(start.get(), original.len())
            .unwrap(),
        original
    );
    assert_eq!(
        engine.image().read_exact_at(descriptor.get(), 8).unwrap(),
        [0; 8]
    );
    assert!(engine.inplace_undo.is_empty());
    assert_eq!(engine.image.buffering_work().pending_bytes, 0);
}

fn fixture(session: bool, mirror: bool) -> (TempPath, WriteEngine, PhysicalAddress, Vec<u8>) {
    let payload = (0..12_000)
        .map(|n| u8::try_from(n % 251).unwrap())
        .collect::<Vec<_>>();
    let mut bytes = global_heap::collection(
        &[
            global_heap::Object::new(1, b"first"),
            global_heap::Object {
                index: 7,
                reference_count: 9,
                data: &payload,
            },
            global_heap::Object::new(23, b"last"),
        ],
        Widths::EIGHT,
    );
    bytes[39] = 0x77;
    let path = test_util::temp::temp_path("heap-mutation.h5");
    let mut builder = FileBuilder::new();
    let mut data = bytes.clone();
    data.extend_from_slice(&[0; 8]);
    builder.create_dataset("bytes").with_u8_data(&data);
    builder.write(&path).unwrap();
    let start = {
        let file = File::open(&path).unwrap();
        let Layout::Contiguous {
            address: Some(start),
            ..
        } = file.dataset("bytes").unwrap().layout().unwrap()
        else {
            panic!("expected contiguous bytes")
        };
        PhysicalAddress::new(start)
    };
    let mut engine = if mirror {
        WriteEngine::open_source_only(&path).unwrap()
    } else {
        WriteEngine::open_sync_counting(&path, SyncPolicy::OnClose, Arc::new(AtomicU64::new(0)))
            .unwrap()
    };
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(if session {
            WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 1_000_000,
            }
        } else {
            WriteBuffering::Unbuffered
        })
        .unwrap();
    (path, engine, start, bytes)
}

fn compaction(start: PhysicalAddress, bytes: &[u8], removed: &[u16]) -> GlobalHeapCompaction {
    let file = FileRegion::new(
        FileRange::new(start, u64::try_from(bytes.len()).unwrap()).unwrap(),
        bytes,
    )
    .unwrap();
    let locator = GlobalHeapCollectionLocator::new(start, LengthWidth::Eight);
    let collection = locator
        .parse(file.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap();
    let mut collector = collection.collect_members();
    while let Some(range) = collector.parse_range() {
        collector.parse(file.subregion(range).unwrap()).unwrap();
    }
    collector
        .finish()
        .unwrap()
        .compact(|object| removed.contains(&object.index().get()))
        .unwrap()
}

fn local(range: FileRange, start: PhysicalAddress) -> core::ops::Range<usize> {
    usize::try_from(range.start().get() - start.get()).unwrap()
        ..usize::try_from(range.end().get() - start.get()).unwrap()
}
fn assert_io(error: Error, expected: &str) {
    let Error::Io(error) = error else {
        panic!("{error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), expected);
}
#[derive(Clone, Copy)]
enum Failure {
    Barrier,
    Issue,
    Restore,
    RestoreBarrier,
    RestoreIssue,
    Write,
}

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}
