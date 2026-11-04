use core::num::NonZeroU32;
use std::collections::BTreeMap;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::ffi::CString;
use std::io::ErrorKind;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use hdf5_pure_format::__private::ObjectReferenceCountUpdate;
use hdf5_pure_format::__private::ObjectReferenceCountUpdateError;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Synchronization;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::super::super::super::tests as protection_tests;
use super::super::super::test_support;
use super::super::super::test_support::Form;
use super::super::tests as completion_tests;
use super::CountPreparationError;
use super::CountWriteFailure;
use crate::UserblockSize;
use crate::edit::SyncPolicy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::commit::PublicationState;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;

#[rstest]
#[case::v1(Form::Version1)]
#[case::v2(Form::Version2)]
#[case::continuation(Form::Continuation)]
fn count_writes_preserve_current_content_and_prior_journal_entries(
    #[case] form: Form,
    #[values(0, 512)] base: u64,
    #[values(2, u32::MAX)] count: u32,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let path = test_util::temp::temp_path("published-count-writes.h5");
    let fixture = test_support::fixture(&path, form, base, NonZeroU32::new(count).unwrap(), c"old");
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(policy);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let bounds = completion_tests::bounds(&mut engine, fixture.retired.to_vec(), &["a"]);
    let root = bounds.root();
    let end = bounds.end();
    let current = form.chunk(count, c"new");
    bounds
        .phase
        .engine
        .write_inplace_journaled(fixture.allocation.start().get(), &current)
        .unwrap();
    let pointer = bounds.phase.engine.inplace_undo[0].1.as_ptr();
    let field = bounds.updates()[0].field();
    let recording = bounds
        .with_publication(BTreeMap::new(), |scope| {
            let scope = scope.unwrap();
            assert_eq!(scope.root, root);
            assert_eq!(scope.end, end);
            disk_log::start();
            scope.write_counts().unwrap();
            Ok(disk_log::finish())
        })
        .unwrap();
    assert_eq!(
        engine
            .image()
            .read_exact_at(
                fixture.allocation.start().get(),
                usize::try_from(fixture.allocation.length()).unwrap(),
            )
            .unwrap(),
        form.chunk(count - 1, c"new")
    );
    assert_eq!(engine.inplace_undo.len(), 1);
    assert_eq!(
        engine.inplace_undo[0],
        (fixture.allocation.start().get(), form.chunk(count, c"old"))
    );
    assert_eq!(engine.inplace_undo[0].1.as_ptr(), pointer);
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(
        engine
            .addresses()
            .resolve(engine.superblock().root_group_address)
            .unwrap(),
        root
    );
    assert_eq!(engine.superblock().eof_address, end.get());
    engine.context.allocations.before_reference().unwrap();
    let writes = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                offset,
                requested,
                outcome,
                ..
            } => Some((*offset, *requested, *outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut expected = vec![(
        field.range().start().get(),
        usize::try_from(field.range().length()).unwrap(),
        IoOutcome::Succeeded,
    )];
    if !matches!(form, Form::Version1) {
        expected.push((
            field.chunk().end().get() - u64::try_from(test_util::checksum::CHECKSUM).unwrap(),
            test_util::checksum::CHECKSUM,
            IoOutcome::Succeeded,
        ));
    }
    assert_eq!(writes, expected);
    assert_eq!(
        recording
            .events
            .iter()
            .filter_map(|event| match event {
                IoEvent::Synchronize { kind, outcome } => Some((*kind, *outcome)),
                _ => None,
            })
            .collect::<Vec<_>>(),
        if matches!(policy, SyncPolicy::Always) {
            vec![(Synchronization::All, IoOutcome::Succeeded)]
        } else {
            vec![]
        }
    );
}

#[rstest]
#[case::count_before(Outcome::Count(Prefix::Before))]
#[case::count_partial(Outcome::Count(Prefix::Partial))]
#[case::count_complete(Outcome::Count(Prefix::Complete))]
#[case::checksum_before(Outcome::Checksum(Prefix::Before))]
#[case::checksum_partial(Outcome::Checksum(Prefix::Partial))]
#[case::checksum_complete(Outcome::Checksum(Prefix::Complete))]
#[case::barrier_before(Outcome::Barrier(FailurePoint::Before))]
#[case::barrier_after(Outcome::Barrier(FailurePoint::After))]
#[case::restoration_write(Outcome::RestoreWrite)]
#[case::restoration_barrier(Outcome::RestoreBarrier)]
fn failed_count_writes_retain_retry_or_seal_uncertain_storage(
    #[case] outcome: Outcome,
    #[values(Form::Version2, Form::Continuation)] form: Form,
) {
    let path = test_util::temp::temp_path("published-count-failure.h5");
    let fixture = test_support::fixture(&path, form, 0, NonZeroU32::new(2).unwrap(), c"old");
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let bounds = completion_tests::bounds(&mut engine, fixture.retired.to_vec(), &["a"]);
    let root = bounds.root();
    let end = bounds.end();
    let current = form.chunk(2, c"new");
    bounds
        .phase
        .engine
        .write_inplace_journaled(fixture.allocation.start().get(), &current)
        .unwrap();
    let pointer = bounds.phase.engine.inplace_undo[0].1.as_ptr();
    let restored = !matches!(outcome, Outcome::RestoreWrite | Outcome::RestoreBarrier);
    let recording = bounds
        .with_publication(BTreeMap::new(), |scope| {
            let scope = scope.unwrap();
            let free = scope.engine.space.reusable_sections().collect::<Vec<_>>();
            disk_log::start();
            outcome.inject();
            let failure = scope.write_counts().unwrap_err();
            let recording = disk_log::finish();
            match *failure {
                CountWriteFailure::Restored { writes, error } => {
                    assert!(restored);
                    assert_io(
                        error,
                        if matches!(outcome, Outcome::Count(_) | Outcome::Checksum(_)) {
                            "injected disk write failure"
                        } else {
                            "injected disk synchronization failure"
                        },
                    );
                    assert_eq!(writes.updates.len(), 1);
                    assert_eq!(
                        writes
                            .scope
                            .engine
                            .space
                            .reusable_sections()
                            .collect::<Vec<_>>(),
                        free
                    );
                    assert_eq!(
                        writes
                            .scope
                            .engine
                            .image()
                            .read_exact_at(fixture.allocation.start().get(), current.len(),)
                            .unwrap(),
                        current
                    );
                    assert_eq!(writes.scope.engine.inplace_undo.len(), 1);
                    assert_eq!(writes.scope.engine.inplace_undo[0].1.as_ptr(), pointer);
                    writes.apply().unwrap();
                }
                CountWriteFailure::Uncertain { error } => {
                    assert!(!restored);
                    let Error::CommitPartiallyApplied { refusal, restore } = error else {
                        panic!("expected failed count restoration, got {error:?}")
                    };
                    assert_io(
                        *refusal,
                        if matches!(outcome, Outcome::RestoreWrite) {
                            "injected disk synchronization failure"
                        } else {
                            "injected disk write failure"
                        },
                    );
                    assert_io(
                        *restore,
                        if matches!(outcome, Outcome::RestoreWrite) {
                            "injected disk write failure"
                        } else {
                            "injected disk synchronization failure"
                        },
                    );
                }
                other => panic!("unexpected count failure: {other:?}"),
            }
            Ok((recording, free))
        })
        .unwrap();
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(
        engine
            .addresses()
            .resolve(engine.superblock().root_group_address)
            .unwrap(),
        root
    );
    assert_eq!(engine.superblock().eof_address, end.get());
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        recording.1
    );
    assert_eq!(engine.inplace_undo.len(), 1);
    assert_eq!(engine.inplace_undo[0].1.as_ptr(), pointer);
    if restored {
        engine.context.allocations.before_reference().unwrap();
        assert_eq!(
            engine
                .image()
                .read_exact_at(fixture.allocation.start().get(), current.len(),)
                .unwrap(),
            form.chunk(1, c"new")
        );
    } else {
        let error = engine.context.allocations.before_reference().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
        let error = engine.commit().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    }
    let syncs = recording
        .0
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Synchronize { kind, outcome } => Some((*kind, *outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let success = IoOutcome::Succeeded;
    let failed = IoOutcome::Failed(ErrorKind::Other);
    assert_eq!(
        syncs,
        match outcome {
            Outcome::Count(_) | Outcome::Checksum(_) => vec![(Synchronization::Data, success)],
            Outcome::Barrier(_) => vec![
                (Synchronization::All, failed),
                (Synchronization::Data, success)
            ],
            Outcome::RestoreWrite => vec![(Synchronization::All, failed)],
            Outcome::RestoreBarrier => vec![(Synchronization::Data, failed)],
        }
    );
}

#[rstest]
#[case::before(0)]
#[case::partial(2)]
#[case::complete(usize::MAX)]
fn failed_version_1_count_writes_can_retry(#[case] prefix: usize) {
    let path = test_util::temp::temp_path("published-v1-count-failure.h5");
    let fixture = test_support::fixture(
        &path,
        Form::Version1,
        0,
        NonZeroU32::new(2).unwrap(),
        c"old",
    );
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    completion_tests::bounds(&mut engine, fixture.retired.to_vec(), &["a"])
        .with_publication(BTreeMap::new(), |scope| {
            disk_log::fail_write_after(0, prefix);
            let failure = scope.unwrap().write_counts().unwrap_err();
            let CountWriteFailure::Restored { writes, error } = *failure else {
                panic!("expected restored version 1 count")
            };
            assert_io(error, "injected disk write failure");
            assert_eq!(
                writes
                    .scope
                    .engine
                    .image()
                    .read_exact_at(
                        fixture.allocation.start().get(),
                        usize::try_from(fixture.allocation.length()).unwrap(),
                    )
                    .unwrap(),
                Form::Version1.chunk(2, c"old")
            );
            writes.apply().unwrap();
            Ok(())
        })
        .unwrap();
    assert_eq!(
        engine
            .image()
            .read_exact_at(
                fixture.allocation.start().get(),
                usize::try_from(fixture.allocation.length()).unwrap(),
            )
            .unwrap(),
        Form::Version1.chunk(1, c"old")
    );
}

#[test]
fn preparation_failure_preserves_the_confirmed_scope_and_original_count() {
    let path = test_util::temp::temp_path("published-count-source-failure.h5");
    let fixture = test_support::fixture(
        &path,
        Form::Version2,
        0,
        NonZeroU32::new(2).unwrap(),
        c"old",
    );
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let bounds = completion_tests::bounds(&mut engine, fixture.retired.to_vec(), &["a"]);
    let field = bounds.updates()[0].field();
    bounds
        .phase
        .engine
        .write_inplace_journaled(
            fixture.allocation.start().get(),
            &Form::Version2.chunk(3, c"new"),
        )
        .unwrap();
    let pointer = bounds.phase.engine.inplace_undo[0].1.as_ptr();
    bounds
        .with_publication(BTreeMap::new(), |scope| {
            disk_log::start();
            let failure = scope.unwrap().write_counts().unwrap_err();
            let recording = disk_log::finish();
            assert!(recording.events.is_empty());
            let CountWriteFailure::Preparation { scope, error } = *failure else {
                panic!("expected current-source preparation failure")
            };
            let CountPreparationError::Update { object, error } = error else {
                panic!("expected changed count, got {error:?}")
            };
            assert_eq!(object, fixture.object);
            assert_eq!(
                error,
                ObjectReferenceCountUpdateError::Value {
                    field: field.range().start(),
                    expected: 2,
                    actual: 3,
                }
            );
            assert_eq!(scope.engine.inplace_undo.len(), 1);
            assert_eq!(scope.engine.inplace_undo[0].1.as_ptr(), pointer);
            scope.engine.context.allocations.before_reference().unwrap();
            assert_eq!(
                scope
                    .engine
                    .image()
                    .read_exact_at(
                        fixture.allocation.start().get(),
                        usize::try_from(fixture.allocation.length()).unwrap(),
                    )
                    .unwrap(),
                Form::Version2.chunk(3, c"new")
            );
            Ok(())
        })
        .unwrap();
}

#[rstest]
#[case::empty(false)]
#[case::writing(true)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn count_writes_skip_empty_work_and_release_metadata_buffers(
    #[case] writing: bool,
    #[values(1_024, 32_768)] body: usize,
) {
    let path = test_util::temp::temp_path("published-count-cost.h5");
    let comment = CString::new(vec![b'x'; body]).unwrap();
    let fixture = test_support::fixture(
        &path,
        Form::Version2,
        0,
        NonZeroU32::new(2).unwrap(),
        &comment,
    );
    let (mut engine, calls) = protection_tests::engine_with_read_call_counter(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let names = if writing { &["a"][..] } else { &["a", "b"][..] };
    let measured = completion_tests::bounds(&mut engine, fixture.retired.to_vec(), names)
        .with_publication(BTreeMap::new(), |scope| {
            let scope = scope.unwrap();
            calls.store(0, Ordering::Relaxed);
            DISCOVERY_WORK.with(|work| work.set((0, 0)));
            let (result, measured) =
                allocation::measure("published-count-writes", || scope.write_counts());
            result.unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), if writing { 3 } else { 0 });
            DISCOVERY_WORK.with(|work| assert_eq!(work.get(), (0, 0)));
            Ok(measured)
        })
        .unwrap();
    if writing {
        assert!(
            measured.blocks > 0 && measured.bytes > u64::try_from(body).unwrap(),
            "{measured}"
        );
        assert!(
            measured.peak_bytes < u64::try_from(body * 3 + 1_024).unwrap(),
            "{measured}"
        );
        assert!(
            measured.live_bytes
                <= u64::try_from(
                    engine.inplace_undo.capacity() * core::mem::size_of::<(u64, Vec<u8>)>()
                )
                .unwrap()
                    + engine.image.buffering_work().write_calls
                        * 2
                        * u64::try_from(core::mem::size_of::<(u64, u64)>()).unwrap(),
            "{measured}"
        );
    } else {
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(measured.bytes, 0, "{measured}");
    }
}

#[rstest]
#[case::preparation(BatchFailure::Preparation)]
#[case::write(BatchFailure::Write)]
fn a_later_batch_failure_does_not_leave_an_earlier_count_changed(#[case] failure: BatchFailure) {
    let path = test_util::temp::temp_path("published-count-batch-failure.h5");
    let mut fixture = Fixture::new();
    let retired = protection_tests::append_retirements(&mut fixture, UserblockSize::NONE);
    let chunk = Form::Version2.chunk(2, c"old");
    let mut addresses = Vec::new();
    for (removed, kept) in [("a", "b"), ("c", "d")] {
        let address = u64::try_from(fixture.image.append(&chunk)).unwrap();
        fixture.link(removed, address);
        fixture.link(kept, address);
        addresses.push(address);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let bounds = completion_tests::bounds(&mut engine, retired.to_vec(), &["a", "c"]);
    let first = bounds.updates()[0].field();
    let second = bounds.updates()[1].field();
    let second_object = bounds.updates()[1].object();
    bounds
        .phase
        .engine
        .write_inplace_journaled(addresses[0], &Form::Version2.chunk(2, c"new"))
        .unwrap();
    if matches!(failure, BatchFailure::Preparation) {
        bounds
            .phase
            .engine
            .write_inplace_journaled(addresses[1], &Form::Version2.chunk(3, c"new"))
            .unwrap();
    }
    let prefix = bounds.phase.engine.inplace_undo.len();
    let recording = bounds
        .with_publication(BTreeMap::new(), |scope| {
            disk_log::start();
            if matches!(failure, BatchFailure::Write) {
                disk_log::fail_write_after(2, 2);
            }
            let result = scope.unwrap().write_counts().unwrap_err();
            let recording = disk_log::finish();
            match *result {
                CountWriteFailure::Preparation { scope, error } => {
                    assert!(matches!(failure, BatchFailure::Preparation));
                    let CountPreparationError::Update { object, error } = error else {
                        panic!("expected later count mismatch, got {error:?}")
                    };
                    assert_eq!(object, second_object);
                    assert_eq!(
                        error,
                        ObjectReferenceCountUpdateError::Value {
                            field: second.range().start(),
                            expected: 2,
                            actual: 3,
                        }
                    );
                    assert_eq!(scope.engine.inplace_undo.len(), prefix);
                    assert_eq!(
                        scope
                            .engine
                            .image()
                            .read_exact_at(addresses[0], chunk.len())
                            .unwrap(),
                        Form::Version2.chunk(2, c"new")
                    );
                    assert!(recording.events.is_empty());
                }
                CountWriteFailure::Restored { writes, error } => {
                    assert!(matches!(failure, BatchFailure::Write));
                    assert_io(error, "injected disk write failure");
                    assert_eq!(writes.updates.len(), 2);
                    assert_eq!(writes.scope.engine.inplace_undo.len(), prefix);
                    assert_eq!(
                        writes
                            .scope
                            .engine
                            .image()
                            .read_exact_at(addresses[0], chunk.len())
                            .unwrap(),
                        Form::Version2.chunk(2, c"new")
                    );
                    assert_eq!(
                        writes
                            .scope
                            .engine
                            .image()
                            .read_exact_at(addresses[1], chunk.len())
                            .unwrap(),
                        chunk
                    );
                    let writes_at = recording
                        .events
                        .iter()
                        .filter_map(|event| match event {
                            IoEvent::Write { offset, .. } => Some(*offset),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let checksum = first.chunk().end().get()
                        - u64::try_from(test_util::checksum::CHECKSUM).unwrap();
                    assert_eq!(
                        writes_at,
                        [
                            first.range().start().get(),
                            checksum,
                            second.range().start().get(),
                            second.range().start().get(),
                            checksum,
                            first.range().start().get()
                        ]
                    );
                    writes.apply().unwrap();
                }
                other => panic!("unexpected batch failure: {other:?}"),
            }
            Ok(recording)
        })
        .unwrap();
    assert_eq!(engine.inplace_undo.len(), prefix);
    assert_eq!(
        engine
            .image()
            .read_exact_at(addresses[0], chunk.len())
            .unwrap(),
        Form::Version2.chunk(
            if matches!(failure, BatchFailure::Write) {
                1
            } else {
                2
            },
            c"new",
        )
    );
    if matches!(failure, BatchFailure::Write) {
        assert!(!recording.events.is_empty());
        assert_eq!(
            engine
                .image()
                .read_exact_at(addresses[1], chunk.len())
                .unwrap(),
            Form::Version2.chunk(1, c"old")
        );
    }
}

#[rstest]
#[case::retry(false)]
#[case::uncertain(true)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn restored_compact_updates_skip_header_reads_and_require_confirmed_storage(
    #[case] uncertain: bool,
) {
    let path = test_util::temp::temp_path("published-count-retry-cost.h5");
    let comment = CString::new(vec![b'x'; 32_768]).unwrap();
    let fixture = test_support::fixture(
        &path,
        Form::Version2,
        0,
        NonZeroU32::new(2).unwrap(),
        &comment,
    );
    let (mut engine, calls) = protection_tests::engine_with_read_call_counter(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    completion_tests::bounds(&mut engine, fixture.retired.to_vec(), &["a"])
        .with_publication(BTreeMap::new(), |scope| {
            calls.store(0, Ordering::Relaxed);
            disk_log::fail_write_after(1, 2);
            let (writes, measured) = allocation::measure("published-count-restored", || {
                let failure = scope.unwrap().write_counts().unwrap_err();
                let CountWriteFailure::Restored { writes, error } = *failure else {
                    panic!("expected compact retry state")
                };
                assert_io(error, "injected disk write failure");
                writes
            });
            assert!(measured.blocks > 0 && measured.bytes > 32_768, "{measured}");
            assert_eq!(writes.updates.len(), 1);
            assert!(
                measured.live_bytes
                    <= u64::try_from(
                        core::mem::size_of::<ObjectReferenceCountUpdate>()
                            * writes.updates.capacity()
                    )
                    .unwrap()
                        + writes.scope.engine.image.buffering_work().write_calls
                            * 2
                            * u64::try_from(core::mem::size_of::<(u64, u64)>()).unwrap(),
                "{measured}"
            );
            assert_eq!(calls.load(Ordering::Relaxed), 3);
            if uncertain {
                writes.scope.engine.make_allocation_uncertain();
            }
            calls.store(0, Ordering::Relaxed);
            disk_log::start();
            let result = writes.apply();
            let recording = disk_log::finish();
            if uncertain {
                let failure = result.unwrap_err();
                let CountWriteFailure::Uncertain { error } = *failure else {
                    panic!("expected uncertain retry rejection")
                };
                assert!(
                    matches!(error, Error::AllocationPublicationUncertain),
                    "{error:?}"
                );
                assert_eq!(calls.load(Ordering::Relaxed), 0);
                assert!(recording.events.is_empty());
            } else {
                result.unwrap();
                assert_eq!(calls.load(Ordering::Relaxed), 2);
                assert_eq!(
                    recording
                        .events
                        .iter()
                        .filter(|event| matches!(event, IoEvent::Write { .. }))
                        .count(),
                    2
                );
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(
        engine
            .image()
            .read_exact_at(
                fixture.allocation.start().get(),
                usize::try_from(fixture.allocation.length()).unwrap(),
            )
            .unwrap(),
        Form::Version2.chunk(if uncertain { 2 } else { 1 }, &comment)
    );
}

#[derive(Clone, Copy)]
enum BatchFailure {
    Preparation,
    Write,
}

fn assert_io(error: Error, expected: &str) {
    let Error::Io(error) = error else {
        panic!("expected I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), expected);
}

#[derive(Clone, Copy)]
enum Outcome {
    Barrier(FailurePoint),
    Checksum(Prefix),
    Count(Prefix),
    RestoreBarrier,
    RestoreWrite,
}

impl Outcome {
    fn inject(self) {
        match self {
            Self::Barrier(point) => disk_log::fail_sync_after(0, point),
            Self::RestoreBarrier => {
                disk_log::fail_write_after(1, 2);
                disk_log::fail_sync_after(0, FailurePoint::Before);
            }
            Self::RestoreWrite => {
                disk_log::fail_sync_after(0, FailurePoint::Before);
                disk_log::fail_write_after(2, 0);
            }
            Self::Checksum(prefix) => disk_log::fail_write_after(1, prefix.written()),
            Self::Count(prefix) => disk_log::fail_write_after(0, prefix.written()),
        }
    }
}

#[derive(Clone, Copy)]
enum Prefix {
    Before,
    Complete,
    Partial,
}

impl Prefix {
    fn written(self) -> usize {
        match self {
            Self::Before => 0,
            Self::Complete => usize::MAX,
            Self::Partial => 2,
        }
    }
}

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}
