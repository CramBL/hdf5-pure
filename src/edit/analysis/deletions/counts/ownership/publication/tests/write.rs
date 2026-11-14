use core::num::NonZeroU32;
use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_format::__private::ObjectReferenceCountField;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Recording;
use test_util::disk_log::Synchronization;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::super::super::tests as protection_tests;
use super::super::CountPublicationBounds;
use super::super::test_util as count_test_util;
use super::super::test_util::CountFixture;
use super::super::test_util::Form;
use super::completion as completion_tests;
use crate::Error;
use crate::PhysicalAddress;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::edit::WriteEngine;
use crate::edit::commit::PublicationState;
use crate::edit::test_util as edit_test_util;
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
    let case = CountCase::new(&path, form, base, count);
    let mut engine = open(&path, policy);
    let bounds = case.prepare(&mut engine);
    let root = bounds.root();
    let end = bounds.end();
    let field = bounds.updates()[0].field();
    let pointer = bounds.phase.engine.inplace_undo[0].1.as_ptr();
    disk_log::start();
    bounds
        .with_ordered_counts(BTreeMap::new(), |_| Ok(()))
        .unwrap();
    let recording = disk_log::finish();
    case.assert_current(&engine, count - 1);
    assert_eq!(engine.inplace_undo.len(), 1);
    assert_eq!(
        engine.inplace_undo[0],
        (case.fixture.allocation.start(), form.chunk(count, c"old"))
    );
    assert_eq!(engine.inplace_undo[0].1.as_ptr(), pointer);
    assert_publication(&engine, root, end);
    engine.context.allocations.before_reference().unwrap();
    let events = count_events(&recording, field, base);
    let writes = events
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
            checksum(field),
            test_util::checksum::CHECKSUM,
            IoOutcome::Succeeded,
        ));
    }
    assert_eq!(writes, expected);
    assert_eq!(
        syncs(events),
        if matches!(policy, SyncPolicy::Always) {
            vec![(Synchronization::All, IoOutcome::Succeeded)]
        } else {
            vec![]
        }
    );
}

#[rstest]
#[case::v1(Form::Version1)]
#[case::v2(Form::Version2)]
#[case::continuation(Form::Continuation)]
fn empty_count_batches_publish_without_touching_original_headers(#[case] form: Form) {
    let path = test_util::temp::temp_path("empty-published-counts.h5");
    let case = CountCase::new(&path, form, 0, 2);
    let mut engine = open(&path, SyncPolicy::Always);
    let bounds = completion_tests::bounds(&mut engine, case.fixture.retired.to_vec(), &["a", "b"]);
    assert!(bounds.updates().is_empty());
    let root = bounds.root();
    let end = bounds.end();
    disk_log::start();
    bounds
        .with_ordered_counts(BTreeMap::new(), |_| Ok(()))
        .unwrap();
    let recording = disk_log::finish();
    assert_publication(&engine, root, end);
    let allocation = case.fixture.allocation;
    assert_eq!(
        engine
            .image()
            .read_exact_at(
                allocation.start().get(),
                usize::try_from(allocation.length()).unwrap()
            )
            .unwrap(),
        form.chunk(2, c"old")
    );
    assert!(recording.events.iter().all(|event| match event {
        IoEvent::Write {
            offset, requested, ..
        } => {
            let end = offset
                .checked_add(u64::try_from(*requested).unwrap())
                .unwrap();
            end <= allocation.start().get() || *offset >= allocation.end().get()
        }
        _ => true,
    }));
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
fn failed_count_writes_restore_the_batch_or_mark_allocation_uncertain(
    #[case] outcome: Outcome,
    #[values(Form::Version2, Form::Continuation)] form: Form,
) {
    let path = test_util::temp::temp_path("published-count-failure.h5");
    let case = CountCase::new(&path, form, 0, 2);
    let mut engine = open(&path, SyncPolicy::Always);
    let bounds = case.prepare(&mut engine);
    let field = bounds.updates()[0].field();
    disk_log::start();
    bounds
        .with_ordered_counts(BTreeMap::new(), |_| Ok(()))
        .unwrap();
    let control = disk_log::finish();
    let free = engine.space.reusable_sections().collect::<Vec<_>>();
    drop(engine);
    fs::write(&path, &case.original).unwrap();
    let mut engine = open(&path, SyncPolicy::Always);
    let bounds = case.prepare(&mut engine);
    let root = bounds.root();
    let end = bounds.end();
    let pointer = bounds.phase.engine.inplace_undo[0].1.as_ptr();
    disk_log::start();
    let failed_event = outcome.inject(&control, field);
    let error = bounds
        .with_ordered_counts(BTreeMap::new(), |_| -> Result<(), Error> {
            panic!("count failure reached completion")
        })
        .unwrap_err();
    disk_log::clear_failures();
    let recording = disk_log::finish();
    assert_eq!(
        recording.events[failed_event].outcome(),
        IoOutcome::Failed(ErrorKind::Other)
    );
    outcome.assert_error(error);
    assert_publication(&engine, root, end);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), free);
    assert_eq!(engine.inplace_undo.len(), 1);
    assert_eq!(engine.inplace_undo[0].1.as_ptr(), pointer);
    if matches!(outcome, Outcome::RestoreWrite | Outcome::RestoreBarrier) {
        assert_uncertain(engine.context.allocations.before_reference().unwrap_err());
        assert_uncertain(engine.commit().unwrap_err());
    } else {
        engine.context.allocations.before_reference().unwrap();
        case.assert_current(&engine, 2);
    }
    let success = IoOutcome::Succeeded;
    let failed = IoOutcome::Failed(ErrorKind::Other);
    assert_eq!(
        syncs(count_events(&recording, field, 0)),
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
#[case::before(Prefix::Before)]
#[case::partial(Prefix::Partial)]
#[case::complete(Prefix::Complete)]
fn failed_version_1_count_writes_restore_current_bytes(#[case] prefix: Prefix) {
    let path = test_util::temp::temp_path("published-v1-count-failure.h5");
    let case = CountCase::new(&path, Form::Version1, 0, 2);
    let mut engine = open(&path, SyncPolicy::Always);
    let bounds = case.prepare(&mut engine);
    let field = bounds.updates()[0].field();
    disk_log::start();
    bounds
        .with_ordered_counts(BTreeMap::new(), |_| Ok(()))
        .unwrap();
    let control = disk_log::finish();
    drop(engine);
    fs::write(&path, &case.original).unwrap();
    let mut engine = open(&path, SyncPolicy::Always);
    let bounds = case.prepare(&mut engine);
    disk_log::start();
    Outcome::Count(prefix).inject(&control, field);
    let error = bounds
        .with_ordered_counts(BTreeMap::new(), |_| -> Result<(), Error> {
            panic!("failed version 1 count reached completion")
        })
        .unwrap_err();
    let recording = disk_log::finish();
    assert_io(error, "injected disk write failure");
    case.assert_current(&engine, 2);
    engine.context.allocations.before_reference().unwrap();
    assert_eq!(
        syncs(count_events(&recording, field, 0)),
        [(Synchronization::Data, IoOutcome::Succeeded)]
    );
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
        addresses.push(PhysicalAddress::new(address));
    }
    fixture.write(&path);
    let original = fs::read(&path).unwrap();
    let mut engine = open(&path, SyncPolicy::Always);
    let bounds = completion_tests::bounds(&mut engine, retired.to_vec(), &["a", "c"]);
    let first = bounds.updates()[0].field();
    let second = bounds.updates()[1].field();
    disk_log::start();
    bounds
        .with_ordered_counts(BTreeMap::new(), |_| Ok(()))
        .unwrap();
    let control = disk_log::finish();
    drop(engine);
    fs::write(&path, &original).unwrap();
    let mut engine = open(&path, SyncPolicy::Always);
    let bounds = completion_tests::bounds(&mut engine, retired.to_vec(), &["a", "c"]);
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
    disk_log::start();
    if matches!(failure, BatchFailure::Write) {
        Outcome::Count(Prefix::Partial).inject(&control, second);
    }
    let error = bounds
        .with_ordered_counts(BTreeMap::new(), |_| -> Result<(), Error> {
            panic!("failed batch reached completion")
        })
        .unwrap_err();
    let recording = disk_log::finish();
    match failure {
        BatchFailure::Preparation => {
            assert!(
                matches!(
                    error,
                    Error::EditUnsupported("an original count field changed during commit")
                ),
                "{error:?}"
            );
            assert!(recording.events.iter().all(|event| !matches!(event, IoEvent::Write { offset, .. } if *offset == first.range().start().get() || *offset == second.range().start().get())));
        }
        BatchFailure::Write => {
            assert_io(error, "injected disk write failure");
            let writes = count_events(&recording, first, 0)
                .iter()
                .filter_map(|event| match event {
                    IoEvent::Write { offset, .. } => Some(*offset),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                writes,
                [
                    first.range().start().get(),
                    checksum(first),
                    second.range().start().get(),
                    second.range().start().get(),
                    checksum(first),
                    first.range().start().get()
                ]
            );
        }
    }
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(engine.inplace_undo.len(), prefix);
    assert_eq!(
        engine
            .image()
            .read_exact_at(addresses[0].get(), chunk.len())
            .unwrap(),
        Form::Version2.chunk(2, c"new")
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(addresses[1].get(), chunk.len())
            .unwrap(),
        Form::Version2.chunk(
            if matches!(failure, BatchFailure::Preparation) {
                3
            } else {
                2
            },
            if matches!(failure, BatchFailure::Preparation) {
                c"new"
            } else {
                c"old"
            }
        )
    );
    engine.context.allocations.before_reference().unwrap();
}

/// Stores an independent count fixture and original bytes for ordered-publication checks.
struct CountCase {
    fixture: CountFixture,
    form: Form,
    count: NonZeroU32,
    original: Vec<u8>,
}

impl CountCase {
    /// Writes the selected header form and source count with an original comment.
    ///
    /// # Panics
    ///
    /// Panics if `count` is zero or fixture construction, writing, or reading fails.
    fn new(path: &Path, form: Form, base: u64, count: u32) -> Self {
        let count = NonZeroU32::new(count).unwrap();
        let fixture = count_test_util::fixture(path, form, base, count, c"old");
        Self {
            fixture,
            form,
            count,
            original: fs::read(path).unwrap(),
        }
    }

    /// Prepares deletion of one alias and journals a current comment change before publication.
    ///
    /// The count remains unchanged in the modified allocation, and the undo entry holds the
    /// complete original allocation.
    ///
    /// # Panics
    ///
    /// Panics if protection, replacement-root preparation, or the journal write fails.
    fn prepare<'a>(&self, engine: &'a mut WriteEngine) -> CountPublicationBounds<'a> {
        let Self {
            fixture,
            form,
            count,
            original: _,
        } = self;
        let bounds = completion_tests::bounds(engine, fixture.retired.to_vec(), &["a"]);
        bounds
            .phase
            .engine
            .write_inplace_journaled(fixture.allocation.start(), &form.chunk(count.get(), c"new"))
            .unwrap();
        bounds
    }

    /// Compares the complete current allocation with the expected count and modified comment.
    fn assert_current(&self, engine: &WriteEngine, count: u32) {
        assert_eq!(
            engine
                .image()
                .read_exact_at(
                    self.fixture.allocation.start().get(),
                    usize::try_from(self.fixture.allocation.length()).unwrap()
                )
                .unwrap(),
            self.form.chunk(count, c"new")
        );
    }
}

#[derive(Clone, Copy)]
enum BatchFailure {
    Preparation,
    Write,
}

/// Selects count, checksum, completion-barrier, or restoration failures after publication.
#[derive(Clone, Copy)]
enum Outcome {
    Barrier(FailurePoint),
    Checksum(Prefix),
    Count(Prefix),
    RestoreBarrier,
    RestoreWrite,
}

impl Outcome {
    /// Schedules failures from a successful recording and returns the first failed event's index.
    ///
    /// Restoration cases also schedule a second failure during rollback.
    ///
    /// # Panics
    ///
    /// Panics if a required count, checksum, or synchronization event is absent.
    fn inject(self, recording: &Recording, field: ObjectReferenceCountField) -> usize {
        let events = &recording.events;
        let count = events.iter().position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == field.range().start().get())).unwrap();
        let checksum = || {
            events.iter().position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == checksum(field))).unwrap()
        };
        let barrier = || {
            count
                + events[count..]
                    .iter()
                    .position(|event| matches!(event, IoEvent::Synchronize { .. }))
                    .unwrap()
        };
        let writes = |end| {
            events[..end]
                .iter()
                .filter(|event| matches!(event, IoEvent::Write { .. }))
                .count()
        };
        let syncs = |end| {
            events[..end]
                .iter()
                .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                .count()
        };
        match self {
            Self::Count(prefix) => {
                disk_log::fail_write_after(writes(count), prefix.written());
                count
            }
            Self::Checksum(prefix) => {
                let event = checksum();
                disk_log::fail_write_after(writes(event), prefix.written());
                event
            }
            Self::Barrier(point) => {
                let event = barrier();
                disk_log::fail_sync_after(syncs(event), point);
                event
            }
            Self::RestoreWrite => {
                let event = barrier();
                disk_log::fail_sync_after(syncs(event), FailurePoint::Before);
                disk_log::fail_write_after(writes(event), 0);
                event
            }
            Self::RestoreBarrier => {
                let event = checksum();
                disk_log::fail_write_after(writes(event), 2);
                disk_log::fail_sync_after(syncs(event), FailurePoint::Before);
                event
            }
        }
    }

    /// Checks exact operation and restoration I/O errors for the selected failure.
    fn assert_error(self, error: Error) {
        match self {
            Self::Count(_) | Self::Checksum(_) => assert_io(error, "injected disk write failure"),
            Self::Barrier(_) => assert_io(error, "injected disk synchronization failure"),
            Self::RestoreWrite | Self::RestoreBarrier => {
                let Error::CommitPartiallyApplied { refusal, restore } = error else {
                    panic!("expected failed count restoration: {error:?}");
                };
                assert_io(
                    *refusal,
                    if matches!(self, Self::RestoreWrite) {
                        "injected disk synchronization failure"
                    } else {
                        "injected disk write failure"
                    },
                );
                assert_io(
                    *restore,
                    if matches!(self, Self::RestoreWrite) {
                        "injected disk write failure"
                    } else {
                        "injected disk synchronization failure"
                    },
                );
            }
        }
    }
}

/// Selects how much of a failed write is issued.
#[derive(Clone, Copy)]
enum Prefix {
    Before,
    Complete,
    Partial,
}

impl Prefix {
    /// Returns the byte limit passed to the write-failure injector.
    fn written(self) -> usize {
        match self {
            Self::Before => 0,
            Self::Complete => usize::MAX,
            Self::Partial => 2,
        }
    }
}

/// Opens an unbuffered editor with the selected synchronization policy.
///
/// # Panics
///
/// Panics if opening or buffering configuration fails.
fn open(path: &Path, policy: SyncPolicy) -> WriteEngine {
    let mut engine = edit_test_util::open_engine(path);
    engine.set_sync_policy(policy);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
}

/// Borrows events from the first count write after checking a prior superblock write at `base`.
///
/// # Panics
///
/// Panics if the recording lacks the count write or the preceding publication write.
fn count_events(recording: &Recording, field: ObjectReferenceCountField, base: u64) -> &[IoEvent] {
    let count = recording.events.iter().position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == field.range().start().get())).unwrap();
    assert!(
        recording.events[..count]
            .iter()
            .any(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == base))
    );
    &recording.events[count..]
}

fn checksum(field: ObjectReferenceCountField) -> u64 {
    field.chunk().end().get() - u64::try_from(test_util::checksum::CHECKSUM).unwrap()
}

fn syncs(events: &[IoEvent]) -> Vec<(Synchronization, IoOutcome)> {
    events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Synchronize { kind, outcome } => Some((*kind, *outcome)),
            _ => None,
        })
        .collect()
}

fn assert_publication(engine: &WriteEngine, root: PhysicalAddress, end: PhysicalAddress) {
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(
        engine
            .addresses()
            .resolve(engine.superblock().root_group_address)
            .unwrap(),
        root
    );
    assert_eq!(engine.superblock().eof_address, end.get());
}

fn assert_io(error: Error, expected: &str) {
    let Error::Io(error) = error else {
        panic!("expected I/O error: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), expected);
}

fn assert_uncertain(error: Error) {
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
}

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}
