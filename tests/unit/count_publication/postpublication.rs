use std::error::Error as StdError;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::Path;

use rstest::rstest;
use test_util::bytes;
use test_util::disk_log::{
    self, DiskOp, FailurePoint, IoEvent, IoOutcome, Recording, Synchronization,
};
use test_util::object_header::v2::HeaderFlags;
use test_util::superblock;

use super::prepublication::{Part, Prefix};
use super::{CountFile, assert_values, stage, type_header};
use crate::{Error, File, FileAccessProperties, FileLocking, MemoryStrategy};

#[rstest]
#[case::count_zero(Fault::Count(Part::Count, Prefix::Zero))]
#[case::count_partial(Fault::Count(Part::Count, Prefix::Half))]
#[case::count_complete(Fault::Count(Part::Count, Prefix::Complete))]
#[case::checksum_zero(Fault::Count(Part::Checksum, Prefix::Zero))]
#[case::checksum_partial(Fault::Count(Part::Checksum, Prefix::Half))]
#[case::checksum_complete(Fault::Count(Part::Checksum, Prefix::Complete))]
#[case::barrier_before(Fault::Barrier(FailurePoint::Before))]
#[case::barrier_after(Fault::Barrier(FailurePoint::After))]
#[case::restore_write_zero(Fault::RestoreWrite(RestorePrefix::Zero))]
#[case::restore_write_complete(Fault::RestoreWrite(RestorePrefix::Complete))]
#[case::restore_barrier_before(Fault::RestoreBarrier(FailurePoint::Before))]
#[case::restore_barrier_after(Fault::RestoreBarrier(FailurePoint::After))]
fn count_decrease_failures_preserve_the_published_tree_and_quarantine_writes(
    #[case] fault: Fault,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("published-count-failure.h5");
    let fixture = CountFile::write(&path, 3, true);
    let control_path = test_util::temp::temp_path("published-count-control.h5");
    std::fs::copy(&path, &control_path).unwrap();
    let properties = FileAccessProperties::new()
        .with_locking(FileLocking::Disabled)
        .with_memory_strategy(memory)
        .with_page_buffer_size(0);
    let reference = File::open_rw_with_options(&control_path, properties).unwrap();
    let before_control = std::fs::read(&control_path).unwrap();
    stage(&reference, 1, true, false);
    disk_log::start();
    reference.commit().unwrap();
    let recording = disk_log::finish();
    let published = std::fs::read(&control_path).unwrap();
    fixture.assert_publication(&before_control, &recording.operations, &published, 2);
    let control = Control::new(&recording, fixture.target);
    reference.close().unwrap();

    let file = File::open_rw_with_options(&path, properties).unwrap();
    let before = std::fs::read(&path).unwrap();
    stage(&file, 1, true, false);
    disk_log::start();
    let event = control.inject(fault);
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    control.assert_failed(fault, event, &failed);
    fault.assert_error(&error);
    assert!(!file.has_staged_edits());
    let mut issued = before.clone();
    for operation in failed.events[..=event]
        .iter()
        .filter_map(IoEvent::operation)
    {
        failed.operations[operation].apply(&mut issued);
    }
    let mut expected = type_header(3);
    let updated = type_header(2);
    let selection = match fault {
        Fault::Count(part, prefix) => Some((part, prefix)),
        Fault::RestoreBarrier(_) => Some((Part::Count, Prefix::Half)),
        Fault::Barrier(_) | Fault::RestoreWrite(_) => None,
    };
    if let Some((part, prefix)) = selection {
        let field = match part {
            Part::Checksum => {
                expected[12..16].copy_from_slice(&updated[12..16]);
                expected.len() - 4
            }
            Part::Count => 12,
        };
        expected[field..field + prefix.bytes()]
            .copy_from_slice(&updated[field..field + prefix.bytes()]);
    } else {
        expected = updated;
    }
    let start = usize::try_from(fixture.target).unwrap();
    assert_eq!(&issued[start..start + expected.len()], expected);
    let after = std::fs::read(&path).unwrap();
    let mut replay = before;
    for operation in &failed.operations {
        operation.apply(&mut replay);
    }
    assert_eq!(replay, after);
    fixture.assert_header(
        &after,
        if matches!(fault, Fault::RestoreWrite(RestorePrefix::Zero)) {
            2
        } else {
            3
        },
    );
    let IoEvent::Write { requested, .. } = recording.events[control.publication] else {
        panic!("missing superblock publication");
    };
    assert_eq!(&after[..requested], &published[..requested]);
    let root = root_range(&published);
    assert_eq!(&after[root.clone()], &published[root.clone()]);
    let sections = file.reusable_free_sections().unwrap();
    for range in [start..start + type_header(3).len(), root] {
        assert!(sections.sections().iter().all(|section| {
            section.end_address().get() <= range.start as u64
                || section.address().get() >= range.end as u64
        }));
    }
    assert_values(&file, 1, false);
    assert_quarantined(&file, &path, &after);
    drop(file);
    File::clear_status_flags(&path).unwrap();
    let reopened = File::open(&path).unwrap();
    assert_values(&reopened, 1, false);
    reopened.close().unwrap();
}

#[derive(Clone, Copy)]
pub(super) enum Fault {
    Barrier(FailurePoint),
    Count(Part, Prefix),
    RestoreBarrier(FailurePoint),
    RestoreWrite(RestorePrefix),
}

impl Fault {
    pub(super) fn assert_error(self, error: &Error) {
        match (self, error) {
            (Self::RestoreBarrier(_), Error::CommitPartiallyApplied { refusal, restore }) => {
                assert_io(refusal, "injected disk write failure");
                assert_io(restore, "injected disk synchronization failure");
                assert_eq!(
                    error.source().unwrap().source().unwrap().to_string(),
                    "injected disk synchronization failure"
                );
            }
            (Self::RestoreWrite(_), Error::CommitPartiallyApplied { refusal, restore }) => {
                assert_io(refusal, "injected disk synchronization failure");
                assert_io(restore, "injected disk write failure");
                assert_eq!(
                    error.source().unwrap().source().unwrap().to_string(),
                    "injected disk write failure"
                );
            }
            (Self::Barrier(_), _) => assert_io(error, "injected disk synchronization failure"),
            (Self::Count(_, _), _) => assert_io(error, "injected disk write failure"),
            _ => panic!("unexpected count journal error: {error:?}"),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum RestorePrefix {
    Complete,
    Zero,
}

impl RestorePrefix {
    fn limit(self) -> usize {
        match self {
            Self::Complete => usize::MAX,
            Self::Zero => 0,
        }
    }
}

pub(super) struct Control<'a> {
    recording: &'a Recording,
    pub(super) publication: usize,
    count: FieldWrite,
    checksum: FieldWrite,
    barrier: usize,
}

impl<'a> Control<'a> {
    pub(super) fn new(recording: &'a Recording, target: u64) -> Self {
        let publication = recording
            .events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
            .unwrap();
        let count = FieldWrite::find(recording, publication, target + 12);
        let checksum = FieldWrite::find(
            recording,
            publication,
            target + type_header(3).len() as u64 - 4,
        );
        assert_eq!(count.event, checksum.event);
        let Some(DiskOp::Write { bytes, .. }) = recording.events[count.event]
            .operation()
            .and_then(|operation| recording.operations.get(operation))
        else {
            panic!("missing completed count write");
        };
        assert_eq!(&bytes[count.before..count.before + 4], 2_u32.to_le_bytes());
        let barrier = recording.events[checksum.event + 1..]
            .iter()
            .position(|event| matches!(event, IoEvent::Synchronize { .. }))
            .unwrap()
            + checksum.event
            + 1;
        Self {
            recording,
            publication,
            count,
            checksum,
            barrier,
        }
    }

    pub(super) fn inject(&self, fault: Fault) -> usize {
        match fault {
            Fault::Count(part, prefix) => {
                let write = match part {
                    Part::Count => self.count,
                    Part::Checksum => self.checksum,
                };
                self.fail_write(write.event, write.before + prefix.bytes());
                write.event
            }
            Fault::Barrier(point) => {
                self.fail_sync(self.barrier, point);
                self.barrier
            }
            Fault::RestoreWrite(prefix) => {
                self.fail_write(self.barrier, prefix.limit());
                self.fail_sync(self.barrier, FailurePoint::Before);
                self.barrier
            }
            Fault::RestoreBarrier(point) => {
                self.fail_sync(self.count.event, point);
                self.fail_write(self.count.event, self.count.before + Prefix::Half.bytes());
                self.count.event
            }
        }
    }

    fn fail_write(&self, event: usize, prefix: usize) {
        disk_log::fail_write_after(
            self.recording.events[..event]
                .iter()
                .filter(|event| matches!(event, IoEvent::Write { .. }))
                .count(),
            prefix,
        );
    }

    fn fail_sync(&self, event: usize, point: FailurePoint) {
        disk_log::fail_sync_after(
            self.recording.events[..event]
                .iter()
                .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                .count(),
            point,
        );
    }

    pub(super) fn assert_failed(&self, fault: Fault, event: usize, failed: &Recording) {
        assert!(event > self.publication);
        assert_eq!(&failed.events[..event], &self.recording.events[..event]);
        assert_eq!(
            failed.events[event].outcome(),
            IoOutcome::Failed(ErrorKind::Other)
        );
        let failures = failed
            .events
            .iter()
            .filter(|event| matches!(event.outcome(), IoOutcome::Failed(_)))
            .collect::<Vec<_>>();
        assert_eq!(
            failures.len(),
            if matches!(fault, Fault::RestoreWrite(_) | Fault::RestoreBarrier(_)) {
                2
            } else {
                1
            }
        );
        assert!(
            failures
                .iter()
                .all(|event| event.outcome() == IoOutcome::Failed(ErrorKind::Other))
        );
        let failed_outcome = IoOutcome::Failed(ErrorKind::Other);
        let syncs = failed.events[event..]
            .iter()
            .filter_map(|event| {
                let IoEvent::Synchronize { kind, outcome } = event else {
                    return None;
                };
                Some((*kind, *outcome))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            syncs,
            match fault {
                Fault::Count(_, _) => vec![(Synchronization::Data, IoOutcome::Succeeded)],
                Fault::Barrier(_) => vec![
                    (Synchronization::All, failed_outcome),
                    (Synchronization::Data, IoOutcome::Succeeded)
                ],
                Fault::RestoreWrite(_) => vec![(Synchronization::All, failed_outcome)],
                Fault::RestoreBarrier(_) => vec![(Synchronization::Data, failed_outcome)],
            }
        );
        match (&self.recording.events[event], &failed.events[event]) {
            (
                IoEvent::Write {
                    offset, requested, ..
                },
                IoEvent::Write {
                    offset: actual,
                    requested: length,
                    written,
                    ..
                },
            ) => {
                assert_eq!((offset, requested), (actual, length));
                let (field, prefix) = match fault {
                    Fault::Count(Part::Count, prefix) => (self.count, prefix),
                    Fault::Count(Part::Checksum, prefix) => (self.checksum, prefix),
                    Fault::RestoreBarrier(_) => (self.count, Prefix::Half),
                    _ => panic!("expected a write failure"),
                };
                assert_eq!(*written, (field.before + prefix.bytes()).min(*requested));
            }
            (IoEvent::Synchronize { kind, .. }, IoEvent::Synchronize { kind: actual, .. }) => {
                assert_eq!(kind, actual)
            }
            _ => panic!("the selected count operation did not fail"),
        }
        match fault {
            Fault::RestoreWrite(prefix) => {
                let IoEvent::Write {
                    offset,
                    requested,
                    written,
                    ..
                } = failures[1]
                else {
                    panic!("missing restoration write failure");
                };
                assert_eq!(*written, prefix.limit().min(*requested));
                let IoEvent::Write {
                    offset: expected,
                    requested: length,
                    ..
                } = self.recording.events[self.count.event]
                else {
                    panic!("missing control write");
                };
                assert_eq!((*offset, *requested), (expected, length));
            }
            Fault::RestoreBarrier(_) => assert!(matches!(failures[1], IoEvent::Synchronize { .. })),
            _ => {}
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct FieldWrite {
    pub(super) event: usize,
    pub(super) before: usize,
}

impl FieldWrite {
    pub(super) fn find(recording: &Recording, publication: usize, field: u64) -> Self {
        recording.events[publication + 1..]
            .iter()
            .enumerate()
            .find_map(|(index, event)| {
                let IoEvent::Write {
                    offset, requested, ..
                } = event
                else {
                    return None;
                };
                let before = usize::try_from(field.checked_sub(*offset)?).ok()?;
                (before + 4 <= *requested).then_some(Self {
                    event: publication + 1 + index,
                    before,
                })
            })
            .unwrap()
    }
}

pub(super) fn root_range(image: &[u8]) -> Range<usize> {
    let root = usize::try_from(superblock::root_header_address(image, 0)).unwrap();
    assert_eq!(&image[root..root + 4], b"OHDR");
    let flags = HeaderFlags(image[root + 5]);
    let prefix = 6
        + 16 * usize::from(flags.stores_times())
        + 4 * usize::from(flags.stores_attribute_phase_change());
    let size = flags.chunk_size_width();
    let length = usize::try_from(bytes::uint_at(image, root + prefix, size)).unwrap();
    root..root + prefix + size + length + 4
}

pub(super) fn assert_quarantined(file: &File, path: &Path, after: &[u8]) {
    file.root()
        .create_dataset("after", |builder| {
            builder.with_i32_data(&[1]);
        })
        .unwrap();
    assert!(file.has_staged_edits());
    disk_log::start();
    for error in [file.commit().unwrap_err(), file.sync().unwrap_err()] {
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    }
    assert!(file.has_staged_edits());
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(std::fs::read(path).unwrap(), after);
}

fn assert_io(error: &Error, message: &str) {
    let Error::Io(error) = error else {
        panic!("expected I/O failure: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), message);
}
