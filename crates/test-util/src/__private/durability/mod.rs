//! Recorded-event schedules for write, resize, and synchronization failures.

use std::io::ErrorKind;
use std::ops::Range;

use crate::disk_log;
use crate::disk_log::DiskOp;
use crate::disk_log::FailurePoint;
use crate::disk_log::IoEvent;
use crate::disk_log::IoOutcome;
use crate::disk_log::Recording;

/// A write, resize, or synchronization failure selected from a borrowed I/O trace.
///
/// Event indices refer to the full borrowed trace, including every I/O kind.
/// For a separately stored event slice, use [`Self::write_events`], [`Self::resize_events`], or
/// [`Self::synchronize_events`].
/// [`disk_log::write_all`] caps write-prefix limits at the request length.
/// Call [`Self::inject`] while recording a new attempt, then check it with [`Self::assert_failed`].
/// For a synchronization and recovery failure, see [`Self::with_recovery_write`].
///
/// # Examples
///
/// ```
/// use std::io::ErrorKind;
///
/// use test_util::__private::durability::FaultSchedule;
/// use test_util::disk_log;
///
/// disk_log::clear_failures();
/// let mut bytes = Vec::new();
/// disk_log::start();
/// disk_log::write_all(&mut bytes, 0, b"abcd")?;
/// let control = disk_log::finish();
/// let schedule = FaultSchedule::write_containing(&control, 1, 1);
///
/// bytes.clear();
/// disk_log::start();
/// schedule.inject();
/// let error = disk_log::write_all(&mut bytes, 0, b"wxyz").unwrap_err();
/// let failed = disk_log::finish();
/// disk_log::clear_failures();
///
/// assert_eq!(error.kind(), ErrorKind::Other);
/// schedule.assert_failed(&failed);
/// assert_eq!(bytes, b"wx");
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct FaultSchedule<'a> {
    /// The immutable trace used for event selection and attempt counts.
    events: &'a [IoEvent],
    /// The primary failure's index in the complete event trace.
    event: usize,
    /// The failure point for the primary resize.
    resize: Option<FailurePoint>,
    /// The byte limit for the primary or recovery write failure.
    write: Option<usize>,
    /// The failure point for the primary or recovery synchronization.
    synchronize: Option<FailurePoint>,
}

impl<'a> FaultSchedule<'a> {
    /// Selects the write at `event` for failure after at most `prefix` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `event` is absent or is not a write.
    #[track_caller]
    pub fn write(recording: &'a Recording, event: usize, prefix: usize) -> Self {
        Self::write_events(&recording.events, event, prefix)
    }

    /// Selects a write from `events` for failure after at most `prefix` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `event` is absent or is not a write.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::ErrorKind;
    ///
    /// use test_util::__private::durability::FaultSchedule;
    /// use test_util::disk_log;
    ///
    /// disk_log::clear_failures();
    /// let mut bytes = Vec::new();
    /// disk_log::start();
    /// disk_log::write_all(&mut bytes, 0, b"seed")?;
    /// let events = disk_log::finish().events;
    /// let schedule = FaultSchedule::write_events(&events, 0, 2);
    ///
    /// bytes.clear();
    /// disk_log::start();
    /// schedule.inject();
    /// let error = disk_log::write_all(&mut bytes, 0, b"data").unwrap_err();
    /// let failed = disk_log::finish();
    /// disk_log::clear_failures();
    ///
    /// assert_eq!(error.kind(), ErrorKind::Other);
    /// schedule.assert_failed(&failed);
    /// assert_eq!(bytes, b"da");
    /// # Ok::<(), std::io::Error>(())
    /// ```
    #[track_caller]
    pub fn write_events(events: &'a [IoEvent], event: usize, prefix: usize) -> Self {
        assert!(
            matches!(events.get(event), Some(IoEvent::Write { .. })),
            "expected a write at event {event}, got {:?}",
            events.get(event)
        );
        Self {
            events,
            event,
            resize: None,
            write: Some(prefix),
            synchronize: None,
        }
    }

    /// Selects the first write whose requested range contains `target`.
    ///
    /// Uses a half-open byte range. Adds the bytes before `target` in that request to `prefix`,
    /// saturating at [`usize::MAX`].
    ///
    /// # Panics
    ///
    /// Panics if a scanned write's request length or range end does not fit in [`u64`], or if
    /// no write contains `target`.
    #[track_caller]
    pub fn write_containing(recording: &'a Recording, target: u64, prefix: usize) -> Self {
        let Some(event) = recording
            .events
            .iter()
            .position(|event| event.write_contains(target))
        else {
            panic!("no write contains byte {target}");
        };
        let Some(IoEvent::Write { offset, .. }) = recording.events.get(event) else {
            panic!("expected the selected write at event {event}");
        };
        let Some(before) = target.checked_sub(*offset) else {
            panic!("selected write starts after byte {target}");
        };
        let Ok(before) = usize::try_from(before) else {
            panic!("write prefix does not fit in usize: {before}");
        };
        Self::write(recording, event, before.saturating_add(prefix))
    }

    /// Selects the resize at `event` for failure at `point`.
    ///
    /// # Panics
    ///
    /// Panics if `event` is absent or is not a resize.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    /// use std::io::{ErrorKind, Read, Seek, Write};
    ///
    /// use test_util::__private::durability::FaultSchedule;
    /// use test_util::disk_log::{self, FailurePoint};
    ///
    /// # let directory = tempfile::tempdir()?;
    /// let mut file = File::options()
    ///     .read(true)
    ///     .write(true)
    ///     .create_new(true)
    ///     .open(directory.path().join("resize.bin"))?;
    /// file.write_all(b"seed")?;
    /// disk_log::clear_failures();
    /// disk_log::start();
    /// disk_log::set_len(&file, 2)?;
    /// let control = disk_log::finish();
    /// let schedule = FaultSchedule::resize(&control, 0, FailurePoint::After);
    ///
    /// file.rewind()?;
    /// file.write_all(b"data")?;
    /// disk_log::start();
    /// schedule.inject();
    /// let error = disk_log::set_len(&file, 2).unwrap_err();
    /// let failed = disk_log::finish();
    /// disk_log::clear_failures();
    /// assert_eq!(error.kind(), ErrorKind::Other);
    /// schedule.assert_failed(&failed);
    /// assert_eq!(file.metadata()?.len(), 2);
    /// file.rewind()?;
    /// let mut bytes = Vec::new();
    /// file.read_to_end(&mut bytes)?;
    /// assert_eq!(bytes, b"da");
    /// drop(file);
    /// # Ok::<(), std::io::Error>(())
    /// ```
    #[track_caller]
    pub fn resize(recording: &'a Recording, event: usize, point: FailurePoint) -> Self {
        Self::resize_events(&recording.events, event, point)
    }

    /// Selects a resize from `events` for failure at `point`.
    ///
    /// # Panics
    ///
    /// Panics if `event` is absent or is not a resize.
    #[track_caller]
    pub fn resize_events(events: &'a [IoEvent], event: usize, point: FailurePoint) -> Self {
        assert!(
            matches!(events.get(event), Some(IoEvent::Resize { .. })),
            "expected resize at event {event}, got {:?}",
            events.get(event)
        );
        Self {
            events,
            event,
            resize: Some(point),
            write: None,
            synchronize: None,
        }
    }

    /// Selects the synchronization at `event` for failure at `point`.
    ///
    /// # Panics
    ///
    /// Panics if `event` is absent or is not a synchronization.
    #[track_caller]
    pub fn synchronize(recording: &'a Recording, event: usize, point: FailurePoint) -> Self {
        Self::synchronize_events(&recording.events, event, point)
    }

    /// Selects a synchronization from `events` for failure at `point`.
    ///
    /// # Panics
    ///
    /// Panics if `event` is absent or is not a synchronization.
    #[track_caller]
    pub fn synchronize_events(events: &'a [IoEvent], event: usize, point: FailurePoint) -> Self {
        assert!(
            matches!(events.get(event), Some(IoEvent::Synchronize { .. })),
            "expected synchronization at event {event}, got {:?}",
            events.get(event)
        );
        Self {
            events,
            event,
            resize: None,
            write: None,
            synchronize: Some(point),
        }
    }

    /// Selects the first synchronization following the event at `after`.
    ///
    /// Reports the injected error at `point`.
    ///
    /// # Panics
    ///
    /// Panics if `after` is absent or no synchronization follows it.
    #[track_caller]
    pub fn synchronize_after(recording: &'a Recording, after: usize, point: FailurePoint) -> Self {
        assert!(after < recording.events.len(), "missing event {after}");
        let Some(event) = recording
            .events
            .iter()
            .enumerate()
            .skip(after + 1)
            .find_map(|(event, entry)| {
                matches!(entry, IoEvent::Synchronize { .. }).then_some(event)
            })
        else {
            panic!("no synchronization follows event {after}");
        };
        Self::synchronize(recording, event, point)
    }

    /// Adds failure of the first write after the primary synchronization failure.
    ///
    /// Uses `prefix` as the byte limit for the recovery write.
    ///
    /// # Panics
    ///
    /// Panics if the primary event is not a synchronization or a recovery failure is already selected.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    /// use std::io::{ErrorKind, Read, Seek};
    ///
    /// use test_util::__private::durability::FaultSchedule;
    /// use test_util::disk_log::{self, FailurePoint, Synchronization};
    ///
    /// # let directory = tempfile::tempdir()?;
    /// let mut file = File::options()
    ///     .read(true)
    ///     .write(true)
    ///     .create_new(true)
    ///     .open(directory.path().join("schedule.bin"))?;
    /// disk_log::clear_failures();
    /// disk_log::start();
    /// disk_log::write_all(&mut file, 0, b"seed")?;
    /// disk_log::synchronize(&file, Synchronization::Data)?;
    /// let control = disk_log::finish();
    /// let schedule = FaultSchedule::synchronize_after(&control, 0, FailurePoint::Before)
    ///     .with_recovery_write(1);
    /// assert_eq!(schedule.event(), 1);
    ///
    /// file.rewind()?;
    /// disk_log::start();
    /// schedule.inject();
    /// disk_log::write_all(&mut file, 0, b"data")?;
    /// let error = disk_log::synchronize(&file, Synchronization::Data).unwrap_err();
    /// assert_eq!(error.kind(), ErrorKind::Other);
    /// file.rewind()?;
    /// let error = disk_log::write_all(&mut file, 0, b"undo").unwrap_err();
    /// assert_eq!(error.kind(), ErrorKind::Other);
    /// let failed = disk_log::finish();
    /// disk_log::clear_failures();
    /// schedule.assert_failed(&failed);
    ///
    /// file.rewind()?;
    /// let mut bytes = Vec::new();
    /// file.read_to_end(&mut bytes)?;
    /// assert_eq!(bytes, b"uata");
    /// drop(file);
    /// # Ok::<(), std::io::Error>(())
    /// ```
    #[track_caller]
    pub fn with_recovery_write(mut self, prefix: usize) -> Self {
        assert!(self.write.is_none() && self.synchronize.is_some());
        self.write = Some(prefix);
        self
    }

    /// Adds failure of the first synchronization after the primary write failure.
    ///
    /// Reports the recovery error at `point`.
    ///
    /// # Panics
    ///
    /// Panics if the primary event is not a write or a recovery failure is already selected.
    #[track_caller]
    pub fn with_recovery_synchronization(mut self, point: FailurePoint) -> Self {
        assert!(self.write.is_some() && self.synchronize.is_none());
        self.synchronize = Some(point);
        self
    }

    /// Schedules this attempt's failures on the current thread.
    ///
    /// Counts preceding attempts of each configured kind in the borrowed trace. Calls
    /// [`disk_log::fail_resize_after`], [`disk_log::fail_write_after`], and
    /// [`disk_log::fail_sync_after`] for those kinds. Use
    /// [`disk_log::clear_failures`] to clear earlier schedules.
    pub fn inject(&self) {
        if let Some(point) = self.resize {
            disk_log::fail_resize_after(self.ordinal(Attempt::Resize), point);
        }
        if let Some(prefix) = self.write {
            disk_log::fail_write_after(self.ordinal(Attempt::Write), prefix);
        }
        if let Some(point) = self.synchronize {
            disk_log::fail_sync_after(self.ordinal(Attempt::Synchronize), point);
        }
    }

    /// Returns the primary failure's index in the borrowed event slice.
    pub fn event(&self) -> usize {
        self.event
    }

    /// Checks the event prefix, primary failure metadata, and failure counts.
    ///
    /// Earlier events must match the control metadata and report success. For a primary write,
    /// checks its offset, request length, written prefix length, and completed operation's offset
    /// and byte count. For a primary synchronization, checks its kind. For a resize, checks its
    /// requested length. Requires no completed operation for [`FailurePoint::Before`], or a
    /// matching [`DiskOp::SetLen`] at the control operation's index for [`FailurePoint::After`].
    /// The primary error kind is [`ErrorKind::Other`]. Requires exact total and per-kind failure
    /// counts for the selected
    /// primary and recovery failures. The caller checks byte values and recovery semantics
    /// separately.
    ///
    /// # Panics
    ///
    /// Panics if the event prefix, primary failure details, completed operation, or failure counts
    /// differ from the schedule.
    #[track_caller]
    pub fn assert_failed(&self, failed: &Recording) {
        let Self {
            events,
            event,
            resize,
            write,
            synchronize,
        } = self;
        let Some(actual) = failed.events.get(*event) else {
            panic!("missing failed event {}", *event);
        };
        assert_eq!(&failed.events[..*event], &events[..*event]);
        assert!(
            failed.events[..*event]
                .iter()
                .all(|event| event.outcome() == IoOutcome::Succeeded)
        );
        assert_eq!(actual.outcome(), IoOutcome::Failed(ErrorKind::Other));
        assert_eq!(
            failed
                .events
                .iter()
                .filter(|event| matches!(event.outcome(), IoOutcome::Failed(_)))
                .count(),
            usize::from(resize.is_some())
                + usize::from(write.is_some())
                + usize::from(synchronize.is_some())
        );
        for (attempt, expected) in [
            (Attempt::Resize, resize.is_some()),
            (Attempt::Synchronize, synchronize.is_some()),
            (Attempt::Write, write.is_some()),
        ] {
            assert_eq!(
                failed
                    .events
                    .iter()
                    .filter(|event| {
                        matches!(event.outcome(), IoOutcome::Failed(_)) && attempt.matches(event)
                    })
                    .count(),
                usize::from(expected)
            );
        }
        match (events.get(*event), actual) {
            (
                Some(IoEvent::Write {
                    offset, requested, ..
                }),
                IoEvent::Write {
                    offset: actual,
                    requested: length,
                    written,
                    operation,
                    ..
                },
            ) => {
                assert_eq!((actual, length), (offset, requested));
                let Some(prefix) = *write else {
                    panic!("missing scheduled write failure");
                };
                assert_eq!(*written, prefix.min(*requested));
                if *written == 0 {
                    assert_eq!(*operation, None);
                } else {
                    let Some(operation) = operation else {
                        panic!("missing completed write prefix");
                    };
                    let Some(DiskOp::Write {
                        offset: start,
                        bytes,
                    }) = failed.operations.get(*operation)
                    else {
                        panic!("missing write operation {operation}");
                    };
                    assert_eq!(start, offset);
                    assert_eq!(bytes.len(), *written);
                }
            }
            (
                Some(IoEvent::Resize {
                    length,
                    operation: expected,
                    ..
                }),
                IoEvent::Resize {
                    length: actual,
                    operation,
                    ..
                },
            ) => {
                assert_eq!(actual, length);
                match resize {
                    Some(FailurePoint::Before) => assert_eq!(*operation, None),
                    Some(FailurePoint::After) => {
                        assert_eq!(operation, expected);
                        let Some(operation) = operation else {
                            panic!("missing completed resize operation");
                        };
                        assert_eq!(
                            failed.operations.get(*operation),
                            Some(&DiskOp::SetLen(*length))
                        );
                    }
                    None => panic!("missing scheduled resize failure"),
                }
            }
            (
                Some(IoEvent::Synchronize { kind, .. }),
                IoEvent::Synchronize { kind: actual, .. },
            ) => assert_eq!(actual, kind),
            (expected, actual) => panic!("expected failed event {expected:?}, got {actual:?}"),
        }
    }

    /// Counts attempts of `attempt` before the primary event.
    fn ordinal(&self, attempt: Attempt) -> usize {
        self.events[..self.event]
            .iter()
            .filter(|event| attempt.matches(event))
            .count()
    }
}

impl Recording {
    /// Checks bytes outside a publication write against this attempt's completed prefix.
    ///
    /// Replays operations before `publication` over `original`. Allows appended bytes in `after`.
    ///
    /// # Panics
    ///
    /// Panics if an index, offset, or range is invalid, the selected event is not the write for
    /// `exempt`, a later write or resize occurs, or the retained bytes differ.
    pub fn assert_publication_retained(
        &self,
        original: &[u8],
        after: &[u8],
        publication: usize,
        exempt: Range<usize>,
    ) {
        assert!(
            self.events[publication + 1..]
                .iter()
                .all(|event| !matches!(event, IoEvent::Write { .. } | IoEvent::Resize { .. }))
        );
        let mut retained = original.to_vec();
        for event in &self.events[..publication] {
            if let Some(operation) = event.operation() {
                self.operations[operation].apply(&mut retained);
            }
        }
        assert!(after.len() >= retained.len());
        let IoEvent::Write {
            offset, requested, ..
        } = self.events[publication]
        else {
            panic!("expected a publication write");
        };
        assert_eq!(usize::try_from(offset).unwrap(), exempt.start);
        assert_eq!(requested, exempt.len());
        assert_eq!(&after[..exempt.start], &retained[..exempt.start]);
        assert_eq!(&after[exempt.end..retained.len()], &retained[exempt.end..]);
    }
}

impl IoEvent {
    /// Returns whether `target` is in this write's half-open requested range.
    ///
    /// # Panics
    ///
    /// Panics if the request length or range end does not fit in [`u64`].
    fn write_contains(&self, target: u64) -> bool {
        let Self::Write {
            offset, requested, ..
        } = self
        else {
            return false;
        };
        let Ok(length) = u64::try_from(*requested) else {
            panic!("write length does not fit in u64: {requested}");
        };
        let Some(end) = offset.checked_add(length) else {
            panic!("write range overflows: {offset} + {length}");
        };
        *offset <= target && target < end
    }
}

/// The I/O kind counted before the primary event.
#[derive(Clone, Copy)]
enum Attempt {
    Resize,
    Synchronize,
    Write,
}

impl Attempt {
    fn matches(self, event: &IoEvent) -> bool {
        match self {
            Self::Resize => matches!(event, IoEvent::Resize { .. }),
            Self::Synchronize => matches!(event, IoEvent::Synchronize { .. }),
            Self::Write => matches!(event, IoEvent::Write { .. }),
        }
    }
}

#[cfg(test)]
mod tests;
