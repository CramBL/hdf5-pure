//! Records file I/O outcomes and completed byte operations on the current thread.
//!
//! Recording begins with [`start`] and ends with [`finish`]. Replaying a prefix of
//! [`Recording::operations`] models interruption between completed writes and resizes, including
//! bytes written before an error. [`Recording::events`] also records synchronization requests and
//! their reported outcomes for ordering assertions. These observations describe I/O calls, without
//! modeling physical power loss or torn metadata writes.

use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::io::Write;

/// A completed write or resize of a file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiskOp {
    /// Bytes written at an absolute file offset.
    Write { offset: u64, bytes: Vec<u8> },
    /// A change to the file length.
    SetLen(u64),
}

impl DiskOp {
    /// Applies the operation, extending writes past the end with zeros.
    ///
    /// # Panics
    ///
    /// Panics if an offset, length, or write end does not fit in [`usize`].
    pub fn apply(&self, image: &mut Vec<u8>) {
        match self {
            Self::Write { offset, bytes } => {
                let start = usize::try_from(*offset).expect("the write offset fits in usize");
                let end = start
                    .checked_add(bytes.len())
                    .expect("the write end fits in usize");
                image.resize(image.len().max(end), 0);
                image[start..end].copy_from_slice(bytes);
            }
            Self::SetLen(len) => {
                image.resize(
                    usize::try_from(*len).expect("the file length fits in usize"),
                    0,
                );
            }
        }
    }

    /// Returns a label with the written range or the new file length.
    pub fn describe(&self) -> String {
        match self {
            Self::Write { offset, bytes } => {
                format!("write {offset}..{}", offset + bytes.len() as u64)
            }
            Self::SetLen(len) => format!("set_len {len}"),
        }
    }
}

/// Completed byte operations and the outcomes of recorded I/O calls.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Recording {
    /// Writes and successful resizes, in completion order, for byte replay.
    pub operations: Vec<DiskOp>,
    /// Call outcomes, including failures and synchronization requests, in call order.
    pub events: Vec<IoEvent>,
}

/// When to report an injected synchronization error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailurePoint {
    /// After the file synchronization succeeds.
    After,
    /// Before calling the file synchronization.
    Before,
}

/// The result reported to the caller, including injected errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IoOutcome {
    /// An error with this kind, possibly after completing part or all of the operation.
    Failed(io::ErrorKind),
    /// Successful completion.
    Succeeded,
}

/// Which file contents to synchronize.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Synchronization {
    /// File data and metadata, through [`File::sync_all`].
    All,
    /// File data, through [`File::sync_data`].
    Data,
}

/// An I/O call with its reported outcome and completed byte operation, if any.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoEvent {
    /// A request to change the file length.
    Resize {
        /// The length passed to [`set_len`], in bytes.
        length: u64,
        outcome: IoOutcome,
        /// An index into [`Recording::operations`] on success, or `None` on failure.
        operation: Option<usize>,
    },
    /// A request to synchronize the file.
    Synchronize {
        kind: Synchronization,
        outcome: IoOutcome,
    },
    /// A request to write bytes at the writer's current position.
    Write {
        /// The absolute file offset supplied for replay.
        offset: u64,
        /// The number of bytes passed to [`write_all`].
        requested: usize,
        /// The number of bytes reported as written by [`Write::write`].
        written: usize,
        outcome: IoOutcome,
        /// An index into [`Recording::operations`] for a nonempty written prefix.
        operation: Option<usize>,
    },
}

impl IoEvent {
    /// Returns the result reported to the caller.
    pub fn outcome(&self) -> IoOutcome {
        match self {
            Self::Resize { outcome, .. }
            | Self::Synchronize { outcome, .. }
            | Self::Write { outcome, .. } => *outcome,
        }
    }

    /// Returns the index of the completed byte operation in [`Recording::operations`].
    ///
    /// Returns `None` for synchronization, a failed resize, or a write of zero bytes.
    pub fn operation(&self) -> Option<usize> {
        match self {
            Self::Resize { operation, .. } | Self::Write { operation, .. } => *operation,
            Self::Synchronize { .. } => None,
        }
    }
}

/// Begins recording on this thread, discarding any previous recording.
pub fn start() {
    LOG.with(|log| *log.borrow_mut() = Some(Recording::default()));
}

/// Stops recording and returns the operations, or an empty vector if recording was disabled.
pub fn take() -> Vec<DiskOp> {
    finish().operations
}

/// Stops recording and returns the byte operations and I/O outcomes.
///
/// Returns an empty [`Recording`] if recording was disabled.
pub fn finish() -> Recording {
    LOG.with(|log| log.borrow_mut().take()).unwrap_or_default()
}

/// Records a completed write if this thread is recording.
///
/// Copies the bytes only when recording is enabled.
pub fn record_write(offset: u64, bytes: &[u8]) {
    record_write_outcome(offset, bytes.len(), bytes, IoOutcome::Succeeded);
}

/// Records a completed resize if this thread is recording.
pub fn record_set_len(len: u64) {
    LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            let operation = log.operations.len();
            log.operations.push(DiskOp::SetLen(len));
            log.events.push(IoEvent::Resize {
                length: len,
                outcome: IoOutcome::Succeeded,
                operation: Some(operation),
            });
        }
    });
}

/// Skips `completed` calls to [`write_all`] before injecting one write error on this thread.
///
/// On the following call, writes at most `prefix` bytes before reporting the error. Replaces any
/// previously scheduled failure, independently of whether recording is enabled.
///
/// # Examples
///
/// ```
/// use std::io::ErrorKind;
///
/// use test_util::disk_log;
///
/// let mut bytes = Vec::new();
/// disk_log::fail_write_after(1, 2);
/// disk_log::write_all(&mut bytes, 0, b"abcd")?;
/// let error = disk_log::write_all(&mut bytes, 4, b"efgh").unwrap_err();
/// assert_eq!(error.kind(), ErrorKind::Other);
/// assert_eq!(bytes, b"abcdef");
/// disk_log::write_all(&mut bytes, 6, b"gh")?;
/// assert_eq!(bytes, b"abcdefgh");
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn fail_write_after(completed: usize, prefix: usize) {
    WRITE_FAILURE.with(|failure| *failure.borrow_mut() = Some((completed, prefix)));
}

/// Writes `bytes` and records them at `offset` if this thread is recording.
///
/// Uses the writer's current position and records `offset` for replay.
/// A failure scheduled with [`fail_write_after`] is consumed before writing its prefix. Records
/// a nonempty prefix if writing it succeeds, then reports the injected error.
///
/// # Errors
///
/// Returns an error from [`Write::write`] if writing fails, retrying
/// [`io::ErrorKind::Interrupted`]. Returns [`io::ErrorKind::WriteZero`] if the writer reports
/// zero bytes before completion, or [`io::ErrorKind::Other`] after an injected failure's prefix.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
///
/// use test_util::disk_log;
///
/// let mut writer = Cursor::new(vec![0; 8]);
/// writer.set_position(2);
/// disk_log::start();
/// disk_log::write_all(&mut writer, 6, b"ab")?;
/// assert_eq!(writer.position(), 4);
/// assert_eq!(writer.into_inner(), b"\0\0ab\0\0\0\0");
/// let mut replay = Vec::new();
/// for operation in disk_log::take() {
///     operation.apply(&mut replay);
/// }
/// assert_eq!(replay, b"\0\0\0\0\0\0ab");
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn write_all(writer: &mut impl Write, offset: u64, bytes: &[u8]) -> io::Result<()> {
    let prefix = WRITE_FAILURE.with(|failure| {
        let mut failure = failure.borrow_mut();
        match failure.as_mut() {
            Some((0, prefix)) => {
                let prefix = *prefix;
                *failure = None;
                Some(prefix)
            }
            Some((remaining, _)) => {
                *remaining -= 1;
                None
            }
            None => None,
        }
    });
    let limit = prefix.map_or(bytes.len(), |prefix| prefix.min(bytes.len()));
    let mut written = 0;
    let result = loop {
        if written == limit {
            break if prefix.is_some() {
                Err(io::Error::other("injected disk write failure"))
            } else {
                Ok(())
            };
        }
        match writer.write(&bytes[written..limit]) {
            Ok(0) => break Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(count) => written += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => break Err(error),
        }
    };
    record_write_outcome(offset, bytes.len(), &bytes[..written], outcome(&result));
    result
}

/// Changes the file length and records the result if this thread is recording.
///
/// # Errors
///
/// Returns an error from [`File::set_len`] if resizing fails.
pub fn set_len(file: &File, length: u64) -> io::Result<()> {
    let result = file.set_len(length);
    if result.is_ok() {
        record_set_len(length);
    } else {
        LOG.with(|log| {
            if let Some(log) = log.borrow_mut().as_mut() {
                log.events.push(IoEvent::Resize {
                    length,
                    outcome: outcome(&result),
                    operation: None,
                });
            }
        });
    }
    result
}

/// Skips `completed` calls to [`synchronize`] before injecting one synchronization error.
///
/// Replaces any previously scheduled synchronization failure on this thread. On the following
/// call, reports [`io::ErrorKind::Other`] at `point`, independently of whether recording is enabled.
pub fn fail_sync_after(completed: usize, point: FailurePoint) {
    SYNC_FAILURE.with(|failure| *failure.borrow_mut() = Some((completed, point)));
}

/// Synchronizes the file and records the reported outcome if this thread is recording.
///
/// # Errors
///
/// Returns an error from [`File::sync_all`] or [`File::sync_data`] if synchronization fails,
/// or [`io::ErrorKind::Other`] for a failure scheduled with [`fail_sync_after`].
pub fn synchronize(file: &File, kind: Synchronization) -> io::Result<()> {
    synchronize_with(kind, || match kind {
        Synchronization::All => file.sync_all(),
        Synchronization::Data => file.sync_data(),
    })
}

/// Clears scheduled write and synchronization failures on this thread.
///
/// Recording remains enabled or disabled as before the call.
pub fn clear_failures() {
    WRITE_FAILURE.with(|failure| *failure.borrow_mut() = None);
    SYNC_FAILURE.with(|failure| *failure.borrow_mut() = None);
}

fn synchronize_with(
    kind: Synchronization,
    operation: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    let point = SYNC_FAILURE.with(|failure| {
        let mut failure = failure.borrow_mut();
        match failure.as_mut() {
            Some((0, point)) => {
                let point = *point;
                *failure = None;
                Some(point)
            }
            Some((remaining, _)) => {
                *remaining -= 1;
                None
            }
            None => None,
        }
    });
    let result = if point == Some(FailurePoint::Before) {
        Err(io::Error::other("injected disk synchronization failure"))
    } else {
        operation().and_then(|()| {
            if point == Some(FailurePoint::After) {
                Err(io::Error::other("injected disk synchronization failure"))
            } else {
                Ok(())
            }
        })
    };
    LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.events.push(IoEvent::Synchronize {
                kind,
                outcome: outcome(&result),
            });
        }
    });
    result
}

fn outcome(result: &io::Result<()>) -> IoOutcome {
    match result {
        Ok(()) => IoOutcome::Succeeded,
        Err(error) => IoOutcome::Failed(error.kind()),
    }
}

fn record_write_outcome(offset: u64, requested: usize, bytes: &[u8], outcome: IoOutcome) {
    LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            let operation = if bytes.is_empty() {
                None
            } else {
                let operation = log.operations.len();
                log.operations.push(DiskOp::Write {
                    offset,
                    bytes: bytes.to_vec(),
                });
                Some(operation)
            };
            log.events.push(IoEvent::Write {
                offset,
                requested,
                written: bytes.len(),
                outcome,
                operation,
            });
        }
    });
}

thread_local! {
    static WRITE_FAILURE: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
    static SYNC_FAILURE: RefCell<Option<(usize, FailurePoint)>> = const { RefCell::new(None) };
    static LOG: RefCell<Option<Recording>> = const { RefCell::new(None) };
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::io;
    use std::io::ErrorKind;
    use std::io::Write;

    use rstest::rstest;

    use super::FailurePoint;
    use super::IoEvent;
    use super::IoOutcome;
    use super::Synchronization;

    #[rstest]
    #[case::before(FailurePoint::Before, 0)]
    #[case::after(FailurePoint::After, 1)]
    fn synchronization_failures_record_the_reported_outcome(
        #[case] point: FailurePoint,
        #[case] calls: usize,
    ) {
        let called = Cell::new(0);
        super::start();
        super::fail_sync_after(0, point);
        let error = super::synchronize_with(Synchronization::All, || {
            called.set(called.get() + 1);
            Ok(())
        })
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk synchronization failure");
        assert_eq!(called.get(), calls);
        let recording = super::finish();
        assert!(recording.operations.is_empty());
        assert_eq!(
            recording.events,
            [IoEvent::Synchronize {
                kind: Synchronization::All,
                outcome: IoOutcome::Failed(ErrorKind::Other)
            }]
        );
        super::synchronize_with(Synchronization::Data, || Ok(())).unwrap();
    }

    #[rstest]
    #[case::data(Synchronization::Data)]
    #[case::all(Synchronization::All)]
    fn successful_file_operations_record_byte_replay_and_synchronization(
        #[case] kind: Synchronization,
    ) {
        let mut file = tempfile::tempfile().unwrap();
        super::start();
        super::write_all(&mut file, 0, b"abcd").unwrap();
        super::set_len(&file, 8).unwrap();
        super::synchronize(&file, kind).unwrap();
        let recording = super::finish();
        assert_eq!(
            recording.events,
            [
                IoEvent::Write {
                    offset: 0,
                    requested: 4,
                    written: 4,
                    outcome: IoOutcome::Succeeded,
                    operation: Some(0)
                },
                IoEvent::Resize {
                    length: 8,
                    outcome: IoOutcome::Succeeded,
                    operation: Some(1)
                },
                IoEvent::Synchronize {
                    kind,
                    outcome: IoOutcome::Succeeded
                },
            ]
        );
        let mut replay = Vec::new();
        for operation in recording.operations {
            operation.apply(&mut replay);
        }
        assert_eq!(replay, b"abcd\0\0\0\0");
    }

    #[test]
    fn an_underlying_partial_write_records_its_completed_prefix_and_error() {
        let mut writer = FailingWriter { bytes: Vec::new() };
        super::start();
        let error = super::write_all(&mut writer, 0, b"abcd").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        let recording = super::finish();
        assert_eq!(
            recording.events,
            [IoEvent::Write {
                offset: 0,
                requested: 4,
                written: 2,
                outcome: IoOutcome::Failed(ErrorKind::PermissionDenied),
                operation: Some(0)
            }]
        );
        let mut replay = Vec::new();
        for operation in recording.operations {
            operation.apply(&mut replay);
        }
        assert_eq!(replay, writer.bytes);
        assert_eq!(writer.bytes, b"ab");
    }

    #[test]
    fn a_failed_file_resize_records_the_error_without_a_completed_operation() {
        let temporary = tempfile::NamedTempFile::new().unwrap();
        let file = std::fs::File::open(temporary.path()).unwrap();
        super::start();
        let error = super::set_len(&file, 8).unwrap_err();
        let recording = super::finish();
        assert!(recording.operations.is_empty());
        assert_eq!(
            recording.events,
            [IoEvent::Resize {
                length: 8,
                outcome: IoOutcome::Failed(error.kind()),
                operation: None
            }]
        );
        assert_eq!(file.metadata().unwrap().len(), 0);
        drop(file);
        drop(temporary);
    }

    struct FailingWriter {
        bytes: Vec<u8>,
    }

    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.bytes.is_empty() {
                let length = bytes.len().min(2);
                self.bytes.extend_from_slice(&bytes[..length]);
                Ok(length)
            } else {
                Err(io::Error::from(ErrorKind::PermissionDenied))
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[rstest]
    #[case::before_write(0, 0)]
    #[case::partial_write(0, 2)]
    #[case::completed_write(0, 4)]
    #[case::after_one_write(1, 2)]
    fn injected_write_failure_records_completed_bytes(
        #[case] completed: usize,
        #[case] prefix: usize,
    ) {
        let mut bytes = Vec::new();
        super::start();
        super::fail_write_after(completed, prefix);
        if completed == 1 {
            super::write_all(&mut bytes, 0, b"abcd").unwrap();
        }
        let error = super::write_all(&mut bytes, completed as u64 * 4, b"efgh").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert_eq!(&bytes[completed * 4..], &b"efgh"[..prefix]);
        let mut replay = Vec::new();
        let recording = super::finish();
        assert_eq!(
            recording.events.last(),
            Some(&IoEvent::Write {
                offset: completed as u64 * 4,
                requested: 4,
                written: prefix,
                outcome: IoOutcome::Failed(ErrorKind::Other),
                operation: if prefix == 0 { None } else { Some(completed) },
            })
        );
        for operation in recording.operations {
            operation.apply(&mut replay);
        }
        assert_eq!(replay, bytes);
        super::write_all(&mut bytes, (completed * 4 + prefix) as u64, b"retry").unwrap();
        assert_eq!(&bytes[completed * 4 + prefix..], b"retry");
    }
}
