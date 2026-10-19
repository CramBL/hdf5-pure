//! Records completed file writes and resizes for replay against a byte buffer.
//!
//! Recording is local to each thread and disabled until [`start`] is called. Callers record
//! operations after successful I/O. With [`write_all`], the recording includes the bytes written
//! before an injected error. Replaying a prefix models interruption between recorded operations.
//! Synchronization is outside the recording.

use std::cell::RefCell;
use std::io;
use std::io::Write;

/// A completed write or resize of a file.
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

/// Begins recording on this thread, discarding any previous recording.
pub fn start() {
    LOG.with(|log| *log.borrow_mut() = Some(Vec::new()));
}

/// Stops recording and returns the operations, or an empty vector if recording was disabled.
pub fn take() -> Vec<DiskOp> {
    LOG.with(|log| log.borrow_mut().take()).unwrap_or_default()
}

/// Records a completed write if this thread is recording.
///
/// Copies the bytes only when recording is enabled.
pub fn record_write(offset: u64, bytes: &[u8]) {
    LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push(DiskOp::Write {
                offset,
                bytes: bytes.to_vec(),
            });
        }
    });
}

/// Records a completed resize if this thread is recording.
pub fn record_set_len(len: u64) {
    LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push(DiskOp::SetLen(len));
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
/// Returns an error from [`Write::write_all`] if writing fails. Returns an error with
/// kind [`io::ErrorKind::Other`] after writing the prefix of an injected failure.
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
    if let Some(prefix) = prefix {
        let bytes = &bytes[..prefix.min(bytes.len())];
        writer.write_all(bytes)?;
        if !bytes.is_empty() {
            record_write(offset, bytes);
        }
        return Err(io::Error::other("injected disk write failure"));
    }
    writer.write_all(bytes)?;
    record_write(offset, bytes);
    Ok(())
}

thread_local! {
    static WRITE_FAILURE: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
    static LOG: RefCell<Option<Vec<DiskOp>>> = const { RefCell::new(None) };
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use rstest::rstest;

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
        for operation in super::take() {
            operation.apply(&mut replay);
        }
        assert_eq!(replay, bytes);
        super::write_all(&mut bytes, (completed * 4 + prefix) as u64, b"retry").unwrap();
        assert_eq!(&bytes[completed * 4 + prefix..], b"retry");
    }
}
