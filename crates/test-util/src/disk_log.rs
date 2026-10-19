//! Records completed file writes and resizes for replay against a byte buffer.
//!
//! Recording is local to each thread and disabled until [`start`] is called. Callers record
//! operations after successful I/O. Replaying a prefix models interruption between completed
//! operations, excluding partial writes and synchronization failures.

use std::cell::RefCell;

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

thread_local! {
    static LOG: RefCell<Option<Vec<DiskOp>>> = const { RefCell::new(None) };
}
