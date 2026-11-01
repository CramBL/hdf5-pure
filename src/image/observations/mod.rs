//! Test observations of buffered bytes, explicit copies, and issued writes.

use super::BufferedWrites;

/// Current payload storage and cumulative buffering observations.
///
/// The caller measures container overhead, allocator bookkeeping, and vector reallocation copies
/// separately. Only the active buffer and retained tail are included in flush peaks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ImageWork {
    /// Pending byte runs.
    pub(crate) pending_runs: u64,
    /// Combined length of pending runs, in bytes.
    pub(crate) pending_bytes: u64,
    /// Combined byte capacity of pending run buffers.
    pub(crate) pending_capacity: u64,
    /// Largest combined length of the active flush buffer and retained tail, in bytes.
    pub(crate) flush_peak_bytes: u64,
    /// Largest combined byte capacity of the active flush buffer and retained tail.
    pub(crate) flush_peak_capacity: u64,
    /// Successful clean-gap read requests.
    pub(crate) clean_gap_calls: u64,
    /// Bytes supplied by successful clean-gap reads.
    pub(crate) clean_gap_bytes: u64,
    /// Bytes explicitly copied into pending or merged flush buffers.
    pub(crate) copied_bytes: u64,
    /// Completed issued write operations.
    pub(crate) write_calls: u64,
    /// Bytes in completed issued writes.
    pub(crate) write_bytes: u64,
    /// Length of a resident file mirror, in bytes, or zero without a mirror.
    pub(crate) mirror_bytes: u64,
    /// Byte capacity of a resident file mirror, or zero without a mirror.
    pub(crate) mirror_capacity: u64,
}

impl BufferedWrites {
    /// Returns current pending storage with recorded work and active-flush peaks.
    pub(super) fn observation(&self) -> ImageWork {
        let Self {
            handle: _,
            mode: _,
            runs,
            pending_bytes,
            on_disk_len: _,
            issued_order,
            work,
        } = self;
        ImageWork {
            pending_runs: runs.len() as u64,
            pending_bytes: *pending_bytes as u64,
            pending_capacity: runs.values().fold(0_u64, |sum, bytes| {
                sum.saturating_add(bytes.capacity() as u64)
            }),
            write_calls: issued_order.len() as u64,
            write_bytes: issued_order
                .iter()
                .fold(0_u64, |sum, &(_, length)| sum.saturating_add(length)),
            ..*work
        }
    }

    /// Adds an explicit buffer copy to the saturating byte count.
    pub(super) fn record_copy(&mut self, bytes: usize) {
        self.work.copied_bytes = self.work.copied_bytes.saturating_add(bytes as u64);
    }

    /// Records flush peaks for the active buffer and an additional retained buffer.
    pub(super) fn observe_flush(
        &mut self,
        bytes: &Vec<u8>,
        extra_len: usize,
        extra_capacity: usize,
    ) {
        self.work.flush_peak_bytes = self
            .work
            .flush_peak_bytes
            .max((bytes.len() as u64).saturating_add(extra_len as u64));
        self.work.flush_peak_capacity = self
            .work
            .flush_peak_capacity
            .max((bytes.capacity() as u64).saturating_add(extra_capacity as u64));
    }
}

#[cfg(test)]
mod tests;
