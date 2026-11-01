//! Test observations of zero generation and its scratch buffers.

use core::cell::Cell;

/// Zero-generation requests recorded on the current thread.
///
/// Call and byte totals saturate at [`u64::MAX`]. The caller measures issued writes separately.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ZeroWork {
    /// Recorded zero-generation requests.
    pub(super) calls: u64,
    /// Sum of the byte lengths passed to [`record_zeros`].
    pub(super) bytes: u64,
    /// Largest scratch buffer length supplied to a recorded request.
    pub(super) scratch_peak: u64,
}

std::thread_local! {
    static ZEROS: Cell<ZeroWork> = const { Cell::new(ZeroWork { calls: 0, bytes: 0, scratch_peak: 0 }) };
}

/// Records one zero-generation request and its scratch buffer length.
pub(super) fn record_zeros(bytes: usize, scratch: usize) {
    ZEROS.with(|observed| {
        let mut work = observed.get();
        work.calls = work.calls.saturating_add(1);
        work.bytes = work.bytes.saturating_add(bytes as u64);
        work.scratch_peak = work.scratch_peak.max(scratch as u64);
        observed.set(work);
    });
}

/// Returns the current thread's recorded zero-generation work.
pub(super) fn zero_work() -> ZeroWork {
    ZEROS.with(Cell::get)
}

/// Clears the current thread's zero-generation observations.
pub(super) fn reset_zeros() {
    ZEROS.with(|observed| observed.set(ZeroWork::default()));
}

#[cfg(test)]
mod tests;
