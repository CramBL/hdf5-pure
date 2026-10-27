//! Thread-local counts of parsed indexes and examined selection candidates.

use core::cell::Cell;

std::thread_local! {
    static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

/// Clears both counters for the current thread.
pub(crate) fn reset() {
    COUNTS.set((0, 0));
}

/// Returns index constructions and examined selection candidates for the current thread.
pub(crate) fn counts() -> (usize, usize) {
    COUNTS.get()
}

/// Records a parsed-index construction on the current thread.
pub(super) fn constructed() {
    let (indexes, candidates) = COUNTS.get();
    COUNTS.set((indexes + 1, candidates));
}

/// Records a selection candidate examined on the current thread.
pub(super) fn candidate() {
    let (indexes, candidates) = COUNTS.get();
    COUNTS.set((indexes, candidates + 1));
}
