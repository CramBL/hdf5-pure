use core::cell::Cell;

thread_local! {
    static DATA_BLOCKS: Cell<usize> = const { Cell::new(0) };
    static SUPER_BLOCKS: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn note_data_block() {
    DATA_BLOCKS.with(|c| c.set(c.get() + 1));
}

pub(crate) fn note_super_block() {
    SUPER_BLOCKS.with(|c| c.set(c.get() + 1));
}

/// Returns both allocation counts and resets the current thread's counters.
pub(crate) fn take() -> (usize, usize) {
    (
        DATA_BLOCKS.with(|c| c.replace(0)),
        SUPER_BLOCKS.with(|c| c.replace(0)),
    )
}
