use core::cell::Cell;

use super::CREATION;
use super::CreationWork;

/// Returns the current thread's cumulative preparation, index, and reservation observations.
pub(crate) fn creation_work() -> CreationWork {
    CREATION.with(Cell::get)
}

/// Clears the current thread's preparation, index, and reservation observations.
pub(crate) fn reset_creation() {
    CREATION.with(|observed| observed.set(CreationWork::default()));
}
