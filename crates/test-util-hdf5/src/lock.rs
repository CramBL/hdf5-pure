//! The locks that keep two tests from calling one C library at once.
//!
//! The libraries are separate, so the locks are too: a libmatio test does not wait on a libhdf5
//! one.

#[cfg(any(feature = "hdf5", feature = "__matio"))]
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Serializes the tests that take it over the whole of their bodies, since libhdf5 is not built
/// thread-safe here.
///
/// The helpers in this crate never take it themselves: a test holding it would deadlock on the
/// first helper it calls.
#[cfg(feature = "hdf5")]
pub fn libhdf5_guard() -> MutexGuard<'static, ()> {
    static LIBHDF5: Mutex<()> = Mutex::new(());
    LIBHDF5.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Serializes all libmatio calls: HDF5, which libmatio calls internally, isn't thread-safe by
/// default and cargo runs tests in parallel.
#[cfg(feature = "__matio")]
pub fn matio_lock() -> MutexGuard<'static, ()> {
    static MATIO: Mutex<()> = Mutex::new(());
    // A panicking test poisons the lock, and should not break the later ones.
    MATIO.lock().unwrap_or_else(PoisonError::into_inner)
}
