//! Single-threaded Mutex replacement for no_std environments.
//!
//! WASM is single-threaded, so a simple `UnsafeCell` wrapper suffices.

use core::cell::UnsafeCell;

pub struct Mutex<T> {
    data: UnsafeCell<T>,
}

// SAFETY: a `no_std` build runs single-threaded (WASM without threads, or bare metal), so no
// two threads hold the value at once.
unsafe impl<T> Send for Mutex<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    pub fn new(val: T) -> Self {
        Self {
            data: UnsafeCell::new(val),
        }
    }

    pub fn lock(&self) -> Result<MutexGuard<'_, T>, ()> {
        Ok(MutexGuard { mutex: self })
    }
}

pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> core::ops::Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: Single-threaded environment (no_std/WASM).
        unsafe { &*self.mutex.data.get() }
    }
}

impl<T> core::ops::DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: Single-threaded environment (no_std/WASM).
        unsafe { &mut *self.mutex.data.get() }
    }
}
