//! The mutex the chunk cache locks in a `no_std` build.

use core::cell::BorrowMutError;
use core::cell::RefCell;
use core::cell::RefMut;

use crate::chunk_cache::CacheLock;

/// A lock for one thread, which holds the state of a [`ChunkCache`] in a `no_std` build.
///
/// A [`RefCell`] holds the value, so a `Mutex` is not `Sync`.
///
/// [`ChunkCache`]: crate::chunk_cache::ChunkCache
pub struct Mutex<T> {
    data: RefCell<T>,
}

impl<T> CacheLock<T> for Mutex<T> {
    type Guard<'a>
        = RefMut<'a, T>
    where
        T: 'a;
    type Error<'a>
        = BorrowMutError
    where
        T: 'a;

    fn new(value: T) -> Self {
        Self {
            data: RefCell::new(value),
        }
    }

    /// Borrows the value mutably until the returned guard drops.
    ///
    /// # Errors
    ///
    /// Returns [`BorrowMutError`] if a guard is live.
    fn lock(&self) -> Result<RefMut<'_, T>, BorrowMutError> {
        self.data.try_borrow_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::Mutex;
    use crate::chunk_cache::CacheLock;

    #[test]
    fn a_second_guard_is_rejected_while_the_first_is_live() {
        let mutex = Mutex::new(0u32);

        let mut first = mutex.lock().unwrap();
        mutex.lock().unwrap_err();
        *first = 1;

        assert_eq!(*first, 1);
    }

    #[test]
    fn a_guard_writes_the_value_the_next_guard_reads() {
        let mutex = Mutex::new(1);

        let mut first = mutex.lock().unwrap();
        *first += 1;
        drop(first);

        assert_eq!(*mutex.lock().unwrap(), 2);
    }
}
