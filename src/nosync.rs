//! The mutex the chunk cache locks in a `no_std` build.

use core::cell::BorrowMutError;
use core::cell::RefCell;
use core::cell::RefMut;

/// A lock for one thread, with the `new` and `lock` methods of `std::sync::Mutex`.
///
/// The chunk cache calls these methods with or without `std`. A [`RefCell`] holds the value, so
/// [`Mutex::lock`] returns an error while a guard is live, and a `Mutex` is not `Sync`.
pub struct Mutex<T> {
    data: RefCell<T>,
}

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            data: RefCell::new(value),
        }
    }

    /// Borrows the value mutably until the returned guard drops.
    ///
    /// # Errors
    ///
    /// Returns [`BorrowMutError`] if a guard is live.
    pub fn lock(&self) -> Result<RefMut<'_, T>, BorrowMutError> {
        self.data.try_borrow_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::Mutex;

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
