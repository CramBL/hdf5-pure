//! Assertions for facade error variants and their complete payloads.

use std::path::Path;

use hdf5_pure::{Error, FormatError};

/// Asserts [`FormatError::PathNotFound`] with the complete expected path.
///
/// # Panics
///
/// Panics if `error` is another variant or its path differs from `path`.
#[track_caller]
pub fn path_not_found(error: Error, path: &str) {
    let Error::Format(FormatError::PathNotFound(actual)) = error else {
        panic!("expected PathNotFound for {path:?}, got {error:?}");
    };
    assert_eq!(actual, path);
}

/// Asserts [`Error::EditUnsupported`] with the complete expected reason.
///
/// # Panics
///
/// Panics if `error` is another variant or its reason differs from `reason`.
#[track_caller]
pub fn edit_unsupported(error: Error, reason: &str) {
    let Error::EditUnsupported(actual) = error else {
        panic!("expected EditUnsupported, got {error:?}");
    };
    assert_eq!(actual, reason);
}

/// Asserts [`Error::AppendInPlaceUnsupported`] with the complete expected reason.
///
/// # Panics
///
/// Panics if `error` is another variant or its reason differs from `reason`.
#[track_caller]
pub fn append_in_place_unsupported(error: Error, reason: &str) {
    let Error::AppendInPlaceUnsupported(actual) = error else {
        panic!("expected AppendInPlaceUnsupported, got {error:?}");
    };
    assert_eq!(actual, reason);
}

/// Asserts [`Error::FileLocked`] with the complete contention diagnostic for `path`.
///
/// # Panics
///
/// Panics if `error` is another variant or its diagnostic differs from the expected message.
#[track_caller]
pub fn file_locked(error: Error, path: &Path) {
    let Error::FileLocked(actual) = error else {
        panic!("expected FileLocked, got {error:?}");
    };
    assert_eq!(
        actual,
        format!(
            "{}: file is already locked by another process. If a previous writer crashed, the OS lock is released automatically (try again); leftover on-disk status flags can be cleared with File::clear_status_flags. Set HDF5_USE_FILE_LOCKING=FALSE or pass FileLocking::Disabled to bypass locking.",
            path.display()
        )
    );
}
