//! Assertions for facade error payloads and native rejection frames.

use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::{MajorErrorCode, MinorErrorCode};
use hdf5_pure::{Error, FormatError, SuperblockStatusFlag, SuperblockStatusFlags};

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

/// Asserts [`Error::RepackUnsupported`] with the complete expected reason.
///
/// # Panics
///
/// Panics if `error` is another variant or its reason differs from `reason`.
#[track_caller]
pub fn repack_unsupported(error: Error, reason: &str) {
    let Error::RepackUnsupported(actual) = error else {
        panic!("expected RepackUnsupported, got {error:?}");
    };
    assert_eq!(actual, reason);
}

/// Asserts [`Error::SwmrAppendUnsupported`] with the complete expected reason.
///
/// # Panics
///
/// Panics if `error` is another variant or its reason differs from `reason`.
#[track_caller]
pub fn swmr_append_unsupported(error: Error, reason: &str) {
    let Error::SwmrAppendUnsupported(actual) = error else {
        panic!("expected SwmrAppendUnsupported, got {error:?}");
    };
    assert_eq!(actual, reason);
}

// H5F_open in H5Fint.c, HDF5 1.10.11, 1.12.3, 1.14.6 and 2.2.0, reports
// this error when a normal reader encounters superblock write access flags.
/// Asserts libhdf5's innermost write-mark rejection frame.
///
/// # Panics
///
/// Panics if `error` has no native error frame or its innermost frame differs from the
/// expected error categories, function, or diagnostic.
#[cfg(feature = "hdf5")]
#[track_caller]
pub fn native_file_marked_in_use(error: &hdf5::Error) {
    let Some(stack) = error.stack() else {
        panic!("expected a native HDF5 error stack, got {error:?}");
    };
    let frame = stack.last().expect("native rejection has no error frames");
    assert_eq!(
        (
            frame.major_code(),
            frame.minor_code(),
            frame.func(),
            frame.desc()
        ),
        (
            MajorErrorCode::File,
            MinorErrorCode::CantOpenFile,
            "H5F_open",
            "file is already open for write (may use <h5clear file> to clear file consistency flags)"
        ),
        "{error:?}"
    );
}

/// The open operation and target used in a status-flag rejection.
#[derive(Clone, Copy)]
pub enum StatusOpen<'a> {
    /// A buffered or streaming read from a filesystem path.
    Read(&'a Path),
    /// A read from a byte source.
    SourceRead,
    /// A SWMR read from a filesystem path.
    SwmrRead(&'a Path),
    /// An ordinary or SWMR writer open from a filesystem path.
    Write(&'a Path),
}

/// Asserts [`Error::FileMarkedInUse`] with the complete diagnostic for `flags` and `open`.
///
/// # Panics
///
/// Panics if `error` is another variant or its diagnostic differs from the expected
/// target, status flags, operation, or recovery guidance.
#[track_caller]
pub fn file_marked_in_use(error: Error, flags: SuperblockStatusFlags, open: StatusOpen<'_>) {
    let Error::FileMarkedInUse(actual) = error else {
        panic!("{error:?}");
    };
    let target = match open {
        StatusOpen::Read(path) | StatusOpen::SwmrRead(path) | StatusOpen::Write(path) => {
            path.display().to_string()
        }
        StatusOpen::SourceRead => "a byte source".into(),
    };
    let swmr = flags.contains(SuperblockStatusFlag::SwmrWriteAccess);
    let flags = flags.bits();
    let reason = match open {
        StatusOpen::Write(_) => format!(
            "the superblock marks the file as open for write (status flags {flags:#04x}), so another writer holds it. Open it read-only, or — if a writer exited without closing the file — clear the flag with File::clear_status_flags"
        ),
        StatusOpen::SwmrRead(_) => format!(
            "the superblock's status flags disagree ({flags:#04x}): a SWMR reader needs a SWMR writer (both the write and SWMR-write bits) or a quiescent file (neither). Clear them with File::clear_status_flags if a writer exited without closing the file"
        ),
        StatusOpen::Read(_) | StatusOpen::SourceRead => {
            let recovery = match (matches!(open, StatusOpen::SourceRead), swmr) {
                (false, true) => {
                    "Use File::open_swmr to follow a live SWMR writer, or File::from_bytes to read the bytes as they stand; if a writer exited without closing the file, clear the flag with File::clear_status_flags"
                }
                (true, true) => {
                    "Use File::from_bytes to read the bytes as they stand; following a live writer with File::open_swmr, and clearing a flag a writer left behind with File::clear_status_flags, both need a filesystem path"
                }
                (false, false) => {
                    "No SWMR writer holds it, so no reader can follow it as one. Pass FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot) to read it as it stands, which is a consistent snapshot once the writer has called File::sync or closed the file, or use File::from_bytes to read the bytes as they stand; if a writer exited without closing the file, clear the flag with File::clear_status_flags"
                }
                (true, false) => {
                    "No SWMR writer holds it, so no reader can follow it as one. Pass FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot) to read it as it stands, which is a consistent snapshot once the writer has called File::sync or closed the file, or use File::from_bytes to read the bytes as they stand; clearing a flag a writer left behind with File::clear_status_flags needs a filesystem path"
                }
            };
            format!(
                "the superblock marks the file as open for write (status flags {flags:#04x}), so a snapshot read is not safe. {recovery}"
            )
        }
    };
    assert_eq!(actual, format!("{target}: {reason}."));
}
