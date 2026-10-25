//! Editing sessions for tests of file contents and editing behavior.
//!
//! These helpers select [`SyncPolicy::OnClose`], which skips intermediate filesystem
//! synchronization and synchronizes when the session closes. Tests of durability semantics select
//! [`SyncPolicy::Always`] explicitly through [`File::open_rw_with_options`].

use std::path::Path;

use hdf5_pure::{Error, File, FileAccessProperties, FileLocking, MemoryStrategy, SyncPolicy};

/// Opens a file for editing with bounded memory and no mirror fallback.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the file requires a mirror. Other opening errors are
/// documented in [`File::open_rw`].
pub fn open_bounded(path: &Path) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose),
    )
}

/// Opens a file for editing with automatic memory strategy selection.
///
/// # Errors
///
/// Returns [`Error::Io`] if opening or reading fails, or another error documented in
/// [`File::open_rw`].
pub fn open_rw(path: &Path) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
}

/// Returns access properties for a page-buffered editing session.
///
/// The page buffer is one mebibyte. Locking is disabled so tests
/// can inspect the file while it is open; OS locks are mandatory on Windows and would otherwise
/// block that observation.
pub fn page_buffered() -> FileAccessProperties {
    FileAccessProperties::new()
        .with_sync_policy(SyncPolicy::OnClose)
        .with_locking(FileLocking::Disabled)
        .with_page_buffer_size(1 << 20)
}
