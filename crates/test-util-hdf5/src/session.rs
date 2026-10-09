//! Editing sessions on a file through `hdf5-pure`'s read-write entry points.

use std::path::Path;

use hdf5_pure::{Error, File, FileAccessProperties, FileLocking, MemoryStrategy, SyncPolicy};

/// Opens `path` read-write on the bounded engine alone, so a file the engine stops accepting
/// fails here and never falls back to the mirror.
pub fn open_bounded(path: &Path) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
}

/// Opens `path` read-write on the bounded engine and flushes changes when the session closes.
///
/// This keeps tests that exercise append batching from paying one filesystem synchronization per
/// append while still making the memory strategy explicit.
pub fn open_bounded_on_close(path: &Path) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose),
    )
}

/// Returns access properties for a page-buffered editing session.
///
/// The session flushes on close and uses a one-mebibyte page buffer. Locking is disabled so tests
/// can inspect the file while it is open; OS locks are mandatory on Windows and would otherwise
/// block that observation.
pub fn page_buffered() -> FileAccessProperties {
    FileAccessProperties::new()
        .with_sync_policy(SyncPolicy::OnClose)
        .with_locking(FileLocking::Disabled)
        .with_page_buffer_size(1 << 20)
}
