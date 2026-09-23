//! Editing sessions on a file through `hdf5-pure`'s read-write entry points.

use std::path::Path;

use hdf5_pure::{Error, File, FileAccessProperties, MemoryStrategy};

/// Opens `path` read-write on the bounded engine alone, so a file the engine stops accepting
/// fails here and never falls back to the mirror.
pub fn open_bounded(path: &Path) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
}
