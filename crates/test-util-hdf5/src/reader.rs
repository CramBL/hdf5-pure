//! Read-only backends for exercising HDF5 file loading through different sources.

use std::path::Path;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::ReadSeekSource;

/// Selects how [`File`] accesses a test file's bytes.
#[derive(Clone, Copy, Debug)]
pub enum ReadBackend {
    /// [`File::open`] loads the file from a path into memory.
    Buffered,
    /// [`File::from_bytes`] opens an owned in-memory image.
    Bytes,
    /// [`File::from_source`] reads through a [`ReadSeekSource`].
    Source,
    /// [`File::open_streaming`] reads file contents on demand.
    Streaming,
}

impl ReadBackend {
    /// Opens `path` with the selected read-only backend.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the file cannot be read or opened.
    pub fn open(self, path: &Path) -> Result<File, Error> {
        match self {
            Self::Buffered => File::open(path),
            Self::Bytes => File::from_bytes(std::fs::read(path)?),
            Self::Source => File::from_source(ReadSeekSource::new(std::fs::File::open(path)?)?),
            Self::Streaming => File::open_streaming(path),
        }
    }
}
