//! Read-only backends for exercising HDF5 file loading through different sources.

use std::path::Path;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
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
        self.open_with_options(path, FileAccessProperties::new())
    }

    /// Opens `path` read-only with `properties` and the selected byte source.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the file cannot be read or opened.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{ChunkCacheConfig, FileAccessProperties};
    /// use test_util_hdf5::reader::ReadBackend;
    ///
    /// # let path = test_util::temp::temp_path("reader_options.h5");
    /// # std::fs::write(&path, test_util::legacy::simple_dataset())?;
    /// let properties = FileAccessProperties::new().with_chunk_cache(
    ///     ChunkCacheConfig::new().with_max_bytes(0).with_max_slots(0),
    /// );
    /// for backend in [
    ///     ReadBackend::Buffered,
    ///     ReadBackend::Bytes,
    ///     ReadBackend::Source,
    ///     ReadBackend::Streaming,
    /// ] {
    ///     let file = backend.open_with_options(&path, properties)?;
    ///     assert_eq!(file.access_properties(), properties);
    ///     assert_eq!(file.dataset("data")?.read_f64()?, [1.0, 2.0, 3.0]);
    ///     file.close()?;
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open_with_options(
        self,
        path: &Path,
        properties: FileAccessProperties,
    ) -> Result<File, Error> {
        match self {
            Self::Buffered => File::open_with_options(path, properties),
            Self::Bytes => File::from_bytes_with_options(std::fs::read(path)?, properties),
            Self::Source => File::from_source_with_options(
                ReadSeekSource::new(std::fs::File::open(path)?)?,
                properties,
            ),
            Self::Streaming => File::open_streaming_with_options(path, properties),
        }
    }
}
