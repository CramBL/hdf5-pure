//! Read-only backends for exercising HDF5 file loading through different sources.

use std::ops::Deref;
use std::path::Path;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::ReadSeekSource;
use tempfile::TempDir;

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

    /// Opens `image`, the bytes of an HDF5 file, with the selected read-only backend.
    ///
    /// Under [`Bytes`](Self::Bytes), this opens `image` in memory. Under the other backends, it
    /// writes `image` to a file in a new temporary directory and opens that file as
    /// [`open`](Self::open) does.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the image cannot be written to its temporary file or opened. The
    /// temporary directory is removed before the error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use test_util_hdf5::reader::ReadBackend;
    ///
    /// let file = ReadBackend::Streaming.open_image(test_util::legacy::simple_dataset())?;
    /// assert_eq!(file.dataset("data")?.read_f64()?, [1.0, 2.0, 3.0]);
    /// file.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open_image(self, image: Vec<u8>) -> Result<ImageFile, Error> {
        self.open_image_with_options(image, FileAccessProperties::new())
    }

    /// Opens `image` read-only with `properties` and the selected byte source.
    ///
    /// It opens `image` under each backend as [`open_image`](Self::open_image) describes.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the image cannot be written to its temporary file or opened. The
    /// temporary directory is removed before the error is returned.
    pub fn open_image_with_options(
        self,
        image: Vec<u8>,
        properties: FileAccessProperties,
    ) -> Result<ImageFile, Error> {
        self.open_image_in(&tempfile::env::temp_dir(), image, properties)
    }

    /// Opens `image` as [`open_image_with_options`](Self::open_image_with_options) does, with
    /// the temporary directory created in `parent`.
    fn open_image_in(
        self,
        parent: &Path,
        image: Vec<u8>,
        properties: FileAccessProperties,
    ) -> Result<ImageFile, Error> {
        match self {
            Self::Bytes => Backing::memory(image, properties),
            Self::Buffered | Self::Source | Self::Streaming => {
                Backing::temporary_file(self, parent, &image, properties)
            }
        }
        .map(ImageFile)
    }
}

/// An HDF5 file opened from an image by [`ReadBackend::open_image`] or
/// [`ReadBackend::open_image_with_options`].
///
/// An `ImageFile` dereferences to its [`File`]. Under a path backend it also owns the temporary
/// directory that holds the image, and dropping it releases the file before it removes the
/// directory.
#[derive(Debug)]
pub struct ImageFile(Backing);

impl ImageFile {
    /// Closes the file, then removes its temporary directory if it has one.
    ///
    /// Dropping an `ImageFile` removes the directory too and ignores the error.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if [`File::close`] fails, and [`Error::Io`] if the directory cannot be
    /// removed.
    pub fn close(self) -> Result<(), Error> {
        match self.0 {
            Backing::Memory(file) => file.close(),
            Backing::TemporaryFile { file, directory } => {
                file.close()?;
                Ok(directory.close()?)
            }
        }
    }
}

impl Deref for ImageFile {
    type Target = File;

    fn deref(&self) -> &File {
        match &self.0 {
            Backing::Memory(file) | Backing::TemporaryFile { file, .. } => file,
        }
    }
}

/// The open file of an [`ImageFile`] and the resource that holds its image.
#[derive(Debug)]
enum Backing {
    Memory(File),
    // Fields drop in declaration order, so the file closes before its directory is removed.
    TemporaryFile { file: File, directory: TempDir },
}

impl Backing {
    fn memory(image: Vec<u8>, properties: FileAccessProperties) -> Result<Self, Error> {
        File::from_bytes_with_options(image, properties).map(Self::Memory)
    }

    /// Writes `image` to a file in a new directory in `parent` and opens that file with `backend`.
    fn temporary_file(
        backend: ReadBackend,
        parent: &Path,
        image: &[u8],
        properties: FileAccessProperties,
    ) -> Result<Self, Error> {
        let directory = TempDir::new_in(parent)?;
        let path = directory.path().join(IMAGE_FILE_NAME);
        std::fs::write(&path, image)?;
        let file = backend.open_with_options(&path, properties)?;
        Ok(Self::TemporaryFile { file, directory })
    }
}

const IMAGE_FILE_NAME: &str = "image.h5";

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;
    use std::path::Path;

    use hdf5_pure::Error;
    use hdf5_pure::FileAccessProperties;
    use hdf5_pure::FormatError;
    use rstest::rstest;
    use test_util::legacy;

    use super::ReadBackend;

    #[rstest]
    #[case::buffered(ReadBackend::Buffered, 1)]
    #[case::bytes_without_file(ReadBackend::Bytes, 0)]
    #[case::source(ReadBackend::Source, 1)]
    #[case::streaming(ReadBackend::Streaming, 1)]
    fn an_opened_image_keeps_its_temporary_file_until_close(
        #[case] backend: ReadBackend,
        #[case] files: usize,
    ) {
        let parent = tempfile::tempdir().unwrap();
        let image = legacy::simple_dataset();
        let file = backend
            .open_image_in(parent.path(), image.clone(), FileAccessProperties::new())
            .unwrap();
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap(),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(stored_images(parent.path()), vec![vec![image]; files]);
        file.close().unwrap();
        assert_eq!(stored_images(parent.path()), Vec::<Vec<Vec<u8>>>::new());
        parent.close().unwrap();
    }

    #[rstest]
    fn an_open_error_leaves_the_parent_directory_empty(
        #[values(
            ReadBackend::Buffered,
            ReadBackend::Bytes,
            ReadBackend::Source,
            ReadBackend::Streaming
        )]
        backend: ReadBackend,
    ) {
        let parent = tempfile::tempdir().unwrap();
        let error = backend
            .open_image_in(parent.path(), vec![0; 64], FileAccessProperties::new())
            .unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::SignatureNotFound)),
            "{error:?}"
        );
        assert_eq!(stored_images(parent.path()), Vec::<Vec<Vec<u8>>>::new());
        parent.close().unwrap();
    }

    #[test]
    fn close_returns_the_temporary_directory_removal_error() {
        let parent = tempfile::tempdir().unwrap();
        let file = ReadBackend::Buffered
            .open_image_in(
                parent.path(),
                legacy::simple_dataset(),
                FileAccessProperties::new(),
            )
            .unwrap();
        for entry in std::fs::read_dir(parent.path()).unwrap() {
            std::fs::remove_dir_all(entry.unwrap().path()).unwrap();
        }
        let error = file.close().unwrap_err();
        let Error::Io(error) = &error else {
            panic!("expected Io, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::NotFound);
        parent.close().unwrap();
    }

    fn stored_images(parent: &Path) -> Vec<Vec<Vec<u8>>> {
        std::fs::read_dir(parent)
            .unwrap()
            .map(|directory| {
                std::fs::read_dir(directory.unwrap().path())
                    .unwrap()
                    .map(|file| std::fs::read(file.unwrap().path()).unwrap())
                    .collect()
            })
            .collect()
    }
}
