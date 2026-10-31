//! Original backend reads paired with the opened image's effective address context.

use hdf5_pure_core::MetadataBytes;

use crate::address::AddressContext;
use crate::edit::WriteEngine;
use crate::error::FormatError;
use crate::image::FileImage;
use crate::source::{BytesSource, MetadataCacheStats, OriginalSource, Source};

use super::{Backend, FileInner};

/// A borrowed original backend and the effective address context of its owning image.
///
/// Reader dispatch and the write engine construct this value from the backend they own.
/// Reads preserve the backend's metadata ownership, physical ranges, and cache behavior.
#[derive(Clone, Copy)]
pub(crate) struct OpenedSource<'a> {
    backing: Backing<'a>,
    addresses: AddressContext,
}

impl<'a> OpenedSource<'a> {
    /// Calls `f` with the file's original backend and effective address context.
    ///
    /// For an editable backend, holds the engine lock during the callback. The callback uses
    /// the supplied source for accesses to this backend.
    pub(super) fn with_file<R>(file: &FileInner, f: impl FnOnce(&OpenedSource<'_>) -> R) -> R {
        match &file.backend {
            Backend::InMemory(bytes) => f(&OpenedSource {
                backing: Backing::Bytes(BytesSource::new(bytes)),
                addresses: file.addresses(),
            }),
            Backend::Streaming(source) => f(&OpenedSource {
                backing: Backing::Source(source.as_ref()),
                addresses: file.addresses(),
            }),
            Backend::Edit(engine) => {
                let engine = engine
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                f(&OpenedSource::from_engine(&engine))
            }
        }
    }

    /// Borrows the write engine's image with its effective address context.
    pub(crate) fn from_engine(engine: &'a WriteEngine) -> Self {
        Self {
            backing: Backing::Image(engine.file_image()),
            addresses: engine.addresses(),
        }
    }

    /// Associates a test backend with the supplied effective address context.
    #[cfg(test)]
    pub(crate) fn for_test(source: &'a dyn Source, addresses: AddressContext) -> Self {
        Self {
            backing: Backing::Source(source),
            addresses,
        }
    }

    fn source(&self) -> &dyn Source {
        match &self.backing {
            Backing::Bytes(bytes) => bytes,
            Backing::Source(source) => *source,
            Backing::Image(image) => *image,
        }
    }
}

impl OriginalSource for OpenedSource<'_> {
    fn addresses(&self) -> AddressContext {
        self.addresses
    }

    fn resident_bytes(&self) -> Option<&[u8]> {
        match self.backing {
            Backing::Bytes(bytes) => Some(bytes.0),
            Backing::Source(_) => None,
            Backing::Image(image) => image.as_slice(),
        }
    }
}

impl Source for OpenedSource<'_> {
    fn len(&self) -> u64 {
        self.source().len()
    }
    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.source().read_at(offset, buffer)
    }
    fn read_exact_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        self.source().read_exact_at(offset, length)
    }
    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        self.source().read_metadata_at(offset, length)
    }
    fn read_metadata(&self, offset: u64, length: usize) -> Result<MetadataBytes<'_>, FormatError> {
        self.source().read_metadata(offset, length)
    }
    fn metadata_origin(&self) -> Result<u64, FormatError> {
        self.source().metadata_origin()
    }
    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.source().metadata_cache_stats()
    }
    fn reset_metadata_cache_stats(&self) {
        self.source().reset_metadata_cache_stats();
    }
}

#[derive(Clone, Copy)]
enum Backing<'a> {
    Bytes(BytesSource<&'a [u8]>),
    Image(&'a dyn FileImage),
    Source(&'a dyn Source),
}

#[cfg(test)]
mod tests {
    use hdf5_pure_core::AddressRange;
    use hdf5_pure_format::__private::HDF5_SIGNATURE;
    use rstest::rstest;
    use test_util_hdf5::reader::ReadBackend;

    use super::*;
    use crate::address::{StoredAddress, StoredAddressExt};
    use crate::reader::test_util as reader_test_util;
    use crate::reader::{File, FileAccessProperties};
    use crate::source::{PhysicalSource, RangeReader};
    use crate::{FileBuilder, MemoryStrategy, UserblockSize};

    #[rstest]
    #[case::zero_base(0, 0)]
    #[case::userblock(512, 512)]
    #[case::forward_relocated(0, 512)]
    #[case::backward_relocated(512, 0)]
    fn opened_readers_establish_original_sources_and_their_effective_context(
        #[case] encoded_base: usize,
        #[case] signature: usize,
        #[values(ReadBackend::Bytes, ReadBackend::Source)] backend: ReadBackend,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("original-source.h5");
        let mut builder = FileBuilder::new();
        builder.with_userblock(UserblockSize::try_from(encoded_base as u64).unwrap());
        builder.create_dataset("values").with_i32_data(&[9, 8, 7]);
        let original = builder.finish().unwrap();
        let expected = &original[encoded_base..encoded_base + HDF5_SIGNATURE.len()];
        let mut relocated = vec![0; signature];
        relocated.extend_from_slice(&original[encoded_base..]);
        std::fs::write(&path, &relocated).unwrap();
        let file = reader_test_util::open(&path, backend).unwrap();
        file.with_opened_source(|source| {
            assert_opened_views(
                source,
                signature,
                expected,
                matches!(backend, ReadBackend::Bytes),
            );
        });
        file.close().unwrap();
    }

    #[rstest]
    #[case::handle(0, MemoryStrategy::Bounded, false)]
    #[case::mirror(512, MemoryStrategy::Mirrored, true)]
    fn writable_images_establish_original_sources_from_their_opened_engine(
        #[case] base: u64,
        #[case] strategy: MemoryStrategy,
        #[case] resident: bool,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("opened-image.h5");
        let mut builder = FileBuilder::new();
        builder.with_userblock(UserblockSize::try_from(base).unwrap());
        builder.create_dataset("values").with_i32_data(&[9, 8, 7]);
        builder.write(&path).unwrap();
        let image = std::fs::read(&path).unwrap();
        let base = usize::try_from(base).unwrap();
        let expected = &image[base..base + HDF5_SIGNATURE.len()];
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(strategy),
        )
        .unwrap();
        file.with_opened_source(|source| assert_opened_views(source, base, expected, resident));
        file.close().unwrap();
    }

    fn assert_opened_views(
        source: &OpenedSource<'_>,
        signature: usize,
        expected: &[u8],
        resident: bool,
    ) {
        assert_eq!(source.addresses().effective_base().get(), signature as u64);
        let physical = PhysicalSource::new(source);
        let bytes = physical.resident();
        assert_eq!(bytes.is_some(), resident);
        let stored = physical.stored();
        assert_eq!(stored.len(), source.len() - signature as u64);
        let range = AddressRange::new(StoredAddress::new(0), expected.len() as u64).unwrap();
        let metadata = stored.read_range(range).unwrap();
        assert_eq!(metadata.as_ref(), expected);
        assert_eq!(metadata.offset(), signature as u64);
        if let Some(bytes) = bytes {
            let stored = bytes.stored().unwrap();
            let actual = stored.borrow_range(range).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                actual.as_ptr(),
                source.resident_bytes().unwrap()[signature..].as_ptr()
            );
            let metadata = stored.read_range(range).unwrap();
            assert_eq!(metadata.as_ptr(), actual.as_ptr());
            assert!(metadata.shared().is_none());
        }
    }
}
