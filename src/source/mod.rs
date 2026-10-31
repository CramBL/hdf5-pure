//! Random-access byte sources for the reader: the [`Source`] trait and its backends.
//!
//! [`BytesSource`] reads from a buffer that holds the whole file, and [`ReadSeekSource`] from a
//! reader that seeks. [`MetadataCachingSource`] wraps either with a bounded cache of the metadata
//! reads and passes a dataset's payload through to the inner source. Internal readers use
//! [`PhysicalSource`] and [`StoredSource`] for coordinate-specific reads.

use alloc::vec::Vec;

use hdf5_pure_core::MetadataBytes;
use hdf5_pure_format::__private::MetadataSource;

#[cfg(test)]
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
#[cfg(test)]
use crate::address::{BaseAddress, BaseAddressExt};
use crate::error::FormatError;

mod backend;
mod original;
mod view;

#[cfg(test)]
pub(crate) use view::PhysicalBytes;
#[cfg(any(feature = "std", test, doc))]
pub(crate) use view::PhysicalSource;
#[cfg(any(test, doc))]
pub(crate) use view::StoredSource;
pub(crate) use view::{StoredBytes, StoredRead};

#[cfg(any(feature = "std", test, doc))]
pub(crate) use backend::BytesSource;
pub(crate) use backend::{
    BaseOffsetSource, RangeReader, SourceMetadata, bytes_in, frame, framed_metadata,
    legacy_framed_region, legacy_stored_range, unexpected_eof,
};
pub use backend::{MetadataCacheConfig, MetadataCacheStats, Source};
#[cfg(feature = "std")]
pub use backend::{MetadataCachingSource, ReadSeekSource};
#[cfg(feature = "std")]
pub(crate) use backend::{MetadataReadCache, ValidatedSource};

pub(crate) use original::OriginalSource;

/// Adapts stored-coordinate reads to the format parsers' byte-offset interface.
///
/// The parser's offsets are interpreted as stored addresses. Metadata reads retain the view's
/// ownership, physical ranges, and cache behavior.
pub(crate) struct StoredMetadata<'a, R: StoredRead + ?Sized>(pub(crate) &'a R);

impl<R: StoredRead + ?Sized> MetadataSource for StoredMetadata<'_, R> {
    type Bytes<'a>
        = MetadataBytes<'a>
    where
        Self: 'a;
    fn read_metadata(&self, offset: u64, length: usize) -> Result<Self::Bytes<'_>, FormatError> {
        self.0.read_metadata(StoredAddress::new(offset), length)
    }
    fn len(&self) -> u64 {
        self.0.len()
    }
    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.0.read_at(StoredAddress::new(offset), buffer)
    }
    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        self.0.read_metadata_at(StoredAddress::new(offset), length)
    }
}

/// Borrows a stored-coordinate view of a source whose effective base is zero.
#[cfg(test)]
pub(crate) fn zero_base_source<S: Source + ?Sized>(source: &S) -> StoredSource<'_, S> {
    PhysicalSource::for_test(
        source,
        AddressContext::from_effective_base(BaseAddress::ZERO),
    )
    .stored()
}

/// Borrows a resident stored-coordinate view of an image whose effective base is zero.
#[cfg(test)]
pub(crate) fn zero_base_bytes(bytes: &[u8]) -> StoredBytes<'_> {
    PhysicalBytes::for_test(
        bytes,
        AddressContext::from_effective_base(BaseAddress::ZERO),
    )
    .stored()
    .unwrap()
}
