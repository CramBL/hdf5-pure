//! Borrowed source views with physical or stored address coordinates.

use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::{AddressRange, MetadataBytes, PhysicalAddress, StoredAddress};

use crate::address::AddressContext;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::{MetadataCacheStats, RangeReader, Source};

/// A borrowed source addressed by positions relative to the original source's start.
///
/// Reads delegate to the original [`Source`]. Metadata retains that source's ownership and
/// physical metadata ranges, including any [`Source::metadata_origin`].
pub(crate) struct PhysicalSource<'a, S: Source + ?Sized> {
    inner: &'a S,
}

impl<'a, S: Source + ?Sized> PhysicalSource<'a, S> {
    /// Borrows an original source in its existing coordinate frame.
    pub(crate) const fn new(inner: &'a S) -> Self {
        Self { inner }
    }

    /// Uses the opened image's effective address context to interpret stored positions.
    ///
    /// The returned view borrows the same source and applies the effective base to each request.
    pub(crate) const fn stored(self, addresses: AddressContext) -> StoredSource<'a, S> {
        StoredSource {
            physical: self,
            addresses,
        }
    }

    /// Returns the original source's byte length.
    pub(crate) fn len(&self) -> u64 {
        self.inner.len()
    }

    /// Reads into `buffer` at a physical position relative to the original source's start.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Source::read_at`].
    pub(crate) fn read_at(
        &self,
        address: PhysicalAddress,
        buffer: &mut [u8],
    ) -> Result<(), FormatError> {
        self.inner.read_at(address.get(), buffer)
    }

    /// Reads a physical source region into a new vector.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Source::read_exact_at`].
    pub(crate) fn read_exact_at(
        &self,
        address: PhysicalAddress,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        self.inner.read_exact_at(address.get(), length)
    }

    /// Reads metadata through the original source's vector adapter and cache, where enabled.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Source::read_metadata_at`].
    pub(crate) fn read_metadata_at(
        &self,
        address: PhysicalAddress,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        self.inner.read_metadata_at(address.get(), length)
    }

    /// Reads metadata with the original source's ownership and physical range.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Source::read_metadata`].
    pub(crate) fn read_metadata(
        &self,
        address: PhysicalAddress,
        length: usize,
    ) -> Result<MetadataBytes<'a>, FormatError> {
        self.inner.read_metadata(address.get(), length)
    }

    /// Returns the physical position represented by original source offset zero.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Source::metadata_origin`].
    pub(crate) fn metadata_origin(&self) -> Result<PhysicalAddress, FormatError> {
        self.inner.metadata_origin().map(PhysicalAddress::new)
    }

    /// Returns the original source's metadata-cache statistics.
    pub(crate) fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.inner.metadata_cache_stats()
    }

    /// Resets the original source's metadata-cache counters, preserving cached entries.
    pub(crate) fn reset_metadata_cache_stats(&self) {
        self.inner.reset_metadata_cache_stats();
    }
}

impl<S: Source + ?Sized> RangeReader for PhysicalSource<'_, S> {
    fn read_range(&self, range: FileRange) -> Result<MetadataBytes<'_>, FormatError> {
        self.read_metadata(range.start(), range.length().to_usize()?)
    }
}

/// A borrowed source addressed relative to the opened image's effective base.
///
/// Construction through [`PhysicalSource::stored`] requires the opened image's [`AddressContext`].
/// Each request is resolved to a physical source position before reading. Metadata reads preserve
/// the original source's borrowed or owned bytes, physical ranges, and cache behavior.
pub(crate) struct StoredSource<'a, S: Source + ?Sized> {
    physical: PhysicalSource<'a, S>,
    addresses: AddressContext,
}

impl<'a, S: Source + ?Sized> StoredSource<'a, S> {
    /// Returns the number of bytes after the effective base, or zero if the base exceeds the source.
    pub(crate) fn len(&self) -> u64 {
        self.physical
            .len()
            .saturating_sub(self.addresses.effective_base().get())
    }

    /// Resolves a stored position and reads into `buffer` through the original source.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if resolving `address` exceeds [`u64::MAX`], and
    /// errors from [`Source::read_at`].
    pub(crate) fn read_at(
        &self,
        address: StoredAddress,
        buffer: &mut [u8],
    ) -> Result<(), FormatError> {
        self.physical
            .read_at(self.addresses.resolve(address)?, buffer)
    }

    /// Reads a stored-coordinate region into a new vector after checking its extent.
    ///
    /// Bounds are checked in stored coordinates before allocating or reading.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the stored end or resolved start exceeds
    /// [`u64::MAX`], [`FormatError::UnexpectedSourceEof`] if the stored end exceeds this view's
    /// length, and errors from [`Source::read_exact_at`].
    pub(crate) fn read_exact_at(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let range = AddressRange::new(address, length.to_u64())?;
        if range.end().get() > self.len() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: range.end().get(),
                available: self.len(),
            });
        }
        self.physical
            .read_exact_at(self.addresses.resolve(address)?, length)
    }

    /// Resolves a stored position and reads metadata through the original source's vector adapter.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if resolving `address` exceeds [`u64::MAX`], and
    /// errors from [`Source::read_metadata_at`].
    pub(crate) fn read_metadata_at(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        self.physical
            .read_metadata_at(self.addresses.resolve(address)?, length)
    }

    /// Resolves a stored position and reads metadata with its original ownership and physical range.
    ///
    /// After resolving the start, the original source checks the request and its physical origin.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if resolving `address` exceeds [`u64::MAX`], and
    /// errors from [`Source::read_metadata`].
    pub(crate) fn read_metadata(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<MetadataBytes<'a>, FormatError> {
        self.physical
            .read_metadata(self.addresses.resolve(address)?, length)
    }

    /// Returns the physical metadata origin corresponding to stored address zero.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the original source's metadata origin plus the
    /// effective base exceeds [`u64::MAX`], and errors from [`Source::metadata_origin`].
    pub(crate) fn metadata_origin(&self) -> Result<PhysicalAddress, FormatError> {
        self.physical
            .metadata_origin()?
            .checked_add(self.addresses.effective_base().get())
    }

    /// Returns the original source's metadata-cache statistics.
    pub(crate) fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        self.physical.metadata_cache_stats()
    }

    /// Resets the original source's metadata-cache counters, preserving cached entries.
    pub(crate) fn reset_metadata_cache_stats(&self) {
        self.physical.reset_metadata_cache_stats();
    }
}

impl<S: Source + ?Sized> RangeReader<AddressRange<StoredAddress>> for StoredSource<'_, S> {
    fn read_range(
        &self,
        range: AddressRange<StoredAddress>,
    ) -> Result<MetadataBytes<'_>, FormatError> {
        self.read_metadata(range.start(), range.length().to_usize()?)
    }
}

#[cfg(test)]
mod tests;
