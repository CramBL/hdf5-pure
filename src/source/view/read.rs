//! Read operations for source views with established stored-address coordinates.

use alloc::vec::Vec;

use hdf5_pure_core::AddressRange;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::StoredAddress;

use crate::error::FormatError;
use crate::source::{RangeReader, Source};

use super::{StoredBytes, StoredSource};

/// A source view addressed using the opened image's stored coordinates.
///
/// Only [`StoredSource`] and [`StoredBytes`] implement this trait. Each view is constructed with
/// the opened image's effective address context. Payload reads use the raw read path, while
/// metadata reads retain the source's physical ranges and ownership.
pub(crate) trait StoredRead:
    sealed::Sealed + RangeReader<AddressRange<StoredAddress>>
{
    /// Returns the usable byte length in the stored-coordinate frame.
    fn len(&self) -> u64;
    /// Returns the physical metadata position corresponding to stored address zero.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the effective base plus the backend's metadata
    /// origin exceeds [`u64::MAX`], and errors from [`Source::metadata_origin`].
    fn metadata_origin(&self) -> Result<PhysicalAddress, FormatError>;
    /// Resolves `address` to a position relative to the original source's start.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if `address` plus the effective base exceeds
    /// [`u64::MAX`].
    fn source_address(&self, address: StoredAddress) -> Result<PhysicalAddress, FormatError>;
    /// Reads at a stored position into the caller's buffer.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if address conversion or endpoint arithmetic
    /// overflows, [`FormatError::UnexpectedSourceEof`] if the request exceeds the source, and
    /// backend read errors.
    fn read_at(&self, address: StoredAddress, buffer: &mut [u8]) -> Result<(), FormatError>;
    /// Reads into a vector after checking the stored extent against [`len`](Self::len).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if address conversion or endpoint arithmetic
    /// overflows, [`FormatError::UnexpectedSourceEof`] if the stored end exceeds the usable length,
    /// and backend read errors.
    fn read_exact_at(&self, address: StoredAddress, length: usize) -> Result<Vec<u8>, FormatError>;
    /// Reads metadata with its original ownership and physical range.
    ///
    /// With [`StoredBytes`], the reader borrows the resident image. With [`StoredSource`], it uses
    /// the backend's metadata operation, including its cache where enabled.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if address conversion or the physical metadata
    /// range overflows, [`FormatError::UnexpectedSourceEof`] if the request exceeds the source,
    /// and backend metadata errors.
    fn read_metadata(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<MetadataBytes<'_>, FormatError>;
    /// Reads metadata through the vector-returning adapter.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Self::read_metadata`] or [`Source::read_metadata_at`].
    fn read_metadata_at(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<Vec<u8>, FormatError>;
}

impl<S: Source + ?Sized> StoredRead for StoredSource<'_, S> {
    fn len(&self) -> u64 {
        Self::len(self)
    }
    fn metadata_origin(&self) -> Result<PhysicalAddress, FormatError> {
        Self::metadata_origin(self)
    }
    fn source_address(&self, address: StoredAddress) -> Result<PhysicalAddress, FormatError> {
        self.physical.addresses.resolve(address)
    }
    fn read_at(&self, address: StoredAddress, buffer: &mut [u8]) -> Result<(), FormatError> {
        Self::read_at(self, address, buffer)
    }
    fn read_exact_at(&self, address: StoredAddress, length: usize) -> Result<Vec<u8>, FormatError> {
        Self::read_exact_at(self, address, length)
    }
    fn read_metadata(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<MetadataBytes<'_>, FormatError> {
        Self::read_metadata(self, address, length)
    }
    fn read_metadata_at(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        Self::read_metadata_at(self, address, length)
    }
}

mod sealed {
    use super::{StoredBytes, StoredSource};
    use crate::source::Source;

    /// A supported view with stored-address coordinates.
    pub trait Sealed {}
    impl<S: Source + ?Sized> Sealed for StoredSource<'_, S> {}
    impl Sealed for StoredBytes<'_> {}
}
