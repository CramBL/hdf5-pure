//! Complete allocations, encoded ranges, and address dependencies for planned blocks.

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::AddressRange;
use hdf5_pure_space::__private::{Extent, ReservationRequest};

use crate::address::{AddressContext, StoredAddress};
use crate::error::FormatError;

/// A complete physical allocation and the contained range used for its encoding.
///
/// The caller retains ownership of the complete allocation when writing only the encoded bytes.
pub(super) struct PlannedBlock {
    allocation: FileRange,
    encoding: FileRange,
    request: BlockRequest,
}

impl PlannedBlock {
    /// Places an encoding at the start of a complete allocation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the allocation length differs from the request.
    pub(super) fn new(allocation: FileRange, request: BlockRequest) -> Result<Self, FormatError> {
        if allocation.length() != request.reservation().size().get() {
            return Err(FormatError::Internal(
                "planned block allocation length differs from its request".into(),
            ));
        }
        let encoding = FileRange::new(allocation.start(), request.encoded_length())?;
        Ok(Self {
            allocation,
            encoding,
            request,
        })
    }

    /// Places an encoding inside a complete stored-address allocation.
    ///
    /// Resolves both ranges through the supplied address context.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SerializationError`] if the encoding is outside the allocation,
    /// [`FormatError::Internal`] if the allocation length differs from the request, or
    /// [`FormatError::OffsetOverflow`] if stored or physical range arithmetic overflows.
    pub(super) fn within(
        addresses: AddressContext,
        allocation: Extent,
        address: StoredAddress,
        request: BlockRequest,
    ) -> Result<Self, FormatError> {
        let encoding = AddressRange::new(address, request.encoded_length())?;
        if !allocation.range().contains(encoding) {
            return Err(FormatError::SerializationError(format!(
                "a commit's tail reserved [{}, {}) but placed {} bytes at {}",
                allocation.start().get(),
                allocation.end().get(),
                request.encoded_length(),
                address.get(),
            )));
        }
        let mut block = Self::new(
            FileRange::new(addresses.resolve(allocation.start())?, allocation.len())?,
            request,
        )?;
        block.encoding = FileRange::new(addresses.resolve(address)?, request.encoded_length())?;
        Ok(block)
    }

    /// Returns the complete owned physical range.
    pub(super) fn allocation(&self) -> FileRange {
        self.allocation
    }

    /// Returns the physical range used for the encoded bytes.
    pub(super) fn encoding(&self) -> FileRange {
        self.encoding
    }

    /// Returns the allocation request, encoded length, and address dependency.
    pub(super) fn request(&self) -> BlockRequest {
        self.request
    }

    /// Associates encoded bytes with their planned physical range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SerializationError`] if the byte length differs from the planned
    /// encoding length.
    pub(super) fn region<'a>(&self, bytes: &'a [u8]) -> Result<FileRegion<'a>, FormatError> {
        if bytes.len() as u64 != self.encoding.length() {
            return Err(FormatError::SerializationError(format!(
                "a placement reserved {} bytes but built {}",
                self.encoding.length(),
                bytes.len(),
            )));
        }
        FileRegion::new(self.encoding(), bytes)
    }
}

/// Allocation capacity, encoded length, and address requirements for one block.
///
/// The encoded length is at most the complete nonempty allocation length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BlockRequest {
    reservation: ReservationRequest,
    encoded_length: u64,
    dependency: EncodingDependency,
}

impl BlockRequest {
    /// Creates a block request whose encoding fits within its allocation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SerializationError`] if `encoded_length` exceeds the reservation size.
    pub(super) fn new(
        reservation: ReservationRequest,
        encoded_length: u64,
        dependency: EncodingDependency,
    ) -> Result<Self, FormatError> {
        if encoded_length > reservation.size().get() {
            return Err(FormatError::SerializationError(format!(
                "a block reserves {} bytes but encodes {encoded_length}",
                reservation.size(),
            )));
        }
        Ok(Self {
            reservation,
            encoded_length,
            dependency,
        })
    }

    /// Creates a request whose encoded length equals its allocation length.
    pub(super) fn complete(
        reservation: ReservationRequest,
        dependency: EncodingDependency,
    ) -> Self {
        Self {
            reservation,
            encoded_length: reservation.size().get(),
            dependency,
        }
    }

    /// Returns the complete allocation request and its effective PAGE type.
    pub(super) fn reservation(self) -> ReservationRequest {
        self.reservation
    }

    /// Returns the encoded byte length, which may be smaller than the allocation.
    pub(super) fn encoded_length(self) -> u64 {
        self.encoded_length
    }

    /// Returns the addresses required before encoding.
    pub(super) fn dependency(self) -> EncodingDependency {
        self.dependency
    }
}

/// The allocation addresses required to encode a block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EncodingDependency {
    /// The block's own allocation address.
    Allocation,
    /// No allocation addresses.
    Independent,
    /// The addresses of reservations in the plan.
    Reservations,
}

#[cfg(test)]
mod tests;
