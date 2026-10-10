//! Address interpretation for an opened HDF5 image.

pub(crate) use hdf5_pure_core::__private::BaseAddressExt;
pub(crate) use hdf5_pure_core::__private::StoredAddress;
pub use hdf5_pure_core::BaseAddress;

use crate::error::FormatError;

/// The mapping between stored addresses and physical positions in an opened image.
///
/// The reader adds the effective base to a [`StoredAddress`] to locate metadata in the source.
/// The editor subtracts it when encoding a physical position in metadata.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AddressContext {
    /// The physical position corresponding to stored address zero.
    effective_base: BaseAddress,
}

impl AddressContext {
    /// Creates an address context with the supplied effective base.
    pub(crate) const fn new(effective_base: BaseAddress) -> Self {
        Self { effective_base }
    }

    /// Returns the physical position of a stored address in this image.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum of the stored address and effective base
    /// exceeds [`u64::MAX`].
    pub(crate) fn absolute(self, stored: StoredAddress) -> Result<u64, FormatError> {
        self.effective_base.absolute(stored)
    }

    /// Returns the stored address corresponding to a physical position in this image.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if the physical position is below the effective
    /// base.
    pub(crate) fn relative(self, absolute: u64) -> Result<StoredAddress, FormatError> {
        self.effective_base.relative(absolute)
    }

    /// Returns the base used to interpret metadata addresses in this image.
    pub(crate) const fn effective_base(self) -> BaseAddress {
        self.effective_base
    }
}
