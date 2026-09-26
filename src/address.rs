//! Base addresses for translating stored HDF5 addresses into file positions.
//!
//! The superblock stores the byte offset where the HDF5 image begins. A stored
//! address is relative to that base, so [`BaseAddress`] converts it to an absolute
//! file position. The base is zero when the file has no userblock.
//!
//! The base address is defined in "Format Signature and Superblock" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super

pub(crate) use hdf5_pure_format::StoredAddress;

use crate::error::FormatError;

/// The byte offset at which a file's HDF5 image begins, the superblock's "Base Address" field.
///
/// Zero for a plain file, and the userblock size for a file that has one. An address a file stores
/// in its metadata is relative to it, so an absolute file position is a stored address plus this
/// value, and [`get`](Self::get) is the number itself.
///
/// The C library sets the size of the userblock with `H5Pset_userblock`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseAddress(u64);

impl BaseAddress {
    /// The base address of a file with no userblock, where a stored address and an absolute file
    /// position are the same number.
    ///
    /// A caller passes this constant where the base shifts nothing it reads: a file it has
    /// established to have no userblock, or a view of the file already framed at the base.
    pub(crate) const ZERO: Self = Self(0);

    /// Creates a base address from the value a superblock stores.
    pub(crate) const fn new(base: u64) -> Self {
        Self(base)
    }

    /// Returns the base as a plain integer.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns `true` if a stored address and an absolute file position are the same number.
    ///
    /// The framing helpers in `hdf5-pure` are the identity for such a file, and a streaming
    /// read passes the inner source along as it is.
    pub(crate) const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Returns the absolute file position of `stored`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds `u64`, which a malformed file
    /// can arrange: the file supplies both operands.
    pub(crate) fn absolute(self, stored: StoredAddress) -> Result<u64, FormatError> {
        stored
            .get()
            .checked_add(self.0)
            .ok_or(FormatError::OffsetOverflow {
                offset: stored.get(),
                length: self.0,
            })
    }

    /// Returns the stored form of the absolute file position `at`.
    ///
    /// A writer stores this value in the metadata that refers to `at`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if `at` lies below the base, inside the
    /// userblock, where no HDF5 structure lives.
    pub(crate) fn relative(self, at: u64) -> Result<StoredAddress, FormatError> {
        at.checked_sub(self.0)
            .map(StoredAddress::new)
            .ok_or(FormatError::AddressBelowBase {
                address: at,
                base: self.0,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_conversions_invert_each_other() {
        for base in [BaseAddress::ZERO, BaseAddress::new(512)] {
            for stored in [0u64, 1, 4096, u32::MAX as u64].map(StoredAddress::new) {
                let at = base.absolute(stored).unwrap();
                assert_eq!(base.relative(at).unwrap(), stored, "base {base:?}");
            }
        }
    }

    #[test]
    fn a_zero_base_is_the_identity() {
        assert!(BaseAddress::ZERO.is_zero());
        assert_eq!(
            BaseAddress::ZERO
                .absolute(StoredAddress::new(1234))
                .unwrap(),
            1234
        );
        assert_eq!(BaseAddress::ZERO.relative(1234).unwrap().get(), 1234);
    }

    #[test]
    fn a_userblock_shifts_by_its_size() {
        let base = BaseAddress::new(512);
        assert!(!base.is_zero());
        assert_eq!(base.absolute(StoredAddress::new(96)).unwrap(), 608);
        assert_eq!(base.relative(608).unwrap().get(), 96);
    }

    #[test]
    fn an_overflowing_sum_is_reported_not_wrapped() {
        let base = BaseAddress::new(512);
        assert_eq!(
            base.absolute(StoredAddress::new(u64::MAX)),
            Err(FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 512,
            })
        );
    }

    #[test]
    fn an_address_below_the_base_returns_address_below_base() {
        let base = BaseAddress::new(512);
        assert_eq!(
            base.relative(511),
            Err(FormatError::AddressBelowBase {
                address: 511,
                base: 512,
            })
        );
        // The base itself is the image's first byte, whose stored address is zero.
        assert_eq!(base.relative(512).unwrap().get(), 0);
    }
}
