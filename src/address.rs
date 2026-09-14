//! The base address of a file's HDF5 image, and the base-relative form of a file address.
//!
//! The superblock stores the byte offset at which the image begins, and the specification makes
//! every other file address relative to that base unless it says otherwise. A writer that reserves
//! a userblock places the image after the start of the file, so an address has two forms: a
//! [`StoredAddress`], as file metadata stores it, and an absolute byte position, where a
//! [`Source`](crate::source::Source) reads. The two forms are the same number for a file with no
//! userblock. [`BaseAddress`] and [`StoredAddress`] are distinct types, so a call that passes a
//! base where an address belongs does not compile.
//!
//! A parser reaches the bytes at a stored address in one of two ways. A parser of an object header
//! or of a group entry adds the base to each address it reads, through [`BaseAddress::absolute`].
//! A parser of raw data, of a chunk index or of dense attribute storage reads against a view of
//! the file framed at the base: [`frame`](crate::source::frame) for a buffer, and
//! [`BaseOffsetSource`](crate::source::BaseOffsetSource) for a stream.
//!
//! The base address is defined in "Format Signature and Superblock" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super

use crate::convert;
use crate::error::FormatError;

/// The byte offset at which a file's HDF5 image begins, the superblock's "Base Address" field.
///
/// Zero for a plain file, and the userblock size for a file that has one, 512 bytes for every
/// `.mat` file this crate writes. An address a file stores in its metadata is relative to it, so
/// an absolute file position is a stored address plus this value, and [`get`](Self::get) is the
/// number itself.
///
/// [`Superblock::base_address`](crate::Superblock::base_address) holds the value a file stores.
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
    /// The framing helpers in [`crate::source`] are the identity for such a file, and a streaming
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

/// A file address in the form file metadata stores it, relative to the [`BaseAddress`].
///
/// The absolute byte position a [`Source`](crate::source::Source) reads at is this address plus
/// the base, which [`BaseAddress::absolute`] computes, and [`BaseAddress::relative`] goes back the
/// other way. A parser that reads against a base-framed view of the file converts nothing: its
/// addresses stay in the stored form.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct StoredAddress(u64);

impl StoredAddress {
    /// Creates a stored address from a base-relative value.
    pub(crate) const fn new(stored: u64) -> Self {
        Self(stored)
    }

    /// Returns the address as a plain integer.
    pub(crate) const fn get(self) -> u64 {
        self.0
    }

    /// Returns `true` if this is the undefined address of a file whose addresses are
    /// `offset_size` bytes wide.
    ///
    /// The undefined address has every bit of the address field set, so the width fixes which
    /// value it is: `0xFFFF_FFFF` is undefined in a file with 4-byte addresses and an ordinary
    /// address in a file with 8-byte ones. Metadata stores it in an address field whose structure
    /// has no storage allocated, such as the section-info address of a free-space manager that
    /// tracks nothing. Returns `false` for an `offset_size` other than 2, 4, or 8.
    ///
    /// The undefined address is defined in "Appendix A: Definitions" of the [format
    /// specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_appendixa
    pub(crate) fn is_undefined(self, offset_size: u8) -> bool {
        convert::is_undefined_addr(self.0, offset_size)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

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

    #[rstest]
    #[case(2, 0xFFFF)]
    #[case(4, 0xFFFF_FFFF)]
    #[case(8, u64::MAX)]
    fn the_undefined_address_sentinel_is_as_wide_as_the_offset_field(
        #[case] offset_size: u8,
        #[case] sentinel: u64,
    ) {
        assert!(StoredAddress::new(sentinel).is_undefined(offset_size));
        assert!(!StoredAddress::new(sentinel - 1).is_undefined(offset_size));
        assert_eq!(
            StoredAddress::new(u64::MAX).is_undefined(offset_size),
            offset_size == 8,
            "a wider file's sentinel is an ordinary address in a narrower one"
        );
    }

    #[test]
    fn an_address_below_the_base_is_refused_rather_than_wrapped() {
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
