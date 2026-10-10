//! Base addresses for translating stored HDF5 addresses into file positions.
//!
//! The superblock stores the byte offset where the HDF5 image begins. A stored
//! address is relative to that base, and the workspace crates convert it to an absolute file
//! position through [`BaseAddressExt`]. The base is zero when the file has no userblock.
//!
//! The base address is defined in "Format Signature and Superblock" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super

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
    /// Returns the base as a plain integer.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The constructors and conversions of a [`BaseAddress`] that the workspace crates share.
///
/// The trait's items are not part of the public API of [`BaseAddress`]. The workspace crates
/// import the trait from the hidden `__private` module.
pub trait BaseAddressExt {
    /// The base address of a file with no userblock, where a stored address and an absolute file
    /// position are the same number.
    ///
    /// A caller passes this constant where the base shifts nothing it reads: a file it has
    /// established to have no userblock, or a view of the file already framed at the base.
    const ZERO: Self;

    /// Creates a base address from the value a superblock stores.
    fn new(base: u64) -> Self;

    /// Returns `true` if a stored address and an absolute file position are the same number.
    ///
    /// The framing helpers in `hdf5-pure` are the identity for such a file, and a streaming
    /// read passes the inner source along as it is.
    fn is_zero(&self) -> bool;

    /// Returns the absolute file position of `stored`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds `u64`, which a malformed file
    /// can arrange: the file supplies both operands.
    fn absolute(self, stored: StoredAddress) -> Result<u64, FormatError>;

    /// Returns the stored form of the absolute file position `at`.
    ///
    /// A writer stores this value in the metadata that refers to `at`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if `at` lies below the base, inside the
    /// userblock, where no HDF5 structure lives.
    fn relative(self, at: u64) -> Result<StoredAddress, FormatError>;
}

#[doc(hidden)]
impl BaseAddressExt for BaseAddress {
    const ZERO: Self = Self(0);

    fn new(base: u64) -> Self {
        Self(base)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    fn absolute(self, stored: StoredAddress) -> Result<u64, FormatError> {
        stored
            .get()
            .checked_add(self.0)
            .ok_or(FormatError::OffsetOverflow {
                offset: stored.get(),
                length: self.0,
            })
    }

    fn relative(self, at: u64) -> Result<StoredAddress, FormatError> {
        at.checked_sub(self.0)
            .map(StoredAddress::new)
            .ok_or(FormatError::AddressBelowBase {
                address: at,
                base: self.0,
            })
    }
}

/// An HDF5 metadata address relative to the file's address base.
///
/// The numeric value returned by [`get`](Self::get) is the encoded address. The reader adds the
/// opened image's effective base when locating metadata in a physical source. A stored address may
/// contain the undefined address, which [`is_undefined`](Self::is_undefined) recognizes
/// at the file's offset width.
///
/// File addresses are defined in "Format Signature and Superblock" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StoredAddress(u64);

impl StoredAddress {
    /// Returns the encoded base-relative value.
    pub const fn get(self) -> u64 {
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
    pub fn is_undefined(self, offset_size: u8) -> bool {
        match offset_size {
            2 => self.0 == 0xFFFF,
            4 => self.0 == 0xFFFF_FFFF,
            8 => self.0 == u64::MAX,
            _ => false,
        }
    }
}

/// The constructors and address arithmetic shared by the workspace crates.
///
/// The trait's items are outside the supported API of [`StoredAddress`]. The workspace crates
/// import the trait from the hidden `__private` module.
pub trait StoredAddressExt {
    /// The stored address at the file's address base.
    const ZERO: Self;

    /// The undefined address for an 8-byte address field.
    const UNDEFINED_64: Self;

    /// Creates a stored address from an encoded base-relative value.
    fn new(stored: u64) -> Self;

    /// Creates the undefined address for an `offset_size`-byte address field.
    ///
    /// Uses [`u64::MAX`] as the stored value for an `offset_size` other than 2, 4, or 8. At those widths,
    /// [`StoredAddress::is_undefined`] returns `false`.
    fn undefined(offset_size: u8) -> Self;

    /// Returns the address `delta` bytes past this one.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the sum exceeds [`u64::MAX`].
    fn offset(self, delta: u64) -> Self;
}

#[doc(hidden)]
impl StoredAddressExt for StoredAddress {
    const ZERO: Self = Self(0);
    const UNDEFINED_64: Self = Self(u64::MAX);

    fn new(stored: u64) -> Self {
        Self(stored)
    }

    fn undefined(offset_size: u8) -> Self {
        Self(match offset_size {
            2 => 0xFFFF,
            4 => 0xFFFF_FFFF,
            _ => 0xFFFF_FFFF_FFFF_FFFF,
        })
    }

    fn offset(self, delta: u64) -> Self {
        Self(self.0 + delta)
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

    #[rstest]
    #[case(2, 0xFFFF)]
    #[case(4, 0xFFFF_FFFF)]
    #[case(8, u64::MAX)]
    fn the_undefined_address_sentinel_is_as_wide_as_the_offset_field(
        #[case] offset_size: u8,
        #[case] sentinel: u64,
    ) {
        assert_eq!(
            StoredAddress::undefined(offset_size),
            StoredAddress::new(sentinel),
            "the constructor and the predicate name the same value at this width"
        );
        assert!(StoredAddress::new(sentinel).is_undefined(offset_size));
        assert!(!StoredAddress::new(sentinel - 1).is_undefined(offset_size));
        assert_eq!(
            StoredAddress::new(u64::MAX).is_undefined(offset_size),
            offset_size == 8,
            "a wider file's sentinel is an ordinary address in a narrower one"
        );
    }
}
