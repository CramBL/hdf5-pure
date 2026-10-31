//! Base addresses for translating stored HDF5 addresses into file positions.
//!
//! A [`StoredAddress`] is relative to an address base. The superblock stores the encoded base,
//! and the reader selects an effective base from the physical signature position when opening
//! the image. The workspace crates use [`BaseAddressExt`] for checked address conversions.
//!
//! The base address is defined in "Format Signature and Superblock" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super

use crate::error::FormatError;
use crate::width::OffsetWidth;

/// A byte position in a physical source.
///
/// Offsets count bytes from the beginning of the opened source, including any userblock prefix.
/// The reader resolves a [`StoredAddress`] against the opened image's effective base to obtain
/// a physical position. Physical positions are represented as [`u64`] on every platform.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::PhysicalAddress;
///
/// let start = PhysicalAddress::new(512);
/// let end = start.checked_add(32)?;
/// assert_eq!(end.get(), 544);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PhysicalAddress(u64);

impl PhysicalAddress {
    /// Creates a position from an absolute byte offset.
    pub const fn new(address: u64) -> Self {
        Self(address)
    }

    /// Returns the absolute byte offset.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the position `length` bytes after this one.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    pub fn checked_add(self, length: u64) -> Result<Self, FormatError> {
        self.0
            .checked_add(length)
            .map(Self)
            .ok_or(FormatError::OffsetOverflow {
                offset: self.0,
                length,
            })
    }
}

/// An encoded address classified at the width of its field.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DecodedStoredAddress {
    /// A base-relative address whose encoded bits are not all set.
    Defined(StoredAddress),
    /// The undefined address encoded with every bit of its field set.
    Undefined,
}

/// An absolute address used as the base for stored HDF5 addresses.
///
/// [`Superblock::base_address`](crate::Superblock::base_address) holds the encoded Base Address.
/// In a newly created file, this is the physical superblock position: zero for a plain file and
/// the userblock size for a file with a userblock. For a relocated image, the reader uses the
/// physical signature position as the effective base when interpreting [`StoredAddress`] values.
///
/// libhdf5 sets the size of the userblock with `H5Pset_userblock`.
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

    /// Creates a base address from an encoded or effective absolute address.
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
    /// Returns the undefined address of a file whose addresses are `offset_width` bytes wide.
    ///
    /// The undefined address has every bit of the address field set, so the width fixes which
    /// value it is: `0xFFFF` is undefined in a file with 2-byte addresses and an ordinary
    /// address in a file with 8-byte ones. Metadata stores it in an address field whose structure
    /// has no storage allocated, such as the section-info address of a free-space manager that
    /// tracks nothing.
    ///
    /// The undefined address is defined in "Appendix A: Definitions" of the [format
    /// specification, version 4.0][spec].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::OffsetWidth;
    /// use hdf5_pure_core::StoredAddress;
    ///
    /// let addr = StoredAddress::undefined(OffsetWidth::Two);
    /// assert_eq!(addr.get(), 0xFFFF);
    /// assert!(addr.is_undefined(OffsetWidth::Two));
    /// ```
    ///
    /// ```compile_fail
    /// use hdf5_pure_core::LengthWidth;
    /// use hdf5_pure_core::StoredAddress;
    ///
    /// let _ = StoredAddress::undefined(LengthWidth::Two);
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_appendixa
    pub const fn undefined(offset_width: OffsetWidth) -> Self {
        Self(match offset_width {
            OffsetWidth::Two => 0xFFFF,
            OffsetWidth::Four => 0xFFFF_FFFF,
            OffsetWidth::Eight => u64::MAX,
        })
    }

    /// Returns the encoded base-relative value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns `true` if this is the undefined address of a file whose addresses are
    /// `offset_width` bytes wide.
    ///
    /// The undefined address has every bit of the address field set, so the width fixes which
    /// value it is: `0xFFFF_FFFF` is undefined in a file with 4-byte addresses and an ordinary
    /// address in a file with 8-byte ones. Metadata stores it in an address field whose structure
    /// has no storage allocated, such as the section-info address of a free-space manager that
    /// tracks nothing.
    ///
    /// The undefined address is defined in "Appendix A: Definitions" of the [format
    /// specification, version 4.0][spec].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::OffsetWidth;
    /// use hdf5_pure_core::StoredAddress;
    ///
    /// let addr = StoredAddress::undefined(OffsetWidth::Two);
    /// assert!(addr.is_undefined(OffsetWidth::Two));
    /// assert!(!addr.is_undefined(OffsetWidth::Four));
    /// ```
    ///
    /// ```compile_fail
    /// use hdf5_pure_core::LengthWidth;
    /// use hdf5_pure_core::OffsetWidth;
    /// use hdf5_pure_core::StoredAddress;
    ///
    /// let addr = StoredAddress::undefined(OffsetWidth::Two);
    /// let _ = addr.is_undefined(LengthWidth::Two);
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_appendixa
    pub const fn is_undefined(self, offset_width: OffsetWidth) -> bool {
        self.0 == Self::undefined(offset_width).0
    }
}

/// The constructors and address arithmetic shared by the workspace crates.
///
/// The trait's items are outside the supported API of [`StoredAddress`]. The workspace crates
/// import the trait from the hidden `__private` module.
pub trait StoredAddressExt {
    /// The stored address at the file's address base.
    const ZERO: Self;

    /// Creates a stored address from an encoded base-relative value.
    fn new(stored: u64) -> Self;

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

    fn new(stored: u64) -> Self {
        Self(stored)
    }

    fn offset(self, delta: u64) -> Self {
        Self(self.0 + delta)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::zero(0, 0, 0)]
    #[case::ordinary(512, 96, 608)]
    #[case::end(u64::MAX - 1, 1, u64::MAX)]
    fn physical_address_addition_preserves_u64_positions(
        #[case] address: u64,
        #[case] length: u64,
        #[case] expected: u64,
    ) {
        assert_eq!(
            PhysicalAddress::new(address).checked_add(length).unwrap(),
            PhysicalAddress::new(expected)
        );
    }

    #[rstest]
    #[case::one(u64::MAX, 1)]
    #[case::many(u64::MAX - 1, 512)]
    fn physical_address_addition_rejects_overflow(#[case] address: u64, #[case] length: u64) {
        assert_eq!(
            PhysicalAddress::new(address)
                .checked_add(length)
                .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: address,
                length
            }
        );
    }

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
    #[case::two(OffsetWidth::Two, 0xFFFF)]
    #[case::four(OffsetWidth::Four, 0xFFFF_FFFF)]
    #[case::eight(OffsetWidth::Eight, u64::MAX)]
    fn the_undefined_address_sentinel_is_as_wide_as_the_offset_field(
        #[case] width: OffsetWidth,
        #[case] sentinel: u64,
    ) {
        assert_eq!(
            StoredAddress::undefined(width),
            StoredAddress::new(sentinel),
        );
        assert!(StoredAddress::new(sentinel).is_undefined(width));
        assert!(!StoredAddress::new(sentinel - 1).is_undefined(width));
        assert_eq!(
            StoredAddress::new(u64::MAX).is_undefined(width),
            width == OffsetWidth::Eight,
        );
    }

    #[rstest]
    #[case::two_under_two(OffsetWidth::Two, OffsetWidth::Two, true)]
    #[case::two_under_four(OffsetWidth::Two, OffsetWidth::Four, false)]
    #[case::two_under_eight(OffsetWidth::Two, OffsetWidth::Eight, false)]
    #[case::four_under_two(OffsetWidth::Four, OffsetWidth::Two, false)]
    #[case::four_under_four(OffsetWidth::Four, OffsetWidth::Four, true)]
    #[case::four_under_eight(OffsetWidth::Four, OffsetWidth::Eight, false)]
    #[case::eight_under_two(OffsetWidth::Eight, OffsetWidth::Two, false)]
    #[case::eight_under_four(OffsetWidth::Eight, OffsetWidth::Four, false)]
    #[case::eight_under_eight(OffsetWidth::Eight, OffsetWidth::Eight, true)]
    fn undefined_sentinel_matches_only_its_own_width(
        #[case] sentinel_width: OffsetWidth,
        #[case] query_width: OffsetWidth,
        #[case] expected: bool,
    ) {
        let addr = StoredAddress::undefined(sentinel_width);
        assert_eq!(addr.is_undefined(query_width), expected);
    }
}
