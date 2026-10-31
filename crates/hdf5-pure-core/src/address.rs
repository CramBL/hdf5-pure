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
        Address::checked_advance(self, length)
    }

    /// Returns the position `length` bytes before this one, or `None` if it would be below zero.
    pub fn checked_sub(self, length: u64) -> Option<Self> {
        Address::checked_retreat(self, length)
    }

    /// Returns the byte distance from `earlier` to this position, or `None` if `earlier` is
    /// greater.
    ///
    /// Equal positions have a distance of zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::PhysicalAddress;
    ///
    /// let start = PhysicalAddress::new(512);
    /// let end = PhysicalAddress::new(608);
    /// assert_eq!(end.checked_distance_from(start), Some(96));
    /// assert_eq!(start.checked_distance_from(end), None);
    /// assert_eq!(start.checked_distance_from(start), Some(0));
    /// ```
    pub fn checked_distance_from(self, earlier: Self) -> Option<u64> {
        Address::distance_from(self, earlier)
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
///
/// The caller classifies undefined markers at the field's [`OffsetWidth`] before converting a
/// defined address.
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

    /// Resolves a stored address against this base.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    fn resolve(self, stored: StoredAddress) -> Result<PhysicalAddress, FormatError>;

    /// Returns the address of a physical position relative to this base.
    ///
    /// A writer stores this value in the metadata that refers to `at`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if `at` lies below the base, inside the
    /// userblock, where no HDF5 structure lives.
    fn relative(self, at: PhysicalAddress) -> Result<StoredAddress, FormatError>;
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

    fn resolve(self, stored: StoredAddress) -> Result<PhysicalAddress, FormatError> {
        stored
            .get()
            .checked_add(self.0)
            .map(PhysicalAddress::new)
            .ok_or(FormatError::OffsetOverflow {
                offset: stored.get(),
                length: self.0,
            })
    }

    fn relative(self, at: PhysicalAddress) -> Result<StoredAddress, FormatError> {
        at.get()
            .checked_sub(self.0)
            .map(StoredAddress::new)
            .ok_or(FormatError::AddressBelowBase {
                address: at.get(),
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

    /// Returns the encoded address `length` bytes after this one.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    pub fn checked_add(self, length: u64) -> Result<Self, FormatError> {
        Address::checked_advance(self, length)
    }

    /// Returns the encoded address `length` bytes before this one, or `None` if it would be below
    /// zero.
    pub fn checked_sub(self, length: u64) -> Option<Self> {
        Address::checked_retreat(self, length)
    }

    /// Returns the byte distance from `earlier` to this address, or `None` if `earlier` is greater.
    ///
    /// Equal addresses have a distance of zero. Both addresses use stored coordinates.
    ///
    /// # Examples
    ///
    /// ```compile_fail,E0308
    /// use hdf5_pure_core::{PhysicalAddress, StoredAddress};
    ///
    /// fn distance(end: StoredAddress, start: PhysicalAddress) -> Option<u64> {
    ///     end.checked_distance_from(start)
    /// }
    /// ```
    pub fn checked_distance_from(self, earlier: Self) -> Option<u64> {
        Address::distance_from(self, earlier)
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

/// The constructors of a [`StoredAddress`] shared by the workspace crates.
///
/// The trait's items are outside the supported API of [`StoredAddress`]. The workspace crates
/// import the trait from the hidden `__private` module.
pub trait StoredAddressExt {
    /// The stored address at the file's address base.
    const ZERO: Self;

    /// Creates a stored address from an encoded base-relative value.
    fn new(stored: u64) -> Self;
}

#[doc(hidden)]
impl StoredAddressExt for StoredAddress {
    const ZERO: Self = Self(0);

    fn new(stored: u64) -> Self {
        Self(stored)
    }
}

/// Checked byte arithmetic within one address coordinate system.
///
/// Only [`StoredAddress`] and [`PhysicalAddress`] implement this trait. Workspace crates import it
/// from the hidden `__private` module, outside the supported public API.
pub trait Address: sealed::Sealed + Copy + Ord {
    /// Returns the byte position in this address's coordinate system.
    fn value(self) -> u64;

    /// Creates an address from a byte position in its coordinate system.
    fn from_u64(value: u64) -> Self;

    /// Returns the position `length` bytes after this one.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    fn checked_advance(self, length: u64) -> Result<Self, FormatError> {
        self.value()
            .checked_add(length)
            .map(Self::from_u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: self.value(),
                length,
            })
    }

    /// Returns the position `length` bytes before this one, or `None` if it would be below zero.
    fn checked_retreat(self, length: u64) -> Option<Self> {
        self.value().checked_sub(length).map(Self::from_u64)
    }

    /// Returns the byte distance from `earlier` to this position, or `None` if `earlier` is
    /// greater.
    fn distance_from(self, earlier: Self) -> Option<u64> {
        self.value().checked_sub(earlier.value())
    }
}

impl Address for StoredAddress {
    fn value(self) -> u64 {
        self.get()
    }

    fn from_u64(value: u64) -> Self {
        Self(value)
    }
}

impl Address for PhysicalAddress {
    fn value(self) -> u64 {
        self.get()
    }

    fn from_u64(value: u64) -> Self {
        Self(value)
    }
}

mod sealed {
    /// An address type supported by the shared arithmetic operations.
    #[expect(
        unnameable_types,
        reason = "the private supertrait prevents implementations outside the supported address types"
    )]
    pub trait Sealed {}

    impl Sealed for super::StoredAddress {}
    impl Sealed for super::PhysicalAddress {}
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::stored(StoredAddress::new)]
    #[case::physical(PhysicalAddress::new)]
    fn generic_address_arithmetic_preserves_coordinates<A: Address + Debug>(
        #[case] address: fn(u64) -> A,
    ) {
        assert_eq!(address(512).checked_advance(96).unwrap(), address(608));
        assert_eq!(address(608).checked_retreat(96), Some(address(512)));
        assert_eq!(address(608).distance_from(address(512)), Some(96));
        assert_eq!(address(512).distance_from(address(608)), None);
        assert_eq!(address(0).checked_retreat(1), None);
        assert_eq!(
            address(u64::MAX).checked_advance(1).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1,
            }
        );
    }

    #[rstest]
    #[case::plain(0)]
    #[case::userblock(512)]
    fn the_two_conversions_invert_each_other(
        #[case] base: u64,
        #[values(0, 1, 4096, u64::from(u32::MAX))] stored: u64,
    ) {
        let base = BaseAddress::new(base);
        let stored = StoredAddress::new(stored);
        let physical = base.resolve(stored).unwrap();
        assert_eq!(physical, PhysicalAddress::new(base.get() + stored.get()));
        assert_eq!(base.relative(physical).unwrap(), stored);
    }

    #[test]
    fn a_zero_base_is_the_identity() {
        assert!(BaseAddress::ZERO.is_zero());
        assert_eq!(
            BaseAddress::ZERO
                .resolve(StoredAddress::new(1234))
                .unwrap()
                .get(),
            1234
        );
        assert_eq!(
            BaseAddress::ZERO
                .relative(PhysicalAddress::new(1234))
                .unwrap()
                .get(),
            1234
        );
    }

    #[test]
    fn a_userblock_shifts_by_its_size() {
        let base = BaseAddress::new(512);
        assert!(!base.is_zero());
        assert_eq!(base.resolve(StoredAddress::new(96)).unwrap().get(), 608);
        assert_eq!(base.relative(PhysicalAddress::new(608)).unwrap().get(), 96);
    }

    #[test]
    fn an_overflowing_sum_is_reported_not_wrapped() {
        let base = BaseAddress::new(512);
        assert_eq!(
            base.resolve(StoredAddress::new(u64::MAX))
                .map(PhysicalAddress::get),
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
            base.relative(PhysicalAddress::new(511)),
            Err(FormatError::AddressBelowBase {
                address: 511,
                base: 512,
            })
        );
        // The base itself is the image's first byte, whose stored address is zero.
        assert_eq!(base.relative(PhysicalAddress::new(512)).unwrap().get(), 0);
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
