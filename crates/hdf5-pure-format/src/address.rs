/// An address stored in file metadata relative to the file base.
///
/// The caller adds the superblock's base address to obtain an absolute byte
/// position. A parser reading bytes framed at the base uses this value directly.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StoredAddress(u64);

impl StoredAddress {
    /// Creates a stored address from a base-relative value.
    pub const fn new(stored: u64) -> Self {
        Self(stored)
    }

    /// Creates the undefined address of a file whose addresses are `offset_size` bytes wide, the
    /// value [`is_undefined`](Self::is_undefined) recognizes at that width.
    ///
    /// An `offset_size` other than 2, 4, or 8 gives `u64::MAX`, for which
    /// [`is_undefined`](Self::is_undefined) returns `false` at that width.
    pub const fn undefined(offset_size: u8) -> Self {
        Self(match offset_size {
            2 => 0xFFFF,
            4 => 0xFFFF_FFFF,
            _ => 0xFFFF_FFFF_FFFF_FFFF,
        })
    }

    /// Returns the address as a plain integer.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the address `delta` bytes past this one.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the sum exceeds `u64::MAX`.
    pub const fn offset(self, delta: u64) -> Self {
        Self(self.0 + delta)
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

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
