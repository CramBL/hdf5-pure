//! The width of a file address and the width of a length.
//!
//! The superblock stores the two widths in one byte each, in its "Size of Offsets" and "Size of
//! Lengths" fields, and a file may pair 2-byte addresses with 4-byte lengths. [`OffsetWidth`] and
//! [`LengthWidth`] are distinct types, so a function that takes one of them cannot be called with
//! the other.
//!
//! Both types parse 2, 4 and 8. `H5Pset_sizes` sets 16 as well, which exceeds the `u64` the
//! readers in [`crate::bytes`] return, so a superblock that stores 16 is rejected with
//! [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`].
//!
//! The 1, 2, 4 or 8 byte width that an object header or a link message encodes in a two-bit flag
//! field is a different value, which [`crate::bytes::read_uint_width`] reads.
//!
//! Both superblock fields are defined in "Format Signature and Superblock" of the [format
//! specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super

use crate::error::FormatError;

/// The width of a file address in bytes, the superblock's "Size of Offsets" field.
///
/// Every file address has this width:
///
/// - the addresses in the superblock, such as the root group address
/// - the addresses in object header messages
/// - the child and sibling addresses in B-tree nodes
/// - the chunk addresses in a chunk index
///
/// The C library calls this value `H5F_SIZEOF_ADDR` and sets it with `H5Pset_sizes`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OffsetWidth {
    Two,
    Four,
    /// Eight bytes per address, the width the C library writes under its default file creation
    /// property list.
    Eight,
}

impl OffsetWidth {
    /// Returns the width in bytes, as the superblock stores it.
    pub(crate) const fn get(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

impl TryFrom<u8> for OffsetWidth {
    type Error = FormatError;

    /// Parses the superblock's "Size of Offsets" byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `size` is not 2, 4, or 8.
    fn try_from(size: u8) -> Result<Self, FormatError> {
        match size {
            2 => Ok(Self::Two),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            other => Err(FormatError::InvalidOffsetSize(other)),
        }
    }
}

/// The width of a length in bytes, the superblock's "Size of Lengths" field.
///
/// Every length has this width:
///
/// - the dimension sizes in a dataspace message
/// - the data segment size of a local heap
/// - the collection size of a global heap
/// - the space and object counts in a fractal heap header
///
/// The C library calls this value `H5F_SIZEOF_SIZE` and sets it with `H5Pset_sizes`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LengthWidth {
    Two,
    Four,
    /// Eight bytes per length, the width the C library writes under its default file creation
    /// property list.
    Eight,
}

impl LengthWidth {
    /// Returns the width in bytes, as the superblock stores it.
    pub(crate) const fn get(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

impl TryFrom<u8> for LengthWidth {
    type Error = FormatError;

    /// Parses the superblock's "Size of Lengths" byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLengthSize`] if `size` is not 2, 4, or 8.
    fn try_from(size: u8) -> Result<Self, FormatError> {
        match size {
            2 => Ok(Self::Two),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            other => Err(FormatError::InvalidLengthSize(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_legal_width_parses_to_its_variant() {
        assert_eq!(OffsetWidth::try_from(2), Ok(OffsetWidth::Two));
        assert_eq!(OffsetWidth::try_from(4), Ok(OffsetWidth::Four));
        assert_eq!(OffsetWidth::try_from(8), Ok(OffsetWidth::Eight));

        assert_eq!(LengthWidth::try_from(2), Ok(LengthWidth::Two));
        assert_eq!(LengthWidth::try_from(4), Ok(LengthWidth::Four));
        assert_eq!(LengthWidth::try_from(8), Ok(LengthWidth::Eight));
    }

    #[test]
    fn a_parsed_width_reports_the_byte_it_came_from() {
        for size in [2u8, 4, 8] {
            assert_eq!(OffsetWidth::try_from(size).unwrap().get(), size);
            assert_eq!(LengthWidth::try_from(size).unwrap().get(), size);
        }
    }

    #[test]
    fn a_width_outside_2_4_and_8_is_rejected_and_the_error_reports_that_byte() {
        for size in [0u8, 1, 3, 16] {
            assert_eq!(
                OffsetWidth::try_from(size),
                Err(FormatError::InvalidOffsetSize(size))
            );
            assert_eq!(
                LengthWidth::try_from(size),
                Err(FormatError::InvalidLengthSize(size))
            );
        }
    }
}
