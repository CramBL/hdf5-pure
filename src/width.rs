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
//! See "Disk Format: Level 0A - Format Signature and Superblock" in the [HDF5 file format
//! specification][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html

use crate::error::FormatError;

/// The number of bytes a file address occupies, from the superblock's "Size of Offsets" field.
///
/// Every address in the file is this wide, from the root group address in the superblock to a
/// chunk address in a B-tree. `H5Pset_sizes` sets the width on a file creation property list, and
/// `H5F_SIZEOF_ADDR` is the C library's name for the value it reads.
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
    /// A byte outside 2, 4 and 8 is rejected with [`FormatError::InvalidOffsetSize`], which
    /// reports that byte.
    fn try_from(size: u8) -> Result<Self, FormatError> {
        match size {
            2 => Ok(Self::Two),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            other => Err(FormatError::InvalidOffsetSize(other)),
        }
    }
}

/// The number of bytes a length occupies, from the superblock's "Size of Lengths" field.
///
/// A length is the size of an object in bytes, such as a local heap's data segment or a fractal
/// heap's managed space. `H5Pset_sizes` sets the width on a file creation property list, and
/// `H5F_SIZEOF_SIZE` is the C library's name for the value it reads.
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
    /// A byte outside 2, 4 and 8 is rejected with [`FormatError::InvalidLengthSize`], which
    /// reports that byte.
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
