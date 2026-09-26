//! The width of a file address, the width of a length, and the width a flag byte selects.
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
//! [`UintWidth`] is a third width, the 1, 2, 4 or 8 bytes an object header or a link message
//! selects with two bits of a flag byte. No superblock field holds it.
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
pub enum OffsetWidth {
    Two,
    Four,
    /// Eight bytes per address, the width the C library writes under its default file creation
    /// property list.
    Eight,
}

impl OffsetWidth {
    /// Returns the width in bytes, as the superblock stores it.
    pub const fn get(self) -> u8 {
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
pub enum LengthWidth {
    Two,
    Four,
    /// Eight bytes per length, the width the C library writes under its default file creation
    /// property list.
    Eight,
}

impl LengthWidth {
    /// Returns the width in bytes, as the superblock stores it.
    pub const fn get(self) -> u8 {
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

/// The width in bytes of an unsigned integer that two bits of a flag byte select: 1, 2, 4, or 8.
///
/// Two fields are encoded this way, each in bits 0-1 of the flag byte ahead of it: the "Size of
/// Chunk #0" field of a version 2 object header, and the "Length of Link Name" field of a link
/// message. Both hold a size, so 1 is a legal width here and malformed in an [`OffsetWidth`].
///
/// The two fields are defined in "Version 2 Data Object Header Prefix" of the [format
/// specification, version 4.0][prefix] and in "The Link Message" of the [same document][link].
///
/// [prefix]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
/// [link]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UintWidth {
    One,
    Two,
    Four,
    Eight,
}

impl UintWidth {
    /// Returns the width that bits 0-1 of `flags` select.
    pub const fn from_flags(flags: u8) -> Self {
        match flags & WIDTH_FLAG_MASK {
            0 => Self::One,
            1 => Self::Two,
            2 => Self::Four,
            _ => Self::Eight,
        }
    }

    /// Returns the narrowest width that holds `len`.
    pub fn smallest_for_len(len: usize) -> Self {
        match u32::try_from(len) {
            Ok(len) if len <= u32::from(u8::MAX) => Self::One,
            Ok(len) if len <= u32::from(u16::MAX) => Self::Two,
            Ok(_) => Self::Four,
            Err(_) => Self::Eight,
        }
    }

    /// Returns the width in bytes.
    pub const fn get(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    /// Returns the two-bit value that selects this width, as bits 0-1 of a flag byte.
    pub const fn flag_bits(self) -> u8 {
        match self {
            Self::One => 0,
            Self::Two => 1,
            Self::Four => 2,
            Self::Eight => 3,
        }
    }
}

/// Bits 0-1 of a flag byte, the two bits that select a [`UintWidth`].
const WIDTH_FLAG_MASK: u8 = 0x03;

#[cfg(test)]
mod tests {
    use rstest::rstest;

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

    #[rstest]
    #[case(0x00, UintWidth::One, 1)]
    #[case(0x01, UintWidth::Two, 2)]
    #[case(0x02, UintWidth::Four, 4)]
    #[case(0x03, UintWidth::Eight, 8)]
    fn each_flag_value_selects_the_width_it_encodes(
        #[case] flags: u8,
        #[case] expected: UintWidth,
        #[case] bytes: u8,
    ) {
        assert_eq!(UintWidth::from_flags(flags), expected);
        assert_eq!(expected.get(), bytes);
        assert_eq!(expected.flag_bits(), flags);
    }

    #[rstest]
    #[case(0, UintWidth::One)]
    #[case(255, UintWidth::One)]
    #[case(256, UintWidth::Two)]
    #[case(65_535, UintWidth::Two)]
    #[case(65_536, UintWidth::Four)]
    fn a_length_takes_the_smallest_width_that_holds_it(
        #[case] len: usize,
        #[case] expected: UintWidth,
    ) {
        assert_eq!(UintWidth::smallest_for_len(len), expected);
    }

    #[cfg(target_pointer_width = "64")]
    #[rstest]
    #[case(4_294_967_295, UintWidth::Four)]
    #[case(4_294_967_296, UintWidth::Eight)]
    fn a_length_past_a_four_byte_field_takes_the_eight_byte_one(
        #[case] len: usize,
        #[case] expected: UintWidth,
    ) {
        assert_eq!(UintWidth::smallest_for_len(len), expected);
    }

    #[rstest]
    #[case(0xFC, UintWidth::One)]
    #[case(0xFD, UintWidth::Two)]
    #[case(0xFE, UintWidth::Four)]
    #[case(0xFF, UintWidth::Eight)]
    fn only_bits_zero_and_one_select_the_width(#[case] flags: u8, #[case] expected: UintWidth) {
        assert_eq!(UintWidth::from_flags(flags), expected);
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
