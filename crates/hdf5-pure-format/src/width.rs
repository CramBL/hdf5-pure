//! The width of a file address, the width of a length, and the width a flag byte selects.
//!
//! The superblock stores the two widths in one byte each, in its "Size of Offsets" and "Size of
//! Lengths" fields, and a file may pair 2-byte addresses with 4-byte lengths. [`OffsetWidth`] and
//! [`LengthWidth`] are distinct types, so a function that takes one of them cannot be called with
//! the other. [`FormatWidths`] holds the two widths of one file.
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

use alloc::vec::Vec;

use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_core::__private::StoredAddressExt;
pub use hdf5_pure_core::LengthWidth;
pub use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;

use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;

/// The width of a file's addresses and the width of its lengths, as its superblock stores them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormatWidths {
    pub offsets: OffsetWidth,
    pub lengths: LengthWidth,
}

impl FormatWidths {
    /// Parses the superblock's "Size of Offsets" and "Size of Lengths" bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and otherwise
    /// [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8.
    pub fn from_sizes(offset_size: u8, length_size: u8) -> Result<Self, FormatError> {
        Ok(Self {
            offsets: OffsetWidth::try_from(offset_size)?,
            lengths: LengthWidth::try_from(length_size)?,
        })
    }
}

/// Returns the minimum number of bytes needed to encode `value`, with one byte for zero.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the bit length of a u64 needs at most eight bytes"
)]
pub(crate) fn encoded_unsigned_width(value: u64) -> u8 {
    (u64::BITS - value.leading_zeros()).div_ceil(8).max(1) as u8
}

pub(crate) const fn offset_width_max(width: OffsetWidth) -> u64 {
    u64::MAX >> (64 - 8 * width.get())
}

pub(crate) fn offset_width_holds(width: OffsetWidth, address: u64) -> bool {
    address == u64::MAX
        || address
            .checked_shr(8 * u32::from(width.get()))
            .is_none_or(|high| high == 0)
}

pub(crate) fn length_width_holds(width: LengthWidth, length: u64) -> bool {
    length
        .checked_shr(8 * u32::from(width.get()))
        .is_none_or(|high| high == 0)
}

pub(crate) const fn length_width_max(width: LengthWidth) -> u64 {
    u64::MAX >> (64 - 8 * width.get())
}

pub(crate) fn decode_stored_address(
    width: OffsetWidth,
    data: &[u8],
    pos: usize,
) -> Result<DecodedStoredAddress, FormatError> {
    let stored = StoredAddress::new(bytes::read_offset_width(data, pos, width)?);
    Ok(if stored.is_undefined(width) {
        DecodedStoredAddress::Undefined
    } else {
        DecodedStoredAddress::Defined(stored)
    })
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
        Self::smallest_for_value(len.to_u64())
    }

    /// Returns the narrowest width that can represent `value`.
    pub fn smallest_for_value(value: u64) -> Self {
        match u32::try_from(value) {
            Ok(value) if value <= u32::from(u8::MAX) => Self::One,
            Ok(value) if value <= u32::from(u16::MAX) => Self::Two,
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

    /// Appends `value` to `buf` as a little-endian integer of this width.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not fit this width.
    pub(crate) fn write(self, buf: &mut Vec<u8>, value: usize) {
        let value = value.to_u64();
        assert!(
            value
                .checked_shr(8 * u32::from(self.get()))
                .is_none_or(|high| high == 0),
            "{value} does not fit a {}-byte field",
            self.get()
        );
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the assertion above admits only a value that fits the width of each arm"
        )]
        match self {
            Self::One => buf.push(value as u8),
            Self::Two => buf.extend_from_slice(&(value as u16).to_le_bytes()),
            Self::Four => buf.extend_from_slice(&(value as u32).to_le_bytes()),
            Self::Eight => buf.extend_from_slice(&value.to_le_bytes()),
        }
    }
}

/// Bits 0-1 of a flag byte, the two bits that select a [`UintWidth`].
const WIDTH_FLAG_MASK: u8 = 0x03;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::zero(0, 1)]
    #[case::one_byte_max(0xff, 1)]
    #[case::two_byte_min(0x100, 2)]
    #[case::two_byte_max(0xffff, 2)]
    #[case::three_byte_min(0x1_0000, 3)]
    #[case::three_byte_max(0xff_ffff, 3)]
    #[case::four_byte_min(0x100_0000, 4)]
    #[case::four_byte_max(0xffff_ffff, 4)]
    #[case::five_byte_min(0x1_0000_0000, 5)]
    #[case::five_byte_max(0xff_ffff_ffff, 5)]
    #[case::six_byte_min(0x100_0000_0000, 6)]
    #[case::six_byte_max(0xffff_ffff_ffff, 6)]
    #[case::seven_byte_min(0x1_0000_0000_0000, 7)]
    #[case::seven_byte_max(0xff_ffff_ffff_ffff, 7)]
    #[case::eight_byte_min(0x100_0000_0000_0000, 8)]
    #[case::eight_byte_max(u64::MAX, 8)]
    fn unsigned_encoding_uses_the_minimum_number_of_bytes(
        #[case] value: u64,
        #[case] expected: u8,
    ) {
        assert_eq!(encoded_unsigned_width(value), expected);
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, &[0xFF, 0xFF])]
    #[case::four(OffsetWidth::Four, &[0xFF, 0xFF, 0xFF, 0xFF])]
    #[case::eight(OffsetWidth::Eight, &[0xFF; 8])]
    fn address_decoding_classifies_the_undefined_field(
        #[case] width: OffsetWidth,
        #[case] bytes: &[u8],
    ) {
        assert_eq!(
            decode_stored_address(width, bytes, 0).unwrap(),
            DecodedStoredAddress::Undefined
        );
    }

    #[rstest]
    #[case::zero(OffsetWidth::Two, &[0, 0], 0, 0)]
    #[case::two(OffsetWidth::Two, &[0xFE, 0xFF], 0, 0xFFFE)]
    #[case::four(OffsetWidth::Four, &[0xFE, 0xFF, 0xFF, 0xFF], 0, 0xFFFF_FFFE)]
    #[case::eight(OffsetWidth::Eight, &[0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], 0, u64::MAX - 1)]
    #[case::two_in_four(OffsetWidth::Four, &[0xFF, 0xFF, 0, 0], 0, 0xFFFF)]
    #[case::four_in_eight(OffsetWidth::Eight, &[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0], 0, 0xFFFF_FFFF)]
    #[case::prefix(OffsetWidth::Two, &[0, 0x34, 0x12, 0xFF], 1, 0x1234)]
    fn address_decoding_preserves_defined_fields(
        #[case] width: OffsetWidth,
        #[case] bytes: &[u8],
        #[case] pos: usize,
        #[case] expected: u64,
    ) {
        assert_eq!(
            decode_stored_address(width, bytes, pos).unwrap(),
            DecodedStoredAddress::Defined(StoredAddress::new(expected))
        );
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, &[0xFF], 0, 2)]
    #[case::four(OffsetWidth::Four, &[0xFF; 3], 0, 4)]
    #[case::eight(OffsetWidth::Eight, &[0xFF; 7], 0, 8)]
    #[case::prefix(OffsetWidth::Two, &[0xFF; 2], 1, 3)]
    #[case::overflow(OffsetWidth::Two, &[], usize::MAX, usize::MAX)]
    fn address_decoding_reports_incomplete_local_fields(
        #[case] width: OffsetWidth,
        #[case] bytes: &[u8],
        #[case] pos: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(
            decode_stored_address(width, bytes, pos).unwrap_err(),
            FormatError::UnexpectedEof {
                expected,
                available: bytes.len()
            }
        );
    }

    #[rstest]
    #[case::mixed(4, 8, FormatWidths { offsets: OffsetWidth::Four, lengths: LengthWidth::Eight })]
    #[case::equal(2, 2, FormatWidths { offsets: OffsetWidth::Two, lengths: LengthWidth::Two })]
    fn sizes_parse_to_their_widths(
        #[case] offset_size: u8,
        #[case] length_size: u8,
        #[case] expected: FormatWidths,
    ) {
        assert_eq!(
            FormatWidths::from_sizes(offset_size, length_size),
            Ok(expected)
        );
    }

    #[rstest]
    #[case::offset_size(0, 8, FormatError::InvalidOffsetSize(0))]
    #[case::length_size(8, 16, FormatError::InvalidLengthSize(16))]
    #[case::both(3, 16, FormatError::InvalidOffsetSize(3))]
    fn a_size_other_than_2_4_or_8_returns_its_width_error(
        #[case] offset_size: u8,
        #[case] length_size: u8,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            FormatWidths::from_sizes(offset_size, length_size),
            Err(expected)
        );
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
    #[case(u32::MAX.into(), UintWidth::Four)]
    #[case(u64::from(u32::MAX) + 1, UintWidth::Eight)]
    #[case(u64::MAX, UintWidth::Eight)]
    fn a_value_selects_the_smallest_width_that_holds_it(
        #[case] value: u64,
        #[case] expected: UintWidth,
    ) {
        assert_eq!(UintWidth::smallest_for_value(value), expected);
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

    #[rstest]
    #[case(UintWidth::One, 0xFF, vec![0xFF])]
    #[case(UintWidth::Two, 0x0102, vec![0x02, 0x01])]
    #[case(UintWidth::Four, 0x0102, vec![0x02, 0x01, 0, 0])]
    #[case(UintWidth::Eight, 0x0102, vec![0x02, 0x01, 0, 0, 0, 0, 0, 0])]
    fn a_value_is_written_little_endian_at_its_width(
        #[case] width: UintWidth,
        #[case] value: usize,
        #[case] expected: Vec<u8>,
    ) {
        let mut buf = Vec::new();
        width.write(&mut buf, value);

        assert_eq!(buf, expected);
    }

    #[test]
    #[should_panic(expected = "256 does not fit a 1-byte field")]
    fn a_value_wider_than_its_field_panics() {
        UintWidth::One.write(&mut Vec::new(), 256);
    }
}
