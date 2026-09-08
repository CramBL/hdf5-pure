//! Bounds-checked reads of the little-endian fields HDF5 writes on disk.
//!
//! Every format module parses the same handful of shapes: check that `needed`
//! bytes are available at `offset`, then read a 2/4/8-byte file address, a
//! 2/4/8-byte length, or a 1/2/4/8-byte variable-width integer. Before this
//! module each one carried its own private copy — 42 of them across 21 files,
//! several byte-identical — which is the duplication [`crate::source`] named as
//! step 2 of its plan and never collapsed.
//!
//! Three distinctions are worth keeping, because they are the ones the copies
//! actually disagreed on:
//!
//! * [`read_offset`] and [`read_length`] differ only in which error they report
//!   for an unsupported width. That error names the on-disk field the width came
//!   from ("size of offsets" against "size of lengths"), so a malformed file is
//!   reported against the field that is wrong rather than a generic one.
//! * [`read_uint_width`] is **not** an address read. Object header and link
//!   messages store a width in a two-bit flag field, where 1 is a legal value;
//!   a file address is 2, 4, or 8 and a 1 there is malformed. Giving the two
//!   cases separate names keeps a caller from widening its own accepted set by
//!   reaching for the wrong helper.
//! * The bound is checked without `offset + needed` ever overflowing. `offset`
//!   is file-derived, so on a 32-bit target that sum can wrap and admit a read
//!   past the end of the buffer (issue #140).
//!
//! Each address and length read comes in two forms. A `_width` function takes a
//! width already parsed into an [`OffsetWidth`] or a [`LengthWidth`]. The plain
//! form takes the `u8` the superblock stores and parses it first.
//!
//! Reads are from an in-memory slice. A parser working through [`crate::source`]
//! reads its bytes into a buffer first and then uses these on that buffer.

use crate::convert::is_undefined_addr;
use crate::error::FormatError;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Checks that `needed` bytes are readable at `offset`.
///
/// Returns [`FormatError::UnexpectedEof`] otherwise. `offset + needed` is
/// computed with [`usize::checked_add`], so a file-derived `offset` near
/// `usize::MAX` reports the truncation rather than wrapping into a valid-looking
/// range.
#[inline]
pub(crate) fn ensure_len(data: &[u8], offset: usize, needed: usize) -> Result<(), FormatError> {
    match offset.checked_add(needed) {
        Some(end) if end <= data.len() => Ok(()),
        _ => Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(needed),
            available: data.len(),
        }),
    }
}

/// Reads a little-endian unsigned integer of `width` bytes at `pos`.
///
/// Each caller restricts `width` to 1, 2, 4 or 8 before the call.
///
/// Written as a match over fixed-size arrays rather than a loop over `width`
/// bytes. A loop whose trip count is the runtime `width` keeps its per-byte
/// bounds check and a live panic edge — LLVM cannot use the `ensure_len` that
/// ran immediately above, because the index is not a constant — so an 8-byte
/// address costs eight loads and eight branches. Each arm here hands
/// `from_le_bytes` a constant-length array, which compiles to one load.
#[inline]
fn read_le(data: &[u8], pos: usize, width: u8) -> u64 {
    match width {
        1 => u64::from(data[pos]),
        2 => u64::from(u16::from_le_bytes([data[pos], data[pos + 1]])),
        4 => u64::from(u32::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ])),
        8 => u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]),
        // Unreachable: `read_offset_width` and `read_length_width` take a width
        // parsed into its type, and `read_uint_width` matches its own legal
        // set. Not `unreachable!()`, which would turn a later caller's omitted
        // check into a panic inside a parser.
        _ => {
            debug_assert!(false, "width validated by the caller");
            0
        }
    }
}

/// Reads a file address of `offset_size` bytes at `pos`.
///
/// `offset_size` is the superblock's "Size of Offsets" field. For the 1, 2, 4
/// or 8 byte width that an object header or a link message encodes in a flag
/// field, use [`read_uint_width`].
///
/// # Errors
///
/// [`FormatError::InvalidOffsetSize`] for an `offset_size` outside 2, 4 and 8,
/// and [`FormatError::UnexpectedEof`] when fewer than `offset_size` bytes
/// remain at `pos`.
#[inline]
pub(crate) fn read_offset(data: &[u8], pos: usize, offset_size: u8) -> Result<u64, FormatError> {
    read_offset_width(data, pos, OffsetWidth::try_from(offset_size)?)
}

/// Reads a file address of `width` bytes at `pos`.
///
/// # Errors
///
/// [`FormatError::UnexpectedEof`] when fewer than `width` bytes remain at `pos`.
#[inline]
pub(crate) fn read_offset_width(
    data: &[u8],
    pos: usize,
    width: OffsetWidth,
) -> Result<u64, FormatError> {
    ensure_len(data, pos, usize::from(width.get()))?;
    Ok(read_le(data, pos, width.get()))
}

/// Reads a length of `length_size` bytes at `pos`.
///
/// `length_size` is the superblock's "Size of Lengths" field.
///
/// # Errors
///
/// [`FormatError::InvalidLengthSize`] for a `length_size` outside 2, 4 and 8,
/// and [`FormatError::UnexpectedEof`] when fewer than `length_size` bytes
/// remain at `pos`.
#[inline]
pub(crate) fn read_length(data: &[u8], pos: usize, length_size: u8) -> Result<u64, FormatError> {
    read_length_width(data, pos, LengthWidth::try_from(length_size)?)
}

/// Reads a length of `width` bytes at `pos`.
///
/// # Errors
///
/// [`FormatError::UnexpectedEof`] when fewer than `width` bytes remain at `pos`.
#[inline]
pub(crate) fn read_length_width(
    data: &[u8],
    pos: usize,
    width: LengthWidth,
) -> Result<u64, FormatError> {
    ensure_len(data, pos, usize::from(width.get()))?;
    Ok(read_le(data, pos, width.get()))
}

/// Reads a variable-width unsigned integer of `width` bytes at `pos`, where
/// `width` is 1, 2, 4, or 8.
///
/// This is the width an object header or link message encodes in a two-bit flag
/// field, for a value that is a size or an index rather than a file address —
/// so 1 is legal here and malformed in [`read_offset`].
///
/// Its error is [`FormatError::InvalidOffsetSize`], whose message names a
/// superblock field this width did not come from and a legal set that excludes
/// the 1 this function accepts. That contradicts the naming principle in the
/// module doc above, and it stands because the arm is unreachable: both callers
/// derive the width as `1 << (flags & 3)`, which is 1, 2, 4, or 8 by
/// construction. Adding a public error variant for a case no input can reach
/// would cost more than it states.
#[inline]
pub(crate) fn read_uint_width(data: &[u8], pos: usize, width: u8) -> Result<u64, FormatError> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(FormatError::InvalidOffsetSize(width));
    }
    ensure_len(data, pos, width as usize)?;
    Ok(read_le(data, pos, width))
}

/// Reads a file address of `offset_size` bytes at `pos`, as `None` when the file
/// stored the all-`0xFF` undefined address there.
///
/// The return type separates the two outcomes a caller must not confuse:
/// `Ok(None)` is the undefined address the file stored, and an address that
/// cannot be read is an `Err`. Each field this reads is optional in that sense,
/// such as a contiguous dataset's data address or a chunk index root.
///
/// # Errors
///
/// [`FormatError::InvalidOffsetSize`] for an `offset_size` outside 2, 4 and 8,
/// and [`FormatError::UnexpectedEof`] when fewer than `offset_size` bytes
/// remain at `pos`.
#[inline]
pub(crate) fn read_optional_offset(
    data: &[u8],
    pos: usize,
    offset_size: u8,
) -> Result<Option<u64>, FormatError> {
    read_optional_offset_width(data, pos, OffsetWidth::try_from(offset_size)?)
}

/// Reads a file address of `width` bytes at `pos`, as `None` when the file
/// stored the all-`0xFF` undefined address there.
///
/// The undefined address is as wide as the address field: `0xFFFF` is undefined
/// in a file with 2-byte addresses and an ordinary address in a file with 8-byte
/// ones.
///
/// # Errors
///
/// [`FormatError::UnexpectedEof`] when fewer than `width` bytes remain at `pos`.
#[inline]
pub(crate) fn read_optional_offset_width(
    data: &[u8],
    pos: usize,
    width: OffsetWidth,
) -> Result<Option<u64>, FormatError> {
    let addr = read_offset_width(data, pos, width)?;
    if is_undefined_addr(addr, width.get()) {
        Ok(None)
    } else {
        Ok(Some(addr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_len_accepts_an_exact_fit_and_rejects_one_byte_more() {
        let data = [0u8; 8];
        assert!(ensure_len(&data, 4, 4).is_ok());
        assert!(ensure_len(&data, 4, 5).is_err());
        assert!(ensure_len(&data, 8, 0).is_ok());
    }

    #[test]
    fn ensure_len_reports_a_wrapping_offset_instead_of_admitting_it() {
        // `offset + needed` overflows usize. The pre-collapse copies that wrote
        // `pos + s` would wrap to a small number and pass the bound check,
        // admitting a read past the end of the buffer (issue #140).
        let data = [0u8; 16];
        let err = ensure_len(&data, usize::MAX - 1, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn the_eof_payload_reports_what_was_wanted_and_what_was_there() {
        // The payload, not just the variant. Three differently-written copies
        // fed this collapse and all three reported `pos + needed` against
        // `data.len()`; asserting only the variant leaves the numbers free to
        // drift, and they are what a caller debugging a truncated file reads.
        let data = [0u8; 8];
        assert_eq!(
            ensure_len(&data, 6, 4).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 10,
                available: 8,
            }
        );
        // On overflow the sum saturates rather than wrapping into a small,
        // plausible-looking `expected`.
        assert_eq!(
            ensure_len(&data, usize::MAX - 1, 8).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: 8,
            }
        );
        // A read reports the same payload its own bound check produced.
        assert_eq!(
            read_offset(&data, 6, 4).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 10,
                available: 8,
            }
        );
    }

    #[test]
    fn ensure_len_of_nothing_past_the_end_is_still_out_of_range() {
        // `offset > len` with `needed == 0`. The two bound formulations this
        // collapse merged reach this answer by different routes, so it is worth
        // pinning: an empty read at an impossible offset is not "fits".
        let data = [0u8; 4];
        assert!(ensure_len(&data, 4, 0).is_ok());
        assert!(ensure_len(&data, 5, 0).is_err());
    }

    #[test]
    fn every_helper_reads_every_width_it_accepts() {
        // Each helper's full accepted set, not just the one width its call
        // sites happen to use most: `read_uint_width` is otherwise only
        // exercised at 1, and `read_length` at 4.
        let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        for (width, expected) in [
            (2u8, 0x2211u64),
            (4, 0x4433_2211),
            (8, 0x8877_6655_4433_2211),
        ] {
            assert_eq!(read_offset(&data, 0, width).unwrap(), expected);
            assert_eq!(read_length(&data, 0, width).unwrap(), expected);
            assert_eq!(read_uint_width(&data, 0, width).unwrap(), expected);
        }
        assert_eq!(read_uint_width(&data, 0, 1).unwrap(), 0x11);
    }

    #[test]
    fn each_supported_width_reads_little_endian() {
        let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert_eq!(read_offset(&data, 0, 2).unwrap(), 0x2211);
        assert_eq!(read_offset(&data, 0, 4).unwrap(), 0x4433_2211);
        assert_eq!(read_offset(&data, 0, 8).unwrap(), 0x8877_6655_4433_2211);
        assert_eq!(read_length(&data, 0, 4).unwrap(), 0x4433_2211);
        assert_eq!(read_uint_width(&data, 0, 1).unwrap(), 0x11);
        // Reading at a non-zero position uses that position's bytes, not the
        // buffer's first ones — a `read_le` that ignored `pos` would still pass
        // every assertion above.
        assert_eq!(read_offset(&data, 2, 2).unwrap(), 0x4433);
    }

    #[test]
    fn a_file_address_of_width_one_is_refused_where_a_flag_width_is_not() {
        // The distinction the two names exist to hold: 1 is a legal flag width
        // and a malformed size-of-offsets.
        let data = [0xABu8; 8];
        assert_eq!(
            read_offset(&data, 0, 1).unwrap_err(),
            FormatError::InvalidOffsetSize(1)
        );
        assert_eq!(read_uint_width(&data, 0, 1).unwrap(), 0xAB);
    }

    #[test]
    fn an_unsupported_width_names_the_field_it_came_from() {
        let data = [0u8; 8];
        assert_eq!(
            read_offset(&data, 0, 3).unwrap_err(),
            FormatError::InvalidOffsetSize(3)
        );
        assert_eq!(
            read_length(&data, 0, 3).unwrap_err(),
            FormatError::InvalidLengthSize(3)
        );
        assert_eq!(
            read_uint_width(&data, 0, 3).unwrap_err(),
            FormatError::InvalidOffsetSize(3)
        );
    }

    #[test]
    fn width_is_validated_before_the_bound_is_checked() {
        // An empty buffer would fail `ensure_len` for any width, so checking the
        // bound first would report EOF and hide the real defect (the width).
        let err = read_offset(&[], 0, 3).unwrap_err();
        assert_eq!(err, FormatError::InvalidOffsetSize(3));
    }

    #[test]
    fn the_all_ones_sentinel_reads_as_none_at_each_width() {
        let ones = [0xFFu8; 8];
        assert_eq!(read_optional_offset(&ones, 0, 2).unwrap(), None);
        assert_eq!(read_optional_offset(&ones, 0, 4).unwrap(), None);
        assert_eq!(read_optional_offset(&ones, 0, 8).unwrap(), None);

        // One byte short of the sentinel is a real address, and it is returned.
        let mut nearly = [0xFFu8; 8];
        nearly[0] = 0xFE;
        assert_eq!(
            read_optional_offset(&nearly, 0, 8).unwrap(),
            Some(0xFFFF_FFFF_FFFF_FFFE)
        );
    }

    #[test]
    fn the_sentinel_is_the_width_the_caller_asked_for() {
        // `0xFFFF` is the undefined address of a 2-byte file and an ordinary
        // address in an 8-byte one. A check written against a fixed width would
        // read the same bytes and disagree with the file it came from.
        let mut data = [0u8; 8];
        data[0] = 0xFF;
        data[1] = 0xFF;
        assert_eq!(read_optional_offset(&data, 0, 2).unwrap(), None);
        assert_eq!(read_optional_offset(&data, 0, 8).unwrap(), Some(0xFFFF));
        assert_eq!(
            read_optional_offset_width(&data, 0, OffsetWidth::Two).unwrap(),
            None
        );
        assert_eq!(
            read_optional_offset_width(&data, 0, OffsetWidth::Eight).unwrap(),
            Some(0xFFFF)
        );
    }

    #[test]
    fn a_typed_width_and_the_u8_it_parsed_from_read_the_same_value() {
        let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        for size in [2u8, 4, 8] {
            let offsets = OffsetWidth::try_from(size).unwrap();
            let lengths = LengthWidth::try_from(size).unwrap();
            assert_eq!(
                read_offset_width(&data, 0, offsets).unwrap(),
                read_offset(&data, 0, size).unwrap()
            );
            assert_eq!(
                read_length_width(&data, 0, lengths).unwrap(),
                read_length(&data, 0, size).unwrap()
            );
            assert_eq!(
                read_optional_offset_width(&data, 0, offsets).unwrap(),
                read_optional_offset(&data, 0, size).unwrap()
            );
        }
    }

    #[test]
    fn a_read_at_a_parsed_width_reports_unexpected_eof_on_a_short_buffer() {
        let data = [0xFFu8; 8];
        assert_eq!(
            read_offset_width(&data, 4, OffsetWidth::Eight).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 12,
                available: 8,
            }
        );
        assert_eq!(
            read_length_width(&data, 6, LengthWidth::Four).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 10,
                available: 8,
            }
        );
    }

    #[test]
    fn an_unreadable_address_is_an_error_rather_than_a_none() {
        // The distinction the return type exists to hold: `None` is what the
        // file said, not what could not be read. Past the end and an
        // unsupported width both report, so neither is silently a "no address
        // here" that a caller would skip over.
        let ones = [0xFFu8; 8];
        assert!(matches!(
            read_optional_offset(&ones, 4, 8),
            Err(FormatError::UnexpectedEof { .. })
        ));
        assert_eq!(
            read_optional_offset(&ones, 0, 3).unwrap_err(),
            FormatError::InvalidOffsetSize(3)
        );
    }
}
