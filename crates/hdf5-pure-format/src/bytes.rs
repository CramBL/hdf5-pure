//! Bounds-checked reads of the little-endian fields HDF5 writes on disk.
//!
//! Every format module parses the same handful of shapes: check that `needed`
//! bytes are available at `offset`, then read a 2/4/8-byte file address, a
//! 2/4/8-byte length, or a 1/2/4/8-byte variable-width integer.
//!
//! Three distinctions are worth keeping, because they are the ones the copies
//! actually disagreed on:
//!
//! * The offset and length wrappers report different errors for unsupported raw widths. That
//!   error identifies the on-disk field the width came from ("size of offsets" against "size of
//!   lengths"), so a malformed file is reported against the field that is wrong.
//! * [`read_uint_width`] is **not** an address read. Object header and link
//!   messages store a width in a two-bit flag field, where 1 is a legal value.
//!   A file address is 2, 4, or 8 and a 1 there is malformed. Giving the two
//!   cases separate names keeps a caller from widening its own accepted set by
//!   reaching for the wrong helper.
//! * The bound is checked without `offset + needed` ever overflowing. `offset`
//!   is file-derived, so on a 32-bit target that sum can wrap and admit a read
//!   past the end of the buffer (issue #140).
//!
//! Production parsing uses the `_width` functions with validated [`OffsetWidth`] or
//! [`LengthWidth`] values. Test-only helpers also exercise the raw `u8` forms to verify
//! malformed-width handling.
//!
//! Reads are from an in-memory slice.

use alloc::vec::Vec;

use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;
use crate::width;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;
use crate::width::UintWidth;

/// Checks that `needed` bytes are readable at `offset`.
///
/// `offset + needed` is computed with [`usize::checked_add`].
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `offset + needed` exceeds `data.len()` or
/// overflows `usize`.
#[inline]
pub fn ensure_len(data: &[u8], offset: usize, needed: usize) -> Result<(), FormatError> {
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
/// A match over fixed-size arrays reads each width. A loop whose trip count is the runtime `width` keeps its per-byte
/// bounds check and a live panic edge  -  LLVM cannot use the `ensure_len` that
/// ran immediately above, because the index is not a constant  -  so an 8-byte
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
        // Unreachable: every caller takes its width parsed into an
        // `OffsetWidth`, a `LengthWidth` or a `UintWidth`. Not
        // `unreachable!()`, which would turn a later caller's omitted check into
        // a panic inside a parser.
        _ => {
            debug_assert!(false, "width validated by the caller");
            0
        }
    }
}

/// Reads a file address of `offset_size` bytes at `pos`.
///
/// `offset_size` is the superblock's "Size of Offsets" field.
///
/// # Errors
///
/// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, or
/// [`FormatError::UnexpectedEof`] if fewer than `offset_size` bytes remain at `pos`.
#[inline]
#[cfg(test)]
pub(crate) fn read_offset(data: &[u8], pos: usize, offset_size: u8) -> Result<u64, FormatError> {
    read_offset_width(data, pos, OffsetWidth::try_from(offset_size)?)
}

/// Reads a file address of `width` bytes at `pos`.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if fewer than `width` bytes remain at `pos`.
#[inline]
pub fn read_offset_width(data: &[u8], pos: usize, width: OffsetWidth) -> Result<u64, FormatError> {
    ensure_len(data, pos, usize::from(width.get()))?;
    Ok(read_le(data, pos, width.get()))
}

/// Reads a length of `length_size` bytes at `pos`.
///
/// `length_size` is the superblock's "Size of Lengths" field.
///
/// # Errors
///
/// Returns [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8, or
/// [`FormatError::UnexpectedEof`] if fewer than `length_size` bytes remain at `pos`.
#[inline]
#[cfg(test)]
pub(crate) fn read_length(data: &[u8], pos: usize, length_size: u8) -> Result<u64, FormatError> {
    read_length_width(data, pos, LengthWidth::try_from(length_size)?)
}

/// Reads a length of `width` bytes at `pos`.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if fewer than `width` bytes remain at `pos`.
#[inline]
pub(crate) fn read_length_width(
    data: &[u8],
    pos: usize,
    width: LengthWidth,
) -> Result<u64, FormatError> {
    ensure_len(data, pos, usize::from(width.get()))?;
    Ok(read_le(data, pos, width.get()))
}

/// Reads an unsigned integer of `width` bytes at `pos`.
///
/// `width` is a [`UintWidth`], the width an object header or a link message
/// encodes in two bits of a flag byte. It sizes a value such as a chunk size or
/// a name length, so 1 is legal here but malformed as a file-address width.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if fewer than `width` bytes remain at
/// `pos`.
#[inline]
pub(crate) fn read_uint_width(
    data: &[u8],
    pos: usize,
    width: UintWidth,
) -> Result<u64, FormatError> {
    ensure_len(data, pos, usize::from(width.get()))?;
    Ok(read_le(data, pos, width.get()))
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
/// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, or
/// [`FormatError::UnexpectedEof`] if fewer than `offset_size` bytes remain at `pos`.
#[inline]
#[cfg(test)]
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
/// Returns [`FormatError::UnexpectedEof`] if fewer than `width` bytes remain at `pos`.
#[inline]
pub(crate) fn read_optional_offset_width(
    data: &[u8],
    pos: usize,
    width: OffsetWidth,
) -> Result<Option<u64>, FormatError> {
    let addr = read_offset_width(data, pos, width)?;
    if StoredAddress::new(addr).is_undefined(width) {
        Ok(None)
    } else {
        Ok(Some(addr))
    }
}

/// Appends the address `val` to `buf` as a little-endian field of `width` bytes, as
/// [`write_offset`] does, and returns an error where [`write_offset`] panics.
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if `val` does not fit `width` and is not `u64::MAX`,
/// and leaves `buf` unchanged.
pub(crate) fn try_write_offset(
    buf: &mut Vec<u8>,
    val: u64,
    width: OffsetWidth,
) -> Result<(), FormatError> {
    if !width::offset_width_holds(width, val) {
        return Err(FormatError::AddressTooLarge {
            address: val,
            offset_size: width.get(),
        });
    }
    write_offset(buf, val, width);
    Ok(())
}

/// Writes `addr` into the little-endian address field of `width` bytes at `pos`.
///
/// Encodes `u64::MAX` as the undefined address at `width`. Preserves bytes outside the field.
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if `addr` exceeds `width` and is not `u64::MAX`,
/// or [`FormatError::UnexpectedEof`] if the field extends past `buf`. Leaves `buf` unchanged
/// on error.
pub fn write_stored_address_at(
    buf: &mut [u8],
    pos: usize,
    addr: StoredAddress,
    width: OffsetWidth,
) -> Result<(), FormatError> {
    if !width::offset_width_holds(width, addr.get()) {
        return Err(FormatError::AddressTooLarge {
            address: addr.get(),
            offset_size: width.get(),
        });
    }
    let len = usize::from(width.get());
    ensure_len(buf, pos, len)?;
    buf[pos..pos + len].copy_from_slice(&addr.get().to_le_bytes()[..len]);
    Ok(())
}

/// Appends the decoded address `addr` to `buf` as a little-endian field of `width` bytes.
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if a defined address does not fit `width`, or
/// [`FormatError::DefinedAddressIsUndefined`] if a defined address equals the undefined
/// sentinel value for `width`, and leaves `buf` unchanged.
pub(crate) fn try_write_decoded_address(
    buf: &mut Vec<u8>,
    addr: DecodedStoredAddress,
    width: OffsetWidth,
) -> Result<(), FormatError> {
    match addr {
        DecodedStoredAddress::Undefined => {
            write_offset(buf, StoredAddress::undefined(width).get(), width);
            Ok(())
        }
        DecodedStoredAddress::Defined(addr) => {
            let val = addr.get();
            if val > width::offset_width_max(width) {
                return Err(FormatError::AddressTooLarge {
                    address: val,
                    offset_size: width.get(),
                });
            }
            if addr.is_undefined(width) {
                return Err(FormatError::DefinedAddressIsUndefined {
                    address: val,
                    offset_size: width.get(),
                });
            }
            write_offset(buf, val, width);
            Ok(())
        }
    }
}

/// Appends the length `val` to `buf` as a little-endian field of `width` bytes, as
/// [`write_length`] does, and returns an error where [`write_length`] panics.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if `val` does not fit in `width` bytes, and leaves `buf`
/// unchanged.
pub(crate) fn try_write_length(
    buf: &mut Vec<u8>,
    val: u64,
    width: LengthWidth,
) -> Result<(), FormatError> {
    if !width::length_width_holds(width, val) {
        return Err(FormatError::LengthTooLarge {
            length: val,
            length_size: width.get(),
        });
    }
    write_length(buf, val, width);
    Ok(())
}

/// Appends `val` to `buf` as a little-endian integer of `width` bytes.
///
/// Writes `u64::MAX` as the undefined address at `width`, all ones.
///
/// # Panics
///
/// Panics if `val` does not fit `width` and is not `u64::MAX`.
pub(crate) fn write_offset(buf: &mut Vec<u8>, val: u64, width: OffsetWidth) {
    assert!(
        width::offset_width_holds(width, val),
        "address {val:#x} does not fit a {}-byte address field",
        width.get()
    );
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the assertion above admits a value that fits `width`, or `u64::MAX`, whose \
                  low `width` bytes are the undefined address at `width`"
    )]
    match width {
        OffsetWidth::Two => buf.extend_from_slice(&(val as u16).to_le_bytes()),
        OffsetWidth::Four => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        OffsetWidth::Eight => buf.extend_from_slice(&val.to_le_bytes()),
    }
}

/// Appends the length `val` to `buf` as a little-endian field of `width` bytes.
///
/// # Panics
///
/// Panics if `val` does not fit in `width` bytes.
pub(crate) fn write_length(buf: &mut Vec<u8>, val: u64, width: LengthWidth) {
    assert!(
        width::length_width_holds(width, val),
        "length {val} does not fit a {}-byte length field",
        width.get()
    );
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the assertion above admits only a value that fits `width`"
    )]
    match width {
        LengthWidth::Two => buf.extend_from_slice(&(val as u16).to_le_bytes()),
        LengthWidth::Four => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        LengthWidth::Eight => buf.extend_from_slice(&val.to_le_bytes()),
    }
}

/// A cursor over the little-endian fields of a block.
///
/// Each method reads the field at the cursor and moves the cursor past it, so a parser reads a
/// block with one call per field, in the order the block stores them. A read of a field that runs past
/// the end of the block returns [`FormatError::UnexpectedEof`] and leaves the cursor in place.
pub(crate) struct Fields<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Fields<'a> {
    /// Returns a cursor at byte `pos` of the block `data`.
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    /// Returns the offset in the block of the next field.
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn u8(&mut self) -> Result<u8, FormatError> {
        let [value] = self.array()?;
        Ok(value)
    }

    pub(crate) fn u16(&mut self) -> Result<u16, FormatError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    /// Reads a length of `width` bytes.
    pub(crate) fn length(&mut self, width: LengthWidth) -> Result<u64, FormatError> {
        let value = read_length_width(self.data, self.pos, width)?;
        self.pos += usize::from(width.get());
        Ok(value)
    }

    /// Reads an absolute address of `width` bytes, such as the superblock's base address or its
    /// end-of-file address.
    pub(crate) fn offset(&mut self, width: OffsetWidth) -> Result<u64, FormatError> {
        let value = read_offset_width(self.data, self.pos, width)?;
        self.pos += usize::from(width.get());
        Ok(value)
    }

    /// Reads a file address of `width` bytes.
    pub(crate) fn address(&mut self, width: OffsetWidth) -> Result<StoredAddress, FormatError> {
        self.offset(width).map(StoredAddress::new)
    }

    /// Reads a file address of `width` bytes, decoded as [`DecodedStoredAddress`].
    pub(crate) fn decoded_address(
        &mut self,
        width: OffsetWidth,
    ) -> Result<DecodedStoredAddress, FormatError> {
        let value = width::decode_stored_address(width, self.data, self.pos)?;
        self.pos += usize::from(width.get());
        Ok(value)
    }

    /// Reads an unsigned integer of `len` bytes, such as the chunk size of a filtered chunk
    /// index element.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the block holds `len` bytes at the cursor and `len`
    /// is more than 8.
    pub(crate) fn uint(&mut self, len: usize) -> Result<u64, FormatError> {
        let bytes = self
            .data
            .get(self.pos..)
            .and_then(|rest| rest.get(..len))
            .ok_or(FormatError::UnexpectedEof {
                expected: self.pos.saturating_add(len),
                available: self.data.len(),
            })?;
        let mut value = [0; 8];
        value
            .get_mut(..len)
            .ok_or_else(|| FormatError::Internal("an integer field is wider than 8 bytes".into()))?
            .copy_from_slice(bytes);
        self.pos += len;
        Ok(u64::from_le_bytes(value))
    }

    /// Reads the next `N` bytes.
    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], FormatError> {
        let value = self
            .data
            .get(self.pos..)
            .and_then(|rest| rest.first_chunk::<N>())
            .copied()
            .ok_or(FormatError::UnexpectedEof {
                expected: self.pos.saturating_add(N),
                available: self.data.len(),
            })?;
        self.pos += N;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_len_accepts_an_exact_fit_and_rejects_one_byte_more() {
        let data = [0u8; 8];
        ensure_len(&data, 4, 4).unwrap();
        assert_eq!(
            ensure_len(&data, 4, 5).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 9,
                available: 8
            }
        );
        ensure_len(&data, 8, 0).unwrap();
    }

    #[test]
    fn ensure_len_reports_a_wrapping_offset_instead_of_admitting_it() {
        // `offset + needed` overflows usize. The pre-collapse copies that wrote
        // `pos + s` would wrap to a small number and pass the bound check,
        // admitting a read past the end of the buffer (issue #140).
        let data = [0u8; 16];
        let err = ensure_len(&data, usize::MAX - 1, 8).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: 16
            }
        );
    }

    #[test]
    fn the_eof_payload_reports_what_was_wanted_and_what_was_there() {
        // The payload, not just the variant. Three differently-written copies
        // fed this collapse and all three reported `pos + needed` against
        // `data.len()`. Asserting only the variant leaves the numbers free to
        // drift, and they are what a caller debugging a truncated file reads.
        let data = [0u8; 8];
        assert_eq!(
            ensure_len(&data, 6, 4).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 10,
                available: 8,
            }
        );
        // On overflow the sum saturates at `usize::MAX`.
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
        // collapse merged reach this result by different routes, so the test must
        // pinning: an empty read at an impossible offset is not "fits".
        let data = [0u8; 4];
        ensure_len(&data, 4, 0).unwrap();
        assert_eq!(
            ensure_len(&data, 5, 0).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 5,
                available: 4
            }
        );
    }

    #[test]
    fn every_helper_reads_every_width_it_accepts() {
        // Each helper's full accepted set, not just the one width its call
        // sites happen to use most: `read_uint_width` is otherwise only
        // exercised at 1, and `read_length` at 4.
        let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        for (width, uint_width, expected) in [
            (2u8, UintWidth::Two, 0x2211u64),
            (4, UintWidth::Four, 0x4433_2211),
            (8, UintWidth::Eight, 0x8877_6655_4433_2211),
        ] {
            assert_eq!(read_offset(&data, 0, width).unwrap(), expected);
            assert_eq!(read_length(&data, 0, width).unwrap(), expected);
            assert_eq!(read_uint_width(&data, 0, uint_width).unwrap(), expected);
        }
        assert_eq!(read_uint_width(&data, 0, UintWidth::One).unwrap(), 0x11);
    }

    #[test]
    fn each_supported_width_reads_little_endian() {
        let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert_eq!(read_offset(&data, 0, 2).unwrap(), 0x2211);
        assert_eq!(read_offset(&data, 0, 4).unwrap(), 0x4433_2211);
        assert_eq!(read_offset(&data, 0, 8).unwrap(), 0x8877_6655_4433_2211);
        assert_eq!(read_length(&data, 0, 4).unwrap(), 0x4433_2211);
        assert_eq!(read_uint_width(&data, 0, UintWidth::One).unwrap(), 0x11);
        // Reading at a non-zero position uses that position's bytes, not the
        // buffer's first ones  -  a `read_le` that ignored `pos` would still pass
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
        assert_eq!(read_uint_width(&data, 0, UintWidth::One).unwrap(), 0xAB);
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

        // One byte short of the marker is a real address, and it is returned.
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

    #[rstest::rstest]
    #[case::fits(OffsetWidth::Four, 0xFFFF_FFFF, vec![0xFF; 4])]
    #[case::undefined(OffsetWidth::Two, u64::MAX, vec![0xFF; 2])]
    #[case::eight_bytes(OffsetWidth::Eight, 0x0102, vec![2, 1, 0, 0, 0, 0, 0, 0])]
    fn an_address_that_fits_its_field_is_written_whole(
        #[case] width: OffsetWidth,
        #[case] val: u64,
        #[case] expected: Vec<u8>,
    ) {
        let mut buf = Vec::new();
        write_offset(&mut buf, val, width);

        assert_eq!(buf, expected);
    }

    #[rstest::rstest]
    #[case::two(OffsetWidth::Two, 0x0102, vec![2, 1])]
    #[case::four(OffsetWidth::Four, 0x0102, vec![2, 1, 0, 0])]
    #[case::eight(OffsetWidth::Eight, 0x0102, vec![2, 1, 0, 0, 0, 0, 0, 0])]
    #[case::undefined(OffsetWidth::Two, u64::MAX, vec![0xFF; 2])]
    fn a_stored_address_patch_preserves_the_bytes_around_its_field(
        #[case] width: OffsetWidth,
        #[case] address: u64,
        #[case] encoded: Vec<u8>,
    ) {
        let mut buf = vec![0xAA; 12];

        write_stored_address_at(&mut buf, 3, StoredAddress::new(address), width).unwrap();

        let expected = [
            vec![0xAA; 3],
            encoded.clone(),
            vec![0xAA; 9 - encoded.len()],
        ]
        .concat();
        assert_eq!(buf, expected);
    }

    #[rstest::rstest]
    #[case::too_wide(
        OffsetWidth::Two,
        0,
        0x1_0000,
        FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 },
    )]
    #[case::past_the_buffer(
        OffsetWidth::Eight,
        5,
        0x0102,
        FormatError::UnexpectedEof { expected: 13, available: 12 },
    )]
    #[case::overflowing_position(
        OffsetWidth::Two,
        usize::MAX,
        0x0102,
        FormatError::UnexpectedEof { expected: usize::MAX, available: 12 },
    )]
    fn an_invalid_stored_address_patch_leaves_the_buffer_unchanged(
        #[case] width: OffsetWidth,
        #[case] pos: usize,
        #[case] address: u64,
        #[case] expected: FormatError,
    ) {
        let mut buf = vec![0xAA; 12];

        assert_eq!(
            write_stored_address_at(&mut buf, pos, StoredAddress::new(address), width),
            Err(expected),
        );
        assert_eq!(buf, vec![0xAA; 12]);
    }

    #[test]
    #[should_panic(expected = "address 0x10000 does not fit a 2-byte address field")]
    fn an_address_wider_than_its_field_panics() {
        write_offset(&mut Vec::new(), 0x1_0000, OffsetWidth::Two);
    }

    #[rstest::rstest]
    #[case::two_sentinel(OffsetWidth::Two, 0xFFFF)]
    #[case::four_sentinel(OffsetWidth::Four, 0xFFFF_FFFF)]
    #[case::eight_sentinel(OffsetWidth::Eight, u64::MAX)]
    fn writing_a_defined_address_matching_the_undefined_sentinel_fails(
        #[case] width: OffsetWidth,
        #[case] sentinel: u64,
    ) {
        let mut buf = vec![1, 2, 3];
        let err = try_write_decoded_address(
            &mut buf,
            DecodedStoredAddress::Defined(StoredAddress::new(sentinel)),
            width,
        )
        .unwrap_err();

        assert_eq!(
            err,
            FormatError::DefinedAddressIsUndefined {
                address: sentinel,
                offset_size: width.get(),
            }
        );
        assert_eq!(buf, vec![1, 2, 3]);
    }

    #[rstest::rstest]
    #[case::two_overflow(OffsetWidth::Two, 0x1_0000)]
    #[case::two_max_token(OffsetWidth::Two, u64::MAX)]
    #[case::four_overflow(OffsetWidth::Four, 0x1_0000_0000)]
    #[case::four_max_token(OffsetWidth::Four, u64::MAX)]
    fn writing_a_defined_address_exceeding_width_fails(
        #[case] width: OffsetWidth,
        #[case] addr: u64,
    ) {
        let mut buf = vec![4, 5, 6];
        let err = try_write_decoded_address(
            &mut buf,
            DecodedStoredAddress::Defined(StoredAddress::new(addr)),
            width,
        )
        .unwrap_err();

        assert_eq!(
            err,
            FormatError::AddressTooLarge {
                address: addr,
                offset_size: width.get(),
            }
        );
        assert_eq!(buf, vec![4, 5, 6]);
    }

    #[rstest::rstest]
    #[case::undefined_two(OffsetWidth::Two, DecodedStoredAddress::Undefined, vec![0xFF, 0xFF])]
    #[case::undefined_four(OffsetWidth::Four, DecodedStoredAddress::Undefined, vec![0xFF, 0xFF, 0xFF, 0xFF])]
    #[case::undefined_eight(OffsetWidth::Eight, DecodedStoredAddress::Undefined, vec![0xFF; 8])]
    #[case::defined_two_max(OffsetWidth::Two, DecodedStoredAddress::Defined(StoredAddress::new(0xFFFE)), vec![0xFE, 0xFF])]
    #[case::defined_four_sentinel_two(OffsetWidth::Four, DecodedStoredAddress::Defined(StoredAddress::new(0xFFFF)), vec![0xFF, 0xFF, 0, 0])]
    #[case::defined_four_max(OffsetWidth::Four, DecodedStoredAddress::Defined(StoredAddress::new(0xFFFF_FFFE)), vec![0xFE, 0xFF, 0xFF, 0xFF])]
    #[case::defined_eight_sentinel_four(OffsetWidth::Eight, DecodedStoredAddress::Defined(StoredAddress::new(0xFFFF_FFFF)), vec![0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0])]
    #[case::defined_eight_max(OffsetWidth::Eight, DecodedStoredAddress::Defined(StoredAddress::new(u64::MAX - 1)), vec![0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])]
    fn writing_a_valid_decoded_address_succeeds(
        #[case] width: OffsetWidth,
        #[case] addr: DecodedStoredAddress,
        #[case] expected: Vec<u8>,
    ) {
        let mut buf = Vec::new();
        try_write_decoded_address(&mut buf, addr, width).unwrap();
        assert_eq!(buf, expected);
    }

    #[rstest::rstest]
    #[case::empty(0, 0)]
    #[case::one(1, 0x11)]
    #[case::three(3, 0x33_2211)]
    #[case::eight(8, 0x8877_6655_4433_2211)]
    fn an_integer_field_reads_little_endian_at_its_width(#[case] len: usize, #[case] value: u64) {
        let data = [0xAA, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let mut fields = Fields::new(&data, 1);
        assert_eq!(fields.uint(len).unwrap(), value);
        assert_eq!(fields.pos(), 1 + len);
    }

    #[test]
    fn an_integer_field_past_the_block_is_an_error() {
        let data = [0x11, 0x22, 0x33];
        let mut fields = Fields::new(&data, 1);
        assert_eq!(
            fields.uint(3).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 4,
                available: 3,
            }
        );
        assert_eq!(fields.pos(), 1);
    }

    #[test]
    fn an_integer_field_wider_than_8_bytes_is_an_internal_error() {
        let data = [0; 9];
        let err = Fields::new(&data, 0).uint(9).unwrap_err();
        let FormatError::Internal(message) = &err else {
            panic!("expected Internal, got {err:?}");
        };
        assert_eq!(message, "an integer field is wider than 8 bytes");
    }
}
