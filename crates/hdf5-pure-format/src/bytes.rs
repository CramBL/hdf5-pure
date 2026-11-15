//! Borrowed byte regions and little-endian metadata fields.
//!
//! Readers acquire input spans through [`ByteCursor`] and report incomplete spans with
//! [`UnexpectedEof`]. Scalar readers decode little-endian integers at validated address, length,
//! or flag-field widths. Compatibility helpers report [`FormatError`].

use alloc::vec::Vec;

use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::{self, Narrow};
use crate::error::FormatError;
use crate::width;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;
use crate::width::UintWidth;

/// A byte span that cannot be acquired in full.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct UnexpectedEof {
    /// The span's end in the original buffer, saturated at [`usize::MAX`].
    pub(crate) expected: usize,
    /// The length of the original buffer.
    pub(crate) available: usize,
}

impl UnexpectedEof {
    /// Computes an EOF report from the unread suffix and requested byte count.
    ///
    /// Saturates the span's end at [`usize::MAX`].
    #[cold]
    fn for_suffix(available: usize, remaining: usize, needed: usize) -> Self {
        Self {
            expected: available.saturating_sub(remaining).saturating_add(needed),
            available,
        }
    }
}

impl From<UnexpectedEof> for FormatError {
    fn from(
        UnexpectedEof {
            expected,
            available,
        }: UnexpectedEof,
    ) -> Self {
        Self::UnexpectedEof {
            expected,
            available,
        }
    }
}

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
    Ok(match width {
        OffsetWidth::Two => u64::from(u16::from_le_bytes(*array_at(data, pos)?)),
        OffsetWidth::Four => u64::from(u32::from_le_bytes(*array_at(data, pos)?)),
        OffsetWidth::Eight => u64::from_le_bytes(*array_at(data, pos)?),
    })
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
    Ok(match width {
        LengthWidth::Two => u64::from(u16::from_le_bytes(*array_at(data, pos)?)),
        LengthWidth::Four => u64::from(u32::from_le_bytes(*array_at(data, pos)?)),
        LengthWidth::Eight => u64::from_le_bytes(*array_at(data, pos)?),
    })
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
    Ok(match width {
        UintWidth::One => u64::from(u8::from_le_bytes(*array_at(data, pos)?)),
        UintWidth::Two => u64::from(u16::from_le_bytes(*array_at(data, pos)?)),
        UintWidth::Four => u64::from(u32::from_le_bytes(*array_at(data, pos)?)),
        UintWidth::Eight => u64::from_le_bytes(*array_at(data, pos)?),
    })
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the address fits the width or is the all-ones undefined sentinel"
    )]
    match width {
        OffsetWidth::Two => *array_at_mut::<2>(buf, pos)? = (addr.get() as u16).to_le_bytes(),
        OffsetWidth::Four => *array_at_mut::<4>(buf, pos)? = (addr.get() as u32).to_le_bytes(),
        OffsetWidth::Eight => *array_at_mut::<8>(buf, pos)? = addr.get().to_le_bytes(),
    }
    #[cfg(test)]
    crate::field_observations::offset(width);
    Ok(())
}

/// Writes a defined address into `buf` at `pos`, preserving bytes outside the field.
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if `address` exceeds `width`,
/// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined marker, or
/// [`FormatError::UnexpectedEof`] if the field extends past `buf`. Leaves `buf` unchanged on error.
pub fn write_defined_stored_address_at(
    buf: &mut [u8],
    pos: usize,
    address: StoredAddress,
    width: OffsetWidth,
) -> Result<(), FormatError> {
    require_defined_offset(address.get(), width)?;
    write_stored_address_at(buf, pos, address, width)
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
    let stored = addr.to_stored(width)?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the converted address fits the selected width"
    )]
    match width {
        OffsetWidth::Two => buf.extend_from_slice(&(stored.get() as u16).to_le_bytes()),
        OffsetWidth::Four => buf.extend_from_slice(&(stored.get() as u32).to_le_bytes()),
        OffsetWidth::Eight => buf.extend_from_slice(&stored.get().to_le_bytes()),
    }
    #[cfg(test)]
    crate::field_observations::offset(width);
    Ok(())
}

/// Appends `val` as a defined little-endian address at `width`.
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if `val` exceeds `width`, or
/// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined marker.
/// Leaves `buf` unchanged on error.
pub(crate) fn try_write_defined_offset(
    buf: &mut Vec<u8>,
    val: u64,
    width: OffsetWidth,
) -> Result<(), FormatError> {
    try_write_decoded_address(
        buf,
        DecodedStoredAddress::Defined(StoredAddress::new(val)),
        width,
    )
}

/// Checks whether `val` is a defined address at `width`.
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if `val` exceeds `width`, or
/// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined marker.
pub(crate) fn require_defined_offset(val: u64, width: OffsetWidth) -> Result<(), FormatError> {
    DecodedStoredAddress::Defined(StoredAddress::new(val))
        .to_stored(width)
        .map(|_| ())
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
    require_length(val, width)?;
    write_length(buf, val, width);
    Ok(())
}

/// Checks whether `val` fits a `width`-byte length field.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if `val` exceeds `width`.
pub(crate) fn require_length(val: u64, width: LengthWidth) -> Result<(), FormatError> {
    if !width::length_width_holds(width, val) {
        return Err(FormatError::LengthTooLarge {
            length: val,
            length_size: width.get(),
        });
    }
    Ok(())
}

/// Writes a little-endian length into `buf` at `pos`, preserving surrounding bytes.
///
/// Checks that `value` fits `width` before checking the output range.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if `value` exceeds `width`, or
/// [`FormatError::UnexpectedEof`] if the output range exceeds `buf`. Preserves `buf` on error.
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::{__private as format, FormatError, LengthWidth};
///
/// let mut bytes = [0xcc; 4];
/// format::write_length_at(&mut bytes, 1, 0x1234, LengthWidth::Two)?;
/// assert_eq!(bytes, [0xcc, 0x34, 0x12, 0xcc]);
/// # Ok::<(), FormatError>(())
/// ```
pub fn write_length_at(
    buf: &mut [u8],
    pos: usize,
    value: u64,
    width: LengthWidth,
) -> Result<(), FormatError> {
    match width {
        LengthWidth::Two => {
            let value = u16::try_from(value).map_err(|_error| FormatError::LengthTooLarge {
                length: value,
                length_size: width.get(),
            })?;
            *array_at_mut::<2>(buf, pos)? = value.to_le_bytes();
        }
        LengthWidth::Four => {
            let value = u32::try_from(value).map_err(|_error| FormatError::LengthTooLarge {
                length: value,
                length_size: width.get(),
            })?;
            *array_at_mut::<4>(buf, pos)? = value.to_le_bytes();
        }
        LengthWidth::Eight => *array_at_mut::<8>(buf, pos)? = value.to_le_bytes(),
    }
    #[cfg(test)]
    crate::field_observations::length(width);
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
    #[cfg(test)]
    crate::field_observations::offset(width);
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
    #[cfg(test)]
    crate::field_observations::length(width);
}

/// Borrows an input array at `position`.
///
/// A zero-length array succeeds at any position through the input's end.
///
/// # Errors
///
/// Returns [`UnexpectedEof`] if the array is incomplete or its end exceeds [`usize::MAX`].
/// Reports `position.saturating_add(N)` in `expected` and the input length in `available`,
/// including when `position` exceeds the input length.
pub(crate) fn array_at<const N: usize>(
    data: &[u8],
    position: usize,
) -> Result<&[u8; N], UnexpectedEof> {
    data.get(position..)
        .and_then(<[u8]>::first_chunk::<N>)
        .ok_or_else(|| UnexpectedEof {
            expected: position.saturating_add(N),
            available: data.len(),
        })
}

/// Borrows an output array at `position`.
///
/// A zero-length array succeeds at any position through the buffer's end.
///
/// # Errors
///
/// Returns [`UnexpectedEof`] if the array is incomplete or its end exceeds [`usize::MAX`].
#[inline]
pub(crate) fn array_at_mut<const N: usize>(
    data: &mut [u8],
    position: usize,
) -> Result<&mut [u8; N], UnexpectedEof> {
    let available = data.len();
    data.get_mut(position..)
        .and_then(<[u8]>::first_chunk_mut::<N>)
        .ok_or_else(|| UnexpectedEof {
            expected: position.saturating_add(N),
            available,
        })
}

/// A byte cursor with little-endian scalar readers.
///
/// Successful reads advance past the acquired bytes. Failed reads leave the cursor unchanged.
/// Acquired spans borrow the original input and remain usable after subsequent cursor operations.
/// Positions and EOF offsets are relative to the original input, including for a cursor
/// constructed with [`Self::with_position`].
#[derive(Clone, Copy)]
pub(crate) struct ByteCursor<'a> {
    remaining: &'a [u8],
    input_len: usize,
}

impl<'a> ByteCursor<'a> {
    /// Returns a cursor at the start of `data`.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            remaining: data,
            input_len: data.len(),
        }
    }

    /// Returns a cursor at `position`, including at the end of `data`.
    ///
    /// # Errors
    ///
    /// Returns [`UnexpectedEof`] if `position` exceeds the input length, with `expected` set to
    /// `position` and `available` set to the input length.
    pub(crate) fn with_position(data: &'a [u8], position: usize) -> Result<Self, UnexpectedEof> {
        let Some((_, remaining)) = data.split_at_checked(position) else {
            return Err(UnexpectedEof {
                expected: position,
                available: data.len(),
            });
        };
        Ok(Self {
            remaining,
            input_len: data.len(),
        })
    }

    /// Returns the offset of the next byte in the original input.
    pub(crate) fn position(&self) -> usize {
        // Every constructor and mutation retains a suffix of the original input.
        // This subtraction is exact for every constructible cursor.
        let Self {
            remaining,
            input_len,
        } = self;
        input_len.saturating_sub(remaining.len())
    }

    /// Borrows the unread suffix for the original input's lifetime.
    pub(crate) fn remaining(&self) -> &'a [u8] {
        self.remaining
    }

    /// Borrows the next `len` bytes and advances the cursor.
    ///
    /// An empty span succeeds at any position through the input's end.
    ///
    /// # Errors
    ///
    /// Returns [`UnexpectedEof`] if the span is incomplete or its end exceeds [`usize::MAX`].
    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8], UnexpectedEof> {
        let Self {
            remaining,
            input_len,
        } = self;
        let Some((bytes, rest)) = remaining.split_at_checked(len) else {
            return Err(UnexpectedEof::for_suffix(*input_len, remaining.len(), len));
        };
        *remaining = rest;
        Ok(bytes)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, UnexpectedEof> {
        let [value] = self.array()?;
        Ok(value)
    }

    pub(crate) fn u16(&mut self) -> Result<u16, UnexpectedEof> {
        Ok(u16::from_le_bytes(*self.take_array()?))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, UnexpectedEof> {
        Ok(u32::from_le_bytes(*self.take_array()?))
    }

    /// Reads an eight-byte little-endian integer.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::take_array`].
    pub(crate) fn u64(&mut self) -> Result<u64, UnexpectedEof> {
        Ok(u64::from_le_bytes(*self.take_array()?))
    }

    /// Reads a length of `width` bytes.
    pub(crate) fn length(&mut self, width: LengthWidth) -> Result<u64, UnexpectedEof> {
        match width {
            LengthWidth::Two => self.u16().map(u64::from),
            LengthWidth::Four => self.u32().map(u64::from),
            LengthWidth::Eight => self.u64(),
        }
    }

    /// Reads an absolute address of `width` bytes, such as the superblock's base address or its
    /// end-of-file address.
    #[inline]
    pub(crate) fn offset(&mut self, width: OffsetWidth) -> Result<u64, UnexpectedEof> {
        Ok(match width {
            OffsetWidth::Two => u64::from(u16::from_le_bytes(*self.take_array()?)),
            OffsetWidth::Four => u64::from(u32::from_le_bytes(*self.take_array()?)),
            OffsetWidth::Eight => u64::from_le_bytes(*self.take_array()?),
        })
    }

    /// Reads a file address of `width` bytes.
    pub(crate) fn address(&mut self, width: OffsetWidth) -> Result<StoredAddress, UnexpectedEof> {
        Ok(StoredAddress::new(match width {
            OffsetWidth::Two => u64::from(u16::from_le_bytes(*self.take_array()?)),
            OffsetWidth::Four => u64::from(u32::from_le_bytes(*self.take_array()?)),
            OffsetWidth::Eight => u64::from_le_bytes(*self.take_array()?),
        }))
    }

    /// Reads a file address of `width` bytes, decoded as [`DecodedStoredAddress`].
    pub(crate) fn decoded_address(
        &mut self,
        width: OffsetWidth,
    ) -> Result<DecodedStoredAddress, UnexpectedEof> {
        let address = self.address(width)?;
        Ok(if address.is_undefined(width) {
            DecodedStoredAddress::Undefined
        } else {
            DecodedStoredAddress::Defined(address)
        })
    }

    /// Reads an unsigned integer of `len` bytes, such as the chunk size of a filtered chunk
    /// index element.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if fewer than `len` bytes remain at the cursor, or
    /// [`FormatError::Internal`] if `len` is more than 8 and those bytes are available.
    pub(crate) fn uint(&mut self, len: usize) -> Result<u64, FormatError> {
        let mut trial = *self;
        let bytes = trial.take(len)?;
        let mut value = [0; 8];
        if len > value.len() {
            return Err(FormatError::Internal(
                "an integer field is wider than 8 bytes".into(),
            ));
        }
        for (destination, source) in value.iter_mut().zip(bytes) {
            *destination = *source;
        }
        *self = trial;
        Ok(u64::from_le_bytes(value))
    }

    /// Reads the next `len` bytes without copying.
    ///
    /// On success, advances the cursor by `len`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the range is incomplete,
    /// [`FormatError::OffsetOverflow`] if its end exceeds [`u64::MAX`], and
    /// [`FormatError::ValueTooLargeForPlatform`] if its end exceeds [`usize::MAX`].
    pub(crate) fn slice(&mut self, len: usize) -> Result<&'a [u8], FormatError> {
        convert::slice_range(self.position().to_u64(), len.to_u64())?;
        Ok(self.take(len)?)
    }

    /// Copies the next `N` bytes and advances the cursor.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::take_array`].
    #[inline]
    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], UnexpectedEof> {
        self.take_array().copied()
    }

    /// Borrows the next `N` bytes and advances the cursor.
    ///
    /// The returned reference remains valid for the original input's lifetime, including after
    /// subsequent cursor operations. A zero-length read succeeds at any position through the
    /// input's end.
    ///
    /// # Errors
    ///
    /// Returns [`UnexpectedEof`] if the range is incomplete or its end exceeds [`usize::MAX`].
    /// Leaves the cursor unchanged on error.
    #[inline]
    pub(crate) fn take_array<const N: usize>(&mut self) -> Result<&'a [u8; N], UnexpectedEof> {
        let Self {
            remaining,
            input_len,
        } = self;
        let Some((array, rest)) = remaining.split_first_chunk::<N>() else {
            return Err(UnexpectedEof::for_suffix(*input_len, remaining.len(), N));
        };
        *remaining = rest;
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::test_runner::FileFailurePersistence;
    use proptest::test_runner::TestCaseResult;

    use super::*;

    #[test]
    fn acquired_arrays_borrow_the_original_bytes_across_cursor_operations() {
        let data = [0xaa, 1, 2, 3, 4, 5, 6, 7];
        let mut cursor = ByteCursor::with_position(&data, 1).unwrap();
        let first: &[u8; 2] = cursor.take_array().unwrap();
        let second: &[u8; 4] = cursor.take_array().unwrap();
        assert_eq!(cursor.u8().unwrap(), 7);
        assert_eq!(cursor.position(), data.len());
        assert_eq!(first, &[1, 2]);
        assert_eq!(second, &[3, 4, 5, 6]);
        assert_eq!(first.as_ptr(), data[1..].as_ptr());
        assert_eq!(second.as_ptr(), data[3..].as_ptr());
    }

    #[rstest::rstest]
    #[case::empty_prefix(0, 0, true)]
    #[case::empty_end(6, 0, true)]
    #[case::empty_past_end(7, 0, false)]
    #[case::empty_largest_position(usize::MAX, 0, false)]
    #[case::exact_end(2, 4, true)]
    #[case::short(3, 4, false)]
    #[case::end_overflow(usize::MAX, 2, false)]
    #[case::largest_length(1, usize::MAX, false)]
    fn acquisition_boundaries_preserve_failed_state(
        #[case] position: usize,
        #[case] len: usize,
        #[case] succeeds: bool,
    ) {
        let data = [1, 2, 3, 4, 5, 6];
        let cursor = ByteCursor::with_position(&data, position);
        let mut output = data;
        let error = UnexpectedEof {
            expected: position.saturating_add(len),
            available: data.len(),
        };
        if position > data.len() {
            assert_eq!(
                cursor.err().unwrap(),
                UnexpectedEof {
                    expected: position,
                    available: data.len(),
                }
            );
            let array_error = UnexpectedEof {
                expected: position.saturating_add(2),
                available: data.len(),
            };
            assert_eq!(array_at::<2>(&data, position).unwrap_err(), array_error);
            for (offsets, lengths, uint) in [
                (OffsetWidth::Two, LengthWidth::Two, UintWidth::Two),
                (OffsetWidth::Four, LengthWidth::Four, UintWidth::Four),
                (OffsetWidth::Eight, LengthWidth::Eight, UintWidth::Eight),
            ] {
                let error = FormatError::UnexpectedEof {
                    expected: position.saturating_add(usize::from(offsets.get())),
                    available: data.len(),
                };
                assert_eq!(
                    read_offset_width(&data, position, offsets).unwrap_err(),
                    error
                );
                assert_eq!(
                    read_length_width(&data, position, lengths).unwrap_err(),
                    error
                );
                assert_eq!(read_uint_width(&data, position, uint).unwrap_err(), error);
            }
            assert_eq!(
                read_uint_width(&data, position, UintWidth::One).unwrap_err(),
                FormatError::UnexpectedEof {
                    expected: position.saturating_add(1),
                    available: data.len(),
                }
            );
            assert_eq!(
                array_at_mut::<2>(&mut output, position).unwrap_err(),
                array_error
            );
            if len == 0 {
                assert_eq!(array_at::<0>(&data, position).unwrap_err(), error);
                assert_eq!(array_at_mut::<0>(&mut output, position).unwrap_err(), error);
            }
            assert_eq!(output, data);
            return;
        }
        let mut cursor = cursor.unwrap();
        if succeeds {
            assert_eq!(cursor.take(len).unwrap(), &data[position..position + len]);
            assert_eq!(cursor.position(), position + len);
        } else {
            assert_eq!(cursor.take(len).unwrap_err(), error);
            assert_eq!(cursor.position(), position);
        }
        if len == 0 {
            assert_eq!(cursor.take_array::<0>().unwrap(), &[]);
            assert_eq!(cursor.array::<0>().unwrap(), []);
            assert_eq!(array_at_mut::<0>(&mut output, position).unwrap(), &[]);
            assert_eq!(cursor.position(), position);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128,
            failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
                env!("CARGO_MANIFEST_DIR"), "/tests/proptest-regressions/bytes.txt"
            )))),
            ..ProptestConfig::default()
        })]

        #[test]
        fn acquired_bytes_follow_requested_bounds(
            data in proptest::collection::vec(any::<u8>(), 8..128),
            origin in any::<usize>(),
            requested in any::<usize>(),
            succeeds in any::<bool>(),
            width in proptest::sample::select(vec![LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight]),
        ) {
            let position = origin % (data.len() + 1);
            let remaining = data.len() - position;
            let len = if succeeds {
                requested % (remaining + 1)
            } else {
                remaining + 1 + requested % 128
            };
            let mut cursor = ByteCursor::with_position(&data, position).unwrap();
            if succeeds {
                let bytes = cursor.take(len).unwrap();
                prop_assert_eq!(bytes, &data[position..position + len]);
                prop_assert_eq!(bytes.as_ptr(), data[position..].as_ptr());
                prop_assert_eq!(cursor.position(), position + len);
                prop_assert_eq!(cursor.remaining(), &data[position + len..]);
            } else {
                let error = UnexpectedEof { expected: position + len, available: data.len() };
                prop_assert_eq!(cursor.take(len).unwrap_err(), error);
                prop_assert_eq!(cursor.position(), position);
                prop_assert_eq!(cursor.remaining(), &data[position..]);
            }
            match width {
                LengthWidth::Two => assert_array_bounds::<2>(&data, origin, requested, succeeds)?,
                LengthWidth::Four => assert_array_bounds::<4>(&data, origin, requested, succeeds)?,
                LengthWidth::Eight => assert_array_bounds::<8>(&data, origin, requested, succeeds)?,
            }
        }
    }

    fn assert_array_bounds<const N: usize>(
        data: &[u8],
        origin: usize,
        requested: usize,
        succeeds: bool,
    ) -> TestCaseResult {
        let position = if succeeds {
            origin % (data.len() - N + 1)
        } else {
            data.len() - N + 1 + origin % N
        };
        let mut cursor = ByteCursor::with_position(data, position).unwrap();
        let mut output = data.to_vec();
        if succeeds {
            let array = cursor.take_array::<N>().unwrap();
            prop_assert_eq!(cursor.array::<0>().unwrap(), []);
            prop_assert_eq!(array.as_slice(), &data[position..position + N]);
            prop_assert_eq!(array.as_ptr(), data[position..].as_ptr());
            prop_assert_eq!(cursor.position(), position + N);
            let copied = ByteCursor::with_position(data, position)
                .unwrap()
                .array::<N>()
                .unwrap();
            prop_assert_eq!(copied, *array);
            array_at_mut::<N>(&mut output, position).unwrap().fill(0xaa);
            let mut expected = data.to_vec();
            expected[position..position + N].fill(0xaa);
            prop_assert_eq!(output, expected);
        } else {
            let error = UnexpectedEof {
                expected: position + N,
                available: data.len(),
            };
            prop_assert_eq!(cursor.take_array::<N>().unwrap_err(), error);
            prop_assert_eq!(cursor.position(), position);
            prop_assert_eq!(array_at_mut::<N>(&mut output, position).unwrap_err(), error);
            prop_assert_eq!(output, data);
        }
        Ok(())
    }

    #[rstest::rstest]
    #[case::two(LengthWidth::Two, u64::from(u16::MAX), &[0xff, 0xff])]
    #[case::four(LengthWidth::Four, u64::from(u32::MAX), &[0xff, 0xff, 0xff, 0xff])]
    #[case::eight(LengthWidth::Eight, u64::MAX, &[0xff; 8])]
    fn length_field_writes_preserve_surrounding_bytes(
        #[case] width: LengthWidth,
        #[case] value: u64,
        #[case] field: &[u8],
    ) {
        let mut output = [0xcc; 10];
        write_length_at(&mut output, 1, value, width).unwrap();
        assert_eq!(&output[1..1 + field.len()], field);
        assert_eq!(output[0], 0xcc);
        assert_eq!(&output[1 + field.len()..], &vec![0xcc; 9 - field.len()]);
    }

    #[rstest::rstest]
    #[case::two_overflow(LengthWidth::Two, 65_536, 1, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
    #[case::four_overflow(LengthWidth::Four, 4_294_967_296, 1, FormatError::LengthTooLarge { length: 4_294_967_296, length_size: 4 })]
    #[case::two_overflow_before_capacity(LengthWidth::Two, 65_536, usize::MAX, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
    #[case::four_overflow_before_capacity(LengthWidth::Four, 4_294_967_296, usize::MAX, FormatError::LengthTooLarge { length: 4_294_967_296, length_size: 4 })]
    #[case::short(LengthWidth::Eight, 1, 3, FormatError::UnexpectedEof { expected: 11, available: 10 })]
    #[case::offset_overflow(LengthWidth::Two, 1, usize::MAX, FormatError::UnexpectedEof { expected: usize::MAX, available: 10 })]
    fn length_field_failures_preserve_the_buffer(
        #[case] width: LengthWidth,
        #[case] value: u64,
        #[case] offset: usize,
        #[case] error: FormatError,
    ) {
        let mut output = [0xcc; 10];
        assert_eq!(
            write_length_at(&mut output, offset, value, width).unwrap_err(),
            error
        );
        assert_eq!(output, [0xcc; 10]);
    }

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
        // The field starts two bytes after the buffer's beginning.
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
        assert_eq!(
            read_optional_offset(&ones, 4, 8).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: 12,
                available: 8
            }
        );
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
        let mut cursor = ByteCursor::with_position(&data, 1).unwrap();
        assert_eq!(cursor.uint(len).unwrap(), value);
        assert_eq!(cursor.position(), 1 + len);
    }

    #[test]
    fn an_integer_field_past_the_block_is_an_error() {
        let data = [0x11, 0x22, 0x33];
        let mut cursor = ByteCursor::with_position(&data, 1).unwrap();
        for len in [3, 9] {
            assert_eq!(
                cursor.uint(len).unwrap_err(),
                FormatError::UnexpectedEof {
                    expected: 1 + len,
                    available: 3,
                }
            );
            assert_eq!(cursor.position(), 1);
        }
    }

    #[test]
    fn an_integer_field_wider_than_8_bytes_is_an_internal_error() {
        let data = [0; 10];
        let mut cursor = ByteCursor::with_position(&data, 1).unwrap();
        let err = cursor.uint(9).unwrap_err();
        let FormatError::Internal(message) = &err else {
            panic!("expected Internal, got {err:?}");
        };
        assert_eq!(message, "an integer field is wider than 8 bytes");
        assert_eq!(cursor.position(), 1);
    }

    #[rstest::rstest]
    #[case::eof(2, 2, FormatError::UnexpectedEof { expected: 4, available: 3 })]
    #[case::end_overflow(1, usize::MAX, if usize::BITS == 64 {
        FormatError::OffsetOverflow { offset: 1, length: u64::MAX }
    } else {
        FormatError::ValueTooLargeForPlatform { value: 0x1_0000_0000, target: "usize" }
    })]
    fn slice_errors_preserve_conversion_precedence_and_position(
        #[case] position: usize,
        #[case] len: usize,
        #[case] error: FormatError,
    ) {
        let mut cursor = ByteCursor::with_position(&[1, 2, 3], position).unwrap();
        assert_eq!(cursor.slice(len).unwrap_err(), error);
        assert_eq!(cursor.position(), position);
    }
}
