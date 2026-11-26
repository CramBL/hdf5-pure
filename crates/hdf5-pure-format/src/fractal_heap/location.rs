use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::error::FormatError;
use crate::width::FormatWidths;

/// A huge object's address and length at the field widths used to encode them.
pub(crate) struct HugeObjectLocationEncoding {
    address: StoredAddress,
    length: u64,
    widths: FormatWidths,
}

impl HugeObjectLocationEncoding {
    /// Prepares the address and length fields for an unfiltered huge object.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if `address` exceeds the offset width,
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals the undefined address at that
    /// width, or [`FormatError::LengthTooLarge`] if `length` exceeds the length width.
    pub(crate) fn new(
        widths: FormatWidths,
        address: StoredAddress,
        length: u64,
    ) -> Result<Self, FormatError> {
        bytes::require_defined_offset(address.get(), widths.offsets)?;
        bytes::require_length(length, widths.lengths)?;
        Ok(Self {
            address,
            length,
            widths,
        })
    }

    /// Appends the address followed by the length in little-endian order.
    pub(crate) fn append(self, buf: &mut Vec<u8>) {
        let Self {
            address,
            length,
            widths,
        } = self;
        bytes::write_offset(buf, address.get(), widths.offsets);
        bytes::write_length(buf, length, widths.lengths);
    }
}

/// Reads the address and positive byte length of an unfiltered huge object.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if either field is truncated,
/// [`FormatError::InvalidFractalHeapGeometry`] if the address is undefined or the length is zero,
/// or [`FormatError::OffsetOverflow`] if the end of the object exceeds [`u64::MAX`].
pub(crate) fn read_unfiltered(
    cursor: &mut bytes::ByteCursor<'_>,
    widths: FormatWidths,
) -> Result<(StoredAddress, u64), FormatError> {
    let address = cursor.address(widths.offsets)?;
    let length = cursor.length(widths.lengths)?;
    if address.get() >= StoredAddress::undefined(widths.offsets).get() || length == 0 {
        return Err(FormatError::InvalidFractalHeapGeometry);
    }
    address
        .get()
        .checked_add(length)
        .ok_or(FormatError::OffsetOverflow {
            offset: address.get(),
            length,
        })?;
    Ok((address, length))
}
