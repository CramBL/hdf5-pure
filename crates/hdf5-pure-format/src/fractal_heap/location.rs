use alloc::vec::Vec;
use core::num::NonZeroU64;

use crate::address::StoredAddress;
use crate::bytes;
use crate::error::FormatError;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

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
) -> Result<(StoredAddress, NonZeroU64), FormatError> {
    let address = cursor.address(widths.offsets)?;
    let length = cursor.length(widths.lengths)?;
    if address.get() >= StoredAddress::undefined(widths.offsets).get() {
        return Err(FormatError::InvalidFractalHeapGeometry);
    }
    let length = NonZeroU64::new(length).ok_or(FormatError::InvalidFractalHeapGeometry)?;
    address
        .get()
        .checked_add(length.get())
        .ok_or(FormatError::OffsetOverflow {
            offset: address.get(),
            length: length.get(),
        })?;
    Ok((address, length))
}

/// The stored address, encoded and decoded sizes, and filter mask of a filtered huge object.
///
/// Both sizes are positive. The reader resolves the base-relative address before reading
/// [`encoded_size`](Self::encoded_size) bytes, then reverses the active filters and checks the
/// result against [`decoded_size`](Self::decoded_size).
/// If every filter is skipped, the encoded and decoded sizes are equal.
///
/// Obtained from [`FractalHeapIdLayout::parse`](super::FractalHeapIdLayout::parse) or
/// [`HugeObjectRecordDecoder::parse`](crate::btree::v2::HugeObjectRecordDecoder::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilteredHugeObjectLocation {
    address: StoredAddress,
    encoded_size: NonZeroU64,
    filter_mask: u32,
    decoded_size: NonZeroU64,
}

impl FilteredHugeObjectLocation {
    pub(crate) fn parse(
        bytes: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
        filter_count: usize,
    ) -> Result<Self, FormatError> {
        let mut cursor = bytes::ByteCursor::new(bytes);
        let address = cursor.address(offsets)?;
        let encoded_size = cursor.length(lengths)?;
        let filter_mask = cursor.u32()?;
        let decoded_size = cursor.length(lengths)?;
        if address.get() >= StoredAddress::undefined(offsets).get()
            || filter_count > u32::BITS as usize
            || (filter_count < u32::BITS as usize && filter_mask >> filter_count != 0)
        {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let encoded_size =
            NonZeroU64::new(encoded_size).ok_or(FormatError::InvalidFractalHeapGeometry)?;
        let decoded_size =
            NonZeroU64::new(decoded_size).ok_or(FormatError::InvalidFractalHeapGeometry)?;
        address
            .get()
            .checked_add(encoded_size.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: address.get(),
                length: encoded_size.get(),
            })?;
        let skipped = filter_mask.count_ones() as usize == filter_count;
        if skipped && encoded_size != decoded_size {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(Self {
            address,
            encoded_size,
            filter_mask,
            decoded_size,
        })
    }

    /// Returns the base-relative address of the encoded payload.
    pub const fn address(self) -> StoredAddress {
        self.address
    }
    /// Returns the encoded physical allocation length in bytes.
    pub const fn encoded_size(self) -> u64 {
        self.encoded_size.get()
    }
    /// Returns the skipped filters as set bits at their pipeline indices.
    pub const fn filter_mask(self) -> u32 {
        self.filter_mask
    }
    /// Returns the expected payload length after reversing its active filters.
    pub const fn decoded_size(self) -> u64 {
        self.decoded_size.get()
    }
}
