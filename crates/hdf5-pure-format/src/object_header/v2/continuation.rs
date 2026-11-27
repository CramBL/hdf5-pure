use core::ops::Range;

use super::CHECKSUM_LEN;
use crate::address::StoredAddress;
use crate::bytes;
use crate::codec::Decode;
use crate::error::FormatError;
use crate::width::FormatWidths;

/// The stored address and byte length of an object header continuation block.
///
/// The fields are defined in "The Object Header Continuation Message" of the [format
/// specification, version 4.0][spec]. The continuation address is defined at the file’s offset width.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_continuation
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectHeaderContinuation {
    /// The address of the continuation block.
    address: StoredAddress,
    /// The length of the continuation block in bytes.
    length: u64,
}

impl ObjectHeaderContinuation {
    /// Parses the body of an Object Header Continuation message.
    ///
    /// Checks `offset_size` and then `length_size` before reading the body.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a
    /// width is not 2, 4, or 8, and otherwise the errors
    /// [`parse_with_widths`](Self::parse_with_widths) returns.
    pub fn parse(body: &[u8], offset_size: u8, length_size: u8) -> Result<Self, FormatError> {
        Self::parse_with_widths(body, FormatWidths::from_sizes(offset_size, length_size)?)
    }

    /// Reads the continuation address and length using the widths from the superblock.
    ///
    /// Reads both fields as one region before interpreting the address. Ignores bytes after
    /// the length field.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `body` is shorter than the sum of the offset
    /// and length widths. Reports that sum as `expected` and `body.len()` as `available`.
    /// Returns [`FormatError::UndefinedContinuationAddress`] if both fields are present and
    /// the address is undefined at the offset width.
    ///
    /// # Examples
    ///
    /// Reads a two-byte address followed by a four-byte length.
    ///
    /// ```
    /// use hdf5_pure_format::FormatError;
    /// use hdf5_pure_format::__private::{FormatWidths, ObjectHeaderContinuation};
    ///
    /// let widths = FormatWidths::from_sizes(2, 4)?;
    /// let body = [0x34, 0x12, 0x00, 0x1a, 1, 0, 0xa5];
    /// let continuation = ObjectHeaderContinuation::parse_with_widths(&body, widths)?;
    /// assert_eq!(continuation.address().get(), 0x1234);
    /// assert_eq!(continuation.length(), 0x0001_1a00);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn parse_with_widths(body: &[u8], widths: FormatWidths) -> Result<Self, FormatError> {
        Self::decode(body, widths)
    }

    /// Returns the address of the continuation block, relative to the base address.
    pub fn address(&self) -> StoredAddress {
        self.address
    }

    /// Returns the length of the continuation block in bytes.
    pub fn length(&self) -> u64 {
        self.length
    }
}

impl<'input> Decode<'input> for ObjectHeaderContinuation {
    type Context = FormatWidths;
    type Error = FormatError;

    fn decode(
        input: &'input [u8],
        FormatWidths { offsets, lengths }: FormatWidths,
    ) -> Result<Self, Self::Error> {
        let record_len = usize::from(offsets.get()) + usize::from(lengths.get());
        let record = bytes::ByteCursor::new(input).take(record_len)?;
        let mut fields = bytes::ByteCursor::new(record);
        let address = fields.address(offsets)?;
        let length = fields.length(lengths)?;
        if address.is_undefined(offsets) {
            return Err(FormatError::UndefinedContinuationAddress);
        }
        Ok(Self { address, length })
    }
}

/// Returns the range of the message records in the continuation block `block`, between its `OCHK`
/// signature and its checksum.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `block` is shorter than the signature and the
/// checksum, and [`FormatError::InvalidObjectHeaderSignature`] if it does not begin with `OCHK`.
pub fn continuation_block_messages(block: &[u8]) -> Result<Range<usize>, FormatError> {
    let end = block
        .len()
        .checked_sub(CHECKSUM_LEN)
        .filter(|&end| end >= OCHK_SIGNATURE.len())
        .ok_or(FormatError::UnexpectedEof {
            expected: OCHK_SIGNATURE.len() + CHECKSUM_LEN,
            available: block.len(),
        })?;
    if block[..OCHK_SIGNATURE.len()] != OCHK_SIGNATURE {
        return Err(FormatError::InvalidObjectHeaderSignature);
    }
    Ok(OCHK_SIGNATURE.len()..end)
}

/// OCHK signature for v2 continuation chunks.
pub(super) const OCHK_SIGNATURE: [u8; 4] = *b"OCHK";

#[cfg(test)]
#[path = "tests/continuation.rs"]
mod tests;
