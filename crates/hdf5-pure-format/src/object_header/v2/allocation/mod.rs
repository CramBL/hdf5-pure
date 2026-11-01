//! Chunk planning within a supplied version 2 object-header allocation.

use alloc::format;
use alloc::vec::Vec;
use core::ops::Range;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::width::{self, FormatWidths, UintWidth};

use super::worklist;
use super::{CHECKSUM_LEN, OCHK_SIGNATURE, ObjectHeaderPrefix};

/// Holds checked chunk geometry for a version 2 header in one contiguous allocation.
///
/// The plan borrows the source message records and retains their ranges and the output chunk
/// geometry. Non-NIL records remain in chunk 0. Additional chunks contain padding and continuation
/// records. [`Self::encode`] materializes the complete allocation.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_format::__private::{
///     FormatWidths, ObjectHeaderAllocation, ObjectHeaderPrefix,
/// };
///
/// let address = StoredAddress::new(128);
/// let plan = ObjectHeaderAllocation::new(
///     ObjectHeaderPrefix::PLAIN,
///     FormatWidths::from_sizes(8, 8)?,
///     address,
///     512,
///     &[],
/// )?;
/// assert_eq!(plan.chunks().collect::<Vec<_>>(), [(address, 512)]);
/// assert_eq!(plan.encode().len(), 512);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Debug)]
pub struct ObjectHeaderAllocation<'a> {
    prefix: ObjectHeaderPrefix,
    widths: FormatWidths,
    messages: &'a [u8],
    records: Vec<Range<usize>>,
    root_width: UintWidth,
    chunks: Vec<Chunk>,
    allocation_len: usize,
}

impl<'a> ObjectHeaderAllocation<'a> {
    /// Plans chunks that cover exactly `allocation_len` bytes beginning at `address`.
    ///
    /// The caller establishes ownership and availability of the allocation and passes a stored,
    /// base-relative address. `messages` contains records in `prefix.layout`, with source
    /// continuations already flattened. Discards source NIL records and trailing bytes shorter
    /// than a record prefix. Retains the other records, including their flags and creation indices.
    ///
    /// Empty input may require a wider Size of Chunk #0 field with a zero-byte body. With
    /// [`ObjectHeaderPrefix::PLAIN`], the three smallest supported allocation lengths are 11, 12,
    /// and 14 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the allocation cannot be materialized
    /// on this host, [`FormatError::OffsetOverflow`] if its exclusive end exceeds [`u64::MAX`],
    /// [`FormatError::AddressTooLarge`] or [`FormatError::DefinedAddressIsUndefined`] if either
    /// endpoint is not a defined address at the output width, and [`FormatError::UnexpectedEof`]
    /// if a message body extends beyond `messages`.
    ///
    /// Returns [`FormatError::Internal`] if `messages` contains a continuation record or the
    /// allocation cannot represent the header, and [`FormatError::NestingDepthExceeded`] if more
    /// than 256 chunks, including chunk 0, are required.
    pub fn new(
        prefix: ObjectHeaderPrefix,
        widths: FormatWidths,
        address: StoredAddress,
        allocation_len: u64,
        messages: &'a [u8],
    ) -> Result<Self, FormatError> {
        if allocation_len > isize::MAX as u64 {
            return Err(FormatError::ValueTooLargeForPlatform {
                value: allocation_len,
                target: "object header allocation",
            });
        }
        let end = address.checked_add(allocation_len)?;
        Self::require_address(widths, address)?;
        Self::require_address(widths, end)?;
        let mut records = Vec::new();
        let mut records_len = 0;
        let mut start = 0;
        while let Some(record) = prefix.layout.next_message(messages, start)? {
            let end = record.body_range.end;
            if record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                return Err(FormatError::Internal(
                    "an object-header allocation requires flattened message records".into(),
                ));
            }
            if record.msg_type != MessageType::NIL {
                records.push(start..end);
                records_len += end - start;
            }
            start = end;
        }
        let allocation_len = allocation_len.to_usize()?;
        for root_width in [
            UintWidth::One,
            UintWidth::Two,
            UintWidth::Four,
            UintWidth::Eight,
        ] {
            let fixed = super::OHDR_SIGNATURE.len()
                + 2
                + prefix.optional_len()
                + usize::from(root_width.get())
                + CHECKSUM_LEN;
            let Some(padding) = allocation_len.checked_sub(fixed + records_len) else {
                continue;
            };
            let first = Chunk::new(prefix, widths, address, fixed + records_len, padding, true)?;
            let root_body = first.len - fixed;
            if (root_body > 0 && root_body < prefix.layout.prefix_len())
                || UintWidth::smallest_for_len(root_body).get() > root_width.get()
            {
                continue;
            }
            let mut chunks = Vec::new();
            let mut chunk = first;
            while let Some(remaining) = chunk.remaining {
                if chunks.len() + 1 >= worklist::MAX_CONTINUATIONS {
                    return Err(FormatError::NestingDepthExceeded);
                }
                let next_address = chunk.address.checked_add(chunk.len.to_u64())?;
                let next = Chunk::new(
                    prefix,
                    widths,
                    next_address,
                    OCHK_SIGNATURE.len() + CHECKSUM_LEN,
                    remaining - OCHK_SIGNATURE.len() - CHECKSUM_LEN,
                    false,
                )?;
                chunk.next = Some((next.address, next.len));
                chunks.push(chunk);
                chunk = next;
            }
            chunks.push(chunk);
            return Ok(Self {
                prefix,
                widths,
                messages,
                records,
                root_width,
                chunks,
                allocation_len,
            });
        }
        Err(FormatError::Internal(format!(
            "a {allocation_len}-byte allocation cannot represent an object header with {records_len} message bytes"
        )))
    }

    /// Returns the complete allocation, with a checksum for every chunk.
    ///
    /// Each chunk contains at most one NIL record. Nonempty padding shorter than a record prefix
    /// is a trailing gap. Checksums are emitted regardless of the reader's checksum feature.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.allocation_len);
        for (index, chunk) in self.chunks.iter().enumerate() {
            let start = out.len();
            if index == 0 {
                let fixed = super::OHDR_SIGNATURE.len()
                    + 2
                    + self.prefix.optional_len()
                    + usize::from(self.root_width.get())
                    + CHECKSUM_LEN;
                self.prefix
                    .write_prefix(&mut out, self.root_width, chunk.len - fixed);
                for record in &self.records {
                    out.extend_from_slice(&self.messages[record.clone()]);
                }
            } else {
                out.extend_from_slice(&OCHK_SIGNATURE);
            }
            self.write_padding(&mut out, chunk.padding);
            if let Some((address, len)) = chunk.next {
                self.prefix.layout.write_record_prefix(
                    &mut out,
                    MessageType::OBJECT_HEADER_CONTINUATION
                        .to_u16()
                        .to_le_bytes()[0],
                    MessageFlags::NONE,
                    0,
                    u16::from(self.widths.offsets.get()) + u16::from(self.widths.lengths.get()),
                );
                bytes::write_offset(&mut out, address.get(), self.widths.offsets);
                bytes::write_length(&mut out, len.to_u64(), self.widths.lengths);
            }
            let checksum = checksum::jenkins_lookup3(&out[start..]);
            out.extend_from_slice(&checksum.to_le_bytes());
            assert_eq!(
                out.len() - start,
                chunk.len,
                "a planned header chunk has its exact owned length"
            );
        }
        assert_eq!(
            out.len(),
            self.allocation_len,
            "header chunks cover the complete allocation"
        );
        out
    }

    /// Returns chunk addresses and complete lengths in ascending address order.
    ///
    /// Addresses are stored, base-relative addresses. The chunks are contiguous and cover the
    /// supplied allocation, including signatures, record prefixes, and checksums.
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.chunks
            .iter()
            .map(|chunk| (chunk.address, chunk.len.to_u64()))
    }

    fn require_address(widths: FormatWidths, address: StoredAddress) -> Result<(), FormatError> {
        if address.get() > width::offset_width_max(widths.offsets) {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: widths.offsets.get(),
            });
        }
        if address.is_undefined(widths.offsets) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: widths.offsets.get(),
            });
        }
        Ok(())
    }

    // H5O__chunk_deserialize in H5Ocache.c, HDF5 2.2.0, rejects gaps after NIL
    // records and coalesces adjacent NIL records on read-write opening within each chunk.
    fn write_padding(&self, out: &mut Vec<u8>, len: usize) {
        let prefix = self.prefix.layout.prefix_len();
        if len >= prefix {
            let body = len - prefix;
            // Chunk::new bounds every NIL body to its two-byte field.
            let body = u16::try_from(body).expect("a planned NIL body fits its size field");
            self.prefix.layout.write_record_prefix(
                out,
                MessageType::NIL.to_u16().to_le_bytes()[0],
                MessageFlags::NONE,
                0,
                body,
            );
            out.resize(out.len() + usize::from(body), 0);
        } else {
            out.resize(out.len() + len, 0);
        }
    }
}

/// Holds one chunk's geometry while dividing the allocation.
#[derive(Debug)]
struct Chunk {
    address: StoredAddress,
    len: usize,
    padding: usize,
    /// The complete bytes reserved for subsequent chunks.
    remaining: Option<usize>,
    /// The address and complete length of the next chunk, once planned.
    next: Option<(StoredAddress, usize)>,
}

impl Chunk {
    fn new(
        prefix: ObjectHeaderPrefix,
        widths: FormatWidths,
        address: StoredAddress,
        fixed: usize,
        available: usize,
        root: bool,
    ) -> Result<Self, FormatError> {
        let maximum = prefix.layout.prefix_len() + OBJECT_HEADER_MESSAGE_MAX;
        let limit = if root {
            usize::MAX
        } else {
            width::length_width_max(widths.lengths)
                .min(usize::MAX as u64)
                .to_usize()?
        };
        if available <= maximum && fixed + available <= limit {
            return Ok(Self {
                address,
                len: fixed + available,
                padding: available,
                remaining: None,
                next: None,
            });
        }
        let continuation = prefix.layout.prefix_len()
            + usize::from(widths.offsets.get())
            + usize::from(widths.lengths.get());
        let minimum = OCHK_SIGNATURE.len() + prefix.layout.prefix_len() + CHECKSUM_LEN;
        let padding = maximum
            .min(limit.saturating_sub(fixed + continuation))
            .min(available - continuation - minimum);
        let len = fixed + padding + continuation;
        let remaining = available - padding - continuation;
        Ok(Self {
            address,
            len,
            padding,
            remaining: Some(remaining),
            next: None,
        })
    }
}

#[cfg(test)]
mod tests;
