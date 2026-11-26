use alloc::format;
use alloc::vec::Vec;

use super::frame::{FractalHeapIdKind, FractalHeapIdType};
use crate::bytes;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;

impl FractalHeapHeader {
    /// Returns the heap ID of the managed object of `length` bytes at heap offset `heap_offset`.
    ///
    /// After its first byte, the ID stores the offset and length in separate little-endian byte
    /// fields. The offset width is [`max_heap_size`](Self::max_heap_size) rounded up to whole
    /// bytes. The length width is derived from the direct-block offset width and
    /// [`max_managed_object_size`](Self::max_managed_object_size), as `H5HF_hdr_finish_init_phase1`
    /// in `H5HFhdr.c` computes it (HDF5 1.10.11). Trailing ID padding is zeroed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the header cannot describe the
    /// managed-ID fields. Returns [`FormatError::Internal`] if `heap_offset` requires more bits
    /// than [`max_heap_size`](Self::max_heap_size) or `length` does not fit its encoded field.
    pub fn encode_managed_id(&self, heap_offset: u64, length: u64) -> Result<Vec<u8>, FormatError> {
        ManagedHeapIdLayout::new(self)
            .ok_or(FormatError::InvalidFractalHeapGeometry)?
            .encode(heap_offset, length)
    }
}

/// The encoded byte widths and offset bit count of a managed heap ID.
///
/// The length field follows the whole-byte offset field. libhdf5 computes these widths in
/// `H5HF_hdr_finish_init_phase1` in `H5HFhdr.c` (HDF5 1.10.11).
#[derive(Clone, Copy, Debug)]
pub(super) struct ManagedHeapIdLayout {
    encoded_len: usize,
    offset_bits: u32,
    offset_bytes: usize,
    length_bytes: usize,
}

impl ManagedHeapIdLayout {
    pub(super) fn new(header: &FractalHeapHeader) -> Option<Self> {
        if !(1..=64).contains(&header.max_heap_size)
            || !header.max_direct_block_size.is_power_of_two()
            || header.max_managed_object_size == 0
        {
            return None;
        }
        let offset_bits = header.max_heap_size;
        let offset_bytes = usize::from(offset_bits).div_ceil(8);
        // H5HF_hdr_finish_init_phase1 in H5HFhdr.c, HDF5 1.10.11, bounds the
        // length field by both the direct-block offset width and managed-object limit.
        let direct_bits = header.max_direct_block_size.trailing_zeros();
        let managed_bits = u32::BITS - header.max_managed_object_size.leading_zeros();
        let length_bytes =
            usize::try_from(direct_bits.div_ceil(8).min(managed_bits.div_ceil(8))).ok()?;
        let encoded_len = usize::from(header.heap_id_length);
        if length_bytes == 0
            || encoded_len < offset_bytes.checked_add(length_bytes)?.checked_add(1)?
        {
            return None;
        }
        Some(Self {
            encoded_len,
            offset_bits: u32::from(offset_bits),
            offset_bytes,
            length_bytes,
        })
    }

    fn encode(&self, heap_offset: u64, length: u64) -> Result<Vec<u8>, FormatError> {
        if !fits_bits(heap_offset, self.offset_bits) {
            return Err(FormatError::Internal(format!(
                "heap offset {heap_offset} does not fit the heap's {}-bit offsets",
                self.offset_bits
            )));
        }
        let length_bytes = self.length_bytes;
        if length_bytes < 8
            && !fits_bits(
                length,
                u32::try_from(length_bytes * 8)
                    .map_err(|_conversion_error| FormatError::InvalidFractalHeapGeometry)?,
            )
        {
            return Err(FormatError::Internal(format!(
                "a managed object length does not fit a {}-byte heap ID",
                self.encoded_len
            )));
        }
        let mut bytes = Vec::with_capacity(self.encoded_len);
        bytes.push(FractalHeapIdType::Managed.first_byte());
        bytes.extend(
            heap_offset
                .to_le_bytes()
                .into_iter()
                .take(self.offset_bytes),
        );
        bytes.extend(length.to_le_bytes().into_iter().take(length_bytes));
        bytes.resize(self.encoded_len, 0);
        Ok(bytes)
    }

    pub(super) fn parse(&self, payload: &[u8]) -> Result<FractalHeapIdKind<'static>, FormatError> {
        let mut cursor = bytes::ByteCursor::new(payload);
        let heap_offset = cursor.uint(self.offset_bytes)?;
        if !fits_bits(heap_offset, self.offset_bits) {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(FractalHeapIdKind::Managed {
            heap_offset,
            object_length: cursor.uint(self.length_bytes)?,
        })
    }
}

fn fits_bits(value: u64, bits: u32) -> bool {
    value.checked_shr(bits).is_none_or(|high| high == 0)
}

#[cfg(test)]
#[path = "tests/managed.rs"]
mod tests;
