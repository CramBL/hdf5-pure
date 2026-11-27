use alloc::vec;
use alloc::vec::Vec;

use super::FixedArrayPlacement;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::chunk_record;
use crate::chunk_record::ArrayClient;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::IndexSlots;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fixed_array::data_block::DATA_BLOCK_SIGNATURE;
use crate::fixed_array::emit;
use crate::fixed_array::framing::VERSION;
use crate::fixed_array::header::HEADER_SIGNATURE;

impl FixedArrayPlacement {
    /// Serializes the header and data block separately, with their reciprocal stored references.
    ///
    /// The returned buffers follow header, then data block order. Unallocated slots contain the
    /// undefined address. All pages are initialized, and the last page holds only its remaining
    /// entries. The writer computes all checksums regardless of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FixedArrayEntryCount`] if the supplied slot count differs from
    /// the planned count, [`FormatError::DefinedAddressIsUndefined`] if a chunk address is
    /// undefined, [`FormatError::AddressTooLarge`] if a chunk address exceeds the address width,
    /// or [`FormatError::Internal`] if a filtered chunk's stored size exceeds its field width.
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if an output length does not fit
    /// [`usize`], or [`FormatError::AllocationFailed`] if an output buffer reservation fails.
    pub fn serialize_blocks(
        &self,
        slots: &IndexSlots<'_>,
    ) -> Result<[(StoredAddress, Vec<u8>); 2], FormatError> {
        let [header, data_block] = self.ranges();
        let geometry = self.plan().geometry();
        let actual = slots.len().to_u64();
        if actual != geometry.entries() {
            return Err(FormatError::FixedArrayEntryCount {
                expected: geometry.entries(),
                actual,
            });
        }
        Ok([
            (header.start(), self.serialize_header()?),
            (data_block.start(), self.serialize_data_block(slots)?),
        ])
    }

    /// Serializes the version 0 header with the placed data-block address and its checksum.
    ///
    /// The caller obtains both blocks from [`Self::serialize_blocks`] or emits an empty data
    /// block with [`Self::emit_empty_data_block`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the header length exceeds
    /// [`usize::MAX`], or [`FormatError::AllocationFailed`] if its buffer reservation fails.
    /// Placement and planning check the encoded address and entry count.
    pub fn serialize_header(&self) -> Result<Vec<u8>, FormatError> {
        let [header, data_block] = self.ranges();
        let geometry = self.plan().geometry();
        let encoding = geometry.encoding();
        let mut bytes = emit::output_buffer(header.length())?;
        bytes.extend_from_slice(&HEADER_SIGNATURE);
        bytes.push(VERSION);
        bytes.push(encoding.client().id());
        bytes.push(encoding.stride().get());
        bytes.push(geometry.page_bits());
        bytes::try_write_length(&mut bytes, geometry.entries(), self.plan().widths().lengths)?;
        bytes::try_write_decoded_address(
            &mut bytes,
            DecodedStoredAddress::Defined(data_block.start()),
            encoding.offsets(),
        )?;
        let checksum = checksum::jenkins_lookup3(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(bytes.len().to_u64(), header.length());
        Ok(bytes)
    }

    fn serialize_data_block(&self, slots: &IndexSlots<'_>) -> Result<Vec<u8>, FormatError> {
        let [header, data_block] = self.ranges();
        let geometry = self.plan().geometry();
        let encoding = geometry.encoding();
        let mut bytes = emit::output_buffer(data_block.length())?;
        bytes.extend_from_slice(&DATA_BLOCK_SIGNATURE);
        bytes.push(VERSION);
        bytes.push(encoding.client().id());
        bytes::try_write_decoded_address(
            &mut bytes,
            DecodedStoredAddress::Defined(header.start()),
            encoding.offsets(),
        )?;
        match geometry.paging() {
            None => {
                for slot in 0..slots.len() {
                    self.write_element(&mut bytes, slots.at(slot))?;
                }
                let checksum = checksum::jenkins_lookup3(&bytes);
                bytes.extend_from_slice(&checksum.to_le_bytes());
            }
            Some(paging) => {
                let pages = paging.pages().to_usize()?;
                let mut bitmap = vec![0; paging.pages().div_ceil(8).to_usize()?];
                for page in 0..pages {
                    // Page 0 occupies the most significant bit. H5VM_bit_set uses this order
                    // in H5VMprivate.h, HDF5 2.2.0.
                    bitmap[page / 8] |= 1 << (7 - page % 8);
                }
                bytes.extend_from_slice(&bitmap);
                let checksum = checksum::jenkins_lookup3(&bytes);
                bytes.extend_from_slice(&checksum.to_le_bytes());
                let page_size = paging.page_entries().to_usize()?;
                for page in 0..pages {
                    let start = page * page_size;
                    let end = start.saturating_add(page_size).min(slots.len());
                    let mut page_bytes =
                        Vec::with_capacity((end - start) * usize::from(encoding.stride().get()));
                    for slot in start..end {
                        self.write_element(&mut page_bytes, slots.at(slot))?;
                    }
                    let checksum = checksum::jenkins_lookup3(&page_bytes);
                    page_bytes.extend_from_slice(&checksum.to_le_bytes());
                    bytes.extend_from_slice(&page_bytes);
                }
            }
        }
        debug_assert_eq!(bytes.len().to_u64(), data_block.length());
        Ok(bytes)
    }

    fn write_element(
        &self,
        bytes: &mut Vec<u8>,
        chunk: Option<&ChunkRecord>,
    ) -> Result<(), FormatError> {
        let encoding = self.plan().geometry().encoding();
        let (filtered, chunk_size_bytes) = match encoding.client() {
            ArrayClient::Chunks => (false, 0),
            ArrayClient::FilteredChunks { chunk_size_len } => (true, usize::from(chunk_size_len)),
        };
        match chunk {
            Some(chunk) => chunk_record::write_chunk_element(
                bytes,
                chunk,
                encoding.offsets(),
                filtered,
                chunk_size_bytes,
            ),
            None => {
                chunk_record::write_undefined_element(
                    bytes,
                    encoding.offsets(),
                    filtered,
                    chunk_size_bytes,
                );
                Ok(())
            }
        }
    }
}
