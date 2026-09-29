//! HDF5 Fixed Array index parsing for chunked datasets (v4 index type 3).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use hdf5_pure_format::__private::MetadataSource;

use crate::address::StoredAddress;
use crate::bytes::{read_length, read_offset, read_optional_offset};
use crate::chunked_write::ChunkRecord;
use crate::convert::Narrow;
use crate::error::FormatError;

/// Parsed Fixed Array header (FAHD).
#[derive(Debug, Clone)]
pub struct FixedArrayHeader {
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size of each array element in bytes.
    pub element_size: u8,
    /// Log2 of max number of elements in a data block page.
    pub max_nelmts_bits: u8,
    /// Total number of elements (chunks) in the array.
    pub num_elements: u64,
    /// Address of the data block.
    pub data_block_address: StoredAddress,
}

impl FixedArrayHeader {
    /// Parse a Fixed Array header from file data at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        // FAHD signature(4) + version(1) + client_id(1) + element_size(1) +
        // max_nelmts_bits(1) + num_elements(length_size) + data_block_addr(offset_size) + checksum(4)
        let min_size = 4 + 1 + 1 + 1 + 1 + length_size as usize + offset_size as usize + 4;
        if min_size > file_data.len() || offset > file_data.len() - min_size {
            return Err(FormatError::UnexpectedEof {
                expected: offset.saturating_add(min_size),
                available: file_data.len(),
            });
        }

        let d = &file_data[offset..];
        if &d[0..4] != b"FAHD" {
            return Err(FormatError::ChunkedReadError(
                "invalid Fixed Array header signature".into(),
            ));
        }

        let version = d[4];
        if version != 0 {
            return Err(FormatError::ChunkedReadError(format!(
                "unsupported Fixed Array header version: {version}"
            )));
        }

        let client_id = d[5];
        let element_size = d[6];
        let max_nelmts_bits = d[7];

        let mut pos = 8;
        let num_elements = read_length(d, pos, length_size)?;
        pos += length_size as usize;
        let data_block_address = StoredAddress::new(read_offset(d, pos, offset_size)?);

        crate::checksum::verify_trailing(&d[..min_size])?;

        Ok(FixedArrayHeader {
            client_id,
            element_size,
            max_nelmts_bits,
            num_elements,
            data_block_address,
        })
    }

    /// Parse a Fixed Array header from a [`MetadataSource`] (bounded window).
    pub fn parse_from_source(
        source: &(impl MetadataSource + ?Sized),
        address: StoredAddress,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        let min_size = 4 + 1 + 1 + 1 + 1 + length_size as usize + offset_size as usize + 4;
        let buf = source.read_metadata_at(address.get(), min_size)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// On-disk byte spans `(addr, len)` of a Fixed Array chunk index's own
/// structure — its header (`FAHD`) and its single data block (`FADB`, paged or
/// not) — for reclaiming a deleted chunked dataset. The chunk *data* blocks the
/// array points at are enumerated separately (via [`read_fixed_array_chunks`]),
/// so they are not included here.
///
/// `fa_base` is the Fixed Array header address taken from the data-layout
/// message. The returned spans are exact (the writer allocates each block at the
/// size computed here); the caller validates them against the file bounds.
pub(crate) fn fixed_array_index_spans(
    source: &(impl MetadataSource + ?Sized),
    fa_base: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let header = FixedArrayHeader::parse_from_source(source, fa_base, offset_size, length_size)?;
    let os = offset_size as usize;

    // FAHD: sig(4)+ver(1)+client(1)+elem_size(1)+max_bits(1)+num_elements(ls)+dblk_addr(os)+checksum(4).
    let fahd_size = (4 + 1 + 1 + 1 + 1 + length_size as usize + os + 4) as u64;
    let mut spans = vec![(fa_base.get(), fahd_size)];

    // An empty array has no data block (the address is the undefined sentinel);
    // there is nothing more to reclaim.
    if header.data_block_address.is_undefined(offset_size) {
        return Ok(spans);
    }

    // FADB element stride: just the chunk address when unfiltered, the full
    // filtered element record otherwise.
    let elem_size = if header.client_id == 0 {
        os
    } else {
        header.element_size as usize
    };
    if header.max_nelmts_bits >= 64 {
        return Err(FormatError::ChunkedReadError(
            "Fixed Array page exponent out of range".into(),
        ));
    }
    let page_size = 1usize << header.max_nelmts_bits;
    let num_elements = header.num_elements.to_usize()?;
    let db_prefix = 4 + 1 + 1 + os; // FADB sig(4)+ver(1)+client(1)+header_addr(os)

    let fadb_size: u64 = if num_elements <= page_size {
        // Non-paged: prefix + element records + checksum.
        (db_prefix
            + num_elements
                .checked_mul(elem_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: num_elements as u64,
                    length: elem_size as u64,
                })?
            + 4) as u64
    } else {
        // Paged: prefix + page-init bitmap + prefix checksum, then the element
        // records (each written exactly once) split into `npages` pages, each
        // followed by its own 4-byte checksum. Only the last page is partial;
        // the writer does not pad it to full stride (see `build_fixed_array_at`),
        // so the data block is `num_elements * elem_size + npages * 4` element
        // and checksum bytes after the prefix.
        let npages = num_elements.div_ceil(page_size);
        let bitmap_size = npages.div_ceil(8);
        let elements_bytes =
            num_elements
                .checked_mul(elem_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: num_elements as u64,
                    length: elem_size as u64,
                })?;
        (db_prefix + bitmap_size + 4 + elements_bytes + npages * 4) as u64
    };

    spans.push((header.data_block_address.get(), fadb_size));
    Ok(spans)
}

/// Decode one Fixed/Extensible-Array element record from `block` at `elem_pos`.
///
/// Returns `None` for the all-`0xFF` sentinel (an unallocated chunk). `block`
/// may be the whole-file buffer (buffered path, `elem_pos` absolute) or a
/// data-block region read from a source (streaming path, `elem_pos` relative).
fn parse_fa_element(
    block: &[u8],
    elem_pos: usize,
    client_id: u8,
    chunk_byte_size: u64,
    elem_size: usize,
    chunk_size_bytes: usize,
    offset_size: u8,
) -> Result<Option<ChunkRecord>, FormatError> {
    if elem_size > block.len() || elem_pos > block.len() - elem_size {
        return Err(FormatError::UnexpectedEof {
            expected: elem_pos.saturating_add(elem_size),
            available: block.len(),
        });
    }
    let Some(address) = read_optional_offset(block, elem_pos, offset_size)?.map(StoredAddress::new)
    else {
        return Ok(None);
    };
    if client_id == 0 {
        Ok(Some(ChunkRecord {
            address,
            stored_size: chunk_byte_size,
            filter_mask: 0,
        }))
    } else {
        let os = offset_size as usize;
        let chunk_size = read_variable_length(&block[elem_pos + os..], chunk_size_bytes)?;
        let fm_off = elem_pos + os + chunk_size_bytes;
        let filter_mask = u32::from_le_bytes([
            block[fm_off],
            block[fm_off + 1],
            block[fm_off + 2],
            block[fm_off + 3],
        ]);
        Ok(Some(ChunkRecord {
            address,
            stored_size: chunk_size,
            filter_mask,
        }))
    }
}

/// Read chunk records from a Fixed Array data block.
///
/// Handles both the non-paged layout (elements stored directly after the data
/// block prefix) and the paged layout used when the chunk count exceeds the
/// page size (`2^max_nelmts_bits`). In the paged layout the data block prefix
/// is followed by a page-initialization bitmap and a checksum, after which the
/// elements live in fixed-stride pages, each terminated by its own checksum.
pub fn read_fixed_array_chunks(
    file_data: &[u8],
    header: &FixedArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    mut visit: impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let db_offset = header.data_block_address.get().to_usize()?;
    let os = offset_size as usize;

    // Parse data block prefix: FADB(4) + version(1) + client_id(1) + header_address(offset_size)
    let db_header_size = 4 + 1 + 1 + os;
    if db_header_size > file_data.len() || db_offset > file_data.len() - db_header_size {
        return Err(FormatError::UnexpectedEof {
            expected: db_offset.saturating_add(db_header_size),
            available: file_data.len(),
        });
    }

    if &file_data[db_offset..db_offset + 4] != b"FADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Fixed Array data block signature".into(),
        ));
    }

    // Per-element encoding width. Non-filtered: just the chunk address.
    // Filtered: address + variable-width chunk size + 4-byte filter mask.
    let chunk_size_bytes = if header.client_id == 0 {
        0
    } else {
        (header.element_size as usize)
            .checked_sub(os + 4)
            .ok_or_else(|| {
                FormatError::ChunkedReadError("Fixed Array element size too small".into())
            })?
    };
    let elem_size = if header.client_id == 0 {
        os
    } else {
        header.element_size as usize
    };

    let num_elements = header.num_elements.to_usize()?;
    let page_size = (1u64 << header.max_nelmts_bits).to_usize()?;
    let is_paged = num_elements > page_size;

    if !is_paged {
        // Elements stored directly after the data block prefix, then one
        // checksum over the prefix and every slot -- including the unallocated
        // ones, which the writer fills with the undefined address.
        let db_len = num_elements
            .checked_mul(elem_size)
            .and_then(|elems| elems.checked_add(db_header_size + 4))
            .ok_or(FormatError::OffsetOverflow {
                offset: num_elements as u64,
                length: elem_size as u64,
            })?;
        if db_len > file_data.len() || db_offset > file_data.len() - db_len {
            return Err(FormatError::UnexpectedEof {
                expected: db_offset.saturating_add(db_len),
                available: file_data.len(),
            });
        }
        crate::checksum::verify_trailing(&file_data[db_offset..db_offset + db_len])?;

        let mut pos = db_offset + db_header_size;
        for index in 0..num_elements {
            if let Some(record) = parse_fa_element(
                file_data,
                pos,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
            )? {
                visit(index as u64, record)?;
            }
            pos += elem_size;
        }
        return Ok(());
    }

    // Paged: prefix is followed by a page-init bitmap (one bit per page,
    // most-significant-bit first) and a 4-byte checksum. Pages then follow at a
    // fixed stride of `page_size` elements plus a 4-byte checksum each; the
    // whole block is allocated contiguously, so the last (partial) page still
    // begins at its full-stride offset.
    let npages = num_elements.div_ceil(page_size);
    let bitmap_size = npages.div_ceil(8);
    let bitmap_pos = db_offset + db_header_size;
    // `bitmap_size` derives from a crafted element count, so the end offset can
    // overflow `usize`; bound it with checked arithmetic (issue #140).
    let fits = bitmap_pos
        .checked_add(bitmap_size)
        .and_then(|x| x.checked_add(4))
        .is_some_and(|end| end <= file_data.len());
    if !fits {
        return Err(FormatError::UnexpectedEof {
            expected: bitmap_pos.saturating_add(bitmap_size).saturating_add(4),
            available: file_data.len(),
        });
    }
    let bitmap = &file_data[bitmap_pos..bitmap_pos + bitmap_size];
    // A paged block checksums its prefix and bitmap together, then each page.
    crate::checksum::verify_trailing(&file_data[db_offset..bitmap_pos + bitmap_size + 4])?;
    let pages_start = bitmap_pos + bitmap_size + 4;
    let page_stride = page_size
        .checked_mul(elem_size)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(FormatError::OffsetOverflow {
            offset: page_size as u64,
            length: elem_size as u64,
        })?;

    for page in 0..npages {
        let nelem_in_page = core::cmp::min(page_size, num_elements - page * page_size);
        // A cleared bit means the page was never initialized: every chunk it
        // would hold is unallocated, so skip it without reading.
        let initialized = (bitmap[page / 8] >> (7 - (page % 8))) & 1 == 1;
        if !initialized {
            continue;
        }
        let page_offset = page
            .checked_mul(page_stride)
            .ok_or(FormatError::OffsetOverflow {
                offset: page as u64,
                length: page_stride as u64,
            })?;
        let page_start = pages_start + page_offset;
        // The page checksum follows its elements. Only the last page is
        // partial: the writer does not pad it, so its checksum covers the
        // elements it actually holds.
        let page_end = page_start
            .checked_add(nelem_in_page * elem_size + 4)
            .filter(|&end| end <= file_data.len())
            .ok_or(FormatError::UnexpectedEof {
                expected: page_start.saturating_add(nelem_in_page * elem_size + 4),
                available: file_data.len(),
            })?;
        crate::checksum::verify_trailing(&file_data[page_start..page_end])?;
        for j in 0..nelem_in_page {
            let index = page * page_size + j;
            let elem_pos = page_start + j * elem_size;
            if let Some(record) = parse_fa_element(
                file_data,
                elem_pos,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
            )? {
                visit(index as u64, record)?;
            }
        }
    }

    Ok(())
}

/// Read chunk records from a Fixed Array via a [`MetadataSource`].
///
/// Streaming counterpart of [`read_fixed_array_chunks`]: it reads the data-block
/// prefix, then the element array (non-paged) or each initialized page
/// (paged) as bounded windows via `read_at`, decoding the same records.
pub fn read_fixed_array_chunks_from_source(
    source: &(impl MetadataSource + ?Sized),
    header: &FixedArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    mut visit: impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let db_address = header.data_block_address.get();
    let os = offset_size as usize;
    let db_header_size = 4 + 1 + 1 + os;

    // Both branches below read from the start of the data block, so its prefix
    // -- FADB(4) + version(1) + client_id(1) + header_address -- arrives with
    // the bytes they verify the checksum over, and the signature is checked
    // there rather than in a read of its own.
    let signature = |block: &[u8]| -> Result<(), FormatError> {
        if &block[0..4] != b"FADB" {
            return Err(FormatError::ChunkedReadError(
                "invalid Fixed Array data block signature".into(),
            ));
        }
        Ok(())
    };

    let chunk_size_bytes = if header.client_id == 0 {
        0
    } else {
        (header.element_size as usize)
            .checked_sub(os + 4)
            .ok_or_else(|| {
                FormatError::ChunkedReadError("Fixed Array element size too small".into())
            })?
    };
    let elem_size = if header.client_id == 0 {
        os
    } else {
        header.element_size as usize
    };

    let num_elements = header.num_elements.to_usize()?;
    let page_size = (1u64 << header.max_nelmts_bits).to_usize()?;
    let is_paged = num_elements > page_size;

    if !is_paged {
        // The whole block in one window: prefix, every element slot, and the
        // single checksum over them.
        let db_len = num_elements
            .checked_mul(elem_size)
            .and_then(|elems| elems.checked_add(db_header_size + 4))
            .ok_or(FormatError::OffsetOverflow {
                offset: num_elements as u64,
                length: elem_size as u64,
            })?;
        let region = source.read_metadata_at(db_address, db_len)?;
        signature(&region)?;
        crate::checksum::verify_trailing(&region)?;
        for index in 0..num_elements {
            if let Some(record) = parse_fa_element(
                &region,
                db_header_size + index * elem_size,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
            )? {
                visit(index as u64, record)?;
            }
        }
        return Ok(());
    }

    // Paged: read the page-init bitmap, then each initialized page's elements.
    let npages = num_elements.div_ceil(page_size);
    let bitmap_size = npages.div_ceil(8);
    let bitmap_addr = db_address + db_header_size as u64;
    // A paged block checksums its prefix and bitmap together, then each page.
    let prefix_and_bitmap =
        source.read_metadata_at(db_address, db_header_size + bitmap_size + 4)?;
    signature(&prefix_and_bitmap)?;
    crate::checksum::verify_trailing(&prefix_and_bitmap)?;
    let bitmap = prefix_and_bitmap[db_header_size..db_header_size + bitmap_size].to_vec();
    let pages_start_addr = bitmap_addr + bitmap_size as u64 + 4;
    let page_stride = page_size
        .checked_mul(elem_size)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(FormatError::OffsetOverflow {
            offset: page_size as u64,
            length: elem_size as u64,
        })?;

    for page in 0..npages {
        let nelem_in_page = core::cmp::min(page_size, num_elements - page * page_size);
        let initialized = (bitmap[page / 8] >> (7 - (page % 8))) & 1 == 1;
        if !initialized {
            continue;
        }
        let page_offset = page
            .checked_mul(page_stride)
            .ok_or(FormatError::OffsetOverflow {
                offset: page as u64,
                length: page_stride as u64,
            })?;
        let page_addr = pages_start_addr + page_offset as u64;
        // Elements plus the page's own checksum. Only the last page is partial:
        // the writer does not pad it, so its checksum covers the elements it
        // actually holds.
        let region = source.read_metadata_at(
            page_addr,
            nelem_in_page
                .checked_mul(elem_size)
                .and_then(|elems| elems.checked_add(4))
                .ok_or(FormatError::OffsetOverflow {
                    offset: nelem_in_page as u64,
                    length: elem_size as u64,
                })?,
        )?;
        crate::checksum::verify_trailing(&region)?;
        for j in 0..nelem_in_page {
            if let Some(record) = parse_fa_element(
                &region,
                j * elem_size,
                header.client_id,
                chunk_byte_size,
                elem_size,
                chunk_size_bytes,
                offset_size,
            )? {
                visit((page * page_size + j) as u64, record)?;
            }
        }
    }

    Ok(())
}

/// Read a variable-length little-endian unsigned integer.
fn read_variable_length(data: &[u8], size: usize) -> Result<u64, FormatError> {
    if size > 8 || data.len() < size {
        return Err(FormatError::ChunkedReadError(
            "invalid variable-length size".into(),
        ));
    }
    let mut val = 0u64;
    for (i, &byte) in data.iter().enumerate().take(size) {
        val |= (byte as u64) << (i * 8);
    }
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::stamp_trailing as stamp;

    /// FAHD: 8 fixed bytes, the element count, the data-block address, and the
    /// checksum.
    const fn fahd_len(os: u8, ls: u8) -> usize {
        8 + ls as usize + os as usize + 4
    }

    /// The chunks the buffered walk reports, with the slot of each.
    fn records(
        file_data: &[u8],
        header: &FixedArrayHeader,
        chunk_byte_size: u64,
    ) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_fixed_array_chunks(file_data, header, 8, chunk_byte_size, |slot, record| {
            records.push((slot, record));
            Ok(())
        })?;
        Ok(records)
    }

    /// The chunks the streaming walk reports, with the slot of each.
    fn records_from_source(
        source: &[u8],
        header: &FixedArrayHeader,
        chunk_byte_size: u64,
    ) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_fixed_array_chunks_from_source(source, header, 8, chunk_byte_size, |slot, record| {
            records.push((slot, record));
            Ok(())
        })?;
        Ok(records)
    }

    #[test]
    fn read_variable_length_values() {
        assert_eq!(read_variable_length(&[0x78, 0x56], 2).unwrap(), 0x5678);
        assert_eq!(
            read_variable_length(&[0x01, 0x02, 0x03, 0x04], 4).unwrap(),
            0x04030201
        );
        assert_eq!(read_variable_length(&[0xFF], 1).unwrap(), 0xFF);
    }

    #[test]
    fn parse_fixed_array_header_valid() {
        let mut buf = vec![0u8; 256];
        // FAHD signature
        buf[0..4].copy_from_slice(b"FAHD");
        buf[4] = 0; // version
        buf[5] = 1; // client_id = filtered
        buf[6] = 16; // element_size
        buf[7] = 10; // max_nelmts_bits (page_size = 1024)
        // num_elements (length_size=8)
        buf[8..16].copy_from_slice(&5u64.to_le_bytes());
        // data_block_address (offset_size=8)
        buf[16..24].copy_from_slice(&0x1000u64.to_le_bytes());
        stamp(&mut buf, 0, fahd_len(8, 8));

        let header = FixedArrayHeader::parse(&buf, 0, 8, 8).unwrap();
        assert_eq!(header.client_id, 1);
        assert_eq!(header.element_size, 16);
        assert_eq!(header.max_nelmts_bits, 10);
        assert_eq!(header.num_elements, 5);
        assert_eq!(header.data_block_address, StoredAddress::new(0x1000));
    }

    #[test]
    fn parse_fixed_array_header_invalid_signature() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"XXXX");
        let result = FixedArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn parse_fixed_array_header_invalid_version() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"FAHD");
        buf[4] = 1; // unsupported version
        let result = FixedArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    /// Build a synthetic Fixed Array (non-filtered) and verify reading.
    #[test]
    fn read_non_filtered_chunks() {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 5u64;

        let mut file_data = vec![0u8; 0x3000];

        // Build FAHD at offset 0x100
        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0; // version
        file_data[fahd_offset + 5] = 0; // client_id = non-filtered
        file_data[fahd_offset + 6] = os as u8; // element_size = just address
        file_data[fahd_offset + 7] = 10; // max_nelmts_bits
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());
        stamp(
            &mut file_data,
            fahd_offset,
            fahd_len(offset_size, length_size),
        );

        // Build FADB at db_offset
        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0; // version
        file_data[db_offset + 5] = 0; // client_id
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes()); // header_address

        // Elements: 5 addresses
        let elem_start = db_offset + 6 + os;
        let base_addr = 0x1000u64;
        let chunk_byte_size = 20 * 8; // 20 elements × 8 bytes
        for i in 0..5 {
            let addr = base_addr + i as u64 * chunk_byte_size;
            let pos = elem_start + i * os;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
        }

        // Prefix, one address per element slot, checksum.
        stamp(
            &mut file_data,
            db_offset,
            (6 + os) + num_chunks as usize * os + 4,
        );

        let expected: Vec<(u64, ChunkRecord)> = (0..num_chunks)
            .map(|slot| {
                let record = ChunkRecord {
                    address: StoredAddress::new(base_addr + slot * chunk_byte_size),
                    stored_size: chunk_byte_size,
                    filter_mask: 0,
                };
                (slot, record)
            })
            .collect();
        assert_eq!(
            walk_both(&file_data, fahd_offset, chunk_byte_size),
            expected
        );
    }

    /// The chunks both walks report for the array whose header is at
    /// `header_offset`, which must agree.
    fn walk_both(
        file_data: &[u8],
        header_offset: usize,
        chunk_byte_size: u64,
    ) -> Vec<(u64, ChunkRecord)> {
        let header = FixedArrayHeader::parse(file_data, header_offset, 8, 8).unwrap();
        let buffered = records(file_data, &header, chunk_byte_size).unwrap();
        let header = FixedArrayHeader::parse_from_source(
            file_data,
            StoredAddress::new(header_offset as u64),
            8,
            8,
        )
        .unwrap();
        let streamed = records_from_source(file_data, &header, chunk_byte_size).unwrap();
        assert_eq!(buffered, streamed, "the two walks disagree");
        buffered
    }

    /// Build a synthetic Fixed Array (filtered) and verify reading.
    #[test]
    fn read_filtered_chunks() {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 3u64;
        // element_size for filtered: offset_size + chunk_size_bytes + 4(filter_mask)
        let chunk_size_bytes = 5usize;
        let elem_size = os + chunk_size_bytes + 4;

        let mut file_data = vec![0u8; 0x3000];

        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0;
        file_data[fahd_offset + 5] = 1; // client_id = filtered
        file_data[fahd_offset + 6] = u8::try_from(elem_size).unwrap();
        file_data[fahd_offset + 7] = 10;
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());
        stamp(
            &mut file_data,
            fahd_offset,
            fahd_len(offset_size, length_size),
        );

        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0;
        file_data[db_offset + 5] = 1;
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes());

        let elem_start = db_offset + 6 + os;
        let test_chunks = [
            (0x1000u64, u64::from(u32::MAX) + 120, 0u32),
            (0x2000u64, 115u64, 0u32),
            (0x3000u64, 100u64, 0u32),
        ];

        for (i, &(addr, csize, fmask)) in test_chunks.iter().enumerate() {
            let pos = elem_start + i * elem_size;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
            file_data[pos + os..pos + os + chunk_size_bytes]
                .copy_from_slice(&csize.to_le_bytes()[..chunk_size_bytes]);
            file_data[pos + os + chunk_size_bytes..pos + os + chunk_size_bytes + 4]
                .copy_from_slice(&fmask.to_le_bytes());
        }

        // Prefix, one filtered element record per slot, checksum.
        stamp(
            &mut file_data,
            db_offset,
            (6 + os) + num_chunks as usize * elem_size + 4,
        );

        let expected: Vec<(u64, ChunkRecord)> = (0..)
            .zip(test_chunks)
            .map(|(slot, (address, stored_size, filter_mask))| {
                let record = ChunkRecord {
                    address: StoredAddress::new(address),
                    stored_size,
                    filter_mask,
                };
                (slot, record)
            })
            .collect();
        assert_eq!(walk_both(&file_data, fahd_offset, 20 * 8), expected);
    }

    /// Build a synthetic paged (non-filtered) Fixed Array and verify both that
    /// initialized pages are read and that pages with a cleared bitmap bit are
    /// skipped as entirely unallocated.
    ///
    /// `bitmap` is the single page-init byte (MSB-first: page 0 = 0x80).
    /// Returns the chunks decoded from a 3-element array split across two
    /// pages of size 2 (`max_nelmts_bits = 1`).
    fn read_paged(bitmap: u8) -> Vec<(u64, ChunkRecord)> {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 3u64; // > page_size(2) => paged, npages = 2

        let mut file_data = vec![0u8; 0x400];

        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0; // version
        file_data[fahd_offset + 5] = 0; // client_id = non-filtered
        file_data[fahd_offset + 6] = os as u8; // element_size
        file_data[fahd_offset + 7] = 1; // max_nelmts_bits => page_size = 2
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());
        stamp(
            &mut file_data,
            fahd_offset,
            fahd_len(offset_size, length_size),
        );

        // FADB prefix: sig + version + client_id + header_addr + bitmap + checksum
        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0; // version
        file_data[db_offset + 5] = 0; // client_id
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes());
        file_data[db_offset + 14] = bitmap; // page-init bitmap (1 byte for 2 pages)
        // A paged block checksums its prefix and bitmap together.
        stamp(&mut file_data, db_offset, 6 + os + 1 + 4);

        // Pages: stride = page_size(2)*elem_size(8) + 4 checksum = 20 bytes.
        let pages_start = db_offset + 14 + 1 + 4;
        let stride = 2 * os + 4;
        let addrs = [0x1000u64, 0x2000, 0x3000];
        for (i, &addr) in addrs.iter().enumerate() {
            let page = i / 2;
            let j = i % 2;
            let pos = pages_start + page * stride + j * os;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
        }
        // Each page carries its own checksum after its elements. Page 0 is
        // full at two slots; page 1 holds the array's odd third and is not
        // padded, so its checksum sits one slot in.
        stamp(&mut file_data, pages_start, 2 * os + 4);
        stamp(&mut file_data, pages_start + stride, os + 4);

        let header =
            FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size).unwrap();
        assert_eq!(header.num_elements, 3);
        walk_both(&file_data, fahd_offset, 8)
    }

    /// The record of an unfiltered 8-byte chunk at `address`.
    fn unfiltered(address: u64) -> ChunkRecord {
        ChunkRecord {
            address: StoredAddress::new(address),
            stored_size: 8,
            filter_mask: 0,
        }
    }

    #[test]
    fn read_paged_all_pages_initialized() {
        // bitmap 0xC0 => both pages initialized (page 0 and page 1).
        assert_eq!(
            read_paged(0b1100_0000),
            vec![
                (0, unfiltered(0x1000)),
                (1, unfiltered(0x2000)),
                (2, unfiltered(0x3000)),
            ]
        );
    }

    #[test]
    fn read_paged_skips_uninitialized_page() {
        // bitmap 0x80 => only page 0 initialized; page 1's chunk is unallocated.
        assert_eq!(
            read_paged(0b1000_0000),
            vec![(0, unfiltered(0x1000)), (1, unfiltered(0x2000))]
        );
    }

    /// Every checksummed structure of a Fixed Array is verified on read, by both
    /// backends (issue #312).
    ///
    /// The array has two: the header and its one data block, which when paged
    /// checksums its prefix and bitmap together and then each page separately.
    /// Each is corrupted in its stored checksum -- bytes that carry no other
    /// meaning, so a refusal can only be the checksum. Before this the reader
    /// returned the same chunk list it did for the sound file.
    /// Refusal is what the `checksum` feature buys, so this asserts it only
    /// where it is compiled in: with the feature off `verify_trailing` is a
    /// no-op and a corrupt index reads as it did before.
    #[cfg(feature = "checksum")]
    #[test]
    fn a_corrupted_fixed_array_structure_is_refused() {
        use crate::chunked_write::build_fixed_array_at;

        let os: u8 = 8;
        let ls: u8 = 8;
        let base = 0x800u64;
        for has_filters in [false, true] {
            // 1024 = the page size, so 1025 and 3000 are paged: 3000 also
            // leaves a partial final page, which the writer does not pad.
            for &n in &[5u64, 1024, 1025, 3000] {
                let chunks: Vec<ChunkRecord> = (0..n)
                    .map(|i| ChunkRecord {
                        address: StoredAddress::new(0x100000 + i * 8),
                        stored_size: if has_filters { 8 + (i % 7) } else { 8 },
                        filter_mask: 0,
                    })
                    .collect();
                let fa = build_fixed_array_at(
                    &crate::chunked_write::IndexSlots::dense(&chunks),
                    8,
                    os,
                    ls,
                    has_filters,
                    StoredAddress::new(base),
                );
                let mut file = vec![0u8; base as usize + fa.len()];
                file[base as usize..].copy_from_slice(&fa);

                let read_both = |file: &[u8]| {
                    let buffered = FixedArrayHeader::parse(file, base as usize, os, ls)
                        .and_then(|h| records(file, &h, 8));
                    let streamed =
                        FixedArrayHeader::parse_from_source(file, StoredAddress::new(base), os, ls)
                            .and_then(|h| records_from_source(file, &h, 8));
                    (buffered, streamed.is_err())
                };
                let sound = read_both(&file).0.expect("the sound file must read");
                assert_eq!(
                    sound.len() as u64,
                    n,
                    "the fixture must read before it is corrupted (filters={has_filters}, n={n})"
                );

                let spans =
                    fixed_array_index_spans(file.as_slice(), StoredAddress::new(base), os, ls)
                        .unwrap();
                assert_eq!(spans.len(), 2, "FA index = FAHD + FADB");

                let header = FixedArrayHeader::parse(&file, base as usize, os, ls).unwrap();
                let elem_size = if has_filters {
                    header.element_size as usize
                } else {
                    os as usize
                };
                let page_size = 1usize << header.max_nelmts_bits;
                let db_prefix = 4 + 1 + 1 + os as usize;

                // The last byte of a structure is the top byte of its trailing
                // checksum -- for a paged data block, of its final page's. The
                // writer marks every page initialized, so the reader reads them
                // all and that final checksum is one it verifies.
                // The header is the one structure the reclaim walk reads: it
                // sizes the data block from the header's own fields without
                // touching it. A corrupt header must stop that walk rather than
                // have it release spans computed from bytes nothing vouched for.
                let mut poke_sites: Vec<(u64, bool)> = spans
                    .iter()
                    .enumerate()
                    .map(|(i, &(at, len))| (at + len - 1, i == 0))
                    .collect();
                if n as usize > page_size {
                    let (db_at, _) = spans[1];
                    let bitmap = (n as usize).div_ceil(page_size).div_ceil(8);
                    // The prefix-and-bitmap checksum, then the first page's.
                    poke_sites.push((db_at + (db_prefix + bitmap + 4) as u64 - 1, false));
                    poke_sites.push((
                        db_at + (db_prefix + bitmap + 4 + page_size * elem_size + 4) as u64 - 1,
                        false,
                    ));
                }

                for (site, walked) in poke_sites {
                    let at = site.to_usize().unwrap();
                    let original = file[at];
                    file[at] ^= 0x01;
                    let (buffered, streamed_err) = read_both(&file);
                    assert!(
                        matches!(buffered, Err(FormatError::ChecksumMismatch { .. })),
                        "filters={has_filters}, n={n}: a corrupted checksum at {at:#x} must be \
                         refused, got {buffered:?}"
                    );
                    assert!(
                        streamed_err,
                        "filters={has_filters}, n={n}: the streaming backend must refuse what the \
                         buffered one does, at {at:#x}"
                    );
                    if walked {
                        let walk = fixed_array_index_spans(
                            file.as_slice(),
                            StoredAddress::new(base),
                            os,
                            ls,
                        );
                        assert!(
                            matches!(walk, Err(FormatError::ChecksumMismatch { .. })),
                            "filters={has_filters}, n={n}: the reclaim walk must refuse a corrupt \
                             header at {at:#x} rather than release spans read out of it, got \
                             {walk:?}"
                        );
                    }
                    file[at] = original;
                }
            }
        }
    }

    /// `fixed_array_index_spans` must tile exactly the contiguous FAHD + FADB
    /// blob the writer produces, in both the non-paged and paged regimes and for
    /// filtered and unfiltered element records. This pins the reclaim sizing to
    /// `build_fixed_array_at` so the two cannot drift.
    #[test]
    fn index_spans_tile_fixed_array_blob() {
        use crate::chunked_write::build_fixed_array_at;

        let os: u8 = 8;
        let ls: u8 = 8;
        let base = 0x800u64;
        for has_filters in [false, true] {
            // 1024 = page_size, so 1025+ exercises the paged FADB layout.
            for &n in &[1u64, 5, 1024, 1025, 3000] {
                let chunks: Vec<ChunkRecord> = (0..n)
                    .map(|i| ChunkRecord {
                        address: StoredAddress::new(0x100000 + i * 8),
                        stored_size: if has_filters { 8 + (i % 7) } else { 8 },
                        filter_mask: 0,
                    })
                    .collect();
                let fa = build_fixed_array_at(
                    &crate::chunked_write::IndexSlots::dense(&chunks),
                    8,
                    os,
                    ls,
                    has_filters,
                    StoredAddress::new(base),
                );
                let mut file = vec![0u8; base as usize + fa.len()];
                file[base as usize..].copy_from_slice(&fa);

                let spans =
                    fixed_array_index_spans(file.as_slice(), StoredAddress::new(base), os, ls)
                        .unwrap();
                assert_eq!(
                    spans.len(),
                    2,
                    "FA index = FAHD + FADB (filters={has_filters}, n={n})"
                );

                let mut sorted = spans.clone();
                sorted.sort_by_key(|&(a, _)| a);
                // FAHD at the base, FADB immediately after, together covering the
                // whole contiguous blob with no gap or overlap.
                assert_eq!(sorted[0].0, base);
                assert_eq!(
                    sorted[0].0 + sorted[0].1,
                    sorted[1].0,
                    "FAHD must abut FADB (filters={has_filters}, n={n})"
                );
                assert_eq!(
                    sorted[1].0 + sorted[1].1,
                    base + fa.len() as u64,
                    "FA index spans must tile the blob (filters={has_filters}, n={n})"
                );
            }
        }
    }
}
