//! The Fixed Array chunk index, index type 3 of a version 4 data layout message.
//!
//! A Fixed Array indexes the chunks of a dataset whose maximum shape is fixed. Its header
//! (`FAHD`) points at one data block (`FADB`), which holds an element per slot. Past `2^page_bits`
//! slots the data block holds a page-init bitmap and then the elements in pages, each with its
//! own checksum. The index is defined in "The Fixed Array Index" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::chunk_record;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::IndexSlots;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// A Fixed Array header, signature `FAHD`, version 0.
#[derive(Debug, Clone)]
pub struct FixedArrayHeader {
    /// The client ID: 0 for unfiltered chunks and 1 for filtered ones.
    pub client_id: u8,
    /// The width in bytes of one element, the "Entry Size" field.
    pub element_size: u8,
    /// The base 2 logarithm of the number of elements in a data block page, the "Page Bits"
    /// field.
    pub max_nelmts_bits: u8,
    /// The number of elements in the array.
    pub num_elements: u64,
    /// The address of the data block.
    pub data_block_address: StoredAddress,
}

impl FixedArrayHeader {
    /// Parses the Fixed Array header at `offset` in `file_data`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the header runs past the end of `file_data`,
    /// [`FormatError::ChunkedReadError`] if the signature is not `FAHD` or the version is not 0,
    /// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not
    /// 2, 4, or 8, and [`FormatError::ChecksumMismatch`] if the checksum does not match.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        // FAHD signature(4) + version(1) + client_id(1) + element_size(1) + max_nelmts_bits(1) +
        // num_elements(length_size) + data_block_addr(offset_size) + checksum(4)
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
        let num_elements = bytes::read_length(d, pos, length_size)?;
        pos += length_size as usize;
        let data_block_address = StoredAddress::new(bytes::read_offset(d, pos, offset_size)?);

        crate::checksum::verify_trailing(&d[..min_size])?;

        Ok(FixedArrayHeader {
            client_id,
            element_size,
            max_nelmts_bits,
            num_elements,
            data_block_address,
        })
    }

    /// Parses the Fixed Array header at `address` in `source`.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
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

/// Returns the `(address, length)` spans of the header and the data block of the Fixed Array at
/// `fa_base`.
///
/// The spans cover the index alone, and [`read_fixed_array_chunks`] reports the chunks it points
/// at. An array whose data block address is undefined has the header span alone. The caller
/// checks the spans against the end of the file.
///
/// # Errors
///
/// Returns the errors [`FixedArrayHeader::parse_from_source`] returns,
/// [`FormatError::ChunkedReadError`] if the page exponent is 64 or more, and
/// [`FormatError::OffsetOverflow`] or [`FormatError::ValueTooLargeForPlatform`] if the length of
/// the data block does not fit a `usize`.
pub fn fixed_array_index_spans(
    source: &(impl MetadataSource + ?Sized),
    fa_base: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let header = FixedArrayHeader::parse_from_source(source, fa_base, offset_size, length_size)?;
    let os = offset_size as usize;

    // FAHD: sig(4) + ver(1) + client(1) + elem_size(1) + max_bits(1) + num_elements(ls) +
    // dblk_addr(os) + checksum(4).
    let fahd_size = (4 + 1 + 1 + 1 + 1 + length_size as usize + os + 4) as u64;
    let mut spans = vec![(fa_base.get(), fahd_size)];

    // An array with the undefined data block address has no data block.
    if header.data_block_address.is_undefined(offset_size) {
        return Ok(spans);
    }

    // An unfiltered element is the chunk address alone, whatever the entry size.
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
        // Not paged: the prefix, the elements, and the checksum.
        (db_prefix
            + num_elements
                .checked_mul(elem_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: num_elements as u64,
                    length: elem_size as u64,
                })?
            + 4) as u64
    } else {
        // Paged: the prefix, the page-init bitmap, and their checksum, then `npages` pages with
        // a checksum each. The writer does not pad the last page.
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

/// Parses the element at `elem_pos` in `block`, or returns `None` where it stores the undefined
/// address.
///
/// `block` is the whole file for the buffered reader, with `elem_pos` absolute, or a block the
/// streaming reader read, with `elem_pos` relative to its start.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the element runs past the end of `block`,
/// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and
/// [`FormatError::ChunkedReadError`] if `chunk_size_bytes` is more than 8.
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
    let Some(address) =
        bytes::read_optional_offset(block, elem_pos, offset_size)?.map(StoredAddress::new)
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

/// Reads the data block of the Fixed Array `header` describes from `file_data`, and calls `visit`
/// with the slot and the record of each element that stores a chunk address.
///
/// A record of an unfiltered chunk has `chunk_byte_size` as its stored size. A reader of a paged
/// data block skips a page the page-init bitmap does not mark initialized.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the signature is not `FADB` or the entry size is
/// too small for a filtered element or leaves more than 8 bytes for its chunk size,
/// [`FormatError::UnexpectedEof`] if the data block runs past the end of `file_data`,
/// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8,
/// [`FormatError::ChecksumMismatch`] if a checksum does not match, [`FormatError::OffsetOverflow`]
/// or [`FormatError::ValueTooLargeForPlatform`] if a position does not fit a `usize`, and the error
/// `visit` returns.
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
        // Not paged: the prefix, every element, and one checksum over both. A slot no chunk
        // occupies stores the undefined address.
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

    // Paged: the prefix, a page-init bitmap with page 0 as its most significant bit, and a
    // checksum, then the pages at a stride of `page_size` elements and a checksum.
    let npages = num_elements.div_ceil(page_size);
    let bitmap_size = npages.div_ceil(8);
    let bitmap_pos = db_offset + db_header_size;
    // A crafted element count can put the end of the bitmap past `usize::MAX`.
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
        // A cleared bit marks an unoccupied page.
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
        // The writer does not pad the last page, so its checksum follows its last element.
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

/// Reads the data block of the Fixed Array `header` describes from `source`, as
/// [`read_fixed_array_chunks`] reads it from a buffer.
///
/// An unpaged data block takes one read, and a paged one a read of its prefix and bitmap and one
/// read per initialized page.
///
/// # Errors
///
/// Returns the errors [`read_fixed_array_chunks`] returns, and the error `source` returns if a
/// read fails.
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

    // Both branches read the prefix in the read its checksum covers, and check the signature
    // there.
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
        // Not paged: the prefix, every element, and one checksum, in one read.
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

    // Paged: the page-init bitmap, then each initialized page.
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
        // The writer does not pad the last page, so its checksum follows its last element.
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

/// Reads a little-endian unsigned integer `size` bytes wide from the start of `data`.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if `size` is more than 8 or `data` is shorter than
/// `size`.
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

/// The base 2 logarithm of the number of elements in a data block page of a Fixed Array the
/// writer builds, 1024 elements.
///
/// The writer stores it in the "Page Bits" field of both the header and the version 4 data layout
/// message. The C library calls this value `H5D_FARRAY_MAX_DBLK_PAGE_NELMTS_BITS`. A reader takes
/// the page size from the header.
pub const FIXED_ARRAY_PAGE_BITS: u8 = 10;

/// The layout of a Fixed Array that does not depend on its address: the element encoding, the
/// paging, and the lengths.
///
/// [`fixed_array_len`] computes the layout without building the array, for a caller that reserves
/// the span of the array before writing it.
struct FixedArrayLayout {
    encoding: ChunkElementEncoding,
    /// The base 2 logarithm of `page_size`, which the header stores.
    page_bits: u8,
    /// The number of elements in a page, past which the data block is paged.
    page_size: usize,
    /// The length of the header in bytes.
    fahd_size: usize,
    /// The length of the header and the data block in bytes.
    total_len: u64,
}

/// Returns the layout of the Fixed Array that holds `slots`.
fn fixed_array_layout(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> FixedArrayLayout {
    let os = usize::from(offset_size.get());
    let num_elements = slots.len();
    let encoding = chunk_record::chunk_element_encoding(chunk_bytes, offset_size, has_filters);

    let fahd_size = 4 + 1 + 1 + 1 + 1 + usize::from(length_size.get()) + os + 4;

    // The data block is the prefix, then either the elements and one checksum, or the page-init
    // bitmap, its checksum, and the pages with a checksum each. The element bytes are the same in
    // both.
    let fadb_prefix = 4 + 1 + 1 + os;
    let page_bits = FIXED_ARRAY_PAGE_BITS;
    let page_size = 1usize << page_bits;
    let elements = num_elements * encoding.elem_size;
    let fadb_size = if num_elements <= page_size {
        fadb_prefix + elements + 4
    } else {
        let npages = num_elements.div_ceil(page_size);
        fadb_prefix + npages.div_ceil(8) + 4 + elements + npages * 4
    };

    FixedArrayLayout {
        encoding,
        page_bits,
        page_size,
        fahd_size,
        total_len: (fahd_size + fadb_size) as u64,
    }
}

/// Returns the length in bytes of the Fixed Array [`build_fixed_array_at`] builds for `slots`,
/// without building it.
pub fn fixed_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> u64 {
    fixed_array_layout(slots, chunk_bytes, offset_size, length_size, has_filters).total_len
}

/// Builds the Fixed Array for `slots` at `fa_address`: the header, then the data block.
///
/// `chunk_bytes` is the size of a chunk before filtering, which sets the width of the chunk size
/// field of a filtered element. A slot no chunk occupies stores the undefined address. Past
/// `1 << FIXED_ARRAY_PAGE_BITS` slots the data block is paged, and every page is marked
/// initialized.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the stored size of a chunk does not fit the chunk size
/// field.
pub fn build_fixed_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    fa_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = slots.len();

    let layout = fixed_array_layout(slots, chunk_bytes, offset_size, length_size, has_filters);
    let ChunkElementEncoding {
        chunk_size_bytes,
        elem_size,
        client_id,
    } = layout.encoding;
    let fahd_total_size = layout.fahd_size;
    let fadb_address = fa_address.offset(fahd_total_size as u64);

    // Build FAHD
    let mut fahd = Vec::with_capacity(fahd_total_size);
    fahd.extend_from_slice(b"FAHD");
    fahd.push(0); // version
    fahd.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an element record is an address of at most 8 bytes, a chunk size of at most \
                  8 and a 4-byte filter mask, so its size fits the 1-byte FAHD field"
    )]
    fahd.push(elem_size as u8);

    fahd.push(layout.page_bits);

    bytes::write_length(&mut fahd, num_elements.to_u64(), length_size);
    bytes::write_offset(&mut fahd, fadb_address.get(), offset_size);

    // Checksum
    let checksum = checksum::jenkins_lookup3(&fahd);
    fahd.extend_from_slice(&checksum.to_le_bytes());

    debug_assert_eq!(fahd.len(), fahd_total_size);

    let write_element = |buf: &mut Vec<u8>, chunk: Option<&ChunkRecord>| match chunk {
        Some(chunk) => chunk_record::write_chunk_element(
            buf,
            chunk,
            offset_size,
            has_filters,
            chunk_size_bytes,
        ),
        None => {
            chunk_record::write_undefined_element(buf, offset_size, has_filters, chunk_size_bytes);
            Ok(())
        }
    };

    // Build the `FADB` prefix: signature + version + `client_id` + header address.
    let mut fadb = Vec::new();
    fadb.extend_from_slice(b"FADB");
    fadb.push(0); // version
    fadb.push(client_id);
    bytes::write_offset(&mut fadb, fa_address.get(), offset_size);

    let page_size = layout.page_size;
    if num_elements <= page_size {
        // Non-paged: elements stored directly, then a single checksum.
        for slot in 0..num_elements {
            write_element(&mut fadb, slots.at(slot))?;
        }
        let fadb_checksum = checksum::jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&fadb_checksum.to_le_bytes());
    } else {
        // Paged: the page-init bitmap and its checksum, then each page and its checksum.
        let npages = num_elements.div_ceil(page_size);
        let bitmap_size = npages.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_size];
        for page in 0..npages {
            // Page 0 is the most significant bit, as `H5VM_bit_set` sets it (`H5VMprivate.h`,
            // HDF5 2.2.0).
            bitmap[page / 8] |= 1 << (7 - (page % 8));
        }
        fadb.extend_from_slice(&bitmap);
        let prefix_checksum = checksum::jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&prefix_checksum.to_le_bytes());

        for page in 0..npages {
            let start = page * page_size;
            let end = core::cmp::min(start + page_size, num_elements);
            let mut page_buf = Vec::with_capacity((end - start) * elem_size);
            for slot in start..end {
                write_element(&mut page_buf, slots.at(slot))?;
            }
            let page_checksum = checksum::jenkins_lookup3(&page_buf);
            page_buf.extend_from_slice(&page_checksum.to_le_bytes());
            fadb.extend_from_slice(&page_buf);
        }
    }

    let mut combined = fahd;
    combined.extend_from_slice(&fadb);
    // The caller reserved the length `fixed_array_len` returns.
    debug_assert_eq!(
        combined.len() as u64,
        layout.total_len,
        "a fixed array must fill the length its layout promised"
    );
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::checksum::restamp as stamp;

    use super::*;

    /// Returns the length of a header: 8 fixed bytes, the element count, the data block address,
    /// and the checksum.
    const fn fahd_len(os: u8, ls: u8) -> usize {
        8 + ls as usize + os as usize + 4
    }

    /// Returns the chunks the buffered walk reports, with the slot of each.
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

    /// Returns the chunks the streaming walk reports, with the slot of each.
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

    /// Returns the chunks both walks report for the array whose header is at `header_offset`, and
    /// asserts that the two agree.
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

    /// Returns the chunks both walks read from an unfiltered array of 3 elements in two pages of
    /// 2, with `bitmap` as its page-init byte and page 0 as its most significant bit.
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

    /// Returns the record of an unfiltered 8-byte chunk at `address`.
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

    // Flips a checksum byte of each structure for every reader of the structure to reject.
    #[cfg(feature = "checksum")]
    #[test]
    fn a_corrupted_fixed_array_structure_is_refused() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let base = 0x800u64;
        for has_filters in [false, true] {
            // 1025 and 3000 are paged, and 3000 leaves a partial last page.
            for &n in &[5u64, 1024, 1025, 3000] {
                let chunks: Vec<ChunkRecord> = (0..n)
                    .map(|i| ChunkRecord {
                        address: StoredAddress::new(0x100000 + i * 8),
                        stored_size: if has_filters { 8 + (i % 7) } else { 8 },
                        filter_mask: 0,
                    })
                    .collect();
                let fa = build_fixed_array_at(
                    &IndexSlots::dense(&chunks),
                    8,
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                    has_filters,
                    StoredAddress::new(base),
                )
                .unwrap();
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

                // The last byte of a structure is the top byte of its checksum, for a paged data
                // block the checksum of its last page, which the reader verifies since the writer
                // marks every page initialized. The span walk reads the header alone and sizes
                // the data block from it.
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

    // The oracle is the array `build_fixed_array_at` builds, paged and not, filtered and not.
    #[test]
    fn index_spans_tile_fixed_array_blob() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let base = 0x800u64;
        for has_filters in [false, true] {
            // 1025 and 3000 are paged.
            for &n in &[1u64, 5, 1024, 1025, 3000] {
                let chunks: Vec<ChunkRecord> = (0..n)
                    .map(|i| ChunkRecord {
                        address: StoredAddress::new(0x100000 + i * 8),
                        stored_size: if has_filters { 8 + (i % 7) } else { 8 },
                        filter_mask: 0,
                    })
                    .collect();
                let fa = build_fixed_array_at(
                    &IndexSlots::dense(&chunks),
                    8,
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                    has_filters,
                    StoredAddress::new(base),
                )
                .unwrap();
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
                // The header at the base, and the data block after it to the end of the array.
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

    #[test]
    fn build_fixed_array_valid_structure() {
        let chunks = vec![
            ChunkRecord {
                address: StoredAddress::new(0x1000),
                stored_size: 160,
                filter_mask: 0,
            },
            ChunkRecord {
                address: StoredAddress::new(0x10A0),
                stored_size: 160,
                filter_mask: 0,
            },
        ];
        let fa = build_fixed_array_at(
            &IndexSlots::dense(&chunks),
            160,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(0x2000),
        )
        .unwrap();
        // Should start with FAHD
        assert_eq!(&fa[0..4], b"FAHD");
        // FAHD size = 4+1+1+1+1+8+8+4 = 28
        // FADB starts at offset 28
        assert_eq!(&fa[28..32], b"FADB");
    }

    // Sweeps every count across the page size, where the data block becomes paged, and counts
    // that leave a partial last page.
    #[test]
    fn fixed_array_len_matches_what_it_builds() {
        fn check(
            n: u64,
            chunk_bytes: u64,
            offset_size: OffsetWidth,
            length_size: LengthWidth,
            has_filters: bool,
        ) {
            let chunks: Vec<ChunkRecord> = (0..n)
                .map(|i| ChunkRecord {
                    address: StoredAddress::new(0x1000 + i * 8),
                    stored_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let planned = fixed_array_len(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
            );
            let built = build_fixed_array_at(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
                StoredAddress::new(0x10_0000),
            )
            .unwrap();
            assert_eq!(
                planned,
                built.len() as u64,
                "planned length must match the emitted array at n={n}, \
                 chunk_bytes={chunk_bytes}, offset_size={offset_size:?}, \
                 has_filters={has_filters}"
            );
        }

        for (offset_size, length_size) in [
            (OffsetWidth::Eight, LengthWidth::Eight),
            (OffsetWidth::Four, LengthWidth::Four),
        ] {
            for &has_filters in &[false, true] {
                // The array is paged past `1 << FIXED_ARRAY_PAGE_BITS` elements.
                for n in 0..=1_100u64 {
                    check(n, 8, offset_size, length_size, has_filters);
                }
                // Several whole pages, and a count that leaves a partial one.
                for &n in &[4_096u64, 5_000, 100_000] {
                    check(n, 8, offset_size, length_size, has_filters);
                }
            }
        }

        // Each chunk size selects a different chunk size field width.
        for &chunk_bytes in &chunk_record::CHUNK_BYTES {
            for &n in &[1u64, 1_024, 1_025, 5_000] {
                check(n, chunk_bytes, OffsetWidth::Eight, LengthWidth::Eight, true);
            }
        }
    }

    #[rstest]
    fn an_array_reads_back_at_every_width(
        #[values(
            (OffsetWidth::Two, LengthWidth::Two),
            (OffsetWidth::Four, LengthWidth::Four),
            (OffsetWidth::Eight, LengthWidth::Eight)
        )]
        widths: (OffsetWidth, LengthWidth),
        #[values(5, 3_000)] n: u64,
        #[values(false, true)] has_filters: bool,
    ) {
        let (offset_size, length_size) = widths;
        let chunks: Vec<ChunkRecord> = (0..n)
            .map(|i| ChunkRecord {
                address: StoredAddress::new(0x100 + i * 8),
                stored_size: if has_filters { 8 + (i % 7) } else { 8 },
                filter_mask: 0,
            })
            .collect();
        let base = 0x80;
        let fa = build_fixed_array_at(
            &IndexSlots::dense(&chunks),
            8,
            offset_size,
            length_size,
            has_filters,
            StoredAddress::new(base),
        )
        .unwrap();
        let mut file = vec![0u8; base as usize];
        file.extend_from_slice(&fa);

        let header =
            FixedArrayHeader::parse(&file, base as usize, offset_size.get(), length_size.get())
                .unwrap();
        let mut read = Vec::new();
        read_fixed_array_chunks(&file, &header, offset_size.get(), 8, |_, record| {
            read.push(record);
            Ok(())
        })
        .unwrap();
        assert_eq!(read, chunks);
    }
}
