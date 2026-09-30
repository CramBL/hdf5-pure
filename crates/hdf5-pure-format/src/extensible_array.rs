//! The Extensible Array chunk index, index type 4 of a version 4 data layout message.
//!
//! An Extensible Array indexes the chunks of a dataset with one unlimited dimension. Its header
//! (`EAHD`) points at an index block (`EAIB`), which holds the first elements, the addresses of
//! the first data blocks (`EADB`), and the addresses of super blocks (`EASB`), each of which
//! addresses further data blocks. A data block of more than one page of elements is paged, and
//! the super block that addresses it holds its page-init bitmap. The index is defined in "The
//! Extensible Array Index" of the [format specification, version 4.0][spec], which calls a super
//! block a secondary block.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr

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

/// An Extensible Array header, signature `EAHD`, version 0.
#[derive(Debug, Clone)]
pub struct ExtensibleArrayHeader {
    /// The client ID: 0 for unfiltered chunks and 1 for filtered ones.
    pub client_id: u8,
    /// The width in bytes of one element.
    pub element_size: u8,
    /// The number of bits that hold the maximum number of elements.
    pub max_nelmts_bits: u8,
    /// The number of elements the index block holds.
    pub idx_blk_elmts: u8,
    /// The number of elements in the smallest data block.
    pub min_dblk_nelmts: u8,
    /// The fewest data block addresses a super block holds.
    pub super_blk_min_data_ptrs: u8,
    /// The base 2 logarithm of the number of elements in a data block page.
    pub max_dblk_nelmts_bits: u8,
    /// One more than the highest element index set, the "Max Index Set" field.
    pub max_idx_set: u64,
    /// The address of the index block.
    pub index_block_address: StoredAddress,
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

impl ExtensibleArrayHeader {
    /// Parses the Extensible Array header at `offset` in `file_data`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the header runs past the end of `file_data`,
    /// [`FormatError::ChunkedReadError`] if the signature is not `EAHD` or the version is not 0,
    /// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not
    /// 2, 4, or 8, and [`FormatError::ChecksumMismatch`] if the checksum does not match.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        // EAHD: the signature (4), eight 1-byte fields, six statistics of `length_size` bytes,
        // the index block address, and the checksum (4).
        let min_size =
            4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * length_size as usize + offset_size as usize + 4;
        if min_size > file_data.len() || offset > file_data.len() - min_size {
            return Err(FormatError::UnexpectedEof {
                expected: offset.saturating_add(min_size),
                available: file_data.len(),
            });
        }

        let d = &file_data[offset..];
        if &d[0..4] != b"EAHD" {
            return Err(FormatError::ChunkedReadError(
                "invalid Extensible Array header signature".into(),
            ));
        }

        let version = d[4];
        if version != 0 {
            return Err(FormatError::ChunkedReadError(format!(
                "unsupported Extensible Array header version: {version}"
            )));
        }

        let client_id = d[5];
        let element_size = d[6];
        let max_nelmts_bits = d[7];
        let idx_blk_elmts = d[8];
        let min_dblk_nelmts = d[9];
        let super_blk_min_data_ptrs = d[10];
        let max_dblk_nelmts_bits = d[11];

        let mut pos = 12;
        // The six statistics, in the order the header stores them:
        //   [0] `nsuper_blks`   [1] `super_blk_size`   [2] `ndata_blks`
        //   [3] `data_blk_size` [4] `max_idx_set`      [5] `nelmts`
        // The parser reads `max_idx_set`, one more than the highest element index set, which
        // bounds the elements in use. `nelmts` counts every slot of the allocated blocks.
        let ls = length_size as usize;
        pos += 4 * ls; // skip [0]..[3]
        let max_idx_set = bytes::read_length(d, pos, length_size)?; // [4]
        pos += ls;
        pos += ls; // skip [5] nelmts
        let index_block_address = StoredAddress::new(bytes::read_offset(d, pos, offset_size)?);

        crate::checksum::verify_trailing(&d[..min_size])?;

        Ok(ExtensibleArrayHeader {
            client_id,
            element_size,
            max_nelmts_bits,
            idx_blk_elmts,
            min_dblk_nelmts,
            super_blk_min_data_ptrs,
            max_dblk_nelmts_bits,
            max_idx_set,
            index_block_address,
        })
    }

    /// Returns the length in bytes of a header with `offset_size`-byte addresses and
    /// `length_size`-byte lengths.
    pub fn serialized_size(offset_size: u8, length_size: u8) -> usize {
        4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * length_size as usize + offset_size as usize + 4
    }

    /// Parses the Extensible Array header at `address` in `source`.
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
        let size = Self::serialized_size(offset_size, length_size);
        let buf = source.read_metadata_at(address.get(), size)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// Returns the width in bytes of an element: the address alone for an unfiltered index, and the
/// element size of the header for a filtered one.
fn ea_elem_stride(header: &ExtensibleArrayHeader, offset_size: u8) -> usize {
    if header.client_id == 0 {
        offset_size as usize
    } else {
        header.element_size as usize
    }
}

/// The number and the size of the data blocks of each super block of an Extensible Array, from
/// the creation parameters of its header.
///
/// Super block `i` has `2^floor(i/2)` data blocks of `min_dblk_nelmts * 2^ceil(i/2)` elements
/// each, as `H5EA__hdr_init` computes them (`H5EAhdr.c`, HDF5 2.2.0):
///
/// ```text
/// SB0: 1 x 16   SB1: 1 x 32   SB2: 2 x 32   SB3: 2 x 64   SB4: 4 x 64 ...
/// ```
///
/// The index block holds the data block addresses of the first `super_blk_min_data_ptrs` super
/// blocks, and the addresses of the super blocks after them.
#[derive(Debug, Clone)]
pub struct ExtensibleArrayGeometry {
    /// The `(ndblks, dblk_nelmts)` of each super block.
    pub(crate) sblks: Vec<(u64, u64)>,
    /// The number of elements in each data block the index block addresses.
    pub(crate) direct_dblk_nelmts: Vec<u64>,
    /// The number of super block addresses in the index block.
    pub(crate) nsblk_addrs: usize,
    /// The index in `sblks` of the super block the first super block address points at,
    /// `super_blk_min_data_ptrs`.
    pub(crate) first_indirect_sblk: usize,
}

impl ExtensibleArrayGeometry {
    /// Returns the geometry of the super block the `j`th super block address of the index block
    /// points at, in an array whose pages hold `page_nelmts` elements.
    ///
    /// `j` counts the super block addresses, and the first `first_indirect_sblk` super blocks have
    /// none.
    pub(crate) fn super_block_at(&self, j: usize, page_nelmts: u64) -> SuperBlockGeometry {
        let (ndblks, dblk_nelmts) = self.sblks[self.first_indirect_sblk + j];
        SuperBlockGeometry {
            ndblks,
            blocks: DataBlockGeometry {
                dblk_nelmts,
                page_nelmts,
            },
        }
    }

    /// Returns the geometry of the array `h` describes.
    pub fn from_header(h: &ExtensibleArrayHeader) -> Self {
        let min_dblk = h.min_dblk_nelmts as u64;
        let sup_blk_min = h.super_blk_min_data_ptrs as usize;
        // `nsblks = max_nelmts_bits - log2(min_dblk_nelmts) + 1`
        let log2_min = if min_dblk <= 1 {
            0
        } else {
            min_dblk.trailing_zeros() as u64
        };
        // `max_nelmts_bits` is a bit-count header field, so the difference is a
        // small super-block count that always fits `usize`.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "max_nelmts_bits is a bit count (<= 64); the super-block count fits usize"
        )]
        let nsblks = (h.max_nelmts_bits as u64).saturating_sub(log2_min) as usize + 1;

        let mut sblks = Vec::with_capacity(nsblks);
        let mut ndblks = 1u64;
        let mut dblk_nelmts = min_dblk;
        for u in 0..nsblks {
            sblks.push((ndblks, dblk_nelmts));
            if u % 2 == 0 {
                dblk_nelmts = dblk_nelmts.saturating_mul(2);
            } else {
                ndblks = ndblks.saturating_mul(2);
            }
        }

        let mut direct_dblk_nelmts = Vec::new();
        for sb in sblks.iter().take(sup_blk_min.min(nsblks)) {
            let (nd, dn) = *sb;
            for _ in 0..nd {
                direct_dblk_nelmts.push(dn);
            }
        }

        let nsblk_addrs = nsblks.saturating_sub(sup_blk_min);
        ExtensibleArrayGeometry {
            sblks,
            direct_dblk_nelmts,
            nsblk_addrs,
            first_indirect_sblk: sup_blk_min,
        }
    }

    /// Returns the `(ndblks, dblk_nelmts)` of each super block: the number of its data blocks and
    /// the number of elements in each.
    pub fn sblks(&self) -> &[(u64, u64)] {
        &self.sblks
    }

    /// Returns the number of elements in each data block the index block addresses.
    pub fn direct_dblk_nelmts(&self) -> &[u64] {
        &self.direct_dblk_nelmts
    }

    /// Returns the number of super block addresses in the index block.
    pub fn nsblk_addrs(&self) -> usize {
        self.nsblk_addrs
    }

    /// Returns the index in [`sblks`](Self::sblks) of the super block the first super block
    /// address points at.
    pub fn first_indirect_sblk(&self) -> usize {
        self.first_indirect_sblk
    }
}

/// The number of elements in an Extensible Array data block, and the number in one page.
///
/// A data block of more elements than a page is paged, and one of exactly a page is not, as in
/// `H5EA__dblock_alloc` (`H5EAdblock.c`, HDF5 2.2.0). The rule holds for a data block the
/// index block addresses as for one a super block addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataBlockGeometry {
    /// The number of elements in the data block.
    pub dblk_nelmts: u64,
    /// The number of elements in a page.
    pub page_nelmts: u64,
}

impl DataBlockGeometry {
    /// Returns whether the data block is stored in pages.
    pub const fn is_paged(self) -> bool {
        self.dblk_nelmts > self.page_nelmts
    }

    /// Returns the number of pages in the data block, or 0 if it is not paged.
    pub const fn npages(self) -> u64 {
        if self.is_paged() {
            self.dblk_nelmts / self.page_nelmts
        } else {
            0
        }
    }
}

/// The number of data blocks of an Extensible Array super block (`EASB`), and their geometry.
///
/// A super block whose data blocks are paged stores a page-init bitmap between its block offset
/// and its data block addresses, so the position of every address in the block depends on
/// [`DataBlockGeometry::is_paged`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperBlockGeometry {
    /// The number of data block addresses in the super block.
    pub ndblks: u64,
    /// The geometry of each of its data blocks.
    pub blocks: DataBlockGeometry,
}

impl SuperBlockGeometry {
    /// Returns the length in bytes of the page-init bitmap, or 0 if the data blocks are not paged.
    ///
    /// The length is `ceil(npages / 8)` bytes per data block. The bits form one stream indexed by
    /// `dblk_local * npages + page`, so the bits of data block `k` begin at bit `k * npages`.
    ///
    /// The length is at most `ndblks * dblk_nelmts`, and every caller computes that product or
    /// the length of the enclosing block first, so the `u64` product does not overflow.
    pub const fn bitmap_size(self) -> u64 {
        self.ndblks * self.blocks.npages().div_ceil(8)
    }
}

/// Parses the element at `pos` in `data`, and returns its record, or `None` where it stores the
/// undefined address, with its width in bytes.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if `element_size` is too small for a filtered element
/// or leaves more than 8 bytes for its chunk size, [`FormatError::InvalidOffsetSize`] if
/// `offset_size` is not 2, 4, or 8, and [`FormatError::UnexpectedEof`] if the element runs past the
/// end of `data`.
fn read_element(
    data: &[u8],
    pos: usize,
    client_id: u8,
    element_size: u8,
    offset_size: u8,
    chunk_byte_size: u64,
) -> Result<(Option<ChunkRecord>, usize), FormatError> {
    let os = offset_size as usize;

    if client_id == 0 {
        // Non-filtered: just address
        if os > data.len() || pos > data.len() - os {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(os),
                available: data.len(),
            });
        }
        let Some(address) =
            bytes::read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new)
        else {
            return Ok((None, os));
        };
        Ok((
            Some(ChunkRecord {
                address,
                stored_size: chunk_byte_size,
                filter_mask: 0,
            }),
            os,
        ))
    } else {
        // Filtered: the address, the chunk size, and the filter mask. The width comes from the
        // header, where a crafted file can store one too small for the address and the mask.
        let chunk_size_bytes = (element_size as usize).checked_sub(os + 4).ok_or_else(|| {
            FormatError::ChunkedReadError("Extensible Array element size too small".into())
        })?;
        let elem_total = os + chunk_size_bytes + 4;
        if elem_total > data.len() || pos > data.len() - elem_total {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(elem_total),
                available: data.len(),
            });
        }
        let Some(address) =
            bytes::read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new)
        else {
            return Ok((None, elem_total));
        };
        let chunk_size = read_variable_length(&data[pos + os..], chunk_size_bytes)?;
        let fm_off = pos + os + chunk_size_bytes;
        let filter_mask = u32::from_le_bytes([
            data[fm_off],
            data[fm_off + 1],
            data[fm_off + 2],
            data[fm_off + 3],
        ]);
        Ok((
            Some(ChunkRecord {
                address,
                stored_size: chunk_size,
                filter_mask,
            }),
            elem_total,
        ))
    }
}

/// Reads the unpaged data block of `nelmts` elements at `db_offset` in `file_data`, whose first
/// element is element `start_index` of the array, and calls `visit` for each element before
/// `total_elements` that stores a chunk address.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the block runs past the end of `file_data`,
/// [`FormatError::ChunkedReadError`] if the signature is not `EADB`,
/// [`FormatError::ChecksumMismatch`] if the checksum does not match,
/// [`FormatError::OffsetOverflow`] if the length of the block does not fit a `usize`, the errors
/// [`read_element`] returns, and the error `visit` returns.
#[allow(clippy::too_many_arguments)]
fn read_data_block_elements(
    file_data: &[u8],
    db_offset: usize,
    nelmts: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    // EADB: signature(4) + version(1) + client_id(1) + header_address(offset_size)
    let db_header_size = 4 + 1 + 1 + offset_size as usize;
    // Block offset is encoded in ceil(max_nelmts_bits/8) bytes.
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    // The checksum covers all `nelmts` elements, and the reader parses those before
    // `total_elements`.
    let db_len = eadb_extent(nelmts, header, offset_size, blk_off_size)?;
    if db_len > file_data.len() || db_offset > file_data.len() - db_len {
        return Err(FormatError::UnexpectedEof {
            expected: db_offset.saturating_add(db_len),
            available: file_data.len(),
        });
    }

    let d = &file_data[db_offset..db_offset + db_len];
    if &d[0..4] != b"EADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array data block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(d)?;
    // Skips the version, the client ID, the header address, and the block offset.
    let mut pos = db_offset + db_header_size + blk_off_size;

    // A SWMR writer grows a block before it raises the header's count, so an element past
    // `total_elements` may hold stale bytes.
    let limit = total_elements.saturating_sub(start_index).min(nelmts);
    for i in 0..limit {
        let (record, consumed) = read_element(
            file_data,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
        )?;
        if let Some(record) = record {
            visit((start_index + i) as u64, record)?;
        }
        pos += consumed;
    }

    Ok(())
}

/// Returns the length in bytes of an unpaged data block (`EADB`) of `nelmts` elements: the prefix,
/// the elements, and the checksum.
///
/// [`data_block_len`] computes the same length for the writer. This one uses checked arithmetic,
/// since `nelmts` comes from a file, and `eadb_extent_matches_the_writer` checks that the two
/// agree.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the length does not fit a `usize`.
fn eadb_extent(
    nelmts: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    blk_off_size: usize,
) -> Result<usize, FormatError> {
    let elem_stride = ea_elem_stride(header, offset_size);
    nelmts
        .checked_mul(elem_stride)
        .and_then(|elems| elems.checked_add(4 + 1 + 1 + offset_size as usize + blk_off_size + 4))
        .ok_or(FormatError::OffsetOverflow {
            offset: nelmts as u64,
            length: elem_stride as u64,
        })
}

/// Returns whether the page-init bitmap `bitmap` marks page `page_idx` initialized.
///
/// Page 0 is the most significant bit of byte 0, and page 8 the most significant bit of byte 1. A
/// page past the end of the bitmap is not initialized.
fn page_is_initialized(bitmap: &[u8], page_idx: usize) -> bool {
    let byte = page_idx / 8;
    let mask = 0x80u8 >> (page_idx % 8);
    byte < bitmap.len() && (bitmap[byte] & mask) != 0
}

/// Reads the paged data block at `db_offset` in `file_data`, and calls `visit` for each element
/// before `total_elements` that stores a chunk address.
///
/// A paged data block has a prefix with its own checksum, then `npages` pages of `page_nelmts`
/// elements and a checksum each. The super block that addresses it marks page `p` of its data
/// block `db_local_idx` initialized at bit `db_local_idx * npages + p` of `page_bitmap`. A page
/// the bitmap does not mark takes its full length in the block, and the reader steps over it.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the prefix or an initialized page runs past the end
/// of `file_data`, [`FormatError::ChunkedReadError`] if the signature is not `EADB`,
/// [`FormatError::ChecksumMismatch`] if a checksum does not match,
/// [`FormatError::OffsetOverflow`] if the length of a page does not fit a `usize`, the errors
/// [`read_element`] returns, and the error `visit` returns.
#[allow(clippy::too_many_arguments)]
fn read_paged_data_block(
    file_data: &[u8],
    db_offset: usize,
    page_nelmts: usize,
    npages: usize,
    db_local_idx: usize,
    page_bitmap: &[u8],
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    // Header includes its own checksum: sig(4)+ver(1)+cid(1)+hdr_addr+block_offset+checksum(4)
    let db_header_size = 4 + 1 + 1 + offset_size as usize + blk_off_size + 4;
    if db_header_size > file_data.len() || db_offset > file_data.len() - db_header_size {
        return Err(FormatError::UnexpectedEof {
            expected: db_offset.saturating_add(db_header_size),
            available: file_data.len(),
        });
    }
    if &file_data[db_offset..db_offset + 4] != b"EADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array data block signature".into(),
        ));
    }
    // A paged block checksums its header on its own, then each page separately.
    crate::checksum::verify_trailing(&file_data[db_offset..db_offset + db_header_size])?;

    let mut pos = db_offset + db_header_size;
    // A writer initializes a page when it writes a chunk into it, in any order.
    let page_stride = page_nelmts
        .checked_mul(ea_elem_stride(header, offset_size))
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(FormatError::OffsetOverflow {
            offset: page_nelmts as u64,
            length: ea_elem_stride(header, offset_size) as u64,
        })?;
    for page in 0..npages {
        let global_page = db_local_idx * npages + page;
        if !page_is_initialized(page_bitmap, global_page) {
            pos += page_stride;
            continue;
        }
        // The checksum covers all `page_nelmts` elements, including those past
        // `total_elements`.
        let page_end = pos
            .checked_add(page_stride)
            .filter(|&end| end <= file_data.len())
            .ok_or(FormatError::UnexpectedEof {
                expected: pos.saturating_add(page_stride),
                available: file_data.len(),
            })?;
        crate::checksum::verify_trailing(&file_data[pos..page_end])?;
        let page_start = start_index + page * page_nelmts;
        // The walk stops at `total_elements`, as in `read_data_block_elements`.
        let limit = total_elements.saturating_sub(page_start).min(page_nelmts);
        for i in 0..limit {
            let (record, consumed) = read_element(
                file_data,
                pos,
                header.client_id,
                header.element_size,
                offset_size,
                chunk_byte_size,
            )?;
            if let Some(record) = record {
                visit((page_start + i) as u64, record)?;
            }
            pos += consumed;
        }
        if limit < page_nelmts {
            break; // the rest are past `total_elements`
        }
        // Skip the page checksum.
        pos += 4;
    }

    Ok(())
}

/// Reads the Extensible Array `header` describes from `file_data`, and calls `visit` with the slot
/// and the record of each element that stores a chunk address.
///
/// The reader walks the elements of the index block, the data blocks it addresses, and then the
/// super blocks and their data blocks, and stops at the header's "Max Index Set". A record of an
/// unfiltered chunk has `chunk_byte_size` as its stored size. A block whose address is undefined
/// is unoccupied.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if a signature does not match or the element size is
/// too small for a filtered element or leaves more than 8 bytes for its chunk size,
/// [`FormatError::UnexpectedEof`] if a block runs past the end of `file_data`,
/// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8,
/// [`FormatError::ChecksumMismatch`] if a checksum does not match, [`FormatError::OffsetOverflow`]
/// or [`FormatError::ValueTooLargeForPlatform`] if a position does not fit a `usize`, and the error
/// `visit` returns.
pub fn read_extensible_array_chunks(
    file_data: &[u8],
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    mut visit: impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let os = offset_size as usize;

    // The writer builds from the same geometry.
    let geom = ExtensibleArrayGeometry::from_header(header);

    // The geometry fixes the length of the index block, so the reader bounds it and verifies its
    // checksum before it reads a field.
    let ib_offset = header.index_block_address.get().to_usize()?;
    // The signature, the version, the client ID, and the header address.
    let ib_header_size = 4 + 1 + 1 + offset_size as usize;
    let ib_len = index_block_len(
        offset_size,
        header.idx_blk_elmts as usize,
        ea_elem_stride(header, offset_size),
        geom.direct_dblk_nelmts.len(),
        geom.nsblk_addrs,
    );
    if ib_len > file_data.len() || ib_offset > file_data.len() - ib_len {
        return Err(FormatError::UnexpectedEof {
            expected: ib_offset.saturating_add(ib_len),
            available: file_data.len(),
        });
    }

    let ib = &file_data[ib_offset..ib_offset + ib_len];
    if &ib[0..4] != b"EAIB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array index block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(ib)?;
    // Skip version(1) + client_id(1) + header_address(offset_size)
    let mut pos = ib_offset + ib_header_size;

    let mut global_index = 0usize;
    // A SWMR writer raises the header's count before the dataspace grows, so an interrupted
    // append leaves elements past the dataspace, which the caller drops. The geometry bounds the
    // walk, whatever the count.
    let total_elements = header.max_idx_set.to_usize()?;

    // 1. Read inline elements in index block
    let n_inline = header.idx_blk_elmts as usize;
    for i in 0..n_inline {
        if global_index + i >= total_elements {
            break;
        }
        let (record, consumed) = read_element(
            file_data,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
        )?;
        if let Some(record) = record {
            visit((global_index + i) as u64, record)?;
        }
        pos += consumed;
    }
    global_index += n_inline.min(total_elements);

    // Every element is in the index block.
    if global_index >= total_elements {
        return Ok(());
    }

    // 2. Direct data blocks: their addresses are listed in the index block,
    //    one per entry in `geom.direct_dblk_nelmts`.
    let mut direct_addrs: Vec<StoredAddress> = Vec::with_capacity(geom.direct_dblk_nelmts.len());
    for _ in 0..geom.direct_dblk_nelmts.len() {
        direct_addrs.push(StoredAddress::new(bytes::read_offset(
            file_data,
            pos,
            offset_size,
        )?));
        pos += os;
    }
    for (i, &addr) in direct_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let nelmts = geom.direct_dblk_nelmts[i].to_usize()?;
        if !addr.is_undefined(offset_size) {
            read_data_block_elements(
                file_data,
                addr.get().to_usize()?,
                nelmts,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                &mut visit,
            )?;
        }
        // A block whose address is undefined still spans its slots.
        global_index += nelmts;
    }

    // 3. Super blocks: the remaining `geom.nsblk_addrs` index-block entries are
    //    addresses of on-disk super blocks (`EASB`). Super-block pointer `j`
    //    refers to super block `first_indirect_sblk + j`.
    let mut sblk_addrs: Vec<StoredAddress> = Vec::with_capacity(geom.nsblk_addrs);
    for _ in 0..geom.nsblk_addrs {
        sblk_addrs.push(StoredAddress::new(bytes::read_offset(
            file_data,
            pos,
            offset_size,
        )?));
        pos += os;
    }
    for (j, &sb_addr) in sblk_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let sblk_idx = geom.first_indirect_sblk + j;
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let total_in_sb = (ndblks * dblk_nelmts).to_usize()?;
        if !sb_addr.is_undefined(offset_size) {
            read_super_block(
                file_data,
                sb_addr.get().to_usize()?,
                ndblks.to_usize()?,
                dblk_nelmts.to_usize()?,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                &mut visit,
            )?;
        }
        global_index += total_in_sb;
    }

    Ok(())
}

/// Reads the super block at `sb_offset` in `file_data`, which addresses `ndblks` data blocks of
/// `nelmts_per_dblk` elements each, and the data blocks it addresses.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the super block runs past the end of `file_data`,
/// [`FormatError::ChunkedReadError`] if its signature is not `EASB`,
/// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8,
/// [`FormatError::ChecksumMismatch`] if its checksum does not match,
/// [`FormatError::OffsetOverflow`] or [`FormatError::ValueTooLargeForPlatform`] if a length or an
/// address does not fit a `usize`, and the errors [`read_data_block_elements`] and
/// [`read_paged_data_block`] return.
#[allow(clippy::too_many_arguments)]
fn read_super_block(
    file_data: &[u8],
    sb_offset: usize,
    ndblks: usize,
    nelmts_per_dblk: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let os = offset_size as usize;

    // EASB: signature(4) + version(1) + client_id(1) + header_address(offset_size)
    //       + block_offset(ceil(max_nelmts_bits/8))
    //       + [page-init bitmap, if data blocks are paged]
    //       + data block addresses
    //       + checksum
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let sb_header_size = 4 + 1 + 1 + os + blk_off_size;

    let page_nelmts = 1usize << header.max_dblk_nelmts_bits;
    let sb = SuperBlockGeometry {
        ndblks: ndblks as u64,
        blocks: DataBlockGeometry {
            dblk_nelmts: nelmts_per_dblk as u64,
            page_nelmts: page_nelmts as u64,
        },
    };
    let is_paged = sb.blocks.is_paged();
    let npages = sb.blocks.npages().to_usize()?;
    let bitmap_size = sb.bitmap_size().to_usize()?;

    // The geometry fixes the length of the super block, so the reader bounds it before it reads
    // a field.
    let sb_len = ndblks
        .checked_mul(os)
        .and_then(|addrs| addrs.checked_add(sb_header_size + bitmap_size + 4))
        .ok_or(FormatError::OffsetOverflow {
            offset: ndblks as u64,
            length: os as u64,
        })?;
    if sb_len > file_data.len() || sb_offset > file_data.len() - sb_len {
        return Err(FormatError::UnexpectedEof {
            expected: sb_offset.saturating_add(sb_len),
            available: file_data.len(),
        });
    }

    if &file_data[sb_offset..sb_offset + 4] != b"EASB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array super block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&file_data[sb_offset..sb_offset + sb_len])?;

    let mut pos = sb_offset + sb_header_size;

    let page_bitmap: Vec<u8> = if is_paged {
        let bm = file_data[pos..pos + bitmap_size].to_vec();
        pos += bitmap_size;
        bm
    } else {
        Vec::new()
    };

    // Read data block addresses.
    let mut dblk_addrs: Vec<StoredAddress> = Vec::with_capacity(ndblks);
    for _ in 0..ndblks {
        dblk_addrs.push(StoredAddress::new(bytes::read_offset(
            file_data,
            pos,
            offset_size,
        )?));
        pos += os;
    }

    let mut global_idx = start_index;

    for (db_local, &addr) in dblk_addrs.iter().enumerate() {
        if !addr.is_undefined(offset_size) {
            if is_paged {
                read_paged_data_block(
                    file_data,
                    addr.get().to_usize()?,
                    page_nelmts,
                    npages,
                    db_local,
                    &page_bitmap,
                    header,
                    offset_size,
                    chunk_byte_size,
                    global_idx,
                    total_elements,
                    visit,
                )?
            } else {
                read_data_block_elements(
                    file_data,
                    addr.get().to_usize()?,
                    nelmts_per_dblk,
                    header,
                    offset_size,
                    chunk_byte_size,
                    global_idx,
                    total_elements,
                    visit,
                )?
            };
        }
        global_idx += nelmts_per_dblk;
    }

    Ok(())
}

/// Returns the `(address, length)` spans of the header, the index block, and every super block and
/// data block of the Extensible Array at `ea_base`.
///
/// The spans cover the index alone, and [`read_extensible_array_chunks`] reports the chunks it
/// points at. The walk reads the block addresses as that reader does, and takes each length from
/// the geometry the writer builds from. A block whose address is undefined has no span. The
/// caller checks the spans against the end of the file.
///
/// # Errors
///
/// Returns the errors [`ExtensibleArrayHeader::parse_from_source`] returns,
/// [`FormatError::ChunkedReadError`] if the page exponent is 64 or more or a signature does not
/// match, [`FormatError::ChecksumMismatch`] if the checksum of the index block or a super block
/// does not match, and the error `source` returns if a read fails.
pub fn extensible_array_index_spans(
    source: &(impl MetadataSource + ?Sized),
    ea_base: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let header =
        ExtensibleArrayHeader::parse_from_source(source, ea_base, offset_size, length_size)?;
    let os = offset_size as usize;
    let elem_size = ea_elem_stride(&header, offset_size);
    if header.max_dblk_nelmts_bits >= 64 {
        return Err(FormatError::ChunkedReadError(
            "Extensible Array page exponent out of range".into(),
        ));
    }
    let page_nelmts = 1u64 << header.max_dblk_nelmts_bits;
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);

    // EAHD header block.
    let header_len = ExtensibleArrayHeader::serialized_size(offset_size, length_size) as u64;
    let mut spans = vec![(ea_base.get(), header_len)];

    if header.index_block_address.is_undefined(offset_size) {
        return Ok(spans);
    }

    // EAIB index block, whose length comes from the geometry the writer builds from.
    let geom = ExtensibleArrayGeometry::from_header(&header);
    let ndblk_addrs = geom.direct_dblk_nelmts.len();
    let nsblk_addrs = geom.nsblk_addrs;
    let inline = header.idx_blk_elmts as usize;
    // The signature, the version, the client ID, and the header address.
    let ib_header = 4 + 1 + 1 + os;
    let index_block_len = index_block_len(offset_size, inline, elem_size, ndblk_addrs, nsblk_addrs);
    let ib_addr = header.index_block_address;
    // The read fails for an index block that runs past the end of the file.
    let ib = source.read_metadata_at(ib_addr.get(), index_block_len)?;
    if &ib[..4] != b"EAIB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array index block signature".into(),
        ));
    }
    // The caller frees the spans, so the walk rejects a block whose checksum does not match
    // before it reads an address from it.
    crate::checksum::verify_trailing(&ib)?;
    spans.push((ib_addr.get(), index_block_len as u64));

    // Read the index block's direct data-block addresses, then its super-block
    // addresses, immediately following the inline element slots.
    let mut pos = ib_header + inline * elem_size;
    for &dblk_nelmts in &geom.direct_dblk_nelmts {
        let addr = StoredAddress::new(bytes::read_offset(&ib, pos, offset_size)?);
        pos += os;
        if addr.is_undefined(offset_size) {
            continue;
        }
        spans.push((
            addr.get(),
            data_block_len(
                DataBlockGeometry {
                    dblk_nelmts,
                    page_nelmts,
                },
                elem_size,
                offset_size,
                blk_off_size,
            ),
        ));
    }
    for j in 0..nsblk_addrs {
        let addr = StoredAddress::new(bytes::read_offset(&ib, pos, offset_size)?);
        pos += os;
        if addr.is_undefined(offset_size) {
            continue;
        }
        let sb = geom.super_block_at(j, page_nelmts);
        spans.push((addr.get(), super_block_len(sb, offset_size, blk_off_size)));
        easb_data_block_spans(
            source,
            addr,
            sb,
            offset_size,
            blk_off_size,
            elem_size,
            &mut spans,
        )?;
    }

    Ok(spans)
}

/// Appends to `spans` the span of each data block the super block at `sb_addr` addresses, and
/// reads the addresses as [`read_super_block`] does.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the signature is not `EASB`,
/// [`FormatError::ChecksumMismatch`] if the checksum does not match, and the error `source`
/// returns if a read fails.
fn easb_data_block_spans(
    source: &(impl MetadataSource + ?Sized),
    sb_addr: StoredAddress,
    sb: SuperBlockGeometry,
    offset_size: u8,
    blk_off_size: usize,
    elem_size: usize,
    spans: &mut Vec<(u64, u64)>,
) -> Result<(), FormatError> {
    let os = offset_size as usize;
    // The signature, the version, the client ID, the header address, and the block offset.
    let sb_header = 4 + 1 + 1 + os + blk_off_size;
    // The caller frees the spans, so the walk verifies the checksum of the whole block before it
    // reads an address.
    let sb_len = super_block_len(sb, offset_size, blk_off_size).to_usize()?;
    let block = source.read_metadata_at(sb_addr.get(), sb_len)?;
    if &block[..4] != b"EASB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array super block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&block)?;

    let mut pos = sb_header + sb.bitmap_size().to_usize()?;
    for _ in 0..sb.ndblks {
        let addr = StoredAddress::new(bytes::read_offset(&block, pos, offset_size)?);
        pos += os;
        if addr.is_undefined(offset_size) {
            continue;
        }
        spans.push((
            addr.get(),
            data_block_len(sb.blocks, elem_size, offset_size, blk_off_size),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming traversal (read each block from a `MetadataSource` on demand)
// ---------------------------------------------------------------------------

/// Reads the Extensible Array `header` describes from `source`, as [`read_extensible_array_chunks`]
/// reads it from a buffer.
///
/// Each block takes one read.
///
/// # Errors
///
/// Returns the errors [`read_extensible_array_chunks`] returns, and the error `source` returns if
/// a read fails.
pub fn read_extensible_array_chunks_from_source(
    source: &(impl MetadataSource + ?Sized),
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    mut visit: impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let os = offset_size as usize;

    // See `read_extensible_array_chunks` on why the dataspace does not bound
    // this count.
    let total_elements = header.max_idx_set.to_usize()?;

    let geom = ExtensibleArrayGeometry::from_header(header);
    let elem_stride = ea_elem_stride(header, offset_size);

    // Read the whole index block: header + inline element slots + direct
    // data-block addresses + super-block addresses + its checksum.
    let ib_header_size = 4 + 1 + 1 + os;
    let n_inline = header.idx_blk_elmts as usize;
    let ndirect = geom.direct_dblk_nelmts.len();
    let nsblk = geom.nsblk_addrs;
    let inline_bytes = n_inline
        .checked_mul(elem_stride)
        .ok_or(FormatError::OffsetOverflow {
            offset: n_inline as u64,
            length: elem_stride as u64,
        })?;
    let addr_bytes = (ndirect + nsblk)
        .checked_mul(os)
        .ok_or(FormatError::OffsetOverflow {
            offset: (ndirect + nsblk) as u64,
            length: os as u64,
        })?;
    let ib_len = ib_header_size + inline_bytes + addr_bytes + 4;
    let ib = source.read_metadata_at(header.index_block_address.get(), ib_len)?;
    if &ib[0..4] != b"EAIB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array index block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&ib)?;

    let mut pos = ib_header_size;
    let mut global_index = 0usize;

    // 1. Inline elements stored directly in the index block.
    for i in 0..n_inline {
        if global_index + i >= total_elements {
            break;
        }
        let (record, consumed) = read_element(
            &ib,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
        )?;
        if let Some(record) = record {
            visit((global_index + i) as u64, record)?;
        }
        pos += consumed;
    }
    global_index += n_inline.min(total_elements);
    if global_index >= total_elements {
        return Ok(());
    }

    // After all inline slots, `pos` sits at the direct data-block addresses.
    let mut direct_addrs: Vec<StoredAddress> = Vec::with_capacity(ndirect);
    for _ in 0..ndirect {
        direct_addrs.push(StoredAddress::new(bytes::read_offset(
            &ib,
            pos,
            offset_size,
        )?));
        pos += os;
    }
    for (i, &addr) in direct_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let nelmts = geom.direct_dblk_nelmts[i].to_usize()?;
        if !addr.is_undefined(offset_size) {
            read_data_block_elements_from_source(
                source,
                addr,
                nelmts,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                &mut visit,
            )?;
        }
        global_index += nelmts;
    }

    // 3. Super-block addresses.
    let mut sblk_addrs: Vec<StoredAddress> = Vec::with_capacity(nsblk);
    for _ in 0..nsblk {
        sblk_addrs.push(StoredAddress::new(bytes::read_offset(
            &ib,
            pos,
            offset_size,
        )?));
        pos += os;
    }
    for (j, &sb_addr) in sblk_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let sblk_idx = geom.first_indirect_sblk + j;
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let total_in_sb = (ndblks * dblk_nelmts).to_usize()?;
        if !sb_addr.is_undefined(offset_size) {
            read_super_block_from_source(
                source,
                sb_addr,
                ndblks.to_usize()?,
                dblk_nelmts.to_usize()?,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                &mut visit,
            )?;
        }
        global_index += total_in_sb;
    }

    Ok(())
}

/// Reads the unpaged data block at `db_address` in `source`, as [`read_data_block_elements`]
/// reads it from a buffer.
///
/// # Errors
///
/// Returns the errors [`read_data_block_elements`] returns, and the error `source` returns if a
/// read fails.
#[allow(clippy::too_many_arguments)]
fn read_data_block_elements_from_source(
    source: &(impl MetadataSource + ?Sized),
    db_address: StoredAddress,
    nelmts: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let os = offset_size as usize;
    let db_header_size = 4 + 1 + 1 + os;
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let limit = total_elements.saturating_sub(start_index).min(nelmts);

    // The whole block, since its checksum covers the elements past `limit` too. An unpaged block
    // holds at most one page of elements.
    let region_len = eadb_extent(nelmts, header, offset_size, blk_off_size)?;
    let block = source.read_metadata_at(db_address.get(), region_len)?;
    if &block[0..4] != b"EADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array data block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&block)?;

    let mut pos = db_header_size + blk_off_size;
    for i in 0..limit {
        let (record, consumed) = read_element(
            &block,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
        )?;
        if let Some(record) = record {
            visit((start_index + i) as u64, record)?;
        }
        pos += consumed;
    }
    Ok(())
}

/// Reads the paged data block at `db_address` in `source`, as [`read_paged_data_block`] reads it
/// from a buffer.
///
/// The read runs from the start of the block through the last page the bitmap marks initialized.
///
/// # Errors
///
/// Returns the errors [`read_paged_data_block`] returns, and the error `source` returns if a read
/// fails.
#[allow(clippy::too_many_arguments)]
fn read_paged_data_block_from_source(
    source: &(impl MetadataSource + ?Sized),
    db_address: StoredAddress,
    page_nelmts: usize,
    npages: usize,
    db_local_idx: usize,
    page_bitmap: &[u8],
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let db_header_size = 4 + 1 + 1 + offset_size as usize + blk_off_size + 4;
    let elem_stride = ea_elem_stride(header, offset_size);
    let page_stride = page_nelmts
        .checked_mul(elem_stride)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(FormatError::OffsetOverflow {
            offset: page_nelmts as u64,
            length: elem_stride as u64,
        })?;

    let mut init_pages = 0usize;
    for page in 0..npages {
        if page_is_initialized(page_bitmap, db_local_idx * npages + page) {
            init_pages = page + 1;
        }
    }
    let pages_bytes = init_pages
        .checked_mul(page_stride)
        .ok_or(FormatError::OffsetOverflow {
            offset: init_pages as u64,
            length: page_stride as u64,
        })?;
    // A writer allocates the space of every page with the block, so a short read is a truncated
    // block.
    let region_len = db_header_size + pages_bytes;
    let block = source.read_metadata_at(db_address.get(), region_len)?;
    if block.len() < 4 || &block[0..4] != b"EADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array data block signature".into(),
        ));
    }
    // A paged block checksums its header on its own, then each page separately.
    crate::checksum::verify_trailing(&block[..db_header_size])?;

    let mut pos = db_header_size;
    for page in 0..npages {
        let global_page = db_local_idx * npages + page;
        if !page_is_initialized(page_bitmap, global_page) {
            pos += page_stride;
            continue;
        }
        // See `read_paged_data_block`: the page checksum covers every slot.
        crate::checksum::verify_trailing(&block[pos..pos + page_stride])?;
        let page_start = start_index + page * page_nelmts;
        let limit = total_elements.saturating_sub(page_start).min(page_nelmts);
        for i in 0..limit {
            let (record, consumed) = read_element(
                &block,
                pos,
                header.client_id,
                header.element_size,
                offset_size,
                chunk_byte_size,
            )?;
            if let Some(record) = record {
                visit((page_start + i) as u64, record)?;
            }
            pos += consumed;
        }
        if limit < page_nelmts {
            break;
        }
        pos += 4; // page checksum
    }
    Ok(())
}

/// Reads the super block at `sb_address` in `source` and the data blocks it addresses, as
/// [`read_super_block`] reads them from a buffer.
///
/// # Errors
///
/// Returns the errors [`read_super_block`] returns, and the error `source` returns if a read
/// fails.
#[allow(clippy::too_many_arguments)]
fn read_super_block_from_source(
    source: &(impl MetadataSource + ?Sized),
    sb_address: StoredAddress,
    ndblks: usize,
    nelmts_per_dblk: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    visit: &mut impl FnMut(u64, ChunkRecord) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let os = offset_size as usize;
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let sb_header_size = 4 + 1 + 1 + os + blk_off_size;

    let page_nelmts = 1usize << header.max_dblk_nelmts_bits;
    let sb = SuperBlockGeometry {
        ndblks: ndblks as u64,
        blocks: DataBlockGeometry {
            dblk_nelmts: nelmts_per_dblk as u64,
            page_nelmts: page_nelmts as u64,
        },
    };
    let is_paged = sb.blocks.is_paged();
    let npages = sb.blocks.npages().to_usize()?;
    let bitmap_size = sb.bitmap_size().to_usize()?;

    // Header + (optional) page-init bitmap + data-block addresses + checksum.
    let addr_bytes = ndblks.checked_mul(os).ok_or(FormatError::OffsetOverflow {
        offset: ndblks as u64,
        length: os as u64,
    })?;
    let region_len = sb_header_size + bitmap_size + addr_bytes + 4;
    let block = source.read_metadata_at(sb_address.get(), region_len)?;
    if &block[0..4] != b"EASB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array super block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&block)?;

    let mut pos = sb_header_size;
    let page_bitmap: Vec<u8> = if is_paged {
        let bm = block[pos..pos + bitmap_size].to_vec();
        pos += bitmap_size;
        bm
    } else {
        Vec::new()
    };

    let mut dblk_addrs: Vec<StoredAddress> = Vec::with_capacity(ndblks);
    for _ in 0..ndblks {
        dblk_addrs.push(StoredAddress::new(bytes::read_offset(
            &block,
            pos,
            offset_size,
        )?));
        pos += os;
    }

    let mut global_idx = start_index;
    for (db_local, &addr) in dblk_addrs.iter().enumerate() {
        if !addr.is_undefined(offset_size) {
            if is_paged {
                read_paged_data_block_from_source(
                    source,
                    addr,
                    page_nelmts,
                    npages,
                    db_local,
                    &page_bitmap,
                    header,
                    offset_size,
                    chunk_byte_size,
                    global_idx,
                    total_elements,
                    visit,
                )?
            } else {
                read_data_block_elements_from_source(
                    source,
                    addr,
                    nelmts_per_dblk,
                    header,
                    offset_size,
                    chunk_byte_size,
                    global_idx,
                    total_elements,
                    visit,
                )?
            };
        }
        global_idx += nelmts_per_dblk;
    }

    Ok(())
}

/// The slots of an Extensible Array that hold a chunk.
///
/// [`extensible_array_stats`] counts the blocks that hold a chunk and [`build_extensible_array_at`]
/// builds them, and the length the first computes must equal the length the second builds. Both
/// test the same `SlotOccupancy`.
#[derive(Clone, Copy)]
pub enum SlotOccupancy<'a> {
    /// Slots `0..len` are occupied, and no slot past them.
    Dense(u64),
    /// The occupied slots of an index.
    Sparse(&'a IndexSlots<'a>),
    /// The occupied slots as slot numbers in ascending order, for a caller that sizes an index
    /// before its chunks have addresses.
    Slots(&'a [u64]),
}

impl SlotOccupancy<'_> {
    /// Returns whether any of the `count` slots from `start` holds a chunk, and so whether the
    /// block over them is allocated.
    ///
    /// The C library allocates a data block when it writes the first chunk into it (`H5EA.c`,
    /// HDF5 2.2.0), so an index grows with its chunks and not with the maximum shape.
    fn any_occupied(&self, start: u64, count: u64) -> bool {
        match self {
            Self::Dense(len) => start < *len,
            Self::Sparse(slots) => slots.any_occupied(start, count),
            Self::Slots(sorted) => {
                let end = start.saturating_add(count);
                let from = sorted.partition_point(|&s| s < start);
                sorted.get(from).is_some_and(|&s| s < end)
            }
        }
    }
}

/// Appends `addr` to `buf` as a little-endian field of `offset_size` bytes.
pub fn write_stored_address(buf: &mut Vec<u8>, addr: StoredAddress, offset_size: OffsetWidth) {
    bytes::write_offset(buf, addr.get(), offset_size);
}

/// Appends a block's offset in the array's element space as a little-endian field of
/// `blk_off_size` bytes, the width `ceil(max_nelmts_bits / 8)` the header gives it.
///
/// # Panics
///
/// Panics if `block_offset` does not fit `blk_off_size` bytes.
fn write_block_offset(buf: &mut Vec<u8>, block_offset: u64, blk_off_size: usize) {
    let bytes = block_offset.to_le_bytes();
    assert!(
        blk_off_size <= bytes.len() && bytes[blk_off_size..].iter().all(|&byte| byte == 0),
        "block offset {block_offset} does not fit a {blk_off_size}-byte field"
    );
    buf.extend_from_slice(&bytes[..blk_off_size]);
}

/// Builds the data block (`EADB`) of the elements in slots `elem_start..elem_start + dblk_nelmts`,
/// with `block_offset_rel` as its block offset.
///
/// A slot no chunk occupies stores the undefined address. A block of more than `page_nelmts`
/// elements is paged: its prefix has its own checksum, and every page follows with a checksum
/// each.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the stored size of a chunk does not fit the chunk size
/// field.
///
/// # Panics
///
/// Panics if `block_offset_rel` does not fit `blk_off_size` bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_data_block(
    slots: &IndexSlots<'_>,
    elem_start: usize,
    dblk_nelmts: usize,
    block_offset_rel: u64,
    ea_address: StoredAddress,
    offset_size: OffsetWidth,
    has_filters: bool,
    chunk_size_bytes: usize,
    client_id: u8,
    page_nelmts: usize,
    blk_off_size: usize,
) -> Result<Vec<u8>, FormatError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EADB");
    buf.push(0); // version
    buf.push(client_id);
    write_stored_address(&mut buf, ea_address, offset_size);
    write_block_offset(&mut buf, block_offset_rel, blk_off_size);

    // The counts are `usize` loop bounds, and the page arithmetic below stays in `usize`.
    let blocks = DataBlockGeometry {
        dblk_nelmts: dblk_nelmts as u64,
        page_nelmts: page_nelmts as u64,
    };
    if !blocks.is_paged() {
        // Non-paged: elements inline, single checksum.
        for slot in 0..dblk_nelmts {
            if let Some(chunk) = slots.at(elem_start + slot) {
                chunk_record::write_chunk_element(
                    &mut buf,
                    chunk,
                    offset_size,
                    has_filters,
                    chunk_size_bytes,
                )?;
            } else {
                chunk_record::write_undefined_element(
                    &mut buf,
                    offset_size,
                    has_filters,
                    chunk_size_bytes,
                );
            }
        }
        let cks = checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&cks.to_le_bytes());
        Ok(buf)
    } else {
        // Paged: the prefix and its checksum, then every page, as the C library allocates them
        // (`H5EAdblock.c`, HDF5 2.2.0).
        let header_cks = checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&header_cks.to_le_bytes());

        let npages = dblk_nelmts / page_nelmts;
        for page in 0..npages {
            let page_start = elem_start + page * page_nelmts;
            let mut page_buf = Vec::new();
            for slot in 0..page_nelmts {
                if let Some(chunk) = slots.at(page_start + slot) {
                    chunk_record::write_chunk_element(
                        &mut page_buf,
                        chunk,
                        offset_size,
                        has_filters,
                        chunk_size_bytes,
                    )?;
                } else {
                    chunk_record::write_undefined_element(
                        &mut page_buf,
                        offset_size,
                        has_filters,
                        chunk_size_bytes,
                    );
                }
            }
            let page_cks = checksum::jenkins_lookup3(&page_buf);
            page_buf.extend_from_slice(&page_cks.to_le_bytes());
            buf.extend_from_slice(&page_buf);
        }
        Ok(buf)
    }
}

/// Builds the super block (`EASB`) that addresses `dblk_addrs`, with `block_offset_rel` as its
/// block offset.
///
/// A non-empty `page_bitmap` is the page-init bitmap of paged data blocks, which the super block
/// stores between its block offset and its data block addresses.
///
/// # Panics
///
/// Panics if `block_offset_rel` does not fit `blk_off_size` bytes.
pub fn encode_super_block(
    ea_address: StoredAddress,
    block_offset_rel: u64,
    page_bitmap: &[u8],
    dblk_addrs: &[StoredAddress],
    offset_size: OffsetWidth,
    blk_off_size: usize,
    client_id: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EASB");
    buf.push(0); // version
    buf.push(client_id);
    write_stored_address(&mut buf, ea_address, offset_size);
    write_block_offset(&mut buf, block_offset_rel, blk_off_size);
    buf.extend_from_slice(page_bitmap);
    for &addr in dblk_addrs {
        write_stored_address(&mut buf, addr, offset_size);
    }
    let cks = checksum::jenkins_lookup3(&buf);
    buf.extend_from_slice(&cks.to_le_bytes());
    buf
}

/// Returns the length in bytes of an index block (`EAIB`): the prefix, `inline_elmts` elements,
/// the data block and super block addresses, and the checksum.
///
/// [`build_extensible_array_at`] and [`extensible_array_index_spans`] both size the index block
/// with it.
pub(crate) fn index_block_len(
    offset_size: u8,
    inline_elmts: usize,
    elem_size: usize,
    ndblk_addrs: usize,
    nsblk_addrs: usize,
) -> usize {
    let os = offset_size as usize;
    4 + 1 + 1 + os // signature + version + client id + header address
        + inline_elmts * elem_size // inline element slots (always all written)
        + ndblk_addrs * os // direct data-block addresses
        + nsblk_addrs * os // super-block addresses
        + 4 // checksum
}

/// The six statistics of an Extensible Array header, in the order the header stores them.
///
/// An in-place writer that appends to an array rewrites them, and [`extensible_array_layout`] takes
/// the length of the blocks after the index block from `super_blk_size` and `data_blk_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensibleArrayStats {
    /// The number of super blocks allocated.
    pub nsuper_blks: u64,
    /// The total length in bytes of the super blocks.
    pub super_blk_size: u64,
    /// The number of data blocks allocated.
    pub ndata_blks: u64,
    /// The total length in bytes of the data blocks.
    pub data_blk_size: u64,
    /// One more than the highest element index set, the "Max Index Set" field.
    pub max_idx_set: u64,
    /// The number of element slots in the index block and the allocated data blocks.
    pub nelmts: u64,
}

/// Returns the length in bytes of a data block (`EADB`) with the geometry `blocks`, paged or not.
pub(crate) fn data_block_len(
    blocks: DataBlockGeometry,
    elem_size: usize,
    offset_size: u8,
    blk_off_size: usize,
) -> u64 {
    let prefix = (4 + 1 + 1 + offset_size as u64) + blk_off_size as u64;
    if blocks.is_paged() {
        // Paged: the prefix and its checksum, then every page and its checksum.
        prefix + 4 + blocks.npages() * (blocks.page_nelmts * elem_size as u64 + 4)
    } else {
        prefix + blocks.dblk_nelmts * elem_size as u64 + 4
    }
}

/// Returns the length in bytes of a super block (`EASB`): the prefix, the page-init bitmap of
/// paged data blocks, the data block addresses, and the checksum.
///
/// The length is computed in `u64`, so the counts are not narrowed to a 32-bit `usize` before
/// they are multiplied.
pub(crate) fn super_block_len(
    geom: SuperBlockGeometry,
    offset_size: u8,
    blk_off_size: usize,
) -> u64 {
    let os = offset_size as u64;
    let header = 4 + 1 + 1 + os + blk_off_size as u64;
    header + geom.bitmap_size() + geom.ndblks * os + 4
}

/// Returns the header statistics of the array whose occupied slots are `occupancy`, from the
/// blocks [`build_extensible_array_at`] allocates for them.
pub fn extensible_array_stats(
    geom: &ExtensibleArrayGeometry,
    idx_blk_elmts: u64,
    elem_size: usize,
    page_nelmts: u64,
    offset_size: OffsetWidth,
    blk_off_size: usize,
    max_idx_set: u64,
    occupancy: SlotOccupancy<'_>,
) -> ExtensibleArrayStats {
    let offset_size = offset_size.get();
    let mut s = ExtensibleArrayStats {
        nsuper_blks: 0,
        super_blk_size: 0,
        ndata_blks: 0,
        data_blk_size: 0,
        max_idx_set,
        nelmts: idx_blk_elmts,
    };
    let mut elem = idx_blk_elmts;
    for &dn in &geom.direct_dblk_nelmts {
        if occupancy.any_occupied(elem, dn) {
            let blocks = DataBlockGeometry {
                dblk_nelmts: dn,
                page_nelmts,
            };
            s.ndata_blks += 1;
            s.data_blk_size += data_block_len(blocks, elem_size, offset_size, blk_off_size);
            s.nelmts += dn;
        }
        elem += dn;
    }
    for j in 0..geom.nsblk_addrs {
        let sb = geom.super_block_at(j, page_nelmts);
        let (ndblks, dn) = (sb.ndblks, sb.blocks.dblk_nelmts);
        let span = ndblks * dn;
        if occupancy.any_occupied(elem, span) {
            s.nsuper_blks += 1;
            s.super_blk_size += super_block_len(sb, offset_size, blk_off_size);
            let mut le = elem;
            for _ in 0..ndblks {
                if occupancy.any_occupied(le, dn) {
                    s.ndata_blks += 1;
                    s.data_blk_size +=
                        data_block_len(sb.blocks, elem_size, offset_size, blk_off_size);
                    s.nelmts += dn;
                }
                le += dn;
            }
        }
        elem += span;
    }
    s
}

/// The layout of an Extensible Array that does not depend on its address: the element encoding,
/// the geometry, and the lengths.
///
/// [`extensible_array_len`] computes the layout without building the array.
/// [`build_extensible_array_at`] builds from the geometry of the layout and walks the blocks
/// itself, and a debug assertion checks its length against the layout's.
pub struct ExtensibleArrayLayout {
    encoding: ChunkElementEncoding,
    max_nelmts_bits: u8,
    idx_blk_elmts: u8,
    min_dblk_nelmts: u8,
    super_blk_min_data_ptrs: u8,
    max_dblk_nelmts_bits: u8,
    geom: ExtensibleArrayGeometry,
    page_nelmts: usize,
    /// The width in bytes of a block offset.
    blk_off_size: usize,
    /// The number of elements in the index block, `idx_blk_elmts`.
    inline: usize,
    /// The length of the header in bytes.
    header_len: usize,
    index_block_len: usize,
    /// The header statistics.
    pub stats: ExtensibleArrayStats,
    /// The length of the array in bytes.
    total_len: u64,
}

/// Returns the layout of the Extensible Array of `num_slots` slots whose occupied slots are
/// `occupancy`.
pub fn extensible_array_layout(
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> ExtensibleArrayLayout {
    let encoding = chunk_record::chunk_element_encoding(chunk_bytes, offset_size, has_filters);
    let ChunkElementEncoding {
        elem_size,
        client_id,
        ..
    } = encoding;

    let (
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_data_ptrs,
        max_dblk_nelmts_bits,
    ) = (
        EA_MAX_NELMTS_BITS,
        EA_IDX_BLK_ELMTS,
        EA_MIN_DBLK_NELMTS,
        EA_SUPER_BLK_MIN_DATA_PTRS,
        EA_MAX_DBLK_NELMTS_BITS,
    );

    // The reader derives its geometry the same way.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    let geom_header = ExtensibleArrayHeader {
        client_id,
        element_size: elem_size as u8,
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_data_ptrs,
        max_dblk_nelmts_bits,
        max_idx_set: 0,
        index_block_address: StoredAddress::new(0),
    };
    let geom = ExtensibleArrayGeometry::from_header(&geom_header);
    let page_nelmts = 1usize << max_dblk_nelmts_bits;
    let blk_off_size = (max_nelmts_bits as usize).div_ceil(8);
    let inline = idx_blk_elmts as usize;

    let header_len = ExtensibleArrayHeader::serialized_size(offset_size.get(), length_size.get());
    let index_block_len = index_block_len(
        offset_size.get(),
        inline,
        elem_size,
        geom.direct_dblk_nelmts.len(),
        geom.nsblk_addrs,
    );

    let stats = extensible_array_stats(
        &geom,
        idx_blk_elmts as u64,
        elem_size,
        page_nelmts as u64,
        offset_size,
        blk_off_size,
        num_slots,
        occupancy,
    );
    let total_len =
        (header_len + index_block_len) as u64 + stats.data_blk_size + stats.super_blk_size;

    ExtensibleArrayLayout {
        encoding,
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_data_ptrs,
        max_dblk_nelmts_bits,
        geom,
        page_nelmts,
        blk_off_size,
        inline,
        header_len,
        index_block_len,
        stats,
        total_len,
    }
}

/// Returns the length in bytes of the Extensible Array [`build_extensible_array_at`] builds for
/// `slots`, without building it.
///
/// A caller that reserves the span of an array before writing it, such as an in-place writer
/// that places an array in freed space, takes the length from here.
pub fn extensible_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> u64 {
    extensible_array_layout(
        SlotOccupancy::Sparse(slots),
        slots.len() as u64,
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    )
    .total_len
}

/// Builds the Extensible Array for `slots` at `ea_address`: the header, the index block, and the
/// blocks after it.
///
/// The creation parameters are the C library's defaults. The index block holds the first
/// `idx_blk_elmts` elements and addresses the data blocks and super blocks after them. A block
/// none of whose slots holds a chunk is not allocated, and its address is undefined. A data block
/// of more than a page of elements is paged, and its super block marks the pages that hold a
/// chunk initialized.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the stored size of a chunk does not fit the chunk size
/// field, and [`FormatError::ValueTooLargeForPlatform`] if the length of the blocks or the count of
/// a block that holds a chunk does not fit a `usize`.
pub fn build_extensible_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    ea_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    let max_idx_set = slots.len();

    let layout = extensible_array_layout(
        SlotOccupancy::Sparse(slots),
        slots.len() as u64,
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    );
    let ChunkElementEncoding {
        chunk_size_bytes,
        elem_size,
        client_id,
    } = layout.encoding;
    let ExtensibleArrayLayout {
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_data_ptrs,
        max_dblk_nelmts_bits,
        ref geom,
        page_nelmts,
        blk_off_size,
        inline,
        header_len,
        index_block_len,
        ..
    } = layout;

    let index_block_address = ea_address.offset(header_len as u64);
    let body_base = index_block_address.offset(index_block_len as u64);

    let undef_addr = StoredAddress::undefined(offset_size.get());

    // ---- Build the body (direct data blocks, then super blocks) -----------
    // Each block's address is computed from `body_base`, so the body can be
    // built before the index block that references it.
    let mut body: Vec<u8> =
        Vec::with_capacity((layout.stats.data_blk_size + layout.stats.super_blk_size).to_usize()?);
    let mut direct_addrs: Vec<StoredAddress> = Vec::with_capacity(geom.direct_dblk_nelmts.len());
    let mut sblk_addrs: Vec<StoredAddress> = Vec::with_capacity(geom.nsblk_addrs);

    // The header statistics.
    let mut ndata_blks: u64 = 0;
    let mut data_blk_size: u64 = 0;
    let mut nsuper_blks: u64 = 0;
    let mut super_blk_size: u64 = 0;
    // `nelmts`: the slots of the index block and of every allocated data block.
    let mut alloc_slots: u64 = inline as u64;

    // The slot past the inline slots. The array spans `extensible_array_capacity()` slots, more
    // than a 32-bit `usize` holds, so the cursor and the block spans are `u64`, and a count is
    // narrowed only for a block that holds a chunk.
    let mut elem_cursor: u64 = inline as u64;

    // The data blocks the index block addresses. A block is written only where it holds a chunk,
    // as in `extensible_array_stats`, which applies the same test.
    let occupancy = SlotOccupancy::Sparse(slots);
    for &dblk_nelmts in &geom.direct_dblk_nelmts {
        if !occupancy.any_occupied(elem_cursor, dblk_nelmts) {
            direct_addrs.push(undef_addr);
            elem_cursor += dblk_nelmts;
            continue;
        }
        let addr = body_base.offset(body.len() as u64);
        let db_bytes = encode_data_block(
            slots,
            elem_cursor.to_usize()?,
            dblk_nelmts.to_usize()?,
            elem_cursor - inline as u64,
            ea_address,
            offset_size,
            has_filters,
            chunk_size_bytes,
            client_id,
            page_nelmts,
            blk_off_size,
        )?;
        ndata_blks += 1;
        data_blk_size += db_bytes.len() as u64;
        alloc_slots += dblk_nelmts;
        body.extend_from_slice(&db_bytes);
        direct_addrs.push(addr);
        elem_cursor += dblk_nelmts;
    }

    // Super blocks: addresses stored in the index block. Super-block pointer `j`
    // refers to super block `first_indirect_sblk + j`.
    for j in 0..geom.nsblk_addrs {
        let sblk_idx = geom.first_indirect_sblk + j;
        // The span of a super block can exceed a 32-bit `usize`, so it stays `u64`.
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let sb_span = ndblks * dblk_nelmts;
        if !occupancy.any_occupied(elem_cursor, sb_span) {
            sblk_addrs.push(undef_addr);
            elem_cursor += sb_span;
            continue;
        }

        // This super block holds a chunk, so the number of chunks bounds its counts.
        let sb = geom.super_block_at(j, page_nelmts as u64);
        let is_paged = sb.blocks.is_paged();
        let npages = sb.blocks.npages();
        let sb_block_offset = elem_cursor - inline as u64;
        let mut page_bitmap = vec![0u8; sb.bitmap_size().to_usize()?];

        let mut sb_dblk_addrs: Vec<StoredAddress> = Vec::with_capacity(ndblks.to_usize()?);
        let mut local_elem = elem_cursor;
        for db_local in 0..ndblks {
            if !occupancy.any_occupied(local_elem, dblk_nelmts) {
                sb_dblk_addrs.push(undef_addr);
                local_elem += dblk_nelmts;
                continue;
            }
            let addr = body_base.offset(body.len() as u64);
            let db_bytes = encode_data_block(
                slots,
                local_elem.to_usize()?,
                dblk_nelmts.to_usize()?,
                local_elem - inline as u64,
                ea_address,
                offset_size,
                has_filters,
                chunk_size_bytes,
                client_id,
                page_nelmts,
                blk_off_size,
            )?;
            ndata_blks += 1;
            data_blk_size += db_bytes.len() as u64;
            alloc_slots += dblk_nelmts;
            body.extend_from_slice(&db_bytes);
            sb_dblk_addrs.push(addr);
            if is_paged {
                // Marks initialized each page that holds a chunk, as the C library marks it
                // (`H5EA.c`, HDF5 2.2.0). No test checks these bits: the tests read back an array
                // with gaps in both libraries, which also passes with every page marked.
                let base = local_elem.to_usize()?;
                for p in 0..npages.to_usize()? {
                    let page_start = base + p * page_nelmts;
                    if !(0..page_nelmts).any(|s| slots.at(page_start + s).is_some()) {
                        continue;
                    }
                    let global_page = (db_local * npages).to_usize()? + p;
                    page_bitmap[global_page / 8] |= 0x80 >> (global_page % 8);
                }
            }
            local_elem += dblk_nelmts;
        }

        let super_block_address = body_base.offset(body.len() as u64);
        let super_block = encode_super_block(
            ea_address,
            sb_block_offset,
            &page_bitmap,
            &sb_dblk_addrs,
            offset_size,
            blk_off_size,
            client_id,
        );
        nsuper_blks += 1;
        super_blk_size += super_block.len() as u64;
        body.extend_from_slice(&super_block);
        sblk_addrs.push(super_block_address);

        elem_cursor += sb_span;
    }

    // ---- Build the header (EAHD) ------------------------------------------
    let write_length = |buf: &mut Vec<u8>, val: u64| bytes::write_length(buf, val, length_size);

    let mut header = Vec::with_capacity(header_len);
    header.extend_from_slice(b"EAHD");
    header.push(0); // version
    header.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    header.push(elem_size as u8);
    header.push(max_nelmts_bits);
    header.push(idx_blk_elmts);
    header.push(min_dblk_nelmts);
    header.push(super_blk_min_data_ptrs);
    header.push(max_dblk_nelmts_bits);

    // 6 statistics, in the C library's order:
    //   [0] `nsuper_blks`   [1] `super_blk_size`   [2] `ndata_blks`
    //   [3] `data_blk_size` [4] `max_idx_set`      [5] `nelmts`
    write_length(&mut header, nsuper_blks);
    write_length(&mut header, super_blk_size);
    write_length(&mut header, ndata_blks);
    write_length(&mut header, data_blk_size);
    write_length(&mut header, max_idx_set.to_u64()); // one past the last slot
    write_length(&mut header, alloc_slots); // nelmts (allocated slots)

    write_stored_address(&mut header, index_block_address, offset_size);

    let header_checksum = checksum::jenkins_lookup3(&header);
    header.extend_from_slice(&header_checksum.to_le_bytes());
    debug_assert_eq!(header.len(), header_len);

    // ---- Build the index block (EAIB) -------------------------------------
    let mut index_block = Vec::with_capacity(index_block_len);
    index_block.extend_from_slice(b"EAIB");
    index_block.push(0); // version
    index_block.push(client_id);
    write_stored_address(&mut index_block, ea_address, offset_size);

    // Inline elements: always `idx_blk_elmts` slots, the unused ones undefined.
    #[allow(clippy::needless_range_loop)]
    for i in 0..inline {
        if let Some(chunk) = slots.at(i) {
            chunk_record::write_chunk_element(
                &mut index_block,
                chunk,
                offset_size,
                has_filters,
                chunk_size_bytes,
            )?;
        } else {
            chunk_record::write_undefined_element(
                &mut index_block,
                offset_size,
                has_filters,
                chunk_size_bytes,
            );
        }
    }
    // Direct data block addresses, then super block addresses.
    for &addr in &direct_addrs {
        write_stored_address(&mut index_block, addr, offset_size);
    }
    for &addr in &sblk_addrs {
        write_stored_address(&mut index_block, addr, offset_size);
    }

    let index_block_checksum = checksum::jenkins_lookup3(&index_block);
    index_block.extend_from_slice(&index_block_checksum.to_le_bytes());
    debug_assert_eq!(index_block.len(), index_block_len);

    let mut combined = header;
    combined.extend_from_slice(&index_block);
    combined.extend_from_slice(&body);
    // The caller reserved the length `extensible_array_len` returns, and a longer array would
    // overlap the next object.
    debug_assert_eq!(
        combined.len() as u64,
        layout.total_len,
        "an extensible array must fill the length its layout promised"
    );
    Ok(combined)
}

/// Returns the capacity in slots of an Extensible Array [`build_extensible_array_at`] builds: the
/// elements of the index block, the data blocks it addresses, and the data blocks of every super
/// block.
///
/// The capacity is 8,589,934,580 slots: 4 in the index block, 240 in six data blocks, and
/// 8,589,934,336 in 25 super blocks.
pub fn extensible_array_capacity() -> u64 {
    let geom_header = ExtensibleArrayHeader {
        client_id: 0,
        element_size: 8,
        max_nelmts_bits: EA_MAX_NELMTS_BITS,
        idx_blk_elmts: EA_IDX_BLK_ELMTS,
        min_dblk_nelmts: EA_MIN_DBLK_NELMTS,
        super_blk_min_data_ptrs: EA_SUPER_BLK_MIN_DATA_PTRS,
        max_dblk_nelmts_bits: EA_MAX_DBLK_NELMTS_BITS,
        max_idx_set: 0,
        index_block_address: StoredAddress::new(0),
    };
    let geom = ExtensibleArrayGeometry::from_header(&geom_header);
    let direct: u64 = geom.direct_dblk_nelmts.iter().sum();
    let indirect: u64 = (0..geom.nsblk_addrs)
        .map(|j| {
            let (ndblks, dn) = geom.sblks[geom.first_indirect_sblk + j];
            ndblks * dn
        })
        .sum();
    u64::from(EA_IDX_BLK_ELMTS) + direct + indirect
}

// The creation parameters the writer stores and `extensible_array_capacity` derives the capacity
// from, the C library's defaults (`H5Dpkg.h`, HDF5 2.2.0).

/// The number of bits that hold the maximum number of elements, `H5D_EARRAY_MAX_NELMTS_BITS`.
const EA_MAX_NELMTS_BITS: u8 = 32;
/// The number of elements in the index block, `H5D_EARRAY_IDX_BLK_ELMTS`.
const EA_IDX_BLK_ELMTS: u8 = 4;
/// The number of elements in the smallest data block, `H5D_EARRAY_DATA_BLK_MIN_ELMTS`.
const EA_MIN_DBLK_NELMTS: u8 = 16;
/// The fewest data block addresses a super block holds, `H5D_EARRAY_SUP_BLK_MIN_DATA_PTRS`.
const EA_SUPER_BLK_MIN_DATA_PTRS: u8 = 4;
/// The base 2 logarithm of the number of elements in a data block page,
/// `H5D_EARRAY_MAX_DBLOCK_PAGE_NELMTS_BITS`.
const EA_MAX_DBLK_NELMTS_BITS: u8 = 10;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::checksum::restamp as stamp;

    use super::*;

    // The oracle is `H5EA__dblock_alloc`, which pages a block of more elements than a page.
    #[test]
    fn a_data_block_is_paged_only_past_a_full_page() {
        let at = |dblk_nelmts| DataBlockGeometry {
            dblk_nelmts,
            page_nelmts: 16,
        };
        assert!(!at(15).is_paged(), "under a page");
        assert!(!at(16).is_paged(), "exactly a page is not paged");
        assert!(at(17).is_paged(), "past a page");
        assert_eq!(at(16).npages(), 0, "a block that is not paged has no pages");
        assert_eq!(at(32).npages(), 2);
    }

    #[test]
    fn the_page_init_bitmap_is_a_byte_per_eight_pages_per_block() {
        let sb = |ndblks, dblk_nelmts| SuperBlockGeometry {
            ndblks,
            blocks: DataBlockGeometry {
                dblk_nelmts,
                page_nelmts: 16,
            },
        };
        // Not paged: no bitmap.
        assert_eq!(sb(4, 16).bitmap_size(), 0);
        // 2 blocks x 8 pages each -> one byte per block.
        assert_eq!(sb(2, 8 * 16).bitmap_size(), 2);
        // 2 blocks x 9 pages each -> the round-up costs a second byte per block.
        assert_eq!(sb(2, 9 * 16).bitmap_size(), 4);
        // One block, two pages: a single byte, not two bits.
        assert_eq!(sb(1, 2 * 16).bitmap_size(), 1);
    }

    /// Returns the length of a header: 12 fixed bytes, six length-sized statistics, the index
    /// block address, and the checksum.
    const fn eahd_len(os: u8, ls: u8) -> usize {
        12 + 6 * ls as usize + os as usize + 4
    }
    #[test]
    fn parse_header_valid() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 0; // version
        buf[5] = 0; // `client_id` = non-filtered
        buf[6] = 8; // `element_size`
        buf[7] = 10; // `max_nelmts_bits`
        buf[8] = 2; // `idx_blk_elmts`
        buf[9] = 4; // `min_dblk_nelmts`
        buf[10] = 2; // `super_blk_min_data_ptrs`
        buf[11] = 8; // `max_dblk_nelmts_bits`
        // 6 stats fields (each 8 bytes)
        buf[12..20].copy_from_slice(&0u64.to_le_bytes()); // stat[0]
        buf[20..28].copy_from_slice(&0u64.to_le_bytes()); // stat[1]
        buf[28..36].copy_from_slice(&0u64.to_le_bytes()); // stat[2]
        buf[36..44].copy_from_slice(&0u64.to_le_bytes()); // stat[3]
        buf[44..52].copy_from_slice(&5u64.to_le_bytes()); // stat[4] = `max_idx_set`
        buf[52..60].copy_from_slice(&0u64.to_le_bytes()); // stat[5]
        buf[60..68].copy_from_slice(&0x1000u64.to_le_bytes()); // `index_block_address`
        stamp(&mut buf, 0, eahd_len(os, ls));

        let hdr = ExtensibleArrayHeader::parse(&buf, 0, os, ls).unwrap();
        assert_eq!(hdr.client_id, 0);
        assert_eq!(hdr.element_size, 8);
        assert_eq!(hdr.idx_blk_elmts, 2);
        assert_eq!(hdr.min_dblk_nelmts, 4);
        assert_eq!(hdr.max_idx_set, 5);
        assert_eq!(hdr.index_block_address, StoredAddress::new(0x1000));
    }

    #[test]
    fn parse_header_invalid_signature() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"XXXX");
        let result = ExtensibleArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_invalid_version() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 1;
        let result = ExtensibleArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    /// Returns the chunks the buffered walk reports, with the slot of each.
    fn records(
        file_data: &[u8],
        header: &ExtensibleArrayHeader,
        chunk_byte_size: u64,
    ) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_extensible_array_chunks(file_data, header, 8, chunk_byte_size, |slot, record| {
            records.push((slot, record));
            Ok(())
        })?;
        Ok(records)
    }

    /// Returns the chunks the streaming walk reports, with the slot of each.
    fn records_from_source(
        source: &[u8],
        header: &ExtensibleArrayHeader,
        chunk_byte_size: u64,
    ) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let mut records = Vec::new();
        read_extensible_array_chunks_from_source(
            source,
            header,
            8,
            chunk_byte_size,
            |slot, record| {
                records.push((slot, record));
                Ok(())
            },
        )?;
        Ok(records)
    }

    /// Returns the chunks both walks report for the array whose header is at `header_offset`, and
    /// asserts that the two agree.
    fn walk_both(
        file_data: &[u8],
        header_offset: usize,
        chunk_byte_size: u64,
    ) -> Vec<(u64, ChunkRecord)> {
        let header = ExtensibleArrayHeader::parse(file_data, header_offset, 8, 8).unwrap();
        let buffered = records(file_data, &header, chunk_byte_size).unwrap();
        let header = ExtensibleArrayHeader::parse_from_source(
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

    /// Returns the records of unfiltered chunks of `chunk_byte_size` bytes stored back
    /// to back from `base_addr`, one per slot of `slots`.
    fn back_to_back(
        slots: core::ops::Range<u64>,
        base_addr: u64,
        chunk_byte_size: u64,
    ) -> Vec<(u64, ChunkRecord)> {
        slots
            .map(|slot| {
                let record = ChunkRecord {
                    address: StoredAddress::new(base_addr + slot * chunk_byte_size),
                    stored_size: chunk_byte_size,
                    filter_mask: 0,
                };
                (slot, record)
            })
            .collect()
    }

    #[test]
    fn read_inline_only() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let num_chunks = 2usize;
        let chunk_byte_size = 20u64 * 8; // 20 elements × 8 bytes

        let mut file_data = vec![0u8; 0x3000];

        // EAHD at offset 0x100
        let header_offset = 0x100usize;
        let index_block_offset = 0x200usize;

        // Build EAHD
        file_data[header_offset..header_offset + 4].copy_from_slice(b"EAHD");
        file_data[header_offset + 4] = 0; // version
        file_data[header_offset + 5] = 0; // `client_id` = non-filtered
        file_data[header_offset + 6] = osv as u8; // `element_size`
        file_data[header_offset + 7] = 10; // `max_nelmts_bits`
        file_data[header_offset + 8] = num_chunks as u8; // `idx_blk_elmts` (all inline)
        file_data[header_offset + 9] = 4; // `min_dblk_nelmts`
        file_data[header_offset + 10] = 2; // `super_blk_min_data_ptrs`
        file_data[header_offset + 11] = 8; // `max_dblk_nelmts_bits`
        // 6 stats fields (each 8 bytes), `max_idx_set` at stat[4]
        file_data[header_offset + 44..header_offset + 52]
            .copy_from_slice(&(num_chunks as u64).to_le_bytes());
        file_data[header_offset + 60..header_offset + 68]
            .copy_from_slice(&(index_block_offset as u64).to_le_bytes());
        stamp(&mut file_data, header_offset, eahd_len(os, ls));

        // Build EAIB at `index_block_offset`
        file_data[index_block_offset..index_block_offset + 4].copy_from_slice(b"EAIB");
        file_data[index_block_offset + 4] = 0; // version
        file_data[index_block_offset + 5] = 0; // `client_id`
        file_data[index_block_offset + 6..index_block_offset + 14]
            .copy_from_slice(&(header_offset as u64).to_le_bytes());

        // Inline elements
        let elem_start = index_block_offset + 6 + osv;
        let base_addr = 0x1000u64;
        for i in 0..num_chunks {
            let addr = base_addr + i as u64 * chunk_byte_size;
            let p = elem_start + i * osv;
            file_data[p..p + osv].copy_from_slice(&addr.to_le_bytes());
        }

        // The index block is sized by the header's geometry whatever the array
        // holds: prefix, two inline slots, then a pointer per direct data block
        // (min_dblk_nelmts=4 and super_blk_min_data_ptrs=2 give two) and one per
        // super block (nine levels less the two taken as direct, so seven).
        // The pointers stay zero, and the read stops at the inline slots.
        stamp(
            &mut file_data,
            index_block_offset,
            (6 + osv) + 2 * osv + 2 * osv + 7 * osv + 4,
        );

        assert_eq!(
            walk_both(&file_data, header_offset, chunk_byte_size),
            back_to_back(0..2, base_addr, chunk_byte_size)
        );
    }

    #[test]
    fn read_inline_plus_data_blocks() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let chunk_byte_size = 10u64 * 8; // 10 elements × 8 bytes
        let idx_blk_elmts = 2u8;
        let min_dblk_nelmts = 2u8;
        let sblk_min = 2u8;
        let total_chunks = 4usize; // 2 inline and 2 in the first data block

        let mut file_data = vec![0u8; 0x5000];
        let header_offset = 0x100usize;
        let index_block_offset = 0x200usize;
        let data_block_offset = 0x300usize;

        // EAHD
        file_data[header_offset..header_offset + 4].copy_from_slice(b"EAHD");
        file_data[header_offset + 4] = 0;
        file_data[header_offset + 5] = 0; // `client_id`
        file_data[header_offset + 6] = osv as u8; // `element_size`
        file_data[header_offset + 7] = 10;
        file_data[header_offset + 8] = idx_blk_elmts;
        file_data[header_offset + 9] = min_dblk_nelmts;
        file_data[header_offset + 10] = sblk_min;
        file_data[header_offset + 11] = 8;
        // 6 stats fields (each 8 bytes), `max_idx_set` at stat[4] (offset 12 + 4*8 = 44)
        file_data[header_offset + 44..header_offset + 52]
            .copy_from_slice(&(total_chunks as u64).to_le_bytes());
        // The index block address at offset 12 + 6*8 = 60
        file_data[header_offset + 60..header_offset + 68]
            .copy_from_slice(&(index_block_offset as u64).to_le_bytes());
        stamp(&mut file_data, header_offset, eahd_len(os, ls));

        // EAIB
        file_data[index_block_offset..index_block_offset + 4].copy_from_slice(b"EAIB");
        file_data[index_block_offset + 4] = 0;
        file_data[index_block_offset + 5] = 0;
        file_data[index_block_offset + 6..index_block_offset + 14]
            .copy_from_slice(&(header_offset as u64).to_le_bytes());

        let mut pos = index_block_offset + 6 + osv;

        // Inline elements (2 chunks)
        let base_addr = 0x1000u64;
        for i in 0..idx_blk_elmts as usize {
            let addr = base_addr + i as u64 * chunk_byte_size;
            file_data[pos..pos + osv].copy_from_slice(&addr.to_le_bytes());
            pos += osv;
        }

        // The index block addresses the data blocks of the first `super_blk_min_data_ptrs` super
        // blocks, which with `min_dblk_nelmts = 2` are 1 block of 2 and 1 block of 4. The first
        // holds the two elements past the two inline ones.
        let n_direct_dblks = 2;
        file_data[pos..pos + osv].copy_from_slice(&(data_block_offset as u64).to_le_bytes());
        pos += osv;
        for _ in 1..n_direct_dblks {
            file_data[pos..pos + osv].copy_from_slice(&u64::MAX.to_le_bytes());
            pos += osv;
        }

        // EADB at `data_block_offset` (`min_dblk_nelmts` elements)
        file_data[data_block_offset..data_block_offset + 4].copy_from_slice(b"EADB");
        file_data[data_block_offset + 4] = 0;
        file_data[data_block_offset + 5] = 0;
        file_data[data_block_offset + 6..data_block_offset + 14]
            .copy_from_slice(&(header_offset as u64).to_le_bytes());
        // The block offset takes ceil(10 / 8) = 2 bytes, and is 0 for the first data block.
        let blk_off_size = (10usize).div_ceil(8); // max_nelmts_bits=10
        let mut dbpos = data_block_offset + 6 + osv + blk_off_size;
        for i in 0..min_dblk_nelmts as usize {
            let addr = base_addr + (idx_blk_elmts as u64 + i as u64) * chunk_byte_size;
            file_data[dbpos..dbpos + osv].copy_from_slice(&addr.to_le_bytes());
            dbpos += osv;
        }

        // Prefix, two inline slots, the two direct pointers written above, and
        // one pointer per remaining super block (ten levels less the two taken
        // as direct).
        stamp(
            &mut file_data,
            index_block_offset,
            (6 + osv) + 2 * osv + n_direct_dblks * osv + 8 * osv + 4,
        );
        // Prefix, block offset, `min_dblk_nelmts` element slots, checksum.
        stamp(
            &mut file_data,
            data_block_offset,
            (6 + osv) + blk_off_size + min_dblk_nelmts as usize * osv + 4,
        );

        assert_eq!(
            walk_both(&file_data, header_offset, chunk_byte_size),
            back_to_back(0..4, base_addr, chunk_byte_size)
        );
    }

    // The writer builds the arrays, whose counts reach the super blocks and the paged data blocks.
    #[test]
    fn streaming_ea_super_blocks_and_paged_match_buffered() {
        // n covers: inline+direct (2000), several super blocks (50000), and
        // paged super-block data blocks (140000, since `dblk_nelmts` exceeds the
        // 1024-element page size at the higher super blocks).
        for &n in &[2000u64, 50000, 140000] {
            let expected = back_to_back(0..n, 0x10, 8);
            let chunks: Vec<ChunkRecord> = expected.iter().map(|&(_, record)| record).collect();
            let base = 0x1000u64;
            let ea = build_extensible_array_at(
                &IndexSlots::dense(&chunks),
                8,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                false,
                StoredAddress::new(base),
            )
            .unwrap();
            let mut file = vec![0u8; base as usize + ea.len()];
            file[base as usize..].copy_from_slice(&ea);

            assert_eq!(
                walk_both(&file, base as usize, 8),
                expected,
                "chunk records at n={n}"
            );
        }
    }

    // Covers what a round trip does not write: 4-byte addresses, a 4-byte block offset, and a
    // block of no elements.
    #[test]
    fn eadb_extent_matches_the_writer() {
        for &(client_id, element_size) in &[(0u8, 8u8), (1, 20)] {
            for &offset_size in &[4u8, 8] {
                for &max_nelmts_bits in &[10u8, 16, 32] {
                    let header = ExtensibleArrayHeader {
                        client_id,
                        element_size,
                        max_nelmts_bits,
                        idx_blk_elmts: 4,
                        min_dblk_nelmts: 4,
                        super_blk_min_data_ptrs: 2,
                        max_dblk_nelmts_bits: 10,
                        max_idx_set: 0,
                        index_block_address: StoredAddress::new(0),
                    };
                    let blk_off = (max_nelmts_bits as usize).div_ceil(8);
                    let stride = ea_elem_stride(&header, offset_size);
                    let page_nelmts = 1u64 << header.max_dblk_nelmts_bits;
                    // Every count a non-paged block can have, up to the page
                    // size that would make it paged instead.
                    for nelmts in [0u64, 1, 2, 4, 16, 64, 255, 256, 1023, page_nelmts] {
                        assert_eq!(
                            eadb_extent(nelmts as usize, &header, offset_size, blk_off).unwrap()
                                as u64,
                            data_block_len(
                                DataBlockGeometry {
                                    dblk_nelmts: nelmts,
                                    page_nelmts,
                                },
                                stride,
                                offset_size,
                                blk_off,
                            ),
                            "client={client_id} os={offset_size} bits={max_nelmts_bits} \
                             nelmts={nelmts}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn header_serialized_size() {
        // 12 fixed + 6*8 stats + 8 address + 4 checksum = 72
        assert_eq!(ExtensibleArrayHeader::serialized_size(8, 8), 72);
        // 12 fixed + 6*4 stats + 4 address + 4 checksum = 44
        assert_eq!(ExtensibleArrayHeader::serialized_size(4, 4), 44);
    }

    #[test]
    fn read_element_unallocated() {
        let data = vec![0xFFu8; 16];
        let (record, consumed) = read_element(&data, 0, 0, 8, 8, 80).unwrap();
        assert_eq!(record, None);
        assert_eq!(consumed, 8);
    }

    #[test]
    fn read_element_filtered() {
        let os: u8 = 8;
        let chunk_size_bytes = 5usize;
        let elem_size = os as usize + chunk_size_bytes + 4;
        let mut data = vec![0u8; elem_size + 16];
        // Address
        data[0..8].copy_from_slice(&0x2000u64.to_le_bytes());
        let stored_size = u64::from(u32::MAX) + 120;
        data[8..8 + chunk_size_bytes]
            .copy_from_slice(&stored_size.to_le_bytes()[..chunk_size_bytes]);
        // Filter mask
        data[8 + chunk_size_bytes..12 + chunk_size_bytes].copy_from_slice(&0u32.to_le_bytes());

        let (record, consumed) =
            read_element(&data, 0, 1, u8::try_from(elem_size).unwrap(), os, 80).unwrap();
        assert_eq!(
            record,
            Some(ChunkRecord {
                address: StoredAddress::new(0x2000),
                stored_size,
                filter_mask: 0,
            })
        );
        assert_eq!(consumed, elem_size);
    }

    // The address and the 4-byte mask of a filtered element take `offset_size + 4` bytes.
    #[test]
    fn a_filtered_element_smaller_than_its_own_fields_is_an_error() {
        let os: u8 = 8;
        let data = vec![0u8; 64];

        // `element_size` must be at least os + 4 = 12 to hold what it claims.
        for element_size in 0..(os + 4) {
            let err = read_element(&data, 0, 1, element_size, os, 80)
                .expect_err("a filtered element narrower than its own fields must be rejected");
            assert!(
                matches!(err, FormatError::ChunkedReadError(_)),
                "element_size {element_size} gave {err:?}, want a ChunkedReadError"
            );
        }

        // The first width that can hold them is accepted.
        read_element(&data, 0, 1, os + 4 + 1, os, 80)
            .expect("a width that fits address + 1-byte size + mask must parse");
    }

    // The oracle is the end of the index block, which both readers bound before they read a field.
    #[test]
    fn truncated_super_block_addresses_fail_to_read_in_both_backends() {
        let n = 100u64;
        let chunks: Vec<ChunkRecord> = (0..n)
            .map(|i| ChunkRecord {
                address: StoredAddress::new(0x10 + i * 8),
                stored_size: 8,
                filter_mask: 0,
            })
            .collect();
        let base = 0x1000u64;
        let ea = build_extensible_array_at(
            &IndexSlots::dense(&chunks),
            8,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(base),
        )
        .unwrap();
        let mut file = vec![0u8; base as usize + ea.len()];
        file[base as usize..].copy_from_slice(&ea);

        let built = ExtensibleArrayHeader::parse(&file, base as usize, 8, 8).unwrap();
        let geom = ExtensibleArrayGeometry::from_header(&built);
        assert!(
            geom.nsblk_addrs > 0,
            "fixture must reach the super-block address array"
        );

        // The writer places the index block before the data blocks, so a cut in the index block
        // would cut the data blocks too. The format allows the index block after them, so the
        // test copies the index block, whose checksum covers its own bytes alone, to the end of
        // the file, repoints the header, and recomputes the header's checksum.
        let old_ib = built.index_block_address.get().to_usize().unwrap();
        let ib_len = 4 + 1 + 1 + 8                            // signature to header address
            + built.idx_blk_elmts as usize * 8                // inline elements
            + geom.direct_dblk_nelmts.len() * 8               // direct-block addresses
            + geom.nsblk_addrs * 8                            // super-block addresses
            + 4; // checksum
        let block = file[old_ib..old_ib + ib_len].to_vec();
        let new_ib = file.len();
        file.extend_from_slice(&block);
        // The header's index-block address sits after the 12-byte prefix and
        // the six length-sized statistics fields.
        let addr_field = base as usize + 12 + 6 * 8;
        file[addr_field..addr_field + 8].copy_from_slice(&(new_ib as u64).to_le_bytes());
        stamp(&mut file, base as usize, eahd_len(8, 8));

        let header = ExtensibleArrayHeader::parse(&file, base as usize, 8, 8).unwrap();
        assert_eq!(header.index_block_address.get() as usize, new_ib);
        // The relocated file must still read identically before it is cut.
        let intact = records(&file, &header, 8).unwrap();
        assert_eq!(intact.len() as u64, n, "relocation must preserve the read");

        // Where the first super-block address begins: index block prefix,
        // then the inline elements, then the direct-block addresses.
        let sblk_start = new_ib
            + 4
            + 1
            + 1
            + 8
            + header.idx_blk_elmts as usize * 8
            + geom.direct_dblk_nelmts.len() * 8;

        // Cuts the first super block address in half, past every data block.
        file.truncate(sblk_start + 4);

        let buffered = records(&file, &header, 8);
        match buffered {
            Err(FormatError::UnexpectedEof {
                expected,
                available,
            }) => {
                assert_eq!(
                    expected,
                    new_ib + ib_len,
                    "the buffered read must fault on the cut index block, not earlier"
                );
                assert_eq!(available, file.len());
            }
            other => panic!("buffered read must reject a truncated address array, got {other:?}"),
        }

        let hm = ExtensibleArrayHeader::parse_from_source(
            file.as_slice(),
            StoredAddress::new(base),
            8,
            8,
        )
        .unwrap();
        let streamed = records_from_source(&file, &hm, 8);
        assert!(
            streamed.is_err(),
            "streaming read must reject the same file, got {streamed:?}"
        );
    }

    // Flips a checksum byte of each structure the span walk reaches, for every reader of the
    // structure to reject.
    #[cfg(feature = "checksum")]
    #[test]
    fn a_corrupted_extensible_array_structure_fails_its_checksum() {
        // The same progression as `streaming_ea_super_blocks_and_paged_match_buffered`:
        // inline and direct blocks, then super blocks, then paged data blocks.
        for &n in &[2000u64, 50000, 140000] {
            let chunks: Vec<ChunkRecord> = (0..n)
                .map(|i| ChunkRecord {
                    address: StoredAddress::new(0x10 + i * 8),
                    stored_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let base = 0x1000u64;
            let ea = build_extensible_array_at(
                &IndexSlots::dense(&chunks),
                8,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                false,
                StoredAddress::new(base),
            )
            .unwrap();
            let mut file = vec![0u8; base as usize + ea.len()];
            file[base as usize..].copy_from_slice(&ea);

            let read_both = |file: &[u8]| {
                let buffered = ExtensibleArrayHeader::parse(file, base as usize, 8, 8)
                    .and_then(|h| records(file, &h, 8));
                let streamed =
                    ExtensibleArrayHeader::parse_from_source(file, StoredAddress::new(base), 8, 8)
                        .and_then(|h| records_from_source(file, &h, 8));
                (buffered, streamed.is_err())
            };
            assert_eq!(
                read_both(&file).0.expect("the sound file must read").len() as u64,
                n,
                "the fixture must read before it is corrupted, at n={n}"
            );

            let spans =
                extensible_array_index_spans(file.as_slice(), StoredAddress::new(base), 8, 8)
                    .unwrap();
            assert!(spans.len() > 1, "the sweep must reach past the header");

            let header = ExtensibleArrayHeader::parse(&file, base as usize, 8, 8).unwrap();
            let stride = ea_elem_stride(&header, 8);
            let page_nelmts = 1usize << header.max_dblk_nelmts_bits;
            // The prefix of a data block, and the length of the largest unpaged one. A paged block
            // holds at least two pages and is longer.
            let db_prefix = 4 + 1 + 1 + 8 + (header.max_nelmts_bits as usize).div_ceil(8);
            let max_unpaged = db_prefix + page_nelmts * stride + 4;

            // Each site is a byte to flip and whether the span walk reads its structure. The walk
            // reads the header, the index block, and the super blocks, and sizes the data blocks
            // from the header.
            let mut poke_sites: Vec<(u64, bool)> = Vec::new();
            let mut paged = 0;
            for &(at, len) in &spans {
                let start = at.to_usize().unwrap();
                let kind = &file[start..start + 4];
                if kind == b"EADB" && len as usize > max_unpaged {
                    paged += 1;
                    // A paged block has a checksum over its prefix and one per page. The reader
                    // steps over a page the bitmap does not mark, so the test flips the first
                    // page alone.
                    poke_sites.push((at + db_prefix as u64 + 4 - 1, false));
                    poke_sites.push((at + (max_unpaged + 4) as u64 - 1, false));
                } else {
                    // The last byte of every other structure is the top byte of
                    // its one trailing checksum.
                    poke_sites.push((at + len - 1, kind != b"EADB"));
                }
            }
            if n == 140000 {
                assert!(paged > 0, "n={n} must reach a paged data block");
            }
            assert!(
                poke_sites.iter().any(|&(_, walked)| walked),
                "n={n}: the sweep must reach a structure the reclaim walk reads"
            );

            for (site, walked) in poke_sites {
                let at = site.to_usize().unwrap();
                let original = file[at];
                file[at] ^= 0x01;
                let (buffered, streamed_err) = read_both(&file);
                assert!(
                    matches!(buffered, Err(FormatError::ChecksumMismatch { .. })),
                    "n={n}: a corrupted checksum at {at:#x} must be rejected, got {buffered:?}"
                );
                assert!(
                    streamed_err,
                    "n={n}: the streaming backend must reject what the buffered one does, at {at:#x}"
                );
                if walked {
                    let walk = extensible_array_index_spans(
                        file.as_slice(),
                        StoredAddress::new(base),
                        8,
                        8,
                    );
                    assert!(
                        matches!(walk, Err(FormatError::ChecksumMismatch { .. })),
                        "n={n}: the reclaim walk must reject a corrupt structure at {at:#x} \
                         rather than release spans read out of it, got {walk:?}"
                    );
                }
                file[at] = original;
            }
        }
    }

    // The oracle is the header, the index block, and the two size statistics the writer stores.
    #[test]
    fn index_spans_match_builder_layout() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let base = 0x4000u64;
        // Counts that reach the index block, the data blocks it addresses, the super blocks, and
        // the paged data blocks.
        for &n in &[1u64, 4, 20, 100, 244, 300, 2000, 50000, 140000] {
            let chunks: Vec<ChunkRecord> = (0..n)
                .map(|i| ChunkRecord {
                    address: StoredAddress::new(0x100000 + i * 8),
                    stored_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let ea = build_extensible_array_at(
                &IndexSlots::dense(&chunks),
                8,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                false,
                StoredAddress::new(base),
            )
            .unwrap();
            let mut file = vec![0u8; base as usize + ea.len()];
            file[base as usize..].copy_from_slice(&ea);

            let spans =
                extensible_array_index_spans(file.as_slice(), StoredAddress::new(base), os, ls)
                    .unwrap();

            // Expected total = EAHD + EAIB + `super_blk_size` + `data_blk_size`, the
            // last two read straight from the statistics the builder wrote.
            let header = ExtensibleArrayHeader::parse(&file, base as usize, os, ls).unwrap();
            let geom = ExtensibleArrayGeometry::from_header(&header);
            let header_len = ExtensibleArrayHeader::serialized_size(os, ls) as u64;
            // Unfiltered EA: element stride equals the offset size.
            let index_block_len = index_block_len(
                os,
                header.idx_blk_elmts as usize,
                os as usize,
                geom.direct_dblk_nelmts.len(),
                geom.nsblk_addrs,
            ) as u64;
            let stat = |k: usize| {
                let off = base as usize + 12 + k * ls as usize;
                u64::from_le_bytes(file[off..off + 8].try_into().unwrap())
            };
            let super_blk_size = stat(1);
            let data_blk_size = stat(3);
            let expected_total = header_len + index_block_len + super_blk_size + data_blk_size;

            let total: u64 = spans.iter().map(|&(_, l)| l).sum();
            assert_eq!(
                total, expected_total,
                "EA index span total mismatch at n={n}"
            );

            // Disjoint and inside the built blob.
            let mut sorted = spans.clone();
            sorted.sort_by_key(|&(a, _)| a);
            for w in sorted.windows(2) {
                assert!(
                    w[0].0 + w[0].1 <= w[1].0,
                    "EA index spans overlap at n={n}: {sorted:?}"
                );
            }
            for &(a, l) in &spans {
                assert!(
                    a >= base && a + l <= base + ea.len() as u64,
                    "EA index span out of the blob at n={n}: ({a}, {l})"
                );
            }
        }
    }

    #[test]
    fn build_extensible_array_valid_structure() {
        let chunks = vec![
            ChunkRecord {
                address: StoredAddress::new(0x1000),
                stored_size: 80,
                filter_mask: 0,
            },
            ChunkRecord {
                address: StoredAddress::new(0x1050),
                stored_size: 80,
                filter_mask: 0,
            },
        ];
        let ea = build_extensible_array_at(
            &IndexSlots::dense(&chunks),
            80,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(0x2000),
        )
        .unwrap();
        assert_eq!(&ea[0..4], b"EAHD");
        // `EAIB` follows `EAHD`: 12 fixed + 6*8 stats + 8 address + 4 checksum = 72
        let header_len = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * 8 + 8 + 4;
        assert_eq!(&ea[header_len..header_len + 4], b"EAIB");
    }

    #[test]
    fn extensible_array_stats_matches_what_it_builds() {
        let geom_header = ExtensibleArrayHeader {
            client_id: 0,
            element_size: 8,
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            min_dblk_nelmts: 16,
            super_blk_min_data_ptrs: 4,
            max_dblk_nelmts_bits: 10,
            max_idx_set: 0,
            index_block_address: StoredAddress::new(0),
        };
        let geom = ExtensibleArrayGeometry::from_header(&geom_header);
        for &n in &[1u64, 4, 20, 100, 244, 300, 2000, 50000, 131056, 140000] {
            let chunks: Vec<ChunkRecord> = (0..n)
                .map(|i| ChunkRecord {
                    address: StoredAddress::new(0x1000 + i * 8),
                    stored_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let ea = build_extensible_array_at(
                &IndexSlots::dense(&chunks),
                8,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                false,
                StoredAddress::new(0x100000),
            )
            .unwrap();
            // Parse the 6 stats from the EAHD (12-byte fixed prefix, then 6 * ls).
            let stat =
                |k: usize| u64::from_le_bytes(ea[12 + k * 8..12 + k * 8 + 8].try_into().unwrap());
            let built = super::ExtensibleArrayStats {
                nsuper_blks: stat(0),
                super_blk_size: stat(1),
                ndata_blks: stat(2),
                data_blk_size: stat(3),
                max_idx_set: stat(4),
                nelmts: stat(5),
            };
            let computed = super::extensible_array_stats(
                &geom,
                4,
                8,
                1024,
                OffsetWidth::Eight,
                4,
                n,
                SlotOccupancy::Dense(n),
            );
            assert_eq!(computed, built, "stats mismatch at n={n}");
        }
    }

    // Sweeps every count through the first super block, then the deeper super blocks and the
    // first paged data block.
    #[test]
    fn extensible_array_len_matches_what_it_builds() {
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
            let planned = extensible_array_len(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
            );
            let built = build_extensible_array_at(
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
                // Contiguous across the inline, direct-block and first
                // super-block ranges.
                for n in 0..=250u64 {
                    check(n, 8, offset_size, length_size, has_filters);
                }
                // The deeper super blocks, and the paged boundary: 131,060 is the
                // last element the unpaged super block 12 holds, 131,061 the first
                // that allocates a paged data block.
                for &n in &[300u64, 2_000, 50_000, 131_060, 131_061, 140_000] {
                    check(n, 8, offset_size, length_size, has_filters);
                }
            }
        }

        // The chunk size field of a filtered element, and with it every block, grows with the chunk
        // size.
        for &chunk_bytes in &chunk_record::CHUNK_BYTES {
            for &n in &[1u64, 5, 244, 300, 2_000] {
                check(n, chunk_bytes, OffsetWidth::Eight, LengthWidth::Eight, true);
            }
        }
        // The four chunk sizes select four different widths, measured on an array of no chunks,
        // whose width comes from `chunk_bytes` alone.
        let widths: Vec<usize> = chunk_record::CHUNK_BYTES
            .iter()
            .map(|&chunk_bytes| {
                super::extensible_array_layout(
                    SlotOccupancy::Dense(0),
                    0,
                    chunk_bytes,
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                    true,
                )
                .encoding
                .chunk_size_bytes
            })
            .collect();
        let mut distinct = widths.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            chunk_record::CHUNK_BYTES.len(),
            "each chunk size must select a different compressed-size field width, got {widths:?}"
        );
    }

    #[rstest]
    fn an_array_reads_back_at_every_width(
        #[values(
            (OffsetWidth::Two, LengthWidth::Two),
            (OffsetWidth::Four, LengthWidth::Four),
            (OffsetWidth::Eight, LengthWidth::Eight)
        )]
        widths: (OffsetWidth, LengthWidth),
        #[values(3, 2_000)] n: u64,
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
        let ea = build_extensible_array_at(
            &IndexSlots::dense(&chunks),
            8,
            offset_size,
            length_size,
            has_filters,
            StoredAddress::new(base),
        )
        .unwrap();
        let mut file = vec![0u8; base as usize];
        file.extend_from_slice(&ea);

        let header = ExtensibleArrayHeader::parse(
            &file,
            base as usize,
            offset_size.get(),
            length_size.get(),
        )
        .unwrap();
        let mut read = Vec::new();
        read_extensible_array_chunks(&file, &header, offset_size.get(), 8, |_, record| {
            read.push(record);
            Ok(())
        })
        .unwrap();
        assert_eq!(read, chunks);
    }
}
