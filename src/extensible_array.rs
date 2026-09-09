//! HDF5 Extensible Array index parsing for chunked datasets (v4 index type 4).
//!
//! Extensible Arrays are used for datasets with exactly one unlimited dimension.
//! Structures: AEHD (header), AEIB (index block), AEDB (data block), AESB (super block).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use crate::bytes::{read_length, read_offset, read_optional_offset};
use crate::chunk_grid::ChunkGrid;
use crate::chunked_read::ChunkInfo;
use crate::convert::{TryToUsize, is_undefined_addr, u32_from};
use crate::error::FormatError;
use crate::source::Source;

/// Parsed Extensible Array header (AEHD).
#[derive(Debug, Clone)]
pub struct ExtensibleArrayHeader {
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size of each array element in bytes.
    pub element_size: u8,
    /// Max number of elements bits (log2 of the max number of data block elements per page).
    pub max_nelmts_bits: u8,
    /// Number of elements in the index block.
    pub idx_blk_elmts: u8,
    /// Minimum number of data block elements.
    pub min_dblk_nelmts: u8,
    /// Minimum number of elements in a super block.
    pub super_blk_min_nelmts: u8,
    /// Max number of data block elements bits.
    pub max_dblk_nelmts_bits: u8,
    /// Total number of elements stored.
    pub num_elements: u64,
    /// Address of the index block.
    pub index_block_address: u64,
}

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
    /// Parse an Extensible Array header from file data at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        // EAHD: signature(4) + version(1) + client_id(1) + element_size(1) +
        //   max_nelmts_bits(1) + idx_blk_elmts(1) + min_dblk_nelmts(1) +
        //   super_blk_min_nelmts(1) + max_dblk_nelmts_bits(1) +
        //   6 stats fields (each length_size) + index_block_address(offset_size) + checksum(4)
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
        let super_blk_min_nelmts = d[10];
        let max_dblk_nelmts_bits = d[11];

        let mut pos = 12;
        // 6 statistics fields, in the HDF5 C library's stored order:
        //   [0] nsuper_blks   [1] super_blk_size   [2] ndata_blks
        //   [3] data_blk_size [4] max_idx_set      [5] nelmts
        // We read max_idx_set (field [4]) — the count of elements that have been
        // set, which bounds the live region. For a densely written array it
        // equals the chunk count; `nelmts` (field [5]) is the larger
        // rounded-up-to-block-boundary slot count, which we do not need.
        let ls = length_size as usize;
        pos += 4 * ls; // skip [0]..[3]
        let num_elements = read_length(d, pos, length_size)?; // [4] max_idx_set
        pos += ls;
        pos += ls; // skip [5] nelmts
        let index_block_address = read_offset(d, pos, offset_size)?;

        crate::checksum::verify_trailing(&d[..min_size])?;

        Ok(ExtensibleArrayHeader {
            client_id,
            element_size,
            max_nelmts_bits,
            idx_blk_elmts,
            min_dblk_nelmts,
            super_blk_min_nelmts,
            max_dblk_nelmts_bits,
            num_elements,
            index_block_address,
        })
    }

    /// Compute the size of this header in bytes (for write support).
    pub fn serialized_size(offset_size: u8, length_size: u8) -> usize {
        4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * length_size as usize + offset_size as usize + 4
    }

    /// Parse an Extensible Array header from a [`Source`] (bounded window).
    pub fn parse_from_source<S: Source + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        let size = Self::serialized_size(offset_size, length_size);
        let buf = source.read_metadata_at(address, size)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// Per-element stride in an Extensible Array data block: just the address for
/// non-filtered chunks, or the full filtered element record otherwise.
fn ea_elem_stride(header: &ExtensibleArrayHeader, offset_size: u8) -> usize {
    if header.client_id == 0 {
        offset_size as usize
    } else {
        header.element_size as usize
    }
}

/// Geometry of an Extensible Array, derived from its header creation parameters.
///
/// This is the single source of truth for the super-block / data-block size
/// progression, shared by both the reader (this module) and the writer
/// (`chunked_write`). It mirrors libhdf5's `H5EA__hdr_init`: starting from
/// `(ndblks = 1, dblk_nelmts = min_dblk_nelmts)`, each successive super block
/// alternately doubles the elements-per-data-block (even index) and the
/// number-of-data-blocks (odd index):
///
/// ```text
///   super block i:  ndblks = 2^floor(i/2)   dblk_nelmts = min_dblk * 2^ceil(i/2)
///   SB0: 1 x 16   SB1: 1 x 32   SB2: 2 x 32   SB3: 2 x 64   SB4: 4 x 64 ...
/// ```
///
/// The first `super_blk_min_nelmts` super blocks have their data-block
/// addresses stored directly in the index block; the rest are reached through
/// on-disk super blocks (`EASB`) whose addresses are stored in the index block.
#[derive(Debug, Clone)]
pub(crate) struct EaGeometry {
    /// `(ndblks, dblk_nelmts)` for each super block index `0..nsblks`.
    pub sblks: Vec<(u64, u64)>,
    /// Element count of each direct data block whose address is stored in the
    /// index block. `len()` equals the number of direct data-block pointers.
    pub direct_dblk_nelmts: Vec<u64>,
    /// Number of super-block pointers stored in the index block.
    pub nsblk_addrs: usize,
    /// Super-block index of the first super block reached via a super-block
    /// pointer in the index block (i.e. `super_blk_min_nelmts`).
    pub first_indirect_sblk: usize,
}

impl EaGeometry {
    /// The shape of the super block reached by the `j`-th super-block pointer in
    /// the index block, in an array whose pages hold `page_nelmts` elements.
    ///
    /// `j` indexes the *pointers*, not [`sblks`](Self::sblks): the first
    /// `first_indirect_sblk` super blocks are stored directly in the index block
    /// and have no pointer, which is the offset every caller was applying by
    /// hand before this existed.
    pub(crate) fn super_block_at(&self, j: usize, page_nelmts: u64) -> SuperBlockGeom {
        let (ndblks, dblk_nelmts) = self.sblks[self.first_indirect_sblk + j];
        SuperBlockGeom {
            ndblks,
            blocks: DataBlockGeom {
                dblk_nelmts,
                page_nelmts,
            },
        }
    }

    /// Derive the geometry from a parsed header.
    pub fn from_header(h: &ExtensibleArrayHeader) -> Self {
        let min_dblk = h.min_dblk_nelmts as u64;
        let sup_blk_min = h.super_blk_min_nelmts as usize;
        // nsblks = max_nelmts_bits - log2(min_dblk_nelmts) + 1
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
        EaGeometry {
            sblks,
            direct_dblk_nelmts,
            nsblk_addrs,
            first_indirect_sblk: sup_blk_min,
        }
    }
}

/// How one Extensible Array data block is paged: the elements it holds, against
/// the elements that fill one page.
///
/// A data block holding more than one page's worth of elements is stored as a
/// sequence of individually-checksummed pages. A block holding *exactly* one
/// page's worth is not paged — the boundary is strictly greater, matching
/// libhdf5's `H5EA__dblock_alloc`.
///
/// Separate from [`SuperBlockGeom`] because that is the whole dependency: a
/// direct data block, whose address sits in the index block and which has no
/// super block at all, is paged by the same rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DataBlockGeom {
    /// Elements the data block holds.
    pub(crate) dblk_nelmts: u64,
    /// Elements that fill one page.
    pub(crate) page_nelmts: u64,
}

impl DataBlockGeom {
    /// Whether the block is stored as pages.
    pub(crate) const fn is_paged(self) -> bool {
        self.dblk_nelmts > self.page_nelmts
    }

    /// Pages in the block, or zero when it is not paged.
    pub(crate) const fn npages(self) -> u64 {
        if self.is_paged() {
            self.dblk_nelmts / self.page_nelmts
        } else {
            0
        }
    }
}

/// The shape of one Extensible Array super block (`EASB`): the data blocks it
/// points at, and how those blocks are paged.
///
/// A super block whose data blocks are paged carries a page-init bitmap between
/// its header and its data-block addresses; one whose blocks are not carries no
/// bitmap at all. [`DataBlockGeom::is_paged`] therefore decides the *position* of
/// every data-block address in the block rather than merely a size, and every
/// reader of an `EASB` has to agree with every writer about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SuperBlockGeom {
    /// Data-block pointers the super block holds.
    pub(crate) ndblks: u64,
    /// How each of those data blocks is paged.
    pub(crate) blocks: DataBlockGeom,
}

impl SuperBlockGeom {
    /// Byte size of the page-init bitmap, zero when the data blocks are not
    /// paged.
    ///
    /// One bit per page per data block. The bits are a contiguous stream indexed
    /// by `dblk_local * npages + page`; the per-block round-up to a whole byte is
    /// over-allocation rather than alignment, so block *k*'s bits do not in
    /// general start at byte `k * ceil(npages / 8)`.
    ///
    /// Computed in `u64`, and narrowed by whichever caller needs a `usize`,
    /// rather than narrowing the operands first. The product cannot overflow
    /// where this is called: `bitmap_size <= ndblks * dblk_nelmts`, and every
    /// caller has already computed that product or the enclosing block's length.
    pub(crate) const fn bitmap_size(self) -> u64 {
        self.ndblks * self.blocks.npages().div_ceil(8)
    }
}

/// Read a single element from the extensible array element data.
/// Returns (chunk_info, bytes_consumed) or None if unallocated.
#[allow(clippy::too_many_arguments)]
fn read_element(
    data: &[u8],
    pos: usize,
    client_id: u8,
    element_size: u8,
    offset_size: u8,
    chunk_byte_size: u64,
    linear_index: usize,
    grid: &ChunkGrid,
) -> Result<(Option<ChunkInfo>, usize), FormatError> {
    let os = offset_size as usize;

    if client_id == 0 {
        // Non-filtered: just address
        if os > data.len() || pos > data.len() - os {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(os),
                available: data.len(),
            });
        }
        let Some(address) = read_optional_offset(data, pos, offset_size)? else {
            return Ok((None, os));
        };
        let Some(offsets) = grid.offsets_in_extent(linear_index as u64)? else {
            return Ok((None, os));
        };
        Ok((
            Some(ChunkInfo {
                chunk_size: u32_from(chunk_byte_size)?,
                filter_mask: 0,
                offsets,
                address,
            }),
            os,
        ))
    } else {
        // Filtered: address + compressed_size + filter_mask. `element_size` is
        // the EAHD's own byte (`d[6]`), so a crafted header can name a width
        // that cannot hold the address and mask it must contain; subtract with
        // a check rather than underflowing. This mirrors the Fixed Array twin
        // in `fixed_array::read_fixed_array_chunks`.
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
        let Some(address) = read_optional_offset(data, pos, offset_size)? else {
            return Ok((None, elem_total));
        };
        let Some(offsets) = grid.offsets_in_extent(linear_index as u64)? else {
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
            Some(ChunkInfo {
                chunk_size: u32_from(chunk_size)?,
                filter_mask,
                offsets,
                address,
            }),
            elem_total,
        ))
    }
}

/// Collect elements from a data block at the given offset.
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
    grid: &ChunkGrid,
) -> Result<Vec<ChunkInfo>, FormatError> {
    // AEDB: signature(4) + version(1) + client_id(1) + header_address(offset_size)
    let db_header_size = 4 + 1 + 1 + offset_size as usize;
    // Block offset is encoded in ceil(max_nelmts_bits/8) bytes.
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    // The block's checksum covers all `nelmts` slots, not just the ones this
    // read decodes, so the extent is the writer's -- every slot is written when
    // the block is allocated, whether or not a chunk occupies it.
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
    // Skip version(1) + client_id(1) + header_address(offset_size) + block_offset
    let mut pos = db_offset + db_header_size + blk_off_size;

    // Direct data blocks and non-paged super-block data blocks store their
    // elements inline. Paged data blocks (only ever found inside a super block
    // whose `dblk_nelmts` exceeds the page size) are handled separately by
    // `read_paged_data_block`, which is driven by the page-init bitmap stored
    // in the owning super block.
    // Read only as many elements as the published count allows. A SWMR writer
    // may have written element slots beyond it (it grows the block before
    // bumping the count), and those slots hold whatever was there before.
    let limit = total_elements.saturating_sub(start_index).min(nelmts);
    let mut chunks = Vec::new();
    for i in 0..limit {
        let (info, consumed) = read_element(
            file_data,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
            start_index + i,
            grid,
        )?;
        if let Some(ci) = info {
            chunks.push(ci);
        }
        pos += consumed;
    }

    Ok(chunks)
}

/// On-disk extent of a *non-paged* Extensible Array data block (`EADB`): the
/// prefix, one record per element slot, and the trailing checksum that covers
/// them.
///
/// Every slot is written when the block is allocated, so this is fixed by the
/// block's element count and does not shrink for a read that decodes fewer.
///
/// [`crate::chunked_write::eadb_size`] states the same rule for the writer, and
/// deliberately stays a separate function: it sizes a block from an in-memory
/// write request, which is bounded, while this one sizes it from a header
/// parsed out of a file, which is not — hence the checked arithmetic.
/// `eadb_extent_matches_the_writer` holds the two to the same answer.
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

/// Test whether page `page_idx` is initialized in a super-block page-init
/// bitmap. The bitmap is a contiguous, MSB-first bit stream: page 0 is bit 7 of
/// byte 0, page 1 is bit 6, page 8 is bit 7 of byte 1, and so on.
fn page_is_initialized(bitmap: &[u8], page_idx: usize) -> bool {
    let byte = page_idx / 8;
    let mask = 0x80u8 >> (page_idx % 8);
    byte < bitmap.len() && (bitmap[byte] & mask) != 0
}

/// Read a paged Extensible Array data block.
///
/// A paged data block has a 22-byte header (signature, version, client id,
/// header address, block offset) *followed by its own checksum*, after which
/// the data block's pages are laid out contiguously. Each page holds
/// `page_nelmts` element slots followed by a 4-byte checksum, and is written
/// only when initialized. Whether page `p` of this data block is initialized is
/// recorded in the owning super block's bitmap at global page index
/// `db_local_idx * npages + p`.
///
/// Pages are *not* filled in order: a writer initializes the page a chunk lands
/// in, and a dataset whose maximum shape is wider than its shape leaves gaps
/// between them. An uninitialized page is therefore stepped over — it still
/// occupies its full stride — rather than treated as the end of the block, which
/// is what this walk used to do (issue #299).
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
    grid: &ChunkGrid,
) -> Result<Vec<ChunkInfo>, FormatError> {
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

    let mut chunks = Vec::new();
    let mut pos = db_offset + db_header_size;
    // Every page occupies its full stride whether or not it was initialized, so
    // an uninitialized one is stepped over rather than treated as the end of the
    // block. The reference library populates pages in whatever order chunks are
    // written, and a dataset whose maximum shape is wider than its shape leaves
    // gaps between them — so the first cleared bit is not the last page (issue
    // #299). This is the same walk `fixed_array` does.
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
        // The page's checksum covers all `page_nelmts` slots, including any
        // beyond the published count that this read stops short of decoding.
        let page_end = pos
            .checked_add(page_stride)
            .filter(|&end| end <= file_data.len())
            .ok_or(FormatError::UnexpectedEof {
                expected: pos.saturating_add(page_stride),
                available: file_data.len(),
            })?;
        crate::checksum::verify_trailing(&file_data[pos..page_end])?;
        let page_start = start_index + page * page_nelmts;
        // Ignore element slots beyond the published count; see the same bound
        // in `read_data_block_elements`.
        let limit = total_elements.saturating_sub(page_start).min(page_nelmts);
        for i in 0..limit {
            let (info, consumed) = read_element(
                file_data,
                pos,
                header.client_id,
                header.element_size,
                offset_size,
                chunk_byte_size,
                page_start + i,
                grid,
            )?;
            if let Some(ci) = info {
                chunks.push(ci);
            }
            pos += consumed;
        }
        if limit < page_nelmts {
            break; // remaining slots/pages are beyond the dataset count
        }
        // Skip the page checksum.
        pos += 4;
    }

    Ok(chunks)
}

/// Read chunk records from an Extensible Array.
///
/// Traverses AEHD -> AEIB -> AEDB/AESB to collect all allocated chunks.
#[allow(clippy::too_many_arguments)]
pub fn read_extensible_array_chunks(
    file_data: &[u8],
    header: &ExtensibleArrayHeader,
    grid: &ChunkGrid,
    chunk_dimensions: &[u32],
    element_size: u32,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let os = offset_size as usize;

    let chunk_byte_size: u64 =
        chunk_dimensions.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;

    // Derive the (shared) extensible-array geometry from the header. This is
    // the same progression the writer uses, so reader and writer cannot drift.
    let geom = EaGeometry::from_header(header);

    // Parse index block (AEIB). Its length is fixed by the geometry -- every
    // inline slot and every block pointer is always written -- so the whole
    // block, checksum included, can be bounded before anything in it is read.
    let ib_offset = header.index_block_address.to_usize()?;
    let ib_header_size = 4 + 1 + 1 + offset_size as usize; // sig + ver + client + hdr_addr
    let ib_len = crate::chunked_write::aeib_size(
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

    let mut chunks = Vec::new();
    let mut global_index = 0usize;
    // A SWMR writer publishes the grown chunk index (header element count)
    // before the grown dataspace dimension, so an interrupted append leaves
    // elements the dataspace has not caught up with. Those are dropped one at a
    // time by `ChunkGrid::offsets_in_extent`, which is what makes the read a
    // consistent prefix; this count does not need to bound them as well. It
    // used to, and the walk is no faster for it — the structure traversal is
    // bounded by the array's own geometry, so a header naming ten billion
    // elements over two allocated blocks reads in the same 200us either way.
    let total_elements = header.num_elements.to_usize()?;

    // 1. Read inline elements in index block
    let n_inline = header.idx_blk_elmts as usize;
    for i in 0..n_inline {
        if global_index + i >= total_elements {
            break;
        }
        let (info, consumed) = read_element(
            file_data,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
            global_index + i,
            grid,
        )?;
        if let Some(ci) = info {
            chunks.push(ci);
        }
        pos += consumed;
    }
    global_index += n_inline.min(total_elements);

    // If all elements were inline, we're done
    if global_index >= total_elements {
        return Ok(chunks);
    }

    // 2. Direct data blocks: their addresses are listed in the index block,
    //    one per entry in `geom.direct_dblk_nelmts`.
    let mut direct_addrs: Vec<u64> = Vec::with_capacity(geom.direct_dblk_nelmts.len());
    for _ in 0..geom.direct_dblk_nelmts.len() {
        direct_addrs.push(read_offset(file_data, pos, offset_size)?);
        pos += os;
    }
    for (i, &addr) in direct_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let nelmts = geom.direct_dblk_nelmts[i].to_usize()?;
        if !is_undefined_addr(addr, offset_size) {
            let block_chunks = read_data_block_elements(
                file_data,
                addr.to_usize()?,
                nelmts,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                grid,
            )?;
            chunks.extend(block_chunks);
        }
        // Advance even for undefined blocks so linear indices stay aligned.
        global_index += nelmts;
    }

    // 3. Super blocks: the remaining `geom.nsblk_addrs` index-block entries are
    //    addresses of on-disk super blocks (`EASB`). Super-block pointer `j`
    //    refers to super block `first_indirect_sblk + j`.
    let mut sblk_addrs: Vec<u64> = Vec::with_capacity(geom.nsblk_addrs);
    for _ in 0..geom.nsblk_addrs {
        sblk_addrs.push(read_offset(file_data, pos, offset_size)?);
        pos += os;
    }
    for (j, &sb_addr) in sblk_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let sblk_idx = geom.first_indirect_sblk + j;
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let total_in_sb = (ndblks * dblk_nelmts).to_usize()?;
        if !is_undefined_addr(sb_addr, offset_size) {
            let sb_chunks = read_super_block(
                file_data,
                sb_addr.to_usize()?,
                ndblks.to_usize()?,
                dblk_nelmts.to_usize()?,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                grid,
            )?;
            chunks.extend(sb_chunks);
        }
        global_index += total_in_sb;
    }

    Ok(chunks)
}

/// Read a super block (AESB) and its data blocks.
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
    grid: &ChunkGrid,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let os = offset_size as usize;

    // AESB: signature(4) + version(1) + client_id(1) + header_address(offset_size)
    //       + block_offset(ceil(max_nelmts_bits/8))
    //       + [page-init bitmap, if data blocks are paged]
    //       + data block addresses
    //       + checksum
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let sb_header_size = 4 + 1 + 1 + os + blk_off_size;

    // A super block's data blocks are paged when their element count exceeds
    // the page size. When paged, a page-init bitmap precedes the data block
    // addresses. Its size is `ndblks * ceil(npages / 8)` bytes, and it is a
    // contiguous bit stream indexed by `db_local_idx * npages + page`.
    let page_nelmts = 1usize << header.max_dblk_nelmts_bits;
    let sb = SuperBlockGeom {
        ndblks: ndblks as u64,
        blocks: DataBlockGeom {
            dblk_nelmts: nelmts_per_dblk as u64,
            page_nelmts: page_nelmts as u64,
        },
    };
    let is_paged = sb.blocks.is_paged();
    let npages = sb.blocks.npages().to_usize()?;
    let bitmap_size = sb.bitmap_size().to_usize()?;

    // The whole super block -- header, bitmap, every data-block pointer, and
    // the checksum over them -- is fixed by the geometry, so it is bounded
    // before anything in it is read.
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
    let mut dblk_addrs: Vec<u64> = Vec::with_capacity(ndblks);
    for _ in 0..ndblks {
        let addr = read_offset(file_data, pos, offset_size)?;
        dblk_addrs.push(addr);
        pos += os;
    }

    let mut chunks = Vec::new();
    let mut global_idx = start_index;

    for (db_local, &addr) in dblk_addrs.iter().enumerate() {
        if !is_undefined_addr(addr, offset_size) {
            let block_chunks = if is_paged {
                read_paged_data_block(
                    file_data,
                    addr.to_usize()?,
                    page_nelmts,
                    npages,
                    db_local,
                    &page_bitmap,
                    header,
                    offset_size,
                    chunk_byte_size,
                    global_idx,
                    total_elements,
                    grid,
                )?
            } else {
                read_data_block_elements(
                    file_data,
                    addr.to_usize()?,
                    nelmts_per_dblk,
                    header,
                    offset_size,
                    chunk_byte_size,
                    global_idx,
                    total_elements,
                    grid,
                )?
            };
            chunks.extend(block_chunks);
        }
        global_idx += nelmts_per_dblk;
    }

    Ok(chunks)
}

/// On-disk byte spans `(addr, len)` of an Extensible Array chunk index's own
/// structure — the header (`EAHD`), the index block (`EAIB`), and every
/// allocated super block (`EASB`) and data block (`EADB`, paged or not) — for
/// reclaiming a deleted chunked dataset. The chunk *data* blocks the array
/// points at are enumerated separately (via [`read_extensible_array_chunks`]),
/// so they are not included here.
///
/// `ea_base` is the Extensible Array header address from the data-layout
/// message. The walk mirrors [`read_extensible_array_chunks`] exactly — same
/// [`EaGeometry`], same block addresses read from the index and super blocks —
/// but records each block's extent (sized by [`eadb_size`] / [`aesb_size`],
/// shared with the writer) instead of decoding chunk records. Only blocks with
/// a defined (non-`0xFF`) address are emitted, so a partially-grown array
/// contributes exactly its allocated blocks. The caller validates the spans
/// against the file bounds.
///
/// [`eadb_size`]: crate::chunked_write::eadb_size
/// [`aesb_size`]: crate::chunked_write::aesb_size
#[cfg(feature = "std")]
pub(crate) fn extensible_array_index_spans<S: Source + ?Sized>(
    source: &S,
    ea_base: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    use crate::chunked_write::{aesb_size, eadb_size};

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
    let aehd_size = ExtensibleArrayHeader::serialized_size(offset_size, length_size) as u64;
    let mut spans = vec![(ea_base, aehd_size)];

    if is_undefined_addr(header.index_block_address, offset_size) {
        return Ok(spans);
    }

    // EAIB index block. Its size — inline element slots plus direct- and
    // super-block address pointers — comes from the same geometry the writer
    // uses (see `build_extensible_array_at`).
    let geom = EaGeometry::from_header(&header);
    let ndblk_addrs = geom.direct_dblk_nelmts.len();
    let nsblk_addrs = geom.nsblk_addrs;
    let inline = header.idx_blk_elmts as usize;
    let ib_header = 4 + 1 + 1 + os; // sig + ver + client + hdr_addr
    let aeib_size =
        crate::chunked_write::aeib_size(offset_size, inline, elem_size, ndblk_addrs, nsblk_addrs);
    let ib_addr = header.index_block_address;
    // Read the index block as one bounded window; the read is also the bounds
    // check, so a block claiming to run past end-of-file errors here rather than
    // being recorded as reclaimable.
    let ib = source.read_metadata_at(ib_addr, aeib_size)?;
    if &ib[..4] != b"EAIB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array index block signature".into(),
        ));
    }
    // The addresses read out of this block become free space, so a block that
    // fails its checksum is refused here rather than reclaimed from: releasing
    // a span derived from bytes we cannot trust would hand live storage back to
    // the allocator.
    crate::checksum::verify_trailing(&ib)?;
    spans.push((ib_addr, aeib_size as u64));

    // Read the index block's direct data-block addresses, then its super-block
    // addresses, immediately following the inline element slots.
    let mut pos = ib_header + inline * elem_size;
    for &dblk_nelmts in &geom.direct_dblk_nelmts {
        let addr = read_offset(&ib, pos, offset_size)?;
        pos += os;
        if is_undefined_addr(addr, offset_size) {
            continue;
        }
        spans.push((
            addr,
            eadb_size(
                DataBlockGeom {
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
        let addr = read_offset(&ib, pos, offset_size)?;
        pos += os;
        if is_undefined_addr(addr, offset_size) {
            continue;
        }
        let sb = geom.super_block_at(j, page_nelmts);
        spans.push((addr, aesb_size(sb, offset_size, blk_off_size)));
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

/// Append the `(addr, len)` spans of the data blocks (`EADB`) reachable through
/// one super block (`EASB`) at `sb_addr`. Mirrors [`read_super_block`]'s address
/// reading (header, optional page-init bitmap, then `ndblks` data-block
/// addresses), emitting an extent for each defined address.
#[cfg(feature = "std")]
fn easb_data_block_spans<S: Source + ?Sized>(
    source: &S,
    sb_addr: u64,
    sb: SuperBlockGeom,
    offset_size: u8,
    blk_off_size: usize,
    elem_size: usize,
    spans: &mut Vec<(u64, u64)>,
) -> Result<(), FormatError> {
    use crate::chunked_write::{aesb_size, eadb_size};

    let os = offset_size as usize;
    let sb_header = 4 + 1 + 1 + os + blk_off_size; // sig + ver + client + hdr_addr + block_offset
    // The whole super block in one read: its checksum covers all of it, and the
    // addresses read out of it become free space, so one that fails is refused
    // rather than reclaimed from.
    let sb_len = aesb_size(sb, offset_size, blk_off_size).to_usize()?;
    let block = source.read_metadata_at(sb_addr, sb_len)?;
    if &block[..4] != b"EASB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array super block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&block)?;

    // A paged super block carries its page-init bitmap between the header and
    // the data-block addresses; `bitmap_size` is zero when it does not.
    let mut pos = sb_header + sb.bitmap_size().to_usize()?;
    for _ in 0..sb.ndblks {
        let addr = read_offset(&block, pos, offset_size)?;
        pos += os;
        if is_undefined_addr(addr, offset_size) {
            continue;
        }
        spans.push((
            addr,
            eadb_size(sb.blocks, elem_size, offset_size, blk_off_size),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming traversal (read each block from a `Source` on demand)
// ---------------------------------------------------------------------------

/// Read chunk records from an Extensible Array via a [`Source`].
///
/// Streaming counterpart of [`read_extensible_array_chunks`]: reads the index
/// block, then each direct data block and super block (and its paged data
/// blocks) as bounded windows via `read_at`. The shared `read_element` /
/// `EaGeometry` drive the decoding, so the layout logic stays in one place.
#[allow(clippy::too_many_arguments)]
pub fn read_extensible_array_chunks_from_source<S: Source + ?Sized>(
    source: &S,
    header: &ExtensibleArrayHeader,
    grid: &ChunkGrid,
    chunk_dimensions: &[u32],
    element_size: u32,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let os = offset_size as usize;

    let chunk_byte_size: u64 =
        chunk_dimensions.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;

    // See `read_extensible_array_chunks` on why the dataspace does not bound
    // this count.
    let total_elements = header.num_elements.to_usize()?;

    let geom = EaGeometry::from_header(header);
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
    let ib = source.read_metadata_at(header.index_block_address, ib_len)?;
    if &ib[0..4] != b"EAIB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array index block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&ib)?;

    let mut pos = ib_header_size;
    let mut chunks = Vec::new();
    let mut global_index = 0usize;

    // 1. Inline elements stored directly in the index block.
    for i in 0..n_inline {
        if global_index + i >= total_elements {
            break;
        }
        let (info, consumed) = read_element(
            &ib,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
            global_index + i,
            grid,
        )?;
        if let Some(ci) = info {
            chunks.push(ci);
        }
        pos += consumed;
    }
    global_index += n_inline.min(total_elements);
    if global_index >= total_elements {
        return Ok(chunks);
    }

    // After all inline slots, `pos` sits at the direct data-block addresses.
    let mut direct_addrs: Vec<u64> = Vec::with_capacity(ndirect);
    for _ in 0..ndirect {
        direct_addrs.push(read_offset(&ib, pos, offset_size)?);
        pos += os;
    }
    for (i, &addr) in direct_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let nelmts = geom.direct_dblk_nelmts[i].to_usize()?;
        if !is_undefined_addr(addr, offset_size) {
            chunks.extend(read_data_block_elements_from_source(
                source,
                addr,
                nelmts,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                grid,
            )?);
        }
        global_index += nelmts;
    }

    // 3. Super-block addresses.
    let mut sblk_addrs: Vec<u64> = Vec::with_capacity(nsblk);
    for _ in 0..nsblk {
        sblk_addrs.push(read_offset(&ib, pos, offset_size)?);
        pos += os;
    }
    for (j, &sb_addr) in sblk_addrs.iter().enumerate() {
        if global_index >= total_elements {
            break;
        }
        let sblk_idx = geom.first_indirect_sblk + j;
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let total_in_sb = (ndblks * dblk_nelmts).to_usize()?;
        if !is_undefined_addr(sb_addr, offset_size) {
            chunks.extend(read_super_block_from_source(
                source,
                sb_addr,
                ndblks.to_usize()?,
                dblk_nelmts.to_usize()?,
                header,
                offset_size,
                chunk_byte_size,
                global_index,
                total_elements,
                grid,
            )?);
        }
        global_index += total_in_sb;
    }

    Ok(chunks)
}

/// Collect elements from a (non-paged) data block read from the source.
#[allow(clippy::too_many_arguments)]
fn read_data_block_elements_from_source<S: Source + ?Sized>(
    source: &S,
    db_address: u64,
    nelmts: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    grid: &ChunkGrid,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let os = offset_size as usize;
    let db_header_size = 4 + 1 + 1 + os;
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let limit = total_elements.saturating_sub(start_index).min(nelmts);

    // The whole block, because its checksum covers every slot -- including the
    // ones past `limit` that this read does not decode. That is more than the
    // window this used to take, but a data block is bounded by the array's page
    // size, so it is one short read either way.
    let region_len = eadb_extent(nelmts, header, offset_size, blk_off_size)?;
    let block = source.read_metadata_at(db_address, region_len)?;
    if &block[0..4] != b"EADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array data block signature".into(),
        ));
    }
    crate::checksum::verify_trailing(&block)?;

    let mut pos = db_header_size + blk_off_size;
    let mut chunks = Vec::new();
    for i in 0..limit {
        let (info, consumed) = read_element(
            &block,
            pos,
            header.client_id,
            header.element_size,
            offset_size,
            chunk_byte_size,
            start_index + i,
            grid,
        )?;
        if let Some(ci) = info {
            chunks.push(ci);
        }
        pos += consumed;
    }
    Ok(chunks)
}

/// Read a paged Extensible Array data block from the source.
#[allow(clippy::too_many_arguments)]
fn read_paged_data_block_from_source<S: Source + ?Sized>(
    source: &S,
    db_address: u64,
    page_nelmts: usize,
    npages: usize,
    db_local_idx: usize,
    page_bitmap: &[u8],
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    grid: &ChunkGrid,
) -> Result<Vec<ChunkInfo>, FormatError> {
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

    // Read through the *last* initialized page rather than the first
    // uninitialized one: pages are not populated in order once the chunk grid
    // has gaps, and each occupies its full stride regardless (issue #299).
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
    // Through the last initialized page, checksums included. Every page is
    // written whole when the block is allocated, so this region is present in
    // any file whose pages the bitmap vouches for -- a short read here means a
    // truncated block, which is what the error says.
    let region_len = db_header_size + pages_bytes;
    let block = source.read_metadata_at(db_address, region_len)?;
    if block.len() < 4 || &block[0..4] != b"EADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Extensible Array data block signature".into(),
        ));
    }
    // A paged block checksums its header on its own, then each page separately.
    crate::checksum::verify_trailing(&block[..db_header_size])?;

    let mut chunks = Vec::new();
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
            let (info, consumed) = read_element(
                &block,
                pos,
                header.client_id,
                header.element_size,
                offset_size,
                chunk_byte_size,
                page_start + i,
                grid,
            )?;
            if let Some(ci) = info {
                chunks.push(ci);
            }
            pos += consumed;
        }
        if limit < page_nelmts {
            break;
        }
        pos += 4; // page checksum
    }
    Ok(chunks)
}

/// Read a super block (AESB) and its data blocks from the source.
#[allow(clippy::too_many_arguments)]
fn read_super_block_from_source<S: Source + ?Sized>(
    source: &S,
    sb_address: u64,
    ndblks: usize,
    nelmts_per_dblk: usize,
    header: &ExtensibleArrayHeader,
    offset_size: u8,
    chunk_byte_size: u64,
    start_index: usize,
    total_elements: usize,
    grid: &ChunkGrid,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let os = offset_size as usize;
    let blk_off_size = (header.max_nelmts_bits as usize).div_ceil(8);
    let sb_header_size = 4 + 1 + 1 + os + blk_off_size;

    let page_nelmts = 1usize << header.max_dblk_nelmts_bits;
    let sb = SuperBlockGeom {
        ndblks: ndblks as u64,
        blocks: DataBlockGeom {
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
    let block = source.read_metadata_at(sb_address, region_len)?;
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

    let mut dblk_addrs: Vec<u64> = Vec::with_capacity(ndblks);
    for _ in 0..ndblks {
        dblk_addrs.push(read_offset(&block, pos, offset_size)?);
        pos += os;
    }

    let mut chunks = Vec::new();
    let mut global_idx = start_index;
    for (db_local, &addr) in dblk_addrs.iter().enumerate() {
        if !is_undefined_addr(addr, offset_size) {
            let block_chunks = if is_paged {
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
                    grid,
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
                    grid,
                )?
            };
            chunks.extend(block_chunks);
        }
        global_idx += nelmts_per_dblk;
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The paged/not-paged boundary, which decides whether an `EASB` carries a
    /// page-init bitmap at all — so a reader and a writer that disagree by one
    /// element disagree about the position of every data-block address after it.
    ///
    /// Strictly greater, not greater-or-equal: a data block holding exactly one
    /// page's worth of elements is *not* paged, matching libhdf5's
    /// `H5EA__dblock_alloc`.
    #[test]
    fn a_data_block_is_paged_only_past_a_full_page() {
        let at = |dblk_nelmts| DataBlockGeom {
            dblk_nelmts,
            page_nelmts: 16,
        };
        assert!(!at(15).is_paged(), "under a page");
        assert!(!at(16).is_paged(), "exactly a page is not paged");
        assert!(at(17).is_paged(), "past a page");
        assert_eq!(at(16).npages(), 0, "a block that is not paged has no pages");
        assert_eq!(at(32).npages(), 2);
    }

    /// The bitmap is one bit per page per data block, each block's bits rounded
    /// up to a whole byte — so eight pages fit one byte and nine need two.
    #[test]
    fn the_page_init_bitmap_is_a_byte_per_eight_pages_per_block() {
        let sb = |ndblks, dblk_nelmts| SuperBlockGeom {
            ndblks,
            blocks: DataBlockGeom {
                dblk_nelmts,
                page_nelmts: 16,
            },
        };
        // Not paged: no bitmap at all, which is what makes the boundary above
        // decide a byte offset rather than merely a size.
        assert_eq!(sb(4, 16).bitmap_size(), 0);
        // 2 blocks x 8 pages each -> one byte per block.
        assert_eq!(sb(2, 8 * 16).bitmap_size(), 2);
        // 2 blocks x 9 pages each -> the round-up costs a second byte per block.
        assert_eq!(sb(2, 9 * 16).bitmap_size(), 4);
        // One block, two pages: a single byte, not two bits.
        assert_eq!(sb(1, 2 * 16).bitmap_size(), 1);
    }

    /// The grid of a dataset with no maximum shape: dense row-major, which is
    /// what these tests read. The numbering rule itself is tested in
    /// [`crate::chunk_grid`]; these tests are about the array structures.
    fn dense_grid(dims: &[u64], chunk_dims: &[u32]) -> ChunkGrid {
        let cd: Vec<u64> = chunk_dims.iter().map(|&d| u64::from(d)).collect();
        ChunkGrid::new(&cd, dims, None, crate::chunk_grid::GridOrder::RowMajor).unwrap()
    }

    use crate::checksum::stamp_trailing as stamp;

    /// EAHD: 12 fixed bytes, six length-sized statistics, the index-block
    /// address, and the checksum.
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
        buf[5] = 0; // client_id = non-filtered
        buf[6] = 8; // element_size
        buf[7] = 10; // max_nelmts_bits
        buf[8] = 2; // idx_blk_elmts
        buf[9] = 4; // min_dblk_nelmts
        buf[10] = 2; // super_blk_min_nelmts
        buf[11] = 8; // max_dblk_nelmts_bits
        // 6 stats fields (each 8 bytes)
        buf[12..20].copy_from_slice(&0u64.to_le_bytes()); // stat[0]
        buf[20..28].copy_from_slice(&0u64.to_le_bytes()); // stat[1]
        buf[28..36].copy_from_slice(&0u64.to_le_bytes()); // stat[2]
        buf[36..44].copy_from_slice(&0u64.to_le_bytes()); // stat[3]
        buf[44..52].copy_from_slice(&5u64.to_le_bytes()); // stat[4] = num_elements
        buf[52..60].copy_from_slice(&0u64.to_le_bytes()); // stat[5]
        buf[60..68].copy_from_slice(&0x1000u64.to_le_bytes()); // index_block_address
        stamp(&mut buf, 0, eahd_len(os, ls));

        let hdr = ExtensibleArrayHeader::parse(&buf, 0, os, ls).unwrap();
        assert_eq!(hdr.client_id, 0);
        assert_eq!(hdr.element_size, 8);
        assert_eq!(hdr.idx_blk_elmts, 2);
        assert_eq!(hdr.min_dblk_nelmts, 4);
        assert_eq!(hdr.num_elements, 5);
        assert_eq!(hdr.index_block_address, 0x1000);
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

    /// A grid with a dimension of no chunks reaches the reader as a refusal, not
    /// as a division by zero.
    ///
    /// A crafted dataspace can name a maximum extent of zero beside a non-zero
    /// current one. Before the maximum shape entered the numbering that file read
    /// as ordinary data; it must now be refused rather than panic, since
    /// `fuzz_targets/parse_file.rs` drives exactly this path.
    #[test]
    fn a_dimension_of_no_chunks_refuses_rather_than_dividing_by_zero() {
        let chunks = [crate::chunked_write::WrittenChunk {
            address: 0x1000,
            compressed_size: 8,
            filter_mask: 0,
        }];
        let slots = crate::chunked_write::IndexSlots::dense(&chunks);
        let ea =
            crate::chunked_write::build_extensible_array_at(&slots, 16, 8, 8, false, 0).unwrap();
        let header = ExtensibleArrayHeader::parse(&ea, 0, 8, 8).unwrap();

        // Maximum extent 0 in the trailing dimension, current extent 4.
        let grid = ChunkGrid::new(
            &[2, 2],
            &[3, 4],
            Some(&[u64::MAX, 0]),
            crate::chunk_grid::GridOrder::UnlimitedFirst,
        )
        .unwrap();
        let err = read_extensible_array_chunks(&ea, &header, &grid, &[2, 2], 4, 8, 8).unwrap_err();
        assert!(format!("{err}").contains("numbers nothing"), "{err}");
    }

    /// The Extensible Array twin of
    /// `fixed_array::tests::a_chunk_at_a_slot_outside_the_dataset_is_dropped`.
    ///
    /// The slot has to be one the walk actually reaches, so it is an *interior*
    /// slot rather than a trailing one: with the maximum shape wider than the
    /// shape in the dimension the rotation leaves behind, slots between the
    /// occupied ones decode to coordinates the dataset has not grown into.
    /// A trailing slot would be cut by the walk's own bound instead, leaving
    /// this check untested.
    #[test]
    fn a_chunk_at_an_interior_slot_outside_the_dataset_is_dropped() {
        // Shape [3, 3] with [2, 2] chunks, maximum [8, unlimited]. The rotation
        // puts the unlimited dimension first, so the multipliers are [4, 1] over
        // (column, row): the dataset's own corner chunk is slot 5, and slot 2 in
        // between decodes to chunk row 2 — offsets [4, 0], past a 3-row dataset.
        let chunks: Vec<crate::chunked_write::WrittenChunk> = [0x1000u64, 0x2000, 0x3000]
            .iter()
            .map(|&address| crate::chunked_write::WrittenChunk {
                address,
                compressed_size: 8,
                filter_mask: 0,
            })
            .collect();
        let slots = crate::chunked_write::IndexSlots::new(&chunks, &[0, 1, 2], 3).unwrap();
        let ea =
            crate::chunked_write::build_extensible_array_at(&slots, 16, 8, 8, false, 0).unwrap();

        let grid = ChunkGrid::new(
            &[2, 2],
            &[3, 3],
            Some(&[8, u64::MAX]),
            crate::chunk_grid::GridOrder::UnlimitedFirst,
        )
        .unwrap();
        assert_eq!(
            grid.offsets_in_extent(2).unwrap(),
            None,
            "slot 2 is the one"
        );

        let header = ExtensibleArrayHeader::parse(&ea, 0, 8, 8).unwrap();
        let read = read_extensible_array_chunks(&ea, &header, &grid, &[2, 2], 4, 8, 8).unwrap();
        assert_eq!(
            read.iter().map(|c| c.address).collect::<Vec<_>>(),
            vec![0x1000, 0x2000],
            "the slot-2 chunk lies past the dataset's three rows"
        );
    }

    /// Build a synthetic Extensible Array with only inline elements (simplest case).
    /// All chunks fit in the index block.
    #[test]
    fn read_inline_only() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let num_chunks = 2usize;
        let chunk_byte_size = 20u64 * 8; // 20 elements × 8 bytes

        let mut file_data = vec![0u8; 0x3000];

        // AEHD at offset 0x100
        let aehd_offset = 0x100usize;
        let aeib_offset = 0x200usize;

        // Build AEHD
        file_data[aehd_offset..aehd_offset + 4].copy_from_slice(b"EAHD");
        file_data[aehd_offset + 4] = 0; // version
        file_data[aehd_offset + 5] = 0; // client_id = non-filtered
        file_data[aehd_offset + 6] = osv as u8; // element_size
        file_data[aehd_offset + 7] = 10; // max_nelmts_bits
        file_data[aehd_offset + 8] = num_chunks as u8; // idx_blk_elmts (all inline)
        file_data[aehd_offset + 9] = 4; // min_dblk_nelmts
        file_data[aehd_offset + 10] = 2; // super_blk_min_nelmts
        file_data[aehd_offset + 11] = 8; // max_dblk_nelmts_bits
        // 6 stats fields (each 8 bytes), nelmts at stat[4]
        file_data[aehd_offset + 44..aehd_offset + 52]
            .copy_from_slice(&(num_chunks as u64).to_le_bytes());
        file_data[aehd_offset + 60..aehd_offset + 68]
            .copy_from_slice(&(aeib_offset as u64).to_le_bytes());
        stamp(&mut file_data, aehd_offset, eahd_len(os, ls));

        // Build AEIB at aeib_offset
        file_data[aeib_offset..aeib_offset + 4].copy_from_slice(b"EAIB");
        file_data[aeib_offset + 4] = 0; // version
        file_data[aeib_offset + 5] = 0; // client_id
        file_data[aeib_offset + 6..aeib_offset + 14]
            .copy_from_slice(&(aehd_offset as u64).to_le_bytes());

        // Inline elements
        let elem_start = aeib_offset + 6 + osv;
        let base_addr = 0x1000u64;
        for i in 0..num_chunks {
            let addr = base_addr + i as u64 * chunk_byte_size;
            let p = elem_start + i * osv;
            file_data[p..p + osv].copy_from_slice(&addr.to_le_bytes());
        }

        // The index block is sized by the header's geometry whatever the array
        // holds: prefix, two inline slots, then a pointer per direct data block
        // (min_dblk_nelmts=4 and super_blk_min_nelmts=2 give two) and one per
        // super block (nine levels less the two taken as direct, so seven).
        // Every pointer here is left zero -- the read stops at the inline slots.
        stamp(
            &mut file_data,
            aeib_offset,
            (6 + osv) + 2 * osv + 2 * osv + 7 * osv + 4,
        );

        let header = ExtensibleArrayHeader::parse(&file_data, aehd_offset, os, ls).unwrap();
        let ds_dims = vec![40u64]; // 2 chunks × 20 elements
        let chunk_dims = vec![20u32];
        let chunks = read_extensible_array_chunks(
            &file_data,
            &header,
            &dense_grid(&ds_dims, &chunk_dims),
            &chunk_dims,
            8,
            os,
            ls,
        )
        .unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].address, base_addr);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[0].chunk_size, chunk_byte_size as u32);
        assert_eq!(chunks[1].address, base_addr + chunk_byte_size);
        assert_eq!(chunks[1].offsets, vec![20]);

        #[cfg(feature = "std")]
        assert_ea_streams_match(&file_data, aehd_offset, &ds_dims, &chunk_dims, 8, os, ls);
    }

    /// Assert the streaming Extensible-Array reader matches the buffered one over
    /// both an in-memory and a `Read+Seek` source.
    #[cfg(feature = "std")]
    fn assert_ea_streams_match(
        file_data: &[u8],
        aehd_offset: usize,
        ds_dims: &[u64],
        chunk_dims: &[u32],
        element_size: u32,
        os: u8,
        ls: u8,
    ) {
        use crate::source::{BytesSource, ReadSeekSource};
        let h = ExtensibleArrayHeader::parse(file_data, aehd_offset, os, ls).unwrap();
        let buffered = read_extensible_array_chunks(
            file_data,
            &h,
            &dense_grid(ds_dims, chunk_dims),
            chunk_dims,
            element_size,
            os,
            ls,
        )
        .unwrap();

        let mem = BytesSource::new(file_data);
        let hm =
            ExtensibleArrayHeader::parse_from_source(&mem, aehd_offset as u64, os, ls).unwrap();
        let from_mem = read_extensible_array_chunks_from_source(
            &mem,
            &hm,
            &dense_grid(ds_dims, chunk_dims),
            chunk_dims,
            element_size,
            os,
            ls,
        )
        .unwrap();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data.to_vec())).unwrap();
        let hs =
            ExtensibleArrayHeader::parse_from_source(&seek, aehd_offset as u64, os, ls).unwrap();
        let from_seek = read_extensible_array_chunks_from_source(
            &seek,
            &hs,
            &dense_grid(ds_dims, chunk_dims),
            chunk_dims,
            element_size,
            os,
            ls,
        )
        .unwrap();

        assert_eq!(buffered, from_mem, "BytesSource mismatch");
        assert_eq!(buffered, from_seek, "ReadSeekSource mismatch");
    }

    /// Build a synthetic EA with inline elements + one direct data block.
    #[test]
    fn read_inline_plus_data_blocks() {
        let os: u8 = 8;
        let ls: u8 = 8;
        let osv = os as usize;
        let chunk_byte_size = 10u64 * 8; // 10 elements × 8 bytes
        let idx_blk_elmts = 2u8;
        let min_dblk_nelmts = 2u8;
        let sblk_min = 2u8;
        let total_chunks = 4usize; // 2 inline + 2 in data block (1 dblk from sb_level 0)

        let mut file_data = vec![0u8; 0x5000];
        let aehd_offset = 0x100usize;
        let aeib_offset = 0x200usize;
        let aedb_offset = 0x300usize;

        // EAHD
        file_data[aehd_offset..aehd_offset + 4].copy_from_slice(b"EAHD");
        file_data[aehd_offset + 4] = 0;
        file_data[aehd_offset + 5] = 0; // client_id
        file_data[aehd_offset + 6] = osv as u8; // element_size
        file_data[aehd_offset + 7] = 10;
        file_data[aehd_offset + 8] = idx_blk_elmts;
        file_data[aehd_offset + 9] = min_dblk_nelmts;
        file_data[aehd_offset + 10] = sblk_min;
        file_data[aehd_offset + 11] = 8;
        // 6 stats fields (each 8 bytes), nelmts at stat[4] (offset 12 + 4*8 = 44)
        file_data[aehd_offset + 44..aehd_offset + 52]
            .copy_from_slice(&(total_chunks as u64).to_le_bytes());
        // idx_blk_addr at offset 12 + 6*8 = 60
        file_data[aehd_offset + 60..aehd_offset + 68]
            .copy_from_slice(&(aeib_offset as u64).to_le_bytes());
        stamp(&mut file_data, aehd_offset, eahd_len(os, ls));

        // AEIB
        file_data[aeib_offset..aeib_offset + 4].copy_from_slice(b"EAIB");
        file_data[aeib_offset + 4] = 0;
        file_data[aeib_offset + 5] = 0;
        file_data[aeib_offset + 6..aeib_offset + 14]
            .copy_from_slice(&(aehd_offset as u64).to_le_bytes());

        let mut pos = aeib_offset + 6 + osv;

        // Inline elements (2 chunks)
        let base_addr = 0x1000u64;
        for i in 0..idx_blk_elmts as usize {
            let addr = base_addr + i as u64 * chunk_byte_size;
            file_data[pos..pos + osv].copy_from_slice(&addr.to_le_bytes());
            pos += osv;
        }

        // Direct data-block pointers. The first `super_blk_min_nelmts` super
        // blocks of the doubling table are stored directly in the index block
        // rather than through a super block, and with `min_dblk_nelmts = 2`
        // those are (1 block of 2) and (1 block of 4) -- two pointers. Only the
        // first holds data; the array's four elements are two inline and two
        // here.
        let n_direct_dblks = 2;
        file_data[pos..pos + osv].copy_from_slice(&(aedb_offset as u64).to_le_bytes());
        pos += osv;
        for _ in 1..n_direct_dblks {
            file_data[pos..pos + osv].copy_from_slice(&u64::MAX.to_le_bytes());
            pos += osv;
        }

        // EADB at aedb_offset (min_dblk_nelmts elements)
        file_data[aedb_offset..aedb_offset + 4].copy_from_slice(b"EADB");
        file_data[aedb_offset + 4] = 0;
        file_data[aedb_offset + 5] = 0;
        file_data[aedb_offset + 6..aedb_offset + 14]
            .copy_from_slice(&(aehd_offset as u64).to_le_bytes());
        // block_offset: ceil(max_nelmts_bits/8) = ceil(10/8) = 2 bytes
        // block_offset = 0 for first data block
        let blk_off_size = (10usize).div_ceil(8); // max_nelmts_bits=10
        let mut dbpos = aedb_offset + 6 + osv + blk_off_size;
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
            aeib_offset,
            (6 + osv) + 2 * osv + n_direct_dblks * osv + 8 * osv + 4,
        );
        // Prefix, block offset, `min_dblk_nelmts` element slots, checksum.
        stamp(
            &mut file_data,
            aedb_offset,
            (6 + osv) + blk_off_size + min_dblk_nelmts as usize * osv + 4,
        );

        let header = ExtensibleArrayHeader::parse(&file_data, aehd_offset, os, ls).unwrap();
        let ds_dims = vec![40u64];
        let chunk_dims = vec![10u32];
        let chunks = read_extensible_array_chunks(
            &file_data,
            &header,
            &dense_grid(&ds_dims, &chunk_dims),
            &chunk_dims,
            8,
            os,
            ls,
        )
        .unwrap();

        assert_eq!(chunks.len(), 4);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, base_addr + i as u64 * chunk_byte_size);
            assert_eq!(c.offsets, vec![i as u64 * 10]);
        }

        #[cfg(feature = "std")]
        assert_ea_streams_match(&file_data, aehd_offset, &ds_dims, &chunk_dims, 8, os, ls);
    }

    /// A large Extensible Array built by the writer reaches direct data blocks,
    /// super blocks, and paged super-block data blocks; the streaming reader
    /// must reproduce the buffered read across all those layers. Driving the
    /// layout from the writer avoids hand-building the doubling-table geometry.
    #[cfg(feature = "std")]
    #[test]
    fn streaming_ea_super_blocks_and_paged_match_buffered() {
        use crate::chunked_write::{WrittenChunk, build_extensible_array_at};
        use crate::source::{BytesSource, ReadSeekSource};

        // n covers: inline+direct (2000), several super blocks (50000), and
        // paged super-block data blocks (140000, since dblk_nelmts exceeds the
        // 1024-element page size at the higher super blocks).
        for &n in &[2000u64, 50000, 140000] {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x10 + i * 8,
                    compressed_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let base = 0x1000u64;
            let ea = build_extensible_array_at(
                &crate::chunked_write::IndexSlots::dense(&chunks),
                8,
                8,
                8,
                false,
                base,
            )
            .unwrap();
            let mut file = vec![0u8; base as usize + ea.len()];
            file[base as usize..].copy_from_slice(&ea);

            let ds_dims = vec![n];
            let chunk_dims = vec![1u32];
            let header = ExtensibleArrayHeader::parse(&file, base as usize, 8, 8).unwrap();
            let buffered = read_extensible_array_chunks(
                &file,
                &header,
                &dense_grid(&ds_dims, &chunk_dims),
                &chunk_dims,
                8,
                8,
                8,
            )
            .unwrap();
            assert_eq!(buffered.len() as u64, n, "buffered chunk count at n={n}");

            let mem = BytesSource::new(&file);
            let hm = ExtensibleArrayHeader::parse_from_source(&mem, base, 8, 8).unwrap();
            let from_mem = read_extensible_array_chunks_from_source(
                &mem,
                &hm,
                &dense_grid(&ds_dims, &chunk_dims),
                &chunk_dims,
                8,
                8,
                8,
            )
            .unwrap();

            let seek = ReadSeekSource::new(std::io::Cursor::new(file)).unwrap();
            let hs = ExtensibleArrayHeader::parse_from_source(&seek, base, 8, 8).unwrap();
            let from_seek = read_extensible_array_chunks_from_source(
                &seek,
                &hs,
                &dense_grid(&ds_dims, &chunk_dims),
                &chunk_dims,
                8,
                8,
                8,
            )
            .unwrap();

            assert_eq!(buffered, from_mem, "BytesSource mismatch at n={n}");
            assert_eq!(buffered, from_seek, "ReadSeekSource mismatch at n={n}");
        }
    }

    /// Test serialized_size computation.
    /// [`eadb_extent`] must agree with [`crate::chunked_write::eadb_size`], the
    /// writer's own sizing, for every non-paged block.
    ///
    /// Not the thing that catches a divergence: a reader that sizes a block
    /// four bytes short of the writer hashes the wrong bytes and refuses every
    /// sound file, which reddens twenty-eight tests here. What this adds is the
    /// name of the rule that broke, and the widths a round trip does not write
    /// — a 4-byte-offset file, a 32-bit block-offset field, and a block of no
    /// elements at all.
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
                        super_blk_min_nelmts: 2,
                        max_dblk_nelmts_bits: 10,
                        num_elements: 0,
                        index_block_address: 0,
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
                            crate::chunked_write::eadb_size(
                                DataBlockGeom {
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
        // 12 fixed + 6*8 stats + 8 addr + 4 checksum = 72
        assert_eq!(ExtensibleArrayHeader::serialized_size(8, 8), 72);
        // 12 fixed + 6*4 stats + 4 addr + 4 checksum = 44
        assert_eq!(ExtensibleArrayHeader::serialized_size(4, 4), 44);
    }

    /// Verify read_element for unallocated slots.
    #[test]
    fn read_element_unallocated() {
        let data = vec![0xFFu8; 16];
        let ds_dims = vec![50u64];
        let chunk_dims = vec![10u32];
        let (info, consumed) =
            read_element(&data, 0, 0, 8, 8, 80, 0, &dense_grid(&ds_dims, &chunk_dims)).unwrap();
        assert!(info.is_none());
        assert_eq!(consumed, 8);
    }

    /// Verify filtered element reading.
    #[test]
    fn read_element_filtered() {
        let os: u8 = 8;
        let chunk_size_bytes = 4usize;
        let elem_size = os as usize + chunk_size_bytes + 4;
        let mut data = vec![0u8; elem_size + 16];
        // Address
        data[0..8].copy_from_slice(&0x2000u64.to_le_bytes());
        // Compressed size (4 bytes LE)
        data[8..12].copy_from_slice(&120u32.to_le_bytes());
        // Filter mask
        data[12..16].copy_from_slice(&0u32.to_le_bytes());

        let ds_dims = vec![50u64];
        let chunk_dims = vec![10u32];
        let (info, consumed) = read_element(
            &data,
            0,
            1,
            elem_size as u8,
            os,
            80,
            2,
            &dense_grid(&ds_dims, &chunk_dims),
        )
        .unwrap();
        let ci = info.unwrap();
        assert_eq!(ci.address, 0x2000);
        assert_eq!(ci.chunk_size, 120);
        assert_eq!(ci.filter_mask, 0);
        assert_eq!(ci.offsets, vec![20]);
        assert_eq!(consumed, elem_size);
    }

    /// A filtered element record holds an address, a chunk size, and a 4-byte
    /// filter mask, so its width cannot be smaller than `offset_size + 4`. The
    /// width comes from the EAHD's own `element_size` byte, which no parse
    /// validates, so a crafted header can name a smaller one. That must be
    /// refused: the size arithmetic used to underflow, which panics under the
    /// overflow checks that `cargo test` and the fuzz targets build with.
    #[test]
    fn filtered_element_smaller_than_its_own_fields_is_refused() {
        let os: u8 = 8;
        let ds_dims = vec![50u64];
        let chunk_dims = vec![10u32];
        let data = vec![0u8; 64];

        // `element_size` must be at least os + 4 = 12 to hold what it claims.
        for element_size in 0..(os + 4) {
            let err = read_element(
                &data,
                0,
                1,
                element_size,
                os,
                80,
                0,
                &dense_grid(&ds_dims, &chunk_dims),
            )
            .expect_err("a filtered element narrower than its own fields must be refused");
            assert!(
                matches!(err, FormatError::ChunkedReadError(_)),
                "element_size {element_size} gave {err:?}, want a ChunkedReadError"
            );
        }

        // The first width that can hold them is accepted.
        read_element(
            &data,
            0,
            1,
            os + 4 + 1,
            os,
            80,
            0,
            &dense_grid(&ds_dims, &chunk_dims),
        )
        .expect("a width that fits address + 1-byte size + mask must parse");
    }

    /// A truncated super-block address array must be refused by both backends.
    ///
    /// The buffered reader used to `break` out of the address loop on a short
    /// read and return `Ok` with a silently shortened chunk list, while the
    /// streaming reader returned `UnexpectedEof` for the identical bytes -- so
    /// the same file decoded two different ways depending on whether it was
    /// opened with `File::open` or `File::open_streaming`. The error is
    /// asserted down to the offset it faults at, so this cannot pass on an
    /// unrelated failure earlier in the walk.
    #[cfg(feature = "std")]
    #[test]
    fn truncated_super_block_addresses_are_refused_by_both_backends() {
        use crate::chunked_write::{WrittenChunk, build_extensible_array_at};
        use crate::source::BytesSource;

        let n = 100u64;
        let chunks: Vec<WrittenChunk> = (0..n)
            .map(|i| WrittenChunk {
                address: 0x10 + i * 8,
                compressed_size: 8,
                filter_mask: 0,
            })
            .collect();
        let base = 0x1000u64;
        let ea = build_extensible_array_at(
            &crate::chunked_write::IndexSlots::dense(&chunks),
            8,
            8,
            8,
            false,
            base,
        )
        .unwrap();
        let mut file = vec![0u8; base as usize + ea.len()];
        file[base as usize..].copy_from_slice(&ea);

        let ds_dims = vec![n];
        let chunk_dims = vec![1u32];
        let built = ExtensibleArrayHeader::parse(&file, base as usize, 8, 8).unwrap();
        let geom = EaGeometry::from_header(&built);
        assert!(
            geom.nsblk_addrs > 0,
            "fixture must reach the super-block address array"
        );

        // This crate's writer lays the index block down *before* the data
        // blocks it points at, so truncating into the block's address array
        // also truncates those data blocks and the walk faults on one of them
        // first. The format does not require that order, and a file whose
        // index block trails its data blocks is what reaches the address loop
        // with everything before it intact -- so move the index block to the
        // end and repoint the header at it. The index block is copied byte for
        // byte and its checksum covers only its own bytes, so the copy stays
        // valid; the header is edited, so it is restamped below.
        let old_ib = built.index_block_address.to_usize().unwrap();
        let ib_len = 4 + 1 + 1 + 8                            // sig + ver + client + header addr
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
        assert_eq!(header.index_block_address as usize, new_ib);
        // The relocated file must still read identically before it is cut.
        let intact = read_extensible_array_chunks(
            &file,
            &header,
            &dense_grid(&ds_dims, &chunk_dims),
            &chunk_dims,
            8,
            8,
            8,
        )
        .unwrap();
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

        // Cut the file in the middle of that first address. Every data block
        // now lies before the cut, so the index block is what faults -- both
        // backends bound the whole block before reading any of it, because the
        // checksum they verify covers all of it.
        file.truncate(sblk_start + 4);

        let buffered = read_extensible_array_chunks(
            &file,
            &header,
            &dense_grid(&ds_dims, &chunk_dims),
            &chunk_dims,
            8,
            8,
            8,
        );
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
            other => panic!("buffered read must refuse a truncated address array, got {other:?}"),
        }

        let mem = BytesSource::new(&file);
        let hm = ExtensibleArrayHeader::parse_from_source(&mem, base, 8, 8).unwrap();
        let streamed = read_extensible_array_chunks_from_source(
            &mem,
            &hm,
            &dense_grid(&ds_dims, &chunk_dims),
            &chunk_dims,
            8,
            8,
            8,
        );
        assert!(
            streamed.is_err(),
            "streaming read must refuse the same file, got {streamed:?}"
        );
    }

    /// Every checksummed structure of an Extensible Array is verified on read,
    /// by both backends (issue #312).
    ///
    /// The array is walked by [`extensible_array_index_spans`], so the sweep
    /// reaches whatever the writer laid down rather than a list written here:
    /// the header, the index block, every super block, and every data block,
    /// paged and not. Each is corrupted in its stored checksum and, for a paged
    /// data block, in the separate checksum over its prefix -- bytes that carry
    /// no other meaning, so a refusal can only be the checksum. Before this the
    /// reader returned the same chunk list it did for the sound file.
    /// Refusal is what the `checksum` feature buys, so this asserts it only
    /// where it is compiled in: with the feature off `verify_trailing` is a
    /// no-op and a corrupt index reads as it did before.
    #[cfg(all(feature = "std", feature = "checksum"))]
    #[test]
    fn a_corrupted_extensible_array_structure_is_refused() {
        use crate::chunked_write::{WrittenChunk, build_extensible_array_at};
        use crate::source::BytesSource;

        // The same progression as `streaming_ea_super_blocks_and_paged_match_buffered`:
        // inline and direct blocks, then super blocks, then paged data blocks.
        for &n in &[2000u64, 50000, 140000] {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x10 + i * 8,
                    compressed_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let base = 0x1000u64;
            let ea = build_extensible_array_at(
                &crate::chunked_write::IndexSlots::dense(&chunks),
                8,
                8,
                8,
                false,
                base,
            )
            .unwrap();
            let mut file = vec![0u8; base as usize + ea.len()];
            file[base as usize..].copy_from_slice(&ea);

            let ds_dims = vec![n];
            let chunk_dims = vec![1u32];
            let read_both = |file: &[u8]| -> (Result<Vec<ChunkInfo>, FormatError>, bool) {
                let grid = dense_grid(&ds_dims, &chunk_dims);
                let buffered =
                    ExtensibleArrayHeader::parse(file, base as usize, 8, 8).and_then(|h| {
                        read_extensible_array_chunks(file, &h, &grid, &chunk_dims, 8, 8, 8)
                    });
                let mem = BytesSource::new(file);
                let streamed =
                    ExtensibleArrayHeader::parse_from_source(&mem, base, 8, 8).and_then(|h| {
                        read_extensible_array_chunks_from_source(
                            &mem,
                            &h,
                            &grid,
                            &chunk_dims,
                            8,
                            8,
                            8,
                        )
                    });
                (buffered, streamed.is_err())
            };
            assert_eq!(
                read_both(&file).0.expect("the sound file must read").len() as u64,
                n,
                "the fixture must read before it is corrupted, at n={n}"
            );

            let spans = extensible_array_index_spans(&BytesSource::new(&file), base, 8, 8).unwrap();
            assert!(spans.len() > 1, "the sweep must reach past the header");

            let header = ExtensibleArrayHeader::parse(&file, base as usize, 8, 8).unwrap();
            let stride = ea_elem_stride(&header, 8);
            let page_nelmts = 1usize << header.max_dblk_nelmts_bits;
            // A data block's bytes before any checksum, and the largest an
            // unpaged one can be -- a paged block is always longer, since it
            // holds at least two full pages.
            let db_prefix = 4 + 1 + 1 + 8 + (header.max_nelmts_bits as usize).div_ceil(8);
            let max_unpaged = db_prefix + page_nelmts * stride + 4;

            // Each site is a byte to flip and whether the reclaim walk reads
            // the structure it belongs to. That walk turns what it reads into
            // free space, so where it does read one, a corrupt one must stop it
            // rather than release a span derived from bytes nothing vouched
            // for. It reads pointers out of the header, index block and super
            // blocks; it sizes data blocks from the header without reading
            // them.
            let mut poke_sites: Vec<(u64, bool)> = Vec::new();
            let mut paged = 0;
            for &(at, len) in &spans {
                let start = at.to_usize().unwrap();
                let kind = &file[start..start + 4];
                if kind == b"EADB" && len as usize > max_unpaged {
                    paged += 1;
                    // A paged block checksums its prefix on its own and then
                    // each page. Poke the prefix and the *first* page: a
                    // trailing page the bitmap never marked initialized is one
                    // the reader steps over without reading, so its checksum is
                    // not a soundness property and pinning it here would assert
                    // more than the format requires.
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
                    "n={n}: a corrupted checksum at {at:#x} must be refused, got {buffered:?}"
                );
                assert!(
                    streamed_err,
                    "n={n}: the streaming backend must refuse what the buffered one does, at {at:#x}"
                );
                if walked {
                    let walk = extensible_array_index_spans(&BytesSource::new(&file), base, 8, 8);
                    assert!(
                        matches!(walk, Err(FormatError::ChecksumMismatch { .. })),
                        "n={n}: the reclaim walk must refuse a corrupt structure at {at:#x} \
                         rather than release spans read out of it, got {walk:?}"
                    );
                }
                file[at] = original;
            }
        }
    }

    /// `extensible_array_index_spans` must account for exactly the index
    /// structure the writer lays down: its total must equal the header, index
    /// block, and the super-/data-block bytes recorded in the EAHD statistics,
    /// and the spans must be pairwise disjoint and lie within the built blob.
    /// This pins the reclaim walk to `build_extensible_array_at` so the two
    /// cannot drift (a too-large span would over-reclaim live bytes on delete).
    #[cfg(feature = "std")]
    #[test]
    fn index_spans_match_builder_layout() {
        use crate::chunked_write::{WrittenChunk, build_extensible_array_at};

        let os: u8 = 8;
        let ls: u8 = 8;
        let base = 0x4000u64;
        // Sizes spanning inline-only, direct data blocks, super blocks, and the
        // paged regime — every branch of the walk.
        for &n in &[1u64, 4, 20, 100, 244, 300, 2000, 50000, 140000] {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x100000 + i * 8,
                    compressed_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let ea = build_extensible_array_at(
                &crate::chunked_write::IndexSlots::dense(&chunks),
                8,
                os,
                ls,
                false,
                base,
            )
            .unwrap();
            let mut file = vec![0u8; base as usize + ea.len()];
            file[base as usize..].copy_from_slice(&ea);

            let spans =
                extensible_array_index_spans(&crate::source::BytesSource::new(&file), base, os, ls)
                    .unwrap();

            // Expected total = EAHD + EAIB + super_blk_size + data_blk_size, the
            // last two read straight from the statistics the builder wrote.
            let header = ExtensibleArrayHeader::parse(&file, base as usize, os, ls).unwrap();
            let geom = EaGeometry::from_header(&header);
            let aehd = ExtensibleArrayHeader::serialized_size(os, ls) as u64;
            // Unfiltered EA: element stride equals the offset size.
            let aeib = crate::chunked_write::aeib_size(
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
            let expected_total = aehd + aeib + super_blk_size + data_blk_size;

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
}
