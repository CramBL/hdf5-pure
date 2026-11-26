use alloc::vec::Vec;
use core::num::NonZeroU64;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2::BTreeV2Record;
use crate::btree::v2::integer;
use crate::bytes;
use crate::chunk_record;
use crate::data_layout::LayoutVersion;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The dataset context needed to decode a version 2 B-tree chunk record.
///
/// Types 10 and 11 store scaled chunk coordinates for the dataset dimensions. Type 10 omits
/// chunk size and filter-mask fields because its chunks are unfiltered and have the fixed logical
/// chunk size. Type 11 stores both. For version 4 layout messages, the stored-size field is one
/// byte wider than the fewest bytes needed for the logical chunk size, capped at 8 bytes.
/// `H5D_BT2_COMPUTE_CHUNK_SIZE_LEN` in `H5Dbtree2.c` (HDF5 2.2.0) applies that rule.
///
/// The two record layouts are defined in "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BTreeV2ChunkRecordContext {
    tree_type: u8,
    offset_width: OffsetWidth,
    rank: usize,
    fixed_chunk_size: NonZeroU64,
    stored_size_width: usize,
    record_size: usize,
}

impl BTreeV2ChunkRecordContext {
    /// Creates a context for a type 10 or type 11 chunk index.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeNodeType`] if `tree_type` is not 10 or 11,
    /// [`FormatError::InvalidLayoutVersion`] if `layout_version` is not 4, and
    /// [`FormatError::InvalidChunkGeometry`] if the record size cannot be represented.
    pub fn new(
        tree_type: u8,
        offset_width: OffsetWidth,
        rank: usize,
        fixed_chunk_size: NonZeroU64,
        layout_version: LayoutVersion,
    ) -> Result<Self, FormatError> {
        if !matches!(tree_type, BTREE_V2_CHUNK | BTREE_V2_FILTERED_CHUNK) {
            return Err(FormatError::InvalidBTreeNodeType(tree_type));
        }
        if layout_version != LayoutVersion::Four {
            return Err(FormatError::InvalidLayoutVersion(layout_version.get()));
        }

        let stored_size_width = if tree_type == BTREE_V2_FILTERED_CHUNK {
            chunk_record::chunk_element_encoding(fixed_chunk_size.get(), offset_width, true)
                .chunk_size_bytes
        } else {
            0
        };
        let coordinate_bytes =
            rank.checked_mul(CHUNK_COORDINATE_LEN)
                .ok_or(FormatError::InvalidChunkGeometry(
                    "version 2 B-tree chunk record size overflows",
                ))?;
        let record_size = usize::from(offset_width.get())
            .checked_add(stored_size_width)
            .and_then(|size| {
                size.checked_add(if tree_type == BTREE_V2_FILTERED_CHUNK {
                    FILTER_MASK_LEN
                } else {
                    0
                })
            })
            .and_then(|size| size.checked_add(coordinate_bytes))
            .ok_or(FormatError::InvalidChunkGeometry(
                "version 2 B-tree chunk record size overflows",
            ))?;
        if record_size > usize::from(u16::MAX) {
            return Err(FormatError::InvalidChunkGeometry(
                "version 2 B-tree chunk record is too large",
            ));
        }

        Ok(Self {
            tree_type,
            offset_width,
            rank,
            fixed_chunk_size,
            stored_size_width,
            record_size,
        })
    }

    /// Returns the exact encoded size of one record.
    pub fn record_size(self) -> usize {
        self.record_size
    }

    /// Requires the B-tree header's declared record size to match this record layout.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if `actual` differs from the exact record size.
    pub fn require_record_size(self, actual: u16) -> Result<(), FormatError> {
        let actual = usize::from(actual);
        if actual != self.record_size {
            return Err(FormatError::DataSizeMismatch {
                expected: self.record_size,
                actual,
            });
        }
        Ok(())
    }

    /// Decodes a raw record as the chunk-index client type this context describes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] unless the record has exactly the expected
    /// length, [`FormatError::ChunkedReadError`] for an undefined chunk address, and
    /// [`FormatError::InvalidChunkGeometry`] for a zero stored size in a filtered record.
    pub fn decode(self, record: &BTreeV2Record) -> Result<BTreeV2ChunkRecord, FormatError> {
        if record.data.len() != self.record_size {
            return Err(FormatError::DataSizeMismatch {
                expected: self.record_size,
                actual: record.data.len(),
            });
        }

        let address = bytes::read_optional_offset_width(&record.data, 0, self.offset_width)?
            .ok_or_else(|| {
                FormatError::ChunkedReadError(
                    "version 2 B-tree chunk record has an undefined chunk address".into(),
                )
            })?;
        let mut pos = usize::from(self.offset_width.get());
        let (stored_size, filter_mask) = if self.tree_type == BTREE_V2_FILTERED_CHUNK {
            let stored_size = integer::read_var_uint(&record.data, pos, self.stored_size_width)?;
            if stored_size == 0 {
                return Err(FormatError::InvalidChunkGeometry(
                    "filtered version 2 B-tree chunk record has zero stored size",
                ));
            }
            pos += self.stored_size_width;
            let mut cursor = bytes::ByteCursor::with_position(&record.data, pos)?;
            let filter_mask = cursor.u32()?;
            pos = cursor.position();
            (stored_size, filter_mask)
        } else {
            (self.fixed_chunk_size.get(), 0)
        };

        let mut scaled_offsets = Vec::with_capacity(self.rank);
        for _ in 0..self.rank {
            scaled_offsets.push(integer::read_var_uint(
                &record.data,
                pos,
                CHUNK_COORDINATE_LEN,
            )?);
            pos += CHUNK_COORDINATE_LEN;
        }

        Ok(BTreeV2ChunkRecord {
            address: StoredAddress::new(address),
            stored_size,
            filter_mask,
            scaled_offsets,
        })
    }
}

/// A decoded type 10 or type 11 record of a version 2 B-tree chunk index.
///
/// The B-tree stores chunk coordinates scaled by the dataset's chunk dimensions. The reader
/// converts [`scaled_offsets`](Self::scaled_offsets) to ordinary element offsets before exposing
/// the chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeV2ChunkRecord {
    /// The address of the chunk in the file.
    pub address: StoredAddress,
    /// The size of the chunk in the file in bytes, after its filters.
    pub stored_size: u64,
    /// The filters skipped for the chunk: bit `i` is set where filter `i` was not applied.
    pub filter_mask: u32,
    /// The chunk coordinates, in units of whole chunks, for each dataset dimension.
    pub scaled_offsets: Vec<u64>,
}

/// The type of a version 2 B-tree that indexes unfiltered dataset chunks, from the same table as
/// [`BTREE_V2_HUGE_OBJECT`](super::BTREE_V2_HUGE_OBJECT).
pub const BTREE_V2_CHUNK: u8 = 10;

/// The type of a version 2 B-tree that indexes filtered dataset chunks, from the same table as
/// [`BTREE_V2_HUGE_OBJECT`](super::BTREE_V2_HUGE_OBJECT).
pub const BTREE_V2_FILTERED_CHUNK: u8 = 11;

/// The width of one scaled chunk coordinate in a type 10 or type 11 record.
const CHUNK_COORDINATE_LEN: usize = 8;

/// The width of the filter mask in a type 11 record.
const FILTER_MASK_LEN: usize = 4;

#[cfg(test)]
#[path = "tests/chunk.rs"]
mod tests;
