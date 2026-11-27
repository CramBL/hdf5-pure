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
//! A caller reads the bytes of each range the parser returns and passes them back to it.
//! [`ExtensibleArrayHeaderLocator`] returns the range of the header, and each reference to a block
//! or a page, from [`ExtensibleArrayIndexBlockRef`] to [`ExtensibleArrayPageRef`], the range a
//! caller reads for it. The reference to a super block or a data block also returns whether the
//! block holds a slot below the header's "Max Index Set", one more than the highest slot the array
//! has set.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr

use alloc::format;
use alloc::vec::Vec;

use self::write::ExtensibleArrayPlan;
use crate::address::StoredAddress;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::IndexSlots;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[cfg(test)]
mod measurement_sizes;

mod block;
mod creation;
mod data_block;
mod elements;
mod geometry;
mod header;
mod index_block;
mod page;
mod secondary_block;
mod statistics;
pub(crate) mod write;

pub use block::ExtensibleArrayBlockFraming;
use block::Owner;
pub use creation::{
    BlockOffsetWidth, ExtensibleArrayBitGeometry, ExtensibleArrayCreationParameters, MaxIndexBits,
    PageElementCount,
};
pub(crate) use creation::{EA_IDX_BLK_ELMTS, EA_MAX_NELMTS_BITS, EA_MIN_DBLK_NELMTS};
use data_block::DATA_BLOCK_SIGNATURE;
pub use data_block::{ExtensibleArrayDataBlock, ExtensibleArrayDataBlockRef};
pub use elements::ExtensibleArrayElements;
use geometry::{DataBlock, Layout, SuperBlock};
pub use geometry::{DataBlockGeometry, ExtensibleArrayGeometry, SuperBlockGeometry};
pub use header::{ExtensibleArray, ExtensibleArrayHeader, ExtensibleArrayHeaderLocator};
use header::{HEADER_BYTE_FIELDS, HEADER_PREFIX_LEN, HEADER_SIGNATURE};
use index_block::INDEX_BLOCK_SIGNATURE;
pub use index_block::{ExtensibleArrayIndexBlock, ExtensibleArrayIndexBlockRef};
pub use page::{ExtensibleArrayPageRef, ExtensibleArrayPages};
use secondary_block::SUPER_BLOCK_SIGNATURE;
pub use secondary_block::{ExtensibleArraySuperBlock, ExtensibleArraySuperBlockRef};
pub use statistics::ExtensibleArrayStats;

fn product(a: u64, b: u64) -> Result<u64, FormatError> {
    a.checked_mul(b)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)
}

fn sum<const N: usize>(terms: [u64; N]) -> Result<u64, FormatError> {
    terms
        .into_iter()
        .try_fold(0, u64::checked_add)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)
}

fn internal(detail: &str) -> FormatError {
    FormatError::Internal(format!("an Extensible Array {detail}"))
}

/// A retained plan for serializing an empty Extensible Array header and index block.
///
/// The serialized header is followed immediately by its index block. Both structures include
/// their checksums. The index block's chunk and child block addresses are undefined. The version 0
/// header and index block are defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_format::__private::{
///     ExtensibleArrayCreationParameters, ExtensibleArrayInitialization, FormatWidths,
/// };
///
/// let creation = ExtensibleArrayCreationParameters::parse(&[24, 8, 2, 8, 8])?;
/// let initialization = ExtensibleArrayInitialization::new(
///     creation,
///     16,
///     FormatWidths::from_sizes(8, 8)?,
///     false,
/// )?;
/// assert_eq!(initialization.plan().blocks().len(), 2);
/// let bytes = initialization.encode_at(StoredAddress::new(1024))?;
/// assert_eq!(bytes.len() as u64, initialization.allocation_len());
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub struct ExtensibleArrayInitialization {
    plan: ExtensibleArrayPlan,
}

impl ExtensibleArrayInitialization {
    /// Uses version 4 chunk-record encoding for an empty index.
    ///
    /// `chunk_bytes` is the chunk size before filtering. For filtered chunks, the chunk-size
    /// field width is derived from that size.
    ///
    /// # Errors
    ///
    /// Returns the geometry and width errors of [`ExtensibleArrayPlan::new`].
    pub fn new(
        creation: ExtensibleArrayCreationParameters,
        chunk_bytes: u64,
        widths: FormatWidths,
        has_filters: bool,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            plan: ExtensibleArrayPlan::new(
                creation,
                SlotOccupancy::Dense(0),
                0,
                chunk_bytes,
                widths,
                has_filters,
            )?,
        })
    }

    /// Returns the block geometry defined by the retained creation parameters.
    ///
    /// Includes the geometry of unallocated blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset
    /// exceeds [`u64::MAX`].
    pub fn geometry(&self) -> Result<ExtensibleArrayGeometry, FormatError> {
        self.plan.geometry()
    }

    /// Returns the canonical chunk-record encoding used to serialize the empty index.
    pub fn element_encoding(&self) -> ChunkElementEncoding {
        self.plan.element_encoding()
    }

    /// Returns the combined size of the header and index block, including their checksums.
    pub fn allocation_len(&self) -> u64 {
        self.plan.total_len()
    }

    /// Serializes the header at the base-relative `address`, followed by its index block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the allocation's end exceeds
    /// [`u64::MAX`]. After this check succeeds, returns [`FormatError::DefinedAddressIsUndefined`]
    /// if `address` is undefined, or [`FormatError::AddressTooLarge`] if its start or end exceeds
    /// the address width. Returns
    /// [`FormatError::ValueTooLargeForPlatform`] if the allocation length does not fit [`usize`],
    /// [`FormatError::AllocationFailed`] if a scratch or output reservation fails, or
    /// [`FormatError::Internal`] if the encoding differs from the planned length.
    pub fn encode_at(&self, address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        address
            .get()
            .checked_add(self.allocation_len())
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        self.plan
            .encode_contiguous(&IndexSlots::dense(&[]), address)
    }

    /// Returns the retained plan for the empty header and index block.
    pub fn plan(&self) -> &ExtensibleArrayPlan {
        &self.plan
    }
}

/// The slots of an Extensible Array that hold a chunk.
///
/// [`extensible_array_layout`] counts the blocks that hold a chunk and [`build_extensible_array_at`]
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

/// Header statistics and complete encoded length from Extensible Array measurement.
///
/// The statistics and total length are computed through the same block traversal as
/// [`ExtensibleArrayPlan`].
pub struct ExtensibleArrayLayout {
    /// The header statistics.
    pub stats: ExtensibleArrayStats,
    /// The length of the array in bytes.
    total_len: u64,
}

/// Returns the layout of the Extensible Array of `num_slots` slots whose occupied slots are
/// `occupancy`.
///
/// The creation parameters are [`ExtensibleArrayCreationParameters::canonical`]. Measurement
/// retains numeric totals in the returned value.
///
/// # Errors
///
/// Returns [`FormatError::ExtensibleArraySlotBounds`] if an occupied slot is outside the
/// declared span, [`FormatError::ExtensibleArraySlotCount`] if a sparse index's span differs
/// from `num_slots`, or [`FormatError::ExtensibleArraySlotOrder`] if explicit slots are not
/// strictly increasing. Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the span,
/// block offsets, or complete lengths exceed the creation geometry or checked `u64` arithmetic,
/// or [`FormatError::LengthTooLarge`] if a header statistic does not fit `length_size`.
pub fn extensible_array_layout(
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> Result<ExtensibleArrayLayout, FormatError> {
    write::measure(
        occupancy,
        num_slots,
        chunk_bytes,
        FormatWidths {
            offsets: offset_size,
            lengths: length_size,
        },
        has_filters,
    )
}

/// Returns the length in bytes of the Extensible Array [`build_extensible_array_at`] builds for
/// `slots`, without building it.
///
/// A caller that reserves the span of an array before writing it, such as an in-place writer
/// that places an array in freed space, takes the length from here.
///
/// # Errors
///
/// Returns the geometry and width errors of [`extensible_array_layout`].
pub fn extensible_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> Result<u64, FormatError> {
    Ok(extensible_array_layout(
        SlotOccupancy::Sparse(slots),
        slots.len().to_u64(),
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    )?
    .total_len)
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
/// Returns the planning errors of [`ExtensibleArrayPlan::new`] and record errors of
/// [`ExtensibleArrayPlacement::encoding`](write::ExtensibleArrayPlacement::encoding).
/// Returns [`FormatError::DefinedAddressIsUndefined`] if `ea_address` is undefined,
/// [`FormatError::AddressTooLarge`] if the allocation's start or end exceeds the address width,
/// [`FormatError::OffsetOverflow`] if its end exceeds [`u64::MAX`], or
/// [`FormatError::ValueTooLargeForPlatform`] if the output length does not fit [`usize`]. Returns
/// [`FormatError::AllocationFailed`] if a scratch or output reservation fails, or
/// [`FormatError::Internal`] if the encoding differs from the planned length.
pub fn build_extensible_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
    ea_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    ExtensibleArrayPlan::new(
        ExtensibleArrayCreationParameters::canonical(),
        SlotOccupancy::Sparse(slots),
        slots.len().to_u64(),
        chunk_bytes,
        FormatWidths {
            offsets: offset_size,
            lengths: length_size,
        },
        has_filters,
    )?
    .encode_contiguous(slots, ea_address)
}

/// Returns the capacity in slots of an Extensible Array [`build_extensible_array_at`] builds: the
/// elements of the index block, the data blocks it addresses, and the data blocks of every super
/// block.
///
/// The capacity is 8,589,934,580 slots: 4 in the index block, 240 in six data blocks, and
/// 8,589,934,336 in 25 super blocks.
pub fn extensible_array_capacity() -> u64 {
    // Each super block holds twice as many slots as the previous one. The sum starts at
    // `EA_MIN_DBLK_NELMTS` and includes a final block of `2^EA_MAX_NELMTS_BITS` slots.
    const {
        assert!((EA_MAX_NELMTS_BITS as u32) < u64::BITS - 1);
        (1u64 << (EA_MAX_NELMTS_BITS + 1)) - EA_MIN_DBLK_NELMTS as u64 + EA_IDX_BLK_ELMTS as u64
    }
}

/// The client ID of an array that indexes unfiltered chunks, whose elements each store a chunk
/// address alone.
///
/// The ID is defined in the Client ID table of "The Extensible Array Index" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub const EA_CLIENT_ID_UNFILTERED: u8 = 0;

const VERSION: u8 = 0;
// The version and client ID of a block.
const BLOCK_BYTE_FIELDS: usize = 2;
const CHECKSUM_LEN: usize = 4;

#[cfg(test)]
mod tests;
