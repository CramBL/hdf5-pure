use alloc::vec::Vec;
use core::ops::Range;

use super::occupancy::{self, Region, SlotOccupancy};
use crate::chunk_record::ChunkElementEncoding;
use crate::error::FormatError;
use crate::extensible_array::geometry::{DataBlock, Layout, SuperBlock};
use crate::extensible_array::{
    ExtensibleArrayCreationParameters, ExtensibleArrayGeometry, ExtensibleArrayStats, header,
};
use crate::width::FormatWidths;

/// The complete blocks selected by an Extensible Array's creation geometry and occupied slots.
///
/// Each paged data block has one descriptor, including its prefix, pages, and checksums. Block
/// lengths and logical slot ranges use `u64`. Planning retains descriptors without materializing
/// the blocks' encoded bytes.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
/// use hdf5_pure_format::__private::ExtensibleArrayPlan;
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::SlotOccupancy;
///
/// let plan = ExtensibleArrayPlan::new(
///     ExtensibleArrayCreationParameters::canonical(),
///     SlotOccupancy::Slots(&[4]),
///     5,
///     8,
///     FormatWidths::from_sizes(8, 8)?,
///     false,
/// )?;
/// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [72, 298, 150]);
/// assert_eq!(plan.blocks().nth(2).unwrap().parent(), Some((1, 0)));
///
/// let addresses = [8_192, 1_024, 4_096].map(StoredAddress::new);
/// let placement = plan.place(&addresses)?;
/// assert_eq!(
///     placement.blocks().collect::<Vec<_>>(),
///     addresses.into_iter().zip([72, 298, 150]).collect::<Vec<_>>()
/// );
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Debug)]
pub struct ExtensibleArrayPlan {
    creation: ExtensibleArrayCreationParameters,
    widths: FormatWidths,
    encoding: ChunkElementEncoding,
    layout: Layout,
    blocks: Vec<ExtensibleArrayBlock>,
    stats: ExtensibleArrayStats,
    total_length: u64,
}

impl ExtensibleArrayPlan {
    /// Plans complete blocks for the occupied slots within `0..num_slots`.
    ///
    /// The header and index block are always included. A data block is included when one of its
    /// slots is occupied, together with its secondary block when applicable. `chunk_bytes` is
    /// the unfiltered chunk size used to derive the element encoding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ExtensibleArraySlotBounds`] if an occupied slot is outside the
    /// declared span, [`FormatError::ExtensibleArraySlotCount`] if a sparse index's span differs
    /// from `num_slots`, or [`FormatError::ExtensibleArraySlotOrder`] if explicit slots are not
    /// strictly increasing. Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the span,
    /// block offsets, or complete lengths exceed the creation geometry or checked `u64`
    /// arithmetic. Returns [`FormatError::LengthTooLarge`] if a header statistic does not fit
    /// `widths.lengths`, or [`FormatError::AllocationFailed`] if a descriptor reservation fails.
    pub fn new(
        creation: ExtensibleArrayCreationParameters,
        occupancy: SlotOccupancy<'_>,
        num_slots: u64,
        chunk_bytes: u64,
        widths: FormatWidths,
        filtered: bool,
    ) -> Result<Self, FormatError> {
        let (encoding, layout) =
            occupancy::element_layout(creation, chunk_bytes, widths, filtered)?;
        let mut blocks = vec_blocks(layout, widths)?;
        let mut stats = ExtensibleArrayStats::empty(layout, num_slots);
        occupancy::visit_regions(layout, occupancy, num_slots, |region| {
            match region {
                Region::Direct { block, pointer } => push_data(
                    &mut blocks,
                    &mut stats,
                    layout,
                    block,
                    (INDEX_BLOCK, pointer),
                )?,
                Region::Secondary {
                    block,
                    pointer,
                    data,
                } => {
                    let mut pending = Vec::new();
                    for data in data {
                        pending
                            .try_reserve(1)
                            .map_err(|source| FormatError::AllocationFailed {
                                values: pending.len().saturating_add(1),
                                source,
                            })?;
                        pending.push(data?);
                    }
                    if pending.is_empty() {
                        return Ok(());
                    }
                    let secondary = blocks
                        .len()
                        .checked_add(pending.len())
                        .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
                    for (data, pointer) in pending {
                        push_data(&mut blocks, &mut stats, layout, data, (secondary, pointer))?;
                    }
                    let (length, slots) = stats.secondary(layout, block)?;
                    push(
                        &mut blocks,
                        ExtensibleArrayBlock {
                            length,
                            location: BlockLocation::Secondary(SecondaryBlock {
                                geometry: block,
                                first_slot: slots.start,
                                pointer,
                            }),
                        },
                    )?;
                }
            }
            Ok(())
        })?;
        let stats = stats.with_widths(widths)?;
        let total_length = stats.total_len(layout, widths)?;
        Ok(Self {
            creation,
            widths,
            encoding,
            layout,
            blocks,
            stats,
            total_length,
        })
    }

    /// Returns the creation parameters used to plan the blocks.
    pub fn creation(&self) -> ExtensibleArrayCreationParameters {
        self.creation
    }
    /// Returns the encoded address and length widths.
    pub fn widths(&self) -> FormatWidths {
        self.widths
    }
    /// Returns the chunk element field widths and client ID.
    pub fn element_encoding(&self) -> ChunkElementEncoding {
        self.encoding
    }
    /// Derives the complete direct and secondary block tables from the creation geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset exceeds
    /// [`u64::MAX`].
    pub fn geometry(&self) -> Result<ExtensibleArrayGeometry, FormatError> {
        ExtensibleArrayGeometry::from_layout(self.layout)
    }
    /// Returns the header statistics for the selected blocks and declared slot span.
    pub fn stats(&self) -> ExtensibleArrayStats {
        self.stats
    }
    /// Returns the sum of complete block lengths, excluding gaps between placements.
    pub fn total_len(&self) -> u64 {
        self.total_length
    }
    /// Returns descriptors in the order used by [`place`](Self::place).
    ///
    /// The header and index block come first, followed by direct data blocks. Each remaining
    /// group contains the selected data blocks followed by their secondary block. Data blocks
    /// follow logical slot order within each group.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = &ExtensibleArrayBlock> {
        self.blocks.iter()
    }
    /// Returns complete allocation lengths in [`blocks`](Self::blocks) order.
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> {
        self.blocks.iter().map(ExtensibleArrayBlock::length)
    }

    pub(super) fn descriptors(&self) -> &[ExtensibleArrayBlock] {
        &self.blocks
    }

    pub(super) fn layout(&self) -> Layout {
        self.layout
    }

    pub(super) fn statistics(&self) -> &ExtensibleArrayStats {
        &self.stats
    }
}

/// A complete allocation's length, logical slots, and parent pointer position.
///
/// A paged data block includes every page in its allocation. Its pages have no separate
/// descriptors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayBlock {
    length: u64,
    location: BlockLocation,
}

impl ExtensibleArrayBlock {
    /// Returns the block's role in the index.
    pub fn kind(&self) -> ExtensibleArrayBlockKind {
        match self.location {
            BlockLocation::Header => ExtensibleArrayBlockKind::Header,
            BlockLocation::Index { .. } => ExtensibleArrayBlockKind::Index,
            BlockLocation::Secondary(_) => ExtensibleArrayBlockKind::Secondary,
            BlockLocation::Data(_) => ExtensibleArrayBlockKind::Data,
        }
    }
    /// Returns the complete allocation length, including checksums and any data block pages.
    pub fn length(&self) -> u64 {
        self.length
    }
    /// Returns the logical slot range, including unoccupied slots, or `None` for the header.
    pub fn slots(&self) -> Option<Range<u64>> {
        match &self.location {
            BlockLocation::Header => None,
            BlockLocation::Index { inline_elements } => Some(0..u64::from(*inline_elements)),
            BlockLocation::Secondary(block) => Some(block.slots()),
            BlockLocation::Data(block) => Some(block.slots.clone()),
        }
    }
    /// Returns the block offset relative to the end of the index block's inline slots.
    ///
    /// The header and index block have no block offset.
    pub fn offset(&self) -> Option<u64> {
        match &self.location {
            BlockLocation::Header | BlockLocation::Index { .. } => None,
            BlockLocation::Secondary(block) => Some(block.geometry.offset()),
            BlockLocation::Data(block) => Some(block.offset),
        }
    }
    /// Returns the parent descriptor index and the pointer slot within its address table.
    ///
    /// The pointer slot is relative to the direct data block table or secondary block table
    /// when the parent is the index block. The index block's parent is the header at slot zero.
    /// The header has no parent. A parent descriptor may follow its child.
    pub fn parent(&self) -> Option<(usize, u64)> {
        match &self.location {
            BlockLocation::Header => None,
            BlockLocation::Index { .. } => Some((HEADER_BLOCK, 0)),
            BlockLocation::Secondary(block) => Some((INDEX_BLOCK, block.pointer)),
            BlockLocation::Data(block) => Some(block.parent),
        }
    }

    pub(super) fn location(&self) -> &BlockLocation {
        &self.location
    }
}

/// The block role and its location within the array's index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum BlockLocation {
    Header,
    Index { inline_elements: u8 },
    Secondary(SecondaryBlock),
    Data(ChildBlock),
}

/// A secondary block's checked geometry, first slot, and pointer in the secondary table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SecondaryBlock {
    geometry: SuperBlock,
    first_slot: u64,
    pointer: u64,
}

impl SecondaryBlock {
    pub(super) fn geometry(&self) -> SuperBlock {
        self.geometry
    }

    pub(super) fn first_slot(&self) -> u64 {
        self.first_slot
    }

    pub(super) fn pointer(&self) -> u64 {
        self.pointer
    }

    fn slots(&self) -> Range<u64> {
        // Planning checks the range of every secondary block before building the descriptor.
        self.first_slot..self.first_slot + self.geometry.blocks() * self.geometry.block_elements()
    }
}

/// The logical slot range, stored offset, and parent pointer for a data block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ChildBlock {
    slots: Range<u64>,
    offset: u64,
    parent: (usize, u64),
}

impl ChildBlock {
    pub(super) fn slots(&self) -> &Range<u64> {
        &self.slots
    }

    pub(super) fn offset(&self) -> u64 {
        self.offset
    }

    pub(super) fn parent(&self) -> (usize, u64) {
        self.parent
    }
}

/// The role of a complete Extensible Array allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExtensibleArrayBlockKind {
    /// The `EAHD` header with creation parameters, statistics, and index block address.
    Header,
    /// The `EAIB` block with inline elements and direct and secondary block addresses.
    Index,
    /// An `EASB` block with data block addresses and any page initialization bitmap.
    Secondary,
    /// An `EADB` allocation, including its pages when paged.
    Data,
}

fn vec_blocks(
    layout: Layout,
    widths: FormatWidths,
) -> Result<Vec<ExtensibleArrayBlock>, FormatError> {
    Ok(alloc::vec![
        ExtensibleArrayBlock {
            length: header::header_len(widths),
            location: BlockLocation::Header,
        },
        ExtensibleArrayBlock {
            length: layout.index_block_len()?,
            location: BlockLocation::Index {
                inline_elements: layout.inline_elements()
            },
        },
    ])
}

fn push_data(
    blocks: &mut Vec<ExtensibleArrayBlock>,
    stats: &mut ExtensibleArrayStats,
    layout: Layout,
    block: DataBlock,
    parent: (usize, u64),
) -> Result<(), FormatError> {
    let (length, slots) = stats.data(layout, block)?;
    push(
        blocks,
        ExtensibleArrayBlock {
            length,
            location: BlockLocation::Data(ChildBlock {
                slots,
                offset: block.offset(),
                parent,
            }),
        },
    )?;
    Ok(())
}

fn push(
    blocks: &mut Vec<ExtensibleArrayBlock>,
    block: ExtensibleArrayBlock,
) -> Result<(), FormatError> {
    blocks
        .try_reserve(1)
        .map_err(|source| FormatError::AllocationFailed {
            values: blocks.len().saturating_add(1),
            source,
        })?;
    blocks.push(block);
    Ok(())
}

pub(super) const HEADER_BLOCK: usize = 0;
pub(super) const INDEX_BLOCK: usize = 1;

#[cfg(test)]
#[path = "tests/plan.rs"]
mod tests;
