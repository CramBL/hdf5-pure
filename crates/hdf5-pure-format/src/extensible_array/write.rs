//! Complete Extensible Array block geometry and checked stored placements.

use alloc::vec::Vec;
use core::ops::Range;

use hdf5_pure_core::AddressRange;

use super::DataBlock;
use super::ExtensibleArrayCreationParameters;
use super::ExtensibleArrayGeometry;
use super::ExtensibleArrayStats;
use super::Layout;
use super::SlotOccupancy;
use super::SuperBlock;
use crate::address::StoredAddress;
use crate::chunk_record;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ElementEncoding;
use crate::chunk_record::InvalidElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
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
        let occupancy = Occupancy::new(occupancy, num_slots)?;
        let encoding = chunk_record::chunk_element_encoding(chunk_bytes, widths.offsets, filtered);
        let element_size = encoding
            .elem_size
            .narrow_or_else(|| FormatError::InvalidExtensibleArrayGeometry)?;
        let elements = ElementEncoding::new(widths.offsets, encoding.client_id, element_size)
            .map_err(|error| match error {
                InvalidElementEncoding::Client(client) => {
                    FormatError::InvalidExtensibleArrayClient(client)
                }
                InvalidElementEncoding::EntrySize => FormatError::InvalidExtensibleArrayGeometry,
            })?;
        let layout = creation.layout(elements);
        let mut blocks = vec_blocks(layout, widths)?;
        let mut stats = ExtensibleArrayStats {
            nsuper_blks: 0,
            super_blk_size: 0,
            ndata_blks: 0,
            data_blk_size: 0,
            max_idx_set: num_slots,
            nelmts: u64::from(layout.inline_elements),
        };
        let mut covered = num_slots <= u64::from(layout.inline_elements);
        for (pointer, block) in layout.direct_blocks().enumerate() {
            let block = block?;
            let slots = layout.slots(block.offset, block.elements)?;
            covered |= num_slots <= slots.end;
            if occupancy
                .next(slots.start)
                .is_some_and(|slot| slot < slots.end)
            {
                push_data(
                    &mut blocks,
                    &mut stats,
                    layout,
                    block,
                    (INDEX_BLOCK, pointer.to_u64()),
                )?;
            }
        }
        for index in layout.first_super_block..layout.super_blocks {
            if covered {
                break;
            }
            let block = layout.super_block(index)?;
            let first = layout.first_slot(block.offset)?;
            covered =
                num_slots > first && (num_slots - 1 - first) / block.block_elements < block.blocks;
            let pending = selected_data(occupancy, layout, block)?;
            if pending.is_empty() {
                continue;
            }
            let secondary = blocks
                .len()
                .checked_add(pending.len())
                .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
            for (data, pointer) in pending {
                push_data(&mut blocks, &mut stats, layout, data, (secondary, pointer))?;
            }
            let length = layout.super_block_len(block)?;
            let slots = layout.slots(
                block.offset,
                super::product(block.blocks, block.block_elements)?,
            )?;
            let descriptor = ExtensibleArrayBlock {
                kind: ExtensibleArrayBlockKind::Secondary,
                length,
                slots: Some(slots),
                offset: Some(block.offset),
                parent: Some((INDEX_BLOCK, u64::from(index - layout.first_super_block))),
            };
            push(&mut blocks, descriptor)?;
            stats.nsuper_blks = super::sum([stats.nsuper_blks, 1])?;
            stats.super_blk_size = super::sum([stats.super_blk_size, length])?;
        }
        if !covered {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        let ExtensibleArrayStats {
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        } = stats;
        for length in [
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        ] {
            if !width::length_width_holds(widths.lengths, length) {
                return Err(FormatError::LengthTooLarge {
                    length,
                    length_size: widths.lengths.get(),
                });
            }
        }
        let total_length = super::sum([
            super::header_len(widths),
            layout.index_block_len()?,
            super_blk_size,
            data_blk_size,
        ])?;
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

    /// Checks disjoint complete ranges at the supplied stored addresses.
    ///
    /// Addresses follow [`blocks`](Self::blocks) order and may be in any address order. Starts
    /// must be defined, and exclusive ends must fit the address width. An end may equal the
    /// undefined address value. The caller establishes ownership and placement policy.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ExtensibleArrayBlockCount`] if the number of addresses differs
    /// from the number of blocks, [`FormatError::DefinedAddressIsUndefined`] if a start is
    /// undefined, [`FormatError::AddressTooLarge`] if a start or end exceeds the address width,
    /// [`FormatError::OffsetOverflow`] if an end exceeds [`u64::MAX`], or
    /// [`FormatError::ExtensibleArrayBlockOverlap`] if complete ranges overlap.
    pub fn place(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<ExtensibleArrayPlacement<'_>, FormatError> {
        if addresses.len() != self.blocks.len() {
            return Err(FormatError::ExtensibleArrayBlockCount {
                expected: self.blocks.len(),
                actual: addresses.len(),
            });
        }
        let mut ranges = Vec::with_capacity(addresses.len());
        for (&address, length) in addresses.iter().zip(self.block_lengths()) {
            let maximum = StoredAddress::undefined(self.widths.offsets).get();
            if address.get() == maximum {
                return Err(FormatError::DefinedAddressIsUndefined {
                    address: address.get(),
                    offset_size: self.widths.offsets.get(),
                });
            }
            if address.get() > maximum {
                return Err(FormatError::AddressTooLarge {
                    address: address.get(),
                    offset_size: self.widths.offsets.get(),
                });
            }
            let range = AddressRange::new(address, length)?;
            if range.end().get() > maximum {
                return Err(FormatError::AddressTooLarge {
                    address: range.end().get(),
                    offset_size: self.widths.offsets.get(),
                });
            }
            ranges.push(range);
        }
        let mut ordered: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(index, range)| (range.start(), range.end(), index))
            .collect();
        ordered.sort_unstable();
        for pair in ordered.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(FormatError::ExtensibleArrayBlockOverlap {
                    first: pair[0].2,
                    second: pair[1].2,
                });
            }
        }
        Ok(ExtensibleArrayPlacement { plan: self, ranges })
    }
}

/// Disjoint complete block ranges with stored addresses that fit an Extensible Array plan.
///
/// The plan is borrowed, and ranges follow its descriptor order. Allocation ownership and
/// placement policy are established by the caller.
#[derive(Debug)]
pub struct ExtensibleArrayPlacement<'a> {
    plan: &'a ExtensibleArrayPlan,
    ranges: Vec<AddressRange<StoredAddress>>,
}

impl ExtensibleArrayPlacement<'_> {
    /// Returns the borrowed geometry and block descriptors.
    pub fn plan(&self) -> &ExtensibleArrayPlan {
        self.plan
    }
    /// Returns stored starts and complete lengths in the plan's descriptor order.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.ranges
            .iter()
            .map(|range| (range.start(), range.length()))
    }
}

/// A complete allocation's length, logical slots, and parent pointer position.
///
/// A paged data block includes every page in its allocation. Its pages have no separate
/// descriptors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayBlock {
    kind: ExtensibleArrayBlockKind,
    length: u64,
    slots: Option<Range<u64>>,
    offset: Option<u64>,
    parent: Option<(usize, u64)>,
}

impl ExtensibleArrayBlock {
    /// Returns the block's role in the index.
    pub fn kind(&self) -> ExtensibleArrayBlockKind {
        self.kind
    }
    /// Returns the complete allocation length, including checksums and any data block pages.
    pub fn length(&self) -> u64 {
        self.length
    }
    /// Returns the logical slot range, including unoccupied slots, or `None` for the header.
    pub fn slots(&self) -> Option<Range<u64>> {
        self.slots.clone()
    }
    /// Returns the block offset relative to the end of the index block's inline slots.
    ///
    /// The header and index block have no block offset.
    pub fn offset(&self) -> Option<u64> {
        self.offset
    }
    /// Returns the parent descriptor index and the pointer slot within its address table.
    ///
    /// The pointer slot is relative to the direct data block table or secondary block table
    /// when the parent is the index block. The index block's parent is the header at slot zero.
    /// The header has no parent. A parent descriptor may follow its child.
    pub fn parent(&self) -> Option<(usize, u64)> {
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

/// Occupied slots within the declared span, with explicit slots in strictly increasing order.
#[derive(Clone, Copy)]
struct Occupancy<'a>(SlotOccupancy<'a>);

impl<'a> Occupancy<'a> {
    fn new(occupancy: SlotOccupancy<'a>, num_slots: u64) -> Result<Self, FormatError> {
        match occupancy {
            SlotOccupancy::Dense(len) if len > num_slots => {
                return Err(FormatError::ExtensibleArraySlotBounds {
                    slot: len - 1,
                    slots: num_slots,
                });
            }
            SlotOccupancy::Sparse(slots) if slots.len().to_u64() != num_slots => {
                return Err(FormatError::ExtensibleArraySlotCount {
                    expected: num_slots,
                    actual: slots.len().to_u64(),
                });
            }
            SlotOccupancy::Slots(slots) => {
                if let Some(pair) = slots.windows(2).find(|pair| pair[0] >= pair[1]) {
                    return Err(FormatError::ExtensibleArraySlotOrder {
                        previous: pair[0],
                        current: pair[1],
                    });
                }
                if let Some(&slot) = slots.last().filter(|&&slot| slot >= num_slots) {
                    return Err(FormatError::ExtensibleArraySlotBounds {
                        slot,
                        slots: num_slots,
                    });
                }
            }
            _ => {}
        }
        Ok(Self(occupancy))
    }

    fn next(self, start: u64) -> Option<u64> {
        match self.0 {
            SlotOccupancy::Dense(len) => (start < len).then_some(start),
            SlotOccupancy::Sparse(slots) => slots.next_occupied(start),
            SlotOccupancy::Slots(slots) => slots
                .get(slots.partition_point(|&slot| slot < start))
                .copied(),
        }
    }
}

fn vec_blocks(
    layout: Layout,
    widths: FormatWidths,
) -> Result<Vec<ExtensibleArrayBlock>, FormatError> {
    Ok(alloc::vec![
        ExtensibleArrayBlock {
            kind: ExtensibleArrayBlockKind::Header,
            length: super::header_len(widths),
            slots: None,
            offset: None,
            parent: None
        },
        ExtensibleArrayBlock {
            kind: ExtensibleArrayBlockKind::Index,
            length: layout.index_block_len()?,
            slots: Some(0..u64::from(layout.inline_elements)),
            offset: None,
            parent: Some((HEADER_BLOCK, 0))
        },
    ])
}

fn selected_data(
    occupancy: Occupancy<'_>,
    layout: Layout,
    block: SuperBlock,
) -> Result<Vec<(DataBlock, u64)>, FormatError> {
    let first = layout.first_slot(block.offset)?;
    let mut pending = Vec::new();
    let mut next = occupancy.next(first);
    while let Some(slot) = next {
        let local = (slot - first) / block.block_elements;
        if local >= block.blocks {
            break;
        }
        let data = block.data_block(local)?;
        let slots = layout.slots(data.offset, data.elements)?;
        pending
            .try_reserve(1)
            .map_err(|source| FormatError::AllocationFailed {
                values: pending.len().saturating_add(1),
                source,
            })?;
        pending.push((data, local));
        next = occupancy.next(slots.end);
    }
    Ok(pending)
}

fn push_data(
    blocks: &mut Vec<ExtensibleArrayBlock>,
    stats: &mut ExtensibleArrayStats,
    layout: Layout,
    block: DataBlock,
    parent: (usize, u64),
) -> Result<(), FormatError> {
    let offset_len = usize::from(layout.block_offset_len);
    if block.offset.to_le_bytes()[offset_len..]
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(FormatError::InvalidExtensibleArrayGeometry);
    }
    let length = layout.data_block_len(block)?;
    push(
        blocks,
        ExtensibleArrayBlock {
            kind: ExtensibleArrayBlockKind::Data,
            length,
            slots: Some(layout.slots(block.offset, block.elements)?),
            offset: Some(block.offset),
            parent: Some(parent),
        },
    )?;
    stats.ndata_blks = super::sum([stats.ndata_blks, 1])?;
    stats.data_blk_size = super::sum([stats.data_blk_size, length])?;
    stats.nelmts = super::sum([stats.nelmts, block.elements])?;
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

const HEADER_BLOCK: usize = 0;
const INDEX_BLOCK: usize = 1;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hdf5_pure_core::__private::BaseAddressExt;
    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::BaseAddress;
    use rstest::rstest;

    use super::*;
    use crate::chunk_record::ChunkRecord;
    use crate::chunk_record::IndexSlots;
    use crate::extensible_array;

    fn plan(creation: &[u8], slots: &[u64], count: u64, width: u8) -> ExtensibleArrayPlan {
        ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(creation).unwrap(),
            SlotOccupancy::Slots(slots),
            count,
            8,
            FormatWidths::from_sizes(width, 8).unwrap(),
            false,
        )
        .unwrap()
    }

    #[rstest]
    #[case::empty(&[], 0)]
    #[case::inline(&[0, 3], 4)]
    #[case::direct(&[4, 19, 20, 51, 52, 83], 84)]
    #[case::secondary(&[244, 500, 1_012], 1_013)]
    #[case::paged(&[16_372, 18_420, 32_756], 32_757)]
    #[case::far_sparse(&[4_294_967_284], 4_294_967_285)]
    fn canonical_planning_preserves_contiguous_geometry(#[case] slots: &[u64], #[case] count: u64) {
        let planned = plan(&[32, 4, 4, 16, 10], slots, count, 8);
        let old = extensible_array::extensible_array_layout(
            SlotOccupancy::Slots(slots),
            count,
            8,
            planned.widths().offsets,
            planned.widths().lengths,
            false,
        );
        assert_eq!(planned.total_len(), old.total_len);
        assert_eq!(planned.stats(), old.stats);
        let encoding = planned.element_encoding();
        assert_eq!(
            (
                encoding.elem_size,
                encoding.chunk_size_bytes,
                encoding.client_id
            ),
            (
                old.encoding.elem_size,
                old.encoding.chunk_size_bytes,
                old.encoding.client_id
            )
        );
        let geometry = planned.geometry().unwrap();
        assert_eq!(geometry.sblks, old.geom.sblks);
        assert_eq!(geometry.direct_dblk_nelmts, old.geom.direct_dblk_nelmts);
        assert_eq!(geometry.nsblk_addrs, old.geom.nsblk_addrs);
        assert_eq!(geometry.first_indirect_sblk, old.geom.first_indirect_sblk);
        assert_eq!(planned.block_lengths().sum::<u64>(), planned.total_len());
        assert_eq!(
            planned.creation(),
            ExtensibleArrayCreationParameters::parse(&[32, 4, 4, 16, 10]).unwrap()
        );
        for (index, block) in planned.blocks().enumerate() {
            match block.kind() {
                ExtensibleArrayBlockKind::Header => {
                    assert_eq!(index, 0);
                    assert_eq!(
                        (block.slots(), block.offset(), block.parent()),
                        (None, None, None)
                    );
                }
                ExtensibleArrayBlockKind::Index => {
                    assert_eq!(index, 1);
                    assert_eq!(
                        (block.slots(), block.offset(), block.parent()),
                        (Some(0..4), None, Some((0, 0)))
                    );
                }
                ExtensibleArrayBlockKind::Secondary | ExtensibleArrayBlockKind::Data => {
                    let (parent, _) = block.parent().unwrap();
                    let owner = planned.blocks().nth(parent).unwrap();
                    assert!(matches!(
                        owner.kind(),
                        ExtensibleArrayBlockKind::Index | ExtensibleArrayBlockKind::Secondary
                    ));
                    assert_eq!(block.slots().unwrap().start, 4 + block.offset().unwrap());
                    assert!(
                        owner
                            .slots()
                            .unwrap()
                            .contains(&block.slots().unwrap().start)
                            || parent == 1
                    );
                }
            }
        }
    }

    #[test]
    fn dense_and_explicit_occupancy_plan_the_same_blocks() {
        let slots: Vec<_> = (0..300).collect();
        let explicit = plan(&[32, 4, 4, 16, 10], &slots, 300, 8);
        let dense = ExtensibleArrayPlan::new(
            explicit.creation(),
            SlotOccupancy::Dense(300),
            300,
            8,
            explicit.widths(),
            false,
        )
        .unwrap();
        assert_eq!(dense.blocks, explicit.blocks);
        assert_eq!(dense.stats(), explicit.stats());
    }

    #[test]
    fn scattered_occupancy_preserves_trailing_empty_slots() {
        let chunks = [ChunkRecord {
            address: StoredAddress::new(100),
            stored_size: 8,
            filter_mask: 0,
        }; 3];
        let slots = IndexSlots::new(&chunks, &[500, 4, 0], 600).unwrap();
        let explicit = plan(&[32, 4, 4, 16, 10], &[0, 4, 500], 600, 8);
        let sparse = ExtensibleArrayPlan::new(
            explicit.creation(),
            SlotOccupancy::Sparse(&slots),
            600,
            8,
            explicit.widths(),
            false,
        )
        .unwrap();
        assert_eq!(sparse.blocks, explicit.blocks);
        assert_eq!(sparse.stats(), explicit.stats());
        assert_eq!(sparse.stats().max_idx_set, 600);
    }

    #[rstest]
    #[case::two_byte_addresses(2, 2, false, 8, 30, 72, 31)]
    #[case::four_byte_addresses(4, 4, false, 8, 44, 134, 49)]
    #[case::eight_byte_addresses(8, 8, false, 8, 72, 258, 85)]
    #[case::filtered_elements(8, 8, true, 256, 72, 314, 141)]
    fn nondefault_geometry_measures_complete_blocks(
        #[case] offsets: u8,
        #[case] lengths: u8,
        #[case] filtered: bool,
        #[case] chunk_bytes: u64,
        #[case] header: u64,
        #[case] index: u64,
        #[case] data: u64,
    ) {
        let planned = ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(&[24, 8, 2, 8, 8]).unwrap(),
            SlotOccupancy::Slots(&[8, 32]),
            33,
            chunk_bytes,
            FormatWidths::from_sizes(offsets, lengths).unwrap(),
            filtered,
        )
        .unwrap();
        assert_eq!(
            planned
                .blocks()
                .map(|block| (block.kind(), block.length(), block.slots(), block.parent()))
                .collect::<Vec<_>>(),
            vec![
                (ExtensibleArrayBlockKind::Header, header, None, None),
                (
                    ExtensibleArrayBlockKind::Index,
                    index,
                    Some(0..8),
                    Some((0, 0))
                ),
                (
                    ExtensibleArrayBlockKind::Data,
                    data,
                    Some(8..16),
                    Some((1, 0))
                ),
                (
                    ExtensibleArrayBlockKind::Data,
                    2 * data - (13 + u64::from(offsets)),
                    Some(32..48),
                    Some((4, 0))
                ),
                (
                    ExtensibleArrayBlockKind::Secondary,
                    13 + 3 * u64::from(offsets),
                    Some(32..64),
                    Some((1, 0))
                ),
            ]
        );
        assert_eq!(
            planned.stats(),
            ExtensibleArrayStats {
                nsuper_blks: 1,
                super_blk_size: 13 + 3 * u64::from(offsets),
                ndata_blks: 2,
                data_blk_size: 3 * data - (13 + u64::from(offsets)),
                max_idx_set: 33,
                nelmts: 32,
            }
        );
    }

    #[rstest]
    #[case::unpaged(16, 0, 128)]
    #[case::one_page(32, 0, 256)]
    #[case::two_pages(64, 2, 512)]
    #[case::eight_pages(256, 8, 2_048)]
    fn pages_remain_inside_the_complete_data_allocation(
        #[case] elements: u64,
        #[case] pages: u64,
        #[case] element_bytes: u64,
    ) {
        let index = 2 * (elements / 8).ilog2();
        let first = 2 + 8 * ((1_u64 << index) - 1);
        let planned = plan(&[16, 2, 2, 8, 5], &[first], first + 1, 8);
        let data = planned
            .blocks()
            .find(|block| block.kind() == ExtensibleArrayBlockKind::Data)
            .unwrap();
        assert_eq!(data.slots(), Some(first..first + elements));
        assert_eq!(data.length(), 20 + element_bytes + 4 * pages);
        assert_eq!(planned.blocks().len(), if index < 2 { 3 } else { 4 });
        assert_eq!(planned.stats().ndata_blks, 1);
        assert_eq!(planned.stats().nelmts, 2 + elements);
    }

    #[test]
    fn secondary_pointer_slots_preserve_holes_between_complete_blocks() {
        let planned = plan(&[32, 4, 4, 16, 10], &[244, 436], 437, 8);
        assert_eq!(
            planned
                .blocks()
                .skip(2)
                .map(|block| (block.kind(), block.slots(), block.offset(), block.parent()))
                .collect::<Vec<_>>(),
            vec![
                (
                    ExtensibleArrayBlockKind::Data,
                    Some(244..308),
                    Some(240),
                    Some((4, 0))
                ),
                (
                    ExtensibleArrayBlockKind::Data,
                    Some(436..500),
                    Some(432),
                    Some((4, 3))
                ),
                (
                    ExtensibleArrayBlockKind::Secondary,
                    Some(244..500),
                    Some(240),
                    Some((1, 0))
                ),
            ]
        );
        let addresses = [4_096, 8_192, 12_288, 16_384, 20_480].map(StoredAddress::new);
        let placed = planned.place(&addresses).unwrap();
        assert_eq!(
            placed.blocks().collect::<Vec<_>>(),
            addresses
                .into_iter()
                .zip([72, 298, 534, 534, 54])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn far_sparse_geometry_stays_in_u64_without_materializing_payloads() {
        let slot = 1_u64 << 56;
        let planned = plan(&[64, 4, 4, 16, 10], &[slot], slot + 1, 8);
        assert_eq!(planned.blocks().len(), 4);
        assert_eq!(planned.stats().ndata_blks, 1);
        assert_eq!(planned.stats().nsuper_blks, 1);
        assert!(planned.total_len() > u64::from(u32::MAX));
        assert!(
            planned
                .blocks()
                .any(|block| block.length() > u64::from(u32::MAX))
        );
        assert_eq!(planned.stats().max_idx_set, slot + 1);
    }

    #[rstest]
    #[case::ascending(8, &[1_000, 2_000, 3_000], 1_u64 << 48)]
    #[case::reverse(8, &[3_000, 2_000, 1_000], 0)]
    #[case::adjacent(2, &[0, 66, 220], 512)]
    #[case::exclusive_end(2, &[0, 1_000, 65_535 - 48], 0)]
    fn placement_preserves_independent_complete_ranges(
        #[case] width: u8,
        #[case] starts: &[u64],
        #[case] base: u64,
    ) {
        let planned = plan(&[32, 4, 4, 16, 10], &[4], 5, width);
        let addresses: Vec<_> = starts.iter().copied().map(StoredAddress::new).collect();
        let placed = planned.place(&addresses).unwrap();
        assert!(core::ptr::eq(placed.plan(), &planned));
        let actual: Vec<_> = placed
            .blocks()
            .map(|(address, length)| (address.get(), length))
            .collect();
        assert_eq!(
            actual,
            starts
                .iter()
                .copied()
                .zip(planned.block_lengths())
                .collect::<Vec<_>>()
        );
        let base = BaseAddress::new(base);
        for (stored, length) in placed.blocks() {
            let physical = base.resolve(stored).unwrap();
            assert_eq!(physical.get(), base.get() + stored.get());
            assert_eq!(base.relative(physical).unwrap(), stored);
            assert_eq!(
                base.resolve(StoredAddress::new(stored.get() + length))
                    .unwrap()
                    .get(),
                physical.get() + length
            );
        }
    }

    #[rstest]
    #[case::undefined(2, &[65_535, 100, 1_000], 65_535, true)]
    #[case::wide_start(2, &[65_536, 100, 1_000], 65_536, false)]
    #[case::wide_end(2, &[65_530, 100, 1_000], 65_596, false)]
    fn placement_rejects_unencodable_addresses(
        #[case] width: u8,
        #[case] starts: &[u64],
        #[case] expected: u64,
        #[case] undefined: bool,
    ) {
        let planned = plan(&[32, 4, 4, 16, 10], &[4], 5, width);
        let addresses: Vec<_> = starts.iter().copied().map(StoredAddress::new).collect();
        let error = planned.place(&addresses).unwrap_err();
        if undefined {
            assert!(
                matches!(error, FormatError::DefinedAddressIsUndefined { address, offset_size } if address == expected && offset_size == width),
                "{error:?}"
            );
        } else {
            assert!(
                matches!(error, FormatError::AddressTooLarge { address, offset_size } if address == expected && offset_size == width),
                "{error:?}"
            );
        }
    }

    #[test]
    fn placement_rejects_count_overlap_and_overflow() {
        let planned = plan(&[32, 4, 4, 16, 10], &[4], 5, 8);
        let error = planned.place(&[]).unwrap_err();
        assert!(
            matches!(
                error,
                FormatError::ExtensibleArrayBlockCount {
                    expected: 3,
                    actual: 0
                }
            ),
            "{error:?}"
        );
        let error = planned
            .place(&[
                StoredAddress::new(1_000),
                StoredAddress::new(0),
                StoredAddress::new(1_001),
            ])
            .unwrap_err();
        assert!(
            matches!(
                error,
                FormatError::ExtensibleArrayBlockOverlap {
                    first: 0,
                    second: 2
                }
            ),
            "{error:?}"
        );
        let error = planned
            .place(&[
                StoredAddress::new(u64::MAX - 1),
                StoredAddress::new(0),
                StoredAddress::new(1_000),
            ])
            .unwrap_err();
        assert!(
            matches!(error, FormatError::OffsetOverflow { offset, length: 72 } if offset == u64::MAX - 1),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::descending(&[2, 1], 3, 2, 1)]
    #[case::duplicate(&[1, 1], 2, 1, 1)]
    fn planning_rejects_unordered_slots(
        #[case] slots: &[u64],
        #[case] count: u64,
        #[case] previous: u64,
        #[case] current: u64,
    ) {
        let error = ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(&[32, 4, 4, 16, 10]).unwrap(),
            SlotOccupancy::Slots(slots),
            count,
            8,
            FormatWidths::from_sizes(8, 8).unwrap(),
            false,
        )
        .unwrap_err();
        assert!(
            matches!(error, FormatError::ExtensibleArraySlotOrder { previous: a, current: b } if a == previous && b == current),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::dense(SlotOccupancy::Dense(5))]
    #[case::explicit(SlotOccupancy::Slots(&[4]))]
    fn planning_rejects_slots_past_the_declared_span(#[case] occupancy: SlotOccupancy<'_>) {
        let error = ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(&[32, 4, 4, 16, 10]).unwrap(),
            occupancy,
            4,
            8,
            FormatWidths::from_sizes(8, 8).unwrap(),
            false,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                FormatError::ExtensibleArraySlotBounds { slot: 4, slots: 4 }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn planning_rejects_a_sparse_span_mismatch() {
        let slots = IndexSlots::dense(&[]);
        let error = ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(&[32, 4, 4, 16, 10]).unwrap(),
            SlotOccupancy::Sparse(&slots),
            1,
            8,
            FormatWidths::from_sizes(8, 8).unwrap(),
            false,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                FormatError::ExtensibleArraySlotCount {
                    expected: 1,
                    actual: 0
                }
            ),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::outside_creation(&[8, 4, 4, 16, 8], 10_000)]
    #[case::offset_field(&[32, 4, 4, 16, 10], 4_296_000_000)]
    #[case::unrepresentable_parent(&[64, 4, 4, 16, 10], u64::MAX - 1)]
    fn planning_rejects_unrepresentable_complete_geometry(
        #[case] creation: &[u8],
        #[case] slot: u64,
    ) {
        let error = ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(creation).unwrap(),
            SlotOccupancy::Slots(&[slot]),
            slot + 1,
            8,
            FormatWidths::from_sizes(8, 8).unwrap(),
            false,
        )
        .unwrap_err();
        assert!(
            matches!(error, FormatError::InvalidExtensibleArrayGeometry),
            "{error:?}"
        );
    }

    #[test]
    fn planning_checks_length_fields_before_placement() {
        let error = ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(&[32, 4, 4, 16, 10]).unwrap(),
            SlotOccupancy::Dense(10_000),
            10_000,
            8,
            FormatWidths::from_sizes(8, 2).unwrap(),
            false,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                FormatError::LengthTooLarge {
                    length: 82_892,
                    length_size: 2
                }
            ),
            "{error:?}"
        );
    }
}
