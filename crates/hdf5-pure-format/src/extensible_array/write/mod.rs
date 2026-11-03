//! Complete Extensible Array block geometry and checked stored placements.

use alloc::vec::Vec;
use core::ops::Range;

use hdf5_pure_core::AddressRange;

use super::DataBlock;
use super::ExtensibleArrayCreationParameters;
use super::ExtensibleArrayGeometry;
use super::ExtensibleArrayLayout;
use super::ExtensibleArrayStats;
use super::Layout;
use super::SlotOccupancy;
use super::SuperBlock;
use crate::address::StoredAddress;
use crate::chunk_record;
use crate::chunk_record::ChunkElementEncoding;
use crate::chunk_record::ElementEncoding;
use crate::chunk_record::IndexSlots;
use crate::chunk_record::InvalidElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
use crate::width::FormatWidths;

mod encode;

pub use encode::ExtensibleArrayEmission;
pub use encode::ExtensibleArrayEncoding;

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
        let (encoding, layout) = element_layout(creation, chunk_bytes, widths, filtered)?;
        let mut blocks = vec_blocks(layout, widths)?;
        let mut stats = ExtensibleArrayStats::empty(layout, num_slots);
        visit_regions(layout, occupancy, num_slots, |region| {
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
                            kind: ExtensibleArrayBlockKind::Secondary,
                            length,
                            slots: Some(slots),
                            offset: Some(block.offset),
                            parent: Some((INDEX_BLOCK, pointer)),
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
            ranges.push(self.range(address, length)?);
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
    /// Serializes adjacent complete blocks at the stored `address`.
    ///
    /// A checked whole range covers the blocks in [`blocks`](Self::blocks) order. Each block is
    /// appended directly to one reserved output buffer. The caller establishes allocation
    /// ownership.
    ///
    /// # Errors
    ///
    /// Returns the address errors of [`Self::place`] and record errors of
    /// [`ExtensibleArrayPlacement::encoding`]. Returns [`FormatError::ValueTooLargeForPlatform`]
    /// if the total length does not fit [`usize`], [`FormatError::AllocationFailed`] if a range,
    /// reference, or output reservation fails, or [`FormatError::Internal`] if the ranges or
    /// encoded length differ from the planned total.
    pub(super) fn encode_contiguous(
        &self,
        slots: &IndexSlots<'_>,
        address: StoredAddress,
    ) -> Result<Vec<u8>, FormatError> {
        let whole = self.range(address, self.total_length)?;
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(self.blocks.len())
            .map_err(|source| FormatError::AllocationFailed {
                values: self.blocks.len(),
                source,
            })?;
        let mut cursor = whole.start();
        for length in self.block_lengths() {
            let range = AddressRange::new(cursor, length)?;
            cursor = range.end();
            ranges.push(range);
        }
        let placement = ExtensibleArrayPlacement { plan: self, ranges };
        let encoding = placement.encoding(slots)?;
        let length = self.total_length.to_usize()?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|source| FormatError::AllocationFailed {
                values: length,
                source,
            })?;
        for block in encoding.blocks() {
            block.append_into(&mut output)?;
        }
        if cursor != whole.end() || output.len() != length {
            return Err(FormatError::Internal(alloc::format!(
                "contiguous EA encoding does not fill its planned {length} bytes"
            )));
        }
        Ok(output)
    }

    fn range(
        &self,
        address: StoredAddress,
        length: u64,
    ) -> Result<AddressRange<StoredAddress>, FormatError> {
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
        Ok(range)
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

/// Computes canonical header statistics and wire lengths through the shared region traversal.
///
/// # Errors
///
/// Propagates occupancy and geometry errors from [`visit_regions`] and width errors from the
/// header statistics.
pub(super) fn measure(
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    chunk_bytes: u64,
    widths: FormatWidths,
    filtered: bool,
) -> Result<ExtensibleArrayLayout, FormatError> {
    let creation = ExtensibleArrayCreationParameters::canonical();
    let (_, layout) = element_layout(creation, chunk_bytes, widths, filtered)?;
    let mut stats = ExtensibleArrayStats::empty(layout, num_slots);
    visit_regions(layout, occupancy, num_slots, |region| {
        match region {
            Region::Direct { block, pointer: _ } => {
                stats.data(layout, block)?;
            }
            Region::Secondary {
                block,
                pointer: _,
                data,
            } => {
                let mut allocated = false;
                for data in data {
                    stats.data(layout, data?.0)?;
                    allocated = true;
                }
                if allocated {
                    stats.secondary(layout, block)?;
                }
            }
        }
        Ok(())
    })?;
    let stats = stats.with_widths(widths)?;
    Ok(ExtensibleArrayLayout {
        stats,
        total_len: stats.total_len(layout, widths)?,
    })
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

fn element_layout(
    creation: ExtensibleArrayCreationParameters,
    chunk_bytes: u64,
    widths: FormatWidths,
    filtered: bool,
) -> Result<(ChunkElementEncoding, Layout), FormatError> {
    let encoding = chunk_record::chunk_element_encoding(chunk_bytes, widths.offsets, filtered);
    let ChunkElementEncoding {
        chunk_size_bytes: _,
        elem_size,
        client_id,
    } = encoding;
    let element_size = elem_size.narrow_or_else(|| FormatError::InvalidExtensibleArrayGeometry)?;
    let elements = ElementEncoding::new(widths.offsets, client_id, element_size).map_err(
        |error| match error {
            InvalidElementEncoding::Client(client) => {
                FormatError::InvalidExtensibleArrayClient(client)
            }
            InvalidElementEncoding::EntrySize => FormatError::InvalidExtensibleArrayGeometry,
        },
    )?;
    Ok((encoding, creation.layout(elements)))
}

/// Visits occupied direct blocks and the secondary regions needed to cover `num_slots`.
///
/// Each secondary region includes a lazy cursor over its occupied data blocks. The cursor may
/// be empty.
///
/// # Errors
///
/// Propagates errors from occupancy checks, block geometry, or `visit`. Returns
/// [`FormatError::InvalidExtensibleArrayGeometry`] if the declared span exceeds the hierarchy.
fn visit_regions(
    layout: Layout,
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    mut visit: impl FnMut(Region<'_>) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let occupancy = Occupancy::new(occupancy, num_slots)?;
    let mut covered = num_slots <= u64::from(layout.inline_elements);
    for (pointer, block) in layout.direct_blocks().enumerate() {
        let block = block?;
        let DataBlock {
            offset,
            elements,
            libhdf5_offset: _,
        } = block;
        let slots = layout.slots(offset, elements)?;
        covered |= num_slots <= slots.end;
        if occupancy
            .next(slots.start)
            .is_some_and(|slot| slot < slots.end)
        {
            visit(Region::Direct {
                block,
                pointer: pointer.to_u64(),
            })?;
        }
    }
    for index in layout.first_super_block..layout.super_blocks {
        if covered {
            break;
        }
        let block = layout.super_block(index)?;
        let SuperBlock {
            offset,
            blocks,
            block_elements,
        } = block;
        let first = layout.first_slot(offset)?;
        covered = num_slots > first && (num_slots - 1 - first) / block_elements < blocks;
        visit(Region::Secondary {
            block,
            pointer: u64::from(index - layout.first_super_block),
            data: SelectedData::new(occupancy, layout, block)?,
        })?;
    }
    if !covered {
        return Err(FormatError::InvalidExtensibleArrayGeometry);
    }
    Ok(())
}

/// An occupied direct block or a secondary region with its selected data cursor.
enum Region<'a> {
    Direct {
        block: DataBlock,
        pointer: u64,
    },
    Secondary {
        block: SuperBlock,
        pointer: u64,
        data: SelectedData<'a>,
    },
}

impl ExtensibleArrayStats {
    fn empty(layout: Layout, num_slots: u64) -> Self {
        Self {
            nsuper_blks: 0,
            super_blk_size: 0,
            ndata_blks: 0,
            data_blk_size: 0,
            max_idx_set: num_slots,
            nelmts: u64::from(layout.inline_elements),
        }
    }

    /// Adds a complete data block's counts and returns its checked length and logical slots.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the block offset exceeds its
    /// field width or the length, slots, or statistics overflow [`u64`].
    fn data(&mut self, layout: Layout, block: DataBlock) -> Result<(u64, Range<u64>), FormatError> {
        let DataBlock {
            offset,
            elements,
            libhdf5_offset: _,
        } = block;
        let offset_len = usize::from(layout.block_offset_len);
        if offset.to_le_bytes()[offset_len..]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        let length = layout.data_block_len(block)?;
        let slots = layout.slots(offset, elements)?;
        self.ndata_blks = super::sum([self.ndata_blks, 1])?;
        self.data_blk_size = super::sum([self.data_blk_size, length])?;
        self.nelmts = super::sum([self.nelmts, elements])?;
        Ok((length, slots))
    }

    /// Adds a secondary block's counts and returns its complete length and logical slots.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if its length, logical range, or
    /// accumulated statistics overflow [`u64`].
    fn secondary(
        &mut self,
        layout: Layout,
        block: SuperBlock,
    ) -> Result<(u64, Range<u64>), FormatError> {
        let SuperBlock {
            offset,
            blocks,
            block_elements,
        } = block;
        let length = layout.super_block_len(block)?;
        let slots = layout.slots(offset, super::product(blocks, block_elements)?)?;
        self.nsuper_blks = super::sum([self.nsuper_blks, 1])?;
        self.super_blk_size = super::sum([self.super_blk_size, length])?;
        Ok((length, slots))
    }

    /// Checks that every header statistic fits the encoded length width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if a statistic exceeds `widths.lengths`.
    fn with_widths(self, widths: FormatWidths) -> Result<Self, FormatError> {
        let Self {
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        } = self;
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
        Ok(self)
    }

    fn total_len(self, layout: Layout, widths: FormatWidths) -> Result<u64, FormatError> {
        super::sum([
            super::header_len(widths),
            layout.index_block_len()?,
            self.super_blk_size,
            self.data_blk_size,
        ])
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

/// A cursor over occupied data blocks and their pointer slots in one secondary region.
struct SelectedData<'a> {
    occupancy: Occupancy<'a>,
    layout: Layout,
    block: SuperBlock,
    first: u64,
    next: Option<u64>,
}

impl<'a> SelectedData<'a> {
    fn new(
        occupancy: Occupancy<'a>,
        layout: Layout,
        block: SuperBlock,
    ) -> Result<Self, FormatError> {
        let first = layout.first_slot(block.offset)?;
        Ok(Self {
            occupancy,
            layout,
            block,
            first,
            next: occupancy.next(first),
        })
    }
}

impl Iterator for SelectedData<'_> {
    type Item = Result<(DataBlock, u64), FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            occupancy,
            layout,
            block,
            first,
            next,
        } = self;
        let slot = next.take()?;
        let SuperBlock {
            offset: _,
            blocks,
            block_elements,
        } = *block;
        let local = (slot - *first) / block_elements;
        if local >= blocks {
            return None;
        }
        Some((|| {
            let data = block.data_block(local)?;
            let DataBlock {
                offset,
                elements,
                libhdf5_offset: _,
            } = data;
            *next = occupancy.next(layout.slots(offset, elements)?.end);
            Ok((data, local))
        })())
    }
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
            kind: ExtensibleArrayBlockKind::Data,
            length,
            slots: Some(slots),
            offset: Some(block.offset),
            parent: Some(parent),
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
    #[case::empty(0, false, 370, 0, 0, 4)]
    #[case::filtered_empty(0, true, 394, 0, 0, 4)]
    #[case::first_data_block(5, false, 520, 1, 0, 20)]
    #[case::filtered_data_block(5, true, 640, 1, 0, 20)]
    #[case::second_data_block(21, false, 798, 2, 0, 52)]
    #[case::filtered_second_data_block(21, true, 1_110, 2, 0, 52)]
    #[case::first_secondary(300, false, 3_010, 7, 1, 308)]
    #[case::filtered_secondary(300, true, 4_858, 7, 1, 308)]
    fn one_shot_measurement_preserves_wire_lengths_and_statistics(
        #[case] count: u64,
        #[case] filtered: bool,
        #[case] total: u64,
        #[case] data: u64,
        #[case] secondary: u64,
        #[case] elements: u64,
    ) {
        let ExtensibleArrayLayout { stats, total_len } = measure(
            SlotOccupancy::Dense(count),
            count,
            8,
            FormatWidths::from_sizes(8, 8).unwrap(),
            filtered,
        )
        .unwrap();
        let ExtensibleArrayStats {
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        } = stats;
        assert_eq!(total_len, total);
        assert_eq!(ndata_blks, data);
        assert_eq!(nsuper_blks, secondary);
        assert_eq!(max_idx_set, count);
        assert_eq!(nelmts, elements);
        assert_eq!(super_blk_size, 54 * secondary);
        assert_eq!(
            data_blk_size + super_blk_size + 72 + if filtered { 322 } else { 298 },
            total
        );
    }

    #[rstest]
    #[case::order(&[2, 1], 3, 8, FormatError::ExtensibleArraySlotOrder { previous: 2, current: 1 })]
    #[case::bounds(&[4], 4, 8, FormatError::ExtensibleArraySlotBounds { slot: 4, slots: 4 })]
    #[case::length(&[], 65_536, 2, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
    #[case::geometry(&[], u64::MAX, 8, FormatError::InvalidExtensibleArrayGeometry)]
    #[case::block_offset(&[4_296_000_000], 4_296_000_001, 8, FormatError::InvalidExtensibleArrayGeometry)]
    fn one_shot_measurement_returns_checked_geometry_failures(
        #[case] slots: &[u64],
        #[case] count: u64,
        #[case] lengths: u8,
        #[case] expected: FormatError,
    ) {
        let error = measure(
            SlotOccupancy::Slots(slots),
            count,
            8,
            FormatWidths::from_sizes(8, lengths).unwrap(),
            false,
        )
        .err()
        .unwrap();
        assert_eq!(error, expected);
    }

    #[rstest]
    #[case::first_child(&[0])]
    #[case::last_child(&[1_023])]
    #[case::separated_children(&[0, 1_023])]
    fn sparse_region_iteration_visits_only_selected_data_blocks(#[case] local: &[u64]) {
        let (_, layout) = element_layout(
            ExtensibleArrayCreationParameters::canonical(),
            8,
            FormatWidths::from_sizes(8, 8).unwrap(),
            false,
        )
        .unwrap();
        let block = layout.super_block(20).unwrap();
        assert_eq!(block.blocks, 1_024);
        let first = layout.first_slot(block.offset).unwrap();
        let slots: Vec<_> = local
            .iter()
            .map(|&local| first + local * block.block_elements)
            .collect();
        let occupancy =
            Occupancy::new(SlotOccupancy::Slots(&slots), slots.last().unwrap() + 1).unwrap();
        let selected = SelectedData::new(occupancy, layout, block)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|&(data, local)| (data.offset, local))
                .collect::<Vec<_>>(),
            local
                .iter()
                .map(|&local| (block.offset + local * block.block_elements, local))
                .collect::<Vec<_>>()
        );
        let mut regions = 0;
        let mut data = 0;
        visit_regions(
            layout,
            SlotOccupancy::Slots(&slots),
            slots.last().unwrap() + 1,
            |region| {
                if let Region::Secondary {
                    data: selected,
                    block: _,
                    pointer: _,
                } = region
                {
                    regions += 1;
                    for item in selected {
                        item?;
                        data += 1;
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(regions, 17);
        assert_eq!(data, local.len());
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
        let measured = extensible_array::extensible_array_layout(
            SlotOccupancy::Slots(slots),
            count,
            8,
            planned.widths().offsets,
            planned.widths().lengths,
            false,
        )
        .unwrap();
        assert_eq!(planned.total_len(), measured.total_len);
        assert_eq!(planned.stats(), measured.stats);
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
