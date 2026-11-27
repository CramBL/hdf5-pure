use super::header;
use crate::chunk_record::{self, ChunkElementEncoding, IndexSlots};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::extensible_array::geometry::{DataBlock, Layout, SuperBlock};
use crate::extensible_array::{ExtensibleArrayCreationParameters, ExtensibleArrayStats};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

/// The slots of an Extensible Array that hold a chunk.
///
/// Measurement and planning use the occupied slots to select data blocks and secondary blocks.
#[derive(Clone, Copy)]
pub enum SlotOccupancy<'a> {
    /// The occupied prefix of the array, in slots.
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
/// [`ExtensibleArrayPlan`](super::plan::ExtensibleArrayPlan).
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
    measure(
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

/// Returns the length in bytes of the Extensible Array
/// [`build_extensible_array_at`](super::build_extensible_array_at) builds for `slots`, without
/// building it.
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

/// Computes canonical header statistics and wire lengths through the shared region traversal.
///
/// # Errors
///
/// Propagates occupancy and geometry errors from [`visit_regions`] and width errors from the
/// header statistics.
fn measure(
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

pub(super) fn element_layout(
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
    let elements = header::check_elements(widths.offsets, client_id, element_size)?;
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
pub(super) fn visit_regions(
    layout: Layout,
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    mut visit: impl FnMut(Region<'_>) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let occupancy = Occupancy::new(occupancy, num_slots)?;
    let mut covered = num_slots <= u64::from(layout.inline_elements());
    for (pointer, block) in layout.direct_blocks().enumerate() {
        let block = block?;
        let slots = block.slots(layout)?;
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
    for index in layout.first_super_block()..layout.super_blocks() {
        if covered {
            break;
        }
        let block = layout.super_block(index)?;
        let first = layout.first_slot(block.offset())?;
        covered =
            num_slots > first && (num_slots - 1 - first) / block.block_elements() < block.blocks();
        visit(Region::Secondary {
            block,
            pointer: u64::from(index - layout.first_super_block()),
            data: SelectedData::new(occupancy, layout, block)?,
        })?;
    }
    if !covered {
        return Err(FormatError::InvalidExtensibleArrayGeometry);
    }
    Ok(())
}

/// An occupied direct block or a secondary region with its selected data cursor.
pub(super) enum Region<'a> {
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

/// A cursor over occupied data blocks and their pointer slots in one secondary region.
pub(super) struct SelectedData<'a> {
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
        let first = layout.first_slot(block.offset())?;
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
        let local = (slot - *first) / block.block_elements();
        if local >= block.blocks() {
            return None;
        }
        Some((|| {
            let data = block.data_block(local)?;
            *next = occupancy.next(data.slots(*layout)?.end);
            Ok((data, local))
        })())
    }
}

#[cfg(test)]
#[path = "tests/occupancy.rs"]
mod tests;
