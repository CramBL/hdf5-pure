//! Block encoding from checked placements and borrowed chunk records.

use alloc::vec::Vec;

use super::BlockLocation;
use super::ExtensibleArrayBlock;
use super::ExtensibleArrayPlacement;
use super::ExtensibleArrayPlan;
use super::HEADER_BLOCK;
use super::block::{self, Addresses, Bitmap, ExtensibleArrayBlockWriter};
use crate::address::StoredAddress;
use crate::bytes::UnexpectedEof;
use crate::chunk_record::IndexSlots;
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::error::FormatError;

impl ExtensibleArrayPlacement<'_> {
    /// Checks chunk records against this placement's logical capacity.
    ///
    /// The slot span must match the plan. Occupied slots may differ from those used for planning,
    /// provided each fits the inline slots or a planned data block. The records remain borrowed,
    /// and the index of child references is built once for all emission handles.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ExtensibleArraySlotCount`] if the slot span differs from the plan,
    /// [`FormatError::ExtensibleArrayUnplannedSlot`] if an occupied slot has no planned storage,
    /// [`FormatError::DefinedAddressIsUndefined`] if a chunk address is undefined,
    /// [`FormatError::AddressTooLarge`] if a chunk address exceeds the address width, or
    /// [`FormatError::LengthTooLarge`] if a filtered chunk's stored size exceeds its field width.
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if a block's encoded length does not fit
    /// [`usize`], or [`FormatError::AllocationFailed`] if the reference index reservation fails.
    pub fn encoding<'a>(
        &'a self,
        slots: &'a IndexSlots<'a>,
    ) -> Result<ExtensibleArrayEncoding<'a>, FormatError> {
        ExtensibleArrayEncoding::new(self, slots)
    }
}

/// Borrowed chunk records and an index of child references for a checked placement.
///
/// Every occupied slot fits the plan's inline elements or a complete data block. The writer
/// derives page initialization bits from these records. Each emission handle borrows this
/// encoding and refers to one complete planned allocation.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
/// use hdf5_pure_format::__private::ExtensibleArrayPlan;
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::IndexSlots;
/// use hdf5_pure_format::__private::SlotOccupancy;
///
/// let slots = IndexSlots::dense(&[]);
/// let plan = ExtensibleArrayPlan::new(
///     ExtensibleArrayCreationParameters::canonical(),
///     SlotOccupancy::Sparse(&slots),
///     0,
///     8,
///     FormatWidths::from_sizes(8, 8)?,
///     false,
/// )?;
/// let addresses = [1_024, 4_096].map(StoredAddress::new);
/// let placement = plan.place(&addresses)?;
/// let encoding = placement.encoding(&slots)?;
/// let mut output = Vec::new();
/// let mut emitted = 0;
/// for (block, expected_address) in encoding.blocks().zip(addresses) {
///     assert_eq!(block.address(), expected_address);
///     output.resize(block.encoded_size(), 0);
///     emitted += block.write_into(&mut output)?;
/// }
/// assert_eq!(emitted, 370);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
pub struct ExtensibleArrayEncoding<'a> {
    placement: &'a ExtensibleArrayPlacement<'a>,
    slots: &'a IndexSlots<'a>,
    pointers: Vec<Pointer>,
}

impl<'a> ExtensibleArrayEncoding<'a> {
    fn new(
        placement: &'a ExtensibleArrayPlacement<'a>,
        slots: &'a IndexSlots<'a>,
    ) -> Result<Self, FormatError> {
        let ExtensibleArrayPlan {
            creation: _,
            widths: _,
            encoding: _,
            layout,
            blocks,
            stats,
            total_length: _,
        } = placement.plan;
        if slots.len().to_u64() != stats.max_idx_set {
            return Err(FormatError::ExtensibleArraySlotCount {
                expected: stats.max_idx_set,
                actual: slots.len().to_u64(),
            });
        }
        let mut next = slots.next_occupied(u64::from(layout.inline_elements));
        for block in blocks {
            let BlockLocation::Data(block) = &block.location else {
                continue;
            };
            let range = &block.slots;
            if let Some(slot) = next.filter(|&slot| slot < range.start) {
                return Err(FormatError::ExtensibleArrayUnplannedSlot { slot });
            }
            if next.is_some_and(|slot| slot < range.end) {
                next = slots.next_occupied(range.end);
            }
        }
        if let Some(slot) = next {
            return Err(FormatError::ExtensibleArrayUnplannedSlot { slot });
        }
        for chunk in slots.records() {
            layout.encoding.require_record_fits(chunk)?;
        }
        for block in blocks {
            block.length.to_usize()?;
        }
        let mut pointers = Vec::new();
        pointers
            .try_reserve_exact(blocks.len().saturating_sub(2))
            .map_err(|source| FormatError::AllocationFailed {
                values: blocks.len().saturating_sub(2),
                source,
            })?;
        let direct_blocks = layout.direct_data_blocks();
        for (descriptor, block) in blocks.iter().enumerate() {
            let (parent, slot) = match &block.location {
                BlockLocation::Header | BlockLocation::Index { .. } => continue,
                BlockLocation::Secondary(block) => {
                    (super::INDEX_BLOCK, direct_blocks + block.pointer)
                }
                BlockLocation::Data(block) => block.parent,
            };
            pointers.push(Pointer {
                parent,
                slot,
                descriptor,
            });
        }
        pointers.sort_unstable_by_key(|pointer| (pointer.parent, pointer.slot));
        Ok(Self {
            placement,
            slots,
            pointers,
        })
    }

    /// Returns emission handles in the placement's descriptor order.
    ///
    /// Repeated calls borrow the same records and reference index.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = ExtensibleArrayEmission<'_>> {
        (0..self.placement.ranges.len()).map(|index| ExtensibleArrayEmission {
            encoding: self,
            index,
        })
    }

    /// Returns the retained capacity of the child reference index, in entries.
    pub fn reference_capacity(&self) -> usize {
        self.pointers.capacity()
    }

    fn children(&self, parent: usize) -> &[Pointer] {
        let first = self
            .pointers
            .partition_point(|pointer| pointer.parent < parent);
        let count = self.pointers[first..].partition_point(|pointer| pointer.parent == parent);
        &self.pointers[first..first + count]
    }
}

/// The stored address and encoder of one complete Extensible Array allocation.
///
/// Each handle borrows the encoding from which it was produced. The caller establishes allocation
/// ownership and resolves its stored address for physical I/O.
pub struct ExtensibleArrayEmission<'a> {
    encoding: &'a ExtensibleArrayEncoding<'a>,
    index: usize,
}

impl ExtensibleArrayEmission<'_> {
    /// Returns the stored start address of the complete allocation.
    pub fn address(&self) -> StoredAddress {
        self.encoding.placement.ranges[self.index].start()
    }

    /// Returns the complete allocation descriptor from the borrowed plan.
    pub fn block(&self) -> &ExtensibleArrayBlock {
        &self.encoding.placement.plan.blocks[self.index]
    }

    /// Replaces `output` with this complete block, reusing its capacity.
    ///
    /// A paged data block includes its prefix and every page. The writer computes all checksums
    /// regardless of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if the output reservation fails, leaving
    /// `output` unchanged.
    #[expect(
        clippy::same_name_method,
        reason = "the existing Vec method reserves storage before using the bounded encoder"
    )]
    pub fn encode_into(&self, output: &mut Vec<u8>) -> Result<(), FormatError> {
        let length = self.encoded_len();
        reserve(output, length.saturating_sub(output.len()), length)?;
        output.resize(length, 0);
        self.write_block(output);
        Ok(())
    }

    /// Appends this complete block after the existing `output` bytes.
    ///
    /// Preserves existing bytes. The writer computes the block's checksums over appended bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if the output reservation fails, leaving
    /// `output` unchanged.
    pub(crate) fn append_into(&self, output: &mut Vec<u8>) -> Result<(), FormatError> {
        let length = self.encoded_len();
        let start = output.len();
        let end = start.saturating_add(length);
        reserve(output, length, end)?;
        output.resize(end, 0);
        self.write_block(&mut output[start..]);
        Ok(())
    }

    /// Returns the exact length in bytes of the complete allocation, including checksums.
    pub fn encoded_size(&self) -> usize {
        Encode::encoded_len(self)
    }

    /// Writes the complete allocation at the start of `output` and returns its length in bytes.
    ///
    /// A paged data block includes its prefix and every page. The writer computes all checksums
    /// regardless of the `checksum` feature and preserves any output suffix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is too short, leaving it unchanged.
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(Encode::encode_into(self, output)?)
    }

    /// Writes the allocation into a slice of exactly its encoded size.
    ///
    /// # Panics
    ///
    /// Can panic if `output` has the wrong length.
    fn write_block(&self, output: &mut [u8]) {
        let ExtensibleArrayEncoding {
            placement,
            slots,
            pointers: _,
        } = self.encoding;
        let layout = placement.plan.layout;
        let writer =
            ExtensibleArrayBlockWriter::placed(layout, placement.ranges[HEADER_BLOCK].start());
        let children = self.encoding.children(self.index);
        let addresses = Addresses::Placed {
            children,
            ranges: &placement.ranges,
        };
        let length = output.len();
        let block = match &self.block().location {
            BlockLocation::Header => {
                placement.header_encoding().write_header(output);
                return;
            }
            BlockLocation::Index { .. } => writer.index(length, slots, addresses),
            BlockLocation::Secondary(block) => {
                let offsets = usize::from(layout.encoding.offsets().get());
                #[cfg_attr(
                    target_pointer_width = "32",
                    expect(
                        clippy::cast_possible_truncation,
                        reason = "the checked block length fits usize and includes at least one byte per pointer"
                    )
                )]
                let pointer_count = block.geometry.blocks as usize;
                let bitmap_len = length - layout.pages_offset_usize() - pointer_count * offsets;
                let bitmap = Bitmap::Occupied {
                    slots,
                    children,
                    first_slot: block.first_slot,
                };
                writer.secondary(length, block.geometry, bitmap_len, bitmap, addresses)
            }
            BlockLocation::Data(block) => {
                writer.data(slots, block.slots.clone(), block.offset, length)
            }
        };
        block.write_block(output);
    }
}

impl Encode for ExtensibleArrayEmission<'_> {
    type Error = UnexpectedEof;

    #[cfg_attr(
        target_pointer_width = "32",
        expect(
            clippy::cast_possible_truncation,
            reason = "encoding preparation checks that every complete block length fits usize"
        )
    )]
    fn encoded_len(&self) -> usize {
        self.block().length as usize
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let length = self.encoded_len();
        self.write_block(block::destination(output, length)?);
        Ok(length)
    }
}

/// The parent and child descriptor indices and the encoded pointer slot between them.
pub(super) struct Pointer {
    pub(super) parent: usize,
    pub(super) slot: u64,
    pub(super) descriptor: usize,
}

fn reserve(output: &mut Vec<u8>, additional: usize, values: usize) -> Result<(), FormatError> {
    output
        .try_reserve_exact(additional)
        .map_err(|source| FormatError::AllocationFailed { values, source })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hdf5_pure_core::__private::BaseAddressExt;
    use hdf5_pure_core::__private::FileRange;
    use hdf5_pure_core::__private::FileRegion;
    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::BaseAddress;
    use hdf5_pure_core::PhysicalAddress;
    use rstest::rstest;
    use test_util::checksum as independent_checksum;

    use super::*;
    use crate::chunk_record::ChunkRecord;
    use crate::extensible_array;
    use crate::extensible_array::ExtensibleArrayCreationParameters;
    use crate::extensible_array::ExtensibleArrayHeaderLocator;
    use crate::extensible_array::SlotOccupancy;
    use crate::extensible_array::write::ExtensibleArrayBlockKind;
    use crate::extensible_array::write::ExtensibleArrayPlan;
    use crate::width::FormatWidths;

    fn chunks(count: usize, filtered: bool) -> Vec<ChunkRecord> {
        (0..count)
            .map(|index| ChunkRecord {
                address: StoredAddress::new(256 + u64::try_from(index).unwrap() * 8),
                stored_size: if filtered { 3 } else { 8 },
                filter_mask: if filtered { 5 } else { 0 },
            })
            .collect()
    }

    fn plan(
        creation: [u8; 5],
        slots: &IndexSlots<'_>,
        width: u8,
        filtered: bool,
    ) -> ExtensibleArrayPlan {
        ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::parse(&creation).unwrap(),
            SlotOccupancy::Sparse(slots),
            slots.len().to_u64(),
            8,
            FormatWidths::from_sizes(width, width).unwrap(),
            filtered,
        )
        .unwrap()
    }

    #[rstest]
    #[case::empty(0)]
    #[case::inline(4)]
    #[case::direct(20)]
    #[case::secondary(300)]
    #[case::paged(140_000)]
    fn contiguous_and_individual_encoding_agree_for_each_block(
        #[case] count: usize,
        #[values(false, true)] filtered: bool,
    ) {
        let records = chunks(count, filtered);
        let slots = IndexSlots::dense(&records);
        let plan = plan([32, 4, 4, 16, 10], &slots, 8, filtered);
        let mut cursor = StoredAddress::new(512);
        let addresses: Vec<_> = plan
            .block_lengths()
            .map(|length| {
                let current = cursor;
                cursor = cursor.checked_add(length).unwrap();
                current
            })
            .collect();
        let placement = plan.place(&addresses).unwrap();
        let encoding = placement.encoding(&slots).unwrap();
        let mut actual = Vec::new();
        let mut block = Vec::new();
        for emission in encoding.blocks() {
            emission.encode_into(&mut block).unwrap();
            actual.extend_from_slice(&block);
        }
        let expected = extensible_array::build_extensible_array_at(
            &slots,
            8,
            plan.widths().offsets,
            plan.widths().lengths,
            filtered,
            StoredAddress::new(512),
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.len().to_u64(), plan.total_len());
    }

    #[test]
    fn appending_blocks_preserves_the_prefix_and_each_block_checksum() {
        let records = chunks(4, true);
        let slots = IndexSlots::new(&records, &[0, 8, 32, 200], 201).unwrap();
        let plan = plan([24, 8, 2, 8, 8], &slots, 8, true);
        let addresses: Vec<_> = (0..plan.blocks().len())
            .map(|index| StoredAddress::new(1_024 + index.to_u64() * 4_096))
            .collect();
        let placement = plan.place(&addresses).unwrap();
        let encoding = placement.encoding(&slots).unwrap();
        let mut output = vec![0xa5; 17];
        let mut expected = Vec::new();
        for emission in encoding.blocks() {
            let start = output.len();
            emission.append_into(&mut output).unwrap();
            emission.encode_into(&mut expected).unwrap();
            assert_eq!(&output[start..], expected);
            assert_checksum(&output[start..]);
        }
        assert_eq!(&output[..17], &[0xa5; 17]);
        assert_eq!(output.len().to_u64(), plan.total_len() + 17);
    }

    #[rstest]
    #[case::empty([24, 8, 2, 8, 8], &[], 0)]
    #[case::inline([24, 8, 2, 8, 8], &[0, 7], 8)]
    #[case::direct([24, 8, 2, 8, 8], &[8, 16, 31], 32)]
    #[case::secondary([24, 8, 2, 8, 8], &[0, 8, 32, 63, 200], 201)]
    #[case::paged([16, 2, 2, 8, 5], &[506, 546, 634, 666], 700)]
    fn independent_blocks_preserve_references_records_and_complete_ranges(
        #[case] creation: [u8; 5],
        #[case] occupied: &[u64],
        #[case] count: u64,
        #[values(2, 4, 8)] width: u8,
        #[values(false, true)] filtered: bool,
    ) {
        let records = chunks(occupied.len(), filtered);
        let slots = IndexSlots::new(&records, occupied, count).unwrap();
        let plan = plan(creation, &slots, width, filtered);
        let stride = plan.block_lengths().max().unwrap() + 17;
        let addresses: Vec<_> = (0..plan.blocks().len())
            .map(|index| StoredAddress::new(128 + stride * (plan.blocks().len() - index).to_u64()))
            .collect();
        let placement = plan.place(&addresses).unwrap();
        let encoding = placement.encoding(&slots).unwrap();
        let mut image = vec![
            0;
            (addresses[0].get() + plan.blocks().next().unwrap().length())
                .to_usize()
                .unwrap()
        ];
        let mut output = Vec::new();
        for emission in encoding.blocks() {
            emission.encode_into(&mut output).unwrap();
            let start = emission.address().get().to_usize().unwrap();
            image[start..start + output.len()].copy_from_slice(&output);
            let descriptor = emission.block();
            if descriptor.kind() != ExtensibleArrayBlockKind::Data || descriptor.length() == 0 {
                assert_checksum(&output);
            } else {
                let range = descriptor.slots().unwrap();
                if let Some(pages) = plan.layout.pages(range.end - range.start) {
                    let prefix = plan.layout.pages_offset().to_usize().unwrap();
                    assert_checksum(&output[..prefix]);
                    let page_len = plan.layout.page_len().unwrap().to_usize().unwrap();
                    assert_eq!(
                        output[prefix..].chunks_exact(page_len).len().to_u64(),
                        pages
                    );
                    for page in output[prefix..].chunks_exact(page_len) {
                        assert_checksum(page);
                    }
                } else {
                    assert_checksum(&output);
                }
            }
        }
        let base = BaseAddress::new(1_u64 << 40);
        let region = FileRegion::new(
            FileRange::new(PhysicalAddress::new(base.get()), image.len().to_u64()).unwrap(),
            &image,
        )
        .unwrap();
        let locator = ExtensibleArrayHeaderLocator::new(plan.widths(), base, addresses[0]);
        let array = locator.parse(region).unwrap();
        let header = array.header();
        assert_eq!(header.index_block_address(), addresses[1]);
        assert_eq!(header.creation_parameters().encode(), creation);
        assert_eq!(
            extensible_array::tests::records_in(region, locator, 8).unwrap(),
            occupied.iter().copied().zip(records).collect::<Vec<_>>()
        );
        let index = array.index_block().unwrap().parse(region).unwrap();
        let mut data: Vec<_> = index.data_blocks().collect::<Result<_, _>>().unwrap();
        let mut complete = vec![array.range(), array.index_block().unwrap().range()];
        for reference in index.super_blocks() {
            let reference = reference.unwrap();
            complete.push(reference.parse_range());
            data.extend(
                reference
                    .parse(region)
                    .unwrap()
                    .data_blocks()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
            );
        }
        for reference in data {
            complete.push(reference.range());
            if let Some(pages) = reference.pages().unwrap() {
                assert!(reference.parse_range().length() < reference.range().length());
                for page in pages.pages() {
                    assert!(reference.range().contains(page.unwrap().parse_range()));
                }
            }
        }
        complete.sort_by_key(|range| range.start());
        let mut expected: Vec<_> = placement
            .blocks()
            .map(|(address, length)| {
                FileRange::new(base.resolve(address).unwrap(), length).unwrap()
            })
            .collect();
        expected.sort_by_key(|range| range.start());
        assert_eq!(complete, expected);
    }

    #[test]
    fn paged_bitmap_marks_only_occupied_pages_and_preserves_pointer_holes() {
        let records = chunks(3, false);
        let slots = IndexSlots::new(&records, &[506, 546, 666], 700).unwrap();
        let plan = plan([16, 2, 2, 8, 5], &slots, 8, false);
        let addresses: Vec<_> = (0..plan.blocks().len())
            .map(|index| StoredAddress::new(1_024 + index.to_u64() * 4_096))
            .collect();
        let placement = plan.place(&addresses).unwrap();
        let encoding = placement.encoding(&slots).unwrap();
        let mut output = Vec::new();
        let secondary = encoding
            .blocks()
            .find(|emission| emission.block().kind() == ExtensibleArrayBlockKind::Secondary)
            .unwrap();
        secondary.encode_into(&mut output).unwrap();
        assert_eq!(&output[..4], b"EASB");
        assert_eq!(&output[14..16], &504_u16.to_le_bytes());
        assert_eq!(&output[16..24], &[0xc4, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&output[24..32], &addresses[2].get().to_le_bytes());
        assert_eq!(&output[32..40], &u64::MAX.to_le_bytes());
        assert_eq!(&output[40..48], &addresses[3].get().to_le_bytes());
        assert_checksum(&output);
    }

    #[test]
    fn actual_occupancy_sets_page_bits_within_retained_allocations() {
        let records = chunks(1, false);
        let predecessor = IndexSlots::new(&records, &[506], 700).unwrap();
        let plan = plan([16, 2, 2, 8, 5], &predecessor, 8, false);
        let addresses: Vec<_> = (0..plan.blocks().len())
            .map(|index| StoredAddress::new(1_024 + index.to_u64() * 4_096))
            .collect();
        let placement = plan.place(&addresses).unwrap();
        let actual = IndexSlots::new(&records, &[546], 700).unwrap();
        let encoding = placement.encoding(&actual).unwrap();
        let mut output = Vec::new();
        for emission in encoding.blocks() {
            emission.encode_into(&mut output).unwrap();
            if emission.block().kind() == ExtensibleArrayBlockKind::Secondary {
                assert_eq!(output[16], 0x40);
            } else if emission.block().kind() == ExtensibleArrayBlockKind::Data {
                assert_eq!(&output[20..28], &u64::MAX.to_le_bytes());
                assert_eq!(&output[344..352], &records[0].address.get().to_le_bytes());
            }
        }
    }

    #[test]
    fn independent_empty_initialization_matches_existing_canonical_bytes() {
        let slots = IndexSlots::dense(&[]);
        let plan = plan([32, 4, 4, 16, 10], &slots, 8, true);
        let addresses = [StoredAddress::new(1_024), StoredAddress::new(128)];
        let placement = plan.place(&addresses).unwrap();
        let encoding = placement.encoding(&slots).unwrap();
        let mut output = Vec::new();
        for emission in encoding.blocks() {
            emission.encode_into(&mut output).unwrap();
            assert_checksum(&output);
            assert_eq!(output.len().to_u64(), emission.block().length());
            if emission.block().kind() == ExtensibleArrayBlockKind::Header {
                assert_eq!(&output[60..68], &128_u64.to_le_bytes());
                assert_eq!(&output[12..52], &[0; 40]);
                assert_eq!(&output[52..60], &4_u64.to_le_bytes());
            } else {
                assert_eq!(&output[6..14], &1_024_u64.to_le_bytes());
                assert_eq!(&output[14..22], &u64::MAX.to_le_bytes());
                assert_eq!(&output[22..28], &[0; 6]);
            }
        }
    }

    #[test]
    fn one_reference_index_and_one_block_buffer_serve_repeated_emission() {
        let records = chunks(4, true);
        let slots = IndexSlots::new(&records, &[0, 8, 32, 200], 201).unwrap();
        let plan = plan([24, 8, 2, 8, 8], &slots, 8, true);
        let addresses: Vec<_> = (0..plan.blocks().len())
            .map(|index| StoredAddress::new(1_024 + index.to_u64() * 4_096))
            .collect();
        let placement = plan.place(&addresses).unwrap();
        let encoding = placement.encoding(&slots).unwrap();
        assert_eq!(encoding.reference_capacity(), plan.blocks().len() - 2);
        let maximum = plan.block_lengths().max().unwrap().to_usize().unwrap();
        assert!(maximum.to_u64() < plan.total_len());
        let mut output = Vec::with_capacity(maximum);
        let buffer = output.as_ptr();
        let references = encoding.pointers.as_ptr();
        let mut emitted = 0;
        for emission in encoding.blocks().chain(encoding.blocks()) {
            emission.encode_into(&mut output).unwrap();
            assert_eq!(output.as_ptr(), buffer);
            assert_eq!(output.capacity(), maximum);
            assert_eq!(encoding.pointers.as_ptr(), references);
            emitted += output.len().to_u64();
        }
        assert_eq!(emitted, 2 * plan.total_len());
    }

    #[test]
    fn encoding_rejects_a_changed_slot_span() {
        let empty = IndexSlots::dense(&[]);
        let plan = plan([32, 4, 4, 16, 10], &empty, 8, false);
        let placement = plan
            .place(&[StoredAddress::new(0), StoredAddress::new(1_024)])
            .unwrap();
        let records = chunks(1, false);
        let slots = IndexSlots::dense(&records);
        let error = placement.encoding(&slots).err().unwrap();
        assert!(
            matches!(
                error,
                FormatError::ExtensibleArraySlotCount {
                    expected: 0,
                    actual: 1
                }
            ),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::before_the_first_block(4, 20)]
    #[case::at_the_planned_block_end(20, 4)]
    #[case::after_the_last_block(300, 4)]
    fn encoding_rejects_occupied_slots_without_planned_blocks(
        #[case] actual: u64,
        #[case] original: u64,
    ) {
        let records = chunks(1, false);
        let original_slots = IndexSlots::new(&records, &[original], 301).unwrap();
        let plan = plan([32, 4, 4, 16, 10], &original_slots, 8, false);
        let addresses: Vec<_> = (0..plan.blocks().len())
            .map(|index| StoredAddress::new(index.to_u64() * 4_096))
            .collect();
        let placement = plan.place(&addresses).unwrap();
        let actual_slots = IndexSlots::new(&records, &[actual], 301).unwrap();
        let error = placement.encoding(&actual_slots).err().unwrap();
        assert!(
            matches!(error, FormatError::ExtensibleArrayUnplannedSlot { slot } if slot == actual),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::undefined(65_535, 3, true)]
    #[case::too_wide(65_536, 3, false)]
    #[case::stored_size(256, 65_536, false)]
    fn encoding_rejects_unencodable_records(
        #[case] address: u64,
        #[case] stored_size: u64,
        #[case] undefined: bool,
    ) {
        let records = [ChunkRecord {
            address: StoredAddress::new(address),
            stored_size,
            filter_mask: 0,
        }];
        let slots = IndexSlots::dense(&records);
        let plan = plan([24, 8, 2, 8, 8], &slots, 2, true);
        let placement = plan
            .place(&[StoredAddress::new(0), StoredAddress::new(128)])
            .unwrap();
        let error = placement.encoding(&slots).err().unwrap();
        if undefined {
            assert!(
                matches!(
                    error,
                    FormatError::DefinedAddressIsUndefined {
                        address: 65_535,
                        offset_size: 2
                    }
                ),
                "{error:?}"
            );
        } else if address > 65_535 {
            assert!(
                matches!(
                    error,
                    FormatError::AddressTooLarge {
                        address: 65_536,
                        offset_size: 2
                    }
                ),
                "{error:?}"
            );
        } else {
            assert!(
                matches!(
                    error,
                    FormatError::LengthTooLarge {
                        length: 65_536,
                        length_size: 2
                    }
                ),
                "{error:?}"
            );
        }
    }

    fn assert_checksum(bytes: &[u8]) {
        let (body, checksum) = bytes.split_at(bytes.len() - 4);
        assert_eq!(checksum, independent_checksum::lookup3(body).to_le_bytes());
    }
}
