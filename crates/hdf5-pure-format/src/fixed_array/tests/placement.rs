use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use rstest::rstest;
use test_util::checksum;

use super::super::FixedArrayDataBlock;
use super::super::FixedArrayHeaderLocator;
use super::super::FixedArrayPlan;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::IndexSlots;
use crate::error::FormatError;
use crate::fixed_array;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
fn independent_allocations_preserve_complete_data_blocks(
    #[values(0, 1, 1024, 1025, 2049)] entries: u64,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] reversed: bool,
    #[values(0, u64::from(u32::MAX) + 1)] base: u64,
) {
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Eight,
    };
    let occupied_slots: Vec<_> = match entries {
        0 => Vec::new(),
        1 => vec![0],
        _ => vec![0, entries - 1],
    };
    let chunks: Vec<_> = occupied_slots
        .iter()
        .map(|&slot| ChunkRecord {
            address: StoredAddress::new(0x200 + slot % 16),
            stored_size: 8,
            filter_mask: u32::from(filtered),
        })
        .collect();
    let slots = IndexSlots::new(&chunks, &occupied_slots, entries).unwrap();
    let plan = FixedArrayPlan::new(entries, 8, widths, filtered).unwrap();
    let addresses = if reversed {
        [0xb000, 0x1000]
    } else {
        [0x40, 0x4000]
    }
    .map(StoredAddress::new);
    let placement = plan.place(addresses[0], addresses[1]).unwrap();
    let lengths: Vec<_> = plan.block_lengths().collect();
    assert_eq!(lengths.iter().sum::<u64>(), plan.total_len());
    assert_eq!(
        placement.blocks().collect::<Vec<_>>(),
        [(addresses[0], lengths[0]), (addresses[1], lengths[1])]
    );
    let [(header_at, header), (data_at, data)] = placement.serialize_blocks(&slots).unwrap();
    assert_eq!([header.len() as u64, data.len() as u64], lengths.as_slice());
    assert_checksum(&header);
    let base = BaseAddress::new(base);
    let header_region = FileRegion::new(
        FileRange::new(base.resolve(header_at).unwrap(), lengths[0]).unwrap(),
        &header,
    )
    .unwrap();
    let data_region = FileRegion::new(
        FileRange::new(base.resolve(data_at).unwrap(), lengths[1]).unwrap(),
        &data,
    )
    .unwrap();
    let header = FixedArrayHeaderLocator::new(widths, base, header_at)
        .parse(header_region)
        .unwrap();
    assert_eq!(header.range(), header_region.range());
    let block = header.data_block().unwrap();
    assert_eq!(block.range(), data_region.range());
    assert_checksum(
        data_region
            .subregion(block.parse_range().unwrap())
            .unwrap()
            .bytes(),
    );
    let mut actual = Vec::new();
    match block.parse(data_region).unwrap() {
        FixedArrayDataBlock::Elements(elements) => {
            assert_eq!(block.parse_range().unwrap(), block.range());
            actual.extend(elements.records(8).collect::<Result<Vec<_>, _>>().unwrap());
        }
        FixedArrayDataBlock::PageBitmap(bitmap) => {
            let mut end = block.parse_range().unwrap().end();
            for page in bitmap.pages() {
                let page = page.unwrap();
                assert_eq!(page.parse_range().start(), end);
                assert!(block.range().contains(page.parse_range()));
                end = page.parse_range().end();
                assert_checksum(data_region.subregion(page.parse_range()).unwrap().bytes());
                actual.extend(
                    page.parse(data_region)
                        .unwrap()
                        .records(8)
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                );
            }
            assert_eq!(end, block.range().end());
            assert!(block.parse_range().unwrap().length() < block.range().length());
        }
    }
    assert_eq!(
        actual,
        occupied_slots
            .iter()
            .copied()
            .zip(chunks.iter().copied())
            .collect::<Vec<_>>()
    );
    let contiguous =
        fixed_array::build_fixed_array_at(&slots, 8, offsets, widths.lengths, filtered, header_at)
            .unwrap();
    let [(.., mut first), (.., second)] = plan
        .place(header_at, header_at.checked_add(lengths[0]).unwrap())
        .unwrap()
        .serialize_blocks(&slots)
        .unwrap();
    first.extend(second);
    assert_eq!(contiguous, first);
}

#[test]
fn planning_preserves_geometry_beyond_u32_lengths() {
    let entries = 1u64 << 33;
    let plan = FixedArrayPlan::new(entries, 8, WIDTHS, false).unwrap();
    let expected_data =
        14 + entries.div_ceil(1024).div_ceil(8) + 4 + entries * 8 + entries.div_ceil(1024) * 4;
    assert_eq!(
        plan.block_lengths().collect::<Vec<_>>(),
        [28, expected_data]
    );
    assert_eq!(plan.total_len(), expected_data + 28);
    assert!(plan.total_len() > u64::from(u32::MAX));
}

#[rstest]
#[case::entry_width(65_536, LengthWidth::Two, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
#[case::arithmetic(u64::MAX, LengthWidth::Eight, FormatError::InvalidFixedArrayGeometry)]
fn planning_rejects_unrepresentable_geometry(
    #[case] entries: u64,
    #[case] lengths: LengthWidth,
    #[case] expected: FormatError,
) {
    assert_eq!(
        FixedArrayPlan::new(
            entries,
            8,
            FormatWidths {
                offsets: OffsetWidth::Eight,
                lengths
            },
            false
        )
        .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::undefined_header(65_535, 32, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::undefined_data(16, 65_535, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::wide_header(65_536, 32, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::wide_data(16, 65_536, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::header_end(65_530, 32, FormatError::AddressTooLarge { address: 65_546, offset_size: 2 })]
#[case::data_end(16, 65_530, FormatError::AddressTooLarge { address: 65_542, offset_size: 2 })]
#[case::overlap(16, 20, FormatError::FixedArrayBlockOverlap { header: 16, data_block: 20 })]
fn placement_rejects_invalid_complete_ranges(
    #[case] header: u64,
    #[case] data: u64,
    #[case] expected: FormatError,
) {
    let plan = FixedArrayPlan::new(
        0,
        8,
        FormatWidths {
            offsets: OffsetWidth::Two,
            lengths: LengthWidth::Two,
        },
        false,
    )
    .unwrap();
    assert_eq!(
        plan.place(StoredAddress::new(header), StoredAddress::new(data))
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::header(true)]
#[case::data(false)]
fn placement_checks_u64_exclusive_ends(#[case] header: bool) {
    let plan = FixedArrayPlan::new(0, 8, WIDTHS, false).unwrap();
    let start = u64::MAX - 10;
    let (header_at, data_at, length) = if header {
        (start, 64, 28)
    } else {
        (64, start, 18)
    };
    assert_eq!(
        plan.place(StoredAddress::new(header_at), StoredAddress::new(data_at))
            .unwrap_err(),
        FormatError::OffsetOverflow {
            offset: start,
            length
        }
    );
}

#[test]
fn placement_accepts_adjacent_blocks_and_the_maximum_exclusive_end() {
    let widths = FormatWidths {
        offsets: OffsetWidth::Two,
        lengths: LengthWidth::Two,
    };
    let plan = FixedArrayPlan::new(0, 8, widths, false).unwrap();
    assert_eq!(
        plan.place(StoredAddress::new(65_507), StoredAddress::new(65_523))
            .unwrap()
            .blocks()
            .collect::<Vec<_>>(),
        [
            (StoredAddress::new(65_507), 16),
            (StoredAddress::new(65_523), 12)
        ]
    );
}

#[rstest]
#[case::undefined(65_535, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::wide(65_536, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
fn emission_rejects_unencodable_chunk_addresses(
    #[case] address: u64,
    #[case] expected: FormatError,
) {
    let chunks = [ChunkRecord {
        address: StoredAddress::new(address),
        stored_size: 8,
        filter_mask: 0,
    }];
    let slots = IndexSlots::dense(&chunks);
    let plan = FixedArrayPlan::new(
        1,
        8,
        FormatWidths {
            offsets: OffsetWidth::Two,
            lengths: LengthWidth::Eight,
        },
        false,
    )
    .unwrap();
    let placement = plan
        .place(StoredAddress::new(32), StoredAddress::new(64))
        .unwrap();
    assert_eq!(placement.serialize_blocks(&slots).unwrap_err(), expected);
}

#[test]
fn emission_requires_the_measured_slot_count() {
    let placement = FixedArrayPlan::new(1, 8, WIDTHS, false)
        .unwrap()
        .place(StoredAddress::new(32), StoredAddress::new(64))
        .unwrap();
    assert_eq!(
        placement
            .serialize_blocks(&IndexSlots::dense(&[]))
            .unwrap_err(),
        FormatError::FixedArrayEntryCount {
            expected: 1,
            actual: 0
        }
    );
}

#[test]
fn emission_reports_a_local_capacity_overflow() {
    let entries = u64::try_from(isize::MAX).unwrap() / 8 + 1;
    let slots = IndexSlots::new(&[], &[], entries).unwrap();
    let plan = FixedArrayPlan::new(entries, 8, WIDTHS, false).unwrap();
    let length = plan.block_lengths().nth(1).unwrap();
    let placement = plan
        .place(StoredAddress::new(0), StoredAddress::new(28))
        .unwrap();
    let error = placement.serialize_blocks(&slots).unwrap_err();
    let FormatError::AllocationFailed { values, .. } = error else {
        panic!("expected capacity overflow: {error:?}");
    };
    assert_eq!(values, usize::try_from(length).unwrap());
}

#[test]
fn emission_preserves_filtered_size_errors() {
    let chunks = [ChunkRecord {
        address: StoredAddress::new(512),
        stored_size: 65_536,
        filter_mask: 0,
    }];
    let slots = IndexSlots::dense(&chunks);
    let placement = FixedArrayPlan::new(1, 8, WIDTHS, true)
        .unwrap()
        .place(StoredAddress::new(32), StoredAddress::new(64))
        .unwrap();
    assert_eq!(
        placement.serialize_blocks(&slots).unwrap_err(),
        FormatError::Internal(alloc::format!(
            "a chunk of {} bytes does not fit a 2-byte chunk size field",
            65_536
        ))
    );
}

fn assert_checksum(bytes: &[u8]) {
    let (body, actual) = bytes.split_last_chunk::<{ size_of::<u32>() }>().unwrap();
    assert_eq!(actual, &checksum::lookup3(body).to_le_bytes());
}

const WIDTHS: FormatWidths = FormatWidths {
    offsets: OffsetWidth::Eight,
    lengths: LengthWidth::Eight,
};
