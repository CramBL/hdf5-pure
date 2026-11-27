use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::StoredAddressExt;
use rstest::rstest;

use super::*;
use crate::address::StoredAddress;
use crate::chunk_record::{ChunkRecord, IndexSlots};
use crate::extensible_array::write::test_support;

#[test]
fn dense_and_explicit_occupancy_plan_the_same_blocks() {
    let slots: Vec<_> = (0..300).collect();
    let explicit = test_support::plan(&[32, 4, 4, 16, 10], &slots, 300, 8);
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
    let explicit = test_support::plan(&[32, 4, 4, 16, 10], &[0, 4, 500], 600, 8);
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
    let planned = test_support::plan(&[16, 2, 2, 8, 5], &[first], first + 1, 8);
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
    let planned = test_support::plan(&[32, 4, 4, 16, 10], &[244, 436], 437, 8);
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
    let planned = test_support::plan(&[64, 4, 4, 16, 10], &[slot], slot + 1, 8);
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
fn planning_rejects_unrepresentable_complete_geometry(#[case] creation: &[u8], #[case] slot: u64) {
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
