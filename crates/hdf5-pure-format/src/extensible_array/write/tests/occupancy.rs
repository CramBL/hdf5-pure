use alloc::vec::Vec;

use rstest::rstest;

use super::*;
use crate::extensible_array;
use crate::extensible_array::write::{ExtensibleArrayBlockKind, test_support};

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
    assert_eq!(block.blocks(), 1_024);
    let first = layout.first_slot(block.offset()).unwrap();
    let slots: Vec<_> = local
        .iter()
        .map(|&local| first + local * block.block_elements())
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
            .map(|&(data, local)| (data.offset(), local))
            .collect::<Vec<_>>(),
        local
            .iter()
            .map(|&local| (block.offset() + local * block.block_elements(), local))
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
    let planned = test_support::plan(&[32, 4, 4, 16, 10], slots, count, 8);
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
