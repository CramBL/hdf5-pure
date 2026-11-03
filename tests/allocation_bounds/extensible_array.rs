use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::ExtensibleArrayBlock;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayPlan;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::SlotOccupancy;
use rstest::rstest;
use test_util::allocation;

#[rstest]
#[case::secondary(300)]
#[case::paged(140_000)]
#[case::many_blocks(20_000_000)]
fn one_shot_measurement_keeps_descriptor_allocations_out_of_its_working_set(#[case] count: u64) {
    let region = format!("ea_measurement_{count}");
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let (plan, control) = allocation::measure(&format!("{region}_plan"), || {
        ExtensibleArrayPlan::new(
            ExtensibleArrayCreationParameters::canonical(),
            SlotOccupancy::Dense(count),
            count,
            8,
            widths,
            false,
        )
        .unwrap()
    });
    let (measured, costs) = allocation::measure(&region, || {
        format::extensible_array_layout(
            SlotOccupancy::Dense(count),
            count,
            8,
            widths.offsets,
            widths.lengths,
            false,
        )
        .unwrap()
    });
    assert_eq!(measured.stats, plan.stats());
    let descriptor_floor =
        u64::try_from(plan.blocks().len() * size_of::<ExtensibleArrayBlock>()).unwrap();
    assert!(control.live_bytes >= descriptor_floor, "{control:?}");
    assert!(control.blocks > 0, "{control:?}");
    assert!(costs.live_bytes > 0, "{costs:?}");
    assert!(costs.blocks <= 4, "{costs:?}");
    assert!(costs.bytes <= 2_048, "{costs:?}");
    assert!(costs.peak_bytes <= 2_048, "{costs:?}");
    assert!(
        costs.live_bytes < control.live_bytes,
        "{costs:?}; {control:?}"
    );
}
