use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::AddressRange;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::ExtensibleArrayBlock;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayInitialization;
use hdf5_pure_format::__private::ExtensibleArrayPlan;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::LengthWidth;
use hdf5_pure_format::__private::OffsetWidth;
use hdf5_pure_format::__private::SlotOccupancy;
use rstest::rstest;
use test_util::allocation;
use test_util::allocation::Measured;

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
    let Measured {
        blocks,
        bytes,
        peak_bytes,
        live_bytes,
    } = costs;
    assert_eq!(blocks, 0, "{costs:?}");
    assert_eq!(bytes, 0, "{costs:?}");
    assert_eq!(peak_bytes, 0, "{costs:?}");
    assert_eq!(live_bytes, 0, "{costs:?}");
    assert!(live_bytes < control.live_bytes, "{costs:?}; {control:?}");
}

#[rstest]
#[case::secondary(300)]
#[case::paged(140_000)]
fn contiguous_encoding_retains_only_its_output_and_bounded_plan_scratch(
    #[case] count: usize,
    #[values(false, true)] filtered: bool,
) {
    let chunks: Vec<_> = (0..count)
        .map(|index| ChunkRecord {
            address: StoredAddress::new(4_096 + u64::try_from(index).unwrap() * 8),
            stored_size: 8,
            filter_mask: 0,
        })
        .collect();
    let slots = IndexSlots::dense(&chunks);
    let expected =
        format::extensible_array_len(&slots, 8, OffsetWidth::Eight, LengthWidth::Eight, filtered)
            .unwrap();
    let (encoded, measured) =
        allocation::measure(&format!("ea_contiguous_{count}_{filtered}"), || {
            format::build_extensible_array_at(
                &slots,
                8,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                filtered,
                StoredAddress::new(512),
            )
            .unwrap()
        });
    let Measured {
        blocks,
        bytes,
        peak_bytes,
        live_bytes,
    } = measured;
    assert_eq!(u64::try_from(encoded.len()).unwrap(), expected);
    assert_eq!(
        live_bytes,
        u64::try_from(encoded.capacity()).unwrap(),
        "{measured:?}"
    );
    assert!(bytes >= expected, "{measured:?}");
    assert!(blocks > 0, "{measured:?}");
    assert!(peak_bytes < 2 * expected, "{measured:?}");
}

#[rstest]
#[case::two_bytes(2)]
#[case::eight_bytes(8)]
fn retained_empty_initialization_reuses_its_plan_across_output_addresses(
    #[case] offset_size: u8,
    #[values(false, true)] filtered: bool,
) {
    let widths = FormatWidths::from_sizes(offset_size, 8).unwrap();
    let initialization = ExtensibleArrayInitialization::new(
        ExtensibleArrayCreationParameters::canonical(),
        8,
        widths,
        filtered,
    )
    .unwrap();
    let (encoded, measured) = allocation::measure(
        &format!("ea_initialization_{offset_size}_{filtered}"),
        || initialization.encode_at(StoredAddress::new(1_024)).unwrap(),
    );
    let Measured {
        blocks,
        bytes,
        peak_bytes,
        live_bytes,
    } = measured;
    assert_eq!(
        u64::try_from(encoded.len()).unwrap(),
        initialization.allocation_len()
    );
    assert_eq!(
        live_bytes,
        u64::try_from(encoded.capacity()).unwrap(),
        "{measured:?}"
    );
    assert!(bytes >= initialization.allocation_len(), "{measured:?}");
    assert!(blocks <= 2, "{measured:?}");
    assert!(
        peak_bytes
            <= live_bytes + 2 * u64::try_from(size_of::<AddressRange<StoredAddress>>()).unwrap(),
        "{measured:?}"
    );
    let other = initialization.encode_at(StoredAddress::new(4_096)).unwrap();
    assert_eq!(other.len(), encoded.len());
    assert_ne!(other, encoded);
}
