use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;

use crate::fractal_heap::FractalHeapGeometry;
use crate::fractal_heap::write::{
    GROUP_HEAP_MAX_MANAGED_OBJECT, ManagedHeapPlan, ManagedHeapPlanError,
};
use crate::width::OffsetWidth;

#[rstest]
#[case::attribute_direct(false, vec![1002], 1024, 0, 0, 0)]
#[case::attribute_exact_fill(false, vec![1001, 1], 1024, 0, 0, 0)]
#[case::attribute_skips_starting_blocks(false, vec![1003], 16_384, 3, 10_240, 15_117)]
#[case::attribute_indirect(false, vec![1002, 1], 4096, 1, 2048, 3005)]
#[case::group_direct(true, vec![491], 512, 0, 0, 0)]
#[case::group_indirect(true, vec![491, 1], 2048, 1, 1024, 1472)]
fn header_statistics_match_known_root_shapes(
    #[case] group: bool,
    #[case] sizes: Vec<u64>,
    #[case] managed: u64,
    #[case] rows: u16,
    #[case] iterator: u64,
    #[case] free: u64,
) {
    let plan = if group {
        ManagedHeapPlan::group_links(&sizes, OffsetWidth::Eight)
    } else {
        ManagedHeapPlan::attributes(&sizes, OffsetWidth::Eight)
    }
    .unwrap();
    assert_eq!(plan.managed_space(), managed);
    assert_eq!(plan.root_rows(), rows);
    assert_eq!(plan.allocation_iterator(), iterator);
    assert_eq!(plan.free_space(), free);
}

#[rstest]
#[case::zero(0, 65_514)]
#[case::past_largest_block(65_515, 65_514)]
#[case::unrepresentable_size(u64::MAX, 65_514)]
fn invalid_object_sizes_are_rejected(#[case] size: u64, #[case] maximum: u64) {
    let error = ManagedHeapPlan::attributes(&[size], OffsetWidth::Eight)
        .err()
        .unwrap();
    assert_eq!(error, ManagedHeapPlanError::ObjectSize { size, maximum });
}

#[rstest]
#[case::zero(0)]
#[case::above_limit(GROUP_HEAP_MAX_MANAGED_OBJECT + 1)]
fn invalid_group_object_sizes_report_the_canonical_managed_limit(#[case] size: u64) {
    let error = ManagedHeapPlan::group_links(&[size], OffsetWidth::Eight)
        .err()
        .unwrap();
    assert_eq!(
        error,
        ManagedHeapPlanError::ObjectSize {
            size,
            maximum: GROUP_HEAP_MAX_MANAGED_OBJECT,
        }
    );
}

#[test]
fn blocks_too_small_for_the_metadata_prefix_are_rejected() {
    let geometry = FractalHeapGeometry::new(4, 16, 32, 16).unwrap();
    assert_eq!(
        ManagedHeapPlan::new(&[1], OffsetWidth::Eight, geometry)
            .err()
            .unwrap(),
        ManagedHeapPlanError::Geometry
    );
}

#[rstest]
#[case::at_boundary(2, true)]
#[case::past_boundary(3, false)]
#[case::further_past_boundary(5, false)]
fn heap_allocation_accepts_the_last_block_and_rejects_further_objects(
    #[case] count: usize,
    #[case] succeeds: bool,
) {
    let geometry = FractalHeapGeometry::new(2, 64, 128, 9).unwrap();
    let result = ManagedHeapPlan::new(&vec![100; count], OffsetWidth::Eight, geometry);
    if succeeds {
        let plan = result.unwrap();
        assert_eq!(plan.managed_space(), 512);
        assert_eq!(plan.allocation_iterator(), 512);
    } else {
        assert_eq!(result.err().unwrap(), ManagedHeapPlanError::HeapSpace);
    }
}

#[test]
fn region_lengths_are_checked_before_host_materialization() {
    let geometry = FractalHeapGeometry::new(2, 1u64 << 32, 1u64 << 32, 35).unwrap();
    let result = ManagedHeapPlan::new(&[], OffsetWidth::Eight, geometry);
    if usize::BITS == 32 {
        assert_eq!(
            result.err().unwrap(),
            ManagedHeapPlanError::Host { bytes: 1u64 << 32 }
        );
    } else {
        assert_eq!(result.unwrap().region_size(), 1u64 << 32);
    }
}
