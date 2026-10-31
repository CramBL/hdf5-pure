use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_space::__private::{Extent, FreeList, FreeListError};
use rstest::rstest;

#[rstest]
#[case::empty(&[], &[], 0)]
#[case::single(&[(10, 20)], &[(10, 20)], 20)]
#[case::disjoint(&[(10, 20), (40, 10)], &[(10, 20), (40, 10)], 30)]
#[case::adjacent(&[(10, 20), (30, 10), (40, 5)], &[(10, 35)], 35)]
#[case::mixed(&[(0, 10), (10, 20), (40, 10), (50, 5)], &[(0, 30), (40, 15)], 45)]
#[case::maximum_end(&[(u64::MAX - 10, 10)], &[(u64::MAX - 10, 10)], 10)]
#[case::maximum_total(&[(0, u64::MAX - 10), (u64::MAX - 10, 10)], &[(0, u64::MAX)], u64::MAX)]
fn sorted_construction_coalesces_adjacent_extents(
    #[case] input: &[(u64, u64)],
    #[case] expected: &[(u64, u64)],
    #[case] total: u64,
) {
    let bulk = FreeList::from_sorted(extents(input)).unwrap();
    let mut incremental = FreeList::new();
    for extent in extents(input) {
        incremental.free(extent);
    }
    assert_list(&bulk, expected, total);
    assert_eq!(bulk.extents(), incremental.extents());
    assert_eq!(bulk.total_bytes(), incremental.total_bytes());
}

#[rstest]
#[case::reversed(&[(40, 10), (10, 20)], FreeListError::Unsorted {
    previous: extent(40, 10), next: extent(10, 20),
})]
#[case::reversed_after_coalescing(&[(10, 10), (20, 10), (15, 2)], FreeListError::Unsorted {
    previous: extent(20, 10), next: extent(15, 2),
})]
#[case::partial_overlap(&[(10, 20), (20, 20)], FreeListError::Overlap {
    previous: extent(10, 20), next: extent(20, 20),
})]
#[case::contained(&[(10, 20), (20, 2)], FreeListError::Overlap {
    previous: extent(10, 20), next: extent(20, 2),
})]
#[case::duplicate(&[(10, 20), (10, 20)], FreeListError::Overlap {
    previous: extent(10, 20), next: extent(10, 20),
})]
#[case::overlap_after_coalescing(&[(10, 10), (20, 10), (25, 10)], FreeListError::Overlap {
    previous: extent(20, 10), next: extent(25, 10),
})]
#[case::overlap_at_maximum(&[(0, u64::MAX - 1), (u64::MAX - 2, 2)], FreeListError::Overlap {
    previous: extent(0, u64::MAX - 1), next: extent(u64::MAX - 2, 2),
})]
fn sorted_construction_rejects_invalid_sequences(
    #[case] input: &[(u64, u64)],
    #[case] expected: FreeListError,
) {
    assert_eq!(FreeList::from_sorted(extents(input)).unwrap_err(), expected);
}

#[rstest]
#[case::smallest_fit(15, Some((80, 15)), &[(10, 40), (95, 5), (120, 20)], 65)]
#[case::lowest_address_tie(20, Some((80, 20)), &[(10, 40), (120, 20)], 60)]
#[case::coalesced_fit(30, Some((10, 30)), &[(40, 10), (80, 20), (120, 20)], 50)]
#[case::exact(40, Some((10, 40)), &[(80, 20), (120, 20)], 40)]
#[case::too_large(41, None, &[(10, 40), (80, 20), (120, 20)], 80)]
#[case::empty_request(0, None, &[(10, 40), (80, 20), (120, 20)], 80)]
fn bulk_construction_preserves_best_fit_and_remainders(
    #[case] requested: u64,
    #[case] allocation: Option<(u64, u64)>,
    #[case] expected: &[(u64, u64)],
    #[case] total: u64,
) {
    let input = extents(&[(10, 20), (30, 20), (80, 20), (120, 20)]);
    let mut bulk = FreeList::from_sorted(input.iter().copied()).unwrap();
    let mut incremental = FreeList::new();
    for extent in input {
        incremental.free(extent);
    }
    assert_eq!(
        bulk.alloc(requested),
        allocation.map(|(start, len)| extent(start, len))
    );
    assert_list(&bulk, expected, total);
    assert_eq!(
        incremental.alloc(requested),
        allocation.map(|(start, len)| extent(start, len))
    );
    assert_eq!(bulk.extents(), incremental.extents());
    assert_eq!(bulk.total_bytes(), incremental.total_bytes());
}

#[rstest]
#[case::empty(&[], &[(10, 20)], &[], 0)]
#[case::no_removals(&[(10, 20)], &[], &[(10, 20)], 20)]
#[case::disjoint(&[(10, 20)], &[(0, 5), (30, 5)], &[(10, 20)], 20)]
#[case::complete(&[(10, 20)], &[(10, 20)], &[], 0)]
#[case::containing(&[(10, 20), (40, 10)], &[(0, 60)], &[], 0)]
#[case::partial_edges(&[(10, 20)], &[(0, 15), (25, 15)], &[(15, 10)], 10)]
#[case::several_splits(&[(10, 100)], &[(20, 10), (40, 10), (60, 10)], &[(10, 10), (30, 10), (50, 10), (70, 40)], 70)]
#[case::across_regions(&[(10, 20), (40, 20), (70, 20)], &[(20, 60)], &[(10, 10), (80, 10)], 20)]
#[case::interleaved(&[(10, 20), (40, 20), (70, 20)], &[(0, 20), (25, 20), (50, 25), (85, 20)], &[(20, 5), (45, 5), (75, 10)], 20)]
#[case::adjacent_removals(&[(0, 100)], &[(20, 10), (30, 10)], &[(0, 20), (40, 60)], 80)]
#[case::maximum_span(&[(0, u64::MAX)], &[(1, u64::MAX - 2)], &[(0, 1), (u64::MAX - 1, 1)], 2)]
fn batch_subtraction_matches_individual_removals(
    #[case] initial: &[(u64, u64)],
    #[case] removed: &[(u64, u64)],
    #[case] expected: &[(u64, u64)],
    #[case] total: u64,
) {
    let mut bulk = FreeList::from_sorted(extents(initial)).unwrap();
    let mut incremental = bulk.clone();
    let removed = FreeList::from_sorted(extents(removed)).unwrap();
    bulk.take_ranges(&removed);
    for &extent in removed.extents() {
        incremental.take_range(extent);
    }
    assert_list(&bulk, expected, total);
    assert_eq!(bulk.extents(), incremental.extents());
    assert_eq!(bulk.total_bytes(), incremental.total_bytes());
}

#[rstest]
#[case::empty(&[], &[(10, 20)], &[], 0)]
#[case::empty_other(&[(10, 20)], &[], &[], 0)]
#[case::equal(&[(10, 20), (40, 10)], &[(10, 20), (40, 10)], &[(10, 20), (40, 10)], 30)]
#[case::disjoint(&[(10, 20)], &[(0, 10), (30, 10)], &[], 0)]
#[case::inside(&[(10, 100)], &[(20, 10), (40, 10)], &[(20, 10), (40, 10)], 20)]
#[case::across_regions(&[(10, 20), (40, 20), (70, 20)], &[(20, 60)], &[(20, 10), (40, 20), (70, 10)], 40)]
#[case::interleaved(&[(10, 20), (40, 20), (70, 20)], &[(0, 20), (25, 20), (50, 25), (85, 20)], &[(10, 10), (25, 5), (40, 5), (50, 10), (70, 5), (85, 5)], 40)]
#[case::maximum_span(&[(0, u64::MAX)], &[(1, u64::MAX - 2)], &[(1, u64::MAX - 2)], u64::MAX - 2)]
fn canonical_intersection_preserves_exact_coverage(
    #[case] left: &[(u64, u64)],
    #[case] right: &[(u64, u64)],
    #[case] expected: &[(u64, u64)],
    #[case] total: u64,
) {
    let left = FreeList::from_sorted(extents(left)).unwrap();
    let right = FreeList::from_sorted(extents(right)).unwrap();
    let intersection = left.intersection(&right);
    assert_list(&intersection, expected, total);
    assert_eq!(intersection.extents(), right.intersection(&left).extents());
    assert_eq!(
        intersection.total_bytes(),
        right
            .extents()
            .iter()
            .map(|&extent| left.covered_len(extent))
            .sum::<u64>()
    );
    let mut difference = left.clone();
    difference.take_ranges(&right);
    assert_eq!(
        difference.total_bytes().checked_add(total),
        Some(left.total_bytes())
    );
}

fn assert_list(list: &FreeList, expected: &[(u64, u64)], total: u64) {
    assert_eq!(list.extents(), extents(expected));
    assert_eq!(list.total_bytes(), total);
    assert_eq!(
        u128::from(list.total_bytes()),
        list.extents()
            .iter()
            .map(|extent| u128::from(extent.len()))
            .sum::<u128>()
    );
}

fn extents(ranges: &[(u64, u64)]) -> Vec<Extent> {
    ranges
        .iter()
        .map(|&(start, len)| extent(start, len))
        .collect()
}

fn extent(start: u64, len: u64) -> Extent {
    Extent::new(StoredAddress::new(start), len).unwrap()
}
