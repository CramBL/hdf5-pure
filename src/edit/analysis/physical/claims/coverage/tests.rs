use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use rstest::rstest;

use super::{PhysicalClaim, PhysicalCoverage};
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::analysis::physical::{CapacityKind, FileStorageKind};
use crate::object_graph::ObjectId;

#[rstest]
#[case::empty(100, &[], &[(0, 100)], None)]
#[case::adjacent(100, &[(10, 20, LIVE), (30, 10, FREE)], &[(0, 10), (40, 60)], None)]
#[case::separate(100, &[(40, 10, FREE), (10, 20, LIVE)], &[(0, 10), (30, 10), (50, 50)], None)]
#[case::equal(100, &[(10, 20, LIVE), (10, 20, HEAP)], &[(0, 10), (30, 70)], Some([0, 1]))]
#[case::nested(100, &[(10, 30, LIVE), (20, 10, HEAP)], &[(0, 10), (40, 60)], Some([0, 1]))]
#[case::crossing(100, &[(10, 20, FREE), (20, 20, LIVE)], &[(0, 10), (40, 60)], Some([0, 1]))]
#[case::parent(100, &[(10, 40, CAPACITY), (10, 10, EXTENSION), (20, 20, MANAGER)], &[(0, 10), (50, 50)], None)]
#[case::equal_parent(100, &[(10, 20, MANAGER), (10, 20, CAPACITY)], &[(0, 10), (30, 70)], None)]
#[case::crossing_parent(100, &[(10, 20, CAPACITY), (20, 20, MANAGER)], &[(0, 10), (40, 60)], Some([0, 1]))]
#[case::inverted_parent(100, &[(10, 20, MANAGER), (20, 10, CAPACITY)], &[(0, 10), (30, 70)], Some([0, 1]))]
#[case::duplicate_manager(100, &[(10, 40, CAPACITY), (20, 10, MANAGER), (20, 10, SECOND_MANAGER)], &[(0, 10), (50, 50)], Some([1, 2]))]
#[case::duplicate_capacity(100, &[(10, 20, CAPACITY), (10, 20, ClaimOrigin::MetadataCapacity(1))], &[(0, 10), (30, 70)], Some([0, 1]))]
#[case::unrelated_parent(100, &[(10, 40, CAPACITY), (20, 10, LIVE)], &[(0, 10), (50, 50)], Some([0, 1]))]
#[case::empty_claim(100, &[(10, 0, LIVE)], &[(0, 100)], None)]
#[case::maximum_adjacent(u64::MAX, &[(u64::MAX - 8, 4, LIVE), (u64::MAX - 4, 4, HEAP)], &[(0, u64::MAX - 8)], None)]
#[case::maximum_overlap(u64::MAX, &[(u64::MAX - 8, 6, LIVE), (u64::MAX - 4, 4, HEAP)], &[(0, u64::MAX - 8)], Some([0, 1]))]
#[case::maximum_empty(u64::MAX, &[(u64::MAX, 0, LIVE)], &[(0, u64::MAX)], None)]
fn coverage_preserves_claims_and_reports_intervals(
    #[case] limit: u64,
    #[case] input: &[(u64, u64, ClaimOrigin)],
    #[case] residuals: &[(u64, u64)],
    #[case] conflict: Option<[usize; 2]>,
) {
    let claims = input.iter().copied().map(claim).collect::<Vec<_>>();
    let original = claims.clone();
    let coverage = PhysicalCoverage::new(&claims, limit, limit).unwrap();
    assert_eq!(tuples(coverage.residuals()), residuals);
    assert_eq!(coverage.conflict(), conflict);
    assert_eq!(coverage.first_beyond_source(), None);
    assert_eq!(coverage.first_beyond_eoa(), None);
    assert_eq!(claims, original);
}

#[rstest]
#[case::source_shorter(100, 110, &[(90, 20, FREE)], &[0], &[], &[(0, 90)])]
#[case::logical_shorter(110, 100, &[(90, 20, CAPACITY)], &[], &[0], &[(0, 90)])]
#[case::outside(100, 100, &[(120, 20, CAPACITY)], &[0], &[0], &[(0, 100)])]
#[case::limit(u64::MAX, u64::MAX - 1, &[(u64::MAX - 1, 1, LIVE)], &[], &[0], &[(0, u64::MAX - 1)])]
#[case::several_limits(100, 90, &[(130, 10, FREE), (100, 10, LIVE), (90, 5, HEAP)], &[1, 0], &[2, 1, 0], &[(0, 90), (95, 5)])]
#[case::empty_outside(100, 100, &[(120, 0, CAPACITY)], &[0], &[0], &[(0, 100)])]
fn coverage_keeps_physical_and_recorded_bounds_separate(
    #[case] physical: u64,
    #[case] logical: u64,
    #[case] input: &[(u64, u64, ClaimOrigin)],
    #[case] beyond_source: &[usize],
    #[case] beyond_eoa: &[usize],
    #[case] residuals: &[(u64, u64)],
) {
    let claims = input.iter().copied().map(claim).collect::<Vec<_>>();
    let coverage = PhysicalCoverage::new(&claims, physical, logical).unwrap();
    assert_eq!(tuples(coverage.residuals()), residuals);
    assert_eq!(
        coverage.first_beyond_source(),
        beyond_source.first().copied()
    );
    assert_eq!(coverage.first_beyond_eoa(), beyond_eoa.first().copied());
    assert_eq!(coverage.conflict(), None);
}

fn claim((start, length, origin): (u64, u64, ClaimOrigin)) -> PhysicalClaim {
    PhysicalClaim {
        range: FileRange::new(PhysicalAddress::new(start), length).unwrap(),
        origin,
    }
}

fn tuples(ranges: &[FileRange]) -> Vec<(u64, u64)> {
    ranges
        .iter()
        .map(|range| (range.start().get(), range.length()))
        .collect()
}

const LIVE: ClaimOrigin = ClaimOrigin::Object {
    object: ObjectId::new(PhysicalAddress::new(10)),
    kind: AllocationKind::Header,
};
const HEAP: ClaimOrigin = ClaimOrigin::Heap(PhysicalAddress::new(10));
const FREE: ClaimOrigin = ClaimOrigin::NonLive {
    index: 0,
    kind: CapacityKind::Reusable(FreeClass::Dead),
};
const CAPACITY: ClaimOrigin = ClaimOrigin::MetadataCapacity(0);
const EXTENSION: ClaimOrigin = ClaimOrigin::File {
    index: 0,
    kind: FileStorageKind::Extension,
};
const MANAGER: ClaimOrigin = ClaimOrigin::File {
    index: 1,
    kind: FileStorageKind::Manager,
};
const SECOND_MANAGER: ClaimOrigin = ClaimOrigin::File {
    index: 2,
    kind: FileStorageKind::Manager,
};
