use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use rstest::rstest;

use super::{PhysicalClaim, PhysicalCoverage};
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::analysis::physical::{CapacityKind, FileStorageKind};
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::{FileBuilder, UserblockSize};

#[rstest]
#[case::empty(&[], &[], &[(0, 100)], None)]
#[case::adjacent(&[(10, 20, LIVE), (30, 10, FREE)], &[(10, 30)], &[(0, 10), (40, 60)], None)]
#[case::separate(&[(40, 10, FREE), (10, 20, LIVE)], &[(10, 20), (40, 10)], &[(0, 10), (30, 10), (50, 50)], None)]
#[case::equal(&[(10, 20, LIVE), (10, 20, HEAP)], &[(10, 20)], &[(0, 10), (30, 70)], Some([0, 1]))]
#[case::nested(&[(10, 30, LIVE), (20, 10, HEAP)], &[(10, 30)], &[(0, 10), (40, 60)], Some([0, 1]))]
#[case::crossing(&[(10, 20, FREE), (20, 20, LIVE)], &[(10, 30)], &[(0, 10), (40, 60)], Some([0, 1]))]
#[case::parent(&[(10, 40, CAPACITY), (10, 10, EXTENSION), (20, 20, MANAGER)], &[(10, 40)], &[(0, 10), (50, 50)], None)]
#[case::equal_parent(&[(10, 20, MANAGER), (10, 20, CAPACITY)], &[(10, 20)], &[(0, 10), (30, 70)], None)]
#[case::crossing_parent(&[(10, 20, CAPACITY), (20, 20, MANAGER)], &[(10, 30)], &[(0, 10), (40, 60)], Some([0, 1]))]
#[case::inverted_parent(&[(10, 20, MANAGER), (20, 10, CAPACITY)], &[(10, 20)], &[(0, 10), (30, 70)], Some([0, 1]))]
#[case::duplicate_manager(&[(10, 40, CAPACITY), (20, 10, MANAGER), (20, 10, SECOND_MANAGER)], &[(10, 40)], &[(0, 10), (50, 50)], Some([1, 2]))]
#[case::duplicate_capacity(&[(10, 20, CAPACITY), (10, 20, ClaimOrigin::MetadataCapacity(1))], &[(10, 20)], &[(0, 10), (30, 70)], Some([0, 1]))]
#[case::unrelated_parent(&[(10, 40, CAPACITY), (20, 10, LIVE)], &[(10, 40)], &[(0, 10), (50, 50)], Some([0, 1]))]
#[case::empty_claim(&[(10, 0, LIVE)], &[], &[(0, 100)], None)]
fn coverage_preserves_claims_and_reports_intervals(
    #[case] input: &[(u64, u64, ClaimOrigin)],
    #[case] covered: &[(u64, u64)],
    #[case] residuals: &[(u64, u64)],
    #[case] conflict: Option<[usize; 2]>,
) {
    let claims = input.iter().copied().map(claim).collect::<Vec<_>>();
    let original = claims.clone();
    let coverage = PhysicalCoverage::new(&claims, 100, 100).unwrap();
    assert_eq!(tuples(coverage.covered()), covered);
    assert_eq!(tuples(coverage.residuals()), residuals);
    assert_eq!(coverage.conflict(), conflict);
    assert!(coverage.beyond_source().is_empty());
    assert!(coverage.beyond_eoa().is_empty());
    assert_eq!(claims, original);
}

#[rstest]
#[case::source_shorter(100, 110, &[(90, 20, FREE)], &[0], &[], &[(0, 90)])]
#[case::logical_shorter(110, 100, &[(90, 20, CAPACITY)], &[], &[0], &[(0, 90)])]
#[case::outside(100, 100, &[(120, 20, CAPACITY)], &[0], &[0], &[(0, 100)])]
#[case::limit(u64::MAX, u64::MAX - 1, &[(u64::MAX - 1, 1, LIVE)], &[], &[0], &[(0, u64::MAX - 1)])]
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
    assert_eq!(
        tuples(coverage.covered()),
        input
            .iter()
            .map(|&(start, len, _)| (start, len))
            .collect::<Vec<_>>()
    );
    assert_eq!(tuples(coverage.residuals()), residuals);
    assert_eq!(coverage.beyond_source(), beyond_source);
    assert_eq!(coverage.beyond_eoa(), beyond_eoa);
    assert_eq!(coverage.conflict(), None);
}

#[rstest]
#[case::plain(0)]
#[case::userblock(512)]
fn cold_open_coverage_retains_unexplained_metadata_spare(#[case] base: u64) {
    let path = test_util::temp::temp_path("physical-coverage-spare.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let claims = phase.physical_claims(&source).unwrap();
    assert!(claims.coverage.is_none());
    assert!(claims.problems().is_empty(), "{:?}", claims.problems());
    let data = claims
        .claims()
        .iter()
        .find(|claim| {
            matches!(
                claim.origin(),
                ClaimOrigin::Object {
                    kind: AllocationKind::Data,
                    ..
                }
            )
        })
        .unwrap()
        .range();
    let metadata_end = claims
        .claims()
        .iter()
        .filter(|claim| {
            matches!(
                claim.origin(),
                ClaimOrigin::Object {
                    kind: AllocationKind::Header,
                    ..
                }
            )
        })
        .map(|claim| claim.range().end().get())
        .max()
        .unwrap();
    let before = calls.load(Ordering::Relaxed);
    let coverage = phase.physical_coverage(&source).unwrap();
    assert_eq!(
        tuples(coverage.residuals()),
        [(metadata_end, data.start().get() - metadata_end)]
    );
    assert_eq!(coverage.conflict(), None);
    assert!(coverage.beyond_source().is_empty());
    assert!(coverage.beyond_eoa().is_empty());
    assert_eq!(coverage.covered().first().unwrap().start().get(), 0);
    let pointer = core::ptr::from_ref(coverage);
    assert_eq!(
        core::ptr::from_ref(phase.physical_coverage(&source).unwrap()),
        pointer
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    let mut foreign = AnalysisPhase::new(&engine);
    let foreign_source = foreign.reconstruction_source(&engine, root).unwrap();
    let error = phase.physical_coverage(&foreign_source).err().unwrap();
    assert!(
        matches!(error, crate::Error::Format(crate::FormatError::Internal(ref message)) if message == "physical claims belong to another source phase"),
        "{error:?}"
    );
    drop(foreign);
    drop(phase);
    drop(engine);
    let reader = crate::File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("data").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
    reader.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::long_page_run(1)]
#[case::overlapping_claims(4_096)]
fn coverage_scratch_and_witnesses_scale_with_claim_count(#[case] count: usize) {
    use test_util::allocation;

    let claims = (0..count)
        .map(|_| claim((0, 4_096_000_000, FREE)))
        .collect::<Vec<_>>();
    let (coverage, measured) = allocation::measure("physical_coverage_scratch", || {
        PhysicalCoverage::new(&claims, 4_096_000_000, 4_096_000_000).unwrap()
    });
    assert_eq!(tuples(coverage.covered()), [(0, 4_096_000_000)]);
    assert_eq!(coverage.residuals(), []);
    assert_eq!(coverage.conflict(), (count > 1).then_some([0, 1]));
    assert!(
        measured.blocks > 0
            && measured.bytes > 0
            && measured.bytes < u64::try_from(count).unwrap() * 32 + 1_024
            && measured.peak_bytes < u64::try_from(count).unwrap() * 32 + 1_024,
        "{measured}"
    );
    let (positive, control) =
        allocation::measure("physical_coverage_control", || vec![0_u8; 1_048_576]);
    assert_eq!(positive.len(), 1_048_576);
    assert!(control.bytes >= 1_048_576);
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
