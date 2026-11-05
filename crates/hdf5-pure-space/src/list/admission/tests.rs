use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use rstest::rstest;

use super::FlatAdmission;
use super::FreeList;
use crate::admission::AdmissionWork;
use crate::admission::Release;
use crate::admission::TrackedSpace;
use crate::extent::Extent;

#[rstest]
#[case::existing_high_seed(&[(24, 4)], &[(0, 8), (8, 8), (16, 8)], &[(0, 28)], &[])]
#[case::existing_low_seed(&[(0, 4)], &[(4, 8), (12, 8), (20, 8)], &[(0, 28)], &[])]
#[case::eligible_high_seed(&[], &[(0, 8), (8, 8), (16, 16)], &[(0, 32)], &[])]
#[case::eligible_low_seed(&[], &[(0, 16), (16, 8), (24, 8)], &[(0, 32)], &[])]
#[case::eligible_middle_seed(&[], &[(0, 8), (8, 16), (24, 8)], &[(0, 32)], &[])]
#[case::existing_middle_seed(&[(8, 4)], &[(0, 8), (12, 8)], &[(0, 20)], &[])]
#[case::seedless(&[], &[(0, 8), (8, 8), (16, 8)], &[], &[(0, 8), (8, 8), (16, 8)])]
#[case::live_gap(&[(25, 4)], &[(0, 8), (8, 8), (16, 8)], &[(25, 4)], &[(0, 8), (8, 8), (16, 8)])]
#[case::multiple_components(&[(0, 4), (100, 4)], &[(4, 8), (30, 8), (38, 8), (104, 8)], &[(0, 12), (100, 12)], &[(30, 8), (38, 8)])]
#[case::reordered(&[], &[(16, 16), (8, 8), (0, 8)], &[(0, 32)], &[])]
#[case::maximum_end(&[], &[(u64::MAX - 24, 8), (u64::MAX - 16, 16)], &[(u64::MAX - 24, 24)], &[])]
fn batch_admission_preserves_seeded_runs_and_original_dropped_allocations(
    #[case] existing: &[(u64, u64)],
    #[case] returned: &[(u64, u64)],
    #[case] admitted: &[(u64, u64)],
    #[case] dropped: &[(u64, u64)],
) {
    let mut free = free_list(existing);
    let mut work = AdmissionWork::default();
    let actual = free.release_batch(ByteLength::new(16), returns(returned), &mut work);
    assert_eq!(free.sections(), admitted);
    assert_eq!(
        free.total_bytes(),
        ByteLength::new(admitted.iter().map(|&(_, len)| len).sum::<u64>())
    );
    assert_eq!(
        actual
            .into_iter()
            .map(|(extent, ())| (extent.start().get(), extent.len().get()))
            .collect::<Vec<_>>(),
        dropped
    );
    assert_eq!(work.extent_visits, existing.len() + returned.len());
    assert_eq!(work.integrated_extents, admitted.len());
}

#[rstest]
#[case::zero(0)]
#[case::one(1)]
#[case::exact(8)]
fn eligible_original_lengths_seed_all_runs(#[case] threshold: u64) {
    let mut free = FreeList::new();
    let dropped = free.release_each(ByteLength::new(threshold), returns(&[(0, 8), (16, 8)]));
    assert!(dropped.is_empty());
    assert_eq!(free.sections(), [(0, 8), (16, 8)]);
    assert_eq!(free.total_bytes(), ByteLength::new(16));
}

#[rstest]
#[case::empty(&[], &[(100, 8)], &[])]
#[case::scalar_dropped(&[(0, 8)], &[(100, 8)], &[(0, 8)])]
#[case::scalar_eligible(&[(0, 16)], &[(0, 16), (100, 8)], &[])]
#[case::scalar_merged(&[(92, 8)], &[(92, 16)], &[])]
fn empty_and_scalar_batches_skip_global_preparation(
    #[case] returned: &[(u64, u64)],
    #[case] expected: &[(u64, u64)],
    #[case] dropped: &[(u64, u64)],
) {
    let mut free = free_list(&[(100, 8)]);
    let mut work = AdmissionWork::default();
    let actual = free.release_batch(ByteLength::new(16), returns(returned), &mut work);
    assert_eq!(free.sections(), expected);
    assert_eq!(
        free.total_bytes(),
        ByteLength::new(expected.iter().map(|&(_, len)| len).sum::<u64>())
    );
    assert_eq!(actual, returns(dropped));
    assert_eq!(work.ordered_visits, 0);
    assert_eq!(work.sort_comparisons, 0);
    assert_eq!(work.extent_visits, 0);
    assert_eq!(work.integrated_extents, 0);
}

#[rstest]
#[case::small(64, 32, false)]
#[case::larger(1_024, 512, false)]
#[case::reordered(1_024, 512, true)]
fn high_end_seed_admission_bounds_ordering_and_complete_integration(
    #[case] count: usize,
    #[case] unrelated: usize,
    #[case] reverse: bool,
) {
    let count_u64 = u64::try_from(count).unwrap();
    let unrelated_u64 = u64::try_from(unrelated).unwrap();
    let seed = count_u64 * 8;
    let existing: Vec<_> = core::iter::once((seed, 8))
        .chain((0..unrelated_u64).map(|index| (seed + 100 + index * 16, 8)))
        .collect();
    let mut returned: Vec<_> = (0..count_u64)
        .map(|index| (extent(index * 8, 8), ()))
        .collect();
    if reverse {
        returned.reverse();
    }
    let mut free = free_list(&existing);
    let mut work = AdmissionWork::default();
    assert!(
        free.release_batch(ByteLength::new(16), returned.clone(), &mut work)
            .is_empty()
    );
    let mut expected = vec![(0, seed + 8)];
    expected.extend_from_slice(&existing[1..]);
    assert_eq!(free.sections(), expected);
    assert_eq!(
        free.total_bytes(),
        ByteLength::new(seed + 8 + unrelated_u64 * 8)
    );
    assert_eq!(work.extent_visits, count + unrelated + 1);
    assert_eq!(work.integrated_extents, unrelated + 1);
    assert!(work.ordered_visits <= 2 * count);
    assert!(work.sort_comparisons <= 4 * count * usize::try_from(count.ilog2()).unwrap());
    assert_eq!(work.sort_comparisons == 0, !reverse);

    let (reference, dropped, attempts) = reference_admission(free_list(&[(seed, 8)]), 16, returned);
    assert_eq!(reference.sections(), [(0, seed + 8)]);
    assert_eq!(reference.total_bytes(), ByteLength::new(seed + 8));
    assert!(dropped.is_empty());
    if !reverse {
        assert_eq!(attempts, count * (count + 1) / 2);
        assert!(attempts > work.extent_visits * 8);
    }
}

#[rstest]
#[case::threshold_zero(0, Some(0), true, false)]
#[case::threshold_one(1, Some(0), true, false)]
#[case::eligible_low(16, Some(0), true, false)]
#[case::tracked_low(16, Some(0), false, false)]
#[case::eligible_middle(16, Some(2_048), true, true)]
#[case::tracked_middle(16, Some(2_048), false, true)]
#[case::eligible_high(16, Some(4_095), true, true)]
#[case::tracked_high(16, Some(4_095), false, true)]
#[case::seedless(16, None, true, true)]
fn flat_runs_buffer_original_returns_only_until_seeded(
    #[case] threshold: u64,
    #[case] seed_index: Option<usize>,
    #[case] seed_returned: bool,
    #[case] preceding_dropped: bool,
) {
    let mut admission = FlatAdmission::default();
    let prefix = if preceding_dropped {
        admission.push(extent(0, 8), false, true);
        admission.finish_run();
        vec![(extent(0, 8), ())]
    } else {
        Vec::new()
    };
    let mut expected_dropped = prefix.clone();
    let mut start = 32;
    let mut seeded_capacity = None;
    for index in 0..4_096 {
        let seed = seed_index == Some(index);
        let len = if seed && seed_returned {
            threshold.max(8)
        } else {
            8
        };
        let returned = !seed || seed_returned;
        let original = extent(start, len);
        let previous_capacity = admission.dropped.capacity();
        admission.push(original, seed, returned);
        if seed {
            assert_eq!(admission.dropped.capacity(), previous_capacity);
        }
        start += len;
        if admission.seeded {
            assert_eq!(admission.dropped, prefix);
            let capacity = *seeded_capacity.get_or_insert(admission.dropped.capacity());
            assert_eq!(admission.dropped.capacity(), capacity);
            if seed_index == Some(0) {
                assert_eq!(capacity, 0);
            }
        } else {
            expected_dropped.push((original, ()));
            assert_eq!(admission.dropped, expected_dropped);
            assert!(admission.dropped.capacity() >= expected_dropped.len());
        }
    }
    admission.finish_run();
    if seed_index.is_some() {
        assert_eq!(admission.admitted, [extent(32, start - 32)]);
        assert_eq!(admission.dropped, prefix);
    } else {
        assert!(admission.admitted.is_empty());
        assert_eq!(admission.dropped, expected_dropped);
        assert!(admission.dropped.capacity() >= 4_097);
    }
}

#[rstest]
#[case::seedless_tail(&[], &[(0, 8), (8, 8), (16, 8)], 24, 0, &[])]
#[case::reversed_tail(&[], &[(16, 8), (8, 8), (0, 8)], 24, 0, &[])]
#[case::live_gap(&[], &[(0, 8), (16, 8)], 24, 16, &[])]
#[case::tracked_tail(&[(24, 8)], &[(0, 8), (8, 8), (16, 8)], 32, 32, &[(0, 32)])]
#[case::eligible_tail(&[], &[(0, 8), (8, 8), (16, 16)], 32, 32, &[(0, 32)])]
fn complete_flat_release_shrinks_only_dropped_tails(
    #[case] existing: &[(u64, u64)],
    #[case] returned: &[(u64, u64)],
    #[case] eoa: u64,
    #[case] expected_eoa: u64,
    #[case] expected_free: &[(u64, u64)],
) {
    let mut free = free_list(existing);
    assert_eq!(
        free.release_all(
            ByteLength::new(16),
            StoredAddress::new(eoa),
            returns(returned).into_iter().map(|(extent, ())| extent)
        ),
        StoredAddress::new(expected_eoa)
    );
    assert_eq!(free.sections(), expected_free);
    assert_eq!(
        free.total_bytes(),
        ByteLength::new(expected_free.iter().map(|&(_, len)| len).sum::<u64>())
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/proptest-regressions/flat_admission.txt"
        )))),
        ..ProptestConfig::with_cases(256)
    })]

    #[test]
    fn flat_batch_matches_independent_repeated_admission(
        pieces in proptest::collection::vec((any::<bool>(), any::<bool>(), 0_u8..4), 0..100),
        reverse in any::<bool>(),
    ) {
        let mut start = 0;
        let mut existing = Vec::new();
        let mut returned = Vec::new();
        for (tracked, eligible, gap) in pieces {
            let len = if eligible { 16 } else { 8 };
            let piece = extent(start, len);
            if tracked {
                existing.push(piece);
            } else {
                returned.push((piece, ()));
            }
            start += len + u64::from(gap);
        }
        if reverse {
            returned.reverse();
        }
        let original = FreeList::from_sorted(existing).unwrap();
        let mut actual = original.clone();
        let mut actual_dropped = actual.release_each(ByteLength::new(16), returned.clone());
        let (reference, mut reference_dropped, _) = reference_admission(original, 16, returned);
        actual_dropped.sort_unstable_by_key(|&(extent, ())| extent.start());
        reference_dropped.sort_unstable_by_key(|&(extent, ())| extent.start());
        prop_assert_eq!(actual.extents(), reference.extents());
        prop_assert_eq!(actual.total_bytes(), reference.total_bytes());
        prop_assert_eq!(actual_dropped, reference_dropped);
    }
}

fn reference_admission(
    mut free: FreeList,
    threshold: u64,
    extents: Vec<(Extent, ())>,
) -> (FreeList, Vec<(Extent, ())>, usize) {
    let (mut pending, eligible): (Vec<_>, Vec<_>) = extents
        .into_iter()
        .partition(|&(extent, ())| extent.len() < ByteLength::new(threshold));
    for (extent, ()) in eligible {
        free.free(extent);
    }
    let mut attempts = 0;
    loop {
        let offered = pending.len();
        pending.retain(|&(extent, ())| {
            attempts += 1;
            free.release(ByteLength::new(threshold), extent, ()) == Release::Drop
        });
        if pending.len() == offered {
            break;
        }
    }
    (free, pending, attempts)
}

fn free_list(regions: &[(u64, u64)]) -> FreeList {
    FreeList::from_sorted(regions.iter().map(|&(start, len)| extent(start, len))).unwrap()
}

fn returns(regions: &[(u64, u64)]) -> Vec<(Extent, ())> {
    regions
        .iter()
        .map(|&(start, len)| (extent(start, len), ()))
        .collect()
}

fn extent(start: u64, len: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(len)).unwrap()
}
