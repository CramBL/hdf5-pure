use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::StoredAddress;
use rstest::rstest;

use super::META;
use super::PAGE;
use super::RAW;
use super::normalization;
use crate::admission::AdmissionWork;
use crate::admission::TrackedSpace;
use crate::list::FreeList;
use crate::paged::FreeClass;
use crate::paged::PagedEdit;
use crate::paged::PagedPostFree;

#[rstest]
#[case::high_seed(
    [vec![(4 * PAGE, PAGE)], vec![], vec![]], &[],
    &[(2 * PAGE, PAGE, RAW), (3 * PAGE, PAGE, RAW)], 3 * PAGE,
    [vec![(4 * PAGE, PAGE)], vec![(2 * PAGE, 2 * PAGE)], vec![]], &[]
)]
#[case::low_seed(
    [vec![(PAGE, PAGE)], vec![], vec![]], &[],
    &[(2 * PAGE, PAGE, RAW), (3 * PAGE, PAGE, RAW)], 3 * PAGE,
    [vec![(PAGE, PAGE)], vec![(2 * PAGE, 2 * PAGE)], vec![]], &[]
)]
#[case::seedless(
    [vec![], vec![], vec![]], &[],
    &[(2 * PAGE, PAGE, RAW), (3 * PAGE, PAGE, RAW)], 3 * PAGE,
    [vec![], vec![], vec![]], &[(2 * PAGE, PAGE, RAW), (3 * PAGE, PAGE, RAW)]
)]
#[case::within_page(
    [vec![(PAGE + 96, 32)], vec![], vec![]], &[],
    &[(PAGE + 32, 32, RAW), (PAGE + 64, 32, RAW)], 128,
    [vec![(PAGE + 96, 32)], vec![(PAGE + 32, 64)], vec![]], &[]
)]
#[case::page_boundary(
    [vec![(2 * PAGE, 32)], vec![], vec![]], &[],
    &[(2 * PAGE - 64, 32, META), (2 * PAGE - 32, 32, META)], 128,
    [vec![(2 * PAGE, 32)], vec![], vec![]],
    &[(2 * PAGE - 64, 32, META), (2 * PAGE - 32, 32, META)]
)]
#[case::dead_seed(
    [vec![], vec![], vec![(PAGE + 96, 32)]], &[],
    &[(PAGE + 32, 32, META), (PAGE + 64, 32, RAW)], 128,
    [vec![(PAGE + 32, 32)], vec![(PAGE + 64, 32)], vec![(PAGE + 96, 32)]], &[]
)]
#[case::small_returns_dropped_beside_unclassified_space(
    [vec![], vec![], vec![]], &[(PAGE + 64, 16)],
    &[(PAGE, 32, META), (PAGE + 32, 32, RAW)], 128,
    [vec![], vec![], vec![]], &[(PAGE, 32, META), (PAGE + 32, 32, RAW)]
)]
#[case::unclassified_seeds_large_returns(
    [vec![], vec![], vec![]], &[(4 * PAGE, 16)],
    &[(PAGE, PAGE, META), (2 * PAGE, 2 * PAGE, RAW)], 3 * PAGE,
    [vec![(PAGE, PAGE)], vec![(2 * PAGE, 2 * PAGE)], vec![]], &[]
)]
#[case::unaligned_large_returns(
    [vec![(4 * PAGE + 1, PAGE)], vec![], vec![]], &[],
    &[(2 * PAGE + 1, PAGE, RAW), (3 * PAGE + 1, PAGE, RAW)], 3 * PAGE,
    [vec![(4 * PAGE + 1, PAGE)], vec![], vec![]],
    &[(2 * PAGE + 1, PAGE, RAW), (3 * PAGE + 1, PAGE, RAW)]
)]
#[case::far_endpoint_becomes_eligible(
    [vec![(2 * PAGE, PAGE / 2)], vec![], vec![]], &[],
    &[(PAGE, PAGE, RAW), (2 * PAGE + PAGE / 2, PAGE / 2, META)], 3 * PAGE,
    [vec![(2 * PAGE, PAGE)], vec![(PAGE, PAGE)], vec![]], &[]
)]
#[case::eligible_high_seed(
    [vec![], vec![], vec![]], &[],
    &[(PAGE, PAGE, FreeClass::Dead), (2 * PAGE, 2 * PAGE, META)], 2 * PAGE,
    [vec![(2 * PAGE, 2 * PAGE)], vec![], vec![(PAGE, PAGE)]], &[]
)]
#[case::eligible_originals_only(
    [vec![], vec![], vec![]], &[],
    &[(PAGE, 32, META), (2 * PAGE, 32, RAW)], 0,
    [vec![(PAGE, 32)], vec![(2 * PAGE, 32)], vec![]], &[]
)]
fn page_admission_preserves_classes_and_original_dropped_allocations(
    #[case] tracked: [Vec<(u64, u64)>; 3],
    #[case] unclassified: &[(u64, u64)],
    #[case] returned: &[(u64, u64, FreeClass)],
    #[case] threshold: u64,
    #[case] expected: [Vec<(u64, u64)>; 3],
    #[case] dropped: &[(u64, u64, FreeClass)],
) {
    let mut post = super::paged_free_space(&tracked);
    post.unclassified = FreeList::from_sorted(
        unclassified
            .iter()
            .map(|&(start, len)| super::extent(start, len)),
    )
    .unwrap();
    let mut reference = post.clone();
    let returns: Vec<_> = returned
        .iter()
        .map(|&(start, len, class)| (super::extent(start, len), class))
        .collect();
    let actual = post.release_each(threshold, returns.clone());
    let reference_dropped =
        normalization::release_each_reference(&mut reference, threshold, returns, &mut 0);
    assert_eq!(super::paged_sections(&post), expected);
    assert_eq!(
        actual,
        dropped
            .iter()
            .map(|&(start, len, class)| (super::extent(start, len), class))
            .collect::<Vec<_>>()
    );
    assert_eq!(actual, reference_dropped);
    assert_post(&post, &reference);
}

#[rstest]
#[case::small_high_seed(32, 16, false, false)]
#[case::large_high_seed(1_024, 512, false, false)]
#[case::reordered_high_seed(1_024, 512, false, true)]
#[case::large_low_seed(1_024, 512, true, false)]
fn page_admission_bounds_index_queue_and_reconstruction_work(
    #[case] count: usize,
    #[case] unrelated: usize,
    #[case] low_seed: bool,
    #[case] reverse: bool,
) {
    let count_u64 = u64::try_from(count).unwrap();
    let unrelated_u64 = u64::try_from(unrelated).unwrap();
    let first = if low_seed { 2 * PAGE } else { PAGE };
    let seed = if low_seed {
        PAGE
    } else {
        first + count_u64 * PAGE
    };
    let mut tracked = vec![(seed, PAGE)];
    tracked
        .extend((0..unrelated_u64).map(|index| (first + (count_u64 + 3 + 2 * index) * PAGE, PAGE)));
    let mut post = super::paged_free_space(&[vec![], tracked.clone(), vec![]]);
    let mut returned: Vec<_> = (0..count_u64)
        .map(|index| (super::extent(first + index * PAGE, PAGE), RAW))
        .collect();
    if reverse {
        returned.reverse();
    }
    let mut work = AdmissionWork::default();
    assert!(
        post.release_batch(3 * PAGE, returned.clone(), &mut work)
            .is_empty()
    );
    let start = first.min(seed);
    let mut expected = vec![(start, (count_u64 + 1) * PAGE)];
    expected.extend_from_slice(&tracked[1..]);
    assert_eq!(super::paged_sections(&post), [vec![], expected, vec![]]);
    assert_eq!(
        post.raw.total_bytes(),
        (count_u64 + unrelated_u64 + 1) * PAGE
    );
    assert_eq!(work.indexed_extents, unrelated + 1);
    assert_eq!(work.extent_visits, count);
    assert_eq!(
        work.queue_visits,
        if low_seed { count } else { 2 * count - 1 }
    );
    assert_eq!(work.inserted_extents, unrelated + count + 1);
    assert_eq!(work.removed_extents, count);
    assert_eq!(work.integrated_extents, unrelated + 1);
    assert!(work.lookups <= 8 * work.queue_visits + 4 * count);
    assert!(work.ordered_visits <= 2 * count);
    assert!(work.sort_comparisons <= 4 * count * usize::try_from(count.ilog2()).unwrap());
    assert_eq!(work.sort_comparisons == 0, !reverse);
    if !reverse && !low_seed {
        let mut reference = super::paged_free_space(&[vec![], vec![(seed, PAGE)], vec![]]);
        let mut attempts = 0;
        assert!(
            normalization::release_each_reference(
                &mut reference,
                3 * PAGE,
                returned,
                &mut attempts
            )
            .is_empty()
        );
        assert_eq!(reference.raw.sections(), [(start, (count_u64 + 1) * PAGE)]);
        assert_eq!(attempts, count * (count + 1) / 2);
        assert!(attempts > 8 * work.queue_visits);
    }
}

#[rstest]
#[case::below_page(1, 4 * PAGE, [vec![], vec![(PAGE, 3 * PAGE)], vec![]])]
#[case::equal_page(PAGE, 4 * PAGE, [vec![], vec![(PAGE, 3 * PAGE)], vec![]])]
#[case::above_page(3 * PAGE, PAGE, [vec![], vec![], vec![]])]
fn complete_post_free_preserves_individual_completed_pages_and_original_classes(
    #[case] threshold: u64,
    #[case] expected_eoa: u64,
    #[case] expected: [Vec<(u64, u64)>; 3],
) {
    let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
    paged.meta = FreeList::from_sorted([super::extent(PAGE, PAGE - 100)]).unwrap();
    paged.raw = FreeList::from_sorted([super::extent(2 * PAGE, PAGE - 100)]).unwrap();
    paged.dead = FreeList::from_sorted([super::extent(3 * PAGE, PAGE - 100)]).unwrap();
    let freed = [
        (super::extent(4 * PAGE - 100, 100), META),
        (super::extent(2 * PAGE - 100, 100), RAW),
    ];
    let old_blocks = [super::extent(3 * PAGE - 100, 100)];
    let mut reference = super::paged_free_space(&[
        paged.meta.sections(),
        paged.raw.sections(),
        paged.dead.sections(),
    ]);
    let reference_eoa = normalization::release_reference(
        &mut reference,
        threshold,
        StoredAddress::new(4 * PAGE),
        freed
            .into_iter()
            .chain(old_blocks.into_iter().map(|extent| (extent, META)))
            .collect(),
    );
    normalization::normalize_reference(
        &mut reference.meta,
        &mut reference.raw,
        &mut reference.dead,
        PAGE,
    );
    let mut work = AdmissionWork::default();
    let (post, eoa) = paged.post_free_with_work(
        threshold,
        StoredAddress::new(4 * PAGE),
        freed,
        old_blocks,
        &mut work,
    );
    assert_eq!(eoa, StoredAddress::new(expected_eoa));
    assert_eq!(eoa, reference_eoa);
    assert_eq!(super::paged_sections(&post), expected);
    assert_post(&post, &reference);
    assert!(work.indexed_extents <= 6);
    assert!(work.extent_visits <= 6);
    assert!(work.queue_visits <= 18);
    assert!(work.inserted_extents <= 12);
    assert!(work.removed_extents <= 12);
    assert!(work.integrated_extents <= 6);
    assert!(work.lookups <= 8 * work.queue_visits + 24);
}

#[rstest]
#[case::ordered(256, false)]
#[case::reordered(256, true)]
fn allocation_padding_below_the_return_threshold_is_merged_in_bulk(
    #[case] count: u64,
    #[case] reverse: bool,
) {
    let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
    paged.meta =
        FreeList::from_sorted((1..=count).map(|index| super::extent(index * PAGE, 64))).unwrap();
    paged.meta_pad = (1..=count)
        .map(|index| super::extent(index * PAGE + 64, PAGE - 64))
        .collect();
    if reverse {
        paged.meta_pad.reverse();
    }
    let mut work = AdmissionWork::default();
    let eoa = StoredAddress::new((count + 2) * PAGE);
    let (post, returned_eoa) = paged.post_free_with_work(10 * PAGE, eoa, [], [], &mut work);
    assert_eq!(returned_eoa, eoa);
    assert_eq!(
        super::paged_sections(&post),
        [vec![], vec![(PAGE, count * PAGE)], vec![]]
    );
    assert_eq!(post.raw.total_bytes(), count * PAGE);
    assert_eq!(work.extent_visits, 3 * usize::try_from(count).unwrap());
    assert_eq!(work.integrated_extents, 0);
    assert_eq!(work.indexed_extents, 0);
    assert_eq!(work.queue_visits, 0);
    assert_eq!(work.sort_comparisons == 0, !reverse);
    assert_eq!(paged.meta.extents().len(), usize::try_from(count).unwrap());
    assert_eq!(paged.meta_pad.len(), usize::try_from(count).unwrap());
}

#[rstest]
#[case::one_page(1)]
#[case::billion_pages(1_000_000_000)]
fn one_and_billion_page_returns_have_equal_observed_admission_work(#[case] pages: u64) {
    let paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
    let mut work = AdmissionWork::default();
    let (post, eoa) = paged.post_free_with_work(
        1,
        StoredAddress::new((pages + 10) * PAGE),
        [
            (super::extent(PAGE, pages * PAGE), META),
            (super::extent((pages + 3) * PAGE, PAGE), RAW),
        ],
        [],
        &mut work,
    );
    assert_eq!(eoa, StoredAddress::new((pages + 10) * PAGE));
    assert_eq!(
        super::paged_sections(&post),
        [
            vec![],
            vec![(PAGE, pages * PAGE), ((pages + 3) * PAGE, PAGE)],
            vec![]
        ]
    );
    assert_eq!(post.raw.total_bytes(), (pages + 1) * PAGE);
    assert_eq!(work.extent_visits, 2);
    assert_eq!(work.inserted_extents, 2);
    assert_eq!(work.integrated_extents, 2);
    assert_eq!(work.queue_visits, 0);
    assert_eq!(work.lookups, 4);
}

fn assert_post(actual: &PagedPostFree, expected: &PagedPostFree) {
    let PagedPostFree {
        page_size,
        meta,
        raw,
        dead,
        unclassified,
    } = actual;
    let PagedPostFree {
        page_size: expected_size,
        meta: expected_meta,
        raw: expected_raw,
        dead: expected_dead,
        unclassified: expected_unclassified,
    } = expected;
    assert_eq!(
        [meta, raw, dead, unclassified].map(|list| (list.extents(), list.total_bytes())),
        [
            expected_meta,
            expected_raw,
            expected_dead,
            expected_unclassified
        ]
        .map(|list| (list.extents(), list.total_bytes()))
    );
    assert_eq!(page_size, expected_size);
    assert_eq!(
        actual.admitted_snapshot().reusable_bytes().unwrap(),
        expected.admitted_snapshot().reusable_bytes().unwrap()
    );
}

#[rstest]
#[case::empty(&[], &[(PAGE, 64)], &[])]
#[case::scalar_dropped(&[(2 * PAGE, 32, RAW)], &[(PAGE, 64)], &[(2 * PAGE, 32, RAW)])]
#[case::scalar_merged(&[(PAGE + 64, 32, RAW)], &[(PAGE, 96)], &[])]
fn empty_and_scalar_page_batches_skip_indexes(
    #[case] returned: &[(u64, u64, FreeClass)],
    #[case] expected: &[(u64, u64)],
    #[case] dropped: &[(u64, u64, FreeClass)],
) {
    let mut post = super::paged_free_space(&[vec![], vec![(PAGE, 64)], vec![]]);
    let mut work = AdmissionWork::default();
    let actual = post.release_batch(
        128,
        returned
            .iter()
            .map(|&(start, len, class)| (super::extent(start, len), class)),
        &mut work,
    );
    assert_eq!(post.raw.sections(), expected);
    assert_eq!(
        post.raw.total_bytes(),
        expected.iter().map(|&(_, len)| len).sum()
    );
    assert_eq!(
        actual,
        dropped
            .iter()
            .map(|&(start, len, class)| (super::extent(start, len), class))
            .collect::<Vec<_>>()
    );
    assert_eq!(work.indexed_extents, 0);
    assert_eq!(work.inserted_extents, 0);
    assert_eq!(work.removed_extents, 0);
    assert_eq!(work.integrated_extents, 0);
    assert_eq!(work.queue_visits, 0);
    assert_eq!(work.lookups, 0);
}

#[test]
fn dropped_tail_preserves_its_original_partial_head_class() {
    let paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
    let returned = [
        (super::extent(7 * PAGE + 100, 2 * PAGE - 100), META),
        (super::extent(9 * PAGE, PAGE), FreeClass::Dead),
    ];
    let mut work = AdmissionWork::default();
    let (post, eoa) = paged.post_free_with_work(
        3 * PAGE,
        StoredAddress::new(10 * PAGE),
        returned,
        [],
        &mut work,
    );
    assert_eq!(eoa, StoredAddress::new(8 * PAGE));
    assert_eq!(
        super::paged_sections(&post),
        [vec![(7 * PAGE + 100, PAGE - 100)], vec![], vec![]]
    );
    assert_eq!(post.meta.total_bytes(), PAGE - 100);
    assert_eq!(work.queue_visits, 2);
    assert_eq!(work.inserted_extents, 0);
    assert_eq!(work.integrated_extents, 0);
}
