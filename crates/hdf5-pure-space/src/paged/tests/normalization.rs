use alloc::boxed::Box;

use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

use super::*;

#[rstest]
#[case::mixed_page(
    &[(PAGE, 1_024, META), (PAGE + 1_024, 1_024, RAW),
      (PAGE + 2_048, 2_048, FreeClass::Dead)],
    &[],
    [vec![], vec![(PAGE, PAGE)], vec![]]
)]
#[case::classified_edges(
    &[(PAGE + 100, PAGE, META), (2 * PAGE + 100, PAGE, RAW),
      (3 * PAGE + 100, PAGE + 100, FreeClass::Dead)],
    &[(PAGE, 100), (4 * PAGE + 200, PAGE - 200)],
    [vec![(PAGE + 100, PAGE - 100)], vec![(2 * PAGE, 2 * PAGE)],
     vec![(4 * PAGE, 200)]]
)]
#[case::one_raw_page(&[(PAGE, PAGE, RAW)], &[], [vec![], vec![(PAGE, PAGE)], vec![]])]
#[case::billion_raw_pages(
    &[(PAGE, PAGE * 1_000_000_000, RAW)], &[],
    [vec![], vec![(PAGE, PAGE * 1_000_000_000)], vec![]]
)]
#[case::unclassified_does_not_complete_page(
    &[(PAGE, PAGE - 100, META)], &[(2 * PAGE - 100, 100)],
    [vec![(PAGE, PAGE - 100)], vec![], vec![]]
)]
#[case::disjoint_promotions(
    &[(PAGE, PAGE / 2, META), (PAGE + PAGE / 2, PAGE / 2, RAW),
      (3 * PAGE, PAGE / 2, FreeClass::Dead), (3 * PAGE + PAGE / 2, PAGE / 2, META)],
    &[], [vec![], vec![(PAGE, PAGE), (3 * PAGE, PAGE)], vec![]]
)]
#[case::address_limit(
    &[(u64::MAX - 2 * PAGE + 1, PAGE / 2, META),
      (u64::MAX - PAGE - PAGE / 2 + 1, PAGE / 2, FreeClass::Dead),
      (u64::MAX - PAGE + 1, PAGE - 1, META)],
    &[], [vec![(u64::MAX - PAGE + 1, PAGE - 1)],
          vec![(u64::MAX - 2 * PAGE + 1, PAGE)], vec![]]
)]
fn normalization_preserves_classified_edges_and_promotes_maximal_interiors(
    #[case] tracked: &[(u64, u64, FreeClass)],
    #[case] unclassified: &[(u64, u64)],
    #[case] expected: [Vec<(u64, u64)>; 3],
) {
    let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
    for &(start, len, class) in tracked {
        paged.track(extent(start, len), class);
    }
    for &(start, len) in unclassified {
        paged.seed(extent(start, len), None);
    }
    let mut reference = paged.clone();
    normalize_reference(
        &mut reference.meta,
        &mut reference.raw,
        &mut reference.dead,
        PAGE,
    );
    paged.finish_seed();
    assert_eq!(
        [&paged.meta, &paged.raw, &paged.dead].map(FreeList::sections),
        expected
    );
    assert_eq!(paged.unclassified.sections(), unclassified);
    assert_same_state(&paged, &reference);
    let normalized = paged.clone();
    paged.finish_seed();
    assert_same_state(&paged, &normalized);
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/proptest-regressions/paged_normalization.txt"
        )))),
        ..ProptestConfig::with_cases(64)
    })]

    #[test]
    fn fragmented_normalization_matches_previous_allocations(
        fragments in prop::collection::vec((fragment(), 1u64..=8), 0..=96),
        page_size in prop::sample::select(vec![512u64, 1_000, 4_096]),
    ) {
        let (mut paged, _, _) = fragmented_state(&fragments, page_size);
        let mut reference = paged.clone();
        normalize_reference(
            &mut reference.meta,
            &mut reference.raw,
            &mut reference.dead,
            page_size,
        );
        paged.finish_seed();
        assert_same_state(&paged, &reference);
        paged.finish_seed();
        assert_same_state(&paged, &reference);
        compare_allocations(paged, reference);
    }

    #[test]
    fn fragmented_post_free_matches_previous_release_and_normalization(
        fragments in prop::collection::vec((fragment(), 1u64..=8), 0..=96),
        page_size in prop::sample::select(vec![512u64, 1_000, 4_096]),
        threshold_quarters in 1u64..=12,
    ) {
        let (mut paged, freed, old_blocks) = fragmented_state(&fragments, page_size);
        paged.finish_seed();
        let quarters: u64 = fragments.iter().map(|&(_, quarters)| quarters).sum();
        let eoa = StoredAddress::new(page_size + quarters * page_size / 4);
        let threshold = threshold_quarters * page_size / 4;
        let (post, actual_eoa) = paged.post_free(threshold, eoa, freed.clone(), old_blocks.clone());
        let mut reference = PagedPostFree {
            page_size: paged.page_size,
            meta: paged.meta.clone(),
            raw: paged.raw.clone(),
            dead: paged.dead.clone(),
            unclassified: paged.unclassified.clone(),
        };
        let expected_eoa = release_reference(
            &mut reference,
            threshold,
            eoa,
            freed
                .into_iter()
                .chain(old_blocks.into_iter().map(|extent| (extent, META)))
                .collect(),
        );
        normalize_reference(
            &mut reference.meta,
            &mut reference.raw,
            &mut reference.dead,
            page_size,
        );
        assert_eq!(actual_eoa, expected_eoa);
        assert_eq!(paged_sections(&post), paged_sections(&reference));
        assert_eq!(
            post.unclassified.extents(),
            reference.unclassified.extents()
        );
        let mut expected = paged.clone();
        paged.adopt_post_free(post, false, None);
        expected.adopt_post_free(reference, false, None);
        assert_same_state(&paged, &expected);
        compare_allocations(paged, expected);
    }
}

#[rstest]
#[case::below_page(1, 4 * PAGE, [vec![], vec![(PAGE, 3 * PAGE)], vec![]])]
#[case::equal_to_page(PAGE, 4 * PAGE, [vec![], vec![(PAGE, 3 * PAGE)], vec![]])]
#[case::above_page(3 * PAGE, PAGE, [vec![], vec![], vec![]])]
fn adjacent_completed_pages_retain_individual_threshold_admission(
    #[case] threshold: u64,
    #[case] expected_eoa: u64,
    #[case] expected: [Vec<(u64, u64)>; 3],
) {
    let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
    paged.track(extent(PAGE, PAGE - 100), META);
    paged.track(extent(2 * PAGE, PAGE - 100), RAW);
    paged.track(extent(3 * PAGE, PAGE - 100), FreeClass::Dead);
    let freed = [
        (extent(2 * PAGE - 100, 100), RAW),
        (extent(3 * PAGE - 100, 100), FreeClass::Dead),
        (extent(4 * PAGE - 100, 100), META),
    ];
    let (post, eoa) = paged.post_free(threshold, StoredAddress::new(4 * PAGE), freed, []);
    assert_eq!(eoa, StoredAddress::new(expected_eoa));
    assert_eq!(paged_sections(&post), expected);
}

#[test]
fn completed_page_sweep_deduplicates_candidates_and_excludes_gaps_and_unclassified_bytes() {
    let mut post = paged_free_space(&[
        vec![(PAGE, PAGE / 2), (3 * PAGE, PAGE / 2)],
        vec![(PAGE + PAGE / 2, PAGE / 4), (3 * PAGE + PAGE / 2, PAGE / 2)],
        vec![(PAGE + 3 * PAGE / 4, PAGE / 4)],
    ]);
    post.unclassified.free(extent(2 * PAGE, PAGE));
    let candidates = [
        (extent(3 * PAGE, 100), META),
        (extent(PAGE + 100, 100), META),
        (extent(PAGE + 200, 100), RAW),
        (extent(2 * PAGE + 100, 100), RAW),
        (extent(4 * PAGE, 100), RAW),
    ];
    assert_eq!(
        post.pages_completed_by_merges(&candidates, &[]),
        [extent(PAGE, PAGE), extent(3 * PAGE, PAGE)].into()
    );
    assert_eq!(
        post.pages_completed_by_merges(&candidates, &candidates[..1]),
        [extent(PAGE, PAGE)].into()
    );
}

#[rstest]
#[case::last_complete(
    u64::MAX - 2 * PAGE + 1, PAGE,
    vec![extent(u64::MAX - 2 * PAGE + 1, PAGE)]
)]
#[case::last_partial(u64::MAX - PAGE + 1, PAGE - 1, vec![])]
fn completed_page_sweep_uses_checked_boundaries_at_the_address_limit(
    #[case] start: u64,
    #[case] len: u64,
    #[case] expected: Vec<Extent>,
) {
    let post = paged_free_space(&[vec![(start, len)], vec![], vec![]]);
    let completed = post.pages_completed_by_merges(&[(extent(start + 100, 100), META)], &[]);
    assert_eq!(completed.into_iter().collect::<Vec<_>>(), expected);
}

#[derive(Clone, Copy, Debug)]
enum Fragment {
    Allocated,
    Freed(FreeClass),
    OldManager,
    Tracked(Option<FreeClass>),
}

fn fragment() -> impl Strategy<Value = Fragment> {
    prop_oneof![
        Just(Fragment::Allocated),
        Just(Fragment::Freed(META)),
        Just(Fragment::Freed(RAW)),
        Just(Fragment::Freed(FreeClass::Dead)),
        Just(Fragment::OldManager),
        Just(Fragment::Tracked(None)),
        Just(Fragment::Tracked(Some(META))),
        Just(Fragment::Tracked(Some(RAW))),
        Just(Fragment::Tracked(Some(FreeClass::Dead))),
    ]
}

fn fragmented_state(
    fragments: &[(Fragment, u64)],
    page_size: u64,
) -> (PagedEdit, Vec<(Extent, FreeClass)>, Vec<Extent>) {
    let mut paged = PagedEdit::new(FileSpacePageSize::try_from(page_size).unwrap());
    let mut freed = Vec::new();
    let mut old_blocks = Vec::new();
    let mut quarter = 0;
    for &(fragment, quarters) in fragments {
        let extent = extent(
            page_size + quarter * page_size / 4,
            quarters * page_size / 4,
        );
        match fragment {
            Fragment::Allocated => {}
            Fragment::Freed(class) => freed.push((extent, class)),
            Fragment::OldManager => old_blocks.push(extent),
            Fragment::Tracked(None) => paged.seed(extent, None),
            Fragment::Tracked(Some(class)) => paged.track(extent, class),
        }
        quarter += quarters;
    }
    (paged, freed, old_blocks)
}

fn assert_same_state(actual: &PagedEdit, expected: &PagedEdit) {
    let lists = |paged: &PagedEdit| {
        [&paged.meta, &paged.raw, &paged.dead, &paged.unclassified]
            .map(|list| (list.extents().to_vec(), list.total_bytes()))
    };
    assert_eq!(lists(actual), lists(expected));
    assert_eq!(
        actual.reusable_bytes().unwrap(),
        expected.reusable_bytes().unwrap()
    );
    let mut all = Vec::new();
    for list in [
        &actual.meta,
        &actual.raw,
        &actual.dead,
        &actual.unclassified,
    ] {
        assert_eq!(
            u128::from(list.total_bytes()),
            list.extents()
                .iter()
                .map(|extent| u128::from(extent.len()))
                .sum::<u128>()
        );
        assert!(
            list.extents()
                .windows(2)
                .all(|pair| pair[0].end() < pair[1].start())
        );
        all.extend_from_slice(list.extents());
    }
    all.sort_unstable_by_key(|extent| extent.start());
    assert!(all.windows(2).all(|pair| pair[0].end() <= pair[1].start()));
}

fn compare_allocations(mut actual: PagedEdit, mut expected: PagedEdit) {
    let page = actual.page_size.get();
    for (len, ty) in [
        (1, PageType::Meta),
        (page / 2, PageType::Raw),
        (page + 1, PageType::Meta),
        (page, PageType::Raw),
        (3 * page, PageType::Meta),
    ] {
        assert_eq!(actual.largest(ty), expected.largest(ty));
        assert_eq!(actual.allocate(len, ty), expected.allocate(len, ty));
        assert_same_state(&actual, &expected);
    }
}

fn normalize_reference(meta: &mut FreeList, raw: &mut FreeList, dead: &mut FreeList, page: u64) {
    let mut all = meta.extents().to_vec();
    all.extend_from_slice(raw.extents());
    all.extend_from_slice(dead.extents());
    all.sort_unstable_by_key(|extent| extent.start());
    let mut runs: Vec<Extent> = Vec::new();
    for extent in all {
        match runs.last_mut() {
            Some(run) if run.end() >= extent.start() => {
                *run = Extent::between(run.start(), run.end().max(extent.end())).unwrap();
            }
            _ => runs.push(extent),
        }
    }
    for extent in runs {
        if let Some(pages) = extent.aligned_interior(page) {
            meta.take_range(pages);
            raw.take_range(pages);
            dead.take_range(pages);
            raw.free(pages);
        }
    }
}

fn release_reference(
    post: &mut PagedPostFree,
    threshold: u64,
    eoa: StoredAddress,
    extents: Vec<(Extent, FreeClass)>,
) -> StoredAddress {
    let mut dropped = post.release_each(threshold, extents.iter().copied());
    let page_size = post.page_size.get();
    if threshold > page_size {
        let dropped_extents: BTreeSet<_> = dropped.iter().map(|&(extent, _)| extent).collect();
        let pages: BTreeSet<_> = extents
            .iter()
            .map(|&(extent, _)| extent)
            .filter(|extent| {
                (1..page_size).contains(&extent.len()) && !dropped_extents.contains(extent)
            })
            .flat_map(|extent| [extent.start().get(), extent.end().get() - 1])
            .map(|addr| StoredAddress::new(addr / page_size * page_size))
            .filter_map(|page| Extent::new(page, page_size))
            .filter(|&page| {
                [&post.meta, &post.raw, &post.dead]
                    .into_iter()
                    .map(|list| list.covered_len(page))
                    .sum::<u64>()
                    == page_size
            })
            .collect();
        for &page in &pages {
            for list in [&mut post.meta, &mut post.raw, &mut post.dead] {
                list.take_range(page);
            }
        }
        dropped.extend(post.release_each(threshold, pages.into_iter().map(|page| (page, RAW))));
    }
    post.give_back(eoa, dropped)
}
