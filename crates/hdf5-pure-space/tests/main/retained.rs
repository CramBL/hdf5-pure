use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::PageClassEvidence;
use hdf5_pure_space::__private::PageTransition;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::RetainedKind;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;

#[rstest]
#[case::disabled(FileSpaceStrategy::None, 0)]
#[case::aggregate(FileSpaceStrategy::Aggr, 0)]
#[case::flat(FileSpaceStrategy::FsmAggr, 16)]
#[case::paged(FileSpaceStrategy::Page, 16)]
fn retained_view_excludes_reusable_and_disabled_extents(
    #[case] strategy: FileSpaceStrategy,
    #[case] reusable: u64,
) {
    let space = SessionSpace::from_persisted(
        Some(strategy),
        1,
        FileSpacePageSize::MIN,
        [(Some(ManagerKind::PagedSmallMetadata), extent(520, 16))],
    )
    .unwrap();
    assert_eq!(space.retained_extents().count(), 0);
    assert_eq!(space.reusable_bytes().unwrap(), reusable);
}

#[test]
fn retained_view_preserves_exact_unclassified_dead_and_padding_ranges() {
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::MIN,
        [(None, extent(8, 16))],
    )
    .unwrap();
    space.record_freed(StoredAddress::new(4_096), extent(520, 16), FreeClass::Dead);
    space.record_page_padding(PageTransition::Pad {
        extent: extent(2_056, 504),
        track_as: Some(PageType::Meta),
    });
    space.record_page_padding(PageTransition::Pad {
        extent: extent(1_544, 504),
        track_as: Some(PageType::Raw),
    });
    let expected = [
        (extent(8, 16), RetainedKind::Unclassified),
        (extent(520, 16), RetainedKind::Dead),
        (extent(1_544, 504), RetainedKind::Padding(PageType::Raw)),
        (extent(2_056, 504), RetainedKind::Padding(PageType::Meta)),
    ];
    let mut observed = space
        .retained_extents()
        .map(|entry| (entry.extent(), entry.kind()))
        .collect::<Vec<_>>();
    observed.sort_unstable_by_key(|(extent, _)| extent.start());
    assert_eq!(observed, expected);
    assert_eq!(
        space
            .retained_extents()
            .map(|entry| entry.extent().len())
            .sum::<u64>(),
        1_040
    );
    assert_eq!(space.reusable_bytes().unwrap(), 0);
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        [(extent(8, 16), FreeClass::Dead)]
    );
    assert_eq!(
        space
            .admitted_snapshot()
            .partial_page_class(extent(2_048, 8)),
        PageClassEvidence::Unknown
    );
    assert_eq!(space.allocate(1, AllocationClass::Metadata), None);
}

#[rstest]
#[case::power_of_two(512)]
#[case::non_power_of_two(768)]
fn retained_view_preserves_only_generic_partial_edges(#[case] page: u64) {
    let size = FileSpacePageSize::try_from(page).unwrap();
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        size,
        [(
            Some(ManagerKind::PagedLargeGeneric),
            extent(page + 8, 4 * page - 16),
        )],
    )
    .unwrap();
    let mut retained = space
        .retained_extents()
        .map(|entry| (entry.extent(), entry.kind()))
        .collect::<Vec<_>>();
    retained.sort_unstable_by_key(|(extent, _)| extent.start());
    assert_eq!(
        retained,
        [
            (extent(page + 8, page - 8), RetainedKind::Unclassified),
            (extent(4 * page, page - 8), RetainedKind::Unclassified),
        ]
    );
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        [extent(2 * page, 2 * page)]
    );
    assert_eq!(
        retained.iter().map(|(extent, _)| extent.len()).sum::<u64>()
            + space.reusable_bytes().unwrap(),
        4 * page - 16
    );
}

#[test]
fn retained_view_follows_promotion_and_complete_allocator_restoration() {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::MIN);
    space.record_freed(StoredAddress::new(4_096), extent(512, 256), FreeClass::Dead);
    let prior = space.append_snapshot();
    assert_eq!(
        space
            .retained_extents()
            .map(|entry| (entry.extent(), entry.kind()))
            .collect::<Vec<_>>(),
        [(extent(512, 256), RetainedKind::Dead)]
    );
    assert_eq!(space.reusable_bytes().unwrap(), 0);
    space.record_freed(
        StoredAddress::new(4_096),
        extent(768, 256),
        FreeClass::Page(PageType::Meta),
    );
    assert_eq!(space.retained_extents().count(), 0);
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        [extent(512, 512)]
    );
    space.restore_append(prior);
    assert_eq!(
        space
            .retained_extents()
            .map(|entry| (entry.extent(), entry.kind()))
            .collect::<Vec<_>>(),
        [(extent(512, 256), RetainedKind::Dead)]
    );
    assert_eq!(space.reusable_bytes().unwrap(), 0);
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), length).unwrap()
}
