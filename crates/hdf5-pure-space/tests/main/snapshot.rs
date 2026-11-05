use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
use hdf5_pure_space::__private::PageClassEvidence;
use hdf5_pure_space::__private::{
    AllocationClass, Extent, FreeClass, ManagerKind, ManagerSections, PageTransition, PageType,
    SessionSpace,
};
use rstest::rstest;

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr, None, 224, 0, &[
    (4160, 64, FreeClass::Dead),
    (8256, 32, FreeClass::Dead),
    (12352, 128, FreeClass::Dead),
])]
#[case::page(FileSpaceStrategy::Page, Some(FileSpacePageSize::DEFAULT), 192, 32, &[
    (4160, 64, FreeClass::Page(PageType::Meta)),
    (8256, 32, FreeClass::Dead),
    (12352, 128, FreeClass::Page(PageType::Raw)),
])]
#[case::disabled(FileSpaceStrategy::None, None, 0, 0, &[])]
fn admitted_snapshot_preserves_classes_order_and_cached_totals(
    #[case] strategy: FileSpaceStrategy,
    #[case] page_size: Option<FileSpacePageSize>,
    #[case] reusable: u64,
    #[case] unclassified: u64,
    #[case] expected: &[(u64, u64, FreeClass)],
) {
    let space = SessionSpace::from_persisted(
        Some(strategy),
        ByteLength::new(8192),
        FileSpacePageSize::DEFAULT,
        [
            (Some(ManagerKind::PagedSmallMetadata), extent(4160, 64)),
            (None, extent(8256, 32)),
            (Some(ManagerKind::PagedSmallRaw), extent(12352, 128)),
        ],
    )
    .unwrap();
    let snapshot = space.admitted_snapshot();
    assert_eq!(snapshot.is_enabled(), strategy != FileSpaceStrategy::None);
    assert_eq!(snapshot.page_size(), page_size);
    assert_eq!(
        snapshot.reusable_bytes().unwrap(),
        ByteLength::new(reusable)
    );
    assert_eq!(snapshot.unclassified_bytes(), ByteLength::new(unclassified));
    assert_eq!(
        snapshot.sections().collect::<Vec<_>>(),
        expected
            .iter()
            .map(|&(start, length, class)| (extent(start, length), class))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        snapshot
            .reusable_sections()
            .map(|extent| extent.len().get())
            .sum::<u64>(),
        reusable
    );
}

#[test]
fn admitted_snapshot_excludes_pending_padding_and_seedless_small_returns() {
    let mut space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(64),
        FileSpacePageSize::DEFAULT,
    );
    let admitted = extent(8256, 128);
    space.record_freed(
        StoredAddress::new(16384),
        admitted,
        FreeClass::Page(PageType::Raw),
    );
    space.record_freed(
        StoredAddress::new(16384),
        extent(12352, 32),
        FreeClass::Page(PageType::Meta),
    );
    space.record_transition(PageType::Meta, PageTransition::Unchanged);
    let transition = space
        .plan_transition(StoredAddress::new(16448), PageType::Raw)
        .unwrap()
        .unwrap();
    space.record_transition(PageType::Raw, transition);
    assert_eq!(
        space.pending_padding(PageType::Meta),
        Some([extent(16448, 4032)].as_slice())
    );
    let snapshot = space.admitted_snapshot();
    assert_eq!(snapshot.reusable_bytes().unwrap(), ByteLength::new(128));
    assert_eq!(snapshot.unclassified_bytes(), ByteLength::new(0));
    assert_eq!(
        snapshot.sections().collect::<Vec<_>>(),
        [(admitted, FreeClass::Page(PageType::Raw))]
    );
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr, FreeClass::Dead)]
#[case::page(FileSpaceStrategy::Page, FreeClass::Page(PageType::Raw))]
fn admitted_snapshot_follows_reservation_and_restored_allocator_state(
    #[case] strategy: FileSpaceStrategy,
    #[case] class: FreeClass,
) {
    let mut space = SessionSpace::from_persisted(
        Some(strategy),
        ByteLength::new(1),
        FileSpacePageSize::DEFAULT,
        [(Some(ManagerKind::PagedSmallRaw), extent(4160, 128))],
    )
    .unwrap();
    let prior = space.append_snapshot();
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        [(extent(4160, 128), class)]
    );
    assert_eq!(
        space.allocate(ByteLength::new(32), AllocationClass::Raw),
        Some(extent(4160, 32))
    );
    let snapshot = space.admitted_snapshot();
    assert_eq!(
        snapshot.reusable_sections().collect::<Vec<_>>(),
        [extent(4192, 96)]
    );
    assert_eq!(snapshot.reusable_bytes().unwrap(), ByteLength::new(96));
    assert_eq!(
        snapshot.sections().collect::<Vec<_>>(),
        [(extent(4192, 96), class)]
    );
    space.restore_append(prior);
    let restored = space.admitted_snapshot();
    assert_eq!(
        restored.sections().collect::<Vec<_>>(),
        [(extent(4160, 128), class)]
    );
    assert_eq!(restored.reusable_bytes().unwrap(), ByteLength::new(128));
}

#[test]
fn admitted_snapshot_keeps_long_page_runs_compact() {
    let run = extent(4096, 1 << 40);
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::DEFAULT,
        [(Some(ManagerKind::PagedLargeGeneric), run)],
    )
    .unwrap();
    let snapshot = space.admitted_snapshot();
    assert_eq!(
        snapshot.sections().collect::<Vec<_>>(),
        [(run, FreeClass::Page(PageType::Raw))]
    );
    assert_eq!(snapshot.reusable_bytes().unwrap(), run.len());
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(length)).unwrap()
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr, vec![
    (ManagerKind::Flat, vec![(100, 200), (3740, 4452), (9000, 200), (13000, 300)]),
])]
#[case::page(FileSpaceStrategy::Page, vec![
    (ManagerKind::PagedSmallMetadata, vec![(100, 200), (3740, 356)]),
    (ManagerKind::PagedSmallRaw, vec![(9000, 200)]),
    (ManagerKind::PagedLargeGeneric, vec![(4096, 4096), (13000, 300)]),
])]
#[case::disabled(FileSpaceStrategy::None, vec![])]
fn snapshot_projection_preserves_current_manager_classes(
    #[case] strategy: FileSpaceStrategy,
    #[case] expected: Vec<(ManagerKind, Vec<(u64, u64)>)>,
) {
    let space = SessionSpace::from_persisted(
        Some(strategy),
        ByteLength::new(1),
        FileSpacePageSize::DEFAULT,
        [
            (Some(ManagerKind::PagedSmallMetadata), extent(100, 200)),
            (Some(ManagerKind::PagedSmallMetadata), extent(3740, 4452)),
            (Some(ManagerKind::PagedSmallRaw), extent(9000, 200)),
            (None, extent(13000, 300)),
        ],
    )
    .unwrap();
    let managers = ManagerSections::from_snapshot(space.admitted_snapshot());
    assert_eq!(
        managers
            .iter()
            .map(|manager| (
                manager.kind(),
                manager
                    .sections()
                    .iter()
                    .map(|extent| (extent.start().get(), extent.len().get()))
                    .collect::<Vec<_>>(),
            ))
            .collect::<Vec<_>>(),
        expected,
    );
}

#[test]
fn proposed_page_snapshot_excludes_owned_metadata_capacity() {
    let space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::DEFAULT,
    );
    let tail = space
        .plan_paged_tail(
            StoredAddress::new(8192),
            &[(extent(4096, 4096), FreeClass::Page(PageType::Raw))],
            &[],
            ByteLength::new(128),
            true,
        )
        .unwrap()
        .unwrap();
    assert_eq!(tail.allocation, extent(8192, 128));
    assert_eq!(tail.final_eoa, StoredAddress::new(12288));
    let snapshot = tail.post.admitted_snapshot();
    assert_eq!(snapshot.reusable_bytes().unwrap(), ByteLength::new(8064));
    assert_eq!(
        snapshot.sections().collect::<Vec<_>>(),
        [
            (extent(4096, 4096), FreeClass::Page(PageType::Raw)),
            (extent(8320, 3968), FreeClass::Page(PageType::Meta)),
        ]
    );
    let managers = ManagerSections::from_snapshot(snapshot);
    assert_eq!(managers.len(), 2);
    assert_eq!(managers[0].kind(), ManagerKind::PagedSmallMetadata);
    assert_eq!(managers[0].sections(), [extent(8320, 3968)]);
    assert_eq!(managers[1].kind(), ManagerKind::PagedLargeGeneric);
    assert_eq!(managers[1].sections(), [extent(4096, 4096)]);
}

#[rstest]
#[case::metadata(&[(Some(ManagerKind::PagedSmallMetadata), 400, 16)], 100, 20, PageClassEvidence::Known(PageType::Meta))]
#[case::raw(&[(Some(ManagerKind::PagedSmallRaw), 400, 16)], 100, 20, PageClassEvidence::Known(PageType::Raw))]
#[case::absent(&[], 100, 20, PageClassEvidence::Unknown)]
#[case::unclassified(&[(None, 400, 16)], 100, 20, PageClassEvidence::Unknown)]
#[case::another_page(&[(Some(ManagerKind::PagedSmallMetadata), 912, 16)], 100, 20, PageClassEvidence::Unknown)]
#[case::following_page(&[(Some(ManagerKind::PagedSmallMetadata), 512, 16)], 480, 32, PageClassEvidence::Unknown)]
#[case::preceding_page(&[(Some(ManagerKind::PagedSmallMetadata), 496, 16)], 512, 16, PageClassEvidence::Unknown)]
#[case::same_page_conflict(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16), (Some(ManagerKind::PagedSmallRaw), 400, 16)], 100, 20, PageClassEvidence::Conflict)]
#[case::both_edges_agree(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16), (Some(ManagerKind::PagedSmallMetadata), 1_424, 16)], 100, 1_224, PageClassEvidence::Known(PageType::Meta))]
#[case::different_edge_classes(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16), (Some(ManagerKind::PagedSmallRaw), 1_424, 16)], 100, 1_224, PageClassEvidence::Conflict)]
#[case::missing_edge(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16)], 100, 1_224, PageClassEvidence::Unknown)]
#[case::missing_first_edge(&[(Some(ManagerKind::PagedSmallMetadata), 1_424, 16)], 100, 1_224, PageClassEvidence::Unknown)]
#[case::crossing_without_a_full_page(&[(Some(ManagerKind::PagedSmallMetadata), 450, 16), (Some(ManagerKind::PagedSmallMetadata), 600, 16)], 480, 64, PageClassEvidence::Known(PageType::Meta))]
#[case::conflict_on_first_edge(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16), (Some(ManagerKind::PagedSmallRaw), 400, 16), (Some(ManagerKind::PagedSmallMetadata), 1_424, 16)], 100, 1_224, PageClassEvidence::Conflict)]
#[case::conflict_on_last_edge(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16), (Some(ManagerKind::PagedSmallMetadata), 1_424, 16), (Some(ManagerKind::PagedSmallRaw), 1_456, 16)], 100, 1_224, PageClassEvidence::Conflict)]
#[case::typed_and_unclassified_fragments(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16), (None, 400, 16)], 100, 20, PageClassEvidence::Known(PageType::Meta))]
#[case::whole_free_pages(&[(Some(ManagerKind::PagedLargeGeneric), 512, 1_024)], 700, 16, PageClassEvidence::Unknown)]
#[case::interior_of_typed_run(&[(Some(ManagerKind::PagedSmallMetadata), 100, 1_224)], 700, 16, PageClassEvidence::Unknown)]
#[case::edge_of_typed_run(&[(Some(ManagerKind::PagedSmallMetadata), 100, 1_224)], 20, 16, PageClassEvidence::Known(PageType::Meta))]
#[case::no_partial_edges(&[(Some(ManagerKind::PagedSmallMetadata), 50, 16)], 512, 1_024, PageClassEvidence::Unknown)]
fn partial_page_classes_require_consistent_typed_fragments(
    #[case] sections: &[(Option<ManagerKind>, u64, u64)],
    #[case] start: u64,
    #[case] length: u64,
    #[case] expected: PageClassEvidence,
    #[values(512, 513, 4_096)] page: u64,
    #[values(0, 1_u64 << 32)] page_origin: u64,
) {
    let coordinate = |value| page_origin * page + value / 512 * page + value % 512;
    let allocation = |start, length| {
        extent(
            coordinate(start),
            coordinate(start + length) - coordinate(start),
        )
    };
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::try_from(page).unwrap(),
        sections
            .iter()
            .map(|&(kind, start, length)| (kind, allocation(start, length))),
    )
    .unwrap();
    let snapshot = space.admitted_snapshot();
    let before = snapshot.sections().collect::<Vec<_>>();
    let totals = (
        snapshot.reusable_bytes().unwrap(),
        snapshot.unclassified_bytes(),
    );
    assert_eq!(
        snapshot.partial_page_class(allocation(start, length)),
        expected
    );
    assert_eq!(snapshot.sections().collect::<Vec<_>>(), before);
    assert_eq!(
        (
            snapshot.reusable_bytes().unwrap(),
            snapshot.unclassified_bytes()
        ),
        totals
    );
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::disabled(FileSpaceStrategy::None)]
fn a_snapshot_without_page_geometry_cannot_establish_a_page_class(
    #[case] strategy: FileSpaceStrategy,
) {
    let space = SessionSpace::from_persisted(
        Some(strategy),
        ByteLength::new(1),
        FileSpacePageSize::MIN,
        [(Some(ManagerKind::PagedSmallMetadata), extent(400, 16))],
    )
    .unwrap();
    assert_eq!(
        space
            .admitted_snapshot()
            .partial_page_class(extent(100, 20)),
        PageClassEvidence::Unknown
    );
}

#[test]
fn a_page_class_proves_no_ownership_or_availability_outside_its_fragment() {
    let seed = extent(400, 16);
    let query = extent(100, 20);
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::MIN,
        [(Some(ManagerKind::PagedSmallMetadata), seed)],
    )
    .unwrap();
    let before = space.append_snapshot();
    assert_eq!(
        space.admitted_snapshot().partial_page_class(query),
        PageClassEvidence::Known(PageType::Meta)
    );
    assert_eq!(space.allocate(query.len(), AllocationClass::Metadata), None);
    assert_eq!(
        space.allocate(seed.len(), AllocationClass::Metadata),
        Some(seed)
    );
    assert_eq!(
        space.admitted_snapshot().partial_page_class(query),
        PageClassEvidence::Unknown
    );
    space.restore_append(before);
    assert_eq!(
        space.admitted_snapshot().partial_page_class(query),
        PageClassEvidence::Known(PageType::Meta)
    );
}
