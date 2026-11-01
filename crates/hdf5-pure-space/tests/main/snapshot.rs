use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
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
        8192,
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
    assert_eq!(snapshot.reusable_bytes().unwrap(), reusable);
    assert_eq!(snapshot.unclassified_bytes(), unclassified);
    assert_eq!(
        snapshot.sections().collect::<Vec<_>>(),
        expected
            .iter()
            .map(|&(start, length, class)| (extent(start, length), class))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        snapshot.reusable_sections().map(Extent::len).sum::<u64>(),
        reusable
    );
}

#[test]
fn admitted_snapshot_excludes_pending_padding_and_seedless_small_returns() {
    let mut space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::Page),
        64,
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
    assert_eq!(snapshot.reusable_bytes().unwrap(), 128);
    assert_eq!(snapshot.unclassified_bytes(), 0);
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
        1,
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
        space.allocate(32, AllocationClass::Raw),
        Some(extent(4160, 32))
    );
    let snapshot = space.admitted_snapshot();
    assert_eq!(
        snapshot.reusable_sections().collect::<Vec<_>>(),
        [extent(4192, 96)]
    );
    assert_eq!(snapshot.reusable_bytes().unwrap(), 96);
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
    assert_eq!(restored.reusable_bytes().unwrap(), 128);
}

#[test]
fn admitted_snapshot_keeps_long_page_runs_compact() {
    let run = extent(4096, 1 << 40);
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
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
    Extent::new(StoredAddress::new(start), length).unwrap()
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
        1,
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
                    .map(|extent| (extent.start().get(), extent.len()))
                    .collect::<Vec<_>>(),
            ))
            .collect::<Vec<_>>(),
        expected,
    );
}

#[test]
fn proposed_page_snapshot_excludes_owned_metadata_capacity() {
    let space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    let tail = space
        .plan_paged_tail(
            StoredAddress::new(8192),
            &[(extent(4096, 4096), FreeClass::Page(PageType::Raw))],
            &[],
            128,
            true,
        )
        .unwrap()
        .unwrap();
    assert_eq!(tail.allocation, extent(8192, 128));
    assert_eq!(tail.final_eoa, StoredAddress::new(12288));
    let snapshot = tail.post.admitted_snapshot();
    assert_eq!(snapshot.reusable_bytes().unwrap(), 8064);
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
