use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
use hdf5_pure_space::__private::{
    AllocationClass, Extent, FreeClass, ManagerKind, PageTransition, PageType, SessionSpace,
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
