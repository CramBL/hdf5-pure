use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
use hdf5_pure_space::__private::{
    AllocationClass, Extent, FreeListError, ManagerKind, PageType, SessionSpace,
};
use rstest::rstest;

#[rstest]
#[case::empty(&[])]
#[case::adjacent(&[
    (Some(ManagerKind::PagedSmallMetadata), 4096, 1024),
    (Some(ManagerKind::PagedSmallRaw), 5120, 3072),
    (None, 8192, 64),
])]
#[case::mixed(&[
    (Some(ManagerKind::PagedSmallMetadata), 4112, 32),
    (Some(ManagerKind::PagedSmallMetadata), 4144, 16),
    (Some(ManagerKind::PagedSmallRaw), 8208, 64),
    (Some(ManagerKind::PagedLargeGeneric), 12289, 16384),
    (None, 32768, 4096),
])]
#[case::long_run(&[(Some(ManagerKind::PagedLargeGeneric), 4096, 1u64 << 40)])]
#[case::address_limit(&[
    (Some(ManagerKind::PagedSmallMetadata), u64::MAX - 4095, 1024),
    (Some(ManagerKind::PagedSmallRaw), u64::MAX - 3071, 3071),
])]
fn bulk_recovery_preserves_incremental_state_and_allocations(
    #[case] input: &[(Option<ManagerKind>, u64, u64)],
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(1, 4096, 4097)] threshold: u64,
) {
    let sections = sections(input);
    let mut bulk = SessionSpace::from_persisted(
        Some(strategy),
        threshold,
        FileSpacePageSize::DEFAULT,
        sections.iter().copied(),
    )
    .unwrap();
    let mut incremental =
        SessionSpace::from_strategy(Some(strategy), threshold, FileSpacePageSize::DEFAULT);
    for (kind, extent) in sections {
        incremental.seed_persisted(kind, extent);
    }
    incremental.finish_seed();
    assert_state(&bulk, &incremental);
    for (length, class) in [
        (16, AllocationClass::Metadata),
        (16, AllocationClass::Raw),
        (4096, AllocationClass::Raw),
        (8192, AllocationClass::Metadata),
        (1, AllocationClass::Raw),
    ] {
        assert_eq!(
            bulk.allocate(length, class),
            incremental.allocate(length, class)
        );
        assert_state(&bulk, &incremental);
    }
}

#[test]
fn bulk_page_recovery_preserves_classified_and_unclassified_edges() {
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        8192,
        FileSpacePageSize::DEFAULT,
        sections(&[
            (Some(ManagerKind::PagedSmallMetadata), 4096, 1024),
            (Some(ManagerKind::PagedSmallRaw), 5120, 3072),
            (Some(ManagerKind::PagedLargeGeneric), 8193, 12287),
            (None, 20480, 4096),
            (Some(ManagerKind::Flat), 24576, 64),
            (Some(ManagerKind::PagedSmallMetadata), 28688, 32),
        ]),
    )
    .unwrap();
    let pages = space.paged_sections().unwrap();
    assert_eq!(pages.metadata(), [extent(28688, 32)]);
    assert_eq!(pages.raw(), [extent(4096, 4096), extent(12288, 8192)]);
    assert_eq!(
        pages.unclassified(),
        [extent(8193, 4095), extent(20480, 4160)]
    );
    assert_eq!(space.reusable_bytes().unwrap(), 12320);
    assert_eq!(space.paged_tail_type(), Some(None));
    assert_eq!(space.pending_padding(PageType::Meta), Some([].as_slice()));
    assert_eq!(space.pending_padding(PageType::Raw), Some([].as_slice()));
}

#[rstest]
#[case::unsorted(
    &[(Some(ManagerKind::PagedSmallMetadata), 200, 16), (Some(ManagerKind::PagedSmallRaw), 100, 16)],
    FreeListError::Unsorted { previous: extent(200, 16), next: extent(100, 16) },
)]
#[case::overlap(
    &[(Some(ManagerKind::PagedSmallMetadata), 100, 32), (Some(ManagerKind::PagedSmallRaw), 116, 32)],
    FreeListError::Overlap { previous: extent(100, 32), next: extent(116, 32) },
)]
#[case::unknown_overlap(
    &[(None, 100, 32), (Some(ManagerKind::PagedLargeGeneric), 116, 32)],
    FreeListError::Overlap { previous: extent(100, 32), next: extent(116, 32) },
)]
fn bulk_recovery_rejects_invalid_global_geometry(
    #[case] input: &[(Option<ManagerKind>, u64, u64)],
    #[case] expected: FreeListError,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
) {
    let Err(error) = SessionSpace::from_persisted(
        Some(strategy),
        1,
        FileSpacePageSize::DEFAULT,
        sections(input),
    ) else {
        panic!("expected invalid free-list geometry");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::aggregate(Some(FileSpaceStrategy::Aggr))]
#[case::none(Some(FileSpaceStrategy::None))]
#[case::unreadable(None)]
fn bulk_recovery_keeps_disabled_strategies_empty(#[case] strategy: Option<FileSpaceStrategy>) {
    let mut space = SessionSpace::from_persisted(
        strategy,
        1,
        FileSpacePageSize::DEFAULT,
        sections(&[(None, 100, 32), (None, 116, 32)]),
    )
    .unwrap();
    assert!(!space.is_enabled());
    assert_eq!(space.reusable_bytes().unwrap(), 0);
    assert_eq!(space.reusable_sections().collect::<Vec<_>>(), []);
    assert_eq!(space.allocate(1, AllocationClass::Raw), None);
}

fn assert_state(actual: &SessionSpace, expected: &SessionSpace) {
    assert_eq!(
        actual.reusable_bytes().unwrap(),
        expected.reusable_bytes().unwrap()
    );
    assert_eq!(
        actual.reusable_sections().collect::<Vec<_>>(),
        expected.reusable_sections().collect::<Vec<_>>()
    );
    match (actual.paged_sections(), expected.paged_sections()) {
        (Some(actual), Some(expected)) => {
            assert_eq!(actual.metadata(), expected.metadata());
            assert_eq!(actual.raw(), expected.raw());
            assert_eq!(actual.unclassified(), expected.unclassified());
        }
        (None, None) => {}
        _ => panic!("different allocation strategies"),
    }
}

fn sections(input: &[(Option<ManagerKind>, u64, u64)]) -> Vec<(Option<ManagerKind>, Extent)> {
    input
        .iter()
        .map(|&(kind, start, len)| (kind, extent(start, len)))
        .collect()
}

fn extent(start: u64, len: u64) -> Extent {
    Extent::new(StoredAddress::new(start), len).unwrap()
}
