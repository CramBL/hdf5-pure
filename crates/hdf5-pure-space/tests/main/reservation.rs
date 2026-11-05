use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, FormatError, StoredAddress};
use hdf5_pure_space::__private::{
    AllocationClass, Extent, FreeClass, ManagerKind, PageType, ReservationRequest, ReservationWork,
    SessionSpace,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use rstest::rstest;

#[rstest]
#[case::indexed(8)]
#[case::scalar(1)]
fn batch_preserves_flat_best_fit_ties_and_successful_prefix(#[case] count: usize) {
    let mut space = flat(&[(10, 40), (80, 20), (120, 20)]);
    let mut batch = space.reservation_batch(count);
    let first = batch
        .reserve(ReservationRequest::new(
            NonZeroByteLength::new(15).unwrap(),
            AllocationClass::Raw,
        ))
        .unwrap();
    assert_eq!(first.extent(), extent(80, 15));
    assert_eq!(first.class(), AllocationClass::Raw);
    assert_eq!(first.page_type(), PageType::Raw);
    assert_eq!(batch.reserve(request(41, PageType::Meta)), None);
    let second = batch.reserve(request(20, PageType::Meta)).unwrap();
    assert_eq!(second.extent(), extent(120, 20));
    let third = batch.reserve(request(40, PageType::Raw)).unwrap();
    assert_eq!(third.extent(), extent(10, 40));
    let work = batch.finish();
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        [extent(95, 5)]
    );
    assert_eq!(space.reusable_bytes().unwrap(), ByteLength::new(5));
    assert_eq!(work.indexed_extents, if count > 1 { 3 } else { 0 });
    assert_eq!(work.restored_extents, if count > 1 { 1 } else { 0 });
    let (returned, class, page_type) = first.into_parts();
    space.return_allocation_with_page_type(returned, class, page_type);
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        [extent(80, 20)]
    );
    assert_eq!(second.extent(), extent(120, 20));
    assert_eq!(third.extent(), extent(10, 40));
}

#[rstest]
#[case::metadata(PageType::Meta, AllocationClass::Raw)]
#[case::raw(PageType::Raw, AllocationClass::Metadata)]
fn batch_preserves_effective_page_type_and_claimed_page_remainders(
    #[case] page_type: PageType,
    #[case] class: AllocationClass,
) {
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        64,
        FileSpacePageSize::DEFAULT,
        [(Some(ManagerKind::PagedLargeGeneric), extent(8192, 8192))],
    )
    .unwrap();
    let mut batch = space.reservation_batch(3);
    let first = batch
        .reserve(ReservationRequest::with_page_type(
            NonZeroByteLength::new(17).unwrap(),
            class,
            page_type,
        ))
        .unwrap();
    assert_eq!(first.extent(), extent(8192, 17));
    assert_eq!(first.class(), class);
    assert_eq!(first.page_type(), page_type);
    let second = batch.reserve(request(4079, page_type)).unwrap();
    assert_eq!(second.extent(), extent(8209, 4079));
    drop(batch);
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        [extent(12288, 4096)]
    );
    assert_eq!(space.reusable_bytes().unwrap(), ByteLength::new(4096));
}

#[test]
fn batch_whole_page_best_fit_uses_aligned_interiors_and_original_address_ties() {
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::DEFAULT,
        [
            (Some(ManagerKind::PagedSmallRaw), extent(4097, 12286)),
            (Some(ManagerKind::PagedSmallRaw), extent(20480, 4096)),
        ],
    )
    .unwrap();
    let mut batch = space.reservation_batch(2);
    assert_eq!(
        batch
            .reserve(request(4096, PageType::Meta))
            .unwrap()
            .extent(),
        extent(8192, 4096)
    );
    assert_eq!(
        batch
            .reserve(request(4096, PageType::Meta))
            .unwrap()
            .extent(),
        extent(20480, 4096)
    );
    batch.finish();
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        [
            (extent(4097, 4095), FreeClass::Page(PageType::Raw)),
            (extent(12288, 4095), FreeClass::Page(PageType::Raw)),
        ]
    );
    assert_eq!(space.reusable_bytes().unwrap(), ByteLength::new(8190));
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr, 4096)]
#[case::paged(FileSpaceStrategy::Page, 4096)]
#[case::non_power_of_two_pages(FileSpaceStrategy::Page, 1536)]
fn indexed_batch_matches_scalar_reservations_and_restoration(
    #[case] strategy: FileSpaceStrategy,
    #[case] page: u64,
) {
    let page_size = FileSpacePageSize::try_from(page).unwrap();
    let entries = [
        (
            Some(ManagerKind::PagedSmallMetadata),
            extent(page * 2 + 13, page - 13),
        ),
        (
            Some(ManagerKind::PagedSmallRaw),
            extent(page * 4 + 17, page * 5 - 34),
        ),
        (
            Some(ManagerKind::PagedSmallMetadata),
            extent(page * 12, page * 3),
        ),
        (None, extent(page * 20 + 11, 37)),
    ];
    let mut scalar = SessionSpace::from_persisted(Some(strategy), 32, page_size, entries).unwrap();
    let mut indexed = SessionSpace::from_persisted(Some(strategy), 32, page_size, entries).unwrap();
    let requests = [
        request(19, PageType::Meta),
        request(page + 1, PageType::Raw),
        request(page / 2, PageType::Meta),
        request(page * 2, PageType::Meta),
        request(u64::MAX, PageType::Raw),
        request(1, PageType::Raw),
        request(page * 3, PageType::Raw),
        request(64, PageType::Meta),
    ];
    let mut batch = indexed.reservation_batch(requests.len());
    let mut successes = Vec::new();
    for request in requests {
        let expected = scalar.allocate_with_page_type(
            request.size().length(),
            request.class(),
            request.page_type(),
        );
        let actual = batch.reserve(request);
        assert_eq!(
            actual.as_ref().map(|allocation| allocation.extent()),
            expected
        );
        if let Some(allocation) = actual {
            successes.push(allocation);
        }
    }
    batch.finish();
    assert_eq!(
        indexed.admitted_snapshot().sections().collect::<Vec<_>>(),
        scalar.admitted_snapshot().sections().collect::<Vec<_>>()
    );
    assert_eq!(
        indexed.reusable_bytes().unwrap(),
        scalar.reusable_bytes().unwrap()
    );
    for allocation in successes {
        let (extent, class, page_type) = allocation.into_parts();
        indexed.return_allocation_with_page_type(extent, class, page_type);
        scalar.return_allocation_with_page_type(extent, class, page_type);
    }
    assert_eq!(
        indexed.admitted_snapshot().sections().collect::<Vec<_>>(),
        scalar.admitted_snapshot().sections().collect::<Vec<_>>()
    );
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::paged(FileSpaceStrategy::Page)]
fn failed_later_request_and_batch_drop_preserve_successful_reservations(
    #[case] strategy: FileSpaceStrategy,
) {
    let mut space = SessionSpace::from_persisted(
        Some(strategy),
        1,
        FileSpacePageSize::DEFAULT,
        [
            (Some(ManagerKind::PagedSmallRaw), extent(8192, 100)),
            (Some(ManagerKind::PagedSmallRaw), extent(12288, 200)),
        ],
    )
    .unwrap();
    let mut batch = space.reservation_batch(3);
    let first = batch.reserve(request(60, PageType::Raw)).unwrap();
    assert_eq!(first.extent(), extent(8192, 60));
    assert_eq!(batch.reserve(request(u64::MAX, PageType::Meta)), None);
    let second = batch.reserve(request(40, PageType::Raw)).unwrap();
    assert_eq!(second.extent(), extent(8252, 40));
    drop(batch);
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        [extent(12288, 200)]
    );
    assert_eq!(space.reusable_bytes().unwrap(), ByteLength::new(200));
    assert_eq!(first.extent(), extent(8192, 60));
    assert_eq!(second.extent(), extent(8252, 40));
}

#[rstest]
#[case::empty(&[])]
#[case::contiguous(&[(100, 100)])]
fn contiguous_batches_preserve_the_scalar_fast_path(#[case] initial: &[(u64, u64)]) {
    let mut space = flat(initial);
    let mut batch = space.reservation_batch(100);
    let allocation = batch.reserve(request(10, PageType::Raw));
    assert_eq!(
        allocation.as_ref().map(|allocation| allocation.extent()),
        initial.first().map(|_| extent(100, 10))
    );
    assert_eq!(batch.finish(), ReservationWork::default());
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        if initial.is_empty() {
            Vec::new()
        } else {
            vec![extent(110, 90)]
        }
    );
}

#[test]
fn disabled_batch_preserves_the_empty_fast_path() {
    let mut space = SessionSpace::Disabled;
    let mut batch = space.reservation_batch(10);
    assert_eq!(batch.reserve(request(1, PageType::Raw)), None);
    assert_eq!(batch.finish(), ReservationWork::default());
    assert_eq!(space.reusable_bytes().unwrap(), ByteLength::new(0));
}

#[rstest]
#[case::equal_sizes(128, 8)]
#[case::fragmented(4096, 16)]
fn batch_builds_indexes_once_and_restores_only_remaining_extents(
    #[case] count: u64,
    #[case] size: u64,
) {
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        1,
        FileSpacePageSize::DEFAULT,
        (0..count).map(|i| (None, extent(i * 128, 64))),
    )
    .unwrap();
    let mut batch = space.reservation_batch(128);
    for i in 0..128 {
        let allocation = batch.reserve(request(size, PageType::Raw)).unwrap();
        assert_eq!(
            allocation.extent(),
            extent((i / (64 / size)) * 128 + (i % (64 / size)) * size, size)
        );
    }
    let work = batch.finish();
    assert_eq!(work.indexed_extents, count);
    assert_eq!(work.lookups, 128);
    assert_eq!(work.removed_extents, 128);
    assert_eq!(work.inserted_extents, count + 128 - 128 / (64 / size));
    assert_eq!(work.restored_extents, count - 128 / (64 / size));
    assert_eq!(
        space.reusable_bytes().unwrap(),
        ByteLength::new(count * 64 - 128 * size)
    );
}

#[test]
fn batch_preserves_the_maximum_checked_extent() {
    let mut space = flat(&[(0, 1), (2, u64::MAX - 2)]);
    let mut batch = space.reservation_batch(2);
    assert_eq!(
        batch
            .reserve(request(u64::MAX - 2, PageType::Raw))
            .unwrap()
            .extent(),
        extent(2, u64::MAX - 2)
    );
    assert_eq!(
        batch.reserve(request(1, PageType::Raw)).unwrap().extent(),
        extent(0, 1)
    );
    batch.finish();
    assert_eq!(space.reusable_bytes().unwrap(), ByteLength::new(0));
    assert_eq!(space.reusable_sections().collect::<Vec<_>>(), []);
}

#[test]
fn page_batch_indexes_a_large_run_without_expanding_pages() {
    let capacity = 1_u64 << 40;
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::DEFAULT,
        [(Some(ManagerKind::PagedLargeGeneric), extent(8192, capacity))],
    )
    .unwrap();
    let mut batch = space.reservation_batch(2);
    assert_eq!(
        batch.reserve(request(1, PageType::Meta)).unwrap().extent(),
        extent(8192, 1)
    );
    assert_eq!(
        batch.reserve(request(1, PageType::Raw)).unwrap().extent(),
        extent(12288, 1)
    );
    let work = batch.finish();
    assert_eq!(work.indexed_extents, 1);
    assert_eq!(work.restored_extents, 2);
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        [
            (extent(8193, 4095), FreeClass::Page(PageType::Meta)),
            (
                extent(12289, capacity - 4097),
                FreeClass::Page(PageType::Raw)
            ),
        ]
    );
    assert_eq!(
        space.reusable_bytes().unwrap(),
        ByteLength::new(capacity - 2)
    );
}

#[test]
fn batch_coalesces_claimed_page_remainders_before_later_reservations() {
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        128,
        FileSpacePageSize::DEFAULT,
        [
            (Some(ManagerKind::PagedSmallRaw), extent(8192, 4096)),
            (Some(ManagerKind::PagedSmallMetadata), extent(12288, 64)),
        ],
    )
    .unwrap();
    let mut batch = space.reservation_batch(2);
    assert_eq!(
        batch
            .reserve(request(100, PageType::Meta))
            .unwrap()
            .extent(),
        extent(8192, 100)
    );
    assert_eq!(
        batch
            .reserve(request(4000, PageType::Meta))
            .unwrap()
            .extent(),
        extent(8292, 4000)
    );
    batch.finish();
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        [(extent(12292, 60), FreeClass::Page(PageType::Meta)),]
    );
    assert_eq!(space.reusable_bytes().unwrap(), ByteLength::new(60));
}

#[rstest]
#[case::adjacent(164, 32, Some((100, 96)), &[(300, 64)])]
#[case::seedless(200, 32, None, &[(100, 64), (300, 64)])]
#[case::eligible(500, 100, Some((500, 96)), &[(100, 64), (300, 64), (596, 4)])]
fn aggregate_returns_keep_current_threshold_admission_inside_a_batch(
    #[case] start: u64,
    #[case] length: u64,
    #[case] chosen: Option<(u64, u64)>,
    #[case] remaining: &[(u64, u64)],
    #[values(1, 4)] count: usize,
) {
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        100,
        FileSpacePageSize::DEFAULT,
        [(None, extent(100, 64)), (None, extent(300, 64))],
    )
    .unwrap();
    let mut batch = space.reservation_batch(count);
    assert!(batch.release_aggregate(StoredAddress::new(1000), extent(start, length)));
    assert_eq!(
        batch
            .reserve(request(96, PageType::Raw))
            .map(|reservation| reservation.extent()),
        chosen.map(|(start, length)| extent(start, length))
    );
    batch.finish();
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        remaining
            .iter()
            .map(|&(start, length)| extent(start, length))
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::empty(&[])]
#[case::single(&[(10, 8)])]
fn returning_batches_keep_one_index_as_free_space_grows(#[case] initial: &[(u64, u64)]) {
    let mut space = flat(initial);
    let mut batch = space.reservation_batch_with_returns(3);
    assert!(batch.release_aggregate(StoredAddress::new(1000), extent(100, 32)));
    assert!(batch.release_aggregate(StoredAddress::new(1000), extent(200, 64)));
    assert_eq!(
        batch.reserve(request(16, PageType::Raw)).unwrap().extent(),
        extent(100, 16)
    );
    let work = batch.finish();
    assert_eq!(work.indexed_extents, u64::try_from(initial.len()).unwrap());
    assert_eq!(
        work.inserted_extents,
        u64::try_from(initial.len()).unwrap() + 3
    );
    assert_eq!(work.removed_extents, 1);
    assert_eq!(
        work.restored_extents,
        u64::try_from(initial.len()).unwrap() + 2
    );
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        initial
            .iter()
            .map(|&(start, length)| extent(start, length))
            .chain([extent(116, 16), extent(200, 64)])
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::aligned_small(0, 4095, 0)]
#[case::ending_at_boundary(4000, 96, 4000)]
#[case::crossing_boundary(4000, 97, 4096)]
#[case::exact_page(4000, 4096, 4096)]
#[case::aligned_page(4096, 4096, 4096)]
#[case::large(3, 8193, 4096)]
#[case::last_page(u64::MAX - 100, 50, u64::MAX - 100)]
fn block_placement_uses_complete_page_geometry(
    #[case] start: u64,
    #[case] size: u64,
    #[case] expected: u64,
) {
    let block = request(size, PageType::Meta).for_block();
    let placement = block
        .place_after(StoredAddress::new(start), Some(FileSpacePageSize::DEFAULT))
        .unwrap();
    assert_eq!(placement.extent(), extent(expected, size));
    assert_eq!(placement.request(), block);
    assert!(placement.request().is_block());
    assert_eq!(
        block
            .place_after(StoredAddress::new(start), None)
            .unwrap()
            .extent(),
        extent(start, size)
    );
    assert_eq!(
        request(size, PageType::Meta)
            .place_after(StoredAddress::new(start), Some(FileSpacePageSize::DEFAULT))
            .unwrap()
            .extent(),
        extent(start, size)
    );
}

#[rstest]
#[case::minimum(FileSpacePageSize::MIN)]
#[case::odd(FileSpacePageSize::try_from(513).unwrap())]
#[case::default(FileSpacePageSize::DEFAULT)]
#[case::maximum(FileSpacePageSize::MAX)]
fn block_geometry_uses_the_configured_page_size(#[case] page_size: FileSpacePageSize) {
    let page = page_size.get();
    let small = request(100, PageType::Meta).for_block();
    assert_eq!(
        small
            .place_after(StoredAddress::new(page - 99), Some(page_size))
            .unwrap()
            .extent(),
        extent(page, 100)
    );
    let large = request(page + 1, PageType::Meta).for_block();
    assert_eq!(
        large
            .place_after(StoredAddress::new(page + 1), Some(page_size))
            .unwrap()
            .extent(),
        extent(2 * page, page + 1)
    );
}

#[rstest]
#[case::alignment(u64::MAX - 100, 102, u64::MAX - 100, 101)]
#[case::allocation(u64::MAX - 4095, 4096, u64::MAX - 4095, 4096)]
fn block_placement_rejects_unrepresentable_geometry(
    #[case] start: u64,
    #[case] size: u64,
    #[case] offset: u64,
    #[case] length: u64,
) {
    assert_eq!(
        request(size, PageType::Meta)
            .for_block()
            .place_after(StoredAddress::new(start), Some(FileSpacePageSize::DEFAULT))
            .unwrap_err(),
        FormatError::OffsetOverflow { offset, length }
    );
}

#[rstest]
#[case::small(4000, 200, 100, Some(4096), &[(4000, 96), (4196, 4)])]
#[case::page(4000, 5000, 4096, Some(4096), &[(4000, 96), (8192, 808)])]
#[case::large(4000, 5000, 4097, Some(4096), &[(4000, 96), (8193, 807)])]
#[case::crossing_only(4000, 100, 100, None, &[(4000, 100)])]
#[case::compact_run(4096, 40_960_000, 4097, Some(4096), &[(8193, 40_955_903)])]
#[case::last_page(u64::MAX - 100, 100, 100, Some(u64::MAX - 100), &[])]
fn block_reservations_keep_alignment_remainders_and_compact_runs(
    #[case] start: u64,
    #[case] available: u64,
    #[case] size: u64,
    #[case] expected: Option<u64>,
    #[case] remaining: &[(u64, u64)],
    #[values(1, 2)] count: usize,
    #[values(PageType::Meta, PageType::Raw)] page_type: PageType,
) {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    space.return_allocation_with_page_type(
        extent(start, available),
        AllocationClass::Metadata,
        page_type,
    );
    let request = ReservationRequest::with_page_type(
        NonZeroByteLength::new(size).unwrap(),
        AllocationClass::Metadata,
        page_type,
    )
    .for_block();
    let mut batch = space.reservation_batch(count);
    let reservation = batch.reserve(request);
    assert_eq!(
        reservation.as_ref().map(|reservation| reservation.extent()),
        expected.map(|start| extent(start, size))
    );
    if let Some(reservation) = reservation {
        assert_eq!(reservation.placement().request(), request);
        assert_eq!(reservation.class(), AllocationClass::Metadata);
        assert_eq!(reservation.page_type(), page_type);
    }
    let work = batch.finish();
    assert_eq!(
        space.reusable_sections().collect::<Vec<_>>(),
        remaining
            .iter()
            .map(|&(start, length)| extent(start, length))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        space.reusable_bytes().unwrap(),
        ByteLength::new(available - expected.map_or(0, |_| size))
    );
    assert_eq!(work.indexed_extents, u64::from(count > 1));
    assert!(work.inserted_extents <= 3, "{work:?}");
    assert!(work.lookups <= 2, "{work:?}");
}

#[rstest]
fn block_reservations_claim_only_complete_pages_from_the_other_type(#[values(1, 2)] count: usize) {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    space.return_allocation(extent(4000, 8292), AllocationClass::Raw);
    let mut batch = space.reservation_batch(count);
    let allocation = batch
        .reserve(request(4097, PageType::Meta).for_block())
        .unwrap();
    assert_eq!(allocation.extent(), extent(4096, 4097));
    batch.finish();
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        [
            (extent(4000, 96), FreeClass::Page(PageType::Raw)),
            (extent(8193, 4095), FreeClass::Page(PageType::Meta)),
            (extent(12288, 4), FreeClass::Page(PageType::Raw)),
        ]
    );
}

#[rstest]
fn mixed_reservations_update_block_indexes_after_each_split(#[values(1, 3)] count: usize) {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    space.return_allocation(extent(4000, 200), AllocationClass::Metadata);
    let mut batch = space.reservation_batch(count);
    assert_eq!(
        batch.reserve(request(16, PageType::Meta)).unwrap().extent(),
        extent(4000, 16)
    );
    assert_eq!(
        batch
            .reserve(request(100, PageType::Meta).for_block())
            .unwrap()
            .extent(),
        extent(4096, 100)
    );
    assert_eq!(
        batch.reserve(request(4, PageType::Meta)).unwrap().extent(),
        extent(4196, 4)
    );
    assert_eq!(
        batch
            .reserve(request(80, PageType::Meta).for_block())
            .unwrap()
            .extent(),
        extent(4016, 80)
    );
    assert_eq!(batch.reserve(request(1, PageType::Meta).for_block()), None);
    batch.finish();
    assert_eq!(space.reusable_sections().collect::<Vec<_>>(), []);
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/proptest-regressions/reservations.txt"
        )))),
        ..ProptestConfig::with_cases(64)
    })]

    #[test]
    fn mixed_batch_reservations_match_scalar_policy(
        initial in prop::collection::vec((0u64..4096, 1u64..8192, any::<bool>()), 0..24),
        requests in prop::collection::vec((1u64..16384, any::<bool>(), any::<bool>()), 0..64),
        paged in any::<bool>(),
    ) {
        let strategy = if paged { FileSpaceStrategy::Page } else { FileSpaceStrategy::FsmAggr };
        let entries: Vec<_> = initial.iter().enumerate().map(|(i, &(offset, length, metadata))| {
            (Some(if metadata { ManagerKind::PagedSmallMetadata } else { ManagerKind::PagedSmallRaw }),
             extent(8192 + u64::try_from(i).unwrap() * 16384 + offset, length))
        }).collect();
        let mut scalar = SessionSpace::from_persisted(Some(strategy), 1, FileSpacePageSize::DEFAULT, entries.iter().copied()).unwrap();
        let mut indexed = SessionSpace::from_persisted(Some(strategy), 1, FileSpacePageSize::DEFAULT, entries).unwrap();
        let mut batch = indexed.reservation_batch(requests.len().max(2));
        for (size, metadata, block) in requests {
            let request = request(size, if metadata { PageType::Meta } else { PageType::Raw });
            let request = if block { request.for_block() } else { request };
            let expected = scalar.reservation_batch(1).reserve(request).map(|allocation| allocation.extent());
            let allocation = batch.reserve(request).map(|allocation| allocation.extent());
            prop_assert_eq!(allocation, expected);
            if paged && block && let Some(allocation) = allocation {
                let offset = allocation.start().get() % FileSpacePageSize::DEFAULT.get();
                if size < FileSpacePageSize::DEFAULT.get() {
                    prop_assert!(offset + size <= FileSpacePageSize::DEFAULT.get());
                } else {
                    prop_assert_eq!(offset, 0);
                }
            }
        }
        batch.finish();
        prop_assert_eq!(indexed.admitted_snapshot().sections().collect::<Vec<_>>(), scalar.admitted_snapshot().sections().collect::<Vec<_>>());
        prop_assert_eq!(indexed.reusable_bytes().unwrap(), scalar.reusable_bytes().unwrap());
    }
}

fn flat(extents: &[(u64, u64)]) -> SessionSpace {
    SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        1,
        FileSpacePageSize::DEFAULT,
        extents
            .iter()
            .map(|&(start, len)| (None, extent(start, len))),
    )
    .unwrap()
}

fn request(size: u64, page_type: PageType) -> ReservationRequest {
    ReservationRequest::with_page_type(
        NonZeroByteLength::new(size).unwrap(),
        match page_type {
            PageType::Meta => AllocationClass::Metadata,
            PageType::Raw => AllocationClass::Raw,
        },
        page_type,
    )
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(length)).unwrap()
}
