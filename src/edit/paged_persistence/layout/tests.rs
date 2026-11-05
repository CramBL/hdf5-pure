use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_space::__private::SessionSpace;
use proptest::test_runner::Config as ProptestConfig;
use proptest::test_runner::FileFailurePersistence;
use rstest::rstest;
use test_util::checksum;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::object_header::v2::HeaderFlags;

use crate::FileSpaceStrategy;
use crate::address::StoredAddressExt;

use super::super::PagedPersistencePlan;
use super::super::ReservationPartition;
use super::*;

#[rstest]
#[case::empty(512, 8, 8, 0, false, false)]
#[case::tracked_narrow(768, 2, 8, 248, true, false)]
#[case::byte_boundary(512, 8, 8, 249, false, false)]
#[case::tracked_byte_boundary(512, 8, 8, 255, true, false)]
#[case::narrow_lengths(1_024, 8, 2, 256, false, false)]
#[case::two_byte_boundary(4_096, 8, 8, 65_525, true, false)]
#[case::many_groups(131_072, 8, 8, 65_525, true, true)]
#[case::large_page(1 << 30, 8, 8, 1, false, false)]
fn controlled_append_layouts_preserve_counts_geometry_and_free_bytes(
    #[case] page: u64,
    #[case] offsets: u8,
    #[case] lengths: u8,
    #[case] count: u64,
    #[case] tracked: bool,
    #[case] distinct: bool,
) {
    let page = FileSpacePageSize::try_from(page).unwrap();
    let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        page,
        (0..count).map(|index| {
            let start = if distinct {
                (index + 2) * 2 * page.get()
            } else {
                2 * page.get() + 2 * index
            };
            (
                Some(ManagerKind::PagedSmallMetadata),
                extent(
                    StoredAddress::new(start),
                    if distinct { index + 1 } else { 1 },
                )
                .unwrap(),
            )
        }),
    )
    .unwrap();
    let eoa = if distinct {
        (count + 8) * 2 * page.get()
    } else {
        (2 * page.get() + 2 * count).div_ceil(page.get()) * page.get() + 8 * page.get()
    };
    let eoa = StoredAddress::new(eoa);
    let layout = if tracked {
        MessageRecordLayout::from_header_flags(HeaderFlags::TRACKS_CREATION_ORDER.0)
    } else {
        MessageRecordLayout::PLAIN
    };
    let prefix = ObjectHeaderPrefix::with_layout(layout);
    let records = v2::message_records(
        &[Message::new(MessageType::DATATYPE, &[1, 2])],
        HeaderFlags(if tracked {
            HeaderFlags::TRACKS_CREATION_ORDER.0
        } else {
            0
        }),
    );
    let geometry = ObjectHeaderAllocationGeometry::new(prefix, widths, &records).unwrap();
    let append = space.plan_paged_append(eoa, &[], &[]).unwrap();
    let original = append.admitted_snapshot().reusable_bytes().unwrap();
    let (base, sizes) = ManagerProjection::indexed(append.admitted_snapshot(), widths).unwrap();
    let planned = PageLayout::new(&geometry, page, eoa, None, &base, &sizes, true)
        .unwrap()
        .unwrap();
    assert!(
        planned
            .gaps
            .iter()
            .any(|gap| gap.len() == ByteLength::new(1))
    );
    let capacity = planned.end.checked_distance_from(eoa).unwrap();
    let tail = append
        .reserve(ByteLength::new(capacity))
        .unwrap()
        .with_metadata_padding(&planned.gaps)
        .unwrap();
    let snapshot = tail.post.admitted_snapshot();
    assert_eq!(
        snapshot.reusable_bytes().unwrap(),
        ByteLength::new(
            original.get()
                + tail.final_eoa.checked_distance_from(planned.end).unwrap()
                + planned.gaps.iter().map(|gap| gap.len().get()).sum::<u64>()
        )
    );
    let (projection, sizes) = ManagerProjection::indexed(snapshot, widths).unwrap();
    assert_eq!(
        (
            projection.sections(ManagerKind::PagedSmallMetadata).len() as u64,
            sizes.len() as u64
        ),
        planned.metadata_counts
    );
    let managers = projection
        .place_with_kind(|kind, measured| {
            let allocations = planned
                .managers
                .iter()
                .find(|(candidate, _)| *candidate == kind)
                .unwrap()
                .1;
            assert_eq!(
                [allocations[0].len(), allocations[1].len()],
                measured.map(ByteLength::new)
            );
            Ok(allocations)
        })
        .unwrap();
    let occupied = planned
        .extension
        .iter()
        .copied()
        .chain(managers.allocations())
        .collect::<Vec<_>>();
    for allocation in &occupied {
        if allocation.len() < ByteLength::new(page.get()) {
            assert_eq!(
                allocation.start().get() / page.get(),
                (allocation.end().get() - 1) / page.get(),
                "{allocation:?}"
            );
        } else {
            assert_eq!(allocation.start().get() % page.get(), 0, "{allocation:?}");
        }
        assert!(
            snapshot
                .sections()
                .all(|(free, _)| !free.range().overlaps(allocation.range()))
        );
    }
    let partition = ReservationPartition::new(tail.allocation, occupied).unwrap();
    assert_eq!(partition.gaps, planned.gaps);
    assert_eq!(
        partition
            .occupied
            .iter()
            .map(|allocation| allocation.len().get())
            .sum::<u64>()
            + partition
                .gaps
                .iter()
                .map(|gap| gap.len().get())
                .sum::<u64>(),
        capacity
    );
    let metadata_end = planned
        .managers
        .iter()
        .find(|(kind, _)| *kind == ManagerKind::PagedSmallMetadata)
        .unwrap()
        .1[1]
        .end();
    let suffix = planned
        .extension
        .iter()
        .filter(|allocation| allocation.start() >= metadata_end)
        .map(|allocation| allocation.len().get())
        .sum::<u64>();
    assert!(suffix < 100, "{suffix}");
    assert!(planned.extension.len() <= ObjectHeaderAllocationGeometry::MAX_CHUNKS);
    let encoded = geometry
        .into_placed_allocation(
            planned
                .extension
                .iter()
                .map(|allocation| allocation.range()),
        )
        .unwrap();
    for ((address, bytes), allocation) in encoded.encode_chunks().zip(planned.extension) {
        assert_eq!(
            (address, bytes.len() as u64),
            (allocation.start(), allocation.len().get())
        );
        let data_len = bytes.len() - 4;
        assert_eq!(
            u32::from_le_bytes(bytes[data_len..].try_into().unwrap()),
            checksum::lookup3(&bytes[..data_len])
        );
    }
}

#[test]
fn inactive_control_cells_coalesce_before_the_chunk_limit() {
    let page = FileSpacePageSize::try_from(512).unwrap();
    let prefix = ObjectHeaderPrefix::PLAIN;
    let records = v2::message_records(
        &[Message::new(MessageType::DATATYPE, &[0; 165])],
        HeaderFlags(0),
    );
    let geometry = ObjectHeaderAllocationGeometry::new(
        prefix,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &records,
    )
    .unwrap();
    let arena = Arena::new(&geometry, page, StoredAddress::new(0), 28, 10).unwrap();
    assert_eq!(arena.root, extent(StoredAddress::new(0), 200).unwrap());
    assert_eq!(arena.end(), StoredAddress::new(541));
    let suffix = extent(StoredAddress::new(7_172), 508 + 244 * 65_536).unwrap();
    let (chunks, gaps) = arena.chunks(page, &geometry, 1, suffix).unwrap();
    assert_eq!(gaps, [extent(StoredAddress::new(228), 1).unwrap()]);
    assert_eq!(
        &chunks[..4],
        &[
            extent(StoredAddress::new(0), 200).unwrap(),
            extent(StoredAddress::new(200), 28).unwrap(),
            extent(StoredAddress::new(229), 283).unwrap(),
            extent(StoredAddress::new(512), 29).unwrap(),
        ]
    );
    assert_eq!(chunks.len(), 249);
    let encoding = geometry
        .into_placed_allocation(chunks.iter().map(|allocation| allocation.range()))
        .unwrap();
    assert_eq!(encoding.chunks().len(), 249);
}

#[test]
fn padding_chunks_enforce_the_complete_chunk_limit() {
    let page = FileSpacePageSize::try_from(512).unwrap();
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let mut chunks = vec![extent(StoredAddress::new(0), 200).unwrap()];
    let error = padding_chunks(
        page,
        &geometry,
        extent(StoredAddress::new(4_096), 256 * 65_536).unwrap(),
        true,
        &mut chunks,
    )
    .unwrap_err();
    assert_eq!(error, FormatError::NestingDepthExceeded);
    assert_eq!(chunks.len(), ObjectHeaderAllocationGeometry::MAX_CHUNKS);
}

#[test]
fn appended_layout_rejects_an_unaligned_start() {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    let (base, sizes) =
        ManagerProjection::indexed(space.admitted_snapshot(), geometry.widths()).unwrap();
    let error = PageLayout::new(
        &geometry,
        FileSpacePageSize::DEFAULT,
        StoredAddress::new(1),
        None,
        &base,
        &sizes,
        false,
    )
    .err()
    .unwrap();
    assert_eq!(
        error,
        FormatError::Internal("an appended PAGE reservation is not aligned".into())
    );
}

#[rstest]
#[case::one_byte(1 << 8)]
#[case::two_bytes(1 << 16)]
#[case::three_bytes(1 << 24)]
#[case::four_bytes(1 << 32)]
#[case::five_bytes(1 << 40)]
#[case::six_bytes(1 << 48)]
#[case::seven_bytes(1 << 56)]
fn control_counts_keep_every_alignment_case_at_one_wire_width(
    #[case] boundary: u64,
    #[values(0, 1, 40)] distance: u64,
    #[values(2, 4, 8)] offsets: u8,
    #[values(2, 4, 8)] lengths: u8,
    #[values(false, true)] tracked: bool,
) {
    let flags = if tracked {
        HeaderFlags::TRACKS_CREATION_ORDER
    } else {
        HeaderFlags(0)
    };
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
        FormatWidths::from_sizes(offsets, lengths).unwrap(),
        &[],
    )
    .unwrap();
    let minimum = *geometry.continuation_chunk_lengths(true).start();
    let window = (minimum - 1).div_ceil(u64::from(offsets) + 1);
    let sections = boundary - distance;
    let controls = control_count(sections, window).unwrap();
    assert!((window + 1..=2 * window + 7).contains(&controls));
    let widths = (sections + controls - window..=sections + controls + 6)
        .map(|count| {
            count
                .to_be_bytes()
                .iter()
                .position(|&byte| byte != 0)
                .map_or(1, |index| 8 - index)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(widths.len(), 1);
}

#[test]
fn control_count_overflow_preserves_the_count_and_increment() {
    assert_eq!(
        control_count(u64::MAX - 1, 3).unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX - 1,
            length: 4,
        }
    );
}

proptest::proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(
            FileFailurePersistence::Direct(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/proptest-regressions/page_manager_partitions.txt"
            ))
        )),
        ..ProptestConfig::with_cases(512)
    })]

    #[test]
    fn active_manager_classes_preserve_the_complete_append_partition(
        (page, offsets, lengths) in proptest::sample::select(vec![
            (512_u64, 8, 8), (513, 4, 4), (768, 8, 2), (1_024, 4, 2),
            (4_096, 8, 8), (65_536, 8, 8), (131_072, 4, 8), (1 << 30, 8, 8),
        ]),
        metadata in proptest::collection::vec(0_u64..1_024, 0..300),
        raw in proptest::collection::vec(0_u64..1_024, 0..40),
        generic in 0_u64..4,
        body_len in proptest::prop_oneof![4 => 0_usize..1_201, 1 => 65_515_usize..65_536],
        tracked in proptest::bool::ANY,
    ) {
        let page = FileSpacePageSize::try_from(page).unwrap();
        let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
        let mut original = Vec::new();
        for (kind, lengths) in [
            (ManagerKind::PagedSmallMetadata, &metadata),
            (ManagerKind::PagedSmallRaw, &raw),
        ] {
            for &length in lengths {
                let start = (original.len() as u64 + 2) * 2 * page.get();
                original.push((
                    Some(kind),
                    extent(
                        StoredAddress::new(start),
                        1 + length % if widths.lengths.get() == 2 { 64 } else { page.get() - 1 },
                    ).unwrap(),
                ));
            }
        }
        for _ in 0..generic {
            let start = (original.len() as u64 + 2) * 2 * page.get();
            original.push((
                Some(ManagerKind::PagedLargeGeneric),
                extent(StoredAddress::new(start), 2 * page.get()).unwrap(),
            ));
        }
        let original_bytes = original.iter().map(|(_, free)| free.len().get()).sum::<u64>();
        let eoa = StoredAddress::new((original.len() as u64 + 18) * 2 * page.get());
        let space = SessionSpace::from_persisted(
            Some(FileSpaceStrategy::Page), 1, page, original,
        ).unwrap();
        let append = space.plan_paged_append(eoa, &[], &[]).unwrap();
        assert_eq!(append.admitted_snapshot().reusable_bytes().unwrap(), ByteLength::new(original_bytes));
        let flags = if tracked { HeaderFlags::TRACKS_CREATION_ORDER } else { HeaderFlags(0) };
        let prefix = ObjectHeaderPrefix::with_layout(
            MessageRecordLayout::from_header_flags(flags.0),
        );
        let body = vec![0_u8; body_len];
        let messages = if body.is_empty() {
            Vec::new()
        } else {
            vec![Message::new(MessageType::OBJECT_COMMENT, &body)]
        };
        let records = v2::message_records(&messages, flags);
        let geometry = ObjectHeaderAllocationGeometry::new(prefix, widths, &records).unwrap();
        let (base, sizes) = ManagerProjection::indexed(append.admitted_snapshot(), widths).unwrap();
        let layout = PageLayout::new(&geometry, page, eoa, None, &base, &sizes, true)
            .unwrap().unwrap();
        let capacity = layout.end.checked_distance_from(eoa).unwrap();
        let tail = append.reserve(ByteLength::new(capacity)).unwrap().with_metadata_padding(&layout.gaps).unwrap();
        let snapshot = tail.post.admitted_snapshot();
        assert_eq!(
            snapshot.reusable_bytes().unwrap(),
            ByteLength::new(original_bytes + tail.final_eoa.checked_distance_from(layout.end).unwrap()
                + layout.gaps.iter().map(|gap| gap.len().get()).sum::<u64>())
        );
        let (projection, sizes) = ManagerProjection::indexed(snapshot, widths).unwrap();
        assert_eq!(
            (projection.sections(ManagerKind::PagedSmallMetadata).len() as u64, sizes.len() as u64),
            layout.metadata_counts
        );
        let managers = projection.place_with_kind(|kind, lengths| {
            let allocations = layout.managers.iter().find(|(candidate, _)| *candidate == kind)
                .unwrap().1;
            assert_eq!([allocations[0].len(), allocations[1].len()], lengths.map(ByteLength::new));
            Ok(allocations)
        }).unwrap();
        let mut pieces = layout.extension.iter().copied()
            .chain(managers.allocations()).chain(layout.gaps.iter().copied()).collect::<Vec<_>>();
        pieces.sort_unstable();
        let mut cursor = eoa;
        for piece in pieces {
            assert_eq!(piece.start(), cursor);
            cursor = piece.end();
        }
        assert_eq!(cursor, layout.end);
        for allocation in layout.extension.iter().copied().chain(managers.allocations()) {
            if allocation.len() < ByteLength::new(page.get()) {
                assert_eq!(
                    allocation.start().get() / page.get(),
                    (allocation.end().get() - 1) / page.get(),
                );
            } else {
                assert_eq!(allocation.start().get() % page.get(), 0);
            }
            assert!(snapshot.sections()
                .all(|(free, _)| !free.range().overlaps(allocation.range())));
        }
        let encoding = geometry.into_placed_allocation(
            layout.extension.iter().map(|range| range.range()),
        )
            .unwrap();
        for (address, bytes) in encoding.encode_chunks() {
            assert!(layout.extension.iter().any(|chunk|
                chunk.start() == address && chunk.len() == ByteLength::new(bytes.len() as u64)));
            let data_len = bytes.len() - 4;
            assert_eq!(
                u32::from_le_bytes(bytes[data_len..].try_into().unwrap()),
                checksum::lookup3(&bytes[..data_len]),
            );
        }
        let geometry = ObjectHeaderAllocationGeometry::new(prefix, widths, &records).unwrap();
        let plan = PagedPersistencePlan::new(
            &space, eoa, &[], &[], &geometry, false,
        ).unwrap().unwrap();
        assert_planned_ranges_cover_the_reservation(&plan, page, original_bytes, eoa);
        let encoding = geometry.into_placed_allocation(
            plan.extension.iter().map(|range| range.range()),
        )
            .unwrap();
        assert_eq!(
            encoding.chunks().collect::<Vec<_>>(),
            plan.extension.iter().map(|range| (range.start(), range.len().get())).collect::<Vec<_>>(),
        );
    }
}

fn assert_planned_ranges_cover_the_reservation(
    plan: &PagedPersistencePlan,
    page: FileSpacePageSize,
    original_bytes: u64,
    previous_eoa: StoredAddress,
) {
    let mut allocations = plan.extension.clone();
    for block in &plan.blocks {
        allocations.push(extent(block.header_address, block.header.len() as u64).unwrap());
        allocations
            .push(extent(block.section_list_address, block.section_list.len() as u64).unwrap());
    }
    allocations.sort_unstable();
    let snapshot = plan.tail.post.admitted_snapshot();
    let mut cursor = plan.tail.allocation.start();
    let mut free_bytes = 0;
    for allocation in allocations {
        assert!(plan.tail.allocation.range().contains(allocation.range()));
        assert!(cursor <= allocation.start());
        let gap_length = allocation.start().checked_distance_from(cursor).unwrap();
        let covered = snapshot
            .sections()
            .map(|(free, _)| {
                free.end()
                    .get()
                    .min(allocation.start().get())
                    .saturating_sub(free.start().get().max(cursor.get()))
            })
            .sum::<u64>();
        assert_eq!(covered, gap_length);
        free_bytes += gap_length;
        if allocation.len() < ByteLength::new(page.get()) {
            assert_eq!(
                allocation.start().get() / page.get(),
                (allocation.end().get() - 1) / page.get(),
            );
        } else {
            assert_eq!(allocation.start().get() % page.get(), 0);
        }
        assert!(
            snapshot
                .sections()
                .all(|(free, _)| !free.range().overlaps(allocation.range()))
        );
        cursor = allocation.end();
    }
    let gap_length = plan
        .tail
        .allocation
        .end()
        .checked_distance_from(cursor)
        .unwrap();
    let covered = snapshot
        .sections()
        .map(|(free, _)| {
            free.end()
                .get()
                .min(plan.tail.allocation.end().get())
                .saturating_sub(free.start().get().max(cursor.get()))
        })
        .sum::<u64>();
    assert_eq!(covered, gap_length);
    free_bytes += gap_length;
    let expected = if plan.tail.appended {
        original_bytes
            + free_bytes
            + plan
                .tail
                .final_eoa
                .checked_distance_from(plan.tail.allocation.end())
                .unwrap()
    } else {
        assert_eq!(plan.tail.final_eoa, previous_eoa);
        original_bytes - plan.tail.allocation.len().get() + free_bytes
    };
    assert_eq!(
        snapshot.reusable_bytes().unwrap(),
        ByteLength::new(expected)
    );
}
