use std::io::ErrorKind;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{
    AllocationClass, Extent, PageTransition, PageType, ReservationRequest, SessionSpace,
};
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::DiskOp;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FileLocking;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::MemoryStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::PersistedFreeSpacePolicy;
use crate::address::{
    AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
};
use crate::edit::WriteEngine;
use crate::edit::observations::{self, ZeroWork};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::image::WriteBuffering;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::MetadataCacheConfig;
use crate::source::Source;
use crate::width::OffsetWidth;
use crate::{FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy, UserblockSize};

use super::{AppendCursor, ReservationFailure, ReservationPlan};
use crate::edit::block::{BlockRequest, EncodingDependency};

mod emission;
mod preferred;

#[rstest]
#[case::two(OffsetWidth::Two, 65_535)]
#[case::four(OffsetWidth::Four, 4_294_967_295)]
fn narrow_block_reservations_check_page_capacity_beyond_encoded_bytes(
    #[case] offsets: OffsetWidth,
    #[case] undefined: u64,
    #[values(0, 512)] base: u64,
) {
    let page_size = FileSpacePageSize::DEFAULT;
    let start = undefined - (2 * page_size.get() - 1);
    let mut cursor = AppendCursor {
        offsets,
        addresses: AddressContext::from_effective_base(BaseAddress::new(base)),
        cursor: StoredAddress::new(start),
        page_size: Some(page_size),
        tail_type: Some(PageType::Meta),
        fresh_boundary: None,
    };
    let block = BlockRequest::new(
        request(2 * page_size.get(), PageType::Meta).for_block(),
        page_size.get() + 1,
        EncodingDependency::Allocation,
    )
    .unwrap();
    assert!(base + start + block.encoded_length() < undefined);
    let error = cursor.reserve(block.reservation()).unwrap_err();
    assert_eq!(
        error,
        FormatError::AddressTooLarge {
            address: undefined + 1,
            offset_size: offsets.get()
        }
    );
    assert_eq!(cursor.cursor.get(), start);
    assert_eq!(cursor.tail_type, Some(PageType::Meta));
}

#[rstest]
#[case::plain(0, None, 65_530, 4, 65_534)]
#[case::userblock(512, None, 65_018, 4, 65_022)]
#[case::page(0, Some(FileSpacePageSize::DEFAULT), 61_440, 1, 61_441)]
fn narrow_append_cursor_checks_complete_capacity_before_advancing(
    #[case] base: u64,
    #[case] page_size: Option<FileSpacePageSize>,
    #[case] start: u64,
    #[case] size: u64,
    #[case] end: u64,
) {
    let mut cursor = AppendCursor {
        offsets: OffsetWidth::Two,
        addresses: AddressContext::from_effective_base(BaseAddress::new(base)),
        cursor: StoredAddress::new(start),
        page_size,
        tail_type: Some(PageType::Raw),
        fresh_boundary: None,
    };
    let (range, _, _) = cursor.reserve(request(size, PageType::Raw)).unwrap();
    assert_eq!(range.start().get(), base + start);
    assert_eq!(cursor.cursor.get(), end);
    let before = cursor.cursor;
    let tail = cursor.tail_type;
    let error = cursor.reserve(request(4096, PageType::Meta)).unwrap_err();
    let FormatError::AddressTooLarge {
        address,
        offset_size,
    } = error
    else {
        panic!("expected AddressTooLarge, got {error:?}");
    };
    assert!(address > u64::from(u16::MAX));
    assert_eq!(offset_size, 2);
    assert_eq!(cursor.cursor, before);
    assert_eq!(cursor.tail_type, tail);
}

#[rstest]
#[case::append(FileSpaceStrategy::None, &[0, 16, 48], 72)]
#[case::aggregates(FileSpaceStrategy::Aggr, &[0, 2048, 16], 4096)]
#[case::flat(FileSpaceStrategy::FsmAggr, &[0, 2048, 16], 4096)]
#[case::page(FileSpaceStrategy::Page, &[0, 4096, 8192], 8216)]
fn deferred_reservations_keep_owned_ranges_disjoint_before_emission(
    #[case] strategy: FileSpaceStrategy,
    #[case] offsets: &[u64],
    #[case] capacity: u64,
    #[values(0, 4096)] userblock: u64,
) {
    let (_directory, mut engine) = engine(strategy, userblock);
    let start = engine.image.len();
    let requests = [
        request(16, PageType::Raw),
        request(32, PageType::Meta),
        request(24, PageType::Raw),
    ];
    let mut plan = planned(&mut engine, &requests);
    assert_eq!(plan.engine.image.len(), start);
    assert_eq!(plan.end(), PhysicalAddress::new(start + capacity));
    assert_eq!(
        plan.reservations()
            .iter()
            .map(|entry| entry.range())
            .collect::<Vec<_>>(),
        requests
            .iter()
            .zip(offsets)
            .map(|(request, offset)| FileRange::new(
                PhysicalAddress::new(start + offset),
                request.size().get()
            )
            .unwrap())
            .collect::<Vec<_>>()
    );
    for (index, request) in requests.iter().enumerate() {
        let bytes =
            vec![u8::try_from(index + 1).unwrap(); usize::try_from(request.size().get()).unwrap()];
        assert_eq!(
            plan.write_next(&bytes).unwrap(),
            PhysicalAddress::new(start + offsets[index])
        );
    }
    assert_eq!(plan.finish().unwrap().indexed_extents, 0);
    assert_eq!(engine.image.len(), start + capacity);
    for (index, (request, offset)) in requests.iter().zip(offsets).enumerate() {
        assert_eq!(
            engine
                .image()
                .read_exact_at(
                    start + offset,
                    usize::try_from(request.size().get()).unwrap()
                )
                .unwrap(),
            vec![u8::try_from(index + 1).unwrap(); usize::try_from(request.size().get()).unwrap()]
        );
    }
}

#[test]
fn later_reservation_failure_retains_the_successful_prefix() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.append(&[0; 512]).unwrap();
    let end = engine.image.len();
    engine.space = flat(&[(start, 128), (start + 256, 128)]);
    let requests = [request(64, PageType::Raw), request(u64::MAX, PageType::Raw)];
    let Err(failure) = engine.plan_reservations(&requests) else {
        panic!("expected an overflowing second request");
    };
    let ReservationFailure { mut plan, error } = *failure;
    assert!(
        matches!(error, Error::Format(FormatError::OffsetOverflow { offset, length }) if offset == end && length == u64::MAX),
        "{error:?}"
    );
    assert_eq!(
        plan.reservations()
            .iter()
            .map(|entry| entry.range())
            .collect::<Vec<_>>(),
        [FileRange::new(PhysicalAddress::new(start), 64).unwrap()]
    );
    assert!(plan.unresolved().is_none());
    assert_eq!(plan.end(), PhysicalAddress::new(end));
    assert_eq!(plan.work().indexed_extents, 2);
    plan.write_next(&[17; 64]).unwrap();
    let work = plan.finish().unwrap();
    assert_eq!(work.indexed_extents, 2);
    assert_eq!(work.restored_extents, 2);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [extent(start + 64, 64), extent(start + 256, 128)]
    );
    assert_eq!(engine.image().read_exact_at(start, 64).unwrap(), [17; 64]);
}

#[test]
fn failed_physical_resolution_retains_the_unresolved_stored_allocation() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 512);
    engine.aggregation = None;
    let start = engine.append(&[0; 64]).unwrap();
    let stored = start - 512;
    let invalid = u64::MAX - 31;
    engine.space = flat(&[(stored, 16), (invalid, 8)]);
    let requests = [request(16, PageType::Raw), request(8, PageType::Raw)];
    let Err(failure) = engine.plan_reservations(&requests) else {
        panic!("expected physical address overflow");
    };
    let ReservationFailure { plan, error } = *failure;
    assert_eq!(
        plan.reservations()
            .iter()
            .map(|entry| entry.range())
            .collect::<Vec<_>>(),
        [FileRange::new(PhysicalAddress::new(start), 16).unwrap()]
    );
    assert_eq!(plan.unresolved().unwrap().extent(), extent(invalid, 8));
    assert!(
        matches!(error, Error::Format(FormatError::OffsetOverflow { offset, length }) if offset == invalid && length == 512),
        "{error:?}"
    );
    drop(plan);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
}

#[test]
fn page_gaps_become_pending_only_after_ordered_emission() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::Page, 4096);
    engine
        .space
        .record_transition(PageType::Raw, PageTransition::Unchanged);
    engine.append(&[19]).unwrap();
    let start = engine.image.len();
    let requests = [
        request(15, PageType::Raw),
        request(32, PageType::Meta),
        request(48, PageType::Raw),
    ];
    let mut plan = planned(&mut engine, &requests);
    let boundary = start + 4095;
    assert_eq!(
        plan.engine.space.pending_padding(PageType::Raw).unwrap(),
        []
    );
    assert_eq!(
        plan.engine.space.pending_padding(PageType::Meta).unwrap(),
        []
    );
    assert_eq!(
        plan.reservations()
            .iter()
            .map(|entry| entry.range().start().get())
            .collect::<Vec<_>>(),
        [start, boundary, boundary + 4096]
    );
    plan.write_next(&[23; 15]).unwrap();
    plan.write_next(&[29; 32]).unwrap();
    plan.write_next(&[31; 48]).unwrap();
    plan.finish().unwrap();
    assert_eq!(
        engine.space.pending_padding(PageType::Raw).unwrap(),
        [extent(start + 15 - 4096, 4080)]
    );
    assert_eq!(
        engine.space.pending_padding(PageType::Meta).unwrap(),
        [extent(boundary + 32 - 4096, 4064)]
    );
    assert_eq!(
        engine.image().read_exact_at(start + 15, 4080).unwrap(),
        vec![0; 4080]
    );
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Raw)));
}

#[rstest]
#[case::seedless(100, false)]
#[case::seeded(100, true)]
#[case::eligible(1, false)]
fn aggregate_spare_admission_is_available_to_later_batch_requests(
    #[case] threshold: u64,
    #[case] seed: bool,
) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    let first = engine.reserve(2000, AllocationClass::Metadata).unwrap();
    let start = engine.place(first, &[3; 2000]).unwrap();
    let second = engine.reserve(64, AllocationClass::Raw).unwrap();
    engine.place(second, &[5; 64]).unwrap();
    engine.space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::FsmAggr),
        threshold,
        FileSpacePageSize::DEFAULT,
    );
    if seed {
        engine
            .space
            .return_allocation(extent(start + 1900, 100), AllocationClass::Raw);
    }
    engine
        .space
        .return_allocation(extent(start + 500, 101), AllocationClass::Raw);
    let requests = [
        request(149, PageType::Meta),
        request(if seed { 148 } else { 48 }, PageType::Raw),
    ];
    let mut plan = planned(&mut engine, &requests);
    let next = plan.reservations()[1].range().start().get();
    assert_eq!(
        next,
        if seed {
            start + 1900
        } else if threshold == 1 {
            start + 2000
        } else {
            start + 500
        }
    );
    assert_eq!(plan.work().indexed_extents, if seed { 2 } else { 1 });
    plan.write_next(&[7; 149]).unwrap();
    plan.write_next(&vec![11; if seed { 148 } else { 48 }])
        .unwrap();
    plan.finish().unwrap();
}

#[test]
fn abandoned_plans_keep_reused_allocations_unavailable_without_source_writes() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.append(&[13; 512]).unwrap();
    let end = engine.image.len();
    engine.space = flat(&[(start, 64), (start + 128, 64)]);
    let requests = [
        request(64, PageType::Raw),
        request(64, PageType::Raw),
        request(16, PageType::Raw),
    ];
    let plan = planned(&mut engine, &requests);
    assert_eq!(plan.reservations().len(), 3);
    drop(plan);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    assert_eq!(engine.image.len(), end);
    assert_eq!(engine.image().read_exact_at(start, 512).unwrap(), [13; 512]);
}

#[test]
fn uncertain_publication_rejects_reservation_without_changing_free_space() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.append(&[13; 512]).unwrap();
    engine.space = flat(&[(start, 64), (start + 128, 64)]);
    engine.context.allocations.make_uncertain();
    let requests = [request(64, PageType::Raw), request(64, PageType::Raw)];
    let Err(failure) = engine.plan_reservations(&requests) else {
        panic!("expected uncertain publication rejection");
    };
    assert!(
        matches!(failure.error, Error::AllocationPublicationUncertain),
        "{:?}",
        failure.error
    );
    assert_eq!(failure.plan.reservations().len(), 0);
    drop(failure);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [extent(start, 64), extent(start + 128, 64)]
    );
}

#[test]
fn failed_emission_keeps_reserved_space_until_explicit_rollback() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.aggregation = None;
    let start = engine.append(&[13; 512]).unwrap();
    engine.space = flat(&[(start, 64), (start + 128, 64)]);
    let snapshot = engine.snapshot_free();
    let requests = [request(64, PageType::Raw), request(64, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    disk_log::fail_write_after(0, 0);
    let error = plan.write_next(&[17; 64]).unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected write failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(plan.next, 0);
    let error = plan.write_next(&[17; 64]).unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reservation plan has an earlier failed write"),
        "{error:?}"
    );
    drop(plan);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    engine.restore_free(snapshot);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [extent(start, 64), extent(start + 128, 64)]
    );
    assert_eq!(engine.image().read_exact_at(start, 512).unwrap(), [13; 512]);
}

#[rstest]
fn provisional_geometry_above_32bit_addresses_uses_only_local_lengths(#[values(0, 512)] base: u64) {
    let start = (1u64 << 40) + 16;
    let mut cursor = AppendCursor {
        offsets: OffsetWidth::Eight,
        addresses: AddressContext::from_effective_base(BaseAddress::new(base)),
        cursor: StoredAddress::new(start),
        page_size: Some(FileSpacePageSize::DEFAULT),
        tail_type: Some(PageType::Raw),
        fresh_boundary: None,
    };
    let (raw, _, transition) = cursor.reserve(request(32, PageType::Raw)).unwrap();
    assert_eq!(
        raw,
        FileRange::new(PhysicalAddress::new(base + start), 32).unwrap()
    );
    assert_eq!(transition, Some(PageTransition::Unchanged));
    let (metadata, _, transition) = cursor.reserve(request(64, PageType::Meta)).unwrap();
    assert_eq!(
        metadata,
        FileRange::new(PhysicalAddress::new(base + (1u64 << 40) + 4096), 64).unwrap()
    );
    assert_eq!(
        transition,
        Some(PageTransition::Pad {
            extent: extent(start + 32, 4048),
            track_as: Some(PageType::Raw)
        })
    );
    assert_eq!(cursor.cursor, StoredAddress::new((1u64 << 40) + 4160));
}

#[test]
fn emission_length_mismatch_preserves_the_pending_reservation() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::None, 0);
    let start = engine.image.len();
    let requests = [request(16, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    let error = plan.write_next(&[1; 17]).unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::SerializationError(reason)) if reason == "a placement reserved 16 bytes but built 17"),
        "{error:?}"
    );
    assert_eq!(plan.next, 0);
    assert!(!plan.failed);
    assert_eq!(plan.engine.image.len(), start);
    plan.write_next(&[3; 16]).unwrap();
    plan.finish().unwrap();
    assert_eq!(engine.image().read_exact_at(start, 16).unwrap(), [3; 16]);
}

#[rstest]
#[case::append(FileSpaceStrategy::None, false)]
#[case::aggregate(FileSpaceStrategy::Aggr, false)]
#[case::reused_flat(FileSpaceStrategy::FsmAggr, true)]
#[case::page(FileSpaceStrategy::Page, false)]
#[case::reused_page(FileSpaceStrategy::Page, true)]
fn streamed_reservations_keep_complete_capacity_and_following_blocks(
    #[case] strategy: FileSpaceStrategy,
    #[case] reused: bool,
    #[values(0, 4_096)] base: u64,
    #[values(0, 16)] encoded: usize,
) {
    let (_directory, mut engine) = engine(strategy, base);
    let start = engine.image.len();
    if reused {
        engine.append(&[13; 128]).unwrap();
        engine
            .space
            .return_allocation(extent(start - base, 128), AllocationClass::Raw);
    }
    let requests = [
        BlockRequest::new(
            request(64, PageType::Raw),
            u64::try_from(encoded).unwrap(),
            EncodingDependency::Independent,
        )
        .unwrap(),
        BlockRequest::complete(
            request(8, PageType::Meta).for_block(),
            EncodingDependency::Independent,
        ),
    ];
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let ranges: Vec<_> = plan
        .reservations()
        .iter()
        .map(|reservation| reservation.range())
        .collect();
    let result = plan
        .write_next_with(|writer| {
            writer.write(&[17; 5][..encoded.min(5)])?;
            writer.write(&[])?;
            writer.write(&[19; 11][..encoded.saturating_sub(5)])
        })
        .unwrap();
    assert_eq!(result, ranges[0].start());
    assert_eq!(plan.next, 1);
    assert_eq!(plan.reservations().len(), 2);
    assert!(!plan.failed);
    assert_eq!(plan.write_next(&[23; 8]).unwrap(), ranges[1].start());
    plan.finish().unwrap();
    let mut expected = vec![if reused { 13 } else { 0 }; 64];
    expected[..encoded.min(5)].fill(17);
    expected[encoded.min(5)..encoded].fill(19);
    assert_eq!(
        engine.image().read_exact_at(result.get(), 64).unwrap(),
        expected
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(ranges[1].start().get(), 8)
            .unwrap(),
        [23; 8]
    );
    assert!(
        engine
            .space
            .reusable_sections()
            .all(|extent| extent.end().get() <= result.get() - base
                || extent.start().get() >= result.get() - base + 64)
    );
}

#[rstest]
#[case::short(7, false, "a placement reserved 16 bytes but built 15")]
#[case::excess(9, false, "a placement reserved 16 bytes but built 17")]
#[case::suppressed_excess(9, true, "reserved encoding has an earlier failed write")]
fn incomplete_streamed_encodings_retain_their_failed_reservation(
    #[case] last: usize,
    #[case] suppress: bool,
    #[case] expected: &str,
) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.append(&[13; 128]).unwrap();
    engine.space = flat(&[(start, 128)]);
    let requests = [request(16, PageType::Raw), request(16, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    let error = plan
        .write_next_with(|writer| {
            writer.write(&[17; 8])?;
            let result = writer.write(&vec![19; last]);
            if suppress {
                result.ok();
                Ok(())
            } else {
                result
            }
        })
        .unwrap_err();
    match error {
        Error::Format(FormatError::Internal(reason)) if suppress => assert_eq!(reason, expected),
        Error::Format(FormatError::SerializationError(reason)) if !suppress => {
            assert_eq!(reason, expected)
        }
        error => panic!("unexpected error: {error:?}"),
    }
    assert_eq!(plan.next, 0);
    assert!(plan.failed);
    let error = plan.write_next(&[23; 16]).unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reservation plan has an earlier failed write"),
        "{error:?}"
    );
    drop(plan);
    assert_eq!(engine.image().read_exact_at(start, 8).unwrap(), [17; 8]);
    assert_eq!(
        engine.image().read_exact_at(start + 16, 16).unwrap(),
        [13; 16]
    );
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [extent(start + 32, 96)]
    );
}

#[rstest]
#[case::appended(false)]
#[case::reused(true)]
fn failed_streamed_writes_keep_source_prefixes_after_allocator_rollback(
    #[case] reused: bool,
    #[values(false, true)] suppress: bool,
) {
    let (directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let start = engine.image.len();
    if reused {
        engine.append(&[13; 64]).unwrap();
        engine.space = flat(&[(start, 64)]);
    }
    let snapshot = engine.snapshot_free();
    let requests = [request(16, PageType::Raw), request(16, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    disk_log::start();
    disk_log::fail_write_after(1, 3);
    let error = plan
        .write_next_with(|writer| {
            writer.write(&[17; 8])?;
            let result = writer.write(&[19; 8]);
            if !suppress {
                return result;
            }
            let error = result.unwrap_err();
            let Error::Io(error) = error else {
                panic!("expected an injected write failure, got {error:?}");
            };
            assert_eq!(error.kind(), ErrorKind::Other);
            assert_eq!(error.to_string(), "injected disk write failure");
            let error = writer.write(&[23; 8]).unwrap_err();
            assert!(
                matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reserved encoding has an earlier failed write"),
                "{error:?}"
            );
            Ok(())
        })
        .unwrap_err();
    let operations = disk_log::take();
    match error {
        Error::Format(FormatError::Internal(reason)) if suppress => {
            assert_eq!(reason, "reserved encoding has an earlier failed write");
        }
        Error::Io(error) if !suppress => {
            assert_eq!(error.kind(), ErrorKind::Other);
            assert_eq!(error.to_string(), "injected disk write failure");
        }
        error => panic!("unexpected error: {error:?}"),
    }
    assert_eq!(
        operations,
        [
            DiskOp::Write {
                offset: start,
                bytes: vec![17; 8]
            },
            DiskOp::Write {
                offset: start + 8,
                bytes: vec![19; 3]
            }
        ]
    );
    assert_eq!(plan.next, 0);
    assert!(plan.failed);
    drop(plan);
    let remaining = if reused {
        vec![extent(start + 32, 32)]
    } else {
        Vec::new()
    };
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        remaining
    );
    engine.restore_free(snapshot);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        if reused {
            vec![extent(start, 64)]
        } else {
            Vec::new()
        }
    );
    drop(engine);
    assert_eq!(
        std::fs::read(directory.path().join("reservations.h5")).unwrap()
            [usize::try_from(start).unwrap()..usize::try_from(start + 11).unwrap()],
        [17, 17, 17, 17, 17, 17, 17, 17, 19, 19, 19]
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(8)]
#[case::many(128)]
fn bounded_unbuffered_emission_keeps_heap_below_payload_size(#[case] count: usize) {
    let (directory, engine) = engine(FileSpaceStrategy::None, 0);
    drop(engine);
    let mut engine = WriteEngine::open_rw_with_strategy(
        &directory.path().join("reservations.h5"),
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let chunk = vec![17; 8_192];
    let length = u64::try_from(count * chunk.len()).unwrap();
    let requests = [request(length, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    let start = plan.reservations()[0].range().start().get();
    let before = plan.engine.image.buffering_work();
    let (address, measured) = allocation::measure(&format!("streamed_reservation_{count}"), || {
        plan.write_next_with(|writer| {
            for _ in 0..count {
                writer.write(&chunk)?;
            }
            Ok(())
        })
        .unwrap()
    });
    assert_eq!(address.get(), start);
    let after = plan.engine.image.buffering_work();
    assert_eq!(
        after.write_calls - before.write_calls,
        u64::try_from(count).unwrap()
    );
    assert_eq!(after.write_bytes - before.write_bytes, length);
    assert_eq!(after.mirror_bytes, 0);
    assert_eq!(after.mirror_capacity, 0);
    assert_eq!(after.pending_bytes, 0);
    assert!(
        measured.peak_bytes < length / 4,
        "{measured:?}, payload={length}"
    );
    plan.finish().unwrap();
    assert_eq!(
        engine
            .image()
            .read_exact_at(start, usize::try_from(length).unwrap())
            .unwrap(),
        vec![17; usize::try_from(length).unwrap()]
    );
    let (copied, control) = allocation::measure(&format!("assembled_reservation_{count}"), || {
        chunk.repeat(count)
    });
    assert_eq!(u64::try_from(copied.len()).unwrap(), length);
    assert!(control.live_bytes >= length, "{control:?}");
}

#[rstest]
#[case::before_output(false)]
#[case::after_prefix(true)]
fn producer_errors_keep_streamed_reservations_unavailable(#[case] prefix: bool) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.append(&[13; 64]).unwrap();
    engine.space = flat(&[(start, 64)]);
    let requests = [request(16, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    let error = plan
        .write_next_with(|writer| {
            if prefix {
                writer.write(&[17; 8])?;
            }
            Err(FormatError::Internal("injected producer error".into()).into())
        })
        .unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "injected producer error"),
        "{error:?}"
    );
    assert!(plan.failed);
    assert_eq!(plan.next, 0);
    drop(plan);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [extent(start + 16, 48)]
    );
    assert_eq!(
        engine.image().read_exact_at(start, 8).unwrap(),
        [if prefix { 17 } else { 13 }; 8]
    );
}

#[test]
fn uncertain_publication_rejects_a_stream_before_calling_its_producer() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::None, 0);
    let requests = [request(16, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    plan.engine.context.allocations.make_uncertain();
    let mut called = false;
    let error = plan
        .write_next_with(|_| {
            called = true;
            Ok(())
        })
        .unwrap_err();
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    assert!(!called);
    assert_eq!(plan.next, 0);
}

fn engine(strategy: FileSpaceStrategy, userblock: u64) -> (tempfile::TempDir, WriteEngine) {
    engine_with_page(strategy, userblock, FileSpacePageSize::DEFAULT)
}

fn engine_with_page(
    strategy: FileSpaceStrategy,
    userblock: u64,
    page_size: FileSpacePageSize,
) -> (tempfile::TempDir, WriteEngine) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reservations.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(strategy == FileSpaceStrategy::Page)
            .with_page_size(page_size),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    if strategy == FileSpaceStrategy::Page {
        engine.space = SessionSpace::from_strategy(Some(strategy), 1, page_size);
    }
    (directory, engine)
}

#[rstest]
#[case::appended(false)]
#[case::reused(true)]
fn block_reservations_keep_capacity_separate_from_encoding(#[case] reused: bool) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.image.len();
    if reused {
        engine.append(&[13; 128]).unwrap();
        engine.space = flat(&[(start, 128)]);
    }
    let requests = [
        BlockRequest::new(
            request(64, PageType::Raw),
            16,
            EncodingDependency::Independent,
        )
        .unwrap(),
        BlockRequest::complete(request(32, PageType::Raw), EncodingDependency::Independent),
    ];
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("block reservation failed: {:?}", failure.error));
    assert_eq!(
        plan.reservations()[0].range(),
        FileRange::new(PhysicalAddress::new(start), 64).unwrap()
    );
    assert_eq!(
        plan.reservations()[1].range(),
        FileRange::new(PhysicalAddress::new(start + 64), 32).unwrap()
    );
    assert_eq!(
        plan.write_next(&[7; 16]).unwrap(),
        PhysicalAddress::new(start)
    );
    assert_eq!(
        plan.write_next(&[11; 32]).unwrap(),
        PhysicalAddress::new(start + 64)
    );
    plan.finish().unwrap();
    assert_eq!(engine.image().read_exact_at(start, 16).unwrap(), [7; 16]);
    assert_eq!(
        engine.image().read_exact_at(start + 16, 48).unwrap(),
        [if reused { 13 } else { 0 }; 48]
    );
    assert_eq!(
        engine.image().read_exact_at(start + 64, 32).unwrap(),
        [11; 32]
    );
    assert_eq!(
        engine.space.reusable_bytes().unwrap(),
        if reused { 32 } else { 0 }
    );
}

#[rstest]
#[case::scalar(true)]
#[case::batch(false)]
fn large_page_reservations_emit_bounded_transition_padding(
    #[case] scalar: bool,
    #[values(0, 262_144)] base: u64,
) {
    let page = 262_144;
    let (_directory, mut engine) = engine_with_page(
        FileSpaceStrategy::Page,
        base,
        FileSpacePageSize::try_from(page).unwrap(),
    );
    engine.begin_page(PageType::Meta).unwrap();
    engine.append(&[17; 19]).unwrap();
    let start = engine.image.len();
    observations::reset_zeros();
    let address = if scalar {
        let placement = engine.reserve(16, AllocationClass::Raw).unwrap();
        engine.place(placement, &[23; 16]).unwrap()
    } else {
        let requests = [request(16, PageType::Raw), request(32, PageType::Meta)];
        let mut plan = planned(&mut engine, &requests);
        let address = plan.write_next(&[23; 16]).unwrap().get();
        assert_eq!(
            plan.write_next(&[29; 32]).unwrap().get(),
            start + 2 * page - 19
        );
        plan.finish().unwrap();
        address
    };
    assert_eq!(address, start + page - 19);
    assert_eq!(engine.image().read_exact_at(address, 16).unwrap(), [23; 16]);
    assert_eq!(
        engine
            .image()
            .read_exact_at(start, (page - 19) as usize)
            .unwrap(),
        vec![0; (page - 19) as usize]
    );
    assert_eq!(
        engine.space.pending_padding(PageType::Meta).unwrap(),
        [extent(start - base, page - 19)]
    );
    assert_eq!(
        engine.image.len(),
        if scalar {
            address + 16
        } else {
            address + page + 32
        }
    );
    assert_eq!(
        observations::zero_work(),
        ZeroWork {
            calls: if scalar { 1 } else { 2 },
            bytes: page - 19 + if scalar { 0 } else { page - 16 },
            scratch_peak: 65_536
        }
    );
}

#[test]
fn large_appended_capacity_remains_owned_after_bounded_zero_emission() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::None, 4096);
    let start = engine.image.len();
    let capacity = 196_624;
    let requests = [BlockRequest::new(
        request(capacity, PageType::Raw),
        16,
        EncodingDependency::Independent,
    )
    .unwrap()];
    observations::reset_zeros();
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("planning failed: {:?}", failure.error));
    assert_eq!(
        plan.reservations()[0].range(),
        FileRange::new(PhysicalAddress::new(start), capacity).unwrap()
    );
    assert_eq!(plan.write_next(&[31; 16]).unwrap().get(), start);
    plan.finish().unwrap();
    assert_eq!(engine.image.len(), start + capacity);
    assert_eq!(engine.image().read_exact_at(start, 16).unwrap(), [31; 16]);
    assert_eq!(
        engine
            .image()
            .read_exact_at(start + 16, (capacity - 16) as usize)
            .unwrap(),
        vec![0; (capacity - 16) as usize]
    );
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    assert_eq!(
        observations::zero_work(),
        ZeroWork {
            calls: 1,
            bytes: capacity - 16,
            scratch_peak: 65_536
        }
    );
}

#[rstest]
#[case::transition(false)]
#[case::capacity(true)]
fn partial_zero_emission_blocks_later_reservations_and_retains_owned_capacity(
    #[case] capacity: bool,
) {
    let page = 262_144;
    let (_directory, mut engine) = engine_with_page(
        FileSpaceStrategy::Page,
        page,
        FileSpacePageSize::try_from(page).unwrap(),
    );
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.begin_page(PageType::Raw).unwrap();
    let start = engine.image.len();
    let requests = [
        BlockRequest::complete(request(16, PageType::Raw), EncodingDependency::Independent),
        BlockRequest::new(
            request(
                if capacity { page } else { 32 },
                if capacity {
                    PageType::Raw
                } else {
                    PageType::Meta
                },
            ),
            32,
            EncodingDependency::Independent,
        )
        .unwrap(),
        BlockRequest::complete(request(16, PageType::Raw), EncodingDependency::Independent),
    ];
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("planning failed: {:?}", failure.error));
    plan.write_next(&[17; 16]).unwrap();
    disk_log::fail_write_after(if capacity { 2 } else { 1 }, 7);
    let error = plan.write_next(&[23; 32]).unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(plan.next, 1);
    assert!(plan.failed);
    assert_eq!(
        plan.engine.image.len(),
        start + if capacity { 48 } else { 16 } + 65_536
    );
    assert_eq!(
        plan.engine.space.paged_tail_type(),
        Some(Some(PageType::Raw))
    );
    assert_eq!(
        plan.engine.space.pending_padding(PageType::Raw).unwrap(),
        []
    );
    assert_eq!(
        plan.engine.space.pending_padding(PageType::Meta).unwrap(),
        []
    );
    assert_eq!(
        plan.engine
            .image()
            .read_exact_at(start + if capacity { 48 } else { 16 }, 65_536)
            .unwrap(),
        vec![0; 65_536]
    );
    if capacity {
        assert_eq!(
            plan.engine.image().read_exact_at(start + 16, 32).unwrap(),
            [23; 32]
        );
    }
    assert!(plan.end().get() > plan.engine.image.len());
    assert_eq!(plan.reservations().len(), 3);
    let error = plan.write_next(&[31; 16]).unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reservation plan has an earlier failed write"),
        "{error:?}"
    );
    drop(plan);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    assert_eq!(engine.image().read_exact_at(start, 16).unwrap(), [17; 16]);
}

#[rstest]
#[case::small(100)]
#[case::large(4100)]
fn single_block_plans_preserve_complete_geometry_and_ordered_gaps(
    #[case] capacity: u64,
    #[values(false, true)] reused: bool,
    #[values(0, 4096)] userblock: u64,
) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::Page, userblock);
    let physical = engine.image.len();
    let addresses = engine.context.addresses();
    let stored = addresses
        .relative(PhysicalAddress::new(physical))
        .unwrap()
        .get();
    assert_eq!(stored % 4096, 0);
    if reused {
        engine.append(&vec![13; 12_288]).unwrap();
        engine
            .space
            .return_allocation(extent(stored + 4000, 8192), AllocationClass::Metadata);
    } else {
        engine.append(&[13; 4000]).unwrap();
    }
    engine
        .space
        .record_transition(PageType::Meta, PageTransition::Unchanged);
    let requests = [
        BlockRequest::new(
            request(capacity, PageType::Meta).for_block(),
            16,
            EncodingDependency::Allocation,
        )
        .unwrap(),
        BlockRequest::complete(
            request(32, PageType::Meta).for_block(),
            EncodingDependency::Independent,
        ),
    ];
    let second = physical + if reused { 4000 } else { 4096 + capacity };
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("block reservation failed: {:?}", failure.error));
    assert_eq!(
        plan.reservations()[0].range(),
        FileRange::new(PhysicalAddress::new(physical + 4096), capacity).unwrap()
    );
    assert_eq!(
        plan.reservations()[1].range(),
        FileRange::new(PhysicalAddress::new(second), 32).unwrap()
    );
    assert_eq!(
        plan.engine.space.pending_padding(PageType::Meta).unwrap(),
        []
    );
    assert_eq!(
        plan.write_next(&[17; 16]).unwrap(),
        PhysicalAddress::new(physical + 4096)
    );
    assert_eq!(
        plan.engine.space.pending_padding(PageType::Meta).unwrap(),
        if reused {
            Vec::new()
        } else {
            vec![extent(stored + 4000, 96)]
        }
    );
    assert_eq!(
        plan.write_next(&[23; 32]).unwrap(),
        PhysicalAddress::new(second)
    );
    plan.finish().unwrap();
    assert_eq!(
        engine.image().read_exact_at(physical + 4096, 16).unwrap(),
        [17; 16]
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(physical + 4112, usize::try_from(capacity - 16).unwrap())
            .unwrap(),
        vec![if reused { 13 } else { 0 }; usize::try_from(capacity - 16).unwrap()]
    );
    assert_eq!(engine.image().read_exact_at(second, 32).unwrap(), [23; 32]);
    if reused {
        assert_eq!(
            engine.space.reusable_sections().collect::<Vec<_>>(),
            [
                extent(stored + 4032, 64),
                extent(stored + 4096 + capacity, 8096 - capacity)
            ]
        );
    } else {
        assert_eq!(
            engine.image().read_exact_at(physical + 4000, 96).unwrap(),
            [0; 96]
        );
        assert_eq!(engine.image.len(), physical + 4128 + capacity);
    }
}

#[rstest]
#[case::stored_alignment(0, u64::MAX - 100, 102, u64::MAX - 100, 101)]
#[case::physical_start(512, u64::MAX - 100, 50, u64::MAX - 100, 512)]
#[case::physical_end(512, u64::MAX - 4095, 4000, u64::MAX - 3583, 4000)]
fn failed_block_geometry_preserves_the_append_cursor(
    #[case] base: u64,
    #[case] start: u64,
    #[case] capacity: u64,
    #[case] offset: u64,
    #[case] length: u64,
) {
    let mut cursor = AppendCursor {
        offsets: OffsetWidth::Eight,
        addresses: AddressContext::from_effective_base(BaseAddress::new(base)),
        cursor: StoredAddress::new(start),
        page_size: Some(FileSpacePageSize::DEFAULT),
        tail_type: Some(PageType::Meta),
        fresh_boundary: None,
    };
    assert_eq!(
        cursor
            .reserve(request(capacity, PageType::Meta).for_block())
            .unwrap_err(),
        FormatError::OffsetOverflow { offset, length }
    );
    assert_eq!(cursor.cursor, StoredAddress::new(start));
    assert_eq!(cursor.tail_type, Some(PageType::Meta));
}

fn planned<'a>(
    engine: &'a mut WriteEngine,
    requests: &[ReservationRequest],
) -> ReservationPlan<'a> {
    engine
        .plan_reservations(requests)
        .unwrap_or_else(|failure| panic!("reservation failed: {:?}", failure.error))
}

fn request(length: u64, page_type: PageType) -> ReservationRequest {
    let class = match page_type {
        PageType::Meta => AllocationClass::Metadata,
        PageType::Raw => AllocationClass::Raw,
    };
    ReservationRequest::with_page_type(NonZeroByteLength::new(length).unwrap(), class, page_type)
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), length).unwrap()
}

fn flat(extents: &[(u64, u64)]) -> SessionSpace {
    SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        1,
        FileSpacePageSize::DEFAULT,
        extents
            .iter()
            .map(|&(start, length)| (None, extent(start, length))),
    )
    .unwrap()
}
