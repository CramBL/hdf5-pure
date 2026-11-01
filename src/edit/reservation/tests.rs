use core::num::NonZeroU64;
use std::io::ErrorKind;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{
    AllocationClass, Extent, PageTransition, PageType, ReservationRequest, SessionSpace,
};
use rstest::rstest;
use test_util::disk_log;

use crate::address::{
    AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
};
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::image::WriteBuffering;
use crate::source::Source;
use crate::{FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy, UserblockSize};

use super::{AppendCursor, ReservationFailure, ReservationPlan};

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
        addresses: AddressContext::from_effective_base(BaseAddress::new(base)),
        cursor: StoredAddress::new(start),
        page_size: Some(FileSpacePageSize::DEFAULT),
        tail_type: Some(PageType::Raw),
        fresh_boundary: None,
    };
    let (raw, transition) = cursor.reserve(request(32, PageType::Raw)).unwrap();
    assert_eq!(
        raw,
        FileRange::new(PhysicalAddress::new(base + start), 32).unwrap()
    );
    assert_eq!(transition, Some(PageTransition::Unchanged));
    let (metadata, transition) = cursor.reserve(request(64, PageType::Meta)).unwrap();
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

fn engine(strategy: FileSpaceStrategy, userblock: u64) -> (tempfile::TempDir, WriteEngine) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reservations.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy).with_persistence(strategy == FileSpaceStrategy::Page),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    if strategy == FileSpaceStrategy::Page {
        engine.space = SessionSpace::from_strategy(Some(strategy), 1, FileSpacePageSize::DEFAULT);
    }
    (directory, engine)
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
    ReservationRequest::with_page_type(NonZeroU64::new(length).unwrap(), class, page_type)
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
