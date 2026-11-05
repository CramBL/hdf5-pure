use std::io::ErrorKind;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::AddressRange;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::SlotOccupancy;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::ReservationRequest;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Recording;
use test_util::extensible_array as extensible_array_fixture;

use super::ArrayIndexPlan;
use crate::File;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::MemoryStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_write::ChunkArrayKind;
use crate::dataspace::MaxExtent;
use crate::edit::AppendBuilder;
use crate::edit::WriteEngine;
use crate::edit::block::BlockRequest;
use crate::edit::block::EncodingDependency;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::extensible_array;
use crate::fixed_array;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::MetadataCacheConfig;
use crate::source::PhysicalSource;
use crate::width::FormatWidths;
use crate::writer::FileBuilder;

#[rstest]
#[case::fixed_empty(ChunkArrayKind::FixedArray, 0, &[])]
#[case::fixed_small(ChunkArrayKind::FixedArray, 8, &[0, 7])]
#[case::fixed_paged(ChunkArrayKind::FixedArray, 1_025, &[0, 1_024])]
#[case::extensible_empty(ChunkArrayKind::ExtensibleArray, 0, &[])]
#[case::inline(ChunkArrayKind::ExtensibleArray, 4, &[0, 3])]
#[case::direct(ChunkArrayKind::ExtensibleArray, 32, &[0, 31])]
#[case::secondary(ChunkArrayKind::ExtensibleArray, 3_000, &[2_999])]
#[case::paged_sparse(ChunkArrayKind::ExtensibleArray, 1_000_001, &[1_000_000])]
fn array_blocks_follow_complete_metadata_reservations(
    #[case] kind: ChunkArrayKind,
    #[case] count: u64,
    #[case] occupied: &[u64],
    #[values(512, 4_096)] page: u64,
    #[values(0, 4_096)] base: u64,
    #[values(false, true)] filtered: bool,
) {
    let (_directory, mut engine) = engine(page, base);
    let records: Vec<_> = occupied
        .iter()
        .map(|&slot| ChunkRecord {
            address: StoredAddress::new(64 + slot % 16),
            stored_size: if filtered { 7 } else { 8 },
            filter_mask: u32::from(filtered),
        })
        .collect();
    let slots = IndexSlots::new(&records, occupied, count).unwrap();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let index = ArrayIndexPlan::new(
        kind,
        SlotOccupancy::Sparse(&slots),
        count,
        8,
        widths,
        filtered,
    )
    .unwrap();
    let requests = index.requests().unwrap();
    assert!(requests.len() >= 2);
    assert!(requests.iter().all(|request| {
        let reservation = request.reservation();
        reservation.is_block()
            && reservation.class() == AllocationClass::Metadata
            && reservation.page_type() == PageType::Meta
            && request.encoded_length() == reservation.size().get()
            && request.dependency() == EncodingDependency::Reservations
    }));
    let addresses = engine.context.addresses();
    let mut reservations = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let expected: Vec<_> = reservations
        .reservations()
        .iter()
        .map(|reservation| {
            let range = reservation.range();
            let start = addresses.relative(range.start()).unwrap();
            if range.length() < page {
                assert_eq!(
                    start.get() / page,
                    (start.get() + range.length() - 1) / page
                );
            } else {
                assert_eq!(start.get() % page, 0);
            }
            AddressRange::new(start, range.length()).unwrap()
        })
        .collect();
    let header = index.write(&slots, &mut reservations).unwrap();
    assert_eq!(header, expected[0].start());
    assert_eq!(reservations.pending().len(), 0);
    reservations.finish().unwrap();
    let opened = engine.opened_source();
    let physical = PhysicalSource::new(&opened);
    let source = physical.stored();
    let mut actual = match kind {
        ChunkArrayKind::FixedArray => fixed_array::index_spans(&source, widths, header),
        ChunkArrayKind::ExtensibleArray => extensible_array::index_spans(&source, widths, header),
    }
    .unwrap();
    let mut expected = expected;
    actual.sort_unstable_by_key(|range| range.start());
    expected.sort_unstable_by_key(|range| range.start());
    assert_eq!(actual, expected);
    let mut actual = Vec::new();
    match kind {
        ChunkArrayKind::FixedArray => {
            fixed_array::read_chunks(&source, widths, header, 8, |slot, record| {
                actual.push((slot, record));
                Ok(())
            })
        }
        ChunkArrayKind::ExtensibleArray => {
            extensible_array::read_chunks(&source, widths, header, 8, |slot, record| {
                actual.push((slot, record));
                Ok(())
            })
        }
    }
    .unwrap();
    assert_eq!(
        actual,
        occupied.iter().copied().zip(records).collect::<Vec<_>>()
    );
}

#[rstest]
#[case::missing(Mismatch::Missing)]
#[case::class(Mismatch::Class)]
#[case::compound(Mismatch::Compound)]
#[case::dependency(Mismatch::Dependency)]
#[case::encoding(Mismatch::Encoding)]
fn array_emission_rejects_incompatible_reservations(#[case] mismatch: Mismatch) {
    let (_directory, mut engine) = engine(512, 0);
    let slots = IndexSlots::dense(&[]);
    let index = ArrayIndexPlan::new(
        ChunkArrayKind::ExtensibleArray,
        SlotOccupancy::Dense(0),
        0,
        8,
        FormatWidths::from_sizes(8, 8).unwrap(),
        false,
    )
    .unwrap();
    let mut requests = index.requests().unwrap();
    match mismatch {
        Mismatch::Missing => {
            requests.pop();
        }
        Mismatch::Class | Mismatch::Compound | Mismatch::Dependency | Mismatch::Encoding => {
            let original = requests[0].reservation();
            let reservation = if matches!(mismatch, Mismatch::Class) {
                ReservationRequest::new(original.size(), AllocationClass::Raw).for_block()
            } else if matches!(mismatch, Mismatch::Compound) {
                ReservationRequest::new(original.size(), AllocationClass::Metadata)
            } else {
                original
            };
            let dependency = if matches!(mismatch, Mismatch::Dependency) {
                EncodingDependency::Independent
            } else {
                EncodingDependency::Reservations
            };
            requests[0] = if matches!(mismatch, Mismatch::Encoding) {
                BlockRequest::new(reservation, reservation.size().get() - 1, dependency).unwrap()
            } else {
                BlockRequest::complete(reservation, dependency)
            };
        }
    }
    let prior = engine.image.len();
    let mut reservations = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let owned: Vec<_> = reservations
        .reservations()
        .iter()
        .map(|reservation| reservation.range())
        .collect();
    let error = index.write(&slots, &mut reservations).unwrap_err();
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        reason,
        if matches!(mismatch, Mismatch::Missing) {
            "array index reservations omit complete blocks"
        } else {
            "an array index reservation differs from its block request"
        }
    );
    assert_eq!(reservations.pending().len(), requests.len());
    assert_eq!(
        owned,
        reservations
            .reservations()
            .iter()
            .map(|reservation| reservation.range())
            .collect::<Vec<_>>()
    );
    drop(reservations);
    assert_eq!(engine.image.len(), prior);
}

#[rstest]
#[case::fixed(ChunkArrayKind::FixedArray)]
#[case::extensible(ChunkArrayKind::ExtensibleArray)]
fn array_emission_follows_preceding_payload_reservations(#[case] kind: ChunkArrayKind) {
    let (_directory, mut engine) = engine(512, 0);
    let records = [ChunkRecord {
        address: StoredAddress::new(64),
        stored_size: 8,
        filter_mask: 0,
    }];
    let slots = IndexSlots::dense(&records);
    let index = ArrayIndexPlan::new(
        kind,
        SlotOccupancy::Dense(1),
        1,
        8,
        FormatWidths::from_sizes(8, 8).unwrap(),
        false,
    )
    .unwrap();
    let mut requests = vec![BlockRequest::complete(
        ReservationRequest::new(NonZeroByteLength::new(8).unwrap(), AllocationClass::Raw),
        EncodingDependency::Independent,
    )];
    requests.extend(index.requests().unwrap());
    let addresses = engine.context.addresses();
    let mut reservations = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let expected = addresses
        .relative(reservations.reservations()[1].range().start())
        .unwrap();
    reservations.write_next(&[7; 8]).unwrap();
    assert_eq!(index.write(&slots, &mut reservations).unwrap(), expected);
    assert!(reservations.pending().is_empty());
    reservations.finish().unwrap();
}

#[test]
fn partial_array_emission_keeps_every_reserved_allocation_unavailable() {
    let (_directory, mut engine) = engine(512, 0);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let slots = IndexSlots::dense(&[]);
    let index = ArrayIndexPlan::new(
        ChunkArrayKind::ExtensibleArray,
        SlotOccupancy::Dense(0),
        0,
        8,
        FormatWidths::from_sizes(8, 8).unwrap(),
        false,
    )
    .unwrap();
    let requests = index.requests().unwrap();
    let lengths: Vec<_> = requests
        .iter()
        .map(|request| request.reservation().size().get())
        .collect();
    assert_eq!(lengths.len(), 2);
    let start = engine.append(&[13; 512]).unwrap();
    let addresses = engine.context.addresses();
    let stored = addresses.relative(PhysicalAddress::new(start)).unwrap();
    let second = stored.checked_add(lengths[0] + 16).unwrap();
    assert!(lengths[0] + 16 + lengths[1] <= 512);
    let original = [
        Extent::new(stored, ByteLength::new(lengths[0])).unwrap(),
        Extent::new(second, ByteLength::new(lengths[1])).unwrap(),
    ];
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::MIN,
        original.map(|extent| (Some(ManagerKind::PagedSmallMetadata), extent)),
    )
    .unwrap();
    let rollback = engine.snapshot_free();
    let mut reservations = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    disk_log::fail_write_after(1, 0);
    let error = index.write(&slots, &mut reservations).unwrap_err();
    let Error::Io(error) = error else {
        panic!("{error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(reservations.pending().len(), 1);
    drop(reservations);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    engine.restore_free(rollback);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        original
    );
    assert_eq!(
        &engine
            .image()
            .read_exact_at(start, extensible_array_fixture::SIGNATURE.len())
            .unwrap()[..],
        extensible_array_fixture::SIGNATURE
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(
                addresses.resolve(second).unwrap().get(),
                usize::try_from(lengths[1]).unwrap()
            )
            .unwrap(),
        vec![13; usize::try_from(lengths[1]).unwrap()]
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(512)]
#[case::many(4_096)]
fn extensible_array_emission_keeps_one_block_buffer(#[case] count: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("array-working-set.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::MIN),
    );
    builder.create_dataset("d").with_i32_data(&[1; 16]);
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let records = vec![
        ChunkRecord {
            address: StoredAddress::new(64),
            stored_size: 8,
            filter_mask: 0
        };
        count
    ];
    let slots = IndexSlots::dense(&records);
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let index = ArrayIndexPlan::new(
        ChunkArrayKind::ExtensibleArray,
        SlotOccupancy::Dense(u64::try_from(count).unwrap()),
        u64::try_from(count).unwrap(),
        8,
        widths,
        false,
    )
    .unwrap();
    let requests = index.requests().unwrap();
    let encoded: u64 = requests
        .iter()
        .map(|request| request.encoded_length())
        .sum();
    let largest = requests
        .iter()
        .map(|request| request.encoded_length())
        .max()
        .unwrap();
    assert!(
        encoded > 3 * largest,
        "encoded={encoded}, largest={largest}"
    );
    let addresses = engine.context.addresses();
    let mut reservations = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let (header, measured) = allocation::measure(&format!("array_block_encoding_{count}"), || {
        index.write(&slots, &mut reservations).unwrap()
    });
    assert_eq!(
        header,
        addresses
            .relative(reservations.reservations()[0].range().start())
            .unwrap()
    );
    reservations.finish().unwrap();
    let (control, copied) =
        allocation::measure(&format!("array_contiguous_encoding_{count}"), || {
            hdf5_pure_format::__private::build_extensible_array_at(
                &slots,
                8,
                widths.offsets,
                widths.lengths,
                false,
                StoredAddress::new(0),
            )
            .unwrap()
        });
    assert_eq!(u64::try_from(control.len()).unwrap(), encoded);
    assert!(copied.live_bytes >= encoded, "{copied:?}");
    assert!(
        measured.peak_bytes < encoded,
        "{measured:?}, encoded={encoded}, largest={largest}"
    );
    assert!(
        measured.peak_bytes <= 4 * largest + 256 * u64::try_from(requests.len()).unwrap(),
        "{measured:?}"
    );
}

#[rstest]
#[case::fixed(ChunkArrayKind::FixedArray)]
#[case::extensible(ChunkArrayKind::ExtensibleArray)]
fn array_emission_rejects_incompatible_slot_tables(#[case] kind: ChunkArrayKind) {
    let (_directory, mut engine) = engine(512, 0);
    let index = ArrayIndexPlan::new(
        kind,
        SlotOccupancy::Dense(0),
        32,
        8,
        FormatWidths::from_sizes(8, 8).unwrap(),
        false,
    )
    .unwrap();
    let records = [ChunkRecord {
        address: StoredAddress::new(64),
        stored_size: 8,
        filter_mask: 0,
    }];
    let slots = match kind {
        ChunkArrayKind::FixedArray => IndexSlots::dense(&records),
        ChunkArrayKind::ExtensibleArray => IndexSlots::new(&records, &[31], 32).unwrap(),
    };
    let requests = index.requests().unwrap();
    let mut reservations = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let error = index.write(&slots, &mut reservations).unwrap_err();
    match (kind, error) {
        (
            ChunkArrayKind::FixedArray,
            Error::Format(FormatError::FixedArrayEntryCount { expected, actual }),
        ) => {
            assert_eq!(expected, 32);
            assert_eq!(actual, 1);
        }
        (
            ChunkArrayKind::ExtensibleArray,
            Error::Format(FormatError::ExtensibleArrayUnplannedSlot { slot }),
        ) => assert_eq!(slot, 31),
        (_, error) => panic!("{error:?}"),
    }
    assert_eq!(reservations.pending().len(), requests.len());
}

#[rstest]
#[case::payload(0)]
#[case::metadata(1)]
fn failed_staged_array_writes_restore_reusable_state_and_predecessor_values(#[case] writes: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged-array-failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1)
            .with_page_size(FileSpacePageSize::MIN),
    );
    builder
        .create_dataset("d")
        .with_i32_data(&[1; 16])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[16]);
    builder.create_dataset("hole").with_i32_data(&[7; 1_024]);
    builder.create_dataset("ceiling").with_i32_data(&[9; 16]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.sync_policy = SyncPolicy::OnClose;
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    let before = engine
        .space
        .admitted_snapshot()
        .sections()
        .collect::<Vec<_>>();
    assert!(
        engine
            .space
            .reusable_sections()
            .any(|extent| extent.len() >= ByteLength::new(64))
    );
    let mut append = AppendBuilder::new();
    append.append_i32(&[2; 16]);
    engine
        .stage_dataset_append(&ObjectPathBuf::parse("d"), append)
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(writes, 0);
    let error = engine.commit().unwrap_err();
    let Recording { operations, events } = disk_log::finish();
    let failed: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                requested,
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            } => Some(*requested),
            _ => None,
        })
        .collect();
    let expected = if writes == 0 {
        16 * size_of::<i32>()
    } else {
        usize::try_from(
            ArrayIndexPlan::new(
                ChunkArrayKind::ExtensibleArray,
                SlotOccupancy::Dense(2),
                2,
                64,
                FormatWidths::from_sizes(8, 8).unwrap(),
                false,
            )
            .unwrap()
            .requests()
            .unwrap()[0]
                .encoded_length(),
        )
        .unwrap()
    };
    assert_eq!(failed, [expected]);
    if writes > 0 {
        let payload: Vec<u8> = core::iter::repeat_n(2_i32, 16)
            .flat_map(i32::to_le_bytes)
            .collect();
        assert!(operations.iter().any(
            |operation| matches!(operation, DiskOp::Write { bytes, .. } if bytes == &payload)
        ));
    }
    let Error::Io(error) = error else {
        panic!("{error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        before
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [1; 16]);
    assert_eq!(
        file.dataset("ceiling").unwrap().read_i32().unwrap(),
        [9; 16]
    );
    file.close().unwrap();
}

#[derive(Clone, Copy)]
enum Mismatch {
    Class,
    Compound,
    Dependency,
    Encoding,
    Missing,
}

fn engine(page: u64, base: u64) -> (tempfile::TempDir, WriteEngine) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("array-blocks.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1)
            .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
    );
    builder.create_dataset("d").with_i32_data(&[1; 16]);
    builder.write(&path).unwrap();
    (directory, edit_test_util::open_engine(&path))
}
