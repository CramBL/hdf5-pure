use std::fs;

use hdf5_pure_core::__private::{ByteLength, FileRange};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{Extent, FreeClass, ManagerKind, ReturnedExtents, SessionSpace};
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::IoEvent;

use super::super::storage::PublicationError;
use crate::address::{DecodedStoredAddress, StoredAddress, StoredAddressExt};
use crate::edit::commit::plan::inventory::storage::{OriginalStorage, SpaceClaim, StorageConflict};
use crate::edit::commit::{PublicationPurpose, PublicationState};
use crate::edit::object_header;
use crate::edit::paged_persistence::PagedPersistencePlan;
use crate::edit::test_util as edit_test_util;
use crate::edit::{TailPlacement, WriteEngine};
use crate::file_space_info::{self, NUM_FILE_FSM_MANAGERS};
use crate::image::WriteBuffering;
use crate::{FileBuilder, FileSpaceSettings, FileSpaceStrategy, UserblockSize};

#[rstest]
#[case::legacy("c/1.8/v1_superblock.h5", 0)]
#[case::modern("c/1.8/v2_superblock.h5", 0)]
#[case::modern_userblock("c/1.8/v2_superblock.h5", 512)]
fn nonpersistent_end_preserves_protected_storage_before_publication(
    #[case] fixture: &str,
    #[case] base: usize,
    #[values(false, true)] short: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = test_util::copy_data(fixture, directory.path());
    if base != 0 {
        test_util_hdf5::userblock::prepend(&path, &vec![0xA5; base]);
    }
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let root = root(&engine);
    let required = PhysicalAddress::new(engine.image.len());
    let storage = protection(required.get() - 1, 1);
    let proposed = PhysicalAddress::new(required.get() - u64::from(short));
    let before = engine.disk_bytes().unwrap();
    disk_log::start();
    let result = engine.publish_nonpersisting_excluding(root, Some(proposed), Some(&storage));
    let log = disk_log::finish();
    if short {
        assert_end(result.unwrap_err(), proposed, required);
        assert_no_writes(&log.events);
        assert_eq!(engine.disk_bytes().unwrap(), before);
        assert_eq!(engine.publication, PublicationState::Ready);
    } else {
        result.unwrap();
        assert_eq!(engine.superblock().eof_address, required.get());
        assert_eq!(engine.image.len(), required.get());
        assert_eq!(engine.publication, PublicationState::Published);
    }
}

#[rstest]
#[case::returned(Conflict::Returned)]
#[case::old_manager(Conflict::OldManager)]
#[case::recycle(Conflict::Recycle)]
#[case::reusable(Conflict::Reusable)]
fn persistent_inputs_reject_protected_ranges_before_content_writes(
    #[case] conflict: Conflict,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("inputs.h5");
    let mut engine = fixture(&path, strategy);
    let root = root(&engine);
    let protected = Extent::new(engine.end_of_allocation().unwrap(), ByteLength::new(32)).unwrap();
    engine.append(&[0xA5; 32]).unwrap();
    let storage = protection(
        engine.addresses().resolve(protected.start()).unwrap().get() + 7,
        1,
    );
    let mut returned = ReturnedExtents::default();
    let mut placement = TailPlacement::Anywhere;
    let claim = match conflict {
        Conflict::Returned => {
            returned.insert(protected, FreeClass::Dead).unwrap();
            SpaceClaim::Retirement(FreeClass::Dead)
        }
        Conflict::OldManager => {
            engine.persist.as_mut().unwrap().old_blocks.push(protected);
            SpaceClaim::OldManager
        }
        Conflict::Recycle => {
            placement = TailPlacement::Recycle(protected);
            SpaceClaim::Recycle
        }
        Conflict::Reusable => {
            engine.space = SessionSpace::from_persisted(
                Some(strategy),
                ByteLength::new(1),
                crate::FileSpacePageSize::default(),
                [(
                    Some(if strategy == FileSpaceStrategy::Page {
                        ManagerKind::PagedSmallMetadata
                    } else {
                        ManagerKind::Flat
                    }),
                    protected,
                )],
            )
            .unwrap();
            SpaceClaim::Reusable
        }
    };
    let publication = if matches!(conflict, Conflict::OldManager) {
        PublicationState::Published
    } else {
        PublicationState::Ready
    };
    engine.publication = publication;
    let before = engine.disk_bytes().unwrap();
    let old_blocks = engine.persist.as_ref().unwrap().old_blocks.clone();
    let reusable = engine.space.reusable_sections().collect::<Vec<_>>();
    let retained = engine.space.retained_extents().collect::<Vec<_>>();
    disk_log::start();
    let error = engine
        .publish_persisting_excluding(
            root,
            returned,
            placement,
            PublicationPurpose::Commit,
            Some(&storage),
        )
        .unwrap_err();
    let log = disk_log::finish();
    let PublicationError::Storage(StorageConflict::Overlap {
        range,
        claim: actual,
        ..
    }) = error
    else {
        panic!("expected a protected range conflict, got {error:?}");
    };
    assert_eq!(actual, claim);
    assert_eq!(
        range.start(),
        engine.addresses().resolve(protected.start()).unwrap()
    );
    assert_eq!(range.length(), protected.len().get());
    assert_no_writes(&log.events);
    assert_eq!(engine.persist.as_ref().unwrap().old_blocks, old_blocks);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        reusable
    );
    assert_eq!(
        engine.space.retained_extents().collect::<Vec<_>>(),
        retained
    );
    assert_eq!(engine.publication, publication);
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[rstest]
#[case::flat(Capacity::Flat)]
#[case::page_reservation(Capacity::PageReservation)]
#[case::page_suffix(Capacity::PageSuffix)]
#[case::page_reused(Capacity::PageReused)]
fn persistent_reservation_checks_complete_capacity_before_writing(#[case] capacity: Capacity) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("capacity.h5");
    let strategy = if matches!(capacity, Capacity::Flat) {
        FileSpaceStrategy::FsmAggr
    } else {
        FileSpaceStrategy::Page
    };
    let mut probe = fixture(&path, strategy);
    let reused =
        matches!(capacity, Capacity::PageReused).then(|| reusable_capacity(&mut probe, strategy));
    let returned_extent =
        Extent::new(probe.end_of_allocation().unwrap(), ByteLength::new(32)).unwrap();
    probe.append(&[0xA5; 4096]).unwrap();
    let returned = || {
        let mut returned = ReturnedExtents::default();
        returned.insert(returned_extent, FreeClass::Dead).unwrap();
        returned
    };
    let planned =
        (strategy == FileSpaceStrategy::Page).then(|| paged_capacity(&probe, &returned()));
    let original = probe.disk_bytes().unwrap();
    let root = root(&probe);
    probe
        .publish_persisting(
            root,
            returned(),
            TailPlacement::Anywhere,
            PublicationPurpose::Allocation,
        )
        .unwrap();
    let extension = probe
        .addresses()
        .resolve(probe.superblock().superblock_extension_address.unwrap())
        .unwrap();
    let encoded = probe
        .oh_chunk_spans(extension)
        .unwrap()
        .iter()
        .map(|range| range.length())
        .sum::<u64>()
        + crate::free_space_manager::ManagerProjection::new(
            probe.space.admitted_snapshot(),
            probe.widths(),
        )
        .unwrap()
        .encoded_len();
    let end = PhysicalAddress::new(probe.image.len());
    assert!(end.get() - extension.get() > encoded);
    let (protected, expected_range, expected_claim) = match (capacity, planned) {
        (Capacity::Flat, None) => {
            assert!(extension.get() >= u64::try_from(original.len()).unwrap());
            (
                PhysicalAddress::new(end.get() - 1),
                None,
                SpaceClaim::Metadata,
            )
        }
        (Capacity::PageSuffix, Some(planned)) => {
            assert!(planned.reservation.end() < end);
            (
                PhysicalAddress::new(end.get() - 1),
                None,
                SpaceClaim::Metadata,
            )
        }
        (Capacity::PageReservation | Capacity::PageReused, Some(planned)) => {
            assert_eq!(probe.oh_chunk_spans(extension).unwrap()[0], planned.root);
            assert!(
                planned
                    .reservation
                    .contains(FileRange::new(planned.unused, 1).unwrap())
            );
            assert!(planned.unused < planned.reservation.end());
            assert!(planned.reservation.end() < end);
            let (range, claim) = if let Some(reused) = reused {
                let range = FileRange::new(
                    probe.addresses().resolve(reused.start()).unwrap(),
                    reused.len().get(),
                )
                .unwrap();
                assert!(range.contains(planned.reservation));
                (range, SpaceClaim::Reusable)
            } else {
                (planned.reservation, SpaceClaim::Metadata)
            };
            (planned.unused, Some(range), claim)
        }
        _ => panic!("a capacity case has the matching layout"),
    };
    drop(probe);
    fs::write(&path, &original).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.aggregation = None;
    engine.space = empty_space(strategy);
    if let Some(reused) = reused {
        engine.space = SessionSpace::from_persisted(
            Some(strategy),
            ByteLength::new(1),
            crate::FileSpacePageSize::default(),
            [(Some(ManagerKind::PagedLargeGeneric), reused)],
        )
        .unwrap();
    }
    let storage = protection(protected.get(), 1);
    let old_blocks = engine.persist.as_ref().unwrap().old_blocks.clone();
    let reusable = engine.space.reusable_sections().collect::<Vec<_>>();
    let retained = engine.space.retained_extents().collect::<Vec<_>>();
    disk_log::start();
    let error = engine
        .publish_persisting_excluding(
            root,
            returned(),
            TailPlacement::Anywhere,
            PublicationPurpose::Allocation,
            Some(&storage),
        )
        .unwrap_err();
    let log = disk_log::finish();
    let PublicationError::Storage(StorageConflict::Overlap { range, claim, .. }) = error else {
        panic!("expected a complete metadata conflict, got {error:?}");
    };
    assert_eq!(claim, expected_claim);
    if let Some(expected) = expected_range {
        assert_eq!(range, expected);
    } else {
        assert!(range.end() >= end);
    }
    assert_no_writes(&log.events);
    assert_eq!(engine.persist.as_ref().unwrap().old_blocks, old_blocks);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        reusable
    );
    assert_eq!(
        engine.space.retained_extents().collect::<Vec<_>>(),
        retained
    );
    assert_eq!(engine.publication, PublicationState::Ready);
    assert_eq!(engine.disk_bytes().unwrap(), original);
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn persistent_proposed_end_checks_the_maximum_protected_end(
    #[case] strategy: FileSpaceStrategy,
    #[values(false, true)] short: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("end.h5");
    let mut engine = fixture(&path, strategy);
    let capacity = reusable_capacity(&mut engine, strategy);
    let root = root(&engine);
    let proposed = PhysicalAddress::new(engine.image.len());
    let original = engine.disk_bytes().unwrap();
    let required = PhysicalAddress::new(proposed.get() + u64::from(short));
    let storage = protection(required.get() - 1, 1);
    let snapshot = engine.snapshot_free();
    let old_blocks = engine.persist.as_ref().unwrap().old_blocks.clone();
    disk_log::start();
    let result = engine.publish_persisting_excluding(
        root,
        ReturnedExtents::default(),
        TailPlacement::Anywhere,
        PublicationPurpose::Commit,
        Some(&storage),
    );
    let log = disk_log::finish();
    if short {
        assert_end(result.unwrap_err(), proposed, required);
        assert_no_writes(&log.events);
        assert_eq!(engine.disk_bytes().unwrap(), original);
        assert_eq!(engine.publication, PublicationState::Ready);
        assert_eq!(engine.persist.as_ref().unwrap().old_blocks, old_blocks);
        let remaining = engine.space.reusable_sections().collect::<Vec<_>>();
        if strategy == FileSpaceStrategy::Page {
            assert_eq!(remaining, [capacity]);
        } else {
            assert!(
                remaining
                    .iter()
                    .all(|extent| extent.start() > capacity.start())
            );
        }
        engine.restore_free(snapshot);
        assert_eq!(
            engine.space.reusable_sections().collect::<Vec<_>>(),
            [capacity]
        );
        engine
            .publish_persisting_excluding(
                root,
                ReturnedExtents::default(),
                TailPlacement::Anywhere,
                PublicationPurpose::Commit,
                Some(&protection(proposed.get() - 1, 1)),
            )
            .unwrap();
        assert_eq!(
            engine
                .addresses()
                .resolve(engine.superblock().superblock_extension_address.unwrap())
                .unwrap(),
            engine.addresses().resolve(capacity.start()).unwrap()
        );
    } else {
        result.unwrap();
        assert_eq!(engine.superblock().eof_address, required.get());
        let extension = engine
            .addresses()
            .resolve(engine.superblock().superblock_extension_address.unwrap())
            .unwrap();
        assert_eq!(
            extension,
            engine.addresses().resolve(capacity.start()).unwrap()
        );
        assert_eq!(
            engine.image().read_exact_at(proposed.get() - 1, 1).unwrap(),
            [0xA5]
        );
    }
}

#[rstest]
#[case::suffix_conflict(true)]
#[case::adjacent(false)]
fn recycled_empty_manager_protects_complete_target_before_suffix_shrink(#[case] conflict: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("recycle.h5");
    let mut engine = fixture(&path, FileSpaceStrategy::FsmAggr);
    let target = Extent::new(engine.end_of_allocation().unwrap(), ByteLength::new(4096)).unwrap();
    engine.append(&[0xA5; 4096]).unwrap();
    engine.persist.as_mut().unwrap().old_blocks.clear();
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        ByteLength::new(1),
        crate::FileSpacePageSize::MIN,
        [],
    )
    .unwrap();
    let target_start = engine.addresses().resolve(target.start()).unwrap();
    let protected = if conflict {
        engine.image.len() - 1
    } else {
        target_start.get() - 1
    };
    let storage = protection(protected, 1);
    let root = root(&engine);
    let original = engine.disk_bytes().unwrap();
    let old_blocks = engine.persist.as_ref().unwrap().old_blocks.clone();
    let reusable = engine.space.reusable_sections().collect::<Vec<_>>();
    let retained = engine.space.retained_extents().collect::<Vec<_>>();
    disk_log::start();
    let result = engine.publish_persisting_excluding(
        root,
        ReturnedExtents::default(),
        TailPlacement::Recycle(target),
        PublicationPurpose::Commit,
        Some(&storage),
    );
    let log = disk_log::finish();
    if conflict {
        let error = result.unwrap_err();
        assert!(
            matches!(error, PublicationError::Storage(StorageConflict::Overlap { claim: SpaceClaim::Recycle, range, .. }) if range.start() == target_start && range.length() == target.len().get()),
            "{error:?}"
        );
        assert_no_writes(&log.events);
        assert_eq!(engine.persist.as_ref().unwrap().old_blocks, old_blocks);
        assert_eq!(
            engine.space.reusable_sections().collect::<Vec<_>>(),
            reusable
        );
        assert_eq!(
            engine.space.retained_extents().collect::<Vec<_>>(),
            retained
        );
        assert_eq!(engine.publication, PublicationState::Ready);
        assert_eq!(engine.disk_bytes().unwrap(), original);
    } else {
        result.unwrap();
        assert!(engine.image.len() < target_start.get() + target.len().get());
        assert_eq!(engine.superblock().eof_address, engine.image.len());
        assert_eq!(
            engine.image().read_exact_at(protected, 1).unwrap(),
            [original[usize::try_from(protected).unwrap()]]
        );
    }
}

fn paged_capacity(engine: &WriteEngine, returned: &ReturnedExtents) -> PagedCapacity {
    let persist = engine.persist.as_ref().unwrap();
    let info = file_space_info::persistent_managers(
        persist.strategy,
        persist.threshold,
        persist.page_size,
        [DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS],
        StoredAddress::ZERO,
    );
    let source = engine.image();
    let predecessor = object_header::ExtensionReconstruction::read(
        source,
        engine
            .addresses()
            .resolve(engine.superblock().superblock_extension_address.unwrap())
            .unwrap(),
        engine.addresses(),
        engine.widths(),
    )
    .unwrap();
    let prepared =
        object_header::PreparedHeader::new(&predecessor.rewrite(&info).unwrap()).unwrap();
    let geometry = prepared.geometry().unwrap();
    let minimum = geometry
        .root_chunk_lengths(false)
        .map(|range| *range.start())
        .min()
        .unwrap();
    let plan = PagedPersistencePlan::new(
        &engine.space,
        engine.end_of_allocation().unwrap(),
        returned.entries(),
        &persist.old_blocks,
        &geometry,
        false,
    )
    .unwrap()
    .unwrap();
    let (_, extension, _, _, coverage) = plan.into_parts();
    let root = extension[0];
    let checksum_len = u64::try_from(test_util::checksum::CHECKSUM).unwrap();
    assert!(root.len().get() > minimum + checksum_len);
    let root = FileRange::new(
        engine.addresses().resolve(root.start()).unwrap(),
        root.len().get(),
    )
    .unwrap();
    let reservation = coverage.reservation();
    PagedCapacity {
        reservation: FileRange::new(
            engine.addresses().resolve(reservation.start()).unwrap(),
            reservation.len().get(),
        )
        .unwrap(),
        root,
        unused: PhysicalAddress::new(root.end().get() - checksum_len - 1),
    }
}

struct PagedCapacity {
    reservation: FileRange,
    root: FileRange,
    unused: PhysicalAddress,
}

#[derive(Clone, Copy)]
enum Capacity {
    Flat,
    PageReservation,
    PageReused,
    PageSuffix,
}

fn reusable_capacity(engine: &mut WriteEngine, strategy: FileSpaceStrategy) -> Extent {
    let capacity = Extent::new(engine.end_of_allocation().unwrap(), ByteLength::new(4096)).unwrap();
    engine.append(&[0xA5; 8192]).unwrap();
    let kind = if strategy == FileSpaceStrategy::Page {
        ManagerKind::PagedLargeGeneric
    } else {
        ManagerKind::Flat
    };
    engine.space = SessionSpace::from_persisted(
        Some(strategy),
        ByteLength::new(1),
        crate::FileSpacePageSize::default(),
        [(Some(kind), capacity)],
    )
    .unwrap();
    capacity
}

fn fixture(path: &std::path::Path, strategy: FileSpaceStrategy) -> WriteEngine {
    let mut builder = FileBuilder::new();
    builder
        .with_userblock(UserblockSize::try_from(4096).unwrap())
        .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    builder.create_dataset("value").with_i32_data(&[7]);
    builder.write(path).unwrap();
    let mut engine = edit_test_util::open_engine(path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.aggregation = None;
    engine.space = empty_space(strategy);
    engine
}

fn empty_space(strategy: FileSpaceStrategy) -> SessionSpace {
    SessionSpace::from_persisted(
        Some(strategy),
        ByteLength::new(1),
        crate::FileSpacePageSize::default(),
        [],
    )
    .unwrap()
}

fn root(engine: &WriteEngine) -> PhysicalAddress {
    engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap()
}

fn protection(start: u64, len: u64) -> OriginalStorage {
    OriginalStorage::from_test_ranges(&[FileRange::new(PhysicalAddress::new(start), len).unwrap()])
}

fn assert_no_writes(events: &[IoEvent]) {
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, IoEvent::Write { .. } | IoEvent::Resize { .. })),
        "{events:?}"
    );
}

fn assert_end(error: PublicationError, proposed: PhysicalAddress, required: PhysicalAddress) {
    assert!(
        matches!(error, PublicationError::Storage(StorageConflict::End { proposed: actual, required: end }) if actual == proposed && end == required),
        "{error:?}"
    );
}

#[derive(Clone, Copy, Debug)]
enum Conflict {
    OldManager,
    Recycle,
    Returned,
    Reusable,
}
