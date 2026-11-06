use core::cell::Cell;
use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_core::__private::{ByteLength, FileRange};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{
    AllocationClass, Extent, FreeClass, ManagerKind, ReturnedExtents, SessionSpace,
};
use rstest::rstest;
use test_util::disk_log::{self, DiskOp, FailurePoint, IoEvent, IoOutcome, Recording};

use super::super::{
    OriginalStorage, PendingEditInventory, SpaceClaim, StorageApplyFailure, StorageConflict,
    StoragePublicationFailure, StoragePublicationRejection, StorageUnavailable,
};
use crate::edit::commit::plan::inventory::AppliedEditInventory;
use crate::edit::commit::plan::inventory::dispositions::{PlacementError, TransitionWithheld};
use crate::edit::commit::publish::AppliedTree;
use crate::edit::commit::{PublicationError, PublicationPurpose, PublicationState};
use crate::edit::test_util as edit_test_util;
use crate::edit::{TailPlacement, WriteEngine};
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::source::BytesSource;
use crate::{
    AttrValue, Error, File, FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy,
    SyncPolicy, UserblockSize,
};

#[rstest]
#[case::flat(Strategy::Flat)]
#[case::page(Strategy::Page)]
#[case::page_persistent(Strategy::PagePersistent)]
#[case::flat_persistent(Strategy::FlatPersistent)]
#[case::reclamation(Strategy::Reclaim)]
fn owned_publication_preserves_original_storage_until_actual_completion(
    #[case] strategy: Strategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("owned.h5");
    let mut engine = fixture(&path, strategy, false);
    let old_root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let pending = apply(&mut engine);
    pending.exclusion().unwrap();
    let requested = pending.pending.root;
    let protected = preserved(pending.engine, &pending.storage);
    let maximum = pending.maximum_end();
    disk_log::start();
    let result =
        pending.publish_with_completion(|published, original, dispositions, storage, _| {
            assert!(original.facts().unwrap().counts().changes().is_empty());
            assert_eq!(
                dispositions.replacements().unwrap(),
                &BTreeMap::from([(ObjectId::new(old_root), ObjectId::new(requested))])
            );
            assert_eq!(published.requested_root(), requested);
            if matches!(strategy, Strategy::Reclaim) {
                assert_eq!(published.root(), old_root);
                assert_ne!(published.root(), requested);
            }
            let end = published.end();
            let engine = published.into_engine();
            assert_eq!(end.get(), engine.image.len());
            assert!(end >= maximum);
            assert_eq!(engine.publication, PublicationState::Published);
            storage
                .exclude_publication(engine, &ReturnedExtents::default(), TailPlacement::Anywhere)
                .unwrap();
            assert_eq!(preserved(engine, storage), protected);
            Ok(())
        });
    if let Err(error) = result {
        panic_failure(*error);
    }
    let recording = disk_log::finish();
    let publications = recording
        .events
        .iter()
        .filter(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == 4096))
        .count();
    assert!(
        publications
            >= if matches!(strategy, Strategy::Reclaim) {
                4
            } else {
                1
            }
    );
    assert_intermediate_bounds(&recording, maximum);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [7, 8]);
    assert_eq!(
        file.root().attrs().unwrap()["generation"],
        AttrValue::I64(1)
    );
    drop(file);
}

#[test]
fn publication_rechecks_current_claims_and_retains_the_authentic_pending_batch() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("current.h5");
    let mut engine = fixture(&path, Strategy::PagePersistent, false);
    let pending = apply(&mut engine);
    let allocation = pending.storage.allocations()[0];
    let retired = pending.pending.batch.entries().to_vec();
    let extent = Extent::new(
        pending
            .engine
            .addresses()
            .relative(allocation.range.start())
            .unwrap(),
        ByteLength::new(allocation.range.length()),
    )
    .unwrap();
    pending.engine.space.record_freed(
        pending.engine.end_of_allocation().unwrap(),
        extent,
        FreeClass::Dead,
    );
    let bytes = pending.engine.disk_bytes().unwrap();
    disk_log::start();
    let failure = pending
        .publish_with_completion(|_, _, _, _, _| -> Result<(), Error> {
            panic!("a conflicting allocator cannot complete")
        })
        .err()
        .unwrap();
    let StoragePublicationFailure::BeforePublication { pending, reason } = *failure else {
        panic!("expected a retained pending batch")
    };
    assert!(
        matches!(reason, StoragePublicationRejection::Exclusion),
        "{reason:?}"
    );
    assert!(
        matches!(pending.exclusion(), Err(StorageConflict::Overlap { allocation: actual, .. }) if *actual == allocation)
    );
    assert_eq!(pending.pending.batch.entries(), retired);
    assert_eq!(pending.engine.disk_bytes().unwrap(), bytes);
    assert!(disk_log::finish().events.is_empty());
}

#[test]
fn publication_retains_pending_inputs_when_placement_reconciliation_is_incomplete() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("placement.h5");
    let mut engine = fixture(&path, Strategy::Flat, false);
    let mut prepared = super::prepare(&mut engine);
    prepared.dispositions.record(
        &ObjectPathBuf::root(),
        None,
        ObjectId::new(PhysicalAddress::new(0)),
    );
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    assert_eq!(
        pending.dispositions.replacements().unwrap_err(),
        &TransitionWithheld::Placement(PlacementError::Source(ObjectPathBuf::root()))
    );
    let root = pending.pending.root;
    let entries = pending.pending.batch.entries().to_vec();
    let bytes = pending.engine.disk_bytes().unwrap();
    disk_log::start();
    let failure = pending
        .publish_with_completion(|_, _, _, _, _| -> Result<(), Error> {
            panic!("an incomplete placement cannot complete")
        })
        .err()
        .unwrap();
    let StoragePublicationFailure::BeforePublication { pending, reason } = *failure else {
        panic!("expected retained pending inputs")
    };
    assert!(
        matches!(reason, StoragePublicationRejection::Placement),
        "{reason:?}"
    );
    assert_eq!(pending.pending.root, root);
    assert_eq!(pending.pending.batch.entries(), entries);
    assert_eq!(pending.engine.disk_bytes().unwrap(), bytes);
    assert!(disk_log::finish().events.is_empty());
}

#[rstest]
#[case::target(true)]
#[case::source(false)]
fn root_copy_checks_both_complete_allocations_before_writing(#[case] target: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("root-copy.h5");
    let mut engine = fixture(&path, Strategy::Reclaim, false);
    let old_root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let pending = apply(&mut engine);
    let requested = pending.pending.root;
    let mut returned = returned(pending.engine, pending.pending.batch.entries());
    let reclamation = pending
        .engine
        .plan_root_reclamation(requested, &mut returned)
        .unwrap()
        .unwrap();
    pending
        .engine
        .commit_persisting_releasing_reserve(requested, returned)
        .unwrap();
    let protected = pending
        .engine
        .oh_chunk_spans(if target { old_root } else { requested })
        .unwrap()[0];
    let storage = OriginalStorage::from_test_ranges(&[protected]);
    let bytes = pending.engine.disk_bytes().unwrap();
    let managers = pending.engine.persist.as_ref().unwrap().old_blocks.clone();
    let reusable = pending.engine.space.reusable_sections().collect::<Vec<_>>();
    disk_log::start();
    let error = pending
        .engine
        .reclaim_root_metadata_excluding(reclamation, Some(&storage))
        .unwrap_err();
    let PublicationError::Storage(StorageConflict::Overlap { range, claim, .. }) = error else {
        panic!("expected a root-copy conflict, got {error:?}")
    };
    assert_eq!(range, protected);
    assert_eq!(
        claim,
        if target {
            SpaceClaim::RootCopy
        } else {
            SpaceClaim::Retirement(FreeClass::Dead)
        }
    );
    assert_eq!(pending.engine.disk_bytes().unwrap(), bytes);
    assert_eq!(
        pending.engine.persist.as_ref().unwrap().old_blocks,
        managers
    );
    assert_eq!(
        pending.engine.space.reusable_sections().collect::<Vec<_>>(),
        reusable
    );
    assert_eq!(pending.engine.publication, PublicationState::Published);
    assert!(disk_log::finish().events.is_empty());
}

#[test]
fn aggregate_surrender_rejects_protected_spare_before_ownership_changes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("aggregate.h5");
    let mut engine = fixture(&path, Strategy::FlatPersistent, false);
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        ByteLength::new(1),
        FileSpacePageSize::default(),
        [],
    )
    .unwrap();
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let address = engine
        .alloc_or_append_typed(&[0; 32], AllocationClass::Metadata)
        .unwrap();
    let allocation = Extent::new(
        engine
            .addresses()
            .relative(PhysicalAddress::new(address))
            .unwrap(),
        ByteLength::new(32),
    )
    .unwrap();
    let returned = || ReturnedExtents::from_sorted(vec![(allocation, FreeClass::Dead)]).unwrap();
    let spare = engine.aggregation.unwrap().unused().collect::<Vec<_>>();
    assert!(!spare.is_empty());
    let range = FileRange::new(
        engine.addresses().resolve(spare[0].start()).unwrap(),
        spare[0].len().get(),
    )
    .unwrap();
    let storage = OriginalStorage::from_test_ranges(&[range]);
    let snapshot = engine.snapshot_free();
    let mut surrendered = returned();
    engine
        .release_aggregates_for_tail(&mut surrendered)
        .unwrap();
    assert_eq!(engine.aggregation.unwrap().unused().collect::<Vec<_>>(), []);
    assert!(surrendered.entries().len() > 1);
    engine.restore_free(snapshot);
    let managers = engine.persist.as_ref().unwrap().old_blocks.clone();
    let bytes = engine.disk_bytes().unwrap();
    disk_log::start();
    let error = engine
        .commit_persisting_releasing_reserve_excluding(root, returned(), Some(&storage))
        .unwrap_err();
    assert!(
        matches!(
            error,
            PublicationError::Storage(StorageConflict::Overlap {
                claim: SpaceClaim::Aggregate,
                ..
            })
        ),
        "{error:?}"
    );
    assert_eq!(
        engine.aggregation.unwrap().unused().collect::<Vec<_>>(),
        spare
    );
    assert_eq!(engine.persist.as_ref().unwrap().old_blocks, managers);
    assert_eq!(engine.disk_bytes().unwrap(), bytes);
    assert!(disk_log::finish().events.is_empty());
}

#[rstest]
#[case::uncertain(Failure::FirstPublication)]
#[case::page_content(Failure::PageContent)]
#[case::tail_recycle(Failure::TailRecycle)]
#[case::root_copy(Failure::RootCopy)]
#[case::truncation(Failure::Truncation)]
fn publication_failure_retains_original_facts_without_completion(#[case] failure: Failure) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let strategy = match failure {
        Failure::FirstPublication => Strategy::Flat,
        Failure::PageContent => Strategy::PagePersistent,
        _ => Strategy::Reclaim,
    };
    let mut engine = fixture(&path, strategy, false);
    let initial = engine.disk_bytes().unwrap();
    let old_root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let old_tail = engine
        .superblock()
        .superblock_extension_address
        .map(|address| engine.addresses().resolve(address).unwrap());
    let pending = apply(&mut engine);
    disk_log::start();
    if let Err(error) = pending.publish_with_completion(|_, _, _, _, _| Ok(())) {
        panic_failure(*error);
    }
    let recording = disk_log::finish();
    let writes = recording
        .events
        .iter()
        .filter_map(|event| {
            if let IoEvent::Write { offset, .. } = event {
                Some(*offset)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let write = match failure {
        Failure::FirstPublication | Failure::PageContent => Some(0),
        Failure::TailRecycle => Some(
            writes
                .iter()
                .position(|offset| Some(PhysicalAddress::new(*offset)) == old_tail)
                .unwrap(),
        ),
        Failure::RootCopy => Some(
            writes
                .iter()
                .position(|offset| *offset == old_root.get())
                .unwrap(),
        ),
        Failure::Truncation => None,
    };
    drop(engine);
    fs::write(&path, initial).unwrap();
    let mut engine = stage(&path);
    let pending = apply(&mut engine);
    let requested = pending.pending.root;
    let protected = preserved(pending.engine, &pending.storage);
    let called = Cell::new(false);
    disk_log::start();
    if let Some(write) = write {
        disk_log::fail_write_after(write, 0);
    } else {
        disk_log::fail_resize_after(0, FailurePoint::Before);
    }
    let result = pending.publish_with_completion(|_, _, _, _, _| {
        called.set(true);
        Ok(())
    });
    let recording = disk_log::finish();
    let StoragePublicationFailure::Publication {
        original,
        dispositions,
        storage,
        collections: _,
        error,
    } = *result.err().unwrap()
    else {
        panic!("expected publication failure")
    };
    let PublicationError::Write(Error::Io(error)) = error else {
        panic!("expected injected I/O failure, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if write.is_some() {
            "injected disk write failure"
        } else {
            "injected disk resize failure"
        }
    );
    assert!(!called.get());
    assert!(original.facts().unwrap().counts().changes().is_empty());
    assert_eq!(
        dispositions.replacements().unwrap(),
        &BTreeMap::from([(ObjectId::new(old_root), ObjectId::new(requested))])
    );
    assert_eq!(preserved(&engine, &storage), protected);
    assert_eq!(
        engine.publication,
        match failure {
            Failure::FirstPublication => PublicationState::Uncertain,
            Failure::PageContent => PublicationState::Ready,
            _ => PublicationState::Published,
        }
    );
    if !matches!(failure, Failure::FirstPublication | Failure::PageContent) {
        assert!(recording.events.iter().any(|event| matches!(
            event,
            IoEvent::Write {
                offset: 4096,
                outcome: IoOutcome::Succeeded,
                ..
            }
        )));
    }
    storage
        .exclude_publication(
            &engine,
            &ReturnedExtents::default(),
            TailPlacement::Anywhere,
        )
        .unwrap();
}

#[test]
fn tail_shrink_retains_the_same_original_exclusions_through_its_publication() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("shrink.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_userblock(UserblockSize::try_from(4096).unwrap())
        .with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
        );
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("hole").with_u8_data(&[0; 512]);
    builder.create_dataset("keep").with_i32_data(&[7, 8]);
    builder.create_dataset("scratch").with_u8_data(&[9; 8192]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    engine.force_sync().unwrap();
    drop(engine);
    let mut engine = edit_test_util::open_engine(&path);
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let root_len = engine.oh_chunk_spans(root).unwrap()[0].length();
    let hole = engine
        .space
        .reusable_sections()
        .find(|extent| extent.len().get() >= root_len)
        .unwrap();
    let hole = Extent::new(hole.start(), ByteLength::new(root_len)).unwrap();
    engine.aggregation = None;
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "generation", AttrValue::I64(1))
        .unwrap();
    let available = engine.space.reusable_sections().collect::<Vec<_>>();
    let prepared = super::prepare(&mut engine);
    assert!(
        prepared.storage().unavailable().is_none(),
        "{:?}",
        prepared.storage().unavailable()
    );
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        ByteLength::new(1),
        FileSpacePageSize::default(),
        [(Some(ManagerKind::Flat), hole)],
    )
    .unwrap();
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    let protected = preserved(pending.engine, &pending.storage);
    let maximum = pending.maximum_end();
    let requested = pending.pending.root;
    let returned = returned(pending.engine, pending.pending.batch.entries());
    disk_log::start();
    pending
        .engine
        .publish_persisting_excluding(
            requested,
            returned,
            TailPlacement::Anywhere,
            PublicationPurpose::Commit,
            Some(&pending.storage),
        )
        .unwrap();
    let first_end = pending.engine.image.len();
    for free in available {
        let free = if free.range().contains(hole.range()) {
            Extent::between(hole.end(), free.end())
        } else {
            Some(free)
        };
        if let Some(free) = free {
            pending
                .engine
                .space
                .return_allocation(free, AllocationClass::Metadata);
        }
    }
    pending
        .engine
        .shrink_to_the_trailing_free_run_excluding(requested, Some(&pending.storage))
        .unwrap();
    assert!(pending.engine.image.len() < first_end);
    assert_eq!(preserved(pending.engine, &pending.storage), protected);
    pending
        .storage
        .exclude_publication(
            pending.engine,
            &ReturnedExtents::default(),
            TailPlacement::Anywhere,
        )
        .unwrap();
    let recording = disk_log::finish();
    let published = recording
        .operations
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write {
                offset: 4096,
                bytes,
            } => Some(
                crate::superblock::read(&BytesSource::new(bytes))
                    .unwrap()
                    .superblock
                    .eof_address,
            ),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(published.len() >= 2, "{published:?}");
    assert!(published.last().unwrap() < &published[0], "{published:?}");
    assert_intermediate_bounds(&recording, maximum);
}

#[test]
fn constrained_reference_update_failure_prevents_completion_after_root_publication() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("references.h5");
    let mut engine = fixture(&path, Strategy::Flat, true);
    let applied = super::prepare(&mut engine).apply(&mut engine).unwrap();
    let AppliedEditInventory { tree, storage, .. } = applied;
    assert!(matches!(
        storage.unavailable(),
        Some(StorageUnavailable::Owners)
    ));
    let AppliedTree {
        retirement,
        freed,
        relocations,
    } = tree;
    let pending = retirement.into_pending_frees(freed, engine.image.len());
    let protected = preserved(&engine, &storage);
    let called = Cell::new(false);
    disk_log::start();
    disk_log::fail_write_after(1, 0);
    let error = engine
        .publish_pending_tree_excluding(pending, relocations, Some(&storage), |_| {
            called.set(true);
            Ok(())
        })
        .unwrap_err();
    let recording = disk_log::finish();
    let PublicationError::Write(Error::Io(error)) = error else {
        panic!("expected a reference write failure, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert!(!called.get());
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(preserved(&engine, &storage), protected);
    let writes = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                offset, outcome, ..
            } => Some((*offset, *outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(writes.len(), 2);
    assert_eq!(writes[0], (4096, IoOutcome::Succeeded));
    assert_ne!(writes[1].0, 4096);
    assert_eq!(writes[1].1, IoOutcome::Failed(ErrorKind::Other));
}

fn apply(engine: &mut WriteEngine) -> PendingEditInventory<'_> {
    match super::prepare(engine).apply_excluding(engine) {
        Ok(pending) => pending,
        Err(StorageApplyFailure::BeforePlacement { prepared, error }) => panic!(
            "unexpected application rejection: {error:?}, {:?}",
            prepared.storage().unavailable()
        ),
        Err(StorageApplyFailure::Application(error)) => {
            panic!("unexpected application error: {error:?}")
        }
    }
}

fn fixture(path: &Path, strategy: Strategy, references: bool) -> WriteEngine {
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(4096).unwrap());
    let settings = match strategy {
        Strategy::Flat => FileSpaceSettings::new(FileSpaceStrategy::FsmAggr),
        Strategy::Page => FileSpaceSettings::new(FileSpaceStrategy::Page),
        Strategy::FlatPersistent => {
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true)
        }
        Strategy::PagePersistent => {
            FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true)
        }
        Strategy::Reclaim => FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(u64::MAX),
    };
    builder.with_file_space_settings(settings);
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("keep").with_i32_data(&[7, 8]);
    if references {
        builder.create_dataset("refs").with_path_references(&[""]);
    }
    builder.write(path).unwrap();
    stage(path)
}

fn stage(path: &Path) -> WriteEngine {
    let mut engine = edit_test_util::open_engine(path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "generation", AttrValue::I64(1))
        .unwrap();
    engine
}

fn returned(engine: &WriteEngine, entries: &[(FileRange, FreeClass)]) -> ReturnedExtents {
    ReturnedExtents::from_sorted(
        entries
            .iter()
            .map(|&(range, class)| {
                (
                    Extent::new(
                        engine.addresses().relative(range.start()).unwrap(),
                        ByteLength::new(range.length()),
                    )
                    .unwrap(),
                    class,
                )
            })
            .collect(),
    )
    .unwrap()
}

fn preserved(
    engine: &WriteEngine,
    storage: &OriginalStorage,
) -> BTreeMap<PhysicalAddress, Vec<u8>> {
    storage
        .allocations()
        .iter()
        .map(|allocation| {
            (
                allocation.range.start(),
                engine
                    .image()
                    .read_exact_at(
                        allocation.range.start().get(),
                        usize::try_from(allocation.range.length()).unwrap(),
                    )
                    .unwrap(),
            )
        })
        .collect()
}

fn assert_intermediate_bounds(recording: &Recording, maximum: PhysicalAddress) {
    let ends = recording
        .operations
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::SetLen(end) => Some(*end),
            DiskOp::Write {
                offset: 4096,
                bytes,
            } => {
                let source = BytesSource::new(bytes);
                Some(
                    crate::superblock::read(&source)
                        .unwrap()
                        .superblock
                        .eof_address,
                )
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!ends.is_empty());
    assert!(ends.iter().all(|end| *end >= maximum.get()), "{ends:?}");
}

fn panic_failure(failure: StoragePublicationFailure<'_>) -> ! {
    match failure {
        StoragePublicationFailure::BeforePublication { reason, .. } => {
            panic!("unexpected prepublication rejection: {reason:?}")
        }
        StoragePublicationFailure::Publication { error, .. } => {
            panic!("unexpected publication error: {error:?}")
        }
    }
}

#[derive(Clone, Copy)]
enum Strategy {
    Flat,
    FlatPersistent,
    Page,
    PagePersistent,
    Reclaim,
}

#[derive(Clone, Copy)]
enum Failure {
    FirstPublication,
    PageContent,
    RootCopy,
    TailRecycle,
    Truncation,
}
