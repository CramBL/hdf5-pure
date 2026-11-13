use core::cell::Cell;
use core::num::NonZeroU32;
use std::io::ErrorKind;
use std::path::Path;

use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::PendingEditInventory;
use crate::AttrValue;
use crate::DatasetBuilder;
use crate::Datatype;
use crate::File;
use crate::FixedPointType;
use crate::PhysicalAddress;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::commit::PublicationError;
use crate::edit::commit::PublicationState;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::PreparedTreeCounts;
use crate::edit::commit::plan::inventory::InventoryPreparation;
use crate::edit::commit::plan::inventory::PreparedEditInventory;
use crate::edit::commit::plan::inventory::storage::OriginalStorage;
use crate::edit::commit::plan::inventory::storage::SpaceClaim;
use crate::edit::commit::plan::inventory::storage::StorageAllocation;
use crate::edit::commit::plan::inventory::storage::StorageConflict;
use crate::edit::commit::plan::inventory::storage::StorageOrigin;
use crate::edit::commit::plan::inventory::storage::StoragePublicationFailure;
use crate::edit::commit::plan::inventory::tests as inventory_tests;
use crate::edit::commit::plan::inventory::tests::Target;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::decrease(true, 0, 1)]
#[case::increase(false, 1, 3)]
#[case::balanced(true, 1, 2)]
#[case::repeated(false, 2, 4)]
fn inventory_publication_applies_original_source_count_changes(
    #[case] remove: bool,
    #[case] copies: usize,
    #[case] expected: u32,
) {
    let path = test_util::temp::temp_path("inventory-source-counts.h5");
    let target = inventory_tests::fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    if remove {
        engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    }
    for index in 0..copies {
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse(&format!("copy{index}")),
            )
            .unwrap();
    }
    let prepared = inventory_tests::prepare(&mut engine);
    let original = prepared.original().facts().unwrap();
    let source = original.counts().root();
    let original_chunk = original.counts().changes()[0].field().unwrap().chunk();
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    let Some(PreparedTreeCounts::Complete {
        inventory,
        protected,
        ..
    }) = &tree.counts
    else {
        panic!("expected count preparation");
    };
    assert_eq!(inventory.changes().len(), 1);
    assert_eq!(inventory.changes()[0].object(), target);
    assert_eq!(inventory.changes()[0].remaining(), expected);
    assert_eq!(inventory.root(), source);
    assert!(inventory.same_session(&engine));
    assert!(inventory.shares_phase(prepared.dispositions().source().as_ref().unwrap()));
    assert_eq!(protected.root(), if expected == 2 { None } else { source });
    assert_eq!(protected.updates().len(), usize::from(expected != 2));
    let applied = prepared.apply(&mut engine).unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    disk_log::start();
    let result = pending.publish_with_completion(|published, original, _, storage, _, _| {
        assert_eq!(
            original.facts().unwrap().counts().changes()[0].remaining(),
            expected
        );
        assert!(
            storage
                .allocations()
                .iter()
                .any(|allocation| allocation.range.start() == target.header_address())
        );
        let engine = published.into_engine();
        assert_eq!(read_count(engine, target), expected);
        Ok(())
    });
    let operations = disk_log::take();
    finish_publication(result);
    let publication = operations
        .iter()
        .position(|op| matches!(op, DiskOp::Write { offset: 0, .. }))
        .unwrap();
    let source_write = operations.iter().position(|op| matches!(op, DiskOp::Write { offset, .. } if *offset >= original_chunk.start().get() && *offset < original_chunk.end().get()));
    if expected > 2 {
        assert!(source_write.unwrap() < publication);
    } else if expected < 2 {
        assert!(source_write.unwrap() > publication);
    } else {
        assert_eq!(source_write, None);
    }
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
    for index in 0..copies {
        assert_eq!(
            file.dataset(&format!("copy{index}"))
                .unwrap()
                .read_f64()
                .unwrap(),
            [17.0]
        );
    }
}

#[rstest]
#[case::surviving(2, 1, false, false)]
#[case::last_links(2, 2, true, false)]
#[case::excess_declared(3, 2, false, false)]
#[case::unavailable(2, 2, false, true)]
fn inventory_retains_count_eligible_retirement_inputs(
    #[case] count: u32,
    #[case] removed: usize,
    #[case] retired: bool,
    #[case] ambiguous: bool,
) {
    let path = test_util::temp::temp_path("inventory-retirement-counts.h5");
    let (target, raw, end) = ordinary_fixture(&path, count, ambiguous);
    let mut engine = edit_test_util::open_engine(&path);
    for name in &["a", "b"][..removed] {
        engine.delete(&ObjectPathBuf::parse(name)).unwrap();
    }
    let prepared = inventory_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    assert!(tree.deleted_free.is_empty());
    if ambiguous {
        assert!(matches!(tree.counts, Some(PreparedTreeCounts::Unavailable)));
        assert!(
            prepared
                .reclamation
                .as_ref()
                .unwrap()
                .iter()
                .all(|(range, _)| range.start().get() < raw)
        );
        return;
    }
    let Some(PreparedTreeCounts::Complete {
        retained_roots,
        eligible_owners,
        ..
    }) = &tree.counts
    else {
        panic!("expected count preparation");
    };
    assert_eq!(eligible_owners.contains(&target), retired);
    assert_eq!(retained_roots.contains(&target), !retired);
    let ranges = prepared
        .reclamation
        .as_ref()
        .unwrap()
        .iter()
        .filter(|(range, _)| range.start().get() >= raw)
        .map(|(range, _)| (range.start().get(), range.end().get()))
        .collect::<Vec<_>>();
    assert_eq!(
        ranges,
        if retired {
            vec![(raw, raw + 8), (target.header_address().get(), end)]
        } else {
            vec![]
        }
    );
    let applied = prepared.apply(&mut engine).unwrap();
    assert_eq!(
        applied.tree().counts.as_ref().unwrap().updates().len(),
        usize::from(!retired)
    );
    engine.publish_tree(applied.into_tree()).unwrap();
    if !retired {
        assert_eq!(
            read_count(&engine, target),
            count - u32::try_from(removed).unwrap()
        );
    }
}

#[rstest]
#[case::zero(0)]
#[case::undercount(1)]
fn inventory_rejects_count_contradictions_without_consuming_staging(#[case] count: u32) {
    let path = test_util::temp::temp_path("inventory-count-contradiction.h5");
    let (target, _, _) = ordinary_fixture(&path, count, false);
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    let InventoryPreparation::Rejected { staged, error } =
        PreparedEditInventory::prepare(&mut engine, staged)
    else {
        panic!("expected count rejection");
    };
    let Error::InconsistentReferenceCount {
        address,
        observed,
        minimum,
    } = error
    else {
        panic!("unexpected count error: {error:?}");
    };
    assert_eq!(
        (address, observed, minimum),
        (target.header_address().get(), count, 2)
    );
    assert_eq!(staged.deletes(), [ObjectPathBuf::parse("a")]);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn inventory_projects_the_reconstructed_root_count() {
    let path = test_util::temp::temp_path("inventory-local-counts.h5");
    let mut fixture = Fixture::new().with_root_reference_count(NonZeroU32::new(2).unwrap());
    fixture.link("self", Fixture::ROOT);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("self")).unwrap();
    let mut prepared = inventory_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &mut prepared.prepared.plan else {
        panic!("expected a tree");
    };
    assert_eq!(
        tree.groups
            .get_mut(&ObjectPathBuf::root())
            .unwrap()
            .prepared
            .region
            .reference_count_mut()
            .unwrap()
            .count(),
        1
    );
    let applied = prepared.apply(&mut engine).unwrap();
    assert!(applied.tree().counts.as_ref().unwrap().updates().is_empty());
    engine.publish_tree(applied.into_tree()).unwrap();
    drop(engine);
}

#[test]
fn inventory_keeps_in_place_preparation_lazy() {
    let path = test_util::temp::temp_path("inventory-lazy-values.h5");
    ordinary_fixture(&path, 2, false);
    let mut engine = edit_test_util::open_engine(&path);
    let mut replacement = DatasetBuilder::new("");
    replacement.with_i32_data(&[19, 23]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("a"), replacement)
        .unwrap();
    let staged = std::mem::take(&mut engine.staged);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let InventoryPreparation::InPlace(prepared) =
        PreparedEditInventory::prepare(&mut engine, staged)
    else {
        panic!("expected in-place preparation");
    };
    assert_eq!(DISCOVERY_WORK.with(|work| work.get()), (0, 0));
    engine.apply_commit(prepared).unwrap();
}

#[rstest]
#[case::increase_before_publication(true, 0, PublicationState::Ready, 2)]
#[case::publication_after_increase(true, 2, PublicationState::Uncertain, 3)]
#[case::decrease_after_publication(false, 1, PublicationState::Published, 2)]
fn inventory_count_write_failures_preserve_the_publication_boundary(
    #[case] increase: bool,
    #[case] completed: usize,
    #[case] state: PublicationState,
    #[case] expected: u32,
) {
    let path = test_util::temp::temp_path("inventory-count-failure.h5");
    let target = inventory_tests::fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    if increase {
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse("copy"),
            )
            .unwrap();
    } else {
        engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    }
    let prepared = inventory_tests::prepare(&mut engine);
    let applied = prepared.apply(&mut engine).unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    let called = Cell::new(false);
    disk_log::start();
    disk_log::fail_write_after(completed, 0);
    let result = pending.publish_with_completion(|_, _, _, _, _, _| {
        called.set(true);
        Ok(())
    });
    disk_log::clear_failures();
    let error = match result {
        Ok(()) => panic!("expected a publication failure"),
        Err(error) => match *error {
            StoragePublicationFailure::Publication { error, .. } => error,
            StoragePublicationFailure::BeforePublication { reason, .. } => {
                panic!("unexpected publication rejection: {reason:?}")
            }
        },
    };
    let PublicationError::Write(Error::Io(error)) = error else {
        panic!("unexpected publication error: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert!(!called.get());
    assert_eq!(engine.publication, state);
    assert_eq!(read_count(&engine, target), expected);
    disk_log::take();
}

#[rstest]
#[case::duplicate(true)]
#[case::new(false)]
fn source_count_exclusions_keep_complete_allocations_without_duplicates(#[case] duplicate: bool) {
    let path = test_util::temp::temp_path("inventory-count-protection.h5");
    let target = inventory_tests::fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    let prepared = inventory_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    let Some(PreparedTreeCounts::Complete { protected, .. }) = &tree.counts else {
        panic!("expected counts");
    };
    let mut storage = if duplicate {
        OriginalStorage::from_counts(protected)
    } else {
        OriginalStorage::from_test_ranges(&[])
    };
    let pointer = storage.allocations().as_ptr();
    storage.include_counts(protected).unwrap();
    assert_eq!(
        storage.allocations(),
        [StorageAllocation {
            range: protected.updates()[0].allocation().range(),
            origin: StorageOrigin::Owner {
                object: target,
                kind: AllocationKind::Header,
            },
        }]
    );
    if duplicate {
        assert_eq!(storage.allocations().as_ptr(), pointer);
    }
    let allocation = storage.allocations()[0];
    let mut conflict = OriginalStorage::from_test_ranges(&[allocation.range]);
    let error = conflict.include_counts(protected).unwrap_err();
    let StorageConflict::Overlap {
        allocation: actual,
        range,
        claim,
    } = error
    else {
        panic!("unexpected protection error: {error:?}");
    };
    assert_eq!(actual, conflict.allocations()[0]);
    assert_eq!(range, allocation.range);
    assert_eq!(claim, SpaceClaim::CountHeader);
}

#[test]
fn empty_source_count_batches_keep_heap_parent_publication_exclusions() {
    let path = test_util::temp::temp_path("inventory-empty-count-parent.h5");
    inventory_tests::fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_group_attr(
            &ObjectPathBuf::root(),
            "a",
            AttrValue::VarLenString("value".into()),
        )
        .unwrap();
    engine.commit().unwrap();
    engine
        .remove_group_attr(&ObjectPathBuf::root(), "a")
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    engine
        .copy(
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("copy"),
        )
        .unwrap();
    let prepared = inventory_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    let Some(PreparedTreeCounts::Complete { protected, .. }) = &tree.counts else {
        panic!("expected counts");
    };
    assert!(protected.updates().is_empty());
    let parent = *prepared.collections().contents().keys().next().unwrap();
    assert_eq!(prepared.collections().contents().len(), 1);
    let allocation = *prepared.storage().allocations().iter().find(|allocation| matches!(allocation.origin, StorageOrigin::Parent { address, .. } if address == parent)).unwrap();
    let before = engine
        .image()
        .read_exact_at(
            parent.get(),
            usize::try_from(allocation.range.length()).unwrap(),
        )
        .unwrap();
    let applied = prepared.apply(&mut engine).unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    finish_publication(
        pending.publish_with_completion(|published, _, _, storage, _, _| {
            let error = storage
                .exclude(allocation.range, SpaceClaim::RootCopy)
                .unwrap_err();
            let StorageConflict::Overlap {
                allocation: actual,
                range,
                claim,
            } = error
            else {
                panic!("unexpected protection error: {error:?}");
            };
            assert_eq!(actual, allocation);
            assert_eq!(range, allocation.range);
            assert_eq!(claim, SpaceClaim::RootCopy);
            let engine = published.into_engine();
            assert_eq!(
                engine
                    .image()
                    .read_exact_at(parent.get(), before.len())
                    .unwrap(),
                before
            );
            Ok(())
        }),
    );
}

fn read_count(engine: &WriteEngine, target: ObjectId) -> u32 {
    WriteEngine::gather_oh_messages(
        &engine.image(),
        target.header_address(),
        engine.addresses(),
        engine.widths(),
    )
    .unwrap()
    .reference_count_mut()
    .unwrap()
    .count()
}

/// Writes a dataset with two hard links and the supplied reference count.
///
/// With `ambiguous`, writes duplicate count records. Returns the object identity, payload start,
/// and header end.
///
/// # Panics
///
/// Panics if the fixture cannot be written.
fn ordinary_fixture(path: &Path, count: u32, ambiguous: bool) -> (ObjectId, u64, u64) {
    let mut fixture = Fixture::new();
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let mut records = Form::ContiguousDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &[7_i32.to_le_bytes(), 11_i32.to_le_bytes()].concat(),
    );
    records.push(Message::reference_count(count));
    if ambiguous {
        records.push(Message::reference_count(count));
    }
    let target = fixture.append_header(records);
    let end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    fixture.link("a", target);
    fixture.link("b", target);
    fixture.write(path);
    (ObjectId::new(PhysicalAddress::new(target)), raw, end)
}

fn finish_publication<'a, T>(result: Result<T, Box<StoragePublicationFailure<'a>>>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match *error {
            StoragePublicationFailure::BeforePublication { reason, .. } => {
                panic!("unexpected publication rejection: {reason:?}")
            }
            StoragePublicationFailure::Publication { error, .. } => {
                panic!("unexpected publication failure: {error:?}")
            }
        },
    }
}
