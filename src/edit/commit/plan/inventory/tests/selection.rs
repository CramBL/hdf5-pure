use core::cell::Cell;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use crate::AttrValue;
use crate::CharacterSet;
use crate::DatasetBuilder;
use crate::FileBuilder;
use crate::FormatError;
use crate::PhysicalAddress;
use crate::edit::WriteEngine;
use crate::edit::analysis::CollectionContent;
use crate::edit::analysis::CollectionProvenance;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::ownership::OWNERSHIP_READS;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::CommitPreparation;
use crate::edit::commit::plan::PreparedCommit;
use crate::edit::commit::plan::PreparedTreeCounts;
use crate::edit::commit::plan::inventory::InventoryPreparation;
use crate::edit::commit::plan::inventory::PreparedEditInventory;
use crate::edit::global_heap_liveness::DATASET_VALUE_WORK;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::tree(CandidateFreeEdit::Group)]
#[case::numeric_overwrite(CandidateFreeEdit::Overwrite)]
#[case::numeric_attribute(CandidateFreeEdit::Attribute)]
fn candidate_preparation_preserves_reference_free_read_and_discovery_cost(
    #[case] edit: CandidateFreeEdit,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (legacy_region, checked_region) = match edit {
        CandidateFreeEdit::Attribute => (
            "legacy_candidate_free_numeric_attribute",
            "checked_candidate_free_numeric_attribute",
        ),
        CandidateFreeEdit::Group => (
            "legacy_candidate_free_group",
            "checked_candidate_free_group",
        ),
        CandidateFreeEdit::Overwrite => (
            "legacy_candidate_free_numeric_overwrite",
            "checked_candidate_free_numeric_overwrite",
        ),
    };
    let path = test_util::temp::temp_path("candidate-free-preparation.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("value").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let (mut legacy, calls) = edit_test_util::open_counted_engine(&path);
    stage_candidate_free(&mut legacy, edit);
    let staged = std::mem::take(&mut legacy.staged);
    reset_work(&calls);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (preparation, allocated) = allocation::measure(legacy_region, || {
        PreparedCommit::prepare(&mut legacy, staged)
    });
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let preparation = PreparedCommit::prepare(&mut legacy, staged);
    let CommitPreparation::Prepared(prepared) = preparation else {
        panic!("expected ordinary preparation");
    };
    let baseline = work(&calls);
    assert!(baseline.0 > 0);
    assert_eq!((baseline.2, baseline.3), (0, 0));
    assert_eq!(
        baseline.1,
        if matches!(edit, CandidateFreeEdit::Attribute) {
            (1, 2)
        } else {
            (0, 0)
        }
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert!(allocated.blocks > 0 && allocated.bytes > 0, "{allocated}");
    drop(prepared);
    drop(legacy);
    let (mut engine, calls) = edit_test_util::open_counted_engine(&path);
    stage_candidate_free(&mut engine, edit);
    let staged = std::mem::take(&mut engine.staged);
    reset_work(&calls);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (preparation, checked) = allocation::measure(checked_region, || {
        PreparedEditInventory::prepare(&mut engine, staged)
    });
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let preparation = PreparedEditInventory::prepare(&mut engine, staged);
    let InventoryPreparation::Direct(prepared) = preparation else {
        panic!("expected preparation without heap candidates");
    };
    assert_eq!(work(&calls), baseline);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert_eq!(
        (checked.blocks, checked.bytes),
        (allocated.blocks, allocated.bytes),
        "{checked}, {allocated}"
    );
    assert_eq!(
        matches!(prepared.plan, CommitPlan::InPlace(_)),
        matches!(edit, CandidateFreeEdit::Overwrite)
    );
    drop(prepared);
    drop(engine);
    let (mut observed, calls) = edit_test_util::open_counted_engine(&path);
    stage_candidate_free(&mut observed, CandidateFreeEdit::Group);
    let staged = std::mem::take(&mut observed.staged);
    reset_work(&calls);
    let InventoryPreparation::Tree(_) =
        PreparedEditInventory::prepare_observed(&mut observed, staged)
    else {
        panic!("expected full observation control");
    };
    let control = work(&calls);
    assert!(
        control.0 > 0 && control.1.0 > 0 && control.3 > 0,
        "{control:?}"
    );
    if matches!(edit, CandidateFreeEdit::Group) {
        assert!(control.0 > baseline.0);
    }
}

#[rstest]
#[case::last_link(Counts::Final)]
#[case::surviving_alias(Counts::Shared)]
#[case::unavailable_count(Counts::Ambiguous)]
fn candidate_preparation_keeps_deleted_targets_independent_of_retirement(#[case] counts: Counts) {
    let path = test_util::temp::temp_path("candidate-deleted-target.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed"), Object::new(11, b"keep")]);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    );
    records.extend(match counts {
        Counts::Final => vec![Message::reference_count(1)],
        Counts::Shared => vec![Message::reference_count(2)],
        Counts::Ambiguous => vec![Message::reference_count(1), Message::reference_count(1)],
    });
    let owner = fixture.append_header(records);
    fixture.link("value", owner);
    if matches!(counts, Counts::Shared) {
        fixture.link("alias", owner);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("value")).unwrap();
    let prepared = prepare_candidates(&mut engine);
    assert_parent(&prepared, parent, owner, CollectionContent::Object);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected tree plan");
    };
    match counts {
        Counts::Ambiguous => assert!(matches!(tree.counts, Some(PreparedTreeCounts::Unavailable))),
        Counts::Final | Counts::Shared => {
            let Some(PreparedTreeCounts::Complete {
                eligible_owners, ..
            }) = &tree.counts
            else {
                panic!("expected checked counts");
            };
            assert_eq!(
                eligible_owners,
                &if matches!(counts, Counts::Final) {
                    vec![ObjectId::new(PhysicalAddress::new(owner))]
                } else {
                    vec![]
                }
            );
        }
    }
}

#[rstest]
#[case::dataset_removal(Attribute::Dataset, false)]
#[case::dataset_replacement(Attribute::Dataset, true)]
#[case::group_removal(Attribute::Group, false)]
#[case::group_replacement(Attribute::Group, true)]
fn candidate_preparation_keeps_removed_and_replaced_attribute_parents(
    #[case] attribute: Attribute,
    #[case] replace: bool,
) {
    let path = test_util::temp::temp_path("candidate-attribute.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed"), Object::new(11, b"keep")]);
    let attrs = Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    );
    let records = match attribute {
        Attribute::Dataset => {
            let mut records = Form::CompactDataset.messages(
                &mut fixture,
                &crate::make_i32_type(),
                &17_i32.to_le_bytes(),
            );
            records.extend(attrs);
            records
        }
        Attribute::Group => test_util::group_info::compact_messages(&attrs),
    };
    let owner = fixture.append_header(records);
    fixture.link("value", owner);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let path = ObjectPathBuf::parse("value");
    match (attribute, replace) {
        (Attribute::Dataset, true) => engine.set_dataset_attr(&path, "a", AttrValue::I64(9)),
        (Attribute::Dataset, false) => engine.remove_dataset_attr(&path, "a"),
        (Attribute::Group, true) => engine.set_group_attr(&path, "a", AttrValue::I64(9)),
        (Attribute::Group, false) => engine.remove_group_attr(&path, "a"),
    }
    .unwrap();
    let prepared = prepare_candidates(&mut engine);
    assert_parent(
        &prepared,
        parent,
        owner,
        CollectionContent::Attribute("a".into()),
    );
}

#[rstest]
#[case::compact_raw(Form::CompactDataset, false)]
#[case::contiguous_raw(Form::ContiguousDataset, false)]
#[case::compact_deferred(Form::CompactDataset, true)]
#[case::contiguous_deferred(Form::ContiguousDataset, true)]
fn candidate_preparation_keeps_old_dataset_values_before_placement(
    #[case] form: Form,
    #[case] deferred: bool,
    #[values(0, 7)] original_length: u32,
) {
    let path = test_util::temp::temp_path("candidate-overwrite.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed"), Object::new(11, b"keep")]);
    let datatype = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
    let records = form.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(original_length, parent, 7),
    );
    let owner = fixture.append_header(records);
    fixture.link("value", owner);
    fixture.write(&path);
    let original = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let mut replacement = DatasetBuilder::new("");
    if deferred {
        replacement.with_vlen_strings(&["after"]);
    } else {
        replacement.with_raw_data(datatype, global_heap_fixture::reference(4, parent, 11), 1);
    }
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("value"), replacement)
        .unwrap();
    let prepared = prepare_candidates(&mut engine);
    assert_parent(&prepared, parent, owner, CollectionContent::DatasetValues);
    assert_eq!(engine.disk_bytes().unwrap(), original);
}

#[test]
fn candidate_preparation_keeps_observations_without_a_checked_parent_allocation() {
    let path = test_util::temp::temp_path("candidate-unavailable-framing.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed")]);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    );
    fixture.object("value", records);
    fixture.write(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[usize::try_from(parent).unwrap()] = 0;
    std::fs::write(&path, &bytes).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("value")).unwrap();
    let prepared = prepare_candidates(&mut engine);
    let parent = PhysicalAddress::new(parent);
    assert_eq!(
        prepared
            .collections()
            .addresses()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [parent]
    );
    let Err(error) = &prepared.collections().contents()[&parent] else {
        panic!("expected unavailable parent framing");
    };
    assert!(
        matches!(error, FormatError::InvalidGlobalHeapSignature),
        "{error:?}"
    );
    assert_eq!(prepared.collections().predecessors().unwrap().len(), 1);
}

fn stage_candidate_free(engine: &mut WriteEngine, edit: CandidateFreeEdit) {
    match edit {
        CandidateFreeEdit::Group => engine.create_group(&ObjectPathBuf::parse("new")).unwrap(),
        CandidateFreeEdit::Attribute => engine
            .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
            .unwrap(),
        CandidateFreeEdit::Overwrite => {
            let mut replacement = DatasetBuilder::new("");
            replacement.with_i32_data(&[11]);
            engine
                .stage_dataset_write(&ObjectPathBuf::parse("value"), replacement)
                .unwrap();
        }
    }
}

fn prepare_candidates(engine: &mut WriteEngine) -> PreparedEditInventory {
    let staged = std::mem::take(&mut engine.staged);
    match PreparedEditInventory::prepare(engine, staged) {
        InventoryPreparation::Tree(prepared) => prepared,
        InventoryPreparation::Rejected { error, .. } => panic!("preparation failed: {error:?}"),
        InventoryPreparation::Direct(_) => panic!("expected preparation with heap candidates"),
    }
}

fn assert_parent(
    prepared: &PreparedEditInventory,
    parent: u64,
    owner: u64,
    content: CollectionContent,
) {
    let parent = PhysicalAddress::new(parent);
    assert_eq!(
        prepared
            .collections()
            .addresses()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [parent]
    );
    assert_eq!(
        prepared.collections().contents()[&parent]
            .as_ref()
            .unwrap()
            .members()
            .iter()
            .map(|member| member.index().get())
            .collect::<Vec<_>>(),
        [7, 11]
    );
    assert_eq!(
        prepared
            .collections()
            .predecessors()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        [&CollectionProvenance {
            object: ObjectId::new(PhysicalAddress::new(owner)),
            content
        }]
    );
}

fn reset_work(calls: &AtomicU64) {
    calls.store(0, Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    DATASET_VALUE_WORK.with(|work| work.set(0));
    OWNERSHIP_READS.with(|work| work.set(0));
}

fn work(calls: &AtomicU64) -> (u64, (usize, usize), u64, usize) {
    (
        calls.load(Ordering::Relaxed),
        DISCOVERY_WORK.with(Cell::get),
        DATASET_VALUE_WORK.with(Cell::get),
        OWNERSHIP_READS.with(Cell::get),
    )
}

#[derive(Clone, Copy, Debug)]
enum CandidateFreeEdit {
    Attribute,
    Group,
    Overwrite,
}

#[derive(Clone, Copy, Debug)]
enum Counts {
    Ambiguous,
    Final,
    Shared,
}

#[derive(Clone, Copy, Debug)]
enum Attribute {
    Dataset,
    Group,
}
