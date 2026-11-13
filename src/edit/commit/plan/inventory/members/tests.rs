use std::path::Path;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::ParentRetirement;
use super::ParentRetirementMismatch;
use crate::AttrValue;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::qualification::OriginalQualification;
use crate::edit::analysis::heap::qualification::QualifiedParentAllocation;
use crate::edit::commit::plan::inventory::InventoryPreparation;
use crate::edit::commit::plan::inventory::PreparedEditInventory;
use crate::edit::commit::plan::inventory::prospective::ParentAbsence;
use crate::edit::commit::plan::inventory::prospective::ProspectiveCollections;
use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;

#[test]
fn retirement_keeps_both_source_and_prospective_associations_until_consumed() {
    let path = test_util::temp::temp_path("bound-parent-retirement.h5");
    let Evidence {
        allocation,
        absence,
        source,
        session,
        root,
    } = evidence(&path, Target::Same);
    let original_range = allocation.range();
    let retirement = ParentRetirement::new(allocation, absence).unwrap();
    assert_eq!(retirement.source(), &source);
    assert!(retirement.session().same(&session));
    assert_eq!(retirement.root(), root);
    assert_ne!(retirement.root(), source.object().header_address());
    assert_eq!(retirement.range(), original_range);
    assert_eq!(retirement.class(), FreeClass::Page(PageType::Raw));
    let (allocation, prospective_root) = retirement.into_parts();
    assert_eq!(allocation.source(), &source);
    assert!(allocation.session().same(&session));
    assert_eq!(allocation.range(), original_range);
    assert_eq!(allocation.class(), FreeClass::Page(PageType::Raw));
    assert_eq!(prospective_root, root);
}

#[rstest]
#[case::address(Mismatch::Address)]
#[case::session(Mismatch::Session)]
#[case::original_source(Mismatch::Source)]
fn retirement_rejects_incompatible_evidence(#[case] mismatch: Mismatch) {
    let path = test_util::temp::temp_path("mismatched-parent-retirement.h5");
    let (allocation, source, absence) = if matches!(mismatch, Mismatch::Source) {
        let (mut engine, parent) = fixture(&path);
        let first = prepare(&mut engine);
        let second = prepare(&mut engine);
        let first = apply(&mut engine, parent, first, Target::Same);
        let second = apply(&mut engine, parent, second, Target::Same);
        assert!(first.session.same(&second.session));
        assert_eq!(first.source.object(), second.source.object());
        assert_ne!(first.source, second.source);
        (first.allocation, first.source, second.absence)
    } else {
        let first = evidence(
            &path,
            if matches!(mismatch, Mismatch::Address) {
                Target::Other
            } else {
                Target::Same
            },
        );
        let absence = if matches!(mismatch, Mismatch::Session) {
            let other = test_util::temp::temp_path("other-parent-session.h5");
            evidence(&other, Target::Same).absence
        } else {
            first.absence
        };
        (first.allocation, first.source, absence)
    };
    let address = allocation.range().start();
    let expected = match mismatch {
        Mismatch::Address => ParentRetirementMismatch::Address {
            allocation: address,
            reference: PhysicalAddress::new(address.get() + 8),
        },
        Mismatch::Session => ParentRetirementMismatch::Session,
        Mismatch::Source => ParentRetirementMismatch::Source {
            allocation: source,
            reference: absence.source().clone(),
        },
    };
    let Err(error) = ParentRetirement::new(allocation, absence) else {
        panic!("expected incompatible evidence");
    };
    assert_eq!(error, expected);
}

fn evidence(path: &Path, target: Target) -> Evidence {
    let (mut engine, parent) = fixture(path);
    let prepared = prepare(&mut engine);
    apply(&mut engine, parent, prepared, target)
}

fn fixture(path: &Path) -> (WriteEngine, u64) {
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value")]);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &7_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    ));
    fixture.object("value", messages);
    fixture.write(path);
    let engine = edit_test_util::open_engine(path);
    (engine, parent)
}

fn prepare(engine: &mut WriteEngine) -> PreparedEditInventory {
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let InventoryPreparation::Tree(prepared) =
        PreparedEditInventory::prepare_observed(engine, staged)
    else {
        panic!("expected inventory preparation");
    };
    prepared
}

fn apply(
    engine: &mut WriteEngine,
    parent: u64,
    prepared: PreparedEditInventory,
    target: Target,
) -> Evidence {
    let source = prepared.dispositions().source().as_ref().unwrap().clone();
    let OriginalQualification::Qualified(qualification) = prepared.collections().qualification()
    else {
        panic!("expected original qualification");
    };
    let allocation = qualification
        .whole_parent_allocation(PhysicalAddress::new(parent))
        .unwrap();
    let session = engine.context.allocations.session();
    let applied = prepared.apply(engine).unwrap();
    let root = applied.tree.retirement.root();
    let prospective = ProspectiveCollections::collect(
        engine,
        &applied.original,
        &applied.dispositions,
        &applied.collections,
        root,
    );
    let address = if matches!(target, Target::Other) {
        parent + 8
    } else {
        parent
    };
    let absence = prospective
        .parent_absence(PhysicalAddress::new(address))
        .unwrap();
    Evidence {
        allocation,
        absence,
        source,
        session,
        root,
    }
}

struct Evidence {
    allocation: QualifiedParentAllocation,
    absence: ParentAbsence,
    source: ReconstructionSource,
    session: AllocationSession,
    root: PhysicalAddress,
}

#[derive(Clone, Copy)]
enum Target {
    Other,
    Same,
}

#[derive(Clone, Copy)]
enum Mismatch {
    Address,
    Session,
    Source,
}
