use core::cell::Cell;
use core::num::NonZeroU16;
use std::collections::BTreeSet;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::PendingEditInventory;
use crate::AttrValue;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::COLLECTION_WORK;
use crate::edit::analysis::CollectionContent;
use crate::edit::analysis::CollectionProvenance;
use crate::edit::analysis::heap::qualification::MemberUnavailable;
use crate::edit::analysis::heap::qualification::ParentUnavailable;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::PreparedTreeCounts;
use crate::edit::commit::plan::inventory::members::MemberDecision;
use crate::edit::commit::plan::inventory::prospective::ProspectiveUnavailable;
use crate::edit::commit::plan::inventory::prospective::ReferencePresence;
use crate::edit::test_util as edit_test_util;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::one_origin(1, Restriction::None)]
#[case::three_origins(3, Restriction::None)]
#[case::surviving_reference(3, Restriction::Live)]
#[case::unavailable_qualification(3, Restriction::Qualification)]
fn physical_member_decisions_preserve_each_predecessor(
    #[case] count: usize,
    #[case] restriction: Restriction,
) {
    let path = test_util::temp::temp_path("shared-member-evidence.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value"), Object::new(91, b"unobserved")]);
    let mut owners = Vec::new();
    for index in 0..count {
        let mut messages = Form::CompactDataset.messages(
            &mut fixture,
            &crate::make_i32_type(),
            &7_i32.to_le_bytes(),
        );
        messages.extend(Form::CompactAttribute.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(5, parent, 7),
        ));
        let owner = fixture.append_header(messages);
        fixture.link(&format!("owner{index}"), owner);
        owners.push(ObjectId::new(PhysicalAddress::new(owner)));
    }
    if matches!(restriction, Restriction::Live) {
        let messages = Form::FillValue.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(5, parent, 7),
        );
        fixture.object("keep", messages);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    for index in 0..count {
        engine
            .set_dataset_attr(
                &ObjectPathBuf::parse(&format!("owner{index}")),
                "a",
                AttrValue::I64(9),
            )
            .unwrap();
    }
    let mut prepared = super::prepare(&mut engine);
    if matches!(restriction, Restriction::Qualification) {
        let mut phase = AnalysisPhase::new(&engine);
        let candidates = phase.deleted_collections(&owners);
        prepared.collections = phase.capture_collections(None, candidates);
    }
    assert_eq!(
        prepared.collections().addresses(),
        &BTreeSet::from([PhysicalAddress::new(parent)])
    );
    let expected_origins = owners
        .iter()
        .copied()
        .map(|object| CollectionProvenance {
            object,
            content: if matches!(restriction, Restriction::Qualification) {
                CollectionContent::Object
            } else {
                CollectionContent::Attribute("a".into())
            },
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        prepared
            .collections()
            .predecessors()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        expected_origins
    );
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let applied = prepared.apply(&mut engine).unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    let work = count + 1 + usize::from(matches!(restriction, Restriction::Live));
    assert_eq!(COLLECTION_WORK.with(Cell::get), (work, work));
    let id = GlobalHeapId {
        collection: PhysicalAddress::new(parent),
        index: NonZeroU16::new(7).unwrap(),
    };
    let presence = if matches!(restriction, Restriction::Live) {
        ReferencePresence::ObservedLive
    } else {
        ReferencePresence::Unreferenced
    };
    assert_eq!(pending.prospective().member_presence(id), presence);
    assert_eq!(
        pending.prospective().parent_presence(id.collection),
        presence
    );
    let decisions = pending.member_decisions();
    assert_eq!(decisions.physical_members(), 1);
    assert_eq!(
        decisions
            .members()
            .keys()
            .map(|(origin, member)| {
                assert_eq!(*member, id);
                origin.clone()
            })
            .collect::<BTreeSet<_>>(),
        expected_origins
    );
    let outcome = match restriction {
        Restriction::None => Ok(MemberDecision::Remove),
        Restriction::Live => Ok(MemberDecision::Live),
        Restriction::Qualification => Err(MemberUnavailable::Qualification),
    };
    assert_eq!(
        decisions.members().values().copied().collect::<Vec<_>>(),
        vec![outcome; count]
    );
    assert_eq!(
        decisions.parents()[&id.collection],
        Err(if matches!(restriction, Restriction::Qualification) {
            ParentUnavailable::Qualification
        } else {
            ParentUnavailable::Unmatched
        })
    );
    assert_eq!(
        decisions.plans().len(),
        usize::from(matches!(restriction, Restriction::None))
    );
    if let Some(plan) = decisions.plans().first() {
        assert_eq!(plan.retained(), 1);
        assert_eq!(plan.moves()[0].object().index().get(), 91);
    }
    assert_eq!(COLLECTION_WORK.with(Cell::get), (work, work));
}

#[test]
fn unavailable_counts_preserve_independent_attribute_candidates() {
    let path = test_util::temp::temp_path("shared-unavailable-count-evidence.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value")]);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &7_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    ));
    let owner = fixture.append_header(messages);
    fixture.link("value", owner);
    let mut removed =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &11_i32.to_le_bytes());
    removed.push(Message::reference_count(1));
    removed.push(Message::reference_count(1));
    fixture.object("remove", removed);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("remove")).unwrap();
    let prepared = super::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    assert!(matches!(tree.counts, Some(PreparedTreeCounts::Unavailable)));
    assert!(prepared.original().facts().is_none());
    let id = GlobalHeapId {
        collection: PhysicalAddress::new(parent),
        index: NonZeroU16::new(7).unwrap(),
    };
    let origin = CollectionProvenance {
        object: ObjectId::new(PhysicalAddress::new(owner)),
        content: CollectionContent::Attribute("a".into()),
    };
    assert_eq!(
        prepared.collections().addresses(),
        &BTreeSet::from([id.collection])
    );
    let observations = prepared.collections().predecessors().unwrap();
    assert_eq!(observations.len(), 1);
    assert_eq!(observations[&origin].objects(), &BTreeSet::from([id]));
    let applied = prepared.apply(&mut engine).unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    assert_eq!(
        pending.prospective().unavailable(),
        Some(&ProspectiveUnavailable::Original)
    );
    assert_eq!(
        pending.prospective().member_presence(id),
        ReferencePresence::Unavailable
    );
    assert_eq!(
        pending.prospective().parent_presence(id.collection),
        ReferencePresence::Unavailable
    );
    let decisions = pending.member_decisions();
    assert_eq!(
        decisions.members()[&(origin, id)],
        Ok(MemberDecision::Prospective)
    );
    assert_eq!(
        decisions.parents()[&id.collection],
        Ok(MemberDecision::Prospective)
    );
    assert_eq!(decisions.physical_members(), 1);
    assert!(decisions.plans().is_empty());
}

#[derive(Clone, Copy)]
enum Restriction {
    Live,
    None,
    Qualification,
}
