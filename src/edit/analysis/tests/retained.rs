use core::cell::Cell;
use core::num::NonZeroU32;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::AnalysisPhase;
use super::super::COLLECTION_WORK;
use super::super::CollectionCandidates;
use super::super::CollectionProvenance;
use super::super::CountStorage;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::eligible(Original::Eligible)]
#[case::positive_count(Original::Retained)]
#[case::incomplete_references(Original::Incomplete)]
fn count_candidate_handoff_preserves_original_storage_and_collection_origins(
    #[case] original: Original,
) {
    let path = test_util::temp::temp_path("count-collection-handoff.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    messages.push(Message::reference_count(
        if original == Original::Retained { 3 } else { 2 },
    ));
    if original == Original::Incomplete {
        messages.push(Message::new(MessageType(0x7f), &[]));
    }
    let owner = fixture.append_header(messages);
    fixture.link("a", owner);
    fixture.link("b", owner);
    let replacement = fixture.append_header(vec![]);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut source = AnalysisPhase::new(&engine);
    let mut storage = storage(&mut source, &["a", "b"]);
    let original_allocations = storage.allocations().to_vec();
    assert_eq!(
        original_allocations.len(),
        if original == Original::Retained { 0 } else { 2 }
    );
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut candidates = storage.take_collections(&mut source).unwrap();
    assert_eq!(
        candidates.collections.len(),
        usize::from(original == Original::Eligible)
    );
    assert_eq!(
        COLLECTION_WORK.with(Cell::get).1,
        usize::from(original != Original::Retained)
    );
    assert!(
        storage
            .take_collections(&mut source)
            .unwrap()
            .collections
            .is_empty()
    );
    assert_eq!(storage.allocations(), original_allocations);
    if original != Original::Eligible {
        return;
    }
    let range = candidates.collections[&PhysicalAddress::new(collection)].range;
    let mut attributes = CollectionCandidates::default();
    attributes
        .insert(
            source
                .collection_allocation(PhysicalAddress::new(collection))
                .unwrap(),
            [origin(owner, Some("removed"))],
        )
        .unwrap();
    candidates.extend(attributes).unwrap();
    assert_eq!(
        candidates.collections[&range.start()].origins,
        BTreeSet::from([origin(owner, None), origin(owner, Some("removed"))])
    );
    drop(source);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut prospective = storage
        .reference_phase(&engine, PhysicalAddress::new(replacement), &BTreeMap::new())
        .unwrap();
    assert_eq!(
        prospective.unreferenced_collection_spans(candidates),
        [(range, FreeClass::Page(PageType::Raw))]
    );
    assert_eq!(prospective.work().liveness_results, 1);
    assert_eq!(COLLECTION_WORK.with(Cell::get), (1, 1));
    assert_eq!(storage.allocations(), original_allocations);
}

#[rstest]
#[case::unchanged(Retention::Unchanged)]
#[case::relocated(Retention::Relocated)]
#[case::incomplete(Retention::Incomplete)]
#[case::replacement_root(Retention::Root)]
fn retained_collection_roots_describe_prospective_values_and_share_one_scan(
    #[case] retention: Retention,
) {
    let path = test_util::temp::temp_path("retained-collection-roots.h5");
    let mut fixture = if retention == Retention::Root {
        Fixture::new().with_root_reference_count(NonZeroU32::new(2).unwrap())
    } else {
        Fixture::new()
    };
    let old = fixture.collection(&[Object::new(1, b"old")]);
    let new = fixture.collection(&[Object::new(1, b"new")]);
    let mut messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(3, old, 1),
    );
    messages.push(Message::reference_count(if retention == Retention::Root {
        1
    } else {
        2
    }));
    if retention == Retention::Incomplete {
        messages.push(Message::new(MessageType(0x7f), &[]));
    }
    let owner = fixture.append_header(messages);
    fixture.link("a", owner);
    if retention == Retention::Root {
        fixture.link("self", Fixture::ROOT);
    } else if retention == Retention::Relocated {
        fixture.link("b", owner);
    }
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(3, new, 1),
    );
    let moved = fixture.append_header(messages);
    let replacement = fixture.append_header(if retention == Retention::Relocated {
        vec![global_heap_fixture::hard_link("b", moved)]
    } else {
        vec![]
    });
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut source = AnalysisPhase::new(&engine);
    let storage = storage(
        &mut source,
        if retention == Retention::Root {
            &["self", "a"]
        } else {
            &["a"]
        },
    );
    assert_eq!(
        storage.reference_roots(),
        [id(if retention == Retention::Root {
            Fixture::ROOT
        } else {
            owner
        })]
    );
    let mut candidates = CollectionCandidates::default();
    let old_allocation = source
        .collection_allocation(PhysicalAddress::new(old))
        .unwrap();
    let old_range = old_allocation.range;
    candidates
        .insert(old_allocation, [origin(owner, None)])
        .unwrap();
    let new_allocation = source
        .collection_allocation(PhysicalAddress::new(new))
        .unwrap();
    let new_range = new_allocation.range;
    candidates
        .insert(new_allocation, [origin(owner, Some("replaced"))])
        .unwrap();
    drop(source);
    let relocations = if retention == Retention::Relocated {
        BTreeMap::from([(owner, moved)])
    } else {
        BTreeMap::new()
    };
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut prospective = storage
        .reference_phase(&engine, PhysicalAddress::new(replacement), &relocations)
        .unwrap();
    assert!(
        prospective
            .unreferenced_collection_spans(CollectionCandidates::default())
            .is_empty()
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (0, 0));
    let retired = prospective.unreferenced_collection_spans(candidates);
    let expected = match retention {
        Retention::Unchanged => vec![(new_range, FreeClass::Page(PageType::Raw))],
        Retention::Relocated => vec![(old_range, FreeClass::Page(PageType::Raw))],
        Retention::Incomplete => vec![],
        Retention::Root => vec![
            (old_range, FreeClass::Page(PageType::Raw)),
            (new_range, FreeClass::Page(PageType::Raw)),
        ],
    };
    assert_eq!(retired, expected);
    let visited = if retention == Retention::Root { 1 } else { 2 };
    assert_eq!(COLLECTION_WORK.with(Cell::get), (visited, visited));
    assert_eq!(
        prospective.work().reference_objects,
        u64::try_from(visited).unwrap()
    );
    assert_eq!(prospective.work().liveness_results, 1);
    let facts = prospective.liveness.as_ref().unwrap().as_ref().unwrap();
    assert_eq!(facts.analysis_complete, retention != Retention::Incomplete);
    assert_eq!(
        facts.referenced_collections,
        match retention {
            Retention::Root => BTreeSet::new(),
            Retention::Relocated => BTreeSet::from([new]),
            _ => BTreeSet::from([old]),
        }
    );
    let mut later = CollectionCandidates::default();
    later
        .insert(
            AnalysisPhase::new(&engine)
                .collection_allocation(PhysicalAddress::new(old))
                .unwrap(),
            [origin(owner, Some("later"))],
        )
        .unwrap();
    let expected_later = if matches!(retention, Retention::Root | Retention::Relocated) {
        vec![(old_range, FreeClass::Page(PageType::Raw))]
    } else {
        vec![]
    };
    assert_eq!(
        prospective.unreferenced_collection_spans(later),
        expected_later
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (visited, visited));
}

#[test]
fn count_collection_handoffs_reject_other_phases_and_sessions_without_losing_candidates() {
    let path = test_util::temp::temp_path("count-collection-association.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    let owner = fixture.append_header(messages);
    fixture.link("a", owner);
    fixture.write(&path);
    let other_path = test_util::temp::temp_path("other-count-collection-session.h5");
    fs::copy(&path, &other_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let other_engine = edit_test_util::open_engine(&other_path);
    let mut source = AnalysisPhase::new(&engine);
    let mut storage = storage(&mut source, &["a"]);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let error = match storage.take_collections(&mut AnalysisPhase::new(&engine)) {
        Err(error) => error,
        Ok(_) => panic!("expected a source phase mismatch"),
    };
    assert_eq!(
        error,
        FormatError::Internal("count collection candidates belong to another source phase".into())
    );
    let error = match storage.reference_phase(
        &other_engine,
        PhysicalAddress::new(Fixture::ROOT),
        &BTreeMap::new(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("expected an allocation session mismatch"),
    };
    assert_eq!(
        error,
        FormatError::Internal(
            "retained collection roots belong to another allocation session".into()
        )
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (0, 0));
    assert_eq!(
        storage
            .take_collections(&mut source)
            .unwrap()
            .collections
            .len(),
        1
    );
}

#[test]
fn retained_root_handoffs_reexamine_values_after_writes_and_allocator_restoration() {
    let path = test_util::temp::temp_path("retained-root-source-phase.h5");
    let mut fixture = Fixture::new();
    let old = fixture.collection(&[Object::new(1, b"old")]);
    let new = fixture.collection(&[Object::new(1, b"new")]);
    let payload = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let before = global_heap_fixture::reference(3, old, 1);
    let after = global_heap_fixture::reference(3, new, 1);
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &before,
    );
    messages.push(Message::reference_count(2));
    let owner = fixture.append_header(messages);
    fixture.link("a", owner);
    let replacement = fixture.append_header(vec![]);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let mut source = AnalysisPhase::new(&engine);
    let storage = storage(&mut source, &["a"]);
    let range = source
        .collection_allocation(PhysicalAddress::new(old))
        .unwrap()
        .range;
    drop(source);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let run = |engine: &WriteEngine| {
        let mut phase = storage
            .reference_phase(engine, PhysicalAddress::new(replacement), &BTreeMap::new())
            .unwrap();
        let mut candidates = CollectionCandidates::default();
        candidates
            .insert(
                phase
                    .collection_allocation(PhysicalAddress::new(old))
                    .unwrap(),
                [origin(owner, None)],
            )
            .unwrap();
        phase.unreferenced_collection_spans(candidates)
    };
    assert!(run(&engine).is_empty());
    let snapshot = engine.snapshot_free();
    engine.image.write_at(payload, &after).unwrap();
    assert_eq!(run(&engine), [(range, FreeClass::Page(PageType::Raw))]);
    engine.restore_free(snapshot);
    assert_eq!(run(&engine), [(range, FreeClass::Page(PageType::Raw))]);
    assert_eq!(COLLECTION_WORK.with(Cell::get), (6, 6));
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Original {
    Eligible,
    Incomplete,
    Retained,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Retention {
    Incomplete,
    Relocated,
    Root,
    Unchanged,
}

fn storage(phase: &mut AnalysisPhase<'_>, paths: &[&str]) -> CountStorage {
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletions = paths
        .iter()
        .map(|path| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(path))
                .unwrap()
        })
        .collect::<Vec<_>>();
    phase
        .net_count_plan(&deletions, &added)
        .unwrap()
        .unwrap()
        .collect_storage()
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

fn origin(address: u64, attribute: Option<&str>) -> CollectionProvenance {
    CollectionProvenance {
        object: id(address),
        attribute: attribute.map(String::from),
    }
}
