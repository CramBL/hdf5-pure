use core::cell::Cell;
use std::collections::BTreeSet;
use std::fs;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{FreeClass, PageType};
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::super::{
    AnalysisPhase, COLLECTION_ALLOCATIONS, COLLECTION_WORK, CollectionAllocation,
    CollectionCandidates, CollectionProvenance,
};
use crate::data_layout::DataLayout;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::{ObjectId, TraversalBudget};

#[rstest]
#[case::topology_first(true)]
#[case::references_first(false)]
fn phase_reuses_adjacency_and_reference_summaries(#[case] topology_first: bool) {
    let path = test_util::temp::temp_path("shared-references.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    let dataset = fixture.append_header(messages);
    let group = fixture.append_header(vec![
        global_heap_fixture::hard_link("d", dataset),
        global_heap_fixture::hard_link("back", Fixture::ROOT),
    ]);
    fixture.link("g", group);
    fixture.link("alias", dataset);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::new(&engine);
    if topology_first {
        assert_eq!(phase.topology().unwrap().incoming(dataset), Some(2));
    }
    let references = phase.collections_reachable_from([Fixture::ROOT]).unwrap();
    assert_eq!(
        references.referenced_collections,
        BTreeSet::from([collection])
    );
    assert!(references.analysis_complete);
    assert_eq!(phase.topology().unwrap().incoming(dataset), Some(2));
    let candidates = phase.deleted_collections(&[group, dataset]);
    assert_eq!(candidates.collections.len(), 1);
    let candidate = &candidates.collections[&PhysicalAddress::new(collection)];
    assert_eq!(
        candidate.origins,
        BTreeSet::from([CollectionProvenance {
            object: ObjectId::new(PhysicalAddress::new(dataset)),
            attribute: None,
        }])
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (3, 3));
}

#[test]
fn candidate_union_preserves_origins_and_reuses_complete_liveness() {
    let path = test_util::temp::temp_path("collection-union.h5");
    let mut fixture = Fixture::new();
    let first = fixture.collection(&[Object::new(1, b"first")]);
    let second = fixture.collection(&[Object::new(1, b"second")]);
    let live = fixture.collection(&[Object::new(1, b"live")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(4, live, 1),
    );
    let survivor = fixture.append_header(messages);
    fixture.link("live", survivor);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let provenance = |attribute: Option<&str>| CollectionProvenance {
        object: ObjectId::new(PhysicalAddress::new(survivor)),
        attribute: attribute.map(String::from),
    };
    let mut candidates = CollectionCandidates::default();
    let first_allocation = allocation(&engine, PhysicalAddress::new(first)).unwrap();
    let first_range = first_allocation.range;
    candidates
        .insert(first_allocation, [provenance(None)])
        .unwrap();
    candidates
        .insert(
            allocation(&engine, PhysicalAddress::new(first)).unwrap(),
            [provenance(Some("prior"))],
        )
        .unwrap();
    assert_eq!(candidates.collections.len(), 1);
    assert_eq!(
        candidates.collections[&PhysicalAddress::new(first)].origins,
        BTreeSet::from([provenance(None), provenance(Some("prior"))])
    );
    candidates
        .insert(
            allocation(&engine, PhysicalAddress::new(live)).unwrap(),
            [provenance(Some("shared"))],
        )
        .unwrap();
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT));
    assert_eq!(
        phase.unreferenced_collection_spans(candidates),
        [(first_range, FreeClass::Page(PageType::Raw))]
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (2, 2));
    let mut later = CollectionCandidates::default();
    let second_allocation = allocation(&engine, PhysicalAddress::new(second)).unwrap();
    let second_range = second_allocation.range;
    later
        .insert(second_allocation, [provenance(Some("later"))])
        .unwrap();
    assert_eq!(
        phase.unreferenced_collection_spans(later),
        [(second_range, FreeClass::Page(PageType::Raw))]
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (2, 2));
}

#[test]
fn incomplete_references_do_not_retire_absent_collections_or_invalidate_topology() {
    let path = test_util::temp::temp_path("partial-references.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let unknown = fixture.append_header(vec![Message::new(MessageType(0x7F), &[])]);
    fixture.link("unknown", unknown);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut candidates = CollectionCandidates::default();
    candidates
        .insert(
            allocation(&engine, PhysicalAddress::new(collection)).unwrap(),
            [CollectionProvenance {
                object: ObjectId::new(PhysicalAddress::new(unknown)),
                attribute: None,
            }],
        )
        .unwrap();
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT));
    assert_eq!(phase.topology().unwrap().incoming(unknown), Some(1));
    assert!(phase.unreferenced_collection_spans(candidates).is_empty());
    let Some(Ok(references)) = &phase.liveness else {
        panic!("expected partial reference facts");
    };
    assert!(!references.analysis_complete);
    assert_eq!(references.referenced_collections, BTreeSet::new());
    assert_eq!(phase.topology().unwrap().incoming(unknown), Some(1));
    assert_eq!(COLLECTION_WORK.with(Cell::get), (2, 2));
}

#[test]
fn empty_candidates_skip_source_and_liveness_work() {
    let path = test_util::temp::temp_path("empty-candidates.h5");
    let fixture = Fixture::new();
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::prospective(&engine, PhysicalAddress::new(u64::MAX));
    assert!(
        phase
            .unreferenced_collection_spans(CollectionCandidates::default())
            .is_empty()
    );
    assert!(phase.deleted_collections(&[]).collections.is_empty());
    assert!(phase.liveness.is_none());
    assert_eq!(COLLECTION_WORK.with(Cell::get), (0, 0));
}

#[test]
fn new_source_phase_reexamines_values_at_the_same_root_after_write_and_restore() {
    let path = test_util::temp::temp_path("source-phase.h5");
    let mut fixture = Fixture::new();
    let first = fixture.collection(&[Object::new(1, b"first")]);
    let second = fixture.collection(&[Object::new(1, b"second")]);
    let first_reference = global_heap_fixture::reference(5, first, 1);
    let second_reference = global_heap_fixture::reference(6, second, 1);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &first_reference,
    );
    let layout = messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let DataLayout::Contiguous {
        address: Some(payload),
        ..
    } = DataLayout::parse(&layout.data, 8, 8).unwrap()
    else {
        panic!("expected contiguous storage");
    };
    fixture.object("d", messages);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut before = AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT));
    assert_eq!(
        before
            .collections_reachable_from([Fixture::ROOT])
            .unwrap()
            .referenced_collections,
        BTreeSet::from([first])
    );
    drop(before);
    engine
        .image
        .write_at(payload.get(), &second_reference)
        .unwrap();
    let mut after = AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT));
    assert_eq!(
        after
            .collections_reachable_from([Fixture::ROOT])
            .unwrap()
            .referenced_collections,
        BTreeSet::from([second])
    );
    drop(after);
    engine
        .image
        .write_at(payload.get(), &first_reference)
        .unwrap();
    assert_eq!(
        AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT))
            .collections_reachable_from([Fixture::ROOT])
            .unwrap()
            .referenced_collections,
        BTreeSet::from([first])
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (6, 6));
}

#[rstest]
#[case::object_limit(TraversalBudget::new(1, usize::MAX))]
#[case::edge_limit(TraversalBudget::new(usize::MAX, 0))]
fn failed_topology_keeps_previously_discovered_reference_adjacency(
    #[case] budget: TraversalBudget,
) {
    let path = test_util::temp::temp_path("failed-topology.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    fixture.object("d", messages);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::new(&engine);
    assert!(
        phase
            .collections_reachable_from([Fixture::ROOT])
            .unwrap()
            .analysis_complete
    );
    let original = phase.budget;
    phase.budget = budget;
    assert!(phase.topology().is_none());
    phase.budget = original;
    let references = phase.collections_reachable_from([Fixture::ROOT]).unwrap();
    assert_eq!(
        references.referenced_collections,
        BTreeSet::from([collection])
    );
    assert!(references.analysis_complete);
    assert_eq!(COLLECTION_WORK.with(Cell::get), (2, 2));
}

#[test]
fn candidate_union_rejects_conflicting_complete_allocation_lengths() {
    let path = test_util::temp::temp_path("conflicting-candidates.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let allocation = allocation(&engine, PhysicalAddress::new(collection)).unwrap();
    let range = allocation.range;
    let mut candidates = CollectionCandidates::default();
    candidates
        .insert(allocation, [origin(Fixture::ROOT)])
        .unwrap();
    let error = candidates
        .insert(
            CollectionAllocation {
                session: engine.context.allocations.session(),
                range: FileRange::new(range.start(), range.length() + 1).unwrap(),
            },
            [origin(Fixture::ROOT)],
        )
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal("collection candidates have conflicting allocation lengths".into())
    );
    assert_eq!(candidates.collections[&range.start()].range, range);
}

#[test]
fn candidate_allocation_proofs_are_bound_to_their_editing_session() {
    let path = test_util::temp::temp_path("candidate-session.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    fixture.write(&path);
    let second_path = test_util::temp::temp_path("second-session.h5");
    fs::copy(&path, &second_path).unwrap();
    let first = edit_test_util::open_engine(&path);
    let second = edit_test_util::open_engine(&second_path);
    let mut candidates = CollectionCandidates::default();
    candidates
        .insert(
            allocation(&first, PhysicalAddress::new(collection)).unwrap(),
            [origin(Fixture::ROOT)],
        )
        .unwrap();
    let error = candidates
        .insert(
            allocation(&second, PhysicalAddress::new(collection)).unwrap(),
            [origin(Fixture::ROOT)],
        )
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal("collection candidates belong to different sessions".into())
    );
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    assert!(
        AnalysisPhase::prospective(&second, PhysicalAddress::new(Fixture::ROOT))
            .unreferenced_collection_spans(candidates)
            .is_empty()
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (0, 0));
}

#[test]
fn numeric_payloads_do_not_need_source_bytes_for_collection_liveness() {
    let path = test_util::temp::temp_path("numeric-reference-shortcut.h5");
    let mut fixture = Fixture::new();
    let mut messages = global_heap_fixture::dataset_messages(
        &crate::make_i32_type(),
        &test_util::dataspace::scalar(),
    );
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &test_util::data_layout::contiguous(65_536, 4),
    ));
    fixture.object("numeric", messages);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    assert!(engine.image.len() < 65_536);
    let mut phase = AnalysisPhase::new(&engine);
    let references = phase.collections_reachable_from([Fixture::ROOT]).unwrap();
    assert!(references.analysis_complete);
    assert_eq!(references.referenced_collections, BTreeSet::new());
}

#[test]
fn undefined_hard_link_addresses_do_not_become_physical_references() {
    let path = test_util::temp::temp_path("undefined-reference-link.h5");
    let mut fixture = Fixture::new();
    fixture.link("undefined", global_heap_fixture::UNDEFINED_ADDRESS);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let references = phase.collections_reachable_from([Fixture::ROOT]).unwrap();
    assert!(references.analysis_complete);
    assert_eq!(references.referenced_collections, BTreeSet::new());
    assert!(phase.topology().is_none());
}

fn origin(object: u64) -> CollectionProvenance {
    CollectionProvenance {
        object: ObjectId::new(PhysicalAddress::new(object)),
        attribute: None,
    }
}

#[test]
fn candidate_union_rejects_allocations_without_provenance() {
    let path = test_util::temp::temp_path("candidate-provenance.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut candidates = CollectionCandidates::default();
    let error = candidates
        .insert(
            allocation(&engine, PhysicalAddress::new(collection)).unwrap(),
            [],
        )
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal("collection candidate has no provenance".into())
    );
    assert!(candidates.collections.is_empty());
    assert!(candidates.session.is_none());
}

fn allocation(
    engine: &WriteEngine,
    address: PhysicalAddress,
) -> Result<CollectionAllocation, FormatError> {
    AnalysisPhase::new(engine).collection_allocation(address)
}

#[test]
fn collection_allocation_geometry_is_parsed_once_per_phase() {
    let path = test_util::temp::temp_path("collection-geometry.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let expected = FileRange::new(
        PhysicalAddress::new(collection),
        u64::try_from(fixture.image.as_bytes().len()).unwrap() - collection,
    )
    .unwrap();
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    COLLECTION_ALLOCATIONS.with(|count| count.set(0));
    let mut phase = AnalysisPhase::new(&engine);
    let first = phase
        .collection_allocation(PhysicalAddress::new(collection))
        .unwrap();
    let second = phase
        .collection_allocation(PhysicalAddress::new(collection))
        .unwrap();
    assert_eq!(first.range, expected);
    assert_eq!(second.range, expected);
    assert_eq!(COLLECTION_ALLOCATIONS.with(Cell::get), 1);
    let Err(other) = phase.collection_allocation(PhysicalAddress::new(Fixture::ROOT)) else {
        panic!("expected an invalid collection signature");
    };
    let Err(again) = phase.collection_allocation(PhysicalAddress::new(Fixture::ROOT)) else {
        panic!("expected an invalid collection signature");
    };
    assert_eq!(other, FormatError::InvalidGlobalHeapSignature);
    assert_eq!(other, again);
    assert_eq!(COLLECTION_ALLOCATIONS.with(Cell::get), 2);
}
