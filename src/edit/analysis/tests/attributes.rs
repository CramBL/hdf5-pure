use core::cell::Cell;
use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::attribute::Attribute;
use test_util::dataspace;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::AnalysisPhase;
use super::super::COLLECTION_ALLOCATIONS;
use super::super::CollectionAllocation;
use super::super::CollectionCandidates;
use super::super::CollectionProvenance;
use crate::AttrValue;
use crate::edit::WriteEngine;
use crate::edit::attributes;
use crate::edit::attributes::AttrOp;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

#[test]
fn attribute_and_deletion_origins_share_one_allocation_and_liveness_result() {
    let path = test_util::temp::temp_path("attribute-union.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"before")]);
    let owner = fixture.append_header(vec![
        attribute("one", collection),
        attribute("two", collection),
    ]);
    fixture.link("owner", owner);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, collection, 1),
    );
    let deleted = fixture.append_header(messages);
    fixture.link("deleted", deleted);
    let root = fixture.append_header(vec![Message::new(
        MessageType::LINK_INFO,
        &format::LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: None,
            btree_name_index_address: None,
            btree_creation_order_address: None,
        }
        .serialize(format::OffsetWidth::Eight),
    )]);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let region =
        WriteEngine::gather_oh_messages(engine.image(), owner, engine.context.addresses()).unwrap();
    let operations = [
        AttrOp::Remove { name: "one".into() },
        AttrOp::Remove { name: "two".into() },
    ];
    let edited = attributes::plan_attr_ops(
        &engine.opened_source(),
        &region,
        &operations.iter().collect::<Vec<_>>(),
    )
    .unwrap();
    COLLECTION_ALLOCATIONS.with(|count| count.set(0));
    let mut phase = AnalysisPhase::new(&engine);
    let mut candidates = phase
        .attribute_collections(
            ObjectId::new(PhysicalAddress::new(owner)),
            edited.predecessors,
        )
        .unwrap();
    candidates.extend(CollectionCandidates::default()).unwrap();
    candidates
        .extend(phase.deleted_collections(&[deleted]))
        .unwrap();
    assert_eq!(COLLECTION_ALLOCATIONS.with(Cell::get), 1);
    assert_eq!(candidates.collections.len(), 1);
    let range = candidates.collections[&PhysicalAddress::new(collection)].range;
    assert_eq!(
        range.length(),
        u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap()
    );
    assert_eq!(
        candidates.collections[&PhysicalAddress::new(collection)].origins,
        BTreeSet::from([
            origin(owner, Some("one")),
            origin(owner, Some("two")),
            origin(deleted, None),
        ])
    );
    drop(phase);
    let mut prospective = AnalysisPhase::prospective(&engine, PhysicalAddress::new(root));
    assert_eq!(
        prospective.unreferenced_collection_spans(candidates),
        [(range, FreeClass::Page(PageType::Raw))]
    );
    assert_eq!(prospective.work().reference_objects, 1);
    assert_eq!(prospective.work().liveness_results, 1);
    assert_eq!(prospective.work().candidate_lookups, 1);
}

#[test]
fn allocator_restoration_does_not_restore_attribute_reference_facts() {
    let path = test_util::temp::temp_path("attribute-source-phase.h5");
    let mut fixture = Fixture::new();
    let first = fixture.collection(&[Object::new(1, b"before")]);
    let second = fixture.collection(&[Object::new(1, b"after!")]);
    let original = Header::new().message(attribute("one", first)).build();
    let replacement = Header::new().message(attribute("one", second)).build();
    assert_eq!(original.len(), replacement.len());
    let owner = fixture.image.append(&original);
    let owner = u64::try_from(owner).unwrap();
    fixture.link("owner", owner);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let root = engine.context.superblock().root_group_address;
    let region =
        WriteEngine::gather_oh_messages(engine.image(), owner, engine.context.addresses()).unwrap();
    let edited = attributes::plan_attr_ops(
        &engine.opened_source(),
        &region,
        &[&AttrOp::Remove { name: "one".into() }],
    )
    .unwrap();
    let mut predecessor = AnalysisPhase::new(&engine);
    let candidates = predecessor
        .attribute_collections(
            ObjectId::new(PhysicalAddress::new(owner)),
            edited.predecessors,
        )
        .unwrap();
    let range = candidates.collections[&PhysicalAddress::new(first)].range;
    assert_eq!(
        predecessor
            .collections_reachable_from([Fixture::ROOT])
            .unwrap()
            .referenced_collections,
        BTreeSet::from([first])
    );
    drop(predecessor);
    let snapshot = engine.snapshot_free();
    engine.image.write_at(owner, &replacement).unwrap();
    engine.restore_free(snapshot);
    assert_eq!(engine.context.superblock().root_group_address, root);
    let mut prospective = AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT));
    assert_eq!(
        prospective.unreferenced_collection_spans(candidates),
        [(range, FreeClass::Page(PageType::Raw))]
    );
    let references = prospective.liveness.as_ref().unwrap().as_ref().unwrap();
    assert!(references.analysis_complete);
    assert_eq!(references.referenced_collections, BTreeSet::from([second]));
    drop(prospective);
    engine.image.write_at(owner, &original).unwrap();
    assert_eq!(
        AnalysisPhase::new(&engine)
            .collections_reachable_from([Fixture::ROOT])
            .unwrap()
            .referenced_collections,
        BTreeSet::from([first])
    );
}

#[rstest]
#[case::invalid_collection(UnknownEvidence::Collection)]
#[case::unsupported_descriptor(UnknownEvidence::Descriptor)]
#[case::malformed_element(UnknownEvidence::Element)]
fn unknown_attribute_evidence_does_not_reject_preparation(#[case] evidence: UnknownEvidence) {
    let path = test_util::temp::temp_path("unknown-attribute-evidence.h5");
    let mut fixture = Fixture::new();
    let message = match evidence {
        UnknownEvidence::Descriptor => Message::new(
            MessageType::ATTRIBUTE,
            &Attribute::new(
                "one",
                &format::serialize_datatype(&global_heap_fixture::object_reference()).unwrap(),
                &dataspace::scalar(),
                &[0; 8],
            )
            .build(),
        ),
        UnknownEvidence::Collection => attribute("one", Fixture::ROOT),
        UnknownEvidence::Element => attribute("one", u64::MAX),
    };
    let owner = fixture.append_header(vec![message]);
    fixture.link("owner", owner);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let region =
        WriteEngine::gather_oh_messages(engine.image(), owner, engine.context.addresses()).unwrap();
    let edited = attributes::plan_attr_ops(
        &engine.opened_source(),
        &region,
        &[&AttrOp::Set {
            name: "one".into(),
            value: AttrValue::I64(19),
        }],
    )
    .unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let candidates = phase
        .attribute_collections(
            ObjectId::new(PhysicalAddress::new(owner)),
            edited.predecessors,
        )
        .unwrap();
    assert!(candidates.collections.is_empty());
    assert_eq!(
        phase.work().allocation_results,
        u64::from(matches!(evidence, UnknownEvidence::Collection))
    );
}

#[rstest]
#[case::different_session(Conflict::Session)]
#[case::different_length(Conflict::Length)]
fn candidate_merging_preserves_session_and_geometry_checks(#[case] conflict: Conflict) {
    let path = test_util::temp::temp_path("merge-candidates.h5");
    let other_path = test_util::temp::temp_path("merge-other-session.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"before")]);
    fixture.write(&path);
    std::fs::copy(&path, &other_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let other = edit_test_util::open_engine(&other_path);
    let mut phase = AnalysisPhase::new(&engine);
    let allocation = phase
        .collection_allocation(PhysicalAddress::new(collection))
        .unwrap();
    let range = allocation.range;
    let mut first = CollectionCandidates::default();
    first
        .insert(allocation, [origin(Fixture::ROOT, Some("one"))])
        .unwrap();
    let allocation = match conflict {
        Conflict::Length => CollectionAllocation {
            session: engine.context.allocations.session(),
            range: FileRange::new(range.start(), range.length() + 1).unwrap(),
        },
        Conflict::Session => AnalysisPhase::new(&other)
            .collection_allocation(PhysicalAddress::new(collection))
            .unwrap(),
    };
    let mut second = CollectionCandidates::default();
    second
        .insert(allocation, [origin(Fixture::ROOT, Some("two"))])
        .unwrap();
    let expected = match conflict {
        Conflict::Length => "collection candidates have conflicting allocation lengths",
        Conflict::Session => "collection candidates belong to different sessions",
    };
    assert_eq!(
        first.extend(second).unwrap_err(),
        FormatError::Internal(expected.into())
    );
    assert_eq!(first.collections.len(), 1);
    assert_eq!(first.collections[&range.start()].range, range);
    assert_eq!(
        first.collections[&range.start()].origins,
        BTreeSet::from([origin(Fixture::ROOT, Some("one"))])
    );
}

#[derive(Clone, Copy, Debug)]
enum Conflict {
    Length,
    Session,
}

#[derive(Clone, Copy, Debug)]
enum UnknownEvidence {
    Collection,
    Descriptor,
    Element,
}

/// Returns a scalar string Attribute record referring to object 1 in `collection`.
pub(super) fn attribute(name: &str, collection: u64) -> Message {
    Message::new(
        MessageType::ATTRIBUTE,
        &Attribute::new(
            name,
            &format::serialize_datatype(&global_heap_fixture::vlen_string()).unwrap(),
            &dataspace::scalar(),
            &global_heap_fixture::reference(6, collection, 1),
        )
        .build(),
    )
}

fn origin(object: u64, attribute: Option<&str>) -> CollectionProvenance {
    CollectionProvenance {
        object: ObjectId::new(PhysicalAddress::new(object)),
        attribute: attribute.map(String::from),
    }
}
