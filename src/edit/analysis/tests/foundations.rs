use core::cell::Cell;
use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{FreeClass, PageType};
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::super::{
    AnalysisPhase, AnalysisWork, COLLECTION_WORK, CollectionCandidates, CollectionProvenance,
};
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;

#[rstest]
#[case::complete(false, "foundation_analysis_complete")]
#[case::partial(true, "foundation_analysis_partial")]
fn foundation_analysis_preserves_candidate_origins_and_completeness(
    #[case] partial: bool,
    #[case] snapshot: &str,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collections.h5");
    let mut fixture = Fixture::new();
    let dead = fixture.collection(&[Object::new(1, b"dead")]);
    let dead_range = FileRange::new(
        PhysicalAddress::new(dead),
        u64::try_from(fixture.image.as_bytes().len()).unwrap() - dead,
    )
    .unwrap();
    let live = fixture.collection(&[Object::new(1, b"live")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(4, live, 1),
    );
    let dataset = fixture.append_header(messages);
    fixture.link("dataset", dataset);
    fixture.link("alias", dataset);
    if partial {
        let unknown = fixture.append_header(vec![Message::new(MessageType(0x7f), &[])]);
        fixture.link("unknown", unknown);
    }
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut predecessor = AnalysisPhase::new(&engine);
    let dead_allocation = predecessor
        .collection_allocation(PhysicalAddress::new(dead))
        .unwrap();
    assert_eq!(dead_allocation.range, dead_range);
    let mut candidates = CollectionCandidates::default();
    let origin = |attribute: Option<&str>| CollectionProvenance {
        object: ObjectId::new(PhysicalAddress::new(dataset)),
        attribute: attribute.map(String::from),
    };
    candidates.insert(dead_allocation, [origin(None)]).unwrap();
    candidates
        .insert(
            predecessor
                .collection_allocation(PhysicalAddress::new(dead))
                .unwrap(),
            [origin(Some("prior"))],
        )
        .unwrap();
    candidates
        .insert(
            predecessor
                .collection_allocation(PhysicalAddress::new(live))
                .unwrap(),
            [origin(Some("shared"))],
        )
        .unwrap();
    let candidate_count = candidates.collections.len();
    let origin_count: usize = candidates
        .collections
        .values()
        .map(|candidate| candidate.origins.len())
        .sum();
    assert_eq!(candidate_count, 2);
    assert_eq!(origin_count, 3);
    let captured = predecessor.work();
    assert_eq!(captured.allocation_results, 2);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut prospective = AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT));
    assert_eq!(prospective.topology().unwrap().incoming(dataset), Some(2));
    let topology = prospective.work();
    assert_eq!(topology.topology_objects, if partial { 3 } else { 2 });
    assert_eq!(topology.topology_edges, if partial { 3 } else { 2 });
    assert_eq!(topology.reference_objects, 0);
    let retired = prospective.unreferenced_collection_spans(candidates);
    assert_eq!(
        retired,
        if partial {
            Vec::new()
        } else {
            vec![(dead_range, FreeClass::Page(PageType::Raw))]
        }
    );
    let first = prospective.work();
    let parsed = COLLECTION_WORK.with(Cell::get);
    assert_eq!(parsed, if partial { (3, 3) } else { (2, 2) });
    assert_eq!(first.reference_objects, topology.topology_objects);
    assert_eq!(first.collection_references, 1);
    assert_eq!(first.liveness_results, 1);
    assert_eq!(first.candidate_lookups, if partial { 0 } else { 2 });
    let references = prospective.liveness.as_ref().unwrap().as_ref().unwrap();
    assert_eq!(references.analysis_complete, !partial);
    assert_eq!(references.referenced_collections, BTreeSet::from([live]));
    let mut later = CollectionCandidates::default();
    later
        .insert(
            predecessor
                .collection_allocation(PhysicalAddress::new(dead))
                .unwrap(),
            [origin(Some("later"))],
        )
        .unwrap();
    assert_eq!(prospective.unreferenced_collection_spans(later), retired);
    let later_work = prospective.work();
    assert_eq!(
        later_work,
        AnalysisWork {
            candidate_lookups: if partial { 0 } else { 3 },
            ..first
        }
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), parsed);
    let report = format!(
        "unique candidates: {candidate_count}\nrecorded origins: {origin_count}\npredecessor captured state:\n{captured:#?}\nprospective topology:\n{topology:#?}\nprospective references:\n{first:#?}\nparsed links/references: {parsed:?}\nreference analysis complete: {}\nretired allocations:\n{retired:#?}\nlater candidate observations:\n{later_work:#?}\nlater candidates reuse result: true",
        !partial
    );
    drop(prospective);
    drop(predecessor);
    drop(engine);
    insta::assert_snapshot!(snapshot, report);
}
