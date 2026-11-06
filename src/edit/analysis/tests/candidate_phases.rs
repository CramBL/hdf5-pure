use std::collections::BTreeSet;
use std::sync::Arc;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::super::{AnalysisPhase, CollectionCandidates, CollectionContent, CollectionProvenance};
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;

#[rstest]
#[case::new_source(OtherPhase::Source)]
#[case::same_prospective_root(OtherPhase::SameRoot)]
#[case::different_prospective_root(OtherPhase::DifferentRoot)]
#[case::different_session(OtherPhase::Session)]
fn candidate_union_preserves_original_phase_and_session(#[case] other_phase: OtherPhase) {
    let path = test_util::temp::temp_path("candidate-phase-union.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(7, b"before")]);
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, collection, 7),
    );
    let object = fixture.append_header(messages);
    fixture.link("d", object);
    fixture.write(&path);
    let other_path = test_util::temp::temp_path("candidate-other-session.h5");
    std::fs::copy(&path, &other_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let other_engine = edit_test_util::open_engine(&other_path);
    let mut original = AnalysisPhase::new(&engine);
    let source = original
        .reconstruction_source(&engine, super::object(object))
        .unwrap();
    let mut candidates = CollectionCandidates::default();
    candidates
        .extend(original.deleted_collections(&[super::object(object)]))
        .unwrap();
    let range = candidates.collections[&PhysicalAddress::new(collection)].range;
    let class = candidates.collections[&PhysicalAddress::new(collection)].class;
    let mut other = match other_phase {
        OtherPhase::Source => AnalysisPhase::new(&engine),
        OtherPhase::SameRoot => {
            AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT))
        }
        OtherPhase::DifferentRoot => {
            AnalysisPhase::prospective(&engine, PhysicalAddress::new(object))
        }
        OtherPhase::Session => AnalysisPhase::new(&other_engine),
    };
    let incoming = other.deleted_collections(&[super::object(object)]);
    assert_eq!(
        candidates.extend(incoming).unwrap_err(),
        FormatError::Internal(
            match other_phase {
                OtherPhase::Session => "collection candidates belong to different sessions",
                _ => "collection candidates belong to different source phases",
            }
            .into()
        )
    );
    let observations = candidates.observations.as_ref().unwrap();
    assert!(Arc::ptr_eq(
        observations.identity.as_ref().unwrap(),
        &source.identity
    ));
    assert_eq!(observations.predecessors.len(), 1);
    assert!(observations.incomplete_deletions.is_empty());
    assert_eq!(candidates.collections.len(), 1);
    assert_eq!(candidates.collections[&range.start()].range, range);
    assert_eq!(candidates.collections[&range.start()].class, class);
    candidates
        .extend(original.deleted_collections(&[super::object(object)]))
        .unwrap();
    assert_eq!(
        candidates.observations.as_ref().unwrap().predecessors.len(),
        1
    );
}

#[rstest]
#[case::synthetic_first(true)]
#[case::observed_first(false)]
fn allocation_only_candidates_do_not_create_predecessor_observations(
    #[case] synthetic_first: bool,
) {
    let path = test_util::temp::temp_path("candidate-phase-synthetic.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(7, b"before")]);
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, collection, 7),
    );
    let object = fixture.append_header(messages);
    fixture.link("d", object);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let mut synthetic = CollectionCandidates::default();
    let origin = CollectionProvenance {
        object: super::object(Fixture::ROOT),
        content: CollectionContent::DatasetValues,
    };
    synthetic
        .insert(
            phase
                .collection_allocation(PhysicalAddress::new(collection))
                .unwrap(),
            [origin.clone()],
        )
        .unwrap();
    let observations = synthetic.observations.as_ref().unwrap();
    assert!(observations.identity.is_none());
    assert!(observations.predecessors.is_empty());
    assert!(observations.incomplete_deletions.is_empty());
    assert!(phase.identity.is_none());
    let mut observed = phase.deleted_collections(&[super::object(object)]);
    let combined = if synthetic_first {
        synthetic.extend(observed).unwrap();
        synthetic
    } else {
        observed.extend(synthetic).unwrap();
        observed
    };
    let observations = combined.observations.as_ref().unwrap();
    assert!(Arc::ptr_eq(
        observations.identity.as_ref().unwrap(),
        phase.identity.as_ref().unwrap()
    ));
    assert_eq!(
        observations
            .predecessors
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([CollectionProvenance {
            object: super::object(object),
            content: CollectionContent::Object,
        }])
    );
    assert!(!observations.predecessors.contains_key(&origin));
    assert_eq!(
        combined.collections[&PhysicalAddress::new(collection)]
            .origins
            .len(),
        2
    );
}

#[derive(Clone, Copy)]
enum OtherPhase {
    DifferentRoot,
    SameRoot,
    Session,
    Source,
}
