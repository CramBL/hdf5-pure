use core::cell::Cell;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::VariableLengthType;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::edit::analysis::{AnalysisPhase, COLLECTION_WORK};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::object_graph::{ObjectId, TraversalBudget};
use crate::{Datatype, ReferenceType};

#[rstest]
#[case::attribute(Form::CompactAttribute)]
#[case::compact(Form::CompactDataset)]
#[case::contiguous(Form::ContiguousDataset)]
#[case::fill(Form::FillValue)]
fn reachable_union_shares_parent_and_member_facts(#[case] form: Form) {
    let path = test_util::temp::temp_path("reachable-members.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(1, b"first"), Object::new(7, b"second")]);
    let records = form.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &[
            global_heap_fixture::reference(5, parent, 1),
            global_heap_fixture::reference(0, 0, 0),
        ]
        .concat(),
    );
    let first = fixture.append_header(records);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, parent, 7),
    );
    let second = fixture.append_header(records);
    let group = fixture.append_header(vec![
        global_heap_fixture::hard_link("first", first),
        global_heap_fixture::hard_link("back", Fixture::ROOT),
    ]);
    fixture.link("first", first);
    fixture.link("group", group);
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let parents = phase
        .collections_reachable_from([object(Fixture::ROOT)])
        .unwrap();
    assert!(parents.analysis_complete);
    assert_eq!(parents.referenced_collections, BTreeSet::from([parent]));
    let read_calls = calls.load(Ordering::Relaxed);
    let work = COLLECTION_WORK.with(Cell::get);
    assert_eq!(work, (3, 3));
    assert!(read_calls > 0);
    let first_union = phase.heap_references_reachable_from([
        object(Fixture::ROOT),
        object(group),
        object(first),
        object(Fixture::ROOT),
    ]);
    assert!(first_union.is_complete());
    assert!(first_union.error().is_none());
    assert_eq!(first_union.collections(), &parents.referenced_collections);
    assert_eq!(
        first_union.objects(),
        &BTreeSet::from([super::id(parent, 1)])
    );
    assert_eq!(calls.load(Ordering::Relaxed), read_calls);
    assert_eq!(COLLECTION_WORK.with(Cell::get), work);
    let union = phase.heap_references_reachable_from([object(first), object(second)]);
    assert!(union.is_complete());
    assert_eq!(union.collections(), &BTreeSet::from([parent]));
    assert_eq!(
        union.objects(),
        &BTreeSet::from([super::id(parent, 1), super::id(parent, 7)])
    );
    let read_calls = calls.load(Ordering::Relaxed);
    let work = COLLECTION_WORK.with(Cell::get);
    let repeated = phase.heap_references_reachable_from([object(second), object(first)]);
    assert_eq!(repeated.objects(), union.objects());
    assert_eq!(calls.load(Ordering::Relaxed), read_calls);
    assert_eq!(COLLECTION_WORK.with(Cell::get), work);
}

#[rstest]
#[case::compound(Nesting::Compound)]
#[case::array(Nesting::Array)]
fn reachable_union_preserves_nested_member_identities(#[case] nesting: Nesting) {
    let path = test_util::temp::temp_path("reachable-nested.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value")]);
    let reference = global_heap_fixture::reference(5, parent, 7);
    let (datatype, values) = match nesting {
        Nesting::Compound => (
            global_heap_fixture::with_integer(global_heap_fixture::vlen_string()),
            [vec![0; 4], reference].concat(),
        ),
        Nesting::Array => (
            global_heap_fixture::array_of(global_heap_fixture::vlen_string()),
            reference.repeat(2),
        ),
    };
    let records = Form::CompactDataset.messages(&mut fixture, &datatype, &values);
    let owner = fixture.append_header(records);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let union = AnalysisPhase::new(&engine).heap_references_reachable_from([object(owner)]);
    assert!(union.is_complete());
    assert!(union.error().is_none());
    assert_eq!(union.collections(), &BTreeSet::from([parent]));
    assert_eq!(union.objects(), &BTreeSet::from([super::id(parent, 7)]));
}

#[rstest]
#[case::object(global_heap_fixture::object_reference())]
#[case::region(Datatype::Reference { size: 12, ref_type: ReferenceType::DatasetRegion })]
#[case::nested_vlen(Datatype::VariableLength(VariableLengthType::sequence(
    global_heap_fixture::vlen_string()
)))]
fn unsupported_union_retains_positive_facts_without_completeness(#[case] datatype: Datatype) {
    let path = test_util::temp::temp_path("reachable-unsupported.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(1, b"value")]);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 1),
    );
    let known = fixture.append_header(records);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &datatype,
        &vec![0; usize::try_from(datatype.type_size()).unwrap()],
    );
    let unknown = fixture.append_header(records);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let union = AnalysisPhase::new(&engine)
        .heap_references_reachable_from([object(known), object(unknown)]);
    assert!(!union.is_complete());
    assert!(union.error().is_none());
    assert_eq!(union.collections(), &BTreeSet::from([parent]));
    assert_eq!(union.objects(), &BTreeSet::from([super::id(parent, 1)]));
}

#[rstest]
#[case::malformed(true)]
#[case::unknown_message(false)]
fn union_preserves_decoded_prefix_and_interpretation_outcome(#[case] malformed: bool) {
    let path = test_util::temp::temp_path("reachable-prefix.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(1, b"value")]);
    let mut values = global_heap_fixture::reference(5, parent, 1);
    if malformed {
        values.extend(global_heap_fixture::reference(1, parent, 0));
    }
    let mut records =
        Form::CompactDataset.messages(&mut fixture, &global_heap_fixture::vlen_string(), &values);
    if !malformed {
        records.push(Message::new(MessageType(0x7f), &[]));
    }
    let owner = fixture.append_header(records);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let union = phase.heap_references_reachable_from([object(owner)]);
    assert!(!union.is_complete());
    assert_eq!(union.collections(), &BTreeSet::from([parent]));
    assert_eq!(union.objects(), &BTreeSet::from([super::id(parent, 1)]));
    if malformed {
        let Some(Error::Format(error)) = union.error() else {
            panic!("expected descriptor error");
        };
        assert_eq!(
            error,
            &FormatError::GlobalHeapObjectNotFound {
                collection_address: parent,
                index: 0
            }
        );
        let Error::Format(error) = phase
            .collections_reachable_from([object(owner)])
            .unwrap_err()
        else {
            panic!("expected descriptor error");
        };
        assert_eq!(
            error,
            FormatError::GlobalHeapObjectNotFound {
                collection_address: parent,
                index: 0
            }
        );
    } else {
        assert!(union.error().is_none());
    }
}

#[rstest]
#[case::object_limit(TraversalBudget::new(1, usize::MAX))]
#[case::edge_limit(TraversalBudget::new(usize::MAX, 0))]
fn traversal_limits_withhold_union_completeness(#[case] budget: TraversalBudget) {
    let path = test_util::temp::temp_path("reachable-limit.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(1, b"value")]);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 1),
    );
    let child = fixture.append_header(Vec::new());
    records.push(global_heap_fixture::hard_link("child", child));
    let owner = fixture.append_header(records);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    phase.budget = budget;
    let union = phase.heap_references_reachable_from([object(owner)]);
    assert!(!union.is_complete());
    assert!(union.error().is_none());
    assert_eq!(union.objects(), &BTreeSet::from([super::id(parent, 1)]));
}

#[test]
fn failed_union_retains_decoded_identities_and_cached_source_failure() {
    let path = test_util::temp::temp_path("reachable-error.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(1, b"value")]);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 1),
    );
    let owner = fixture.append_header(records);
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let roots = [object(0), object(owner)];
    let union = phase.heap_references_reachable_from(roots);
    assert!(!union.is_complete());
    assert_eq!(union.collections(), &BTreeSet::from([parent]));
    assert_eq!(union.objects(), &BTreeSet::from([super::id(parent, 1)]));
    let Some(Error::Format(error)) = union.error() else {
        panic!("expected invalid object header");
    };
    assert_eq!(
        error,
        &FormatError::InvalidObjectHeaderVersion(test_util::superblock::SIGNATURE[0])
    );
    let read_calls = calls.load(Ordering::Relaxed);
    let repeated = phase.heap_references_reachable_from(roots);
    assert_eq!(repeated.objects(), union.objects());
    let Some(Error::Format(error)) = repeated.error() else {
        panic!("expected cached invalid object header");
    };
    assert_eq!(
        error,
        &FormatError::InvalidObjectHeaderVersion(test_util::superblock::SIGNATURE[0])
    );
    assert_eq!(calls.load(Ordering::Relaxed), read_calls);
}

fn object(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

#[derive(Clone, Copy)]
enum Nesting {
    Array,
    Compound,
}
