use std::path::Path;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use crate::Datatype;
use crate::FixedPointType;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::TraversalBudget;

use super::*;

#[rstest]
#[case::implicit_dataset(Subject::Dataset, None, 1, Some(OrdinaryCountSubject::Dataset))]
#[case::aliased_dataset(Subject::Dataset, Some(2), 2, Some(OrdinaryCountSubject::Dataset))]
#[case::excess_dataset(Subject::Dataset, Some(7), 2, Some(OrdinaryCountSubject::Dataset))]
#[case::aliased_group(Subject::Group, Some(2), 2, Some(OrdinaryCountSubject::Group))]
#[case::excess_group(Subject::Group, Some(u32::MAX), 1, Some(OrdinaryCountSubject::Group))]
#[case::unclassified(Subject::Unclassified, Some(3), 1, None)]
#[case::conflicting(Subject::Conflicting, Some(2), 1, None)]
#[case::missing_dataset_message(Subject::MissingDatasetMessage, Some(2), 1, None)]
#[case::repeated_dataset_message(Subject::RepeatedDatasetMessage, Some(2), 1, None)]
fn source_count_evidence_preserves_original_bounds_and_supported_subjects(
    #[case] subject: Subject,
    #[case] count: Option<u32>,
    #[case] aliases: u32,
    #[case] expected_subject: Option<OrdinaryCountSubject>,
) {
    let path = test_util::temp::temp_path("count-evidence.h5");
    let address = write_fixture(&path, subject, count, aliases);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let object = ObjectId::new(PhysicalAddress::new(address));
    let evidence = phase.count_evidence(object).unwrap().unwrap();
    let bound = evidence.bound();
    assert_eq!(bound.object(), object);
    assert_eq!(bound.original(), count.unwrap_or(1));
    assert_eq!(bound.incoming(), aliases);
    assert_eq!(bound.field().map(|field| field.value()), count);
    assert!(bound.matches(phase.topology().unwrap()));
    assert_eq!(
        match evidence {
            CountEvidence::Ordinary(ordinary) => Some(ordinary.subject()),
            CountEvidence::Retained(_) => None,
        },
        expected_subject
    );
    assert_eq!(phase.counts.len(), 1);
    assert_eq!(
        phase
            .count_evidence(object)
            .unwrap()
            .unwrap()
            .bound()
            .original(),
        count.unwrap_or(1)
    );
    assert_eq!(phase.counts.len(), 1);
}

#[rstest]
#[case::implicit(None, 2)]
#[case::zero(Some(0), 1)]
#[case::explicit(Some(1), 2)]
fn an_observed_undercount_returns_a_structured_contradiction(
    #[case] count: Option<u32>,
    #[case] incoming: u32,
) {
    let path = test_util::temp::temp_path("count-contradiction.h5");
    let address = write_fixture(&path, Subject::Dataset, count, incoming);
    let engine = edit_test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let object = ObjectId::new(PhysicalAddress::new(address));
    let Err(error) = phase.count_evidence(object) else {
        panic!("expected an undercount")
    };
    assert_eq!(
        error,
        CountEvidenceError::Contradiction(CountContradiction {
            object,
            observed: count.unwrap_or(1),
            incoming
        })
    );
    assert_eq!(phase.topology().unwrap().incoming(address), Some(incoming));
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[rstest]
#[case::ambiguous(true)]
#[case::unsupported(false)]
fn untrusted_count_forms_preserve_topology_without_bound_evidence(#[case] ambiguous: bool) {
    let path = test_util::temp::temp_path("untrusted-count-evidence.h5");
    let mut fixture = Fixture::new();
    let messages = if ambiguous {
        vec![Message::reference_count(2), Message::reference_count(3)]
    } else {
        vec![Message::new(
            MessageType::OBJECT_REFERENCE_COUNT,
            &[1, 2, 0, 0, 0],
        )]
    };
    let address = fixture.append_header(messages);
    fixture.link("value", address);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    assert!(
        phase
            .count_evidence(ObjectId::new(PhysicalAddress::new(address)))
            .unwrap()
            .is_none()
    );
    assert_eq!(phase.topology().unwrap().incoming(address), Some(1));
    assert_eq!(phase.counts.len(), 1);
}

#[rstest]
#[case::incomplete(FactLimit::Topology)]
#[case::root(FactLimit::Root)]
#[case::outside(FactLimit::Outside)]
fn missing_applicable_topology_or_nonroot_identity_supplies_no_count_evidence(
    #[case] limit: FactLimit,
) {
    let path = test_util::temp::temp_path("incomplete-count-evidence.h5");
    let address = write_fixture(&path, Subject::Dataset, Some(1), 1);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let object = ObjectId::new(PhysicalAddress::new(match limit {
        FactLimit::Outside => u64::MAX,
        FactLimit::Root => Fixture::ROOT,
        FactLimit::Topology => address,
    }));
    if matches!(limit, FactLimit::Topology) {
        phase.budget = TraversalBudget::new(1, usize::MAX);
        let ObjectReferenceCount::Version2(field) = phase.object_reference_count(object).unwrap()
        else {
            panic!("expected explicit count framing")
        };
        assert_eq!(field.value(), 1);
    }
    assert!(phase.count_evidence(object).unwrap().is_none());
    assert_eq!(
        phase.counts.len(),
        usize::from(matches!(limit, FactLimit::Topology))
    );
}

#[test]
fn equal_addresses_and_counts_do_not_match_another_topology_identity() {
    let path = test_util::temp::temp_path("count-identity.h5");
    let address = write_fixture(&path, Subject::Dataset, Some(1), 1);
    let engine = edit_test_util::open_engine(&path);
    let mut first = AnalysisPhase::new(&engine);
    let mut second = AnalysisPhase::new(&engine);
    let object = ObjectId::new(PhysicalAddress::new(address));
    let evidence = first.count_evidence(object).unwrap().unwrap();
    let other = second.count_evidence(object).unwrap().unwrap();
    assert_eq!(evidence.bound().object(), other.bound().object());
    assert_eq!(evidence.bound().original(), other.bound().original());
    assert!(evidence.bound().matches(first.topology().unwrap()));
    assert!(!evidence.bound().matches(second.topology().unwrap()));
    assert!(!other.bound().matches(first.topology().unwrap()));
}

#[test]
fn a_cached_source_failure_supplies_no_count_bound() {
    let path = test_util::temp::temp_path("failed-count-evidence.h5");
    let address = write_fixture(&path, Subject::Dataset, Some(1), 1);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    assert_eq!(phase.topology().unwrap().incoming(address), Some(1));
    let object = ObjectId::new(PhysicalAddress::new(address));
    let failure = FormatError::UnexpectedSourceEof {
        expected: address + 1,
        available: address,
    };
    phase.counts.insert(object, Err(failure.clone()));
    let Err(error) = phase.count_evidence(object) else {
        panic!("expected the cached source failure")
    };
    assert_eq!(error, CountEvidenceError::Source(failure.clone()));
    assert_eq!(phase.counts[&object], Err(failure));
    assert_eq!(phase.topology().unwrap().incoming(address), Some(1));
}

fn write_fixture(path: &Path, subject: Subject, count: Option<u32>, aliases: u32) -> u64 {
    let mut fixture = Fixture::new();
    let mut records = match subject {
        Subject::Conflicting
        | Subject::Dataset
        | Subject::MissingDatasetMessage
        | Subject::RepeatedDatasetMessage => Form::CompactDataset.messages(
            &mut fixture,
            &Datatype::FixedPoint(FixedPointType::i32()),
            &17_i32.to_le_bytes(),
        ),
        Subject::Group | Subject::Unclassified => vec![],
    };
    match subject {
        Subject::MissingDatasetMessage => {
            records.retain(|record| record.msg_type != MessageType::DATASPACE)
        }
        Subject::RepeatedDatasetMessage => {
            let layout = records
                .iter()
                .find(|record| record.msg_type == MessageType::DATA_LAYOUT)
                .unwrap()
                .clone();
            records.push(layout);
        }
        Subject::Conflicting | Subject::Dataset | Subject::Group | Subject::Unclassified => {}
    }
    if matches!(subject, Subject::Conflicting | Subject::Group) {
        let child = fixture.append_header(vec![]);
        records.push(global_heap_fixture::hard_link("child", child));
    }
    if let Some(count) = count {
        records.push(Message::reference_count(count));
    }
    let address = fixture.append_header(records);
    for index in 0..aliases {
        fixture.link(&format!("alias{index}"), address);
    }
    fixture.write(path);
    address
}

#[derive(Clone, Copy)]
enum FactLimit {
    Outside,
    Root,
    Topology,
}

#[derive(Clone, Copy)]
enum Subject {
    Conflicting,
    Dataset,
    Group,
    MissingDatasetMessage,
    RepeatedDatasetMessage,
    Unclassified,
}
