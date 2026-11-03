use core::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use crate::Datatype;
use crate::FileLocking;
use crate::FixedPointType;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::PlannedDeletion;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::image::CountingImage;
use crate::image::HandleImage;
use crate::object_graph::TraversalBudget;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;

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
    assert_eq!(phase.counts[&object], Err(failure.clone()));
    assert_eq!(phase.topology().unwrap().incoming(address), Some(1));
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("alias0"))
        .unwrap();
    let (removed, mut queries) = phase.removal_queries([&deletion]).unwrap();
    let Err(error) = queries.evidence(object) else {
        panic!("expected the cached source failure")
    };
    assert_eq!(error, CountEvidenceError::Source(failure));
    assert_eq!(
        removed.count_plan().unwrap().unwrap().remaining(object),
        Some(0)
    );
}

#[test]
fn bound_deletions_and_lazy_count_queries_share_disjoint_phase_fields() {
    let path = test_util::temp::temp_path("count-query-view.h5");
    let address = write_fixture(&path, Subject::Dataset, Some(7), 2);
    let calls = Arc::new(AtomicU64::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    MetadataCacheConfig::disabled(),
                )),
                bytes.clone(),
                Arc::default(),
            )
            .with_read_calls(calls.clone()),
        ))
    })
    .unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("alias0"))
        .unwrap();
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let (removed, mut queries) = phase.removal_queries([&deletion]).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(queries.counts.is_empty());
    let object = ObjectId::new(PhysicalAddress::new(address));
    let evidence = queries.evidence(object).unwrap().unwrap();
    assert_eq!(evidence.bound().original(), 7);
    assert_eq!(evidence.bound().incoming(), 2);
    assert!(evidence.bound().matches(queries.topology.unwrap()));
    let reads = calls.load(Ordering::Relaxed);
    let read_bytes = bytes.load(Ordering::Relaxed);
    assert!(reads > 0 && read_bytes > 0);
    let plan = removed.count_plan().unwrap().unwrap();
    assert_eq!(plan.remaining(object), Some(1));
    assert_eq!(plan.zero_link_objects().count(), 0);
    assert_eq!(
        queries
            .evidence(object)
            .unwrap()
            .unwrap()
            .bound()
            .original(),
        7
    );
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert_eq!(bytes.load(Ordering::Relaxed), read_bytes);
    assert_eq!(queries.counts.len(), 1);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
}

#[rstest]
#[case::empty(false)]
#[case::incomplete(true)]
fn deletion_query_views_do_not_promote_inapplicable_count_facts(#[case] incomplete: bool) {
    let path = test_util::temp::temp_path("inapplicable-count-view.h5");
    let address = write_fixture(&path, Subject::Dataset, Some(1), 1);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    if incomplete {
        phase.budget = TraversalBudget::new(1, usize::MAX);
    }
    let object = ObjectId::new(PhysicalAddress::new(address));
    let deletion =
        PlannedDeletion::unresolved(LinkNameBuf::new("alias0").unwrap(), object.header_address());
    let deletions = incomplete.then_some(&deletion);
    let (removed, mut queries) = phase.removal_queries(deletions).unwrap();
    assert!(queries.topology.is_none());
    assert!(queries.evidence(object).unwrap().is_none());
    assert!(queries.counts.is_empty());
    assert!(removed.count_plan().unwrap().is_none());
    let ObjectReferenceCount::Version2(field) = queries.observation(object).unwrap() else {
        panic!("expected explicit framing")
    };
    assert_eq!(field.value(), 1);
}

#[test]
fn count_query_views_reject_a_topology_from_another_engine() {
    let first_path = test_util::temp::temp_path("first-count-view.h5");
    let second_path = test_util::temp::temp_path("second-count-view.h5");
    let address = write_fixture(&first_path, Subject::Dataset, Some(2), 2);
    assert_eq!(
        write_fixture(&second_path, Subject::Dataset, Some(7), 2),
        address
    );
    let first = edit_test_util::open_engine(&first_path);
    let second = edit_test_util::open_engine(&second_path);
    let mut phase = AnalysisPhase::new(&second);
    let mut counts = BTreeMap::new();
    let mut queries = CountQueries::new(&first, &mut counts, phase.topology());
    let Err(error) = queries.evidence(ObjectId::new(PhysicalAddress::new(address))) else {
        panic!("expected a source identity mismatch")
    };
    assert_eq!(
        error,
        CountEvidenceError::Topology(FormatError::Internal(
            "count queries use another source engine".into()
        ))
    );
    assert!(queries.counts.is_empty());
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
