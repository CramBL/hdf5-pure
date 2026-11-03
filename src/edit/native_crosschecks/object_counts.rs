use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::counts::CountEvidence;
use crate::edit::analysis::counts::OrdinaryCountSubject;
use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::hard_links(Usage::HardLinks)]
#[case::committed_type_users(Usage::CommittedType)]
fn phase_observes_native_counts_without_replacing_incoming_link_facts(#[case] usage: Usage) {
    let path = test_util::temp::temp_path("native-object-counts.h5");
    let (fixture, name, expected, incoming) = match usage {
        Usage::CommittedType => (Fixture::CommittedDatatypeV1, "mytype", 3, 1),
        Usage::HardLinks => (Fixture::HardLinkUndo, "aa", 2, 2),
    };
    fixture.write(&path);
    fixture.assert_native(&path, ByteOrder::LittleEndian);
    let native = file::libhdf5_open(&path);
    let reported = native.loc_info_by_name(name).unwrap().num_links;
    assert_eq!(reported, expected);
    native.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let (_, object) = phase
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse(name))
        .unwrap();
    assert_eq!(
        phase
            .topology()
            .unwrap()
            .incoming(object.header_address().get()),
        Some(incoming)
    );
    let observed = phase.object_reference_count(object).unwrap();
    let field = match (usage, observed) {
        (Usage::CommittedType, ObjectReferenceCount::Version1(field))
        | (Usage::HardLinks, ObjectReferenceCount::Version2(field)) => field,
        (_, other) => panic!("unexpected count framing: {other:?}"),
    };
    assert_eq!(field.value(), u32::try_from(reported).unwrap());
    assert_eq!(field.range().length(), 4);
    assert!(field.chunk().contains(field.range()));
    let evidence = phase.count_evidence(object).unwrap().unwrap();
    assert_eq!(
        evidence.bound().original(),
        u32::try_from(reported).unwrap()
    );
    assert_eq!(evidence.bound().incoming(), incoming);
    assert_eq!(evidence.bound().field(), Some(field));
    assert!(evidence.bound().matches(phase.topology().unwrap()));
    match (usage, evidence) {
        (Usage::HardLinks, CountEvidence::Ordinary(ordinary)) => {
            assert_eq!(ordinary.subject(), OrdinaryCountSubject::Dataset);
        }
        (Usage::CommittedType, CountEvidence::Retained(_)) => {}
        _ => panic!("unexpected count subject"),
    }
    assert_eq!(phase.object_reference_count(object).unwrap(), observed);
    assert_eq!(
        phase
            .topology()
            .unwrap()
            .incoming(object.header_address().get()),
        Some(incoming)
    );
}

#[test]
fn native_version_1_group_counts_retain_the_supported_group_subject() {
    let path = test_util::temp::temp_path("native-group-counts.h5");
    let native = file::libhdf5_create_earliest(&path);
    drop(native.create_group("g").unwrap());
    native.link_hard("g", "alias").unwrap();
    assert_eq!(native.loc_info_by_name("g").unwrap().num_links, 2);
    assert_eq!(native.loc_info_by_name("alias").unwrap().num_links, 2);
    native.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let (_, object) = phase
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("g"))
        .unwrap();
    let observed = phase.object_reference_count(object).unwrap();
    let ObjectReferenceCount::Version1(field) = observed else {
        panic!("expected version 1 count framing")
    };
    let evidence = phase.count_evidence(object).unwrap().unwrap();
    assert_eq!(evidence.bound().original(), 2);
    assert_eq!(evidence.bound().incoming(), 2);
    assert_eq!(evidence.bound().field(), Some(field));
    let CountEvidence::Ordinary(ordinary) = evidence else {
        panic!("expected ordinary group count evidence")
    };
    assert_eq!(ordinary.subject(), OrdinaryCountSubject::Group);
}

#[derive(Clone, Copy)]
enum Usage {
    CommittedType,
    HardLinks,
}
