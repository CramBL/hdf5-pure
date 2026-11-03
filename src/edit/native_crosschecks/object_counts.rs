use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
use test_util::checksum;
use test_util_hdf5::file;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use crate::File;
use crate::MessageType;
use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::counts::CountEvidence;
use crate::edit::analysis::counts::OrdinaryCountSubject;
use crate::edit::group::GroupInfo;
use crate::edit::object_header;
use crate::edit::object_header::PreparedHeader;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_header::ObjectHeader;
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
    fixture.assert_native(
        &path,
        if cfg!(target_endian = "big") {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        },
    );
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
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse(name))
        .unwrap();
    let (removed, mut queries) = phase.removal_queries([&deletion]).unwrap();
    let plan = removed.count_plan().unwrap().unwrap();
    assert_eq!(plan.remaining(object), Some(incoming - 1));
    let evidence = queries.evidence(object).unwrap().unwrap();
    assert_eq!(
        evidence.bound().original(),
        u32::try_from(reported).unwrap()
    );
    assert_eq!(evidence.bound().incoming(), incoming);
    let source = phase.source_count_plan([&deletion]).unwrap().unwrap();
    assert_eq!(source.graph_remaining(object), Some(incoming - 1));
    assert_eq!(
        source.declared_remaining(object),
        Some(u32::try_from(reported).unwrap() - 1)
    );
    assert_eq!(source.counted_zero_objects().count(), 0);
    assert_eq!(source.retained_objects().collect::<Vec<_>>(), [object]);
    let changes = source.changes().collect::<Vec<_>>();
    assert_eq!(changes.len(), 1);
    assert_eq!(
        changes[0].bound().original(),
        u32::try_from(reported).unwrap()
    );
    assert_eq!(changes[0].bound().field(), Some(field));
    assert_eq!(changes[0].remaining(), u32::try_from(reported).unwrap() - 1);
    drop(source);
    if matches!(usage, Usage::HardLinks) {
        let alias = phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("bb"))
            .unwrap();
        let source = phase
            .source_count_plan([&deletion, &alias])
            .unwrap()
            .unwrap();
        assert_eq!(source.graph_remaining(object), Some(0));
        assert_eq!(source.declared_remaining(object), Some(0));
        assert_eq!(
            source
                .counted_zero_objects()
                .map(|object| object.id())
                .collect::<Vec<_>>(),
            [object]
        );
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
    let topology = phase.topology().unwrap();
    let deletions = [
        topology.deletion(&ObjectPathBuf::parse("g")).unwrap(),
        topology.deletion(&ObjectPathBuf::parse("alias")).unwrap(),
    ];
    let source = phase.source_count_plan(&deletions).unwrap().unwrap();
    assert_eq!(source.graph_remaining(object), Some(0));
    assert_eq!(source.declared_remaining(object), Some(0));
    assert_eq!(
        source
            .counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        [object]
    );
    assert_eq!(
        source
            .changes()
            .map(|change| (change.bound().field(), change.remaining()))
            .collect::<Vec<_>>(),
        [(Some(field), 0)]
    );
}

#[rstest]
#[case::single(false)]
#[case::aliased(true)]
fn version_1_group_conversion_preserves_native_prefix_counts(
    #[case] aliased: bool,
    #[values(false, true)] source_only: bool,
) {
    let path = test_util::temp::temp_path("converted-native-counts.h5");
    let native = file::libhdf5_create_earliest(&path);
    let group = native.create_group("g").unwrap();
    group
        .new_attr::<i32>()
        .create("marker")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    group
        .new_dataset::<i32>()
        .shape([1])
        .create("value")
        .unwrap()
        .write_raw(&[17])
        .unwrap();
    drop(group);
    if aliased {
        native.link_hard("g", "alias").unwrap();
    }
    let expected = native.loc_info_by_name("g").unwrap().num_links;
    assert_eq!(expected, if aliased { 2 } else { 1 });
    assert_eq!(
        native
            .group("g")
            .unwrap()
            .attr("marker")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        7
    );
    native.close().unwrap();
    let reader = File::open(&path).unwrap();
    let address = reader.group("g").unwrap().header_address().unwrap();
    reader.close().unwrap();
    let engine = if source_only {
        WriteEngine::open_source_only(&path).unwrap()
    } else {
        edit_test_util::open_engine(&path)
    };
    let GroupInfo { region, links } = engine.reconstruct_v1_group(address).unwrap();
    assert_eq!(links.links().len(), 1);
    let prepared = PreparedHeader::new(&region).unwrap();
    let geometry = prepared.geometry().unwrap();
    let measured = *geometry.root_chunk_lengths(false).next().unwrap().start();
    let encoded = object_header::build_v2_object_header(&region).unwrap();
    assert_eq!(u64::try_from(encoded.len()).unwrap(), measured);
    let checksum_at = encoded.len() - checksum::CHECKSUM;
    assert_eq!(
        &encoded[checksum_at..],
        checksum::lookup3(&encoded[..checksum_at]).to_le_bytes()
    );
    let rebuilt =
        ObjectHeader::parse_with_widths(&encoded, AccessMode::ReadOnly, 0, engine.widths())
            .unwrap();
    let count = rebuilt
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::OBJECT_REFERENCE_COUNT);
    assert_eq!(
        count.map(|message| message.data.as_slice()),
        aliased.then_some(&[0, 2, 0, 0, 0][..])
    );
    let observed = AnalysisPhase::new(&engine)
        .object_reference_count(ObjectId::new(PhysicalAddress::new(address)))
        .unwrap();
    let ObjectReferenceCount::Version1(field) = observed else {
        panic!("expected a native version 1 count")
    };
    assert_eq!(field.value(), u32::try_from(expected).unwrap());
    assert!(encoded.len() > region.len());
}

#[derive(Clone, Copy)]
enum Usage {
    CommittedType,
    HardLinks,
}
