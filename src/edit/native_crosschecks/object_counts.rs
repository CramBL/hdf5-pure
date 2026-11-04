use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
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
use crate::edit::analysis::counts::ROOT_REFERENCE_COUNT;
use crate::edit::group::GroupInfo;
use crate::edit::object_header;
use crate::edit::object_header::PreparedHeader;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::compact_named(false, true)]
#[case::compact_unnamed(false, false)]
#[case::dense_named(true, true)]
#[case::dense_unnamed(true, false)]
fn copied_datatype_users_preserve_native_compact_and_dense_references(
    #[case] dense: bool,
    #[case] named: bool,
) {
    let path = test_util::temp::temp_path("native-copied-type-users.h5");
    let native = file::libhdf5_create_v18(&path);
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("mytype", &datatype).unwrap();
    let group = native.create_group("source").unwrap();
    let payload = vec![7i32; 1_048_576];
    let dataset = group
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape([payload.len()])
        .create("data")
        .unwrap();
    dataset.write(&payload).unwrap();
    assert!(dataset.dtype().unwrap().is_committed());
    drop(dataset);
    let attrs = if dense { 9 } else { 1 };
    for index in 0..attrs {
        group
            .new_attr_builder()
            .empty_as(&datatype)
            .shape([1])
            .create(format!("attr{index}").as_str())
            .unwrap()
            .write(&[index])
            .unwrap();
    }
    drop(group);
    assert_eq!(
        native.loc_info_by_name("mytype").unwrap().num_links,
        attrs + 2
    );
    if !named {
        native.unlink("mytype").unwrap();
    }
    drop(datatype);
    native.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let (_, object) = phase
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("source"))
        .unwrap();
    let tree = WriteEngine::read_copy_subtree(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        object.header_address().get(),
        false,
        engine.context.widths(),
    )
    .unwrap();
    assert_eq!(tree.has_dense_attributes(), dense);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let users = {
        let (copy, control) =
            allocation::measure(&format!("copied_type_payload_{dense}_{named}"), || {
                std::hint::black_box(payload.clone())
            });
        assert_eq!(copy, payload);
        assert!(
            control.blocks > 0
                && control.bytes >= u64::try_from(payload.len() * size_of::<i32>()).unwrap(),
            "{control}"
        );
        drop(copy);
        let (users, measured) =
            allocation::measure(&format!("copied_type_summary_{dense}_{named}"), || {
                tree.datatype_users(engine.context.widths())
            });
        let users = users.unwrap().unwrap();
        assert!(
            measured.blocks > 0 && measured.bytes <= 1_024 && measured.live_bytes <= 1_024,
            "{measured}"
        );
        users
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let users = tree
        .datatype_users(engine.context.widths())
        .unwrap()
        .unwrap();
    assert!(users.matches(&tree));
    let users = users.users().collect::<Vec<_>>();
    assert_eq!(users.len(), 1);
    let (address, count) = users[0];
    assert_eq!(count.get(), u64::try_from(attrs + 1).unwrap());
    let target = ObjectId::resolve(engine.addresses(), address).unwrap();
    assert_eq!(
        phase
            .topology()
            .unwrap()
            .incoming(target.header_address().get()),
        named.then_some(1)
    );
    let observed = phase.object_reference_count(target).unwrap();
    let ObjectReferenceCount::Version2(field) = observed else {
        panic!("expected a version 2 committed-type count, got {observed:?}")
    };
    assert_eq!(u64::from(field.value()), count.get() + u64::from(named));
}

#[rstest]
#[case::version_1(false)]
#[case::version_2(true)]
fn native_root_counts_include_the_superblock_reference(
    #[case] modern: bool,
    #[values(0, 1, 2)] links: u32,
) {
    let path = test_util::temp::temp_path("native-root-counts.h5");
    let native = if modern {
        file::libhdf5_create_v18(&path)
    } else {
        file::libhdf5_create_earliest(&path)
    };
    assert_eq!(
        native.loc_info_by_name("/").unwrap().num_links,
        usize::try_from(ROOT_REFERENCE_COUNT).unwrap()
    );
    for index in 0..links {
        native.link_hard("/", &format!("self{index}")).unwrap();
    }
    assert_eq!(
        native.loc_info_by_name("/").unwrap().num_links,
        usize::try_from(links + ROOT_REFERENCE_COUNT).unwrap()
    );
    if links > 0 {
        native.unlink("self0").unwrap();
    }
    let expected = links.saturating_sub(1) + ROOT_REFERENCE_COUNT;
    assert_eq!(
        native.loc_info_by_name("/").unwrap().num_links,
        usize::try_from(expected).unwrap()
    );
    native.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    assert_eq!(
        phase
            .topology()
            .unwrap()
            .incoming(root.header_address().get()),
        if links > 1 { Some(links - 1) } else { None }
    );
    match phase.object_reference_count(root).unwrap() {
        ObjectReferenceCount::Version1(field) if !modern => assert_eq!(field.value(), expected),
        ObjectReferenceCount::Version2(field) if modern => assert_eq!(field.value(), expected),
        ObjectReferenceCount::ImplicitOne if modern && expected == ROOT_REFERENCE_COUNT => {}
        other => panic!("unexpected root count: {other:?}"),
    }
    let Some(CountEvidence::Retained(bound)) = phase.count_evidence(root).unwrap() else {
        panic!("expected retained native root evidence")
    };
    assert_eq!(bound.original(), expected);
    assert_eq!(bound.incoming(), links.saturating_sub(1));
    if links > 1 {
        let deletion = phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("self1"))
            .unwrap();
        let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
        assert_eq!(plan.graph_remaining(root), Some(0));
        assert_eq!(plan.declared_remaining(root), Some(ROOT_REFERENCE_COUNT));
        assert_eq!(plan.counted_zero_objects().count(), 0);
    }
}

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
    assert_eq!(
        evidence.bound().header_allocation().unwrap().range(),
        field.chunk()
    );
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
    let detached = source.surviving_counts(&engine).unwrap();
    assert!(detached.same_session(&engine));
    assert_eq!(detached.updates().len(), 1);
    assert_eq!(detached.updates()[0].object(), object);
    assert_eq!(detached.updates()[0].field(), field);
    assert_eq!(
        detached.updates()[0].header_allocation().unwrap().range(),
        field.chunk()
    );
    assert_eq!(
        detached.updates()[0].count().get(),
        u32::try_from(reported).unwrap() - 1
    );
    let protected = detached.protect(&engine, []).unwrap();
    assert_eq!(protected.updates().len(), 1);
    assert_eq!(protected.updates()[0].field(), field);
    assert_eq!(
        protected.updates()[0].count().get(),
        u32::try_from(reported).unwrap() - 1
    );
    assert_eq!(protected.updates()[0].allocation().range(), field.chunk());
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

#[rstest]
#[case::version_1(true)]
#[case::version_2(false)]
fn native_count_fields_prepare_only_scalar_and_checksum_bytes(#[case] earliest: bool) {
    let path = test_util::temp::temp_path("native-count-update.h5");
    let native = if earliest {
        file::libhdf5_create_earliest(&path)
    } else {
        file::libhdf5_create_v110(&path)
    };
    let group = native.create_group("g").unwrap();
    group
        .new_attr::<i64>()
        .create("value")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    drop(group);
    native.link_hard("g", "alias").unwrap();
    assert_eq!(native.loc_info_by_name("g").unwrap().num_links, 2);
    native.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let (_, object) = phase
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("g"))
        .unwrap();
    let field = match phase.object_reference_count(object).unwrap() {
        ObjectReferenceCount::Version1(field) if earliest => field,
        ObjectReferenceCount::Version2(field) if !earliest => field,
        other => panic!("unexpected native count form: {other:?}"),
    };
    assert_eq!(field.value(), 2);
    let current = engine
        .image()
        .read_exact_at(
            field.chunk().start().get(),
            usize::try_from(field.chunk().length()).unwrap(),
        )
        .unwrap();
    let region = FileRegion::new(field.chunk(), &current).unwrap();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let update = {
        let (copy, control) = allocation::measure(&format!("native_count_copy_{earliest}"), || {
            std::hint::black_box(current.to_vec())
        });
        assert_eq!(copy, current);
        assert!(
            control.blocks > 0 && control.bytes >= u64::try_from(current.len()).unwrap(),
            "{control}"
        );
        let (update, measured) =
            allocation::measure(&format!("native_count_update_{earliest}"), || {
                field.update(region, 1)
            });
        let update = update.unwrap();
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(measured.bytes, 0, "{measured}");
        update
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let update = field.update(region, 1).unwrap();
    let mut output = current.clone();
    for (range, bytes) in update.writes() {
        assert!(field.chunk().contains(range));
        let start = usize::try_from(range.start().get() - field.chunk().start().get()).unwrap();
        output[start..start + bytes.len()].copy_from_slice(bytes);
    }
    let mut expected = current.clone();
    let at = usize::try_from(field.range().start().get() - field.chunk().start().get()).unwrap();
    assert_eq!(&current[at..at + 4], 2_u32.to_le_bytes());
    expected[at..at + 4].copy_from_slice(&1_u32.to_le_bytes());
    if !earliest {
        let at = expected.len() - checksum::CHECKSUM;
        let value = checksum::lookup3(&expected[..at]);
        expected[at..].copy_from_slice(&value.to_le_bytes());
    }
    assert_eq!(output, expected);
    assert_eq!(
        update.writes().map(|(_, bytes)| bytes.len()).sum::<usize>(),
        if earliest { 4 } else { 8 }
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(field.chunk().start().get(), current.len())
            .unwrap(),
        current
    );
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.loc_info_by_name("g").unwrap().num_links, 2);
    assert_eq!(
        native
            .group("g")
            .unwrap()
            .attr("value")
            .unwrap()
            .read_scalar::<i64>()
            .unwrap(),
        7
    );
    native.close().unwrap();
}

#[derive(Clone, Copy)]
enum Usage {
    CommittedType,
    HardLinks,
}
