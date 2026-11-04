use rstest::rstest;
use test_util::object_header::Message;
use test_util_hdf5::file;

use crate::AttrValue;
use crate::MessageType;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::counts::ROOT_REFERENCE_COUNT;
use crate::edit::attributes;
use crate::edit::attributes::AttrOp;
use crate::edit::group;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;

use super::super::super::ReplacedHeader;
use super::super::PreparedCounts;
use super::super::PreparedGroup;

#[rstest]
#[case::version_1(true)]
#[case::version_2(false)]
fn native_group_counts_project_after_link_and_attribute_transformations(#[case] earliest: bool) {
    let path = test_util::temp::temp_path("native-prepared-counts.h5");
    let native = if earliest {
        file::libhdf5_create_earliest(&path)
    } else {
        file::libhdf5_create_v110(&path)
    };
    let native_group = native.create_group("g").unwrap();
    drop(native_group.create_group("remove").unwrap());
    drop(native_group.create_group("retain").unwrap());
    native_group
        .new_attr::<i64>()
        .create("value")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    drop(native_group);
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
    let source = phase.reconstruction_source(&engine, object).unwrap();
    let mut region = engine.inspect_group(object).unwrap().region;
    let original = region.reference_count_mut().unwrap().count();
    assert_eq!(original, 2);
    region =
        group::remove_links_from_region(&region, [&LinkNameBuf::new("remove").unwrap()]).unwrap();
    let retained = LinkNameBuf::new("retain").unwrap();
    group::LinkPatches::new(&mut region, [retained.as_str()])
        .patch_next(
            engine
                .context
                .addresses()
                .relative(object.header_address())
                .unwrap(),
        )
        .unwrap();
    region = attributes::edit_compact_region(
        &region,
        &[AttrOp::Set {
            name: "value".into(),
            value: AttrValue::I64(19),
        }],
    )
    .unwrap();
    let before = region.to_vec();
    let mut prepared = PreparedGroup {
        region,
        replaces: Some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("g"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    let disposition = PreparedCounts::new(Some(&plan), [&mut prepared])
        .unwrap()
        .apply()
        .unwrap();
    assert_eq!(disposition.rebuilt.len(), 1);
    assert!(disposition.source.is_empty() && disposition.opaque.is_empty());
    assert_eq!(disposition.rebuilt[0].bound().original(), 2);
    assert_eq!(disposition.rebuilt[0].remaining(), 1);
    let mut position = 0;
    let at = loop {
        let (kind, body, end) = prepared.region.next_message(position).unwrap().unwrap();
        if kind == MessageType::OBJECT_REFERENCE_COUNT {
            break body;
        }
        position = end;
    };
    assert_eq!(&before[at..at + 5], Message::reference_count(2).data);
    let mut expected = before;
    expected[at + 1] = 1;
    assert_eq!(&prepared.region[..], expected);
    assert_eq!(prepared.region.reference_count_mut().unwrap().count(), 1);
    assert_eq!(
        plan.changes()
            .next()
            .unwrap()
            .bound()
            .field()
            .unwrap()
            .value(),
        2
    );
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.loc_info_by_name("g").unwrap().num_links, 2);
    assert_eq!(
        native.group("g").unwrap().member_names().unwrap(),
        ["remove", "retain"]
    );
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

#[rstest]
#[case::version_1(true)]
#[case::version_2(false)]
fn native_root_self_link_projections_preserve_the_superblock_reference(#[case] earliest: bool) {
    let path = test_util::temp::temp_path("native-root-count-projection.h5");
    let native = if earliest {
        file::libhdf5_create_earliest(&path)
    } else {
        file::libhdf5_create_v18(&path)
    };
    native.link_hard("/", "self").unwrap();
    native
        .new_attr::<i64>()
        .create("value")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    let original = native.loc_info_by_name("/").unwrap().num_links;
    assert_eq!(original, 2);
    native.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let disk = engine.disk_bytes().unwrap();
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let region = engine.inspect_group(root).unwrap().region;
    let region =
        group::remove_links_from_region(&region, [&LinkNameBuf::new("self").unwrap()]).unwrap();
    let region = attributes::edit_compact_region(
        &region,
        &[AttrOp::Set {
            name: "value".into(),
            value: AttrValue::I64(19),
        }],
    )
    .unwrap();
    let mut prepared = PreparedGroup {
        region,
        replaces: Some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let mut expected = prepared.region.to_vec();
    let mut position = 0;
    let body = loop {
        let (kind, body, end) = prepared.region.next_message(position).unwrap().unwrap();
        if kind == MessageType::OBJECT_REFERENCE_COUNT {
            break body..end;
        }
        position = end;
    };
    assert_eq!(
        &expected[body.clone()],
        Message::reference_count(u32::try_from(original).unwrap()).data
    );
    expected[body].copy_from_slice(&Message::reference_count(ROOT_REFERENCE_COUNT).data);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("self"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    assert_eq!(plan.counted_zero_objects().count(), 0);
    assert_eq!(plan.graph_remaining(root), Some(0));
    assert_eq!(plan.declared_remaining(root), Some(ROOT_REFERENCE_COUNT));
    let result = PreparedCounts::new(Some(&plan), [&mut prepared])
        .unwrap()
        .apply()
        .unwrap();
    assert_eq!(result.rebuilt.len(), 1);
    assert!(result.source.is_empty() && result.opaque.is_empty());
    assert_eq!(result.rebuilt[0].bound().object(), root);
    assert_eq!(result.rebuilt[0].remaining(), ROOT_REFERENCE_COUNT);
    assert_eq!(&prepared.region[..], expected);
    assert_eq!(engine.disk_bytes().unwrap(), disk);
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.loc_info_by_name("/").unwrap().num_links, original);
    assert_eq!(
        native.attr("value").unwrap().read_scalar::<i64>().unwrap(),
        7
    );
    assert_eq!(native.member_names().unwrap(), ["self"]);
    native.close().unwrap();
}
