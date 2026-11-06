use core::mem;
use core::num::NonZeroU32;

use hdf5_pure_format::__private::MessageRecordLayout;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::v2;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

use crate::AttrValue;
use crate::MessageType;
use crate::access_mode::AccessMode;
use crate::edit::OhRegion;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::NetCountPlan;
use crate::edit::analysis::counts::OrdinaryCountSubject;
use crate::edit::analysis::counts::ROOT_REFERENCE_COUNT;
use crate::edit::analysis::counts::changes::OriginalCount;
use crate::edit::analysis::counts::changes::owned::{
    CountDisposition as OwnedCountDisposition, OwnedOriginalCount, UnresolvedCount,
};
use crate::edit::attributes;
use crate::edit::attributes::AttrOp;
use crate::edit::commit::plan::inventory::{InventoryPreparation, PreparedEditInventory};
use crate::edit::group;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;

use super::super::super::ReplacedHeader;
use super::super::CountDisposition;
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
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let plan = phase.net_count_plan([&deletion], &added).unwrap().unwrap();
    let owned_records = project_owned(&engine, &plan, &mut prepared);
    let CountDisposition {
        rebuilt,
        source,
        opaque,
        unchanged,
    } = PreparedCounts::combined(&plan, [&mut prepared])
        .unwrap()
        .apply()
        .unwrap();
    assert_eq!(rebuilt.len(), 1);
    assert!(source.is_empty() && opaque.is_empty());
    assert!(unchanged.is_empty());
    let OriginalCount::Topology(evidence) = rebuilt[0].original() else {
        panic!("expected topology-bound count evidence")
    };
    assert_eq!(evidence.bound().original(), 2);
    let (inventory, surviving) = plan.owned_counts(&engine).unwrap().partition().into_parts();
    assert_eq!(inventory.changes().len(), 1);
    assert_eq!(
        inventory.changes()[0].original(),
        OwnedOriginalCount::Ordinary {
            original: NonZeroU32::new(2).unwrap(),
            incoming: 2,
            subject: OrdinaryCountSubject::Group,
        }
    );
    assert_eq!(
        surviving
            .updates()
            .iter()
            .map(|update| update.count().get())
            .collect::<Vec<_>>(),
        [1]
    );
    assert_eq!(rebuilt[0].remaining(), 1);
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
    assert_eq!(prepared.region.to_vec(), owned_records.to_vec());
    assert_eq!(prepared.region.reference_count_mut().unwrap().count(), 1);
    assert_eq!(plan.changes().next().unwrap().field().unwrap().value(), 2);
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
#[case::named(true)]
#[case::unnamed(false)]
fn native_copy_users_project_combined_counts_without_source_writes(#[case] named: bool) {
    let path = test_util::temp::temp_path("native-combined-counts.h5");
    let native = file::libhdf5_create_v18(&path);
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("type", &datatype).unwrap();
    let dataset = native
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape([1])
        .create("data")
        .unwrap();
    dataset.write(&[17_i32]).unwrap();
    assert!(dataset.dtype().unwrap().is_committed());
    drop(dataset);
    assert_eq!(native.loc_info_by_name("type").unwrap().num_links, 2);
    if !named {
        native.unlink("type").unwrap();
    }
    drop(datatype);
    native.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let disk = engine.disk_bytes().unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let (_, data) = phase
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("data"))
        .unwrap();
    let tree = WriteEngine::read_copy_subtree(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        data,
        false,
        engine.widths(),
    )
    .unwrap();
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase
        .added_copy_users([Some(&copy), Some(&copy)])
        .unwrap()
        .unwrap();
    let (target, users) = added.users().next().unwrap();
    assert_eq!(users.get(), 2);
    let original = if named { 2 } else { 1 };
    let source = phase.reconstruction_source(&engine, target).unwrap();
    let mut prepared = PreparedGroup {
        region: super::gather(&engine, target.header_address().get()),
        replaces: Some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let mut expected = prepared.region.to_vec();
    let mut position = 0;
    let mut count_body = None;
    while let Some((kind, body, end)) = prepared.region.next_message(position).unwrap() {
        if kind == MessageType::OBJECT_REFERENCE_COUNT {
            count_body = Some(body..end);
        }
        position = end;
    }
    if let Some(body) = count_body.clone() {
        assert_eq!(
            &expected[body.clone()],
            Message::reference_count(original).data
        );
        expected[body].copy_from_slice(&Message::reference_count(3).data);
    } else {
        assert_eq!(original, 1);
        assert_eq!(
            prepared.region.layout(),
            MessageRecordLayout::from_header_flags(v2::HeaderFlags::default().0)
        );
        expected.splice(
            position..position,
            v2::message_record(&Message::reference_count(3), v2::HeaderFlags::default()),
        );
    }
    let deletion = named.then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("type"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletion.as_ref(), &added)
        .unwrap()
        .unwrap();
    let owned_records = project_owned(&engine, &plan, &mut prepared);
    let CountDisposition {
        rebuilt,
        source,
        opaque,
        unchanged,
    } = PreparedCounts::combined(&plan, [&mut prepared])
        .unwrap()
        .apply()
        .unwrap();
    assert_eq!(rebuilt.len(), 1);
    assert!(source.is_empty() && opaque.is_empty() && unchanged.is_empty());
    assert_eq!(rebuilt[0].remaining(), 3);
    match rebuilt[0].original() {
        OriginalCount::Topology(original) => {
            assert!(named);
            assert_eq!(original.bound().field().unwrap().value(), 2);
        }
        OriginalCount::Referenced(original) => {
            assert!(!named);
            assert_eq!(original.minimum(), None);
            assert_eq!(
                original.field().map(|field| field.value()),
                count_body.map(|_| 1)
            );
        }
    }
    assert_eq!(&prepared.region[..], expected);
    assert_eq!(prepared.region.to_vec(), owned_records.to_vec());
    assert_eq!(engine.disk_bytes().unwrap(), disk);
    let (inventory, surviving) = plan.owned_counts(&engine).unwrap().partition().into_parts();
    assert_eq!(inventory.changes().len(), 1);
    let owned = &inventory.changes()[0];
    assert_eq!(
        (owned.object(), owned.original().value(), owned.remaining()),
        (target, original, 3)
    );
    assert_eq!(
        surviving.updates().len(),
        usize::from(owned.field().is_some())
    );
    if let Some(field) = owned.field() {
        assert_eq!(surviving.updates()[0].field(), field);
        assert_eq!(
            surviving.updates()[0].header_allocation(),
            owned.header_allocation()
        );
    } else {
        let OwnedCountDisposition::Unresolved { change, reason } = owned.disposition() else {
            panic!("expected a missing source field");
        };
        assert!(std::ptr::eq(change, owned));
        assert_eq!(reason, UnresolvedCount::MissingField);
    }
    drop((rebuilt, source, opaque, unchanged));
    drop(plan);
    drop(added);
    drop(copy);
    drop(tree);
    drop(phase);
    drop(engine);
    let reopened = edit_test_util::open_engine(&path);
    assert!(!inventory.same_session(&reopened));
    drop(reopened);
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [17]
    );
    if named {
        assert_eq!(native.loc_info_by_name("type").unwrap().num_links, 2);
    }
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
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let plan = phase.net_count_plan([&deletion], &added).unwrap().unwrap();
    let removals = plan.removals().unwrap();
    assert_eq!(removals.counted_zero_objects().count(), 0);
    assert_eq!(removals.graph_remaining(root), Some(0));
    assert_eq!(
        removals.declared_remaining(root),
        Some(ROOT_REFERENCE_COUNT)
    );
    let owned_records = project_owned(&engine, &plan, &mut prepared);
    let CountDisposition {
        rebuilt,
        source,
        opaque,
        unchanged,
    } = PreparedCounts::combined(&plan, [&mut prepared])
        .unwrap()
        .apply()
        .unwrap();
    assert_eq!(rebuilt.len(), 1);
    assert!(source.is_empty() && opaque.is_empty());
    assert!(unchanged.is_empty());
    let OriginalCount::Topology(evidence) = rebuilt[0].original() else {
        panic!("expected topology-bound count evidence")
    };
    assert_eq!(evidence.bound().object(), root);
    let (inventory, surviving) = plan.owned_counts(&engine).unwrap().partition().into_parts();
    assert_eq!(inventory.root(), Some(root));
    assert_eq!(inventory.changes().len(), 1);
    assert_eq!(
        inventory.changes()[0].original(),
        OwnedOriginalCount::Retained {
            original: NonZeroU32::new(2).unwrap(),
            incoming: 1,
        }
    );
    assert_eq!(
        surviving
            .updates()
            .iter()
            .map(|update| update.count().get())
            .collect::<Vec<_>>(),
        [1]
    );
    assert_eq!(rebuilt[0].remaining(), ROOT_REFERENCE_COUNT);
    assert_eq!(&prepared.region[..], expected);
    assert_eq!(prepared.region.to_vec(), owned_records.to_vec());
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

/// Applies the owned route, restores the original records, and returns the projected records.
fn project_owned(
    engine: &WriteEngine,
    plan: &NetCountPlan<'_, '_>,
    prepared: &mut PreparedGroup,
) -> OhRegion {
    let original = prepared.region.clone();
    let checked = plan
        .owned_counts(engine)
        .unwrap()
        .preflight(engine, prepared.replaces.iter().map(ReplacedHeader::source))
        .unwrap();
    let (projected, proof) = PreparedCounts::owned(&checked, [&mut *prepared])
        .unwrap()
        .apply()
        .unwrap();
    assert_eq!(projected.rebuilt.len(), 1);
    assert!(
        projected.source.is_empty()
            && projected.opaque.is_empty()
            && projected.unchanged.is_empty()
    );
    assert!(std::ptr::eq(projected.rebuilt[0], &checked.changes()[0]));
    drop(projected);
    assert!(checked.into_parts(proof).unwrap().1.updates().is_empty());
    mem::replace(&mut prepared.region, original)
}

#[test]
fn native_committed_type_edits_retain_actual_copy_and_removal_inputs() {
    let path = test_util::temp::temp_path("native-edit-inventory.h5");
    let native = file::libhdf5_create_v18(&path);
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("mytype", &datatype).unwrap();
    let dataset = native
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape([2])
        .create("first")
        .unwrap();
    dataset.write(&[1_i32, 2]).unwrap();
    drop(dataset);
    drop(datatype);
    assert_eq!(native.loc_info_by_name("mytype").unwrap().num_links, 2);
    let dataset = native.dataset("first").unwrap();
    assert!(dataset.dtype().unwrap().is_committed());
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [1, 2]);
    drop(dataset);
    native.close().unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("mytype")).unwrap();
    engine
        .copy(
            &ObjectPathBuf::parse("first"),
            &ObjectPathBuf::parse("copy"),
        )
        .unwrap();
    let staged = mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let prepared = match PreparedEditInventory::prepare(&mut engine, staged) {
        InventoryPreparation::Tree(prepared) => prepared,
        InventoryPreparation::Rejected { error, .. } => panic!("preparation failed: {error:?}"),
        InventoryPreparation::InPlace(_) => panic!("expected native tree preparation"),
    };
    let facts = prepared.original().facts().unwrap();
    assert_eq!(facts.counts().changes().len(), 1);
    let change = &facts.counts().changes()[0];
    assert_eq!((change.original().value(), change.remaining()), (2, 2));
    assert_eq!(change.field().unwrap().value(), 2);
    let original_target = change.object();
    assert_eq!(facts.storage().reference_roots(), [original_target]);
    assert!(facts.eligible_owners().is_empty());
    let predecessors = prepared.dispositions().predecessors();
    assert_eq!(predecessors.len(), 1);
    let root = &predecessors[&ObjectPathBuf::root()];
    let root_counts = root.counts.as_ref().unwrap().as_ref().unwrap();
    assert_eq!((root_counts.original, root_counts.incoming), (1, 0));
    let original_root = root.source.object();
    let applied = prepared.apply(&mut engine).unwrap();
    assert_eq!(
        applied.dispositions().replacements().unwrap(),
        &applied.tree().relocations
    );
    assert_eq!(
        applied.dispositions().translate(original_target).unwrap(),
        original_target
    );
    assert_eq!(
        applied
            .dispositions()
            .translate(original_root)
            .unwrap()
            .header_address(),
        applied.tree().retirement.root()
    );
    assert_eq!(
        applied.original().facts().unwrap().counts().changes()[0].object(),
        original_target
    );
    engine.publish_tree(applied.into_tree()).unwrap();
    drop(engine);
    let native = file::libhdf5_open(&path);
    let mut names = native.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["copy", "first"]);
    let copied = native.dataset("copy").unwrap();
    assert!(copied.dtype().unwrap().is_committed());
    assert_eq!(copied.read_raw::<i32>().unwrap(), [1, 2]);
    drop(copied);
    native.close().unwrap();
}

#[rstest]
#[case::increase(2, false, false, 3)]
#[case::decrease(1, true, false, 2)]
#[case::balanced(1, false, false, 2)]
#[case::removed_dataset_user(1, false, true, 2)]
fn native_committed_type_counts_follow_copy_and_delete_commits(
    #[case] copies: usize,
    #[case] extra_alias: bool,
    #[case] remove_source: bool,
    #[case] expected: usize,
) {
    let path = test_util::temp::temp_path("native-public-counts.h5");
    let native = file::libhdf5_create_v18(&path);
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("type", &datatype).unwrap();
    let dataset = native
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape([2])
        .create("source")
        .unwrap();
    dataset.write(&[7_i32, 11]).unwrap();
    if extra_alias {
        native.link_hard("type", "alias").unwrap();
    }
    assert_eq!(
        native.loc_info_by_name("type").unwrap().num_links,
        if extra_alias { 3 } else { 2 }
    );
    drop(dataset);
    drop(datatype);
    native.close().unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    for index in 0..copies {
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse(&format!("copy{index}")),
            )
            .unwrap();
    }
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    if extra_alias {
        engine.delete(&ObjectPathBuf::parse("alias")).unwrap();
    }
    if remove_source {
        engine.delete(&ObjectPathBuf::parse("source")).unwrap();
    }
    engine.commit().unwrap();
    drop(engine);
    let native = file::libhdf5_open(&path);
    let dataset = native.dataset("copy0").unwrap();
    let datatype = dataset.dtype().unwrap();
    assert!(datatype.is_committed());
    assert_eq!(
        datatype
            .as_location()
            .unwrap()
            .loc_info()
            .unwrap()
            .num_links,
        expected
    );
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [7, 11]);
    drop(datatype);
    drop(dataset);
    native.close().unwrap();
}
