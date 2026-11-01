use core::cell::Cell;
use std::collections::{BTreeSet, HashMap};

use rstest::rstest;
use test_util::object_header::{Message, MessageType as RecordType};
use test_util_hdf5::global_heap_fixture::{self, Fixture};

use super::{AnalysisPhase, DISCOVERY_WORK};
use crate::access_mode::AccessMode;
use crate::edit::test_util as edit_test_util;
use crate::group_v2;
use crate::link_info::LinkInfoMessage;
use crate::message_type::MessageType;
use crate::object_graph::TraversalBudget;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::SourceMetadata;
use crate::{AttrValue, File, FileBuilder};

mod collections;
mod foundations;

#[rstest]
#[case::compact(4, false)]
#[case::dense(64, true)]
fn phase_reuses_topology_and_indexes_named_edges(#[case] children: usize, #[case] dense: bool) {
    let path = test_util::temp::temp_path("topology.h5");
    let mut builder = FileBuilder::new();
    for index in 0..children {
        builder
            .create_dataset(&format!("d{index}"))
            .with_i32_data(&[17]);
    }
    builder.write(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap()
        .get();
    let header = ObjectHeader::parse_from_source_with_widths(
        &SourceMetadata(&engine.image()),
        AccessMode::ReadWrite,
        root,
        engine.widths(),
        engine.addresses().effective_base(),
    )
    .unwrap();
    let info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    assert_eq!(
        LinkInfoMessage::parse_with_offset_width(&info.data, engine.widths().offsets)
            .unwrap()
            .fractal_heap_address
            .is_some(),
        dense
    );
    let paths: Vec<_> = (0..children)
        .map(|index| ObjectPathBuf::parse(&format!("d{index}")))
        .collect();
    let expected: Vec<_> = paths
        .iter()
        .map(|path| {
            group_v2::resolve_path_any_from_source(
                &engine.opened_source(),
                AccessMode::ReadWrite,
                engine.superblock(),
                engine.widths(),
                &path.as_path(),
            )
            .unwrap()
        })
        .collect();
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::new(&engine);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    let topology = phase.topology().unwrap();
    let edges: Vec<_> = paths
        .iter()
        .map(|path| topology.find_link(path).unwrap())
        .collect();
    assert_eq!(
        edges
            .iter()
            .map(|(_, id)| id.header_address().get())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        expected
            .iter()
            .map(|&address| topology.incoming(address))
            .collect::<Vec<_>>(),
        vec![Some(1); children]
    );
    assert_eq!(
        edges
            .iter()
            .map(|(edge, _)| edge)
            .collect::<BTreeSet<_>>()
            .len(),
        children
    );
    let mut actual_children = topology.children(root).collect::<Vec<_>>();
    actual_children.sort_unstable();
    let mut expected_children = expected.clone();
    expected_children.sort_unstable();
    assert_eq!(actual_children, expected_children);
    assert_eq!(topology.incoming(root), None);
    let topology = phase.topology().unwrap();
    assert_eq!(
        paths
            .iter()
            .map(|path| topology.find_link(path).unwrap())
            .collect::<Vec<_>>(),
        edges
    );
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, children + 1));
}

#[test]
fn topology_accounts_for_aliases_and_cycles_despite_unsupported_content() {
    let path = test_util::temp::temp_path("cycle.h5");
    let mut fixture = Fixture::new();
    let leaf = fixture.append_header(vec![Message::new(RecordType(0x7F), &[])]);
    let group = fixture.append_header(vec![global_heap_fixture::hard_link("back", Fixture::ROOT)]);
    fixture.link("a", leaf);
    fixture.link("b", leaf);
    fixture.link("g", group);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::new(&engine);
    let topology = phase.topology().unwrap();
    assert_eq!(
        topology.incoming,
        HashMap::from([(leaf, 2), (group, 1), (Fixture::ROOT, 1)])
    );
    assert_eq!(
        topology
            .find_link(&ObjectPathBuf::parse("g/back/a"))
            .unwrap()
            .1
            .header_address()
            .get(),
        leaf
    );
    assert_eq!(
        topology.children(group).collect::<Vec<_>>(),
        [Fixture::ROOT]
    );
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 3));
    assert!(!engine.global_heap_liveness().unwrap().analysis_complete);
    assert_eq!(phase.topology().unwrap().incoming(leaf), Some(2));
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 3));
}

#[rstest]
#[case::object_limit(TraversalBudget::new(1, usize::MAX))]
#[case::edge_limit(TraversalBudget::new(usize::MAX, 0))]
fn incomplete_topology_is_cached_without_authorizing_incoming_counts(
    #[case] budget: TraversalBudget,
) {
    let path = test_util::temp::temp_path("limit.h5");
    let mut fixture = Fixture::new();
    fixture.object("leaf", Vec::new());
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::new(&engine);
    phase.budget = budget;
    assert!(phase.topology().is_none());
    assert!(phase.topology().is_none());
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 1));
}

#[test]
fn relocation_and_deletion_share_the_pre_edit_topology() {
    let path = test_util::temp::temp_path("mixed.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("kept").with_i32_data(&[17]);
    builder.create_dataset("removed").with_i32_data(&[29]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("kept"), "value", AttrValue::I64(43))
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("removed")).unwrap();
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    engine.commit().unwrap();
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 3));
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("kept"), "value", AttrValue::I64(47))
        .unwrap();
    engine.commit().unwrap();
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (2, 5));
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("kept").unwrap().attrs().unwrap(),
        HashMap::from([(String::from("value"), AttrValue::I64(47))])
    );
    assert_eq!(file.dataset("kept").unwrap().read_i32().unwrap(), [17]);
    assert_eq!(file.root().datasets().unwrap(), ["kept"]);
}

#[test]
fn in_place_numeric_writes_do_not_discover_topology() {
    let path = test_util::temp::temp_path("in-place.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("d").with_i32_data(&[17, 19]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    file.dataset("d")
        .unwrap()
        .write_staged(|dataset| {
            dataset.with_i32_data(&[23, 29]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [23, 29]);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
}

#[test]
fn unreadable_topology_is_cached_as_incomplete() {
    let path = test_util::temp::temp_path("unreadable.h5");
    let mut fixture = Fixture::new();
    fixture.link("missing", 65_536);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::new(&engine);
    assert!(phase.topology().is_none());
    assert!(phase.topology().is_none());
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 2));
}
