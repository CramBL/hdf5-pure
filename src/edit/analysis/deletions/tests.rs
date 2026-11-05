use core::cell::Cell;

use rstest::rstest;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::PlannedDeletion;
use crate::PhysicalAddress;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_graph::TraversalBudget;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::different_edges(&["first", "second"], 2, false)]
#[case::one_edge_through_owner_aliases(&["first/value", "second/value"], 1, true)]
#[case::repeated_path(&["keep", "keep"], 1, true)]
fn removed_edge_multiplicity_is_separate_from_distinct_targets(
    #[case] paths: &[&str],
    #[case] edges: usize,
    #[case] leaf: bool,
) {
    let path = test_util::temp::temp_path("removed-edges.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    let owner = fixture.append_header(vec![global_heap_fixture::hard_link("value", value)]);
    fixture.link("first", owner);
    fixture.link("second", owner);
    fixture.link("keep", value);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let topology = phase.topology().unwrap();
    assert_eq!(
        topology.incoming(ObjectId::new(PhysicalAddress::new(owner))),
        Some(2)
    );
    assert_eq!(
        topology.incoming(ObjectId::new(PhysicalAddress::new(value))),
        Some(2)
    );
    let deletions = paths
        .iter()
        .map(|path| topology.deletion(&ObjectPathBuf::parse(path)).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        deletions
            .iter()
            .map(|deletion| deletion.name().as_str())
            .collect::<Vec<_>>(),
        paths
            .iter()
            .map(|path| path.rsplit('/').next().unwrap())
            .collect::<Vec<_>>()
    );
    let removed = phase.removed_links(&deletions).unwrap();
    assert_eq!(removed.edges.len(), edges);
    assert!(removed.unresolved.is_empty());
    let mut spans = Vec::new();
    assert_eq!(
        removed.collect_free_spans(&mut spans),
        [ObjectId::new(PhysicalAddress::new(if leaf {
            value
        } else {
            owner
        }))]
    );
    assert!(spans.is_empty());
}

#[test]
fn removed_edges_reject_another_topology_from_the_same_source() {
    let path = test_util::temp::temp_path("removed-edge-phase.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    fixture.link("value", value);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut first = AnalysisPhase::new(&engine);
    let deletion = first
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("value"))
        .unwrap();
    let mut second = AnalysisPhase::new(&engine);
    let Err(error) = second.removed_links([&deletion]) else {
        panic!("expected a topology mismatch");
    };
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected an internal topology error, got {error:?}");
    };
    assert_eq!(
        reason,
        "a removed link belongs to another analysis topology"
    );
    assert_eq!(
        first
            .removed_links([&deletion])
            .unwrap()
            .collect_free_spans(&mut Vec::new()),
        [ObjectId::new(PhysicalAddress::new(value))]
    );
}

#[test]
fn incomplete_topology_preserves_deletion_targets_without_allocation_retirement() {
    let path = test_util::temp::temp_path("unresolved-deletion.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    fixture.link("value", value);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    phase.budget = TraversalBudget::new(1, usize::MAX);
    let deletion = PlannedDeletion::unresolved(
        LinkNameBuf::new("value").unwrap(),
        PhysicalAddress::new(value),
    );
    let mut spans = Vec::new();
    let removed = phase.removed_links([&deletion]).unwrap();
    assert!(removed.topology.is_none());
    assert!(removed.edges.is_empty());
    assert_eq!(
        removed.collect_free_spans(&mut spans),
        [ObjectId::new(PhysicalAddress::new(value))]
    );
    assert!(spans.is_empty());
}

#[test]
fn empty_removed_link_batch_skips_topology_discovery() {
    let path = test_util::temp::temp_path("no-deletions.h5");
    Fixture::new().write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let mut spans = Vec::new();
    assert!(
        phase
            .removed_links([])
            .unwrap()
            .collect_free_spans(&mut spans)
            .is_empty()
    );
    assert!(spans.is_empty());
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
}
