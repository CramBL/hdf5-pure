use core::cell::Cell;

use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::v2::Header;
use test_util::object_header::v2::HeaderFlags;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::PhysicalAddress;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::PlannedDeletion;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_graph::TraversalBudget;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::one_alias(&["first"], 1)]
#[case::final_aliases(&["first", "second"], 0)]
#[case::repeated_path(&["first", "first"], 1)]
fn removed_alias_counts_use_distinct_physical_edges(
    #[case] paths: &[&str],
    #[case] remaining: u32,
) {
    let path = test_util::temp::temp_path("alias-counts.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    fixture.link("first", value);
    fixture.link("second", value);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletions = paths
        .iter()
        .map(|path| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(path))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = phase
        .removed_links(&deletions)
        .unwrap()
        .count_plan()
        .unwrap()
        .unwrap();
    assert_eq!(plan.remaining(id(value)), Some(remaining));
    assert_eq!(
        plan.zero_link_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        if remaining == 0 {
            vec![id(value)]
        } else {
            Vec::new()
        }
    );
    assert_eq!(plan.topology.incoming(value), Some(2));
}

#[test]
fn owner_alias_paths_remove_the_same_child_edge_once() {
    let path = test_util::temp::temp_path("owner-alias-counts.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    let owner = fixture.append_header(vec![global_heap_fixture::hard_link("value", value)]);
    fixture.link("first", owner);
    fixture.link("second", owner);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletions = ["first/value", "second/value"].map(|path| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse(path))
            .unwrap()
    });
    let plan = phase
        .removed_links(&deletions)
        .unwrap()
        .count_plan()
        .unwrap()
        .unwrap();
    assert_eq!(plan.remaining(id(value)), Some(0));
    assert_eq!(plan.remaining(id(owner)), Some(2));
    assert_eq!(
        plan.zero_link_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        [id(value)]
    );
    assert_eq!(plan.work.initial_edges, 1);
    assert_eq!(plan.work.decrements, 1);
}

#[rstest]
#[case::repeated_child_edges(OwnerRemoval::RepeatedChildren)]
#[case::already_removed_child_edge(OwnerRemoval::AlreadyRemovedChild)]
#[case::surviving_outside_link(OwnerRemoval::SurvivingOutsideLink)]
#[case::cascade(OwnerRemoval::Cascade)]
fn zero_count_owners_remove_each_remaining_outgoing_edge(#[case] removal: OwnerRemoval) {
    let path = test_util::temp::temp_path("cascade-counts.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    let branch = fixture.append_header(vec![
        global_heap_fixture::hard_link("left", value),
        global_heap_fixture::hard_link("right", value),
    ]);
    let owner = if removal == OwnerRemoval::Cascade {
        fixture.append_header(vec![global_heap_fixture::hard_link("branch", branch)])
    } else {
        branch
    };
    fixture.link("owner", owner);
    if removal == OwnerRemoval::SurvivingOutsideLink {
        fixture.link("keep", value);
    }
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let mut deletions = vec![
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("owner"))
            .unwrap(),
    ];
    if removal == OwnerRemoval::AlreadyRemovedChild {
        deletions.push(
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse("owner/left"))
                .unwrap(),
        );
    }
    let plan = phase
        .removed_links(&deletions)
        .unwrap()
        .count_plan()
        .unwrap()
        .unwrap();
    let expected = match removal {
        OwnerRemoval::AlreadyRemovedChild | OwnerRemoval::RepeatedChildren => {
            vec![id(owner), id(value)]
        }
        OwnerRemoval::Cascade => vec![id(owner), id(branch), id(value)],
        OwnerRemoval::SurvivingOutsideLink => vec![id(owner)],
    };
    assert_eq!(
        plan.zero_link_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        plan.remaining(id(value)),
        Some(u32::from(removal == OwnerRemoval::SurvivingOutsideLink))
    );
    assert_eq!(
        plan.work.decrements,
        if removal == OwnerRemoval::Cascade {
            4
        } else {
            3
        }
    );
    assert_eq!(
        plan.work.outgoing_edges,
        if removal == OwnerRemoval::Cascade {
            3
        } else {
            2
        }
    );
    assert_eq!(plan.work.objects, expected.len());
    assert_eq!(plan.remaining(id(Fixture::ROOT)), Some(0));
    assert_eq!(plan.remaining(id(u64::MAX)), None);
}

#[rstest]
#[case::root_back_link(CycleRemoval::RootBackLink)]
#[case::positive_cycle(CycleRemoval::KeepCycle)]
#[case::broken_cycle(CycleRemoval::BreakCycle)]
fn count_transitions_keep_the_phase_root_and_positive_cycles(#[case] removal: CycleRemoval) {
    let path = test_util::temp::temp_path("cycle-counts.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    let owner = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    assert_eq!(
        fixture.append_header(vec![
            global_heap_fixture::hard_link(
                "back",
                if removal == CycleRemoval::RootBackLink {
                    Fixture::ROOT
                } else {
                    owner
                }
            ),
            global_heap_fixture::hard_link("value", value),
        ]),
        owner
    );
    fixture.link("owner", owner);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let paths = match removal {
        CycleRemoval::BreakCycle => vec!["owner", "owner/back"],
        CycleRemoval::KeepCycle => vec!["owner"],
        CycleRemoval::RootBackLink => vec!["owner/back"],
    };
    let deletions = paths
        .into_iter()
        .map(|path| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(path))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = phase
        .removed_links(&deletions)
        .unwrap()
        .count_plan()
        .unwrap()
        .unwrap();
    assert_eq!(plan.remaining(id(Fixture::ROOT)), Some(0));
    assert_eq!(
        plan.remaining(id(owner)),
        Some(u32::from(removal != CycleRemoval::BreakCycle))
    );
    assert_eq!(
        plan.remaining(id(value)),
        Some(u32::from(removal != CycleRemoval::BreakCycle))
    );
    assert_eq!(
        plan.zero_link_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        if removal == CycleRemoval::BreakCycle {
            vec![id(owner), id(value)]
        } else {
            Vec::new()
        }
    );
}

#[rstest]
#[case::incomplete(false)]
#[case::complete_unresolved(true)]
fn unresolved_targets_supply_no_zero_count_proof(#[case] complete: bool) {
    let path = test_util::temp::temp_path("unresolved-counts.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    fixture.link("value", value);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    if !complete {
        phase.budget = TraversalBudget::new(1, usize::MAX);
    }
    let deletion = PlannedDeletion::unresolved(
        LinkNameBuf::new("value").unwrap(),
        PhysicalAddress::new(value),
    );
    let plan = phase
        .removed_links([&deletion])
        .unwrap()
        .count_plan()
        .unwrap();
    if complete {
        let plan = plan.unwrap();
        assert_eq!(plan.remaining(id(value)), Some(1));
        assert_eq!(plan.zero_link_objects().count(), 0);
    } else {
        assert!(plan.is_none());
    }
}

#[rstest]
#[case::small(1, 1)]
#[case::many_live_children(1, 256)]
#[case::many_removed_children(128, 1)]
#[case::wide(128, 256)]
fn count_work_visits_only_removed_edges_and_zero_count_adjacency(
    #[case] removed: usize,
    #[case] live: usize,
) {
    let path = test_util::temp::temp_path("count-work.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    let owner = wide_header(
        &mut fixture,
        (0..removed)
            .map(|index| global_heap_fixture::hard_link(&format!("removed{index}"), value))
            .collect(),
    );
    let live_children = (0..live)
        .map(|index| {
            let child = fixture.append_header(Vec::new());
            global_heap_fixture::hard_link(&format!("live{index}"), child)
        })
        .collect();
    let kept = wide_header(&mut fixture, live_children);
    fixture.link("removed", owner);
    fixture.link("kept", kept);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("removed"))
        .unwrap();
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let plan = phase
        .removed_links([&deletion])
        .unwrap()
        .count_plan()
        .unwrap()
        .unwrap();
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    assert_eq!(
        plan.zero_link_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        [id(owner), id(value)]
    );
    assert_eq!(plan.remaining(id(kept)), Some(1));
    assert_eq!(plan.remaining.len(), 2);
    assert_eq!(plan.work.initial_edges, 1);
    assert_eq!(plan.work.objects, 2);
    assert_eq!(plan.work.outgoing_edges, removed);
    assert_eq!(plan.work.decrements, removed + 1);
}

#[rstest]
#[case::missing(None, "a removed link target has no incoming count")]
#[case::underflow(Some(0), "removed links exceed their target's incoming count")]
fn inconsistent_original_counts_return_an_internal_error(
    #[case] count: Option<u32>,
    #[case] reason: &str,
) {
    let path = test_util::temp::temp_path("invalid-counts.h5");
    let mut fixture = Fixture::new();
    let value = fixture.append_header(Vec::new());
    fixture.link("value", value);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("value"))
        .unwrap();
    let topology = phase.topology.as_mut().unwrap().as_mut().unwrap();
    if let Some(count) = count {
        topology.incoming.insert(value, count);
    } else {
        topology.incoming.remove(&value);
    }
    let Err(error) = phase.removed_links([&deletion]).unwrap().count_plan() else {
        panic!("expected an inconsistent count error");
    };
    let FormatError::Internal(actual) = error else {
        panic!("expected Internal, got {error:?}");
    };
    assert_eq!(actual, reason);
}

#[test]
fn empty_batch_has_no_count_plan_or_topology_pass() {
    let path = test_util::temp::temp_path("empty-counts.h5");
    Fixture::new().write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    assert!(
        phase
            .removed_links([])
            .unwrap()
            .count_plan()
            .unwrap()
            .is_none()
    );
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum OwnerRemoval {
    AlreadyRemovedChild,
    Cascade,
    RepeatedChildren,
    SurvivingOutsideLink,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CycleRemoval {
    BreakCycle,
    KeepCycle,
    RootBackLink,
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

fn wide_header(fixture: &mut Fixture, messages: Vec<Message>) -> u64 {
    u64::try_from(
        fixture.image.append(
            &Header::new()
                .flags(HeaderFlags::CHUNK_SIZE_WIDTH)
                .messages(messages)
                .build(),
        ),
    )
    .unwrap()
}
