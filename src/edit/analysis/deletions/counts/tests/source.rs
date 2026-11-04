use core::cell::Cell;
use core::num::NonZeroU32;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use crate::Datatype;
use crate::FileLocking;
use crate::FixedPointType;
use crate::edit::WriteEngine;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::counts::CountContradiction;
use crate::edit::test_util as edit_test_util;
use crate::image::CountingImage;
use crate::image::HandleImage;
use crate::object_graph::TraversalBudget;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;

use super::super::AnalysisPhase;
use super::super::CountEvidenceError;
use super::super::FormatError;
use super::super::ObjectId;
use super::super::PlannedDeletion;
use super::super::SourceCountState;
use super::super::SourceRemovalPlan;

#[rstest]
#[case::one(&["first"], 1)]
#[case::final_aliases(&["first", "second"], 0)]
#[case::duplicate_path(&["first", "first"], 1)]
fn source_alias_transitions_count_distinct_edges_once(
    #[case] paths: &[&str],
    #[case] remaining: u32,
) {
    let path = test_util::temp::temp_path("source-alias-counts.h5");
    let mut fixture = Fixture::new();
    let target = dataset(&mut fixture, 2);
    fixture.link("first", target);
    fixture.link("second", target);
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
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    assert_eq!(plan.graph_remaining(id(target)), Some(remaining));
    assert_eq!(plan.declared_remaining(id(target)), Some(remaining));
    assert_eq!(
        plan.counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        if remaining == 0 {
            vec![id(target)]
        } else {
            vec![]
        }
    );
    assert_eq!(
        plan.retained_objects().collect::<Vec<_>>(),
        if remaining == 0 {
            vec![]
        } else {
            vec![id(target)]
        }
    );
    let changes = plan.changes().collect::<Vec<_>>();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].bound().original(), 2);
    assert_eq!(changes[0].bound().field().unwrap().value(), 2);
    assert_eq!(changes[0].remaining(), remaining);
    assert_eq!(
        plan.plan.work.initial_edges,
        usize::try_from(2 - remaining).unwrap()
    );
    assert_eq!(
        plan.plan.work.decrements,
        usize::try_from(2 - remaining).unwrap()
    );
}

#[rstest]
#[case::ambiguous(Owner::Ambiguous)]
#[case::release(Owner::Release)]
#[case::excess(Owner::Excess)]
#[case::unknown(Owner::Unknown)]
#[case::retained_subject(Owner::Retained)]
#[case::outside_child(Owner::OutsideChild)]
#[case::cascade(Owner::Cascade)]
#[case::explicit_child(Owner::ExplicitChild)]
fn only_eligible_zero_owners_release_their_remaining_child_edges(#[case] mode: Owner) {
    let path = test_util::temp::temp_path("source-owner-counts.h5");
    let mut fixture = Fixture::new();
    let child = dataset(
        &mut fixture,
        if mode == Owner::OutsideChild { 3 } else { 2 },
    );
    let mut records = vec![
        global_heap_fixture::hard_link("left", child),
        global_heap_fixture::hard_link("right", child),
    ];
    match mode {
        Owner::Ambiguous => {
            records.extend([Message::reference_count(1), Message::reference_count(1)])
        }
        Owner::Unknown => records.push(Message::new(
            MessageType::OBJECT_REFERENCE_COUNT,
            &[1, 1, 0, 0, 0],
        )),
        Owner::Retained => {
            records.extend(Form::CompactDataset.messages(
                &mut fixture,
                &Datatype::FixedPoint(FixedPointType::i32()),
                &17_i32.to_le_bytes(),
            ));
            records.push(Message::reference_count(1));
        }
        _ => records.push(Message::reference_count(if mode == Owner::Excess {
            2
        } else {
            1
        })),
    }
    let branch = header(&mut fixture, records);
    let owner = if mode == Owner::Cascade {
        group(
            &mut fixture,
            1,
            vec![global_heap_fixture::hard_link("branch", branch)],
        )
    } else {
        branch
    };
    fixture.link("owner", owner);
    if mode == Owner::OutsideChild {
        fixture.link("keep", child);
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
    if mode == Owner::ExplicitChild {
        deletions.push(
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse("owner/left"))
                .unwrap(),
        );
    }
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    let retained = matches!(
        mode,
        Owner::Ambiguous | Owner::Excess | Owner::Unknown | Owner::Retained
    );
    let expected = if retained {
        vec![]
    } else if mode == Owner::Cascade {
        vec![id(owner), id(branch), id(child)]
    } else if mode == Owner::OutsideChild {
        vec![id(owner)]
    } else {
        vec![id(owner), id(child)]
    };
    assert_eq!(
        plan.counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(plan.graph_remaining(id(owner)), Some(0));
    assert_eq!(
        plan.declared_remaining(id(owner)),
        if matches!(mode, Owner::Ambiguous | Owner::Unknown) {
            None
        } else {
            Some(u32::from(mode == Owner::Excess))
        }
    );
    assert_eq!(
        plan.graph_remaining(id(child)),
        Some(if retained {
            2
        } else {
            u32::from(mode == Owner::OutsideChild)
        })
    );
    assert_eq!(
        plan.declared_remaining(id(child)),
        if retained {
            None
        } else {
            Some(u32::from(mode == Owner::OutsideChild))
        }
    );
    assert_eq!(
        plan.plan.work.outgoing_edges,
        if retained {
            0
        } else if mode == Owner::Cascade {
            3
        } else {
            2
        }
    );
    assert_eq!(plan.plan.work.objects, expected.len());
}

#[rstest]
#[case::positive_cycle(Cycle::Keep)]
#[case::broken_cycle(Cycle::Break)]
#[case::root(Cycle::Root)]
fn source_count_transitions_keep_positive_cycles_and_exclude_root_candidates(#[case] mode: Cycle) {
    let path = test_util::temp::temp_path("source-cycle-counts.h5");
    let mut fixture = if mode == Cycle::Root {
        Fixture::new().with_root_reference_count(NonZeroU32::new(2).unwrap())
    } else {
        Fixture::new()
    };
    let child = dataset(&mut fixture, 1);
    let owner = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    assert_eq!(
        group(
            &mut fixture,
            if mode == Cycle::Root { 1 } else { 2 },
            vec![
                global_heap_fixture::hard_link(
                    "back",
                    if mode == Cycle::Root {
                        Fixture::ROOT
                    } else {
                        owner
                    }
                ),
                global_heap_fixture::hard_link("child", child)
            ]
        ),
        owner
    );
    fixture.link("owner", owner);
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
    if mode == Cycle::Break {
        deletions.push(
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse("owner/back"))
                .unwrap(),
        );
    }
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    assert_eq!(
        plan.counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        if mode == Cycle::Keep {
            vec![]
        } else {
            vec![id(owner), id(child)]
        }
    );
    assert_eq!(
        plan.graph_remaining(id(owner)),
        Some(u32::from(mode == Cycle::Keep))
    );
    assert_eq!(
        plan.graph_remaining(id(child)),
        Some(u32::from(mode == Cycle::Keep))
    );
    assert_eq!(plan.graph_remaining(id(Fixture::ROOT)), Some(0));
    assert_eq!(
        plan.declared_remaining(id(Fixture::ROOT)),
        if mode == Cycle::Root { Some(1) } else { None }
    );
}

#[rstest]
#[case::empty(Applicability::Empty)]
#[case::incomplete(Applicability::Incomplete)]
#[case::unresolved(Applicability::Unresolved)]
#[case::outside(Applicability::Outside)]
fn source_count_queries_require_complete_topology_and_identified_edges(
    #[case] mode: Applicability,
) {
    let path = test_util::temp::temp_path("unresolved-source-counts.h5");
    let mut fixture = Fixture::new();
    let target = dataset(&mut fixture, 1);
    fixture.link("target", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    if mode == Applicability::Incomplete {
        phase.budget = TraversalBudget::new(1, usize::MAX);
    }
    let deletion = PlannedDeletion::unresolved(
        LinkNameBuf::new("target").unwrap(),
        PhysicalAddress::new(if mode == Applicability::Outside {
            u64::MAX
        } else {
            target
        }),
    );
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let plan = phase
        .source_count_plan((mode != Applicability::Empty).then_some(&deletion))
        .unwrap();
    if matches!(mode, Applicability::Empty | Applicability::Incomplete) {
        assert!(plan.is_none());
    } else {
        let plan = plan.unwrap();
        assert_eq!(plan.counted_zero_objects().count(), 0);
        assert_eq!(plan.changes().count(), 0);
        assert_eq!(plan.graph_remaining(id(target)), Some(1));
        assert_eq!(plan.graph_remaining(id(u64::MAX)), None);
        assert_eq!(plan.declared_remaining(id(u64::MAX)), None);
    }
    assert!(phase.counts.is_empty());
    if mode == Applicability::Empty {
        assert!(phase.topology.is_none());
        assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    }
}

#[rstest]
#[case::one(1, 1)]
#[case::wide_removed(128, 1)]
#[case::wide_live(1, 256)]
#[case::wide(128, 256)]
fn source_work_queries_only_targets_of_actual_decrements(
    #[case] removed: usize,
    #[case] live: usize,
) {
    let path = test_util::temp::temp_path("source-count-work.h5");
    let mut fixture = Fixture::new();
    let target = dataset(&mut fixture, u32::try_from(removed).unwrap());
    let owner = group(
        &mut fixture,
        1,
        (0..removed)
            .map(|index| global_heap_fixture::hard_link(&format!("r{index}"), target))
            .collect(),
    );
    let live_links = (0..live)
        .map(|index| {
            let child = dataset(&mut fixture, 1);
            global_heap_fixture::hard_link(&format!("l{index}"), child)
        })
        .collect();
    let kept = group(&mut fixture, 1, live_links);
    fixture.link("removed", owner);
    fixture.link("kept", kept);
    fixture.write(&path);
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
        .deletion(&ObjectPathBuf::parse("removed"))
        .unwrap();
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    assert_eq!(
        plan.counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        [id(owner), id(target)]
    );
    assert_eq!(plan.plan.remaining.len(), 2);
    assert_eq!(plan.plan.work.initial_edges, 1);
    assert_eq!(plan.plan.work.objects, 2);
    assert_eq!(plan.plan.work.outgoing_edges, removed);
    assert_eq!(plan.plan.work.decrements, removed + 1);
    assert_eq!(plan.graph_remaining(id(kept)), Some(1));
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    let reads = calls.load(Ordering::Relaxed);
    let read_bytes = bytes.load(Ordering::Relaxed);
    assert!(reads > 0 && read_bytes > 0);
    drop(plan);
    let repeated = phase.source_count_plan([&deletion]).unwrap().unwrap();
    assert_eq!(
        repeated
            .counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        [id(owner), id(target)]
    );
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert_eq!(bytes.load(Ordering::Relaxed), read_bytes);
    drop(repeated);
    assert_eq!(
        phase.counts.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([id(owner), id(target)])
    );
}

#[test]
fn checked_source_decrements_preserve_original_evidence_and_failure_state() {
    let path = test_util::temp::temp_path("checked-source-counts.h5");
    let mut fixture = Fixture::new();
    let target = dataset(&mut fixture, 2);
    fixture.link("a", target);
    fixture.link("b", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let evidence = phase.count_evidence(id(target)).unwrap().unwrap();
    let topology = phase.topology().unwrap();
    let index = topology.indices[&id(target)];
    let mut evidence = Some(evidence);
    let mut state = SourceCountState::new(topology, index, &mut |_| Ok(evidence.take())).unwrap();
    assert!(!state.remove().unwrap());
    assert_eq!((state.graph, state.declared), (1, Some(1)));
    assert!(state.remove().unwrap());
    assert_eq!((state.graph, state.declared), (0, Some(0)));
    let error = state.remove().unwrap_err();
    assert_eq!(
        error,
        CountEvidenceError::Topology(FormatError::Internal(
            "removed links exceed their target's incoming count".into()
        ))
    );
    assert_eq!((state.graph, state.declared), (0, Some(0)));
    assert_eq!(state.evidence.as_ref().unwrap().bound().original(), 2);
    state.graph = 1;
    let error = state.remove().unwrap_err();
    assert_eq!(
        error,
        CountEvidenceError::Topology(FormatError::Internal(
            "removed links exceed their target's declared count".into()
        ))
    );
    assert_eq!((state.graph, state.declared), (1, Some(0)));
}

#[test]
fn source_planning_propagates_contradictions_and_source_failures() {
    let path = test_util::temp::temp_path("failed-source-planning.h5");
    let mut fixture = Fixture::new();
    let target = dataset(&mut fixture, 1);
    fixture.link("a", target);
    fixture.link("b", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    let Err(error) = phase.source_count_plan([&deletion]) else {
        panic!("expected a contradiction")
    };
    assert_eq!(
        error,
        CountEvidenceError::Contradiction(CountContradiction {
            object: id(target),
            observed: 1,
            minimum: 2
        })
    );
    let failure = FormatError::UnexpectedSourceEof {
        expected: target + 1,
        available: target,
    };
    phase.counts.insert(id(target), Err(failure.clone()));
    let Err(error) = phase.source_count_plan([&deletion]) else {
        panic!("expected a source failure")
    };
    assert_eq!(error, CountEvidenceError::Source(failure));
}

#[rstest]
#[case::matching(Identity::Matching)]
#[case::other_phase(Identity::OtherPhase)]
#[case::other_source(Identity::OtherSource)]
#[case::other_object(Identity::OtherObject)]
#[case::incoming_count(Identity::IncomingCount)]
fn source_plans_require_matching_count_evidence(#[case] mode: Identity) {
    let path = test_util::temp::temp_path("source-plan-identity.h5");
    let other_path = test_util::temp::temp_path("other-source-plan-identity.h5");
    let mut fixture = Fixture::new();
    let target = dataset(&mut fixture, 1);
    let other_target = dataset(&mut fixture, 1);
    fixture.link("a", target);
    fixture.link("other", other_target);
    fixture.write(&path);
    std::fs::copy(&path, &other_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let other_engine = edit_test_util::open_engine(&other_path);
    let mut phase = AnalysisPhase::new(&engine);
    let mut other_phase = AnalysisPhase::new(if mode == Identity::OtherSource {
        &other_engine
    } else {
        &engine
    });
    let evidence = match mode {
        Identity::Matching | Identity::IncomingCount => phase.count_evidence(id(target)),
        Identity::OtherObject => phase.count_evidence(id(other_target)),
        Identity::OtherPhase | Identity::OtherSource => other_phase.count_evidence(id(target)),
    }
    .unwrap();
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    if mode == Identity::IncomingCount {
        phase
            .topology
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap()
            .incoming
            .insert(id(target), 2);
    }
    let removed = phase.removed_links([&deletion]).unwrap();
    let mut evidence = evidence;
    let result = SourceRemovalPlan::new(removed, |_| Ok(evidence.take()));
    if mode == Identity::Matching {
        assert_eq!(
            result
                .unwrap()
                .unwrap()
                .counted_zero_objects()
                .map(|object| object.id())
                .collect::<Vec<_>>(),
            [id(target)]
        );
    } else {
        let Err(error) = result else {
            panic!("expected mismatched count evidence")
        };
        assert_eq!(
            error,
            CountEvidenceError::Topology(FormatError::Internal(
                "a declared count does not match its removal topology".into()
            ))
        );
    }
}

#[test]
fn source_plans_reject_deletions_from_another_phase_before_count_reads() {
    let path = test_util::temp::temp_path("source-deletion-identity.h5");
    let mut fixture = Fixture::new();
    let target = dataset(&mut fixture, 1);
    fixture.link("a", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut first = AnalysisPhase::new(&engine);
    let mut second = AnalysisPhase::new(&engine);
    let deletion = first
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    let Err(error) = second.source_count_plan([&deletion]) else {
        panic!("expected a deletion identity mismatch")
    };
    assert_eq!(
        error,
        CountEvidenceError::Topology(FormatError::Internal(
            "a removed link belongs to another analysis topology".into()
        ))
    );
    assert!(second.counts.is_empty());
}

fn dataset(fixture: &mut Fixture, count: u32) -> u64 {
    let mut records = Form::CompactDataset.messages(
        fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &17_i32.to_le_bytes(),
    );
    records.push(Message::reference_count(count));
    header(fixture, records)
}
fn group(fixture: &mut Fixture, count: u32, mut links: Vec<Message>) -> u64 {
    links.push(Message::reference_count(count));
    header(fixture, links)
}
fn header(fixture: &mut Fixture, records: Vec<Message>) -> u64 {
    u64::try_from(
        fixture.image.append(
            &v2::Header::new()
                .flags(v2::HeaderFlags::CHUNK_SIZE_WIDTH)
                .messages(records)
                .build(),
        ),
    )
    .unwrap()
}
fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Owner {
    Ambiguous,
    Cascade,
    Excess,
    ExplicitChild,
    OutsideChild,
    Release,
    Retained,
    Unknown,
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Cycle {
    Break,
    Keep,
    Root,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Applicability {
    Empty,
    Incomplete,
    Outside,
    Unresolved,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Identity {
    IncomingCount,
    Matching,
    OtherObject,
    OtherPhase,
    OtherSource,
}
