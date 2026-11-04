use core::cell::Cell;
use core::num::NonZeroU32;

use rstest::rstest;
use test_util::datatype;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::edit::analysis::counts::CountContradiction;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::edit::analysis::counts::balance::CountArithmeticError;
use crate::edit::analysis::deletions::counts::source::added::AddedCountError;
use crate::edit::analysis::{AnalysisPhase, DISCOVERY_WORK};
use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;
use crate::{Datatype, FixedPointType};

use super::SourceRemovalPlan;

#[rstest]
#[case::unchanged(&["first"], Some(2), 1, 2)]
#[case::last_edges(&["first", "second"], Some(2), 1, 1)]
#[case::duplicate_edge(&["first", "first"], Some(2), 1, 2)]
#[case::maximum_cancellation(&["first"], Some(u32::MAX), 1, u32::MAX)]
#[case::two_removals(&["first", "second"], Some(u32::MAX), 2, u32::MAX)]
#[case::implicit_cancellation(&["first"], None, 1, 1)]
fn additions_are_applied_after_unique_original_edge_removals(
    #[case] paths: &[&str],
    #[case] original: Option<u32>,
    #[case] added: u64,
    #[case] remaining: u32,
) {
    let path = test_util::temp::temp_path("added-source-aliases.h5");
    let mut fixture = Fixture::new();
    let target = if let Some(original) = original {
        super::dataset(&mut fixture, original)
    } else {
        let records = Form::CompactDataset.messages(
            &mut fixture,
            &Datatype::FixedPoint(FixedPointType::i32()),
            &17_i32.to_le_bytes(),
        );
        super::header(&mut fixture, records)
    };
    fixture.link("first", target);
    if original.is_some() {
        fixture.link("second", target);
    }
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
    let before = DISCOVERY_WORK.with(Cell::get);
    let queries = Cell::new(0);
    let additions = Cell::new(0);
    let (removed, mut counts) = phase.removal_queries(&deletions).unwrap();
    let plan = SourceRemovalPlan::with_additions(
        removed,
        |object| {
            queries.set(queries.get() + 1);
            counts.evidence(object)
        },
        |_| {
            additions.set(additions.get() + 1);
            added
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(DISCOVERY_WORK.with(Cell::get), before);
    assert_eq!((queries.get(), additions.get()), (1, 1));
    assert_eq!(plan.declared_remaining(super::id(target)), Some(remaining));
    assert_eq!(
        plan.graph_remaining(super::id(target)),
        Some(1 + u32::from(original.is_some()) - u32::try_from(plan.plan.work.decrements).unwrap())
    );
    assert_eq!(plan.counted_zero_objects().count(), 0);
    assert_eq!(
        plan.retained_objects().collect::<Vec<_>>(),
        [super::id(target)]
    );
    let changes = plan.changes().collect::<Vec<_>>();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].bound().original(), original.unwrap_or(1));
    assert_eq!(
        changes[0].bound().field().map(|field| field.value()),
        original
    );
    assert_eq!(changes[0].remaining(), remaining);
    assert_eq!(plan.plan.work.initial_edges, plan.plan.work.decrements);
    assert_eq!(plan.plan.work.outgoing_edges, 0);
}

#[rstest]
#[case::overflow(u32::MAX, 2)]
#[case::wide_overflow(2, u64::MAX)]
#[case::original_contradiction(1, 1)]
fn added_count_errors_preserve_original_operands_and_consistency(
    #[case] original: u32,
    #[case] added: u64,
) {
    let path = test_util::temp::temp_path("added-source-errors.h5");
    let mut fixture = Fixture::new();
    let target = super::dataset(&mut fixture, original);
    fixture.link("first", target);
    fixture.link("second", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("first"))
        .unwrap();
    let additions = Cell::new(0);
    let (removed, mut counts) = phase.removal_queries([&deletion]).unwrap();
    let Err(error) = SourceRemovalPlan::with_additions(
        removed,
        |object| counts.evidence(object),
        |_| {
            additions.set(additions.get() + 1);
            added
        },
    ) else {
        panic!("expected original inconsistency or final overflow");
    };
    assert_eq!(
        error,
        if original == 1 {
            AddedCountError::Evidence(CountEvidenceError::Contradiction(CountContradiction {
                object: super::id(target),
                observed: 1,
                minimum: 2,
            }))
        } else {
            AddedCountError::Arithmetic {
                object: super::id(target),
                error: CountArithmeticError::Overflow {
                    original,
                    added,
                    removed: 1,
                },
            }
        }
    );
    assert_eq!(additions.get(), usize::from(original != 1));
}

#[rstest]
#[case::root_without_additions(RetainedSubject::Root, 0)]
#[case::root_with_additions(RetainedSubject::Root, 1)]
#[case::named_type_without_additions(RetainedSubject::NamedType, 0)]
#[case::named_type_with_additions(RetainedSubject::NamedType, 1)]
fn retained_subjects_have_net_counts_and_no_zero_candidates(
    #[case] subject: RetainedSubject,
    #[case] added: u64,
) {
    let path = test_util::temp::temp_path("added-root-count.h5");
    let mut fixture = if subject == RetainedSubject::Root {
        Fixture::new().with_root_reference_count(NonZeroU32::new(2).unwrap())
    } else {
        Fixture::new()
    };
    let target = if subject == RetainedSubject::Root {
        Fixture::ROOT
    } else {
        super::header(
            &mut fixture,
            vec![
                Message::new(MessageType::DATATYPE, &datatype::f64_le()),
                Message::reference_count(1),
            ],
        )
    };
    fixture.link("self", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("self"))
        .unwrap();
    let (removed, mut counts) = phase.removal_queries([&deletion]).unwrap();
    let plan =
        SourceRemovalPlan::with_additions(removed, |object| counts.evidence(object), |_| added)
            .unwrap()
            .unwrap();
    let target = super::id(target);
    assert_eq!(plan.graph_remaining(target), Some(0));
    assert_eq!(
        plan.declared_remaining(target),
        Some(u32::from(subject == RetainedSubject::Root) + u32::try_from(added).unwrap())
    );
    assert_eq!(plan.counted_zero_objects().count(), 0);
    assert_eq!(plan.retained_objects().collect::<Vec<_>>(), [target]);
    assert_eq!(plan.plan.work.outgoing_edges, 0);
}

#[rstest]
#[case::ordinary(Owner::Ordinary)]
#[case::positive(Owner::Positive)]
#[case::excess(Owner::Excess)]
#[case::unknown_count(Owner::UnknownCount)]
#[case::retained(Owner::Retained)]
fn additions_preserve_owner_propagation_restrictions(#[case] owner: Owner) {
    let path = test_util::temp::temp_path("added-owner-counts.h5");
    let mut fixture = Fixture::new();
    let child = super::dataset(&mut fixture, 1);
    let mut records = vec![global_heap_fixture::hard_link("child", child)];
    if owner == Owner::Retained {
        records.extend(Form::CompactDataset.messages(
            &mut fixture,
            &Datatype::FixedPoint(FixedPointType::i32()),
            &17_i32.to_le_bytes(),
        ));
    }
    records.push(Message::reference_count(if owner == Owner::Excess {
        2
    } else {
        1
    }));
    let target = super::header(&mut fixture, records);
    fixture.link("owner", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("owner"))
        .unwrap();
    let (removed, mut counts) = phase.removal_queries([&deletion]).unwrap();
    let plan = SourceRemovalPlan::with_additions(
        removed,
        |object| {
            if object == super::id(target) && owner == Owner::UnknownCount {
                Ok(None)
            } else {
                counts.evidence(object)
            }
        },
        |object| u64::from(object == super::id(target) && owner == Owner::Positive),
    )
    .unwrap()
    .unwrap();
    let released = owner == Owner::Ordinary;
    assert_eq!(plan.graph_remaining(super::id(target)), Some(0));
    assert_eq!(
        plan.declared_remaining(super::id(target)),
        (owner != Owner::UnknownCount)
            .then_some(u32::from(matches!(owner, Owner::Positive | Owner::Excess)))
    );
    assert_eq!(
        plan.graph_remaining(super::id(child)),
        Some(u32::from(!released))
    );
    assert_eq!(
        plan.declared_remaining(super::id(child)),
        released.then_some(0)
    );
    assert_eq!(
        plan.counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        if released {
            vec![super::id(target), super::id(child)]
        } else {
            vec![]
        }
    );
    assert_eq!(plan.plan.work.outgoing_edges, usize::from(released));
}

#[test]
fn unavailable_removal_topology_does_not_query_counts_or_additions() {
    let path = test_util::temp::temp_path("added-empty-topology.h5");
    Fixture::new().write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let before = DISCOVERY_WORK.with(Cell::get);
    let (removed, _) = phase.removal_queries([]).unwrap();
    let calls = Cell::new(0);
    let plan = SourceRemovalPlan::with_additions(
        removed,
        |_| {
            calls.set(calls.get() + 1);
            Ok(None)
        },
        |_| {
            calls.set(calls.get() + 1);
            0
        },
    )
    .unwrap();
    assert!(plan.is_none());
    assert_eq!(calls.get(), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), before);
    assert!(phase.topology.is_none() && phase.counts.is_empty() && phase.identity.is_none());
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Owner {
    Excess,
    Ordinary,
    Positive,
    Retained,
    UnknownCount,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RetainedSubject {
    NamedType,
    Root,
}
