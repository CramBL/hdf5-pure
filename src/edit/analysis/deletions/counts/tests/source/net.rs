use core::cell::Cell;
use core::num::NonZeroU32;
use core::num::NonZeroU64;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::datatype;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::object_header::v2;
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util_hdf5::global_heap_fixture::{Fixture, Form};

use super::super::super::ownership::CountHandoffError;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::super::super::ownership::SurvivingCount;
use super::{CountContradiction, CountEvidenceError};
use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::added_users::AddedCopyUsers;
use crate::edit::analysis::counts::added_users::CopyUsersError;
use crate::edit::analysis::counts::balance::CountArithmeticError;
use crate::edit::analysis::counts::changes::OriginalCount;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::counts::changes::owned::OwnedCountChange;
use crate::edit::analysis::counts::changes::owned::{
    CountDisposition, OwnedOriginalCount, UnresolvedCount,
};
use crate::edit::analysis::counts::test_util as count_test_util;
use crate::edit::analysis::counts::{CountEvidence, OrdinaryCountSubject};
use crate::edit::analysis::deletions::counts::net::NetCountError;
use crate::edit::analysis::{AnalysisPhase, DISCOVERY_WORK, PlannedDeletion};
use crate::edit::copy::CopyTree;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::image::{CountingImage, HandleImage};
use crate::object_graph::ObjectId;
use crate::object_graph::TraversalBudget;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::source::MetadataCacheConfig;
use crate::{Datatype, DatatypeByteOrder, FileLocking, FloatingPointType, PhysicalAddress};

mod storage;

#[rstest]
#[case::named(Some(2), Links::Named, &[], true, false, 3)]
#[case::outside(Some(1), Links::Anonymous, &[], true, false, 2)]
#[case::mixed(Some(2), Links::Named, &["type"], true, false, 2)]
#[case::untouched_target(Some(2), Links::Named, &["source"], true, false, 3)]
#[case::maximum(Some(u32::MAX), Links::Aliases, &["type", "type"], true, false, u32::MAX)]
#[case::removal_only(Some(2), Links::Named, &["type"], false, false, 1)]
#[case::implicit(None, Links::Anonymous, &[], true, false, 2)]
#[case::unproven_zero(Some(0), Links::Anonymous, &[], true, false, 1)]
#[case::partial_allocation(Some(2), Links::Named, &[], true, true, 3)]
fn net_counts_project_each_target_once_with_original_fields(
    #[case] original: Option<u32>,
    #[case] links: Links,
    #[case] paths: &[&str],
    #[case] copied: bool,
    #[case] partial: bool,
    #[case] expected: u32,
) {
    let path = test_util::temp::temp_path("net-count-projection.h5");
    let (engine, tree, target, calls) = prepared(&path, original, links);
    let mut phase = AnalysisPhase::new(&engine);
    let observed = phase.object_reference_count(target).unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);
    if partial {
        phase.counts.insert(
            target,
            Ok(count_test_util::observation_without_allocation(observed)),
        );
    }
    let added = additions(&mut phase, &tree, copied);
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
    if paths.contains(&"source") {
        let (_, source) = phase
            .topology()
            .unwrap()
            .find_link(&ObjectPathBuf::parse("source"))
            .unwrap();
        assert_eq!(
            phase.object_reference_count(source).unwrap(),
            ObjectReferenceCount::ImplicitOne
        );
    }
    let before = DISCOVERY_WORK.with(Cell::get);
    calls.store(0, Ordering::Relaxed);
    let plan = phase.net_count_plan(&deletions, &added).unwrap().unwrap();
    assert!(plan.same_engine(&engine));
    assert_eq!(plan.source().unwrap().object(), super::id(Fixture::ROOT));
    assert_eq!(plan.removals().is_some(), !paths.is_empty());
    let changes = plan.changes().collect::<Vec<_>>();
    let deleted_source = paths.contains(&"source");
    assert_eq!(changes.len(), 1 + usize::from(deleted_source));
    assert_eq!(
        changes
            .iter()
            .filter(|change| change.object() == target)
            .count(),
        1
    );
    let changed = changes
        .iter()
        .find(|change| change.object() == target)
        .unwrap();
    let evidence = plan
        .evidence_changes()
        .find(|change| match change.original() {
            OriginalCount::Topology(evidence) => evidence.bound().object() == target,
            OriginalCount::Referenced(observed) => observed.object() == target,
        })
        .unwrap();
    assert_eq!(evidence.remaining(), expected);
    match evidence.original() {
        OriginalCount::Topology(topology) => {
            assert!(paths.contains(&"type"));
            assert!(matches!(topology, CountEvidence::Retained(_)));
            let bound = topology.bound();
            assert_eq!(bound.original(), original.unwrap_or(1));
            assert_eq!(bound.field(), changed.field());
            assert_eq!(bound.header_allocation(), changed.header_allocation());
            let stored = plan
                .removals()
                .unwrap()
                .plan
                .remaining
                .values()
                .find_map(|state| {
                    state
                        .evidence
                        .as_ref()
                        .filter(|other| other.bound().object() == target)
                })
                .unwrap();
            assert!(std::ptr::eq(topology, stored));
        }
        OriginalCount::Referenced(observed) => {
            assert!(!paths.contains(&"type"));
            assert_eq!(observed.original(), original.unwrap_or(1));
            assert_eq!(observed.field(), changed.field());
            assert_eq!(observed.header_allocation(), changed.header_allocation());
            assert_eq!(observed.incoming(), (!paths.is_empty()).then_some(1));
            assert_eq!(
                observed.minimum().map(|count| count.get()),
                (!paths.is_empty()).then_some(1)
            );
        }
    }
    assert_eq!(changed.original(), original.unwrap_or(1));
    assert_eq!(changed.remaining(), expected);
    assert_eq!(changed.field().map(|field| field.value()), original);
    assert_eq!(
        changed.header_allocation().is_none(),
        partial || original.is_none()
    );
    if let Some(removals) = plan.removals() {
        assert_eq!(
            removals.counted_zero_objects().count(),
            usize::from(deleted_source)
        );
        assert_eq!(
            removals.graph_remaining(target),
            Some(if deleted_source {
                1
            } else {
                u32::from(links == Links::Aliases)
            })
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), before);
    drop(plan);
    assert_eq!(phase.topology.is_some(), !paths.is_empty());
    assert_eq!(phase.object_reference_count(target).unwrap(), observed);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[rstest]
#[case::addition_only(Links::Named, &[])]
#[case::anonymous(Links::Anonymous, &[])]
#[case::source_removed(Links::Named, &["source"])]
#[case::shared_view(Links::Named, &["type"])]
fn referenced_targets_remain_storage_reference_roots(#[case] links: Links, #[case] paths: &[&str]) {
    let path = test_util::temp::temp_path("referenced-storage-roots.h5");
    let (engine, tree, target, _) = prepared(&path, Some(2), links);
    let mut phase = AnalysisPhase::new(&engine);
    let added = additions(&mut phase, &tree, true);
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
    let plan = phase.net_count_plan(&deletions, &added).unwrap().unwrap();
    let storage = plan.collect_storage();
    assert_eq!(storage.reference_roots(), [target]);
    assert!(
        storage
            .allocations()
            .iter()
            .all(|(range, _)| range.start() != target.header_address())
    );
    assert!(storage.same_session(&engine));
    assert!(storage.shares_phase(plan.source().unwrap()));
}

#[rstest]
#[case::overflow(Some(u32::MAX), Links::Anonymous)]
#[case::original_contradiction(Some(1), Links::Aliases)]
fn net_count_errors_describe_original_counts_before_prospective_additions(
    #[case] original: Option<u32>,
    #[case] links: Links,
) {
    let path = test_util::temp::temp_path("net-count-errors.h5");
    let (engine, tree, target, _) = prepared(&path, original, links);
    let mut phase = AnalysisPhase::new(&engine);
    let added = additions(&mut phase, &tree, true);
    let deletions = if links == Links::Aliases {
        vec![
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse("type"))
                .unwrap(),
        ]
    } else {
        vec![]
    };
    let Err(error) = phase.net_count_plan(&deletions, &added) else {
        panic!("expected original contradiction or final overflow")
    };
    assert_eq!(
        error,
        if links == Links::Aliases {
            NetCountError::Evidence(CountEvidenceError::Contradiction(CountContradiction {
                object: target,
                observed: 1,
                minimum: 2,
            }))
        } else {
            NetCountError::Arithmetic {
                object: target,
                error: CountArithmeticError::Overflow {
                    original: u32::MAX,
                    added: 1,
                    removed: 0,
                },
            }
        }
    );
}

#[rstest]
#[case::count_for_additions(Incomplete::Added)]
#[case::count_for_removals(Incomplete::Removed)]
#[case::unresolved_edge(Incomplete::Edge)]
#[case::topology(Incomplete::Topology)]
fn missing_required_proofs_produce_no_complete_net_plan(#[case] mode: Incomplete) {
    let path = test_util::temp::temp_path("incomplete-net-counts.h5");
    let (engine, tree, target, _) = prepared(&path, Some(2), Links::Named);
    let mut phase = AnalysisPhase::new(&engine);
    let added = additions(&mut phase, &tree, mode != Incomplete::Removed);
    if matches!(mode, Incomplete::Added | Incomplete::Removed) {
        phase.counts.insert(
            target,
            Ok(count_test_util::observation_without_allocation(
                ObjectReferenceCount::Ambiguous,
            )),
        );
    }
    let deletions = match mode {
        Incomplete::Added => vec![],
        Incomplete::Removed => vec![
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse("type"))
                .unwrap(),
        ],
        Incomplete::Edge | Incomplete::Topology => vec![PlannedDeletion::unresolved(
            LinkNameBuf::new("type").unwrap(),
            target.header_address(),
        )],
    };
    if mode == Incomplete::Topology {
        phase.budget = TraversalBudget::new(0, 0);
    }
    assert!(phase.net_count_plan(&deletions, &added).unwrap().is_none());
    if mode == Incomplete::Topology {
        assert!(phase.counts.is_empty());
    }
}

#[test]
fn net_count_planning_preserves_source_error_variants_and_fields() {
    let path = test_util::temp::temp_path("failed-net-source.h5");
    let (engine, tree, target, _) = prepared(&path, Some(2), Links::Named);
    let mut phase = AnalysisPhase::new(&engine);
    let added = additions(&mut phase, &tree, true);
    let failure = FormatError::UnexpectedSourceEof {
        expected: 1,
        available: 0,
    };
    phase.counts.insert(target, Err(failure.clone()));
    let Err(error) = phase.net_count_plan([], &added) else {
        panic!("expected the cached source failure")
    };
    assert_eq!(
        error,
        NetCountError::Evidence(CountEvidenceError::Source(failure))
    );
}

#[test]
fn removal_only_plans_preserve_eligible_ordinary_transitions() {
    let path = test_util::temp::temp_path("ordinary-net-counts.h5");
    let mut fixture = Fixture::new();
    let child = super::dataset(&mut fixture, 1);
    let owner = super::group(
        &mut fixture,
        1,
        vec![test_util_hdf5::global_heap_fixture::hard_link(
            "child", child,
        )],
    );
    fixture.link("owner", owner);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("owner"))
        .unwrap();
    let plan = phase.net_count_plan([&deletion], &added).unwrap().unwrap();
    assert_eq!(
        plan.removals()
            .unwrap()
            .counted_zero_objects()
            .map(|object| object.id())
            .collect::<Vec<_>>(),
        [super::id(owner), super::id(child)]
    );
    assert_eq!(
        plan.changes()
            .map(|change| (change.object(), change.original(), change.remaining()))
            .collect::<Vec<_>>(),
        [(super::id(owner), 1, 0), (super::id(child), 1, 0)]
    );
    let evidence = plan
        .evidence_changes()
        .map(|change| {
            let OriginalCount::Topology(CountEvidence::Ordinary(ordinary)) = change.original()
            else {
                panic!("expected the original ordinary count evidence")
            };
            (ordinary.subject(), change.remaining())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        evidence,
        [
            (OrdinaryCountSubject::Group, 0),
            (OrdinaryCountSubject::Dataset, 0)
        ]
    );
}

#[test]
fn retained_zero_counts_keep_their_original_evidence() {
    let path = test_util::temp::temp_path("retained-zero-count-view.h5");
    let mut fixture = Fixture::new();
    let address = fixture.append_header(vec![
        Message::new(MessageType::DATATYPE, &datatype::f64_le()),
        Message::reference_count(1),
    ]);
    fixture.link("type", address);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("type"))
        .unwrap();
    let plan = phase.net_count_plan([&deletion], &added).unwrap().unwrap();
    let evidence = plan.evidence_changes().collect::<Vec<_>>();
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].remaining(), 0);
    let OriginalCount::Topology(CountEvidence::Retained(bound)) = evidence[0].original() else {
        panic!("expected retained topology count evidence")
    };
    assert_eq!(bound.object(), super::id(address));
    assert_eq!(bound.original(), 1);
    assert_eq!(bound.field().unwrap().value(), 1);
    assert_eq!(
        bound.header_allocation().unwrap().range(),
        bound.field().unwrap().chunk()
    );
    assert_eq!(plan.removals().unwrap().counted_zero_objects().count(), 0);
}

#[rstest]
#[case::phase(false)]
#[case::prospective_root(true)]
fn incompatible_origins_fail_before_discovery_or_count_reads(#[case] root: bool) {
    let path = test_util::temp::temp_path("net-count-origin.h5");
    let (engine, tree, target, calls) = prepared(&path, Some(2), Links::Named);
    let mut first = AnalysisPhase::new(&engine);
    let added = additions(&mut first, &tree, true);
    let actual = ObjectId::new(PhysicalAddress::new(Fixture::ROOT + 8));
    let mut other = if root {
        AnalysisPhase::prospective(&engine, actual.header_address())
    } else {
        AnalysisPhase::new(&engine)
    };
    let deletions = vec![PlannedDeletion::unresolved(
        LinkNameBuf::new("type").unwrap(),
        target.header_address(),
    )];
    let empty = other.added_copy_users([]).unwrap().unwrap();
    let before = DISCOVERY_WORK.with(Cell::get);
    let Err(error) = other.net_count_plan(&deletions, if root { &empty } else { &added }) else {
        panic!("expected the incompatible origin")
    };
    assert_eq!(
        error,
        NetCountError::Association(if root {
            CopyUsersError::Root {
                expected: super::id(Fixture::ROOT),
                actual,
            }
        } else {
            CopyUsersError::Phase
        })
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), before);
    assert!(other.topology.is_none() && other.counts.is_empty());
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn an_empty_net_plan_skips_source_and_heap_work() {
    let path = test_util::temp::temp_path("empty-net-counts.h5");
    let (engine, tree, target, calls) = prepared(&path, Some(2), Links::Named);
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let before = DISCOVERY_WORK.with(Cell::get);
    let (plan, measured) =
        allocation::measure("empty_net_count_plan", || phase.net_count_plan([], &added));
    let plan = plan.unwrap().unwrap();
    assert!(plan.same_engine(&engine));
    assert!(plan.source().is_none() && plan.removals().is_none());
    let (entries, empty_views) = allocation::measure("empty_net_original_views", || {
        plan.evidence_changes().count()
    });
    assert_eq!(entries, 0);
    assert_eq!(plan.changes().count(), 0);
    drop(plan);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), before);
    assert!(phase.topology.is_none() && phase.identity.is_none() && phase.counts.is_empty());
    let added = additions(&mut phase, &tree, true);
    let (plan, positive) = allocation::measure("nonempty_net_count_plan", || {
        phase.net_count_plan([], &added)
    });
    let plan = plan.unwrap().unwrap();
    let (sum, original_views) = allocation::measure("nonempty_net_original_views", || {
        plan.evidence_changes()
            .map(|change| u64::from(change.remaining()))
            .sum::<u64>()
    });
    assert_eq!(sum, 3);
    assert_eq!(
        plan.changes()
            .map(|change| (change.object(), change.remaining()))
            .collect::<Vec<_>>(),
        [(target, 3)]
    );
    assert!(positive.blocks > 0 && positive.bytes > 0, "{positive}");
    assert!(calls.load(Ordering::Relaxed) > 0);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(
        (empty_views.blocks, empty_views.bytes),
        (0, 0),
        "{empty_views}"
    );
    assert_eq!(
        (original_views.blocks, original_views.bytes),
        (0, 0),
        "{original_views}"
    );
}

#[rstest]
#[case::outside(
    Some(1),
    Links::Anonymous,
    false,
    true,
    CountContext::Default,
    2,
    Expected::Surviving
)]
#[case::original_zero(
    Some(0),
    Links::Anonymous,
    false,
    true,
    CountContext::Default,
    1,
    Expected::Surviving
)]
#[case::implicit(
    None,
    Links::Anonymous,
    false,
    true,
    CountContext::Default,
    2,
    Expected::Missing
)]
#[case::unchanged(
    Some(2),
    Links::Named,
    true,
    true,
    CountContext::Default,
    2,
    Expected::Unchanged
)]
#[case::maximum(
    Some(u32::MAX),
    Links::Aliases,
    true,
    true,
    CountContext::Default,
    u32::MAX,
    Expected::Unchanged
)]
#[case::implicit_unchanged(
    None,
    Links::Named,
    true,
    true,
    CountContext::Default,
    1,
    Expected::Unchanged
)]
#[case::retained_zero(
    Some(1),
    Links::Named,
    true,
    false,
    CountContext::Default,
    0,
    Expected::Zero
)]
#[case::partial_allocation(
    Some(2),
    Links::Named,
    false,
    true,
    CountContext::PartialAllocation,
    3,
    Expected::Surviving
)]
#[case::outside_topology(
    Some(1),
    Links::Anonymous,
    false,
    true,
    CountContext::Topology,
    2,
    Expected::Surviving
)]
#[case::named_topology(
    Some(2),
    Links::Named,
    false,
    true,
    CountContext::Topology,
    3,
    Expected::Surviving
)]
#[case::zero_topology(
    Some(0),
    Links::Anonymous,
    false,
    true,
    CountContext::Topology,
    1,
    Expected::Surviving
)]
fn owned_inventory_preserves_complete_original_facts_after_phase_release(
    #[case] original: Option<u32>,
    #[case] links: Links,
    #[case] removed: bool,
    #[case] copied: bool,
    #[case] context: CountContext,
    #[case] remaining: u32,
    #[case] expected: Expected,
) {
    let path = test_util::temp::temp_path("owned-net-counts.h5");
    let (engine, tree, target, calls) = prepared(&path, original, links);
    let mut phase = AnalysisPhase::new(&engine);
    let origin = phase.reconstruction_source(&engine, target).unwrap();
    if matches!(context, CountContext::PartialAllocation) {
        let observed = phase.object_reference_count(target).unwrap();
        phase.counts.insert(
            target,
            Ok(count_test_util::observation_without_allocation(observed)),
        );
    }
    if matches!(context, CountContext::Topology) {
        phase.topology().unwrap();
    }
    let added = additions(&mut phase, &tree, copied);
    let deletion = removed.then(|| {
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
    let numeric = plan.changes().next().unwrap();
    let work = DISCOVERY_WORK.with(Cell::get);
    calls.store(0, Ordering::Relaxed);
    let inventory = plan.owned_counts(&engine).unwrap();
    drop(plan);
    drop(phase);
    drop(tree);
    let (inventory, surviving) = inventory.partition().into_parts();
    assert!(inventory.same_session(&engine));
    assert!(inventory.shares_phase(&origin));
    assert_eq!(inventory.root(), Some(super::id(Fixture::ROOT)));
    assert_eq!(inventory.changes().len(), 1);
    let change = &inventory.changes()[0];
    assert_eq!(change.object(), target);
    assert_eq!(change.original().value(), original.unwrap_or(1));
    assert_eq!(change.remaining(), remaining);
    assert_eq!(change.field(), numeric.field());
    assert_eq!(change.header_allocation(), numeric.header_allocation());
    assert_eq!(
        change.header_allocation().is_none(),
        matches!(context, CountContext::PartialAllocation) || original.is_none()
    );
    assert_eq!(
        change.original(),
        if removed {
            OwnedOriginalCount::Retained {
                original: NonZeroU32::new(original.unwrap_or(1)).unwrap(),
                incoming: if links == Links::Aliases { 2 } else { 1 },
            }
        } else {
            OwnedOriginalCount::Referenced {
                original: original.unwrap_or(1),
                incoming: matches!(context, CountContext::Topology)
                    .then_some(u32::from(links != Links::Anonymous)),
                minimum: (matches!(context, CountContext::Topology) && links != Links::Anonymous)
                    .then_some(NonZeroU64::new(1).unwrap()),
            }
        }
    );
    assert_eq!(
        surviving.updates().len(),
        usize::from(matches!(expected, Expected::Surviving))
    );
    match expected {
        Expected::Surviving => {
            let field = numeric.field().unwrap();
            let count = NonZeroU32::new(remaining).unwrap();
            let CountDisposition::Surviving {
                change: observed,
                field: actual,
                count: planned,
            } = change.disposition()
            else {
                panic!("expected a positive explicit count");
            };
            assert!(std::ptr::eq(observed, change));
            assert_eq!((*actual, planned), (field, count));
            let update = &surviving.updates()[0];
            assert_eq!(
                (
                    update.object(),
                    update.field(),
                    update.count(),
                    update.header_allocation()
                ),
                (target, field, count, numeric.header_allocation())
            );
            assert!(surviving.same_session(&engine) && surviving.shares_phase(&origin));
            assert_eq!(surviving.root(), inventory.root());
        }
        Expected::Unchanged => {
            let CountDisposition::Unchanged(observed) = change.disposition() else {
                panic!("expected an unchanged count");
            };
            assert!(std::ptr::eq(observed, change));
        }
        Expected::Missing | Expected::Zero => {
            let CountDisposition::Unresolved {
                change: observed,
                reason,
            } = change.disposition()
            else {
                panic!("expected an unresolved count");
            };
            assert!(std::ptr::eq(observed, change));
            assert_eq!(
                reason,
                if matches!(expected, Expected::Missing) {
                    UnresolvedCount::MissingField
                } else {
                    UnresolvedCount::Zero
                }
            );
        }
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), work);
    let mut other = AnalysisPhase::new(&engine);
    let other_origin = other.reconstruction_source(&engine, target).unwrap();
    assert!(!inventory.shares_phase(&other_origin));
    drop(other);
    drop(engine);
    let reopened = edit_test_util::open_engine(&path);
    assert!(!inventory.same_session(&reopened));
    assert_eq!(
        surviving.same_session(&reopened),
        !matches!(expected, Expected::Surviving)
    );
}

#[rstest]
#[case::empty(false)]
#[case::nonempty(true)]
fn owned_inventory_rejects_foreign_engines_and_preserves_empty_association(#[case] nonempty: bool) {
    let path = test_util::temp::temp_path("owned-count-engine.h5");
    let other_path = test_util::temp::temp_path("owned-count-other.h5");
    let (engine, tree, target, _) = prepared(&path, Some(2), Links::Named);
    drop(engine);
    std::fs::copy(&path, &other_path).unwrap();
    let (engine, calls) = open_counted_engine(&path);
    let other = edit_test_util::open_engine(&other_path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = additions(&mut phase, &tree, nonempty);
    let plan = phase.net_count_plan([], &added).unwrap().unwrap();
    calls.store(0, Ordering::Relaxed);
    let Err(error) = plan.owned_counts(&other) else {
        panic!("expected a different engine");
    };
    assert_eq!(error, CountHandoffError::Engine);
    let inventory = plan.owned_counts(&engine).unwrap();
    assert!(inventory.same_session(&engine));
    assert!(!inventory.same_session(&other));
    assert_eq!(
        inventory.root(),
        nonempty.then_some(super::id(Fixture::ROOT))
    );
    assert_eq!(
        inventory
            .changes()
            .iter()
            .map(|change| change.object())
            .collect::<Vec<_>>(),
        if nonempty { vec![target] } else { vec![] }
    );
    let (inventory, surviving) = inventory.partition().into_parts();
    assert_eq!(surviving.updates().len(), usize::from(nonempty));
    assert_eq!(inventory.changes().len(), usize::from(nonempty));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(false)]
#[case::unchanged(true)]
fn empty_and_unchanged_owned_partitions_skip_update_allocation(#[case] nonempty: bool) {
    let path = test_util::temp::temp_path("owned-count-empty.h5");
    let (engine, tree, _, calls) = prepared(&path, Some(2), Links::Named);
    let mut phase = AnalysisPhase::new(&engine);
    let added = additions(&mut phase, &tree, nonempty);
    let deletion = nonempty.then(|| {
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
    calls.store(0, Ordering::Relaxed);
    let (inventory, measured) = allocation::measure("owned_count_inventory", || {
        plan.owned_counts(&engine).unwrap()
    });
    assert_eq!(inventory.changes().len(), usize::from(nonempty));
    if nonempty {
        assert!(
            measured.blocks > 0
                && measured.live_bytes > 0
                && measured.bytes <= 4 * u64::try_from(size_of::<OwnedCountChange>()).unwrap(),
            "{measured}"
        );
    } else {
        assert_eq!((measured.blocks, measured.bytes), (0, 0), "{measured}");
    }
    let pointer = inventory.changes().as_ptr();
    let (partition, measured) =
        allocation::measure("owned_count_partition_empty", || inventory.partition());
    let (inventory, surviving) = partition.into_parts();
    assert_eq!((measured.blocks, measured.bytes), (0, 0), "{measured}");
    assert_eq!(inventory.changes().as_ptr(), pointer);
    assert!(surviving.updates().is_empty());
    assert_eq!(surviving.root(), None);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::one(1)]
#[case::wide(64)]
fn owned_inventory_and_positive_partition_retain_bounded_per_target_state(#[case] count: usize) {
    let path = test_util::temp::temp_path("owned-count-wide.h5");
    let mut fixture = Fixture::new();
    let mut links = Vec::new();
    let mut expected = BTreeMap::new();
    let record = v2::message_record(
        &Message::reference_count(2),
        v2::HeaderFlags::CHUNK_SIZE_WIDTH,
    );
    let value = 2_u32.to_le_bytes();
    assert!(record.ends_with(&value));
    for index in 0..count {
        let target = super::dataset(&mut fixture, 2);
        let start = usize::try_from(target).unwrap();
        let header = &fixture.image.as_bytes()[start..];
        let positions = header
            .windows(record.len())
            .enumerate()
            .filter_map(|(at, bytes)| (bytes == record).then_some(at))
            .collect::<Vec<_>>();
        assert_eq!(positions.len(), 1);
        let allocation = FileRange::new(
            PhysicalAddress::new(target),
            u64::try_from(header.len()).unwrap(),
        )
        .unwrap();
        let field = FileRange::new(
            PhysicalAddress::new(
                u64::try_from(start + positions[0] + record.len() - value.len()).unwrap(),
            ),
            u64::try_from(value.len()).unwrap(),
        )
        .unwrap();
        expected.insert(
            super::id(target),
            ((2, field, allocation, Some(true)), 1, Some(allocation)),
        );
        links.push(test_util_hdf5::global_heap_fixture::hard_link(
            &format!("a{index}"),
            target,
        ));
        links.push(test_util_hdf5::global_heap_fixture::hard_link(
            &format!("b{index}"),
            target,
        ));
    }
    assert_eq!(expected.len(), count);
    let owner = super::group(&mut fixture, 1, links);
    fixture.link("owner", owner);
    fixture.write(&path);
    let (engine, calls) = open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletions = (0..count)
        .map(|index| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(&format!("owner/a{index}")))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let plan = phase.net_count_plan(&deletions, &added).unwrap().unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    let work = DISCOVERY_WORK.with(Cell::get);
    let (inventory, measured) = allocation::measure("owned_counts_wide_inventory", || {
        plan.owned_counts(&engine).unwrap()
    });
    let bytes = u64::try_from(count * size_of::<OwnedCountChange>()).unwrap();
    assert!(
        measured.blocks > 0
            && measured.bytes >= bytes
            && measured.bytes <= 4 * bytes
            && measured.live_bytes <= 4 * bytes,
        "{measured}"
    );
    let pointer = inventory.changes().as_ptr();
    let (partition, measured) =
        allocation::measure("owned_counts_wide_partition", || inventory.partition());
    let (inventory, surviving) = partition.into_parts();
    let bytes = u64::try_from(count * size_of::<SurvivingCount>()).unwrap();
    assert!(
        measured.blocks > 0
            && measured.bytes >= bytes
            && measured.bytes <= 4 * bytes
            && measured.live_bytes <= 4 * bytes,
        "{measured}"
    );
    assert_eq!(inventory.changes().as_ptr(), pointer);
    assert_eq!(
        (inventory.changes().len(), surviving.updates().len()),
        (count, count)
    );
    assert_eq!(
        inventory
            .changes()
            .iter()
            .map(|change| change.original())
            .collect::<Vec<_>>(),
        vec![
            OwnedOriginalCount::Ordinary {
                original: NonZeroU32::new(2).unwrap(),
                incoming: 2,
                subject: OrdinaryCountSubject::Dataset
            };
            count
        ]
    );
    assert_eq!(
        surviving
            .updates()
            .iter()
            .map(|update| update.count().get())
            .collect::<Vec<_>>(),
        vec![1; count]
    );
    assert_eq!(
        inventory
            .changes()
            .iter()
            .map(|change| {
                let field = change.field().unwrap();
                (
                    change.object(),
                    (
                        (
                            field.value(),
                            field.range(),
                            field.chunk(),
                            field.flags().map(|flags| flags.is_empty()),
                        ),
                        change.remaining(),
                        change
                            .header_allocation()
                            .map(|allocation| allocation.range()),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>(),
        expected
    );
    assert_eq!(
        surviving
            .updates()
            .iter()
            .map(|update| {
                let field = update.field();
                (
                    update.object(),
                    (
                        (
                            field.value(),
                            field.range(),
                            field.chunk(),
                            field.flags().map(|flags| flags.is_empty()),
                        ),
                        update.count().get(),
                        update
                            .header_allocation()
                            .map(|allocation| allocation.range()),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>(),
        expected
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), work);
}

fn additions<'source>(
    phase: &mut AnalysisPhase<'source>,
    tree: &CopyTree,
    copied: bool,
) -> AddedCopyUsers<'source> {
    if copied {
        let copy = phase.copy_users(tree).unwrap().unwrap();
        phase.added_copy_users([Some(&copy)]).unwrap().unwrap()
    } else {
        phase.added_copy_users([]).unwrap().unwrap()
    }
}

fn prepared(
    path: &Path,
    original: Option<u32>,
    links: Links,
) -> (WriteEngine, CopyTree, ObjectId, Arc<AtomicU64>) {
    let mut fixture = Fixture::new();
    let mut records = vec![Message::new(MessageType::DATATYPE, &datatype::f64_le())];
    records.extend(original.map(Message::reference_count));
    let target = fixture.append_header(records);
    if links != Links::Anonymous {
        fixture.link("type", target);
    }
    if links == Links::Aliases {
        fixture.link("alias", target);
    }
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
            DatatypeByteOrder::LittleEndian,
        )),
        &17_f64.to_le_bytes(),
    );
    let message = records
        .iter_mut()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap();
    *message = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(target),
    );
    message.flags = MessageFlags::SHARED;
    let source = fixture.append_header(records);
    fixture.link("source", source);
    fixture.write(path);
    let (engine, calls) = open_counted_engine(path);
    let tree = WriteEngine::read_copy_subtree(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        super::id(source),
        false,
        engine.widths(),
    )
    .unwrap();
    calls.store(0, Ordering::Relaxed);
    (engine, tree, super::id(target), calls)
}

/// Opens an engine without metadata caching and counts image reads.
pub(in crate::edit::analysis::deletions::counts) fn open_counted_engine(
    path: &Path,
) -> (WriteEngine, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    MetadataCacheConfig::disabled(),
                )),
                Arc::default(),
                Arc::default(),
            )
            .with_read_calls(Arc::clone(&calls)),
        ))
    })
    .unwrap();
    (engine, calls)
}

#[derive(Clone, Copy)]
enum Expected {
    Missing,
    Surviving,
    Unchanged,
    Zero,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Links {
    Aliases,
    Anonymous,
    Named,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Incomplete {
    Added,
    Edge,
    Removed,
    Topology,
}

#[derive(Clone, Copy)]
enum CountContext {
    Default,
    PartialAllocation,
    Topology,
}
