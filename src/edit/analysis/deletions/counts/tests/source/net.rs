use core::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::datatype;
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util_hdf5::global_heap_fixture::{Fixture, Form};

use super::{CountContradiction, CountEvidenceError};
use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::added_users::AddedCopyUsers;
use crate::edit::analysis::counts::added_users::CopyUsersError;
use crate::edit::analysis::counts::balance::CountArithmeticError;
use crate::edit::analysis::counts::test_util as count_test_util;
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

#[rstest]
#[case::named(Some(2), Links::Named, &[], true, false, 3)]
#[case::outside(Some(1), Links::Anonymous, &[], true, false, 2)]
#[case::mixed(Some(2), Links::Named, &["type"], true, false, 2)]
#[case::untouched_target(Some(2), Links::Named, &["source"], true, false, 3)]
#[case::maximum(Some(u32::MAX), Links::Aliases, &["type", "type"], true, false, u32::MAX)]
#[case::removal_only(Some(2), Links::Named, &["type"], false, false, 1)]
#[case::implicit(None, Links::Anonymous, &[], true, false, 2)]
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
