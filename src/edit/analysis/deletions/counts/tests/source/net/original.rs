use core::cell::Cell;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use core::mem;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::Links;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FileBuilder;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::counts::changes::owned::OwnedCountChange;
use crate::edit::analysis::deletions::counts::net::original::OriginalOwnerError;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::deletions::counts::ownership::CountHandoffError;
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::analysis::{AnalysisPhase, DISCOVERY_WORK};
use crate::edit::test_util as edit_test_util;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::empty(false, 0, Links::Named, Some(2))]
#[case::unchanged(true, 1, Links::Named, Some(2))]
#[case::two_emissions(true, 2, Links::Named, Some(2))]
#[case::outside_view(false, 1, Links::Anonymous, Some(2))]
#[case::implicit(false, 1, Links::Anonymous, None)]
#[case::original_zero(false, 1, Links::Anonymous, Some(0))]
fn original_owner_facts_retain_complete_counts_and_copy_multiplicity(
    #[case] removed: bool,
    #[case] emissions: usize,
    #[case] links: Links,
    #[case] original: Option<u32>,
) {
    let path = test_util::temp::temp_path("original-owner-counts.h5");
    let (engine, tree, target, calls) = super::prepared(&path, original, links);
    let mut phase = AnalysisPhase::new(&engine);
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase
        .added_copy_users(std::iter::repeat_n(Some(&copy), emissions))
        .unwrap()
        .unwrap();
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
    calls.store(0, Ordering::Relaxed);
    let work = DISCOVERY_WORK.with(Cell::get);
    let facts = plan.original_owners(&engine).unwrap();
    drop(plan);
    drop(phase);
    let represented = removed || emissions > 0;
    let expected = if represented {
        vec![(
            target,
            original.unwrap_or(1),
            original.unwrap_or(1) - u32::from(removed) + u32::try_from(emissions).unwrap(),
        )]
    } else {
        vec![]
    };
    assert_eq!(
        facts
            .counts()
            .changes()
            .iter()
            .map(|change| (
                change.object(),
                change.original().value(),
                change.remaining()
            ))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        facts.storage().reference_roots(),
        if represented { vec![target] } else { vec![] }
    );
    assert!(facts.eligible_owners().is_empty());
    assert!(facts.storage().allocations().is_empty());
    if represented {
        assert_eq!(
            facts.counts().changes()[0]
                .field()
                .map(|field| field.value()),
            original
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), work);
}

#[rstest]
#[case::empty(false)]
#[case::removal_only(true)]
fn original_owner_inventory_retains_matching_source_after_phase_ends(#[case] nonempty: bool) {
    let path = test_util::temp::temp_path("original-owner-source.h5");
    let mut fixture = Fixture::new();
    let address = super::super::dataset(&mut fixture, 2);
    fixture.link("first", address);
    fixture.link("second", address);
    fixture.write(&path);
    let target = super::super::id(address);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, super::super::id(Fixture::ROOT))
        .unwrap();
    let domain = match phase.supported_owner_domain(&source).unwrap() {
        OwnerDomain::Available(domain) => domain,
        OwnerDomain::Unavailable(reason) => {
            panic!("expected a supported original owner domain: {reason:?}")
        }
    };
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletion = nonempty.then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("first"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletion.as_ref(), &added)
        .unwrap()
        .unwrap();
    let facts = plan.original_owners(&engine).unwrap();
    assert_eq!(facts.counts().root(), nonempty.then_some(source.object()));
    drop(plan);
    calls.store(0, Ordering::Relaxed);
    let work = DISCOVERY_WORK.with(Cell::get);
    let inventory = facts
        .qualify(
            &engine,
            &source,
            phase.physical_claims(&source).unwrap(),
            domain,
        )
        .unwrap();
    assert!(inventory.matches(&engine, phase.physical_claims(&source).unwrap()));
    drop(phase);
    assert_eq!(inventory.source().object(), source.object());
    assert!(Arc::ptr_eq(&inventory.source().identity, &source.identity));
    assert_eq!(
        inventory.facts().counts().changes().len(),
        usize::from(nonempty)
    );
    assert_eq!(
        inventory.facts().storage().reference_roots(),
        if nonempty { vec![target] } else { vec![] }
    );
    assert!(inventory.facts().eligible_owners().is_empty());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), work);
}

#[rstest]
#[case::phase(Association::Phase, OriginalOwnerError::Phase)]
#[case::claims(Association::Claims, OriginalOwnerError::Phase)]
#[case::root(Association::Root, OriginalOwnerError::Root)]
#[case::domain(Association::Domain, OriginalOwnerError::Domain)]
#[case::session(Association::Session, OriginalOwnerError::Session)]
fn original_owner_inventory_rejects_incompatible_source_associations(
    #[case] association: Association,
    #[case] expected: OriginalOwnerError,
) {
    let path = test_util::temp::temp_path("original-owner-association.h5");
    let other_path = test_util::temp::temp_path("original-owner-other-session.h5");
    let mut fixture = Fixture::new();
    let address = super::super::dataset(&mut fixture, 2);
    fixture.link("first", address);
    fixture.link("second", address);
    fixture.write(&path);
    std::fs::copy(&path, &other_path).unwrap();
    let target = super::super::id(address);
    let engine = edit_test_util::open_engine(&path);
    let other_engine = edit_test_util::open_engine(&other_path);
    let mut phase = AnalysisPhase::new(&engine);
    let root = super::super::id(Fixture::ROOT);
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let mut other = AnalysisPhase::new(&engine);
    let other_source = other.reconstruction_source(&engine, root).unwrap();
    let domain_source = if matches!(association, Association::Domain | Association::Phase) {
        &other_source
    } else {
        &source
    };
    let domain_phase = if matches!(association, Association::Domain | Association::Phase) {
        &mut other
    } else {
        &mut phase
    };
    let OwnerDomain::Available(domain) =
        domain_phase.supported_owner_domain(domain_source).unwrap()
    else {
        panic!("expected a supported original owner domain");
    };
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("first"))
        .unwrap();
    let plan = phase.net_count_plan([&deletion], &added).unwrap().unwrap();
    let facts = plan.original_owners(&engine).unwrap();
    drop(plan);
    let wrong_root = phase.reconstruction_source(&engine, target).unwrap();
    let selected_source = match association {
        Association::Phase => &other_source,
        Association::Root => &wrong_root,
        _ => &source,
    };
    let claims = if matches!(association, Association::Phase | Association::Claims) {
        other.physical_claims(&other_source).unwrap()
    } else {
        phase.physical_claims(&source).unwrap()
    };
    let selected_engine = if matches!(association, Association::Session) {
        &other_engine
    } else {
        &engine
    };
    let error = facts
        .qualify(selected_engine, selected_source, claims, domain)
        .err()
        .unwrap();
    assert_eq!(error, expected);
}

#[test]
fn distinct_original_owner_plans_keep_their_own_counts_and_retained_roots() {
    let path = test_util::temp::temp_path("original-owner-plans.h5");
    let mut fixture = Fixture::new();
    let address = super::super::dataset(&mut fixture, 2);
    fixture.link("first", address);
    fixture.link("second", address);
    fixture.write(&path);
    let target = super::super::id(address);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let first_edge = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("first"))
        .unwrap();
    let second_edge = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("second"))
        .unwrap();
    let first = phase
        .net_count_plan([&first_edge], &added)
        .unwrap()
        .unwrap()
        .original_owners(&engine)
        .unwrap();
    let second = phase
        .net_count_plan([&first_edge, &second_edge], &added)
        .unwrap()
        .unwrap()
        .original_owners(&engine)
        .unwrap();
    drop(phase);
    assert_eq!(first.counts().changes()[0].remaining(), 1);
    assert_eq!(second.counts().changes()[0].remaining(), 0);
    assert_eq!(first.storage().reference_roots(), [target]);
    assert!(second.storage().reference_roots().is_empty());
    assert!(first.eligible_owners().is_empty());
    assert_eq!(second.eligible_owners(), [target]);
    assert!(first.storage().allocations().is_empty());
    assert_eq!(second.storage().allocations().len(), 1);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(false)]
#[case::nonempty(true)]
fn original_owner_facts_keep_bounded_metadata_and_reject_other_engines(#[case] nonempty: bool) {
    let path = test_util::temp::temp_path("original-owner-resources.h5");
    let other_path = test_util::temp::temp_path("original-owner-resource-other.h5");
    let (engine, tree, target, calls) = super::prepared(&path, Some(2), Links::Named);
    FileBuilder::new().write(&other_path).unwrap();
    let other = edit_test_util::open_engine(&other_path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = super::additions(&mut phase, &tree, nonempty);
    let plan = phase.net_count_plan([], &added).unwrap().unwrap();
    assert_eq!(
        plan.original_owners(&other).err().unwrap(),
        CountHandoffError::Engine
    );
    calls.store(0, Ordering::Relaxed);
    let work = DISCOVERY_WORK.with(Cell::get);
    let (facts, measured) = allocation::measure("original_owner_facts", || {
        plan.original_owners(&engine).unwrap()
    });
    assert_eq!(facts.counts().changes().len(), usize::from(nonempty));
    assert_eq!(
        facts.storage().reference_roots(),
        if nonempty { vec![target] } else { vec![] }
    );
    if nonempty {
        let floor =
            u64::try_from(mem::size_of::<OwnedCountChange>() + mem::size_of::<ObjectId>()).unwrap();
        assert!(
            measured.blocks > 0 && measured.bytes >= floor && measured.bytes <= 4 * floor,
            "{measured}"
        );
    } else {
        assert_eq!((measured.blocks, measured.bytes), (0, 0), "{measured}");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), work);
}

#[derive(Clone, Copy)]
enum Association {
    Claims,
    Domain,
    Phase,
    Root,
    Session,
}
