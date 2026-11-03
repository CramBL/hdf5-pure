use core::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::FileLocking;
use crate::PhysicalAddress;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::image::CountingImage;
use crate::image::HandleImage;
use crate::object_graph::ObjectId;
use crate::object_graph::TraversalBudget;
use crate::source::MetadataCacheConfig;

#[rstest]
fn reconstruction_sources_share_the_lazy_topology_identity(
    #[values(false, true)] topology_first: bool,
) {
    let path = test_util::temp::temp_path("reconstruction-identity.h5");
    Fixture::new().write(&path);
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
    assert!(phase.identity.is_none());
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    if topology_first {
        assert!(phase.topology().is_some());
        assert!(calls.load(Ordering::Relaxed) > 0 && bytes.load(Ordering::Relaxed) > 0);
    }
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let object = ObjectId::new(PhysicalAddress::new(Fixture::ROOT));
    let first = phase.reconstruction_source(&engine, object).unwrap();
    let repeated = phase.reconstruction_source(&engine, object).unwrap();
    assert_eq!(first, repeated);
    assert_eq!(first.object(), object);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    if !topology_first {
        assert!(phase.topology.is_none());
        assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    }
    let topology = phase.topology().unwrap();
    assert!(Arc::ptr_eq(&first.identity, &topology.identity));
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 1));
    assert!(phase.counts.is_empty());
    assert!(phase.references.is_empty());
    assert!(phase.allocations.is_empty());
}

#[rstest]
#[case::matching(Relation::Matching)]
#[case::other_object(Relation::Object)]
#[case::other_phase(Relation::Phase)]
#[case::other_source(Relation::Source)]
fn reconstruction_source_equality_requires_the_same_phase_and_object(#[case] relation: Relation) {
    let first_path = test_util::temp::temp_path("first-reconstruction.h5");
    let second_path = test_util::temp::temp_path("second-reconstruction.h5");
    let mut fixture = Fixture::new();
    let child = fixture.append_header(vec![]);
    fixture.link("child", child);
    fixture.write(&first_path);
    std::fs::copy(&first_path, &second_path).unwrap();
    let first_engine = edit_test_util::open_engine(&first_path);
    let second_engine = edit_test_util::open_engine(&second_path);
    let mut first = AnalysisPhase::new(&first_engine);
    let mut second = AnalysisPhase::new(if matches!(relation, Relation::Source) {
        &second_engine
    } else {
        &first_engine
    });
    let root = ObjectId::new(PhysicalAddress::new(Fixture::ROOT));
    let original = first.reconstruction_source(&first_engine, root).unwrap();
    let compared = match relation {
        Relation::Matching => first.reconstruction_source(&first_engine, root).unwrap(),
        Relation::Object => first
            .reconstruction_source(&first_engine, ObjectId::new(PhysicalAddress::new(child)))
            .unwrap(),
        Relation::Phase | Relation::Source => second
            .reconstruction_source(
                if matches!(relation, Relation::Source) {
                    &second_engine
                } else {
                    &first_engine
                },
                root,
            )
            .unwrap(),
    };
    assert_eq!(original == compared, matches!(relation, Relation::Matching));
    assert!(first.topology.is_none() && second.topology.is_none());
    assert!(first.counts.is_empty() && second.counts.is_empty());
}

#[test]
fn reconstruction_identity_does_not_promote_incomplete_topology() {
    let path = test_util::temp::temp_path("incomplete-reconstruction.h5");
    let mut fixture = Fixture::new();
    let child = fixture.append_header(vec![]);
    fixture.link("child", child);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    phase.budget = TraversalBudget::new(1, usize::MAX);
    let object = ObjectId::new(PhysicalAddress::new(child));
    let before = phase.reconstruction_source(&engine, object).unwrap();
    assert!(phase.topology().is_none());
    assert_eq!(
        before,
        phase.reconstruction_source(&engine, object).unwrap()
    );
    assert!(phase.count_evidence(object).unwrap().is_none());
    assert!(phase.counts.is_empty());
}

#[test]
fn a_foreign_source_is_rejected_before_an_origin_is_allocated() {
    let first_path = test_util::temp::temp_path("first-origin-source.h5");
    let second_path = test_util::temp::temp_path("second-origin-source.h5");
    Fixture::new().write(&first_path);
    std::fs::copy(&first_path, &second_path).unwrap();
    let first = edit_test_util::open_engine(&first_path);
    let second = edit_test_util::open_engine(&second_path);
    let mut phase = AnalysisPhase::new(&first);
    let error = phase
        .reconstruction_source(&second, ObjectId::new(PhysicalAddress::new(Fixture::ROOT)))
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal("a reconstructed header belongs to another source engine".into())
    );
    assert!(phase.identity.is_none());
    assert!(phase.topology.is_none());
    assert!(phase.counts.is_empty());
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn an_unused_analysis_phase_allocates_no_reconstruction_identity() {
    let path = test_util::temp::temp_path("empty-reconstruction.h5");
    Fixture::new().write(&path);
    let engine = edit_test_util::open_engine(&path);
    let (mut phase, empty) =
        allocation::measure("empty_reconstruction_phase", || AnalysisPhase::new(&engine));
    let object = ObjectId::new(PhysicalAddress::new(Fixture::ROOT));
    let (first, control) = allocation::measure("first_reconstruction_identity", || {
        phase.reconstruction_source(&engine, object).unwrap()
    });
    assert!(control.blocks >= 1 && control.bytes > 0, "{control}");
    assert_eq!(empty.blocks, 0, "{empty}");
    assert_eq!(empty.bytes, 0, "{empty}");
    let (repeated, cached) = allocation::measure("repeated_reconstruction_identity", || {
        phase.reconstruction_source(&engine, object).unwrap()
    });
    assert_eq!(cached.blocks, 0, "{cached}");
    assert_eq!(cached.bytes, 0, "{cached}");
    assert_eq!(first, repeated);
    assert!(phase.topology.is_none());
}

#[derive(Clone, Copy)]
enum Relation {
    Matching,
    Object,
    Phase,
    Source,
}
