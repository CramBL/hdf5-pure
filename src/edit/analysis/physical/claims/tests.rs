use core::cell::Cell;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{ClaimOrigin, ClaimProblem};
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::{
    AnalysisPhase, COLLECTION_ALLOCATIONS, COLLECTION_WORK, DISCOVERY_WORK,
};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;

#[rstest]
#[case::cold(false)]
#[case::enriched(true)]
fn physical_claim_queries_reuse_caches_and_parent_identities(#[case] enriched: bool) {
    let path = test_util::temp::temp_path("physical-claim-identities.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"one"), Object::new(7, b"seven")]);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(3, collection, 1),
    );
    let first = fixture.append_header(records);
    fixture.link("a", first);
    fixture.link("alias", first);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 7),
    );
    let second = fixture.append_header(records);
    fixture.link("b", second);
    fixture.write(&path);
    let original = std::fs::read(&path).unwrap();
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let root = id(Fixture::ROOT);
    let source = phase.reconstruction_source(&engine, root).unwrap();
    calls.store(0, Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    COLLECTION_ALLOCATIONS.with(|count| count.set(0));
    assert!(phase.claims.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    if enriched {
        let references = phase.collections_reachable_from([root]).unwrap();
        assert!(references.analysis_complete);
        assert_eq!(
            references.referenced_collections,
            BTreeSet::from([collection])
        );
        phase
            .heap_members(&source, PhysicalAddress::new(collection))
            .unwrap();
        phase.object_ownership(&source).unwrap();
    }
    let claims = phase.physical_claims(&source).unwrap();
    assert_eq!(claims.root(), Some(root));
    assert!(claims.matches_phase(&source));
    assert_eq!(
        (claims.physical_size(), claims.logical_eoa()),
        (original.len() as u64, original.len() as u64)
    );
    assert!(claims.problems().is_empty(), "{:?}", claims.problems());
    let headers = claims
        .claims()
        .iter()
        .filter_map(|claim| match claim.origin() {
            ClaimOrigin::Object {
                object,
                kind: AllocationKind::Header,
            } => Some(object),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(headers, [root, id(first), id(second)]);
    let heap = claims
        .claims()
        .iter()
        .filter(|claim| matches!(claim.origin(), ClaimOrigin::Heap(_)))
        .map(|claim| (claim.range().start().get(), claim.range().length()))
        .collect::<Vec<_>>();
    assert_eq!(
        heap,
        [(
            collection,
            u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap()
        )]
    );
    let pointer = claims.claims().as_ptr();
    let before = calls.load(Ordering::Relaxed);
    assert!(before > 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 3));
    assert_eq!(COLLECTION_WORK.with(Cell::get).1, 3);
    assert_eq!(COLLECTION_ALLOCATIONS.with(Cell::get), 1);
    assert_eq!(phase.ownership.len(), 3);
    assert_eq!(
        phase.physical_claims(&source).unwrap().claims().as_ptr(),
        pointer
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 3));
    let mut other = AnalysisPhase::new(&engine);
    let other_source = other.reconstruction_source(&engine, root).unwrap();
    let error = phase.physical_claims(&other_source).err().unwrap();
    let Error::Format(error) = error else {
        panic!("expected a phase-association error: {error:?}");
    };
    assert_eq!(
        error,
        FormatError::Internal("physical claims belong to another source phase".into())
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    drop(other);
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[rstest]
#[case::unsupported(Input::Unsupported)]
#[case::collection(Input::Collection)]
#[case::topology(Input::Topology)]
#[case::file(Input::File)]
fn partial_claim_collection_preserves_independent_ranges(#[case] input: Input) {
    let path = test_util::temp::temp_path("physical-claim-partial.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"one")]);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(3, collection, 1),
    );
    if matches!(input, Input::Unsupported) {
        records.push(Message::new(MessageType(0x7f), &[]));
    }
    let owner = fixture.append_header(records);
    fixture.link("a", owner);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    if matches!(input, Input::File) {
        engine.write_at(0, &[0]).unwrap();
    }
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    if matches!(input, Input::Topology) {
        phase.topology = Some(None);
    }
    if matches!(input, Input::Collection) {
        phase.allocations.insert(
            PhysicalAddress::new(collection),
            Err(FormatError::UnexpectedEof {
                expected: 16,
                available: 8,
            }),
        );
    }
    let claims = phase.physical_claims(&source).unwrap();
    assert_eq!(claims.root(), Some(id(Fixture::ROOT)));
    assert!(!claims.claims().is_empty());
    match input {
        Input::Unsupported => {
            assert!(
                matches!(claims.problems(), [ClaimProblem::ObjectStorage(object), ClaimProblem::References(reference)] if *object == id(owner) && *reference == id(owner)),
                "{:?}",
                claims.problems()
            );
            assert!(
                claims
                    .claims()
                    .iter()
                    .any(|claim| claim.origin()
                        == ClaimOrigin::Heap(PhysicalAddress::new(collection)))
            );
        }
        Input::Collection => {
            assert!(
                matches!(claims.problems(), [ClaimProblem::Collection { address, error: FormatError::UnexpectedEof { expected: 16, available: 8 } }] if *address == PhysicalAddress::new(collection)),
                "{:?}",
                claims.problems()
            );
            assert!(claims.claims().iter().any(|claim| matches!(claim.origin(), ClaimOrigin::Object { object, .. } if object == id(owner))));
        }
        Input::Topology => {
            assert!(
                matches!(claims.problems(), [ClaimProblem::Topology]),
                "{:?}",
                claims.problems()
            );
            assert!(
                claims
                    .claims()
                    .iter()
                    .all(|claim| matches!(claim.origin(), ClaimOrigin::File { .. }))
            );
        }
        Input::File => {
            assert!(
                matches!(claims.problems(), [ClaimProblem::File(0)]),
                "{:?}",
                claims.problems()
            );
            assert!(claims.claims().iter().any(|claim| matches!(claim.origin(), ClaimOrigin::Object { object, .. } if object == id(owner))));
        }
    }
    let problems = claims.problems().len();
    let before = phase.claims.as_ref().unwrap().claims().as_ptr();
    phase.physical_coverage(&source).unwrap();
    phase.residual_content(&source).unwrap();
    let claims = phase.physical_claims(&source).unwrap();
    assert_eq!(claims.problems().len(), problems);
    assert_eq!(claims.claims().as_ptr(), before);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn numeric_claim_queries_bound_source_reads_and_allocations() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use test_util::allocation;

    let path = test_util::temp::temp_path("physical-claim-numeric.h5");
    let mut fixture = Fixture::new();
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::type_builders::make_i32_type(),
        &vec![0; 1_048_576],
    );
    let owner = fixture.append_header(records);
    fixture.link("data", owner);
    fixture.write(&path);
    let read_bytes = Arc::new(AtomicU64::new(0));
    let (engine, calls) =
        edit_test_util::open_counted_engine_with_bytes(&path, Arc::clone(&read_bytes));
    calls.store(0, Ordering::Relaxed);
    read_bytes.store(0, Ordering::Relaxed);
    let (mut phase, empty) =
        allocation::measure("physical_claims_empty", || AnalysisPhase::new(&engine));
    assert_eq!(
        (empty.blocks, empty.bytes, calls.load(Ordering::Relaxed)),
        (0, 0, 0)
    );
    let source = phase
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    let ((), measured) = allocation::measure("physical_claims_numeric", || {
        let claims = phase.physical_claims(&source).unwrap();
        assert!(claims.problems().is_empty(), "{:?}", claims.problems());
    });
    let claims = phase.physical_claims(&source).unwrap();
    assert_eq!(
        claims
            .claims()
            .iter()
            .filter(|claim| matches!(
                claim.origin(),
                ClaimOrigin::Object {
                    kind: AllocationKind::Data,
                    ..
                }
            ))
            .map(|claim| claim.range().length())
            .collect::<Vec<_>>(),
        [1_048_576]
    );
    assert!(
        measured.blocks > 0
            && measured.bytes > 0
            && measured.bytes < 65_536
            && measured.peak_bytes < 65_536
            && measured.live_bytes < 65_536,
        "{measured}"
    );
    let coverage = phase.physical_coverage(&source).unwrap();
    assert_eq!(coverage.conflict(), None);
    assert!(coverage.beyond_source().is_empty());
    assert!(coverage.beyond_eoa().is_empty());
    let before = (
        calls.load(Ordering::Relaxed),
        read_bytes.load(Ordering::Relaxed),
    );
    assert!(
        before.0 > 0 && before.0 < 64 && before.1 > 0 && before.1 < 8_192,
        "{before:?}"
    );
    let ((), cached) = allocation::measure("physical_claims_cached", || {
        assert_eq!(phase.physical_coverage(&source).unwrap().conflict(), None);
    });
    assert_eq!((cached.blocks, cached.bytes), (0, 0));
    assert_eq!(
        (
            calls.load(Ordering::Relaxed),
            read_bytes.load(Ordering::Relaxed)
        ),
        before
    );
    let (payload, positive) =
        allocation::measure("physical_claims_payload_control", || vec![0_u8; 1_048_576]);
    assert_eq!(payload.len(), 1_048_576);
    assert!(positive.bytes >= 1_048_576);
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

#[derive(Clone, Copy)]
enum Input {
    Collection,
    File,
    Topology,
    Unsupported,
}
