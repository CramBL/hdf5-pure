use std::sync::Arc;
use std::sync::Mutex;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{DomainUnavailable, OwnerDomain};
use crate::edit::WriteEngine;
use crate::edit::analysis::physical::claims::{ClaimOrigin, ClaimProblem, PhysicalClaim};
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use crate::{FileBuilder, UserblockSize};

#[rstest]
#[case::plain(0)]
#[case::userblock(512)]
fn owner_domain_reuses_source_facts_without_reading_numeric_storage(#[case] base: u64) {
    let path = test_util::temp::temp_path("owner-domain-numeric.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder
        .create_dataset("data")
        .with_u8_data(&vec![0x5a; 1_048_576]);
    builder.write(&path).unwrap();
    let reads = Arc::new(Mutex::new(Vec::new()));
    let engine = edit_test_util::open_observed_engine(&path, Arc::clone(&reads));
    let mut phase = AnalysisPhase::new(&engine);
    let source = root_source(&mut phase, &engine);
    phase.residual_content(&source).unwrap();
    reads.lock().unwrap().clear();
    let OwnerDomain::Available(proof) = phase.supported_owner_domain(&source).unwrap() else {
        panic!("expected a supported owner domain");
    };
    assert!(proof.matches(&engine, phase.physical_claims(&source).unwrap()));
    assert!(reads.lock().unwrap().is_empty());
    let mut other = AnalysisPhase::new(&engine);
    let other_source = root_source(&mut other, &engine);
    assert!(!proof.matches(&engine, other.physical_claims(&other_source).unwrap()));
    let error = phase.supported_owner_domain(&other_source).err().unwrap();
    assert!(
        matches!(error, Error::Format(FormatError::Internal(ref message)) if message == "physical claims belong to another source phase"),
        "{error:?}"
    );
}

#[rstest]
#[case::topology(Failure::Topology)]
#[case::references(Failure::References)]
#[case::storage(Failure::Storage)]
#[case::file(Failure::File)]
#[case::collection(Failure::Collection)]
#[case::conflict(Failure::Conflict)]
#[case::authority(Failure::Authority)]
#[case::physical_bound(Failure::PhysicalBound)]
#[case::logical_bound(Failure::LogicalBound)]
#[case::missing_source(Failure::MissingSource)]
#[case::prospective_root(Failure::Root)]
fn owner_domain_preserves_independent_incomplete_outcomes(#[case] failure: Failure) {
    let path = test_util::temp::temp_path("owner-domain-incomplete.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"one")]);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(3, collection, 1),
    );
    let owner = fixture.append_header(records);
    fixture.link("data", owner);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let object = ObjectId::new(PhysicalAddress::new(owner));
    let mut phase = if matches!(failure, Failure::Root) {
        AnalysisPhase::prospective(&engine, object.header_address())
    } else {
        AnalysisPhase::new(&engine)
    };
    let source = root_source(&mut phase, &engine);
    phase.physical_claims(&source).unwrap();
    let claims = phase.claims.as_mut().unwrap();
    let original = claims.claims.clone();
    match failure {
        Failure::Topology => claims.problems.push(ClaimProblem::Topology),
        Failure::References => claims.problems.push(ClaimProblem::References(object)),
        Failure::Storage => claims.problems.push(ClaimProblem::ObjectStorage(object)),
        Failure::File => claims.problems.push(ClaimProblem::File(0)),
        Failure::Collection => claims.problems.push(ClaimProblem::Collection {
            address: PhysicalAddress::new(collection),
            error: FormatError::Source("injected collection framing failure".into()),
        }),
        Failure::Conflict => claims.claims.push(original[0]),
        Failure::Authority => claims.claims.push(PhysicalClaim {
            range: original[0].range(),
            origin: ClaimOrigin::Heap(PhysicalAddress::new(collection)),
        }),
        Failure::PhysicalBound => claims.physical_size = 1,
        Failure::LogicalBound => claims.logical_eoa = 1,
        Failure::MissingSource => claims.logical_eoa += 1,
        Failure::Root => {}
    }
    let OwnerDomain::Unavailable(reason) = phase.supported_owner_domain(&source).unwrap() else {
        panic!("expected unavailable owner domain");
    };
    match (&failure, &reason) {
        (Failure::Topology, DomainUnavailable::Topology)
        | (Failure::MissingSource, DomainUnavailable::MissingSource)
        | (Failure::Root, DomainUnavailable::Root) => {}
        (Failure::References, DomainUnavailable::References(found))
        | (Failure::Storage, DomainUnavailable::ObjectStorage(found)) => assert_eq!(*found, object),
        (Failure::File, DomainUnavailable::File(index)) => assert_eq!(*index, 0),
        (Failure::Collection, DomainUnavailable::Collection(address)) => {
            assert_eq!(address.get(), collection)
        }
        (Failure::Authority, DomainUnavailable::Authority(index)) => {
            assert_eq!(*index, original.len())
        }
        (Failure::Conflict, DomainUnavailable::Conflict(pair)) => {
            assert_eq!(*pair, [0, original.len()])
        }
        (Failure::PhysicalBound, DomainUnavailable::BeyondSource(index))
        | (Failure::LogicalBound, DomainUnavailable::BeyondEoa(index)) => assert_eq!(*index, 0),
        _ => panic!("unexpected owner-domain outcome: {reason:?}"),
    }
    assert_eq!(
        &phase.claims.as_ref().unwrap().claims[..original.len()],
        original
    );
}

#[rstest]
#[case::provisional(false)]
#[case::uncertain(true)]
fn owner_domain_requires_settled_reference_publication(#[case] uncertain: bool) {
    let path = test_util::temp::temp_path("owner-domain-publication.h5");
    FileBuilder::new().write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .context
        .allocations
        .provision(FileRange::new(PhysicalAddress::new(0), 1).unwrap())
        .unwrap();
    if uncertain {
        engine.context.allocations.make_uncertain();
    }
    let mut phase = AnalysisPhase::new(&engine);
    let source = root_source(&mut phase, &engine);
    let OwnerDomain::Unavailable(DomainUnavailable::Publication(error)) =
        phase.supported_owner_domain(&source).unwrap()
    else {
        panic!("expected unavailable allocation publication");
    };
    if uncertain {
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(error, Error::Format(FormatError::Internal(ref message)) if message == "reference publication precedes allocation ownership publication"),
            "{error:?}"
        );
    }
}

#[rstest]
#[case::unknown_header(true)]
#[case::nonzero_residual(false)]
fn owner_domain_keeps_unsupported_or_unexplained_source_unavailable(#[case] unsupported: bool) {
    let path = test_util::temp::temp_path("owner-domain-residual.h5");
    let mut fixture = Fixture::new();
    let owner = unsupported.then(|| {
        let owner = fixture.append_header(vec![Message::new(MessageType(0x7f), &[])]);
        fixture.link("owner", owner);
        ObjectId::new(PhysicalAddress::new(owner))
    });
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = root_source(&mut phase, &engine);
    if !unsupported {
        let range = phase.physical_coverage(&source).unwrap().residuals()[0];
        drop(phase);
        engine.image.write_at(range.start().get(), &[1]).unwrap();
        phase = AnalysisPhase::new(&engine);
    }
    let source = root_source(&mut phase, &engine);
    let OwnerDomain::Unavailable(reason) = phase.supported_owner_domain(&source).unwrap() else {
        panic!("expected unavailable owner domain");
    };
    if let Some(owner) = owner {
        let DomainUnavailable::ObjectStorage(object) = reason else {
            panic!("expected unsupported object storage: {reason:?}");
        };
        assert_eq!(object, owner);
    } else {
        assert!(
            matches!(reason, DomainUnavailable::Residual(0)),
            "{reason:?}"
        );
    }
}

#[test]
fn owner_domain_requires_new_evidence_after_source_changes_and_allocator_restore() {
    let path = test_util::temp::temp_path("owner-domain-changed.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("data").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = root_source(&mut phase, &engine);
    let OwnerDomain::Available(proof) = phase.supported_owner_domain(&source).unwrap() else {
        panic!("expected initial owner domain");
    };
    let range = phase.physical_coverage(&source).unwrap().residuals()[0];
    drop(phase);
    let snapshot = engine.snapshot_free();
    engine.image.write_at(range.start().get(), &[1]).unwrap();
    engine.restore_free(snapshot);
    let mut changed = AnalysisPhase::new(&engine);
    let fresh = root_source(&mut changed, &engine);
    assert!(!proof.matches(&engine, changed.physical_claims(&fresh).unwrap()));
    let error = changed.supported_owner_domain(&source).err().unwrap();
    assert!(
        matches!(error, Error::Format(FormatError::Internal(ref message)) if message == "physical claims belong to another source phase"),
        "{error:?}"
    );
    let OwnerDomain::Unavailable(DomainUnavailable::Residual(index)) =
        changed.supported_owner_domain(&fresh).unwrap()
    else {
        panic!("expected unavailable changed residual");
    };
    assert_eq!(index, 0);
    drop(changed);
    engine.image.write_at(range.start().get(), &[0]).unwrap();
    let mut restored = AnalysisPhase::new(&engine);
    let fresh = root_source(&mut restored, &engine);
    let OwnerDomain::Available(current) = restored.supported_owner_domain(&fresh).unwrap() else {
        panic!("expected restored source qualification");
    };
    let claims = restored.physical_claims(&fresh).unwrap();
    assert!(current.matches(&engine, claims));
    assert!(!proof.matches(&engine, claims));
}

fn root_source(phase: &mut AnalysisPhase<'_>, engine: &WriteEngine) -> ReconstructionSource {
    phase
        .reconstruction_source(
            engine,
            ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap(),
        )
        .unwrap()
}

#[derive(Clone, Copy)]
enum Failure {
    Authority,
    Collection,
    Conflict,
    File,
    LogicalBound,
    MissingSource,
    PhysicalBound,
    References,
    Root,
    Storage,
    Topology,
}
