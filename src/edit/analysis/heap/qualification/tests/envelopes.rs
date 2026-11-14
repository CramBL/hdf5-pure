use core::num::NonZeroU8;
use std::path::Path;
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::global_heap::{self, MIN_COLLECTION_SIZE, Object};
use test_util::object_header::v2::Header;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::super::ParentUnavailable;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::analysis::physical::claims::coverage::content::ResidualOutcome;
use crate::edit::analysis::physical::claims::domain::{DomainUnavailable, OwnerDomain};
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

#[rstest]
#[case::records(Damage::Records)]
#[case::version(Damage::Version)]
#[case::unknown(Damage::Unknown)]
#[case::overlap(Damage::Overlap)]
#[case::source_bounds(Damage::BeyondSource)]
fn collection_envelope_proof_preserves_independent_restrictions(#[case] damage: Damage) {
    let path = test_util::temp::temp_path("collection-envelope.h5");
    let (healthy, damaged, gone, opaque) = fixture(&path, damage);
    let (engine, reads) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let error = phase
        .heap_members(&source, PhysicalAddress::new(damaged))
        .err()
        .unwrap();
    assert_eq!(error, damage.expected_error(damaged, engine.image().len()));
    reads.store(0, Ordering::Relaxed);
    assert_eq!(
        phase
            .heap_members(&source, PhysicalAddress::new(damaged))
            .err()
            .unwrap(),
        error
    );
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    let candidates = phase.deleted_collections(&[gone]);
    let facts = phase.qualify_collections(&source, candidates).unwrap();
    let Some(OwnerDomain::Unavailable(DomainUnavailable::Collection(address))) = facts.domain()
    else {
        panic!("expected strict collection restriction");
    };
    assert_eq!(*address, PhysicalAddress::new(damaged));
    let restriction = phase.heap_domain_restriction(&source).unwrap();
    match (damage, restriction) {
        (Damage::Records, None) => {
            facts.whole_parent(PhysicalAddress::new(healthy)).unwrap();
            let proof = facts.residual_groups().unwrap();
            assert_eq!(proof.source(), &source);
            assert!(proof.session().same(&engine.context.allocations.session()));
        }
        (Damage::Version, Some(DomainUnavailable::Collection(address))) => {
            assert_eq!(address, PhysicalAddress::new(damaged));
        }
        (Damage::Unknown, Some(DomainUnavailable::Residual(index))) => {
            let content = phase.residual_content(&source).unwrap();
            let observed = content.intervals().get(index).unwrap();
            let opaque = opaque.unwrap();
            assert_eq!(observed.range(), opaque);
            assert_eq!(
                observed.outcome(),
                &ResidualOutcome::NonZero {
                    address: opaque.start(),
                    value: NonZeroU8::new(OPAQUE_BYTES[0]).unwrap(),
                }
            );
        }
        (Damage::Overlap, Some(DomainUnavailable::Conflict(pair))) => {
            let claims = phase.physical_claims(&source).unwrap();
            let first = claims.claims()[pair[0]];
            let second = claims.claims()[pair[1]];
            assert!(first.range().overlaps(second.range()));
            assert_eq!(
                first.origin(),
                ClaimOrigin::Heap(PhysicalAddress::new(damaged))
            );
            assert_eq!(
                second.origin(),
                ClaimOrigin::Heap(PhysicalAddress::new(healthy))
            );
        }
        (Damage::BeyondSource, Some(DomainUnavailable::BeyondSource(index))) => {
            let claims = phase.physical_claims(&source).unwrap();
            let claim = claims.claims()[index];
            assert_eq!(
                claim.origin(),
                ClaimOrigin::Heap(PhysicalAddress::new(damaged))
            );
            assert_eq!(
                claim.range().length(),
                u64::try_from(OVERSIZED_COLLECTION).unwrap()
            );
            assert!(claim.range().end().get() > claims.physical_size());
        }
        (_, restriction) => panic!("unexpected domain restriction: {restriction:?}"),
    }
    if !matches!(damage, Damage::Records) {
        assert_eq!(
            facts.whole_parent(PhysicalAddress::new(healthy)),
            Err(ParentUnavailable::Domain)
        );
        assert!(facts.residual_groups().is_none());
    }
}

/// Writes string owners of healthy and damaged heap collections with a numeric keeper.
///
/// `damage` selects the collection damage and any additional source restriction. Returns the
/// physical healthy-parent address, damaged-parent address, `gone` object identity, and optional
/// opaque residual range, in that order.
///
/// # Panics
///
/// Panics if construction, address conversion, or writing the fixture fails.
pub(in crate::edit) fn fixture(
    path: &Path,
    damage: Damage,
) -> (u64, u64, ObjectId, Option<FileRange>) {
    let mut fixture = Fixture::new();
    let mut damaged =
        fixture.collection(&[Object::new(7, b"before"), Object::new(7, b"duplicate")]);
    let healthy = fixture.collection(&[Object::new(7, b"removed")]);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, damaged, 7),
    );
    let bad_header = fixture.append_header(records);
    fixture.link("bad", bad_header);
    let bad_header_length = u64::try_from(fixture.image.as_bytes().len()).unwrap() - bad_header;
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, healthy, 7),
    );
    let gone = fixture.append_header(records);
    fixture.link("gone", gone);
    let records =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &17_i32.to_le_bytes());
    fixture.object("keep", records);
    if matches!(damage, Damage::BeyondSource) {
        fixture.image.place(
            usize::try_from(damaged).unwrap(),
            &vec![0; MIN_COLLECTION_SIZE],
        );
        damaged = u64::try_from(fixture.image.append(&global_heap::collection(
            &[Object::new(7, b"before"), Object::new(7, b"duplicate")],
            Widths::EIGHT,
        )))
        .unwrap();
        let records = Form::CompactDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(6, damaged, 7),
        );
        let header = Header::new().messages(records).build();
        assert_eq!(u64::try_from(header.len()).unwrap(), bad_header_length);
        fixture
            .image
            .place(usize::try_from(bad_header).unwrap(), &header);
    }
    let start = usize::try_from(damaged).unwrap();
    let mut opaque = None;
    match damage {
        Damage::Version => fixture
            .image
            .place(start + global_heap::VERSION_OFFSET, &[2]),
        Damage::Overlap => fixture.image.place(
            start + global_heap::SIZE_OFFSET,
            &u64::try_from(2 * MIN_COLLECTION_SIZE)
                .unwrap()
                .to_le_bytes(),
        ),
        Damage::BeyondSource => fixture.image.place(
            start + global_heap::SIZE_OFFSET,
            &u64::try_from(OVERSIZED_COLLECTION).unwrap().to_le_bytes(),
        ),
        Damage::Unknown => {
            let start = fixture.image.append(&OPAQUE_BYTES);
            opaque = Some(
                FileRange::new(
                    PhysicalAddress::new(u64::try_from(start).unwrap()),
                    u64::try_from(OPAQUE_BYTES.len()).unwrap(),
                )
                .unwrap(),
            );
        }
        Damage::Records => {}
    }
    fixture.write(path);
    (
        healthy,
        damaged,
        ObjectId::new(PhysicalAddress::new(gone)),
        opaque,
    )
}

#[derive(Clone, Copy, Debug)]
pub(in crate::edit) enum Damage {
    BeyondSource,
    Overlap,
    Records,
    Unknown,
    Version,
}

impl Damage {
    /// Returns the collection framing or extent error expected for this fixture.
    ///
    /// The caller passes the damaged-parent address and current source length.
    ///
    /// # Panics
    ///
    /// Panics if the oversized allocation's end overflows.
    pub(in crate::edit) fn expected_error(self, address: u64, source_len: u64) -> FormatError {
        match self {
            Damage::Version => FormatError::InvalidGlobalHeapVersion(2),
            Damage::BeyondSource => FormatError::UnexpectedSourceEof {
                expected: address
                    .checked_add(u64::try_from(OVERSIZED_COLLECTION).unwrap())
                    .unwrap(),
                available: source_len,
            },
            _ => FormatError::DuplicateGlobalHeapObjectIndex {
                collection_address: address,
                index: 7,
            },
        }
    }
}

/// Opaque residual bytes appended for [`Damage::Unknown`].
pub(in crate::edit) const OPAQUE_BYTES: [u8; 17] = [0x5a; 17];

/// Declared allocation length used for source-extent rejection.
const OVERSIZED_COLLECTION: usize = 16 * MIN_COLLECTION_SIZE;
