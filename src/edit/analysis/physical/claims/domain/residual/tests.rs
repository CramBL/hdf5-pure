use core::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::MetadataSource;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::v1;
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{
    MAX_METADATA_BYTES, MAX_RESIDUAL_BYTES, MAX_RESIDUAL_GROUPS, ResidualGroups, ResidualLimit,
    ResidualSource, ResidualUnavailable,
};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::physical::claims::domain::{DomainUnavailable, OwnerDomain};
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;

#[rstest]
#[case::numeric_child(Extra::Keep)]
#[case::removed_child(Extra::Gone)]
#[case::byte_boundary(Extra::ByteBoundary)]
#[case::group_boundary(Extra::GroupBoundary)]
fn residual_group_proof_retains_known_links_and_reuses_source_observations(#[case] extra: Extra) {
    let path = test_util::temp::temp_path("residual-group-cache.h5");
    let (_, group, _) = fixture(&path, extra);
    let (engine, reads) = edit_test_util::open_counted_engine(&path);
    let (mut phase, source) = phase(&engine);
    assert!(matches!(
        phase.supported_owner_domain(&source).unwrap(),
        OwnerDomain::Unavailable(DomainUnavailable::Residual(_))
    ));
    reads.store(0, Ordering::Relaxed);
    let first = phase.residual_group_domain(&source).unwrap();
    let groups = first.as_ref().as_ref().unwrap();
    assert!(
        groups
            .roots()
            .contains(&ObjectId::new(PhysicalAddress::new(group)))
    );
    assert_eq!(
        groups.roots().len(),
        if matches!(extra, Extra::GroupBoundary) {
            MAX_RESIDUAL_GROUPS
        } else {
            1
        }
    );
    assert_eq!(groups.source(), &source);
    assert!(groups.session().same(&engine.context.allocations.session()));
    let observed = reads.load(Ordering::Relaxed);
    assert!(observed > 0);
    let second = phase.residual_group_domain(&source).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(reads.load(Ordering::Relaxed), observed);
}

#[rstest]
#[case::unknown_bytes(Extra::Unknown)]
#[case::unsupported_header(Extra::Unsupported)]
#[case::malformed_header(Extra::Malformed)]
#[case::legacy_storage(Extra::Legacy)]
#[case::unresolved_link(Extra::Unresolved)]
#[case::byte_limit(Extra::Limit)]
#[case::group_limit(Extra::GroupLimit)]
fn residual_group_qualification_withholds_unknown_or_incomplete_forms(#[case] extra: Extra) {
    let path = test_util::temp::temp_path("residual-group-unavailable.h5");
    let (_, group, _) = fixture(&path, extra);
    let engine = edit_test_util::open_engine(&path);
    let (mut phase, source) = phase(&engine);
    let result = phase.residual_group_domain(&source).unwrap();
    let error = match result.as_ref() {
        Ok(_) => panic!("expected unavailable residual group proof"),
        Err(error) => error,
    };
    match (extra, error) {
        (
            Extra::Limit,
            ResidualUnavailable::Limit {
                kind: ResidualLimit::Bytes,
                required,
                maximum,
            },
        ) => {
            assert_eq!(
                (*required, *maximum),
                (
                    u64::try_from(MAX_RESIDUAL_BYTES + 1).unwrap(),
                    u64::try_from(MAX_RESIDUAL_BYTES).unwrap()
                )
            );
        }
        (
            Extra::GroupLimit,
            ResidualUnavailable::Limit {
                kind: ResidualLimit::Groups,
                required,
                maximum,
            },
        ) => {
            assert_eq!(
                (*required, *maximum),
                (
                    u64::try_from(MAX_RESIDUAL_GROUPS + 1).unwrap(),
                    u64::try_from(MAX_RESIDUAL_GROUPS).unwrap()
                )
            );
        }
        (Extra::Legacy | Extra::Unsupported, ResidualUnavailable::Unsupported(object)) => {
            assert_eq!(*object, ObjectId::new(PhysicalAddress::new(group)))
        }
        (Extra::Unresolved, ResidualUnavailable::Link(object)) => {
            assert_eq!(*object, ObjectId::new(PhysicalAddress::new(group)))
        }
        (
            Extra::Malformed,
            ResidualUnavailable::Format(FormatError::InvalidObjectHeaderVersion(version)),
        ) => assert_eq!(*version, 3),
        (
            Extra::Unknown,
            ResidualUnavailable::Format(FormatError::InvalidObjectHeaderVersion(version)),
        ) => assert_eq!(*version, 0x5a),
        _ => panic!("unexpected residual outcome: {error:?}"),
    }
}

#[rstest]
#[case::phase(Association::Phase)]
#[case::session(Association::Session)]
fn residual_group_cache_rejects_incompatible_associations(#[case] association: Association) {
    let path = test_util::temp::temp_path("residual-group-association.h5");
    fixture(&path, Extra::Keep);
    let engine = edit_test_util::open_engine(&path);
    let (mut first, source) = phase(&engine);
    let proof = first.residual_group_domain(&source).unwrap();
    match association {
        Association::Phase => {
            let (mut other, source) = phase(&engine);
            other.physical_claims(&source).unwrap();
            other.claims.as_mut().unwrap().residual_groups = Some(proof);
            assert_association_error(other.residual_group_domain(&source));
        }
        Association::Session => {
            let old = proof.as_ref().as_ref().unwrap();
            let changed = ResidualGroups {
                source: old.source.clone(),
                session: AllocationPublication::new().session(),
                physical_size: old.physical_size,
                logical_eoa: old.logical_eoa,
                roots: old.roots.clone(),
                objects: old.objects.clone(),
            };
            first.claims.as_mut().unwrap().residual_groups = Some(Arc::new(Ok(changed)));
            assert_association_error(first.residual_group_domain(&source));
        }
    }
}

#[test]
fn residual_group_metadata_preserves_physical_bounds_and_read_budget() {
    let start = PhysicalAddress::new(u64::from(u32::MAX) + 1);
    let range = FileRange::new(start, 4).unwrap();
    let budget = Cell::new(MAX_METADATA_BYTES);
    let source = ResidualSource {
        region: FileRegion::new(range, &[1, 2, 3, 4]).unwrap(),
        remaining: &budget,
    };
    assert_eq!(source.read_metadata(start.get() + 1, 2).unwrap(), [2, 3]);
    assert_eq!(source.read_metadata_at(start.get() + 3, 1).unwrap(), [4]);
    assert_eq!(budget.get(), MAX_METADATA_BYTES - 3);
    let error = source.read_metadata(start.get() - 1, 1).unwrap_err();
    assert_eq!(
        error,
        FormatError::InsufficientRegion {
            required_start: start.get() - 1,
            required_length: 1,
            provided_start: start.get(),
            provided_length: 4
        }
    );
    budget.set(2);
    let error = source.read_metadata(start.get(), 4).unwrap_err();
    assert_eq!(
        error,
        FormatError::ResourceLimit {
            limit: u64::try_from(MAX_METADATA_BYTES).unwrap(),
            required: u64::try_from(MAX_METADATA_BYTES + 2).unwrap()
        }
    );
    assert_eq!(budget.get(), 2);
}

fn assert_association_error(
    result: Result<Arc<Result<ResidualGroups, ResidualUnavailable>>, Error>,
) {
    let error = match result {
        Ok(_) => panic!("expected incompatible residual source"),
        Err(error) => error,
    };
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("unexpected association error: {error:?}");
    };
    assert_eq!(
        reason,
        "residual groups belong to another source phase or session"
    );
}

fn phase(engine: &WriteEngine) -> (AnalysisPhase<'_>, ReconstructionSource) {
    let mut phase = AnalysisPhase::new(engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(engine, root).unwrap();
    (phase, source)
}

/// Writes linked string and numeric datasets with the residual storage selected by `extra`.
///
/// Returns the physical heap-parent address for `gone`, the residual start address, and the
/// complete `keep` object-header range, in that order.
///
/// # Panics
///
/// Panics if construction, address conversion, or writing the fixture fails.
pub(in crate::edit) fn fixture(path: &Path, extra: Extra) -> (u64, u64, FileRange) {
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed")]);
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    );
    let gone = fixture.append_header(records);
    fixture.link("gone", gone);
    let records =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &17_i32.to_le_bytes());
    let keep = fixture.append_header(records);
    let keeper = FileRange::new(
        PhysicalAddress::new(keep),
        u64::try_from(fixture.image.as_bytes().len()).unwrap() - keep,
    )
    .unwrap();
    fixture.link("keep", keep);
    let mut records = vec![
        Message::new(
            MessageType::LINK_INFO,
            &format::LinkInfoMessage {
                max_creation_order: None,
                fractal_heap_address: None,
                btree_name_index_address: None,
                btree_creation_order_address: None,
            }
            .serialize(format::OffsetWidth::Eight)
            .unwrap(),
        ),
        global_heap_fixture::hard_link(
            "child",
            match extra {
                Extra::Gone => gone,
                Extra::Unresolved => u64::MAX - 1,
                _ => keep,
            },
        ),
    ];
    if matches!(extra, Extra::Unsupported) {
        records.push(Message::new(MessageType(0x7f), &[]));
    }
    let mut bytes = if matches!(extra, Extra::Legacy) {
        v1::Header::new()
            .messages(vec![Message::new(MessageType::SYMBOL_TABLE, &[0; 16])])
            .build()
    } else {
        Header::new().messages(records).build()
    };
    match extra {
        Extra::Unknown => bytes.fill(0x5a),
        Extra::Malformed => bytes[4] = 3,
        Extra::Limit => bytes.resize(MAX_RESIDUAL_BYTES + 1, 0),
        Extra::ByteBoundary => bytes.resize(MAX_RESIDUAL_BYTES, 0),
        Extra::GroupBoundary => bytes = bytes.repeat(MAX_RESIDUAL_GROUPS),
        Extra::GroupLimit => bytes = bytes.repeat(MAX_RESIDUAL_GROUPS + 1),
        _ => {}
    }
    let group = u64::try_from(fixture.image.append(&bytes)).unwrap();
    fixture.write(path);
    (parent, group, keeper)
}

#[derive(Clone, Copy, Debug)]
pub(in crate::edit) enum Extra {
    ByteBoundary,
    Gone,
    GroupBoundary,
    GroupLimit,
    Keep,
    Legacy,
    Limit,
    Malformed,
    Unknown,
    Unresolved,
    Unsupported,
}

#[derive(Clone, Copy, Debug)]
enum Association {
    Phase,
    Session,
}
