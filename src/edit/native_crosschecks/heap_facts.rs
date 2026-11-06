use core::num::NonZeroU16;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use rstest::rstest;
use test_util::bytes;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;

use crate::LengthWidth;
use crate::PhysicalAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::heap::associations::MemberPresence;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::test_util as edit_test_util;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

#[test]
fn native_replacement_removes_a_manually_aliased_heap_member() {
    let path = test_util::temp::temp_path("native-heap-member-lifetime.h5");
    let fixture = heap_members::MemberLifetime::write(&path);
    let (collection, original_index) = token_id(fixture.original());
    let (retained_collection, retained_index) = token_id(fixture.retained());
    assert_eq!(retained_collection, collection);
    assert_ne!(retained_index, original_index);
    let mut image = std::fs::read(&path).unwrap();
    let mut original_members = collection_members(&image, collection);
    assert_eq!(original_members.len(), 3);
    let original_payload = original_members[&original_index].1.clone();
    assert_eq!(original_payload, original_members[&retained_index].1);
    let descriptor = global_heap::reference(
        u32::try_from(original_payload.len()).unwrap(),
        collection.get(),
        u32::from(original_index.get()),
        Widths::EIGHT,
    );
    let sequence_at = usize::try_from(fixture.sequence_address()).unwrap();
    assert_eq!(
        bytes::slice_at(&image, sequence_at, descriptor.len()),
        vec![0; descriptor.len()]
    );
    bytes::set_slice_at(&mut image, sequence_at, &descriptor);
    std::fs::write(&path, &image).unwrap();
    fixture.assert_aliased_values(&path, &original_payload);
    assert_eq!(std::fs::read(&path).unwrap(), image);
    let retained = fixture.replace_with_empty(&path);
    let replaced = std::fs::read(&path).unwrap();
    assert_eq!(
        bytes::slice_at(&replaced, sequence_at, descriptor.len()),
        vec![0; descriptor.len()]
    );
    assert_eq!(
        original_members.remove(&original_index).unwrap().1,
        original_payload
    );
    assert_eq!(collection_members(&replaced, collection), original_members);
    heap_members::assert_lifetime_replacement(&path, &retained);
    assert_eq!(std::fs::read(&path).unwrap(), replaced);
}

#[rstest]
#[case::ordinary(false)]
#[case::external_region_token(true)]
fn native_heap_members_are_distinct_from_root_reference_completeness(#[case] external_token: bool) {
    let path = test_util::temp::temp_path("native-heap-member-facts.h5");
    let token = heap_members::write(&path, external_token);
    let original = std::fs::read(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let live = phase.collections_reachable_from([root]).unwrap();
    assert!(live.analysis_complete);
    let references = phase.heap_reference_facts(&source).unwrap();
    assert!(references.is_complete());
    assert!(references.error().is_none());
    assert_eq!(references.objects().len(), 2);
    assert_eq!(references.collections(), &live.referenced_collections);
    assert_eq!(live.referenced_collections.len(), 1);
    let referenced = references.objects().clone();
    let collection = PhysicalAddress::new(*live.referenced_collections.first().unwrap());
    phase.collection_allocation(collection).unwrap();
    let contents = phase.heap_members(&source, collection).unwrap();
    assert_eq!(
        contents.range().length(),
        bytes::uint_at(
            &original,
            usize::try_from(collection.get()).unwrap() + global_heap::SIZE_OFFSET,
            Widths::EIGHT.length
        )
    );
    assert_eq!(contents.members().len(), if external_token { 3 } else { 2 });
    let allocated = contents
        .members()
        .iter()
        .map(|member| GlobalHeapId {
            collection,
            index: member.index(),
        })
        .collect::<BTreeSet<_>>();
    assert!(referenced.is_subset(&allocated));
    assert!(
        contents
            .members()
            .iter()
            .all(|member| member.reference_count() == 0)
    );
    let missing = allocated
        .difference(&referenced)
        .copied()
        .collect::<Vec<_>>();
    if let Some(token) = &token {
        let offsets = usize::from(engine.widths().offsets.get());
        let address = engine
            .addresses()
            .resolve(StoredAddress::new(bytes::uint_at(
                token.bytes(),
                0,
                offsets,
            )))
            .unwrap();
        let index = u16::try_from(bytes::u32_at(token.bytes(), offsets)).unwrap();
        assert_eq!(
            missing,
            [GlobalHeapId {
                collection: address,
                index: NonZeroU16::new(index).unwrap()
            }]
        );
        assert_eq!(address, collection);
        let member = contents
            .members()
            .iter()
            .find(|member| member.index().get() == index)
            .unwrap();
        assert!(member.payload().range().length() > 0);
        assert!(member.range().length() > member.payload().range().length());
    } else {
        assert_eq!(missing, []);
    }
    let expected_length = contents.range().length();
    let claims = phase.physical_claims(&source).unwrap();
    assert_eq!(claims.root(), Some(root));
    assert_eq!(
        claims
            .claims()
            .iter()
            .filter(|claim| claim.origin() == ClaimOrigin::Heap(collection))
            .map(|claim| (claim.range().start(), claim.range().length()))
            .collect::<Vec<_>>(),
        [(collection, expected_length)]
    );
    let coverage = phase.physical_coverage(&source).unwrap();
    assert_eq!(coverage.conflict(), None);
    assert!(coverage.beyond_source().is_empty());
    assert!(
        coverage
            .covered()
            .iter()
            .any(|range| range.start() <= collection
                && range.end().get() >= collection.get() + expected_length)
    );
    let associations = phase.member_associations(&source).unwrap();
    assert!(associations.topology_available());
    assert!(associations.incomplete_references().is_empty());
    let association = &associations.collections()[&collection];
    assert!(association.error().is_none());
    assert_eq!(
        association.unmatched(),
        missing.iter().map(|id| id.index).collect::<Vec<_>>()
    );
    assert_eq!(association.observed().len(), 2);
    for observed in association.observed().values() {
        assert_eq!(observed.presence(), MemberPresence::Allocated);
        assert_eq!(observed.referrers(), [root]);
    }
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    heap_members::assert_values(&path, token.as_ref());
}

/// Decodes the fixture's collection address and member index from a native token.
fn token_id(token: &heap_members::RegionToken) -> (PhysicalAddress, NonZeroU16) {
    (
        PhysicalAddress::new(bytes::uint_at(token.bytes(), 0, Widths::EIGHT.offset)),
        NonZeroU16::new(u16::try_from(bytes::u32_at(token.bytes(), Widths::EIGHT.offset)).unwrap())
            .unwrap(),
    )
}

/// Parses allocated member indices, stored counts, and payloads from the image.
fn collection_members(
    bytes: &[u8],
    address: PhysicalAddress,
) -> BTreeMap<NonZeroU16, (u16, Vec<u8>)> {
    let image = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(bytes.len()).unwrap()).unwrap(),
        bytes,
    )
    .unwrap();
    let locator = GlobalHeapCollectionLocator::new(address, LengthWidth::Eight);
    let collection = locator
        .parse(image.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap();
    let mut walk = collection.walk();
    let mut members = BTreeMap::new();
    while let Some(range) = walk.parse_range() {
        if let Some(member) = walk.parse(image.subregion(range).unwrap()).unwrap() {
            assert_eq!(
                members.insert(
                    member.index(),
                    (
                        member.reference_count(),
                        member.payload().parse(image).unwrap().to_vec()
                    )
                ),
                None
            );
        }
    }
    members
}
