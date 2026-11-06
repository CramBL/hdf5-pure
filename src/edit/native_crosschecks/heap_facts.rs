use core::num::NonZeroU16;
use std::collections::BTreeSet;

use rstest::rstest;
use test_util::bytes;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;

use crate::PhysicalAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::test_util as edit_test_util;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

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
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    heap_members::assert_values(&path, token.as_ref());
}
