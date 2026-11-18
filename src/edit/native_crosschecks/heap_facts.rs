use core::num::NonZeroU16;
use std::collections::BTreeMap;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use test_util::bytes;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;

use crate::{LengthWidth, PhysicalAddress};

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
