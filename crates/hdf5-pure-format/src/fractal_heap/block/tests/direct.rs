use alloc::vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;

use super::support;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::block::{CHECKSUM_LEN, FractalHeapBlockFilters};
use crate::width::OffsetWidth;

#[rstest]
#[case::two(OffsetWidth::Two, false, 0)]
#[case::four(OffsetWidth::Four, true, 512)]
#[case::eight(OffsetWidth::Eight, true, 1u64 << 40)]
fn direct_views_borrow_complete_blocks_at_physical_coordinates(
    #[case] offsets: OffsetWidth,
    #[case] checksummed: bool,
    #[case] base: u64,
) {
    let heap = support::heap(offsets, 0, checksummed, false);
    let block = support::direct_root(&heap);
    assert_eq!(block.address().get(), 256);
    assert_eq!(block.heap_offset(), 0);
    assert_eq!(block.decoded_size(), 128);
    assert_eq!(block.filter_mask(), None);
    let reference = block
        .at(PhysicalAddress::new(base + block.address().get()))
        .unwrap();
    let bytes = support::direct_bytes(&heap);
    let view = reference
        .parse(support::region(base + 256, &bytes))
        .unwrap();
    assert_eq!(view.bytes(), bytes);
    assert_eq!(view.bytes().as_ptr(), bytes.as_ptr());
    let data_at = heap.prefix_len() + if checksummed { CHECKSUM_LEN } else { 0 };
    assert_eq!(view.data(), vec![0x5A; 128 - data_at]);
    assert_eq!(reference.range(), reference.parse_range());
    assert_eq!(reference.range().start().get(), base + 256);
}

#[rstest]
#[case::unchecksummed(false)]
#[case::checksummed(true)]
fn bounded_object_refs_retain_unvalidated_checksum_metadata(
    #[case] checksummed: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
) {
    let heap = support::heap(offsets, 0, checksummed, false);
    let bytes = support::direct_bytes(&heap);
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(768))
        .unwrap();
    let prefix = reference.object_prefix_range().unwrap();
    let data_at = prefix.length();
    let object = reference
        .parse_object(support::region(768, &bytes[..data_at as usize]), data_at, 3)
        .unwrap();
    assert_eq!(
        object.range(),
        FileRange::new(PhysicalAddress::new(768 + data_at), 3).unwrap()
    );
    assert_eq!(object.block_range(), reference.range());
    assert_eq!(
        object.stored_checksum(),
        checksummed.then(|| {
            u32::from_le_bytes(
                bytes[heap.prefix_len()..heap.prefix_len() + CHECKSUM_LEN]
                    .try_into()
                    .unwrap(),
            )
        })
    );
    assert_eq!(
        object.parse(support::region(768, &bytes)).unwrap(),
        &[0x5A; 3]
    );
}

#[rstest]
#[case::prefix(0, 3)]
#[case::overrun(127, 2)]
#[case::overflow(u64::MAX, 1)]
fn bounded_object_refs_reject_invalid_object_bounds(#[case] offset: u64, #[case] length: u64) {
    let heap = support::heap(OffsetWidth::Eight, 0, true, false);
    let bytes = support::direct_bytes(&heap);
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(256))
        .unwrap();
    assert_eq!(
        reference
            .parse_object(support::region(256, &bytes), offset, length)
            .unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[test]
fn required_checksum_validation_rejects_corruption_with_every_feature_set() {
    let heap = support::heap(OffsetWidth::Eight, 0, true, false);
    let mut bytes = support::direct_bytes(&heap);
    let checksum_at = heap.prefix_len();
    let stored = u32::from_le_bytes(
        bytes[checksum_at..checksum_at + CHECKSUM_LEN]
            .try_into()
            .unwrap(),
    );
    bytes[127] ^= 1;
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(256))
        .unwrap();
    let object = reference
        .parse_object(support::region(256, &bytes), 19, 3)
        .unwrap();
    assert_eq!(object.stored_checksum(), Some(stored));
    let mut zeroed = bytes.clone();
    zeroed[checksum_at..checksum_at + CHECKSUM_LEN].fill(0);
    assert_eq!(
        reference
            .parse_checked(support::region(256, &bytes))
            .unwrap_err(),
        FormatError::ChecksumMismatch {
            expected: stored,
            computed: checksum::jenkins_lookup3(&zeroed)
        }
    );
}
#[test]
fn filtered_refs_keep_encoded_ranges_separate_from_decoded_geometry() {
    let heap = support::heap(OffsetWidth::Eight, 0, true, true);
    let block = support::direct_root(&heap);
    let reference = block.at(PhysicalAddress::new(768)).unwrap();
    assert_eq!(block.decoded_size(), 128);
    assert_eq!(block.filter_mask(), Some(1));
    assert_eq!(reference.range().length(), 47);
    assert_eq!(
        reference.prefix_range().unwrap_err(),
        FormatError::UnsupportedFilteredHeapObject
    );
    assert_eq!(
        reference.parse(support::region(768, &[0; 47])).unwrap_err(),
        FormatError::UnsupportedFilteredHeapObject
    );
}

#[rstest]
#[case::unfiltered(false)]
#[case::filtered(true)]
fn direct_blocks_parse_from_their_decoded_bytes(#[case] filtered: bool) {
    let heap = support::heap(OffsetWidth::Eight, 0, true, filtered);
    let block = support::direct_root(&heap);
    let reference = block.at(PhysicalAddress::new(768)).unwrap();
    let decoded = support::direct_bytes(&heap);
    let data_at = heap.prefix_len() + CHECKSUM_LEN;
    let view = reference.parse_decoded_checked(&decoded).unwrap();
    assert_eq!(view.bytes(), decoded);
    assert_eq!(view.data(), vec![0x5A; 128 - data_at]);
    assert_eq!(
        reference
            .parse_decoded_object(&decoded, data_at.to_u64() + 2, 3)
            .unwrap(),
        &[0x5A; 3]
    );
    let FractalHeapFiltering::Filtered { pipeline, .. } = &heap.filtering else {
        assert_eq!(block.filters(), None);
        return;
    };
    assert_eq!(
        block.filters(),
        Some(FractalHeapBlockFilters { pipeline, mask: 1 })
    );
}

#[rstest]
#[case::short(Damage::Truncate, FormatError::InvalidFractalHeapGeometry)]
#[case::long(Damage::Extend, FormatError::InvalidFractalHeapGeometry)]
#[case::owner(
    Damage::Byte(support::OWNER_AT, 97),
    FormatError::InvalidFractalHeapBlockOwner { expected: 96, actual: 97 }
)]
#[case::offset(
    Damage::Byte(support::BLOCK_OFFSET_AT, 1),
    FormatError::InvalidFractalHeapBlockOffset { expected: 0, actual: 1 }
)]
fn decoded_blocks_reject_a_wrong_length_or_prefix(
    #[case] damage: Damage,
    #[case] expected: FormatError,
) {
    let heap = support::heap(OffsetWidth::Eight, 0, true, true);
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(768))
        .unwrap();
    let mut decoded = support::direct_bytes(&heap);
    match damage {
        Damage::Truncate => {
            decoded.pop();
        }
        Damage::Extend => decoded.push(0),
        Damage::Byte(at, value) => decoded[at] = value,
    }
    assert_eq!(
        reference.parse_decoded_checked(&decoded).unwrap_err(),
        expected
    );
    let data_at = (heap.prefix_len() + CHECKSUM_LEN).to_u64();
    assert_eq!(
        reference
            .parse_decoded_object(&decoded, data_at, 3)
            .unwrap_err(),
        expected
    );
}

#[derive(Clone, Copy, Debug)]
enum Damage {
    Byte(usize, u8),
    Extend,
    Truncate,
}

#[test]
fn a_decoded_block_checksum_is_validated_by_the_checked_parse_alone() {
    let heap = support::heap(OffsetWidth::Eight, 0, true, true);
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(768))
        .unwrap();
    let mut decoded = support::direct_bytes(&heap);
    let checksum_at = heap.prefix_len();
    let stored = u32::from_le_bytes(
        decoded[checksum_at..checksum_at + CHECKSUM_LEN]
            .try_into()
            .unwrap(),
    );
    *decoded.last_mut().unwrap() ^= 1;
    let mut zeroed = decoded.clone();
    zeroed[checksum_at..checksum_at + CHECKSUM_LEN].fill(0);
    assert_eq!(
        reference.parse_decoded_checked(&decoded).unwrap_err(),
        FormatError::ChecksumMismatch {
            expected: stored,
            computed: checksum::jenkins_lookup3(&zeroed)
        }
    );
    let data_at = (checksum_at + CHECKSUM_LEN).to_u64();
    assert_eq!(
        reference
            .parse_decoded_object(&decoded, data_at, 3)
            .unwrap(),
        &[0x5A; 3]
    );
}

#[rstest]
#[case::checksum_field(ObjectBounds::ChecksumField)]
#[case::past_the_block(ObjectBounds::PastTheBlock)]
#[case::overflow(ObjectBounds::Overflow)]
fn decoded_objects_reject_invalid_bounds(#[case] bounds: ObjectBounds) {
    let heap = support::heap(OffsetWidth::Eight, 0, true, true);
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(768))
        .unwrap();
    let decoded = support::direct_bytes(&heap);
    let (offset, length) = match bounds {
        ObjectBounds::ChecksumField => (heap.prefix_len().to_u64(), 3),
        ObjectBounds::Overflow => (u64::MAX, 1),
        ObjectBounds::PastTheBlock => (decoded.len().to_u64() - 2, 3),
    };
    assert_eq!(
        reference
            .parse_decoded_object(&decoded, offset, length)
            .unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[derive(Clone, Copy, Debug)]
enum ObjectBounds {
    ChecksumField,
    Overflow,
    PastTheBlock,
}
#[test]
fn supplied_larger_regions_are_borrowed_at_the_required_range() {
    let heap = support::heap(OffsetWidth::Eight, 2, false, false);
    let bytes = support::indirect_bytes(&heap, &[(0, 0, 512)]);
    let mut image = vec![0xFF; 9];
    image.extend_from_slice(&bytes);
    image.extend_from_slice(&[0xFF; 7]);
    let reference = support::indirect_root(&heap)
        .at(PhysicalAddress::new(521))
        .unwrap();
    let view = reference.parse(support::region(512, &image)).unwrap();
    assert_eq!(view.bytes(), bytes);
    assert_eq!(view.bytes().as_ptr(), image[9..].as_ptr());
    let error = reference.parse(support::region(522, &bytes)).unwrap_err();
    assert_eq!(
        error,
        FormatError::InsufficientRegion {
            required_start: 521,
            required_length: bytes.len() as u64,
            provided_start: 522,
            provided_length: bytes.len() as u64,
        }
    );
}
