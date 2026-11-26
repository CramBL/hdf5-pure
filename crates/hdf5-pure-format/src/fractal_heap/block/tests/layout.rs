use alloc::vec;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::support;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
#[cfg(feature = "checksum")]
use crate::checksum;
use crate::error::FormatError;
use crate::filter_pipeline::{FilterDescription, FilterPipeline, VERSION_TWO};
use crate::fractal_heap::block::{DIRECT_SIGNATURE, FractalHeapBlockLayout};
use crate::fractal_heap::{FractalHeapFiltering, FractalHeapHeader};
use crate::width::{LengthWidth, OffsetWidth};

#[rstest]
#[case::direct(false)]
#[case::indirect(true)]
fn supplied_short_regions_return_insufficient_region(#[case] indirect: bool) {
    let heap = support::heap(
        OffsetWidth::Eight,
        if indirect { 2 } else { 0 },
        false,
        false,
    );
    let bytes = if indirect {
        support::indirect_bytes(&heap, &[])
    } else {
        support::direct_bytes(&heap)
    };
    let required_len = bytes.len() as u64;
    let supplied = support::region(256, &bytes[..bytes.len() - 1]);
    let error = if indirect {
        support::indirect_root(&heap)
            .at(PhysicalAddress::new(256))
            .unwrap()
            .parse(supplied)
            .unwrap_err()
    } else {
        support::direct_root(&heap)
            .at(PhysicalAddress::new(256))
            .unwrap()
            .parse(supplied)
            .unwrap_err()
    };
    assert_eq!(
        error,
        FormatError::InsufficientRegion {
            required_start: 256,
            required_length: required_len,
            provided_start: 256,
            provided_length: required_len - 1,
        }
    );
}

#[rstest]
#[case::signature(0, 0, FormatError::InvalidFractalHeapSignature)]
#[case::version(4, 1, FormatError::InvalidFractalHeapVersion(1))]
#[case::owner(5, 97, FormatError::InvalidFractalHeapBlockOwner { expected: 96, actual: 97 })]
#[case::offset(13, 1, FormatError::InvalidFractalHeapBlockOffset { expected: 0, actual: 1 })]
fn block_prefixes_reject_malformed_fields(
    #[case] position: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
) {
    let heap = support::heap(OffsetWidth::Eight, 0, false, false);
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(256))
        .unwrap();
    let mut bytes = support::direct_bytes(&heap);
    bytes[position] = value;
    assert_eq!(
        reference.parse(support::region(256, &bytes)).unwrap_err(),
        expected
    );
    assert_eq!(
        reference
            .parse_prefix(support::region(256, &bytes))
            .unwrap_err(),
        expected
    );
}

#[test]
fn prefix_parsing_requires_only_structural_bytes() {
    let heap = support::heap(OffsetWidth::Eight, 0, true, false);
    let reference = support::direct_root(&heap)
        .at(PhysicalAddress::new(256))
        .unwrap();
    let bytes = support::prefix(&heap, &DIRECT_SIGNATURE, 0);
    assert_eq!(
        reference.prefix_range().unwrap().length(),
        bytes.len() as u64
    );
    reference
        .parse_prefix(support::region(256, &bytes))
        .unwrap();
    assert_eq!(
        reference.parse(support::region(256, &bytes)).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: 256,
            required_length: 128,
            provided_start: 256,
            provided_length: bytes.len() as u64,
        }
    );
}
#[test]
fn physical_range_overflow_is_distinct_from_block_geometry() {
    let heap = support::heap(OffsetWidth::Eight, 0, false, false);
    assert_eq!(
        support::direct_root(&heap)
            .at(PhysicalAddress::new(u64::MAX - 127))
            .unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX - 127,
            length: 128
        }
    );
}

#[cfg(feature = "checksum")]
#[rstest]
#[case::direct(false)]
#[case::indirect(true)]
fn complete_block_views_reject_bad_checksums(#[case] indirect: bool) {
    let heap = support::heap(
        OffsetWidth::Eight,
        if indirect { 1 } else { 0 },
        true,
        false,
    );
    let mut bytes = if indirect {
        support::indirect_bytes(&heap, &[])
    } else {
        support::direct_bytes(&heap)
    };
    let checksum_at = if indirect {
        bytes.len() - 4
    } else {
        heap.prefix_len()
    };
    let stored = u32::from_le_bytes(bytes[checksum_at..checksum_at + 4].try_into().unwrap());
    bytes[checksum_at] ^= 1;
    let expected = stored ^ 1;
    let computed = if indirect {
        checksum::jenkins_lookup3(&bytes[..checksum_at])
    } else {
        let mut zeroed = bytes.clone();
        zeroed[checksum_at..checksum_at + 4].fill(0);
        checksum::jenkins_lookup3(&zeroed)
    };
    let error = if indirect {
        support::indirect_root(&heap)
            .at(PhysicalAddress::new(256))
            .unwrap()
            .parse(support::region(256, &bytes))
            .unwrap_err()
    } else {
        support::direct_root(&heap)
            .at(PhysicalAddress::new(256))
            .unwrap()
            .parse(support::region(256, &bytes))
            .unwrap_err()
    };
    assert_eq!(error, FormatError::ChecksumMismatch { expected, computed });
}
#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn empty_managed_heaps_can_still_contain_huge_objects(#[case] offsets: OffsetWidth) {
    let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
    let mut header =
        FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    header.root_block_address = StoredAddress::undefined(offsets);
    header.huge_objects_count = 3;
    header.huge_objects_size = 300_000;
    let heap =
        FractalHeapBlockLayout::new(&header, offsets, LengthWidth::Eight, StoredAddress::new(96))
            .unwrap();
    assert_eq!(heap.root().unwrap(), None);
}

#[rstest]
#[case::undefined_owner(0, 0xFFFF)]
#[case::unrepresentable_owner(0, 0x1_0000)]
#[case::unrepresentable_root(1, 96)]
#[case::invalid_root_rows(2, 96)]
#[case::invalid_table_width(3, 96)]
#[case::zero_encoded_root(4, 96)]
#[case::invalid_root_mask(5, 96)]
#[case::small_direct_root(6, 96)]
fn block_layout_rejects_invalid_header_geometry(#[case] mutation: u8, #[case] owner: u64) {
    let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
    let mut header =
        FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    match mutation {
        0 => {}
        1 => header.root_block_address = StoredAddress::new(0x1_0000),
        2 => header.current_rows_in_root_indirect_block = 9,
        3 => header.table_width = 3,
        4 | 5 => {
            header.filtering = FractalHeapFiltering::Filtered {
                root_direct_block_size: if mutation == 4 { 0 } else { 47 },
                root_filter_mask: if mutation == 5 { 2 } else { 0 },
                pipeline: FilterPipeline::new(
                    VERSION_TWO,
                    vec![FilterDescription {
                        filter_id: support::FILTER_DEFLATE,
                        name: None,
                        flags: 1,
                        client_data: vec![6],
                    }],
                )
                .unwrap(),
            }
        }
        6 => {
            header.starting_block_size = 8;
        }
        _ => panic!("invalid test mutation"),
    }
    assert_eq!(
        FractalHeapBlockLayout::new(
            &header,
            OffsetWidth::Two,
            LengthWidth::Eight,
            StoredAddress::new(owner)
        )
        .unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}
