use hdf5_pure_core::AddressRange;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::v2::{self as independent, HeaderFlags};
use test_util::widths::Widths;

use super::super::{ObjectHeaderAllocation, ObjectHeaderAllocationGeometry};
use crate::address::{StoredAddress, StoredAddressExt};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::object_header::v2::ObjectHeaderPrefix;
use crate::width::FormatWidths;

#[rstest]
#[case::undefined_start(65_535,0,FormatError::DefinedAddressIsUndefined { address:65_535,offset_size:2 })]
#[case::address_too_large(65_536,0,FormatError::AddressTooLarge { address:65_536,offset_size:2 })]
#[case::end_too_large(65_530,10,FormatError::AddressTooLarge { address:65_540,offset_size:2 })]
fn addresses_are_bound_to_the_output_width(
    #[case] address: u64,
    #[case] len: u64,
    #[case] expected: FormatError,
) {
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(2, 2).unwrap(),
        StoredAddress::new(address),
        len,
        &[],
    )
    .err()
    .unwrap();
    assert_eq!(err, expected);
}

#[test]
fn overflowing_complete_addresses_are_rejected() {
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(u64::MAX - 1),
        128,
        &[],
    )
    .unwrap_err();
    assert_eq!(
        err,
        FormatError::OffsetOverflow {
            offset: u64::MAX - 1,
            length: 128
        }
    );
}

#[rstest]
#[case::short(&[32], 33, 32)]
#[case::long(&[32], 31, 32)]
#[case::empty(&[], 31, 0)]
fn chosen_chunks_cover_the_complete_supplied_allocation(
    #[case] lengths: &[u64],
    #[case] expected: u64,
    #[case] actual: u64,
) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(
        geometry
            .into_chunked_allocation(StoredAddress::new(128), expected, lengths)
            .unwrap_err(),
        FormatError::ObjectHeaderAllocationLengthMismatch { expected, actual }
    );
}

#[rstest]
#[case::undefined_start(65_535, 0, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::undefined_end(65_500, 35, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::oversized_end(65_500, 36, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
fn chosen_chunks_keep_output_address_width_checks(
    #[case] address: u64,
    #[case] len: u64,
    #[case] expected: FormatError,
) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(2, 2).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(
        geometry
            .into_chunked_allocation(StoredAddress::new(address), len, &[len])
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::root_child(&[(128, 64), (160, 12)], 0, 1)]
#[case::child_root(&[(160, 64), (128, 40)], 1, 0)]
#[case::duplicate(&[(128, 64), (128, 12)], 0, 1)]
#[case::nonadjacent_chain(&[(128, 64), (1_024, 28), (1_030, 12)], 1, 2)]
fn placed_chunks_reject_complete_allocation_overlap(
    #[case] ranges: &[(u64, u64)],
    #[case] first: usize,
    #[case] second: usize,
) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let err =
        geometry
            .into_placed_allocation(ranges.iter().map(|&(address, len)| {
                AddressRange::new(StoredAddress::new(address), len).unwrap()
            }))
            .unwrap_err();
    assert_eq!(err, FormatError::ObjectHeaderChunkOverlap { first, second });
}

#[rstest]
#[case::missing(&[], 0, 0)]
#[case::empty_root(&[(128, 0)], 0, 0)]
#[case::short_root(&[(128, 30), (512, 12)], 0, 30)]
#[case::short_child(&[(128, 31), (512, 11)], 1, 11)]
#[case::short_continued_child(&[(128, 31), (512, 27), (1_024, 12)], 1, 27)]
#[case::too_long_child(&[(128, 31), (512, 65_548)], 1, 65_548)]
fn placed_chunks_reject_unrepresentable_records(
    #[case] ranges: &[(u64, u64)],
    #[case] index: usize,
    #[case] length: u64,
) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(
        geometry
            .into_placed_allocation(ranges.iter().map(|&(address, len)| {
                AddressRange::new(StoredAddress::new(address), len).unwrap()
            }))
            .unwrap_err(),
        FormatError::InvalidObjectHeaderChunkLength { index, length }
    );
}

#[rstest]
#[case::undefined_start(65_535, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::undefined_end(65_523, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::oversized_end(65_524, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
fn placed_chunks_check_each_output_address(#[case] address: u64, #[case] expected: FormatError) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(2, 2).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(
        geometry
            .into_placed_allocation(
                [(128, 19), (address, 12)]
                    .into_iter()
                    .map(
                        |(address, len)| AddressRange::new(StoredAddress::new(address), len)
                            .unwrap()
                    )
            )
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::last_supported(256)]
#[case::over_limit(257)]
fn placed_chunks_keep_the_chunk_count_limit(#[case] count: usize) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let result = geometry.into_placed_allocation((0..count).map(|index| {
        AddressRange::new(
            StoredAddress::new(128 + index.to_u64() * 64),
            if index == 0 {
                31
            } else if index + 1 == count {
                12
            } else {
                28
            },
        )
        .unwrap()
    }));
    if count > 256 {
        assert_eq!(result.unwrap_err(), FormatError::NestingDepthExceeded);
    } else {
        let plan = result.unwrap();
        assert_eq!(plan.chunks().len(), count);
        assert_eq!(
            plan.encode_chunks()
                .map(|(_, bytes)| bytes.len())
                .sum::<usize>(),
            31 + (count - 2) * 28 + 12
        );
    }
}

#[rstest]
#[case::above_u32_max(1 << 35)]
#[case::last_defined_end(u64::MAX - 32)]
fn placed_chunks_preserve_wide_stored_coordinates(#[case] root: u64) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let ranges = [(root, 31), (512, 12)];
    let plan = geometry
        .into_placed_allocation(
            ranges
                .into_iter()
                .map(|(address, len)| AddressRange::new(StoredAddress::new(address), len).unwrap()),
        )
        .unwrap();
    assert_eq!(
        plan.chunks().collect::<Vec<_>>(),
        ranges.map(|(address, len)| (StoredAddress::new(address), len))
    );
    let chunks = plan.encode_chunks().collect::<Vec<_>>();
    assert_eq!(
        chunks[0],
        (
            StoredAddress::new(root),
            independent::Header::new()
                .message(Message::continuation(512, 12, Widths::EIGHT))
                .build()
        )
    );
    assert_eq!(
        chunks[1],
        (
            StoredAddress::new(512),
            independent::continuation_chunk(&[Message::nil(0)], HeaderFlags::default())
        )
    );
}

#[test]
fn placed_chunks_keep_narrow_length_fields() {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 2).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(
        geometry
            .into_placed_allocation(
                [(128, 25), (512, 65_536)]
                    .into_iter()
                    .map(
                        |(address, len)| AddressRange::new(StoredAddress::new(address), len)
                            .unwrap()
                    )
            )
            .unwrap_err(),
        FormatError::InvalidObjectHeaderChunkLength {
            index: 1,
            length: 65_536
        }
    );
}
