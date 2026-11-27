use core::ops::RangeInclusive;

use rstest::rstest;
use test_util::object_header::v2::{self as independent, HeaderFlags};
use test_util::object_header::{Message, MessageType as RecordType};

use super::super::{ObjectHeaderAllocation, ObjectHeaderAllocationGeometry};
use super::support::{EmptyRecords, TRACKED_FLAGS};
use crate::access_mode::AccessMode;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::object_header::v2::ObjectHeaderPrefix;
use crate::object_header::{MessageRecordLayout, ObjectHeader};
use crate::width::{FormatWidths, UintWidth};

#[test]
fn too_many_owned_chunks_are_rejected() {
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        17_000_000,
        &[],
    )
    .err()
    .unwrap();
    assert!(matches!(err, FormatError::NestingDepthExceeded), "{err:?}");
}

#[test]
fn oversized_materialization_is_rejected() {
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        u64::MAX,
        &[],
    )
    .err()
    .unwrap();
    assert_eq!(
        err,
        FormatError::ValueTooLargeForPlatform {
            value: u64::MAX,
            target: "object header allocation"
        }
    );
}

#[test]
fn narrow_continuation_lengths_bound_each_child_allocation() {
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 2).unwrap(),
        StoredAddress::new(128),
        1_048_576,
        &[],
    )
    .unwrap();
    let image = plan.encode();
    assert!(
        plan.chunks()
            .skip(1)
            .all(|(_, len)| len <= u64::from(u16::MAX))
    );
    let mut data = vec![0; 128];
    data.extend(image);
    assert_eq!(
        ObjectHeader::parse(&data, AccessMode::ReadOnly, 128, 8, 2)
            .unwrap()
            .messages
            .len(),
        0
    );
}

#[test]
fn malformed_later_records_are_rejected_before_encoding() {
    let mut records = independent::message_record(
        &Message::new(RecordType::NIL, &[0; 4]),
        HeaderFlags::default(),
    );
    records.extend([1, 4, 0, 0, 1]);
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        128,
        &records,
    )
    .err()
    .unwrap();
    assert_eq!(
        err,
        FormatError::UnexpectedEof {
            expected: 16,
            available: 13
        }
    );
}

#[test]
fn external_continuation_records_require_flattening() {
    let records = independent::message_record(
        &Message::new(RecordType::OBJECT_HEADER_CONTINUATION, &[0; 16]),
        HeaderFlags::default(),
    );
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        128,
        &records,
    )
    .err()
    .unwrap();
    assert_eq!(
        err,
        FormatError::Internal(
            "an object-header allocation requires flattened message records".into()
        )
    );
}

#[test]
fn insufficient_capacity_is_rejected() {
    let records = independent::message_record(
        &Message::new(RecordType::DATATYPE, &[5, 6]),
        HeaderFlags::default(),
    );
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        16,
        &records,
    )
    .err()
    .unwrap();
    assert_eq!(
        err,
        FormatError::Internal(
            "a 16-byte allocation cannot represent an object header with 6 message bytes".into()
        )
    );
}

#[test]
fn the_last_supported_chunk_fits_without_materializing_its_payload() {
    let records = independent::message_record(
        &Message::new(RecordType::DATATYPE, &[5, 6]),
        HeaderFlags::default(),
    );
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        16_785_144,
        &records,
    )
    .unwrap();
    assert_eq!(plan.chunks().len(), 256);
    assert_eq!(plan.chunks().map(|(_, len)| len).sum::<u64>(), 16_785_144);
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        16_785_145,
        &records,
    )
    .unwrap_err();
    assert!(matches!(err, FormatError::NestingDepthExceeded), "{err:?}");
}

#[rstest]
#[case::plain_gap(HeaderFlags::default(), 13)]
#[case::tracked_first_gap(TRACKED_FLAGS, 13)]
#[case::tracked_second_gap(TRACKED_FLAGS, 15)]
#[case::tracked_third_gap(TRACKED_FLAGS, 16)]
#[case::smaller_than_prefix(HeaderFlags::default(), 10)]
fn empty_capacity_without_a_complete_representation_is_rejected(
    #[case] flags: HeaderFlags,
    #[case] allocation: u64,
    #[values(EmptyRecords::Empty, EmptyRecords::NilOnly)] input: EmptyRecords,
) {
    let records = input.records(flags);
    let err = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        allocation,
        &records,
    )
    .unwrap_err();
    assert_eq!(
        err,
        FormatError::Internal(format!(
            "a {allocation}-byte allocation cannot represent an object header with 0 message bytes"
        ))
    );
}

#[rstest]
#[case::plain(HeaderFlags::default(), &[11..=11, 15..=266, 12..=12, 16..=65_547, 14..=14, 18..=65_553, 18..=18, 22..=65_557], 12..=65_547, 28..=65_567)]
#[case::tracked(TRACKED_FLAGS, &[11..=11, 17..=266, 12..=12, 18..=65_547, 14..=14, 20..=65_555, 18..=18, 24..=65_559], 14..=65_549, 30..=65_571)]
fn empty_geometry_exposes_exact_framing_ranges(
    #[case] flags: HeaderFlags,
    #[case] roots: &[RangeInclusive<u64>],
    #[case] terminal: RangeInclusive<u64>,
    #[case] continued: RangeInclusive<u64>,
) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(
        geometry.root_chunk_lengths(false).collect::<Vec<_>>(),
        roots
    );
    assert_eq!(geometry.continuation_chunk_lengths(false), terminal);
    assert_eq!(geometry.continuation_chunk_lengths(true), continued);
}

#[rstest]
#[case::too_short_root(&[30, 12], 0, 30)]
#[case::too_short_child(&[37, 11], 1, 11)]
#[case::too_short_continued_child(&[37, 27, 12], 1, 27)]
#[case::oversized_padding(&[37, 65_548], 1, 65_548)]
#[case::unrepresentable_empty_root(&[13], 0, 13)]
#[case::missing_root(&[], 0, 0)]
fn chosen_chunk_lengths_reject_unrepresentable_records(
    #[case] lengths: &[u64],
    #[case] index: usize,
    #[case] length: u64,
) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let err = geometry
        .into_chunked_allocation(StoredAddress::new(128), lengths.iter().sum(), lengths)
        .unwrap_err();
    assert_eq!(
        err,
        FormatError::InvalidObjectHeaderChunkLength { index, length }
    );
}

#[rstest]
#[case::one_byte(266, UintWidth::One)]
#[case::two_bytes(267, UintWidth::Two)]
#[case::last_two_bytes(65_547, UintWidth::Two)]
#[case::four_bytes(65_548, UintWidth::Four)]
fn chosen_root_lengths_select_a_representable_size_field(
    #[case] root_len: u64,
    #[case] width: UintWidth,
) {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let plan = geometry
        .into_chunked_allocation(StoredAddress::new(128), root_len + 12, &[root_len, 12])
        .unwrap();
    assert_eq!(
        UintWidth::from_flags(ObjectHeaderPrefix::parse(&plan.encode()).unwrap().flags),
        width
    );
}

#[rstest]
#[case::last_supported(256)]
#[case::over_limit(257)]
fn chosen_chunks_keep_the_chunk_count_limit(#[case] count: usize) {
    let mut lengths = vec![28; count];
    lengths[0] = 31;
    lengths[count - 1] = 12;
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    let result =
        geometry.into_chunked_allocation(StoredAddress::new(128), lengths.iter().sum(), &lengths);
    if count > 256 {
        assert_eq!(result.unwrap_err(), FormatError::NestingDepthExceeded);
    } else {
        let plan = result.unwrap();
        assert_eq!(plan.chunks().len(), count);
        assert_eq!(plan.encode().len().to_u64(), lengths.iter().sum());
    }
}

#[test]
fn narrow_length_fields_bound_exposed_and_chosen_chunks() {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 2).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(geometry.continuation_chunk_lengths(false), 12..=65_535);
    assert_eq!(geometry.continuation_chunk_lengths(true), 22..=65_535);
    let err = geometry
        .into_chunked_allocation(StoredAddress::new(128), 65_567, &[31, 65_536])
        .unwrap_err();
    assert_eq!(
        err,
        FormatError::InvalidObjectHeaderChunkLength {
            index: 1,
            length: 65_536
        }
    );
}

#[test]
fn chosen_chunk_length_sum_overflow_is_rejected() {
    let geometry = ObjectHeaderAllocationGeometry::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(
        geometry
            .into_chunked_allocation(StoredAddress::new(128), 64, &[u64::MAX, 1])
            .unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX,
            length: 1
        }
    );
}
