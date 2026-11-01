use core::slice;

use rstest::rstest;
use test_util::object_header::v2::{self as independent, HeaderFlags};
use test_util::object_header::{Message, MessageType as RecordType};
use test_util::widths::Widths;

use crate::access_mode::AccessMode;
use crate::address::StoredAddressExt;
use crate::object_header::{AttributePhaseChange, MessageRecordLayout, ObjectHeader, ObjectTimes};

use super::*;

#[rstest]
#[case::plain(MessageRecordLayout::PLAIN, HeaderFlags::default())]
#[case::tracked(MessageRecordLayout::from_header_flags(
    (HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER).0),
    HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER)]
fn one_chunk_matches_independent_bytes(
    #[case] layout: MessageRecordLayout,
    #[case] flags: HeaderFlags,
    #[values(0, 1, 3, 4, 6, 17)] extra: usize,
) {
    let prefix = ObjectHeaderPrefix::with_layout(layout);
    let message = Message::new(RecordType::DATATYPE, &[5, 6]).with_creation_order(17);
    let messages = independent::message_records(slice::from_ref(&message), flags);
    let minimal = independent::Header::new()
        .flags(flags)
        .message(message.clone())
        .build();
    let mut expected = independent::Header::new().flags(flags).message(message);
    if extra >= layout.prefix_len() {
        expected = expected.message(Message::nil(extra - layout.prefix_len()));
    }
    let mut expected = expected.build();
    if extra < layout.prefix_len() && extra > 0 {
        let size = test_util::bytes::uint_at(&expected, 6, 1);
        expected.truncate(expected.len() - CHECKSUM_LEN);
        expected.resize(expected.len() + extra, 0);
        test_util::bytes::set_uint_at(&mut expected, 6, 1, size + extra.to_u64());
        test_util::checksum::append(&mut expected);
    }
    let plan = ObjectHeaderAllocation::new(
        prefix,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        (minimal.len() + extra).to_u64(),
        &messages,
    )
    .unwrap();
    assert_eq!(plan.encode(), expected);
    assert_eq!(
        plan.chunks().collect::<Vec<_>>(),
        [(StoredAddress::new(128), expected.len().to_u64())]
    );
}

#[rstest]
#[case::plain(MessageRecordLayout::PLAIN, HeaderFlags::default())]
#[case::tracked(MessageRecordLayout::from_header_flags(
    (HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER).0),
    HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER)]
fn continued_padding_matches_independent_bytes(
    #[case] layout: MessageRecordLayout,
    #[case] flags: HeaderFlags,
    #[values(65_540, 65_541, 65_542, 65_543, 65_544, 65_545, 131_123, 1_048_576)] allocation: usize,
) {
    let message = Message::new(RecordType::DATATYPE, &[5, 6]).with_creation_order(17);
    let messages = independent::message_records(slice::from_ref(&message), flags);
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::with_layout(layout),
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        allocation.to_u64(),
        &messages,
    )
    .unwrap();
    let chunks = plan.chunks().collect::<Vec<_>>();
    let actual = plan.encode();
    let mut expected = Vec::new();
    for (index, &(address, len)) in chunks.iter().enumerate() {
        assert_eq!(address.get(), 128 + expected.len().to_u64());
        let mut records = Vec::new();
        if index == 0 {
            records.push(message.clone());
        }
        let next = chunks
            .get(index + 1)
            .map(|&(address, length)| Message::continuation(address.get(), length, Widths::EIGHT));
        let chunk_flags = if index == 0 {
            let width =
                ObjectHeaderPrefix::parse(&actual).unwrap().flags & HeaderFlags::CHUNK_SIZE_WIDTH.0;
            HeaderFlags(flags.0 | width)
        } else {
            flags
        };
        let fixed = if index == 0 {
            6 + chunk_flags.chunk_size_width() + CHECKSUM_LEN
        } else {
            8
        };
        let encoded_records = independent::message_records(&records, flags).len();
        let continuation_len = next.as_ref().map_or(0, |_| layout.prefix_len() + 16);
        let padding = len.to_usize().unwrap() - fixed - encoded_records - continuation_len;
        assert!(padding >= layout.prefix_len());
        assert!(padding - layout.prefix_len() <= OBJECT_HEADER_MESSAGE_MAX);
        records.push(Message::nil(padding - layout.prefix_len()));
        records.extend(next);
        let chunk = if index == 0 {
            independent::Header::new()
                .flags(chunk_flags)
                .messages(records)
                .build()
        } else {
            independent::continuation_chunk(&records, flags)
        };
        assert_eq!(chunk.len().to_u64(), len);
        expected.extend_from_slice(&chunk);
    }
    assert_eq!(actual, expected);
    let mut image = vec![0; 128];
    image.extend_from_slice(&actual);
    let parsed = ObjectHeader::parse(&image, AccessMode::ReadOnly, 128, 8, 8).unwrap();
    assert_eq!(parsed.messages.len(), 1);
    assert_eq!(parsed.messages[0].data, [5, 6]);
}

#[rstest]
#[case::one_byte(255)]
#[case::two_bytes(256)]
#[case::last_two_bytes(65_535)]
#[case::four_bytes(65_536)]
fn prefix_growth_uses_owned_capacity(
    #[case] len: usize,
    #[values(0, 1, 2, 3, 4, 5, 6, 7)] extra: usize,
) {
    let prefix = ObjectHeaderPrefix::PLAIN;
    let records = prefix.layout.record(
        MessageType::DATATYPE,
        &vec![7; len.min(OBJECT_HEADER_MESSAGE_MAX)],
    );
    let allocation = prefix.encode_header(&records).len() + extra;
    let plan = ObjectHeaderAllocation::new(
        prefix,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        allocation.to_u64(),
        &records,
    )
    .unwrap();
    assert_eq!(plan.encode().len(), allocation);
}

#[rstest]
fn padding_keeps_optional_fields_and_discards_old_nil_records(
    #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    #[values(None, Some(ObjectTimes { access: 1, modification: 2, change: 3, birth: 4 }))]
    times: Option<ObjectTimes>,
    #[values(None, Some(AttributePhaseChange { max_compact: 8, min_dense: 6 }))] phase: Option<
        AttributePhaseChange,
    >,
) {
    let prefix = ObjectHeaderPrefix {
        times,
        attribute_phase_change: phase,
        layout,
    };
    let mut records = prefix
        .layout
        .record(MessageType::NIL, &vec![0; OBJECT_HEADER_MESSAGE_MAX]);
    records.extend(prefix.layout.record_with_creation_index(
        MessageType::DATATYPE,
        MessageFlags::CONSTANT,
        0,
        &[5, 6],
    ));
    records.extend(prefix.layout.record(MessageType::NIL, &[0; 3]));
    let plan = ObjectHeaderAllocation::new(
        prefix,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        131_123,
        &records,
    )
    .unwrap();
    let actual = plan.encode();
    let parsed = ObjectHeaderPrefix::parse(&actual).unwrap();
    assert_eq!(parsed.prefix, prefix);
    let mut image = vec![0; 128];
    image.extend(actual);
    let parsed = ObjectHeader::parse(&image, AccessMode::ReadOnly, 128, 8, 8).unwrap();
    assert_eq!(parsed.messages.len(), 1);
    assert_eq!(parsed.messages[0].data, [5, 6]);
    assert_eq!(parsed.messages[0].flags, MessageFlags::CONSTANT);
}

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

#[rstest]
#[case::three_chunks(131_127, 3)]
#[case::sixteen_chunks(1_048_576, 16)]
fn chunk_geometry_is_bounded_independently_of_native_coalescing(
    #[case] allocation: u64,
    #[case] count: usize,
) {
    let records = ObjectHeaderPrefix::PLAIN
        .layout
        .record(MessageType::DATATYPE, &[5, 6]);
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        allocation,
        &records,
    )
    .unwrap();
    assert_eq!(plan.chunks().len(), count);
    let chunks = plan.chunks().collect::<Vec<_>>();
    assert_eq!(chunks.first().unwrap().0, StoredAddress::new(128));
    assert_eq!(
        chunks.last().unwrap().0.get() + chunks.last().unwrap().1,
        128 + allocation
    );
    assert_eq!(chunks.iter().map(|(_, len)| *len).sum::<u64>(), allocation);
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
fn malformed_later_records_are_rejected_before_encoding() {
    let mut records = ObjectHeaderPrefix::PLAIN
        .layout
        .record(MessageType::NIL, &[0; 4]);
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
    let records = ObjectHeaderPrefix::PLAIN
        .layout
        .record(MessageType::OBJECT_HEADER_CONTINUATION, &[0; 16]);
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
    let records = ObjectHeaderPrefix::PLAIN
        .layout
        .record(MessageType::DATATYPE, &[5, 6]);
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

#[rstest]
#[case::minimum(17, UintWidth::One)]
#[case::last_one_byte(266, UintWidth::One)]
#[case::first_two_bytes(267, UintWidth::Two)]
#[case::last_two_bytes(65_547, UintWidth::Two)]
#[case::first_four_bytes(65_548, UintWidth::Four)]
fn root_size_width_matches_independent_thresholds(
    #[case] allocation: u64,
    #[case] expected: UintWidth,
) {
    let records = ObjectHeaderPrefix::PLAIN
        .layout
        .record(MessageType::DATATYPE, &[5, 6]);
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::PLAIN,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        allocation,
        &records,
    )
    .unwrap();
    let bytes = plan.encode();
    assert_eq!(
        UintWidth::from_flags(ObjectHeaderPrefix::parse(&bytes).unwrap().flags),
        expected
    );
}

#[rstest]
#[case::plain(HeaderFlags::default(), 65_560, 65_548, 65_504, 12)]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER, 65_564, 65_550, 65_500, 14)]
fn the_first_continuation_has_independently_measured_geometry(
    #[case] layout_flags: HeaderFlags,
    #[case] allocation: u64,
    #[case] root_len: u64,
    #[case] root_nil_body: usize,
    #[case] child_len: u64,
) {
    let flags = HeaderFlags(layout_flags.0 | FOUR_BYTE_CHUNK_SIZE.0);
    let layout = MessageRecordLayout::from_header_flags(flags.0);
    let message = Message::new(RecordType::DATATYPE, &[5, 6]);
    let records = independent::message_records(slice::from_ref(&message), layout_flags);
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::with_layout(layout),
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        allocation,
        &records,
    )
    .unwrap();
    assert_eq!(
        plan.chunks().collect::<Vec<_>>(),
        [
            (StoredAddress::new(128), root_len),
            (StoredAddress::new(128 + root_len), child_len)
        ]
    );
    let expected_root = independent::Header::new()
        .flags(flags)
        .message(message)
        .message(Message::nil(root_nil_body))
        .message(Message::continuation(
            128 + root_len,
            child_len,
            Widths::EIGHT,
        ))
        .build();
    let expected_child = independent::continuation_chunk(&[Message::nil(0)], layout_flags);
    assert_eq!(expected_root.len().to_u64(), root_len);
    assert_eq!(expected_child.len().to_u64(), child_len);
    assert_eq!(plan.encode(), [expected_root, expected_child].concat());
}

#[test]
fn the_last_supported_chunk_fits_without_materializing_its_payload() {
    let records = ObjectHeaderPrefix::PLAIN
        .layout
        .record(MessageType::DATATYPE, &[5, 6]);
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

/// Selects a four-byte Size of Chunk #0 field, as defined in "Version 2 Data Object Header Prefix"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
const FOUR_BYTE_CHUNK_SIZE: HeaderFlags = HeaderFlags(0x02);

#[rstest]
#[case::plain_minimum(HeaderFlags::default(), 11, HeaderFlags::default(), None)]
#[case::plain_wider_prefix(HeaderFlags::default(), 12, TWO_BYTE_CHUNK_SIZE, None)]
#[case::plain_four_byte_prefix(HeaderFlags::default(), 14, FOUR_BYTE_CHUNK_SIZE, None)]
#[case::plain_nil_prefix(HeaderFlags::default(), 15, HeaderFlags::default(), Some(0))]
#[case::plain_nil_body(HeaderFlags::default(), 16, HeaderFlags::default(), Some(1))]
#[case::tracked_minimum(TRACKED_FLAGS, 11, HeaderFlags::default(), None)]
#[case::tracked_wider_prefix(TRACKED_FLAGS, 12, TWO_BYTE_CHUNK_SIZE, None)]
#[case::tracked_four_byte_prefix(TRACKED_FLAGS, 14, FOUR_BYTE_CHUNK_SIZE, None)]
#[case::tracked_nil_prefix(TRACKED_FLAGS, 17, HeaderFlags::default(), Some(0))]
#[case::tracked_nil_body(TRACKED_FLAGS, 18, HeaderFlags::default(), Some(1))]
fn empty_chunks_match_independent_zero_body_or_nil_bytes(
    #[case] flags: HeaderFlags,
    #[case] allocation: u64,
    #[case] size_flags: HeaderFlags,
    #[case] nil_body: Option<usize>,
    #[values(EmptyRecords::Empty, EmptyRecords::NilOnly)] input: EmptyRecords,
) {
    let records = input.records(flags);
    let prefix = ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0));
    let plan = ObjectHeaderAllocation::new(
        prefix,
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(128),
        allocation,
        &records,
    )
    .unwrap();
    let expected = independent::Header::new()
        .flags(HeaderFlags(flags.0 | size_flags.0))
        .messages(nil_body.map(Message::nil))
        .build();
    assert_eq!(expected.len().to_u64(), allocation);
    assert_eq!(plan.encode(), expected);
    assert_eq!(
        plan.chunks().collect::<Vec<_>>(),
        [(StoredAddress::new(128), allocation)]
    );
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

#[derive(Clone, Copy)]
enum EmptyRecords {
    Empty,
    NilOnly,
}

impl EmptyRecords {
    fn records(self, flags: HeaderFlags) -> Vec<u8> {
        match self {
            Self::Empty => Vec::new(),
            Self::NilOnly => {
                independent::message_records(&[Message::nil(17), Message::nil(1)], flags)
            }
        }
    }
}

const TRACKED_FLAGS: HeaderFlags =
    HeaderFlags(HeaderFlags::TRACKS_CREATION_ORDER.0 | HeaderFlags::INDEXES_CREATION_ORDER.0);
const TRACKED: MessageRecordLayout = MessageRecordLayout::from_header_flags(TRACKED_FLAGS.0);
/// Selects a two-byte Size of Chunk #0 field, as defined in "Version 2 Data Object Header Prefix"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
const TWO_BYTE_CHUNK_SIZE: HeaderFlags = HeaderFlags(0x01);
