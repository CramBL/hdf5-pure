use core::iter;
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
    #[values(false, true)] placed: bool,
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
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let address = StoredAddress::new(128);
    let len = (minimal.len() + extra).to_u64();
    let plan = if placed {
        ObjectHeaderAllocationGeometry::new(prefix, widths, &messages)
            .unwrap()
            .into_placed_allocation(iter::once(AddressRange::new(address, len).unwrap()))
    } else {
        ObjectHeaderAllocation::new(prefix, widths, address, len, &messages)
    }
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
    #[values(false, true)] separated: bool,
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
    let plan = if separated {
        ObjectHeaderAllocationGeometry::new(
            prefix,
            FormatWidths::from_sizes(8, 8).unwrap(),
            &records,
        )
        .unwrap()
        .into_placed_allocation(plan.chunks().enumerate().map(|(index, (_, len))| {
            AddressRange::new(StoredAddress::new(128 + index.to_u64() * 262_144), len).unwrap()
        }))
        .unwrap()
    } else {
        plan
    };
    let actual = plan.encode();
    let parsed = ObjectHeaderPrefix::parse(&actual).unwrap();
    assert_eq!(parsed.prefix, prefix);
    let end = plan
        .chunks()
        .map(|(address, len)| address.get() + len)
        .max()
        .unwrap();
    let mut image = vec![0; end.to_usize().unwrap()];
    for (address, bytes) in plan.encode_chunks() {
        let start = address.get().to_usize().unwrap();
        image[start..start + bytes.len()].copy_from_slice(&bytes);
    }
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

#[rstest]
#[case::plain(MessageRecordLayout::PLAIN, HeaderFlags::default())]
#[case::tracked(TRACKED, TRACKED_FLAGS)]
fn chosen_chunks_match_independent_bytes(
    #[case] layout: MessageRecordLayout,
    #[case] flags: HeaderFlags,
    #[values(false, true)] separated: bool,
    #[values(2, 4, 8)] width: u8,
    #[values(0, 1, 2, 3, 4, 5, 6, 17)] padding: usize,
) {
    let prefix = ObjectHeaderPrefix::with_layout(layout);
    let message = Message::new(RecordType::DATATYPE, &[5, 6])
        .with_flags(test_util::object_header::MessageFlags::CONSTANT)
        .with_creation_order(17);
    let mut source = independent::message_records(&[Message::nil(31), message.clone()], flags);
    source.extend([1, 2, 3]);
    let records_len = layout.prefix_len() + 2;
    let continuation = layout.prefix_len() + 2 * usize::from(width);
    let lengths = [
        (11 + records_len + continuation + padding).to_u64(),
        (8 + continuation + padding).to_u64(),
        (8 + layout.prefix_len()).to_u64(),
    ];
    let geometry = ObjectHeaderAllocationGeometry::new(
        prefix,
        FormatWidths::from_sizes(width, width).unwrap(),
        &source,
    )
    .unwrap();
    assert!(
        geometry
            .root_chunk_lengths(true)
            .any(|range| range.contains(&lengths[0]))
    );
    assert!(
        geometry
            .continuation_chunk_lengths(true)
            .contains(&lengths[1])
    );
    assert!(
        geometry
            .continuation_chunk_lengths(false)
            .contains(&lengths[2])
    );
    let addresses = if separated {
        [128, 1_024, 512]
    } else {
        [128, 128 + lengths[0], 128 + lengths[0] + lengths[1]]
    };
    let plan = if separated {
        geometry.into_placed_allocation(
            addresses
                .into_iter()
                .zip(lengths)
                .map(|(address, len)| AddressRange::new(StoredAddress::new(address), len).unwrap()),
        )
    } else {
        geometry.into_chunked_allocation(StoredAddress::new(128), lengths.iter().sum(), &lengths)
    }
    .unwrap();
    assert_eq!(
        plan.chunks().collect::<Vec<_>>(),
        addresses
            .into_iter()
            .zip(lengths)
            .map(|(address, len)| (StoredAddress::new(address), len))
            .collect::<Vec<_>>()
    );
    let chunks = plan.chunks().collect::<Vec<_>>();
    let mut expected = Vec::new();
    for (index, &(_, length)) in chunks.iter().enumerate() {
        let mut records = if index == 0 {
            vec![message.clone()]
        } else {
            Vec::new()
        };
        if index + 1 == chunks.len() {
            records.push(Message::nil(0));
        } else {
            if padding >= layout.prefix_len() {
                records.push(Message::nil(padding - layout.prefix_len()));
            }
            records.push(Message::continuation(
                chunks[index + 1].0.get(),
                chunks[index + 1].1,
                Widths::new(usize::from(width), usize::from(width)),
            ));
        }
        let mut chunk = if index == 0 {
            independent::Header::new()
                .flags(flags)
                .messages(records)
                .build()
        } else {
            independent::continuation_chunk(&records, flags)
        };
        if padding < layout.prefix_len() && index + 1 < chunks.len() {
            chunk.truncate(chunk.len() - CHECKSUM_LEN);
            chunk.resize(chunk.len() + padding, 0);
            if index == 0 {
                test_util::bytes::set_uint_at(&mut chunk, 6, 1, length - 11);
            }
            test_util::checksum::append(&mut chunk);
        }
        assert_eq!(chunk.len().to_u64(), length);
        expected.extend(chunk);
    }
    assert_eq!(plan.encode(), expected);
    let encoded = plan.encode_chunks().collect::<Vec<_>>();
    assert_eq!(
        encoded
            .iter()
            .flat_map(|(_, bytes)| bytes.iter())
            .copied()
            .collect::<Vec<_>>(),
        expected
    );
    let end = addresses
        .into_iter()
        .zip(lengths)
        .map(|(address, len)| address + len)
        .max()
        .unwrap();
    let mut image = vec![0xa5; end.to_usize().unwrap()];
    for (address, bytes) in encoded {
        let start = address.get().to_usize().unwrap();
        image[start..start + bytes.len()].copy_from_slice(&bytes);
    }
    if separated {
        assert!(
            image[(128 + lengths[0]).to_usize().unwrap()..512]
                .iter()
                .all(|&byte| byte == 0xa5)
        );
        assert!(
            image[(512 + lengths[2]).to_usize().unwrap()..1_024]
                .iter()
                .all(|&byte| byte == 0xa5)
        );
    }
    let header = ObjectHeader::parse(&image, AccessMode::ReadOnly, 128, width, width).unwrap();
    assert_eq!(header.messages.len(), 1);
    assert_eq!(header.messages[0].data, [5, 6]);
    assert_eq!(header.messages[0].flags, MessageFlags::CONSTANT);
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
#[case::above_host_width(1 << 35)]
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
