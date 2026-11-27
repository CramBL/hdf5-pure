use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::MessageType as RecordType;
use test_util::object_header::v2 as v2_bytes;

use super::*;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::object_header::MessageFilter;

#[test]
fn a_prefix_parses_to_the_fields_its_flags_select() {
    let flags = v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER
        | v2_bytes::HeaderFlags::INDEXES_CREATION_ORDER
        | v2_bytes::HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE;
    let header = v2_bytes::Header::new()
        .flags(flags)
        .timestamps(v2_bytes::Timestamps {
            access: 1,
            modification: 2,
            change: 3,
            birth: 4,
        })
        .message(Message::new(RecordType::DATASPACE, &[9]))
        .build();

    assert_eq!(
        ObjectHeaderPrefix::parse(&header),
        Ok(ParsedObjectHeaderPrefix {
            prefix: ObjectHeaderPrefix {
                layout: TRACKED,
                times: Some(TIMES),
                attribute_phase_change: Some(AttributePhaseChange {
                    max_compact: 8,
                    min_dense: 6,
                }),
            },
            flags: (flags | v2_bytes::HeaderFlags::STORES_TIMES).0,
            chunk0_size: 7,
            len: 27,
        })
    );
}

#[rstest]
fn an_encoded_header_parses_back_to_its_prefix_and_messages(
    #[values(
        MessageRecordLayout::PLAIN,
        MessageRecordLayout::from_header_flags(0x0C)
    )]
    layout: MessageRecordLayout,
    #[values(None, Some(TIMES))] times: Option<ObjectTimes>,
    #[values(None, Some(PHASE))] attribute_phase_change: Option<AttributePhaseChange>,
) {
    let prefix = ObjectHeaderPrefix {
        layout,
        times,
        attribute_phase_change,
    };
    let flags = if layout.tracks_creation_order() {
        v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER
    } else {
        v2_bytes::HeaderFlags::default()
    };
    let records = v2_bytes::message_record(&Message::new(RecordType::DATASPACE, &[1, 2, 3]), flags);
    let header = prefix.encode_header(&records);

    let parsed = ObjectHeaderPrefix::parse(&header).unwrap();
    assert_eq!(parsed.prefix, prefix);
    assert_eq!(parsed.chunk0_size, records.len().to_u64());
    let parsed_header =
        ObjectHeader::parse(&header, crate::access_mode::AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(parsed_header.prefix, Some(prefix));
    assert_eq!(parsed_header.messages.len(), 1);
    assert_eq!(parsed_header.messages[0].data, vec![1, 2, 3]);
    let filtered = ObjectHeader::parse_from_source_filtered(
        header.as_slice(),
        crate::access_mode::AccessMode::ReadOnly,
        0,
        8,
        8,
        BaseAddress::ZERO,
        MessageFilter::Only(&mut |_, _| false),
    )
    .unwrap();
    assert_eq!(filtered.prefix, Some(prefix));
    assert!(filtered.messages.is_empty());
}

#[test]
fn an_encoded_header_matches_the_bytes_of_an_independent_builder() {
    let message = Message::new(RecordType::DATATYPE, &[5, 6])
        .with_flags(test_util::object_header::MessageFlags::CONSTANT)
        .with_creation_order(17);
    let expected = v2_bytes::Header::new()
        .timestamps(v2_bytes::Timestamps {
            access: 1,
            modification: 2,
            change: 3,
            birth: 4,
        })
        .message(message.clone())
        .build();
    let prefix = ObjectHeaderPrefix {
        times: Some(TIMES),
        ..ObjectHeaderPrefix::PLAIN
    };
    let records = v2_bytes::message_records(&[message], v2_bytes::HeaderFlags::default());

    assert_eq!(prefix.encode_header(&records), expected);
}

#[rstest]
#[case::inside_the_signature(3, FormatError::UnexpectedEof { expected: 6, available: 3 })]
#[case::inside_the_times(10, FormatError::UnexpectedEof { expected: 22, available: 10 })]
#[case::inside_the_phase_change(24, FormatError::UnexpectedEof { expected: 26, available: 24 })]
#[case::before_the_chunk_size(26, FormatError::UnexpectedEof { expected: 27, available: 26 })]
fn a_truncated_prefix_is_unexpected_eof(#[case] cut: usize, #[case] expected: FormatError) {
    let header = ObjectHeaderPrefix {
        times: Some(TIMES),
        attribute_phase_change: Some(PHASE),
        ..ObjectHeaderPrefix::PLAIN
    }
    .encode_header(&[]);
    assert_eq!(ObjectHeaderPrefix::parse(&header[..cut]), Err(expected));
}

#[rstest]
#[case::another_signature(0, b'X', FormatError::InvalidObjectHeaderSignature)]
#[case::version_1(4, 1, FormatError::InvalidObjectHeaderVersion(1))]
fn a_prefix_that_is_not_version_2_is_rejected(
    #[case] at: usize,
    #[case] byte: u8,
    #[case] expected: FormatError,
) {
    let mut header = ObjectHeaderPrefix::PLAIN.encode_header(&[]);
    header[at] = byte;
    assert_eq!(ObjectHeaderPrefix::parse(&header), Err(expected));
}

#[rstest]
#[case::neither_block(ObjectHeaderPrefix::PLAIN, 0, 11)]
#[case::the_times(
    ObjectHeaderPrefix { times: Some(TIMES), ..ObjectHeaderPrefix::PLAIN },
    v2_bytes::HeaderFlags::STORES_TIMES.0,
    11 + 16
)]
#[case::the_phase_change(
    ObjectHeaderPrefix { attribute_phase_change: Some(PHASE), ..ObjectHeaderPrefix::PLAIN },
    v2_bytes::HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE.0,
    11 + 4
)]
fn an_optional_block_sets_its_own_flag_bit(
    #[case] prefix: ObjectHeaderPrefix,
    #[case] flags: u8,
    #[case] header_len: usize,
) {
    let header = prefix.encode_header(&[]);
    assert_eq!((header[5], header.len()), (flags, header_len));
}

const TRACKED: MessageRecordLayout = MessageRecordLayout::from_header_flags(
    v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0
        | v2_bytes::HeaderFlags::INDEXES_CREATION_ORDER.0,
);

const TIMES: ObjectTimes = ObjectTimes {
    access: 1,
    modification: 2,
    change: 3,
    birth: 4,
};

const PHASE: AttributePhaseChange = AttributePhaseChange {
    max_compact: 32,
    min_dense: 24,
};
