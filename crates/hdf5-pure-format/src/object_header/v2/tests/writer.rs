use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags as FixtureFlags;
use test_util::object_header::MessageType as FixtureType;
use test_util::object_header::v2 as v2_bytes;

use super::*;
use crate::access_mode::AccessMode;
use crate::error::OBJECT_HEADER_MESSAGE_MAX;
use crate::object_header::AttributePhaseChange;
use crate::object_header::MessageRecordLayout;
use crate::object_header::ObjectHeader;
use crate::object_header::ObjectTimes;

#[rstest]
fn supplied_prefix_and_record_metadata_match_independent_bytes(
    #[values(
        v2_bytes::HeaderFlags::default(),
        v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER,
        v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER | v2_bytes::HeaderFlags::INDEXES_CREATION_ORDER
    )]
    flags: v2_bytes::HeaderFlags,
    #[values(false, true)] times: bool,
    #[values(false, true)] phase_change: bool,
) {
    let prefix = ObjectHeaderPrefix {
        layout: MessageRecordLayout::from_header_flags(flags.0),
        times: times.then_some(ObjectTimes {
            access: 1,
            modification: 2,
            change: 3,
            birth: 4,
        }),
        attribute_phase_change: phase_change.then_some(AttributePhaseChange {
            max_compact: 32,
            min_dense: 24,
        }),
    };
    let mut expected = v2_bytes::Header::new().flags(flags);
    if times {
        expected = expected.timestamps(v2_bytes::Timestamps {
            access: 1,
            modification: 2,
            change: 3,
            birth: 4,
        });
    }
    if phase_change {
        expected = expected.phase_change(32, 24);
    }
    let expected = expected
        .message(
            Message::new(FixtureType::ATTRIBUTE, &[7, 11])
                .with_flags(FixtureFlags::CONSTANT)
                .with_creation_order(17),
        )
        .message(Message::new(FixtureType::ATTRIBUTE, &[19]).with_creation_order(31))
        .build();
    let first = prefix
        .layout
        .prepare_raw_record(MessageType::ATTRIBUTE, MessageFlags::CONSTANT, 17, &[7, 11])
        .unwrap();
    let second = prefix
        .layout
        .prepare_raw_record(MessageType::ATTRIBUTE, MessageFlags::NONE, 31, &[19])
        .unwrap();
    let writer = ObjectHeaderWriter::with_prepared_records(
        prefix,
        &[Some(first.as_record()), None, Some(second.as_record())],
    )
    .unwrap();
    let bytes = writer.serialize().unwrap();
    assert_eq!(bytes, expected);
    let parsed = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(parsed.prefix, Some(prefix));
    assert_eq!(parsed.messages.len(), 2);
    assert_eq!(
        parsed.messages[0].creation_order,
        flags.tracks_creation_order().then_some(17)
    );
    assert_eq!(
        parsed.messages[1].creation_order,
        flags.tracks_creation_order().then_some(31)
    );
}

#[test]
fn prepared_records_reject_different_creation_index_presence() {
    let tracked = ObjectHeaderPrefix {
        layout: MessageRecordLayout::from_header_flags(
            v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0,
        ),
        ..ObjectHeaderPrefix::PLAIN
    };
    let record = tracked
        .layout
        .prepare_raw_record(MessageType::ATTRIBUTE, MessageFlags::NONE, 1, &[7])
        .unwrap();
    let error = ObjectHeaderWriter::with_prepared_records(
        ObjectHeaderPrefix::PLAIN,
        &[Some(record.as_record())],
    )
    .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal(
            "prepared record and header disagree on creation index presence".into()
        )
    );
}

#[test]
fn empty_header_roundtrip() {
    let writer = ObjectHeaderWriter::new();
    let bytes = writer.serialize().unwrap();
    let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(hdr.version, 2);
    assert_eq!(hdr.messages.len(), 0);
}

#[test]
fn two_messages_roundtrip() {
    let mut writer = ObjectHeaderWriter::new();
    writer
        .append_raw_message(MessageType::DATASPACE, &[1, 2, 3, 4])
        .unwrap();
    writer
        .append_raw_message(MessageType::DATATYPE, &[5, 6])
        .unwrap();
    let bytes = writer.serialize().unwrap();
    let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(hdr.messages.len(), 2);
    assert_eq!(hdr.messages[0].msg_type, MessageType::DATASPACE);
    assert_eq!(hdr.messages[0].data, vec![1, 2, 3, 4]);
    assert_eq!(hdr.messages[1].msg_type, MessageType::DATATYPE);
    assert_eq!(hdr.messages[1].data, vec![5, 6]);
}

#[test]
fn a_serialized_record_stores_the_flags_the_message_was_added_with() {
    let mut writer = ObjectHeaderWriter::new();
    writer
        .append_raw_message_with_flags(
            MessageType::DATATYPE,
            &[5, 6],
            MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
        )
        .unwrap();
    let bytes = writer.serialize().unwrap();
    let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(
        hdr.messages[0].flags,
        MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING
    );
}

#[test]
fn a_message_added_without_flags_sets_no_flag_in_its_record() {
    let mut writer = ObjectHeaderWriter::new();
    writer
        .append_raw_message(MessageType::DATATYPE, &[5, 6])
        .unwrap();
    let bytes = writer.serialize().unwrap();
    let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(hdr.messages[0].flags, MessageFlags::NONE);
}

#[test]
fn large_header_uses_2byte_chunk_size() {
    let mut writer = ObjectHeaderWriter::new();
    // Add a message with >255 bytes of payload
    writer
        .append_raw_message(MessageType::DATATYPE, &[0xAA; 300])
        .unwrap();
    let bytes = writer.serialize().unwrap();
    let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(hdr.messages.len(), 1);
    assert_eq!(hdr.messages[0].data.len(), 300);
}

#[rstest]
fn message_at_the_size_field_limit_serializes(
    #[values(
        ObjectHeaderPrefix::PLAIN,
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0))
    )]
    prefix: ObjectHeaderPrefix,
) {
    let mut writer = ObjectHeaderWriter::with_prefix(prefix);
    writer
        .append_raw_message(
            MessageType::ATTRIBUTE,
            &vec![0xAA; OBJECT_HEADER_MESSAGE_MAX],
        )
        .unwrap();
    let bytes = writer.serialize().unwrap();
    let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(hdr.messages.len(), 1);
    assert_eq!(hdr.messages[0].data.len(), OBJECT_HEADER_MESSAGE_MAX);
}

#[rstest]
#[case::oversized_body(MessageType::ATTRIBUTE, FormatError::ObjectHeaderMessageTooLarge {
    message_type: MessageType::ATTRIBUTE.to_u16(),
    size: OBJECT_HEADER_MESSAGE_MAX + 1,
})]
#[case::wide_type(
    MessageType::from_u16(u16::MAX),
    FormatError::ObjectHeaderMessageTypeTooLarge(u16::MAX)
)]
fn writer_preserves_existing_records_after_a_failed_append(
    #[case] message_type: MessageType,
    #[case] error: FormatError,
) {
    let pad = [0xa5];
    let mut writer = ObjectHeaderWriter::new();
    writer.append_raw_message(MessageType::NIL, &pad).unwrap();
    assert_eq!(
        writer.append_raw_message(message_type, &vec![0xaa; OBJECT_HEADER_MESSAGE_MAX + 1]),
        Err(error)
    );
    writer.append_raw_message(MessageType::NIL, &pad).unwrap();
    let expected = v2_bytes::Header::new()
        .message(Message::new(FixtureType::NIL, &pad))
        .message(Message::new(FixtureType::NIL, &pad))
        .build();
    assert_eq!(writer.serialize().unwrap(), expected);
}
