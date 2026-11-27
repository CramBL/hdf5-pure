use alloc::vec::Vec;

use rstest::rstest;
use test_util::image::Image;
use test_util::object_header::v1 as v1_bytes;
use test_util::object_header::v2 as v2_bytes;
use test_util::object_header::{Message, MessageFlags as RecordFlags, MessageType as RecordType};

use super::support;
use super::support::{
    CONTINUATION_OFFSET, ContinuationTree, Driver, LENGTH_SIZE, OFFSET_SIZE, WIDTHS,
};
use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::{MessageFilter, ObjectHeader};

/// A chunk of only Nil padding reserves nothing.
///
/// The reservation is sized from a walk of the chunk. A `HeaderMessage` is
/// much wider than the four-byte message prefix, so skipped Nil messages are
/// excluded from the reservation count.
#[test]
fn a_chunk_of_padding_reserves_no_messages() {
    let data = v2_bytes::Header::new()
        .flags(v2_bytes::HeaderFlags(0x01))
        .messages((0..256).map(|_| Message::nil(0)))
        .build();
    let header = support::parse(&data).unwrap();

    assert!(header.messages.is_empty(), "Nil messages are not kept");
    assert_eq!(
        header.messages.capacity(),
        0,
        "a chunk the parse keeps nothing from must not reserve for it"
    );
}

#[test]
fn parse_v1_unknown_message_ok() {
    let data = v1_bytes::Header::new()
        .message(Message::new(RecordType::UNKNOWN, &UNKNOWN_BODY))
        .build();
    let header = support::parse(&data).unwrap();

    assert_eq!(header.messages.len(), 1);
    assert_eq!(
        header.messages[0].msg_type.unknown_id(),
        Some(RecordType::UNKNOWN.0)
    );
}

/// The must-understand guard applies only to message types the parser cannot
/// name. External Data Files (`0x0007`) is a known type, so the header parses
/// and the reader reports external storage when the message is used.
///
/// The reference library writes flags `0x01` on this message. A crafted
/// `0x08` flag exercises the guard.
#[test]
fn parse_v1_named_message_survives_must_understand() {
    let data = v1_bytes::Header::new()
        .message(
            Message::new(RecordType::EXTERNAL_DATA_FILES, &UNKNOWN_BODY)
                .with_flags(RecordFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE),
        )
        .build();
    let header = support::parse_for_mode(&data, AccessMode::ReadWrite).unwrap();

    assert_eq!(header.messages.len(), 1);
    assert_eq!(
        header.messages[0].msg_type,
        MessageType::EXTERNAL_DATA_FILES
    );
}

#[rstest]
#[case::v1_header(UnknownMessageLocation::V1Header)]
#[case::v1_continuation(UnknownMessageLocation::V1Continuation)]
#[case::v2_header(UnknownMessageLocation::V2Header)]
fn unknown_message_that_must_always_be_understood_is_rejected(
    #[case] location: UnknownMessageLocation,
) {
    let data = unknown_message_data(location, RecordFlags::FAIL_IF_UNKNOWN_ALWAYS);

    assert_eq!(
        support::parse(&data).unwrap_err(),
        FormatError::UnsupportedMessage(RecordType::UNKNOWN.0),
        "buffered {location:?}"
    );
    assert_eq!(
        support::parse_from_source_for_mode(&data, AccessMode::ReadOnly).unwrap_err(),
        FormatError::UnsupportedMessage(RecordType::UNKNOWN.0),
        "streamed {location:?}"
    );
}

#[rstest]
#[case::v1_header(UnknownMessageLocation::V1Header)]
#[case::v1_continuation(UnknownMessageLocation::V1Continuation)]
#[case::v2_header(UnknownMessageLocation::V2Header)]
fn unknown_message_that_a_writer_must_understand_is_rejected_only_for_write(
    #[case] location: UnknownMessageLocation,
) {
    let data = unknown_message_data(location, RecordFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE);

    let buffered = support::parse_for_mode(&data, AccessMode::ReadOnly).unwrap();
    let streamed = support::parse_from_source_for_mode(&data, AccessMode::ReadOnly).unwrap();

    for (parser, header) in [("buffered", buffered), ("streamed", streamed)] {
        match location {
            UnknownMessageLocation::V1Continuation => {
                assert_eq!(header.messages.len(), 2, "{parser}");
                assert_eq!(
                    header.messages[0].msg_type,
                    MessageType::OBJECT_HEADER_CONTINUATION,
                    "{parser}"
                );
                assert_eq!(
                    header.messages[1].msg_type.unknown_id(),
                    Some(RecordType::UNKNOWN.0),
                    "{parser}"
                );
                assert_eq!(
                    header.messages[1].data.as_slice(),
                    &UNKNOWN_BODY,
                    "{parser}"
                );
            }
            UnknownMessageLocation::V1Header => {
                assert_eq!(header.messages.len(), 1, "{parser}");
                assert_eq!(
                    header.messages[0].msg_type.unknown_id(),
                    Some(RecordType::UNKNOWN.0),
                    "{parser}"
                );
                assert_eq!(
                    header.messages[0].data.as_slice(),
                    &UNKNOWN_BODY,
                    "{parser}"
                );
            }
            UnknownMessageLocation::V2Header => {
                assert_eq!(header.messages.len(), 1, "{parser}");
                assert_eq!(
                    header.messages[0].msg_type.unknown_id(),
                    Some(RecordType::UNKNOWN.0),
                    "{parser}"
                );
                assert_eq!(header.messages[0].data.as_slice(), &[0xAA], "{parser}");
            }
        }
    }

    assert_eq!(
        support::parse_for_mode(&data, AccessMode::ReadWrite).unwrap_err(),
        FormatError::UnsupportedMessage(RecordType::UNKNOWN.0),
        "buffered"
    );
    assert_eq!(
        support::parse_from_source_for_mode(&data, AccessMode::ReadWrite).unwrap_err(),
        FormatError::UnsupportedMessage(RecordType::UNKNOWN.0),
        "streamed"
    );
}

#[test]
fn parse_v2_nil_padding_skipped() {
    let data = v2_bytes::Header::new()
        .message(Message::nil(4))
        .message(Message::new(RecordType::DATASPACE, &[42]))
        .build();
    let header = support::parse(&data).unwrap();

    assert_eq!(header.messages.len(), 1);
    assert_eq!(header.messages[0].msg_type, MessageType::DATASPACE);
}

#[rstest]
fn a_filter_that_drops_a_blocks_messages_visits_the_blocks_it_refers_to(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let mut drop_a = |_: MessageType, body: &[u8]| body != b"a";
    let header = driver
        .parse(
            &ContinuationTree::BothNested.image(),
            MessageFilter::Only(&mut drop_a),
        )
        .unwrap();

    assert_eq!(support::bodies(&header), [b"b".as_slice(), b"a1", b"b1"]);
}

#[rstest]
fn a_short_version_2_continuation_is_rejected_under_a_filter_that_keeps_nothing(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    // The body of a continuation message is 16 bytes at 8-byte offsets and lengths.
    let data = v2_bytes::Header::new()
        .message(support::named(b"a"))
        .message(Message::new(
            RecordType::OBJECT_HEADER_CONTINUATION,
            &[0; 15],
        ))
        .build();
    let mut keep_nothing = |_: MessageType, _: &[u8]| false;

    assert_eq!(
        driver
            .parse(&data, MessageFilter::Only(&mut keep_nothing))
            .unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 16,
            available: 15
        }
    );
}

#[test]
fn a_filtered_version_1_parse_follows_unretained_continuations() {
    let nested_offset = 512;
    let nested_data = [0xDE, 0xAD, 0, 0, 0, 0, 0, 0];
    let nested_chunk = support::v1_datatype_chunk(&nested_data);
    let continuation_chunk = v1_bytes::message_record(&Message::continuation(
        nested_offset as u64,
        nested_chunk.len() as u64,
        WIDTHS,
    ));
    let mut image = Image::starting_with(
        &v1_bytes::Header::new()
            .continuation(CONTINUATION_OFFSET, &continuation_chunk, WIDTHS)
            .build(),
    );
    image.place(CONTINUATION_OFFSET, &continuation_chunk);
    image.place(nested_offset, &nested_chunk);
    let data = image.build();

    let mut keep_datatype = |msg_type: MessageType, _: &[u8]| msg_type == MessageType::DATATYPE;
    let buffered = ObjectHeader::parse_filtered(
        &data,
        AccessMode::ReadOnly,
        0,
        OFFSET_SIZE,
        LENGTH_SIZE,
        BaseAddress::ZERO,
        MessageFilter::Only(&mut keep_datatype),
    )
    .unwrap();

    let mut keep_datatype = |msg_type: MessageType, _: &[u8]| msg_type == MessageType::DATATYPE;
    let streamed = ObjectHeader::parse_from_source_filtered(
        data.as_slice(),
        AccessMode::ReadOnly,
        0,
        OFFSET_SIZE,
        LENGTH_SIZE,
        BaseAddress::ZERO,
        MessageFilter::Only(&mut keep_datatype),
    )
    .unwrap();

    for header in [&buffered, &streamed] {
        assert_eq!(header.messages.len(), 1);
        support::assert_message(&header.messages[0], MessageType::DATATYPE, &nested_data);
    }
}

#[rstest]
#[case::v1_header(UnknownMessageLocation::V1Header)]
#[case::v2_header(UnknownMessageLocation::V2Header)]
fn a_filtered_parse_checks_must_understand_before_filtering(
    #[case] location: UnknownMessageLocation,
) {
    let data = unknown_message_data(location, RecordFlags::FAIL_IF_UNKNOWN_ALWAYS);

    let mut buffered_filter_called = false;
    let buffered_error = {
        let mut drop_all = |_: MessageType, _: &[u8]| {
            buffered_filter_called = true;
            false
        };
        ObjectHeader::parse_filtered(
            &data,
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
            MessageFilter::Only(&mut drop_all),
        )
    }
    .unwrap_err();

    let mut streamed_filter_called = false;
    let streamed_error = {
        let mut drop_all = |_: MessageType, _: &[u8]| {
            streamed_filter_called = true;
            false
        };
        ObjectHeader::parse_from_source_filtered(
            data.as_slice(),
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
            MessageFilter::Only(&mut drop_all),
        )
    }
    .unwrap_err();

    assert_eq!(
        buffered_error,
        FormatError::UnsupportedMessage(RecordType::UNKNOWN.0)
    );
    assert_eq!(
        streamed_error,
        FormatError::UnsupportedMessage(RecordType::UNKNOWN.0)
    );
    assert!(
        !buffered_filter_called,
        "buffered parser filtered before must-understand validation"
    );
    assert!(
        !streamed_filter_called,
        "streamed parser filtered before must-understand validation"
    );
}

#[derive(Clone, Copy, Debug)]
enum UnknownMessageLocation {
    V1Header,
    V1Continuation,
    V2Header,
}

fn unknown_message_data(location: UnknownMessageLocation, flags: RecordFlags) -> Vec<u8> {
    let unknown = Message::new(RecordType::UNKNOWN, &UNKNOWN_BODY).with_flags(flags);
    match location {
        UnknownMessageLocation::V1Header => v1_bytes::Header::new().message(unknown).build(),
        UnknownMessageLocation::V1Continuation => {
            let continuation = v1_bytes::message_record(&unknown);
            let mut image = Image::starting_with(
                &v1_bytes::Header::new()
                    .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
                    .build(),
            );
            image.place(CONTINUATION_OFFSET, &continuation);
            image.build()
        }
        UnknownMessageLocation::V2Header => v2_bytes::Header::new()
            .message(Message::new(RecordType::UNKNOWN, &[0xAA]).with_flags(flags))
            .build(),
    }
}

const UNKNOWN_BODY: [u8; 8] = [0xAA, 0, 0, 0, 0, 0, 0, 0];
