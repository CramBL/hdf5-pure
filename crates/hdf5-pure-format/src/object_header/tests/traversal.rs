use rstest::rstest;
use test_util::image::Image;
use test_util::object_header::v1 as v1_bytes;
use test_util::object_header::v2 as v2_bytes;
use test_util::object_header::{Message, MessageType as RecordType};
use test_util::widths::Widths;

use super::support;
use super::support::{
    BLOCK_PAST_THE_HEADER, CONTINUATION_OFFSET, ContinuationTree, Driver, LENGTH_SIZE, WIDTHS,
};
use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::{MessageFilter, ObjectHeader};
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
#[case::eight_byte_widths(OffsetWidth::Eight, LengthWidth::Eight)]
#[case::mixed_widths(OffsetWidth::Two, LengthWidth::Four)]
fn parse_v2_continuation(#[case] offset_width: OffsetWidth, #[case] length_width: LengthWidth) {
    let widths = FormatWidths {
        offsets: offset_width,
        lengths: length_width,
    };
    let encoded_widths = Widths::new(
        usize::from(offset_width.get()),
        usize::from(length_width.get()),
    );
    let continuation = support::v2_continuation(&[0xDE, 0xAD]);
    let mut image = Image::starting_with(
        &v2_bytes::Header::new()
            .message(Message::new(RecordType::DATASPACE, &[42]))
            .continuation(CONTINUATION_OFFSET, &continuation, encoded_widths)
            .build(),
    );
    image.place(CONTINUATION_OFFSET, &continuation);
    let data = image.build();
    let header = ObjectHeader::parse_with_widths(&data, AccessMode::ReadOnly, 0, widths).unwrap();

    let streamed = ObjectHeader::parse_from_source_with_widths(
        data.as_slice(),
        AccessMode::ReadOnly,
        0,
        widths,
        BaseAddress::ZERO,
    )
    .unwrap();

    assert_eq!(header.messages.len(), 2);
    assert_eq!(header.messages[0].msg_type, MessageType::DATASPACE);
    support::assert_message(&header.messages[1], MessageType::DATATYPE, &[0xDE, 0xAD]);
    assert_eq!(streamed.messages.len(), 2);
    assert_eq!(streamed.messages[0].msg_type, MessageType::DATASPACE);
    support::assert_message(&streamed.messages[1], MessageType::DATATYPE, &[0xDE, 0xAD]);
}

#[rstest]
#[case::empty(&[], FormatError::UnexpectedEof { expected: 16, available: 0 })]
#[case::the_address_alone(&[0; 8], FormatError::UnexpectedEof { expected: 16, available: 8 })]
#[case::one_byte_short(&[0; 15], FormatError::UnexpectedEof { expected: 16, available: 15 })]
fn a_v2_continuation_message_too_short_for_its_fields_is_rejected(
    #[case] body: &[u8],
    #[case] expected: FormatError,
) {
    let data = v2_bytes::Header::new()
        .message(Message::new(RecordType::OBJECT_HEADER_CONTINUATION, body))
        .build();

    assert_eq!(support::parse(&data).unwrap_err(), expected);
    assert_eq!(
        support::parse_from_source_for_mode(&data, AccessMode::ReadOnly).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::two_byte_addresses(2)]
#[case::four_byte_addresses(4)]
#[case::eight_byte_addresses(8)]
fn a_v2_continuation_to_the_undefined_address_is_rejected(#[case] offset_size: u8) {
    let widths = Widths::new(usize::from(offset_size), usize::from(LENGTH_SIZE));
    let data = v2_bytes::Header::new()
        .message(Message::undefined_continuation(64, widths))
        .build();

    assert_eq!(
        ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, offset_size, LENGTH_SIZE).unwrap_err(),
        FormatError::UndefinedContinuationAddress
    );
    assert_eq!(
        ObjectHeader::parse_from_source(
            &data[..],
            AccessMode::ReadOnly,
            0,
            offset_size,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        )
        .unwrap_err(),
        FormatError::UndefinedContinuationAddress
    );
}

#[rstest]
#[case::siblings(ContinuationTree::Siblings, &[b"a".as_slice(), b"b"])]
#[case::a_block_found_while_visiting(ContinuationTree::Nested, &[b"a".as_slice(), b"b", b"a1"])]
fn version_2_continuation_blocks_are_visited_in_the_order_they_are_found(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
    #[case] tree: ContinuationTree,
    #[case] expected: &[&[u8]],
) {
    let header = driver.parse(&tree.image(), MessageFilter::All).unwrap();

    assert_eq!(support::bodies(&header), expected);
}

#[rstest]
#[case::at_the_limit(256, Ok(256))]
#[case::past_the_limit(257, Err(FormatError::NestingDepthExceeded))]
fn a_version_2_header_parse_accepts_at_most_256_continuation_messages(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
    #[case] count: usize,
    #[case] expected: Result<usize, FormatError>,
) {
    // Every continuation message refers to the same block, which the parse reads once per message.
    let block = support::continuation_block(&[support::named(b"a")]);
    let data = support::continuation_image(
        v2_bytes::Header::new()
            .flags(v2_bytes::HeaderFlags(0x01))
            .messages((0..count).map(|_| support::continuation_to(BLOCK_PAST_THE_HEADER, &block))),
        &[(BLOCK_PAST_THE_HEADER, &block)],
    );

    assert_eq!(
        driver
            .parse(&data, MessageFilter::All)
            .map(|header| header.messages.len()),
        expected
    );
}

#[test]
fn streaming_v2_simple_matches_buffered() {
    let data = v2_bytes::Header::new()
        .timestamps(v2_bytes::Timestamps {
            access: 1,
            modification: 2,
            change: 3,
            birth: 4,
        })
        .message(Message::new(RecordType::DATASPACE, &[1, 2, 3]))
        .build();

    support::assert_all_parse_paths_match(&data);
}

#[test]
fn streaming_v2_with_continuation_matches_buffered() {
    let continuation = support::v2_continuation(&[0xDE, 0xAD]);
    let mut image = Image::starting_with(
        &v2_bytes::Header::new()
            .message(Message::new(RecordType::DATASPACE, &[42]))
            .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
            .build(),
    );
    image.place(CONTINUATION_OFFSET, &continuation);

    support::assert_all_parse_paths_match(image.as_bytes());
}

#[test]
fn streaming_v1_with_continuation_matches_buffered() {
    let continuation = support::v1_datatype_chunk(&[0xBE, 0xEF, 0, 0, 0, 0, 0, 0]);
    let mut image = Image::starting_with(
        &v1_bytes::Header::new()
            .message(Message::new(
                RecordType::DATASPACE,
                &[42, 0, 0, 0, 0, 0, 0, 0],
            ))
            .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
            .build(),
    );
    image.place(CONTINUATION_OFFSET, &continuation);

    support::assert_all_parse_paths_match(image.as_bytes());
}

#[test]
fn version_1_message_overrunning_chunk0_is_rejected() {
    let continuation = support::v1_datatype_chunk(&[0xBE, 0xEF, 0, 0, 0, 0, 0, 0]);
    let mut image = Image::starting_with(
        &v1_bytes::Header::new()
            .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
            .declared_data_size(16)
            .build(),
    );
    image.place(CONTINUATION_OFFSET, &continuation);

    support::assert_unexpected_eof_all_parse_paths(image.as_bytes(), 24, 16);
}

#[test]
fn version_1_continuation_with_partial_message_prefix_is_rejected() {
    let mut continuation = support::v1_datatype_chunk(&[0xAB; 8]);

    // One trailing byte cannot form the eight-byte prefix of another v1 message.
    continuation.push(0);

    let mut image = Image::starting_with(
        &v1_bytes::Header::new()
            .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
            .build(),
    );
    image.place(CONTINUATION_OFFSET, &continuation);

    support::assert_unexpected_eof_all_parse_paths(image.as_bytes(), 8, 1);
}

#[test]
fn version_1_continuation_message_with_unaligned_size_is_rejected() {
    let continuation = support::v1_datatype_chunk(&[0xAB; 7]);

    let mut image = Image::starting_with(
        &v1_bytes::Header::new()
            .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
            .build(),
    );
    image.place(CONTINUATION_OFFSET, &continuation);

    support::assert_invalid_v1_message_size_all_parse_paths(image.as_bytes(), 7);
}
