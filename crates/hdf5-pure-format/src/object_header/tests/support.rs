use alloc::vec::Vec;

use test_util::image::Image;
use test_util::object_header::v1 as v1_bytes;
use test_util::object_header::v2 as v2_bytes;
use test_util::object_header::{Message, MessageType as RecordType};
use test_util::widths::Widths;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::{HeaderMessage, MessageFilter, ObjectHeader};

/// A version 1 continuation chunk of one Datatype message carrying `data`.
pub(super) fn v1_datatype_chunk(data: &[u8]) -> Vec<u8> {
    v1_bytes::message_record(&Message::new(RecordType::DATATYPE, data))
}

/// A version 2 continuation chunk of one Datatype message carrying `data`.
pub(super) fn v2_continuation(data: &[u8]) -> Vec<u8> {
    v2_bytes::continuation_chunk(
        &[Message::new(RecordType::DATATYPE, data)],
        v2_bytes::HeaderFlags::default(),
    )
}

/// The two entry points of the parser: a buffer that holds the whole file, and a
/// [`MetadataSource`](crate::metadata_source::MetadataSource).
#[derive(Clone, Copy, Debug)]
pub(super) enum Driver {
    Buffered,
    Source,
}

impl Driver {
    pub(super) fn parse(
        self,
        data: &[u8],
        filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        match self {
            Driver::Buffered => ObjectHeader::parse_filtered(
                data,
                AccessMode::ReadOnly,
                0,
                OFFSET_SIZE,
                LENGTH_SIZE,
                BaseAddress::ZERO,
                filter,
            ),
            Driver::Source => ObjectHeader::parse_from_source_filtered(
                data,
                AccessMode::ReadOnly,
                0,
                OFFSET_SIZE,
                LENGTH_SIZE,
                BaseAddress::ZERO,
                filter,
            ),
        }
    }
}

/// A version 2 header whose chunk 0 refers to two continuation blocks, `a` and then `b`.
///
/// Each block stores one Datatype message whose body is the name of the block.
#[derive(Clone, Copy, Debug)]
pub(super) enum ContinuationTree {
    /// Block `a` refers to block `a1`, and block `b` to block `b1`.
    BothNested,
    /// Block `a` refers to block `a1`.
    Nested,
    /// Neither block refers to a further block.
    Siblings,
}

impl ContinuationTree {
    pub(super) fn image(self) -> Vec<u8> {
        let a1 = continuation_block(&[named(b"a1")]);
        let b1 = continuation_block(&[named(b"b1")]);
        let (a, b) = match self {
            ContinuationTree::BothNested => (
                continuation_block(&[named(b"a"), continuation_to(BLOCK_A1, &a1)]),
                continuation_block(&[named(b"b"), continuation_to(BLOCK_B1, &b1)]),
            ),
            ContinuationTree::Nested => (
                continuation_block(&[named(b"a"), continuation_to(BLOCK_A1, &a1)]),
                continuation_block(&[named(b"b")]),
            ),
            ContinuationTree::Siblings => (
                continuation_block(&[named(b"a")]),
                continuation_block(&[named(b"b")]),
            ),
        };
        continuation_image(
            v2_bytes::Header::new()
                .continuation(BLOCK_A, &a, WIDTHS)
                .continuation(BLOCK_B, &b, WIDTHS),
            &[
                (BLOCK_A, &a),
                (BLOCK_B, &b),
                (BLOCK_A1, &a1),
                (BLOCK_B1, &b1),
            ],
        )
    }
}

/// Returns the bytes of `header`, with each block of `blocks` placed at its offset.
pub(super) fn continuation_image(header: v2_bytes::Header, blocks: &[(usize, &[u8])]) -> Vec<u8> {
    let mut image = Image::starting_with(&header.build());
    for &(at, block) in blocks {
        image.place(at, block);
    }
    image.build()
}

pub(super) fn continuation_block(messages: &[Message]) -> Vec<u8> {
    v2_bytes::continuation_chunk(messages, v2_bytes::HeaderFlags::default())
}

pub(super) fn continuation_to(at: usize, block: &[u8]) -> Message {
    Message::continuation(at as u64, block.len() as u64, WIDTHS)
}

/// Returns a Datatype message whose body is `body`, the name [`bodies`] lists it by.
pub(super) fn named(body: &[u8]) -> Message {
    Message::new(RecordType::DATATYPE, body)
}

pub(super) fn bodies(header: &ObjectHeader) -> Vec<&[u8]> {
    header
        .messages
        .iter()
        .map(|message| message.data.as_slice())
        .collect()
}

pub(super) fn assert_message(message: &HeaderMessage, msg_type: MessageType, data: &[u8]) {
    assert_eq!(message.msg_type, msg_type);
    assert_eq!(message.data.as_slice(), data);
}

pub(super) fn assert_same_header(actual: &ObjectHeader, expected: &ObjectHeader) {
    assert_eq!(actual.version, expected.version);
    assert_eq!(actual.reference_count, expected.reference_count);
    assert_eq!(actual.flags, expected.flags);
    assert_eq!(actual.access_time, expected.access_time);
    assert_eq!(actual.modification_time, expected.modification_time);
    assert_eq!(actual.change_time, expected.change_time);
    assert_eq!(actual.birth_time, expected.birth_time);
    assert_eq!(
        actual.messages.len(),
        expected.messages.len(),
        "message count"
    );

    for (index, (actual, expected)) in actual.messages.iter().zip(&expected.messages).enumerate() {
        assert_eq!(actual.msg_type, expected.msg_type, "msg {index} type");
        assert_eq!(actual.size, expected.size, "msg {index} size");
        assert_eq!(actual.flags, expected.flags, "msg {index} flags");
        assert_eq!(
            actual.creation_order, expected.creation_order,
            "msg {index} creation_order"
        );
        assert_eq!(actual.data, expected.data, "msg {index} data");
    }
}

pub(super) fn assert_all_parse_paths_match(data: &[u8]) {
    let buffered = ObjectHeader::parse_with_base(
        data,
        AccessMode::ReadOnly,
        0,
        OFFSET_SIZE,
        LENGTH_SIZE,
        BaseAddress::ZERO,
    )
    .unwrap();
    let memory = parse_from_source_for_mode(data, AccessMode::ReadOnly).unwrap();

    assert_same_header(&memory, &buffered);
}

pub(super) fn assert_unexpected_eof_all_parse_paths(
    data: &[u8],
    expected: usize,
    available: usize,
) {
    let buffered = ObjectHeader::parse_with_base(
        data,
        AccessMode::ReadOnly,
        0,
        OFFSET_SIZE,
        LENGTH_SIZE,
        BaseAddress::ZERO,
    );
    let memory = parse_from_source_for_mode(data, AccessMode::ReadOnly);

    assert_eq!(
        buffered.unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
    assert_eq!(
        memory.unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

pub(super) fn assert_invalid_v1_message_size_all_parse_paths(data: &[u8], size: u16) {
    let buffered = ObjectHeader::parse_with_base(
        data,
        AccessMode::ReadOnly,
        0,
        OFFSET_SIZE,
        LENGTH_SIZE,
        BaseAddress::ZERO,
    );
    let memory = parse_from_source_for_mode(data, AccessMode::ReadOnly);

    for (parser, result) in [("buffered", buffered), ("memory source", memory)] {
        assert_eq!(
            result.unwrap_err(),
            FormatError::InvalidObjectHeaderMessageSize(size),
            "{parser}"
        );
    }
}

pub(super) fn parse(data: &[u8]) -> Result<ObjectHeader, FormatError> {
    parse_for_mode(data, AccessMode::ReadOnly)
}

pub(super) fn parse_for_mode(
    data: &[u8],
    access_mode: AccessMode,
) -> Result<ObjectHeader, FormatError> {
    ObjectHeader::parse(data, access_mode, 0, OFFSET_SIZE, LENGTH_SIZE)
}

pub(super) fn parse_from_source_for_mode(
    data: &[u8],
    access_mode: AccessMode,
) -> Result<ObjectHeader, FormatError> {
    ObjectHeader::parse_from_source(
        data,
        access_mode,
        0,
        OFFSET_SIZE,
        LENGTH_SIZE,
        BaseAddress::ZERO,
    )
}

pub(super) const WIDTHS: Widths = Widths::EIGHT;
pub(super) const OFFSET_SIZE: u8 = 8;

pub(super) const LENGTH_SIZE: u8 = 8;
pub(super) const CONTINUATION_OFFSET: usize = 256;

pub(super) const BLOCK_B: usize = 320;
pub(super) const BLOCK_A1: usize = 384;

pub(super) const BLOCK_B1: usize = 448;
pub(super) const BLOCK_PAST_THE_HEADER: usize = 8192;

pub(super) const BLOCK_A: usize = 256;
