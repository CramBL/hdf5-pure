//! The object header, version 1 and version 2: the parser, and the codec of the prefix and the
//! message records of a version 2 header.

use alloc::vec::Vec;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::bytes::ensure_len;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;

mod v1;
mod v2;

pub use v2::AttributePhaseChange;
pub use v2::MessageRecord;
pub use v2::MessageRecordLayout;
pub use v2::OBJECT_HEADER_PREFIX_MAX_LEN;
pub use v2::ObjectHeaderContinuation;
pub use v2::ObjectHeaderPrefix;
pub use v2::ObjectTimes;
pub use v2::ParsedObjectHeaderPrefix;
pub use v2::continuation_block_messages;

#[derive(Clone, Copy)]
struct ParseContext {
    access_mode: AccessMode,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
}

/// OHDR signature for v2 object headers.
pub(super) const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// Controls which parsed messages are retained.
///
/// A parse owns a copy of every retained message. Path resolution usually needs
/// one Link message from a group that may contain many Link messages. Filtering
/// retains only messages that can satisfy that lookup while still walking the
/// complete header.
///
/// Filtering affects retention only. A filter that cannot determine whether a
/// message is relevant retains it for the message reader. Continuation messages
/// are always followed.
pub enum MessageFilter<'f> {
    /// Keeps every message.
    All,
    /// Keep a message only if this says so, given its type and its body.
    Only(&'f mut dyn FnMut(MessageType, &[u8]) -> bool),
}

impl MessageFilter<'_> {
    /// Whether the message of type `msg_type` with body `data` is kept.
    fn keeps(&mut self, msg_type: MessageType, data: &[u8]) -> bool {
        match self {
            MessageFilter::All => true,
            MessageFilter::Only(f) => f(msg_type, data),
        }
    }

    /// Whether this filter keeps everything, and so the message vector can be
    /// sized from a count of the chunk. A filter that drops most of a header
    /// would make that reservation the largest thing the parse allocates.
    fn keeps_all(&self) -> bool {
        matches!(self, MessageFilter::All)
    }
}

/// A single parsed header message.
///
/// `size` and `creation_order` are decoded from the on-disk message prefix for
/// completeness but are not consulted by the current reader.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct HeaderMessage {
    /// The message type.
    pub msg_type: MessageType,
    /// Size of the message data in bytes.
    pub size: usize,
    /// The record's flags. `data` holds a reference to the message when
    /// [`MessageFlags::SHARED`] is set.
    pub flags: MessageFlags,
    /// Creation order (v2 only, when tracking is enabled).
    pub creation_order: Option<u16>,
    /// Raw message data bytes.
    pub data: Vec<u8>,
}

/// Parsed HDF5 object header.
///
/// The version, reference count, flags, and v2 timestamps are decoded for
/// on-disk format completeness. The current reader does not consult them.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Header version (1 or 2).
    pub version: u8,
    /// All non-NIL messages collected from all chunks.
    pub messages: Vec<HeaderMessage>,
    /// Object reference count (v1 only).
    pub reference_count: Option<u32>,
    /// Contains the v2 object header flags. Version 1 uses zero.
    pub flags: u8,
    /// Access time (v2, when flags bit 2 set).
    pub access_time: Option<u32>,
    /// Modification time (v2, when flags bit 2 set).
    pub modification_time: Option<u32>,
    /// Change time (v2, when flags bit 2 set).
    pub change_time: Option<u32>,
    /// Birth time (v2, when flags bit 2 set).
    pub birth_time: Option<u32>,
}

impl ObjectHeader {
    /// Parses the object header at `offset` in `data`.
    ///
    /// `offset_size` and `length_size` come from the superblock, and `access_mode` is
    /// the mode the caller reads under.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedMessage`] if a message record holds a type
    /// this parser cannot name and [`MessageFlags::must_be_understood`] returns
    /// `true` for `access_mode`.
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a continuation message of a version 2 header
    /// ends inside its fields, and [`FormatError::UndefinedContinuationAddress`] if one stores
    /// the undefined address.
    pub fn parse(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_with_base(
            data,
            access_mode,
            offset,
            offset_size,
            length_size,
            BaseAddress::ZERO,
        )
    }

    /// Parses an object header, applying `base_address` to continuation offsets.
    ///
    /// A continuation message stores a *file address*, and the format specification
    /// says of the superblock's base address that "unless otherwise noted, all
    /// other file addresses are relative to this base address". So for a file with
    /// a userblock (a `.mat` carries a 512-byte one) the stored offset is short of
    /// the real position by exactly the base, and this method adds it back before
    /// reading the block. Both header versions store the address the same way.
    ///
    /// `offset` itself is already absolute: every caller resolves an address to a
    /// file position before parsing there. Every message record is tested against
    /// `access_mode`, as in [`parse`](Self::parse).
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns.
    pub fn parse_with_base(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_filtered(
            data,
            access_mode,
            offset,
            offset_size,
            length_size,
            base_address,
            MessageFilter::All,
        )
    }

    /// Parses an object header, keeping only the messages `filter` names.
    ///
    /// The header uses the same validation, checksum checks, and continuation
    /// traversal as [`parse_with_base`](Self::parse_with_base). The result differs
    /// only in which messages it carries. See [`MessageFilter`] for targeted
    /// parsing.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns.
    pub fn parse_filtered(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
        mut filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        let context = ParseContext {
            access_mode,
            offset_size,
            length_size,
            base_address,
        };
        ensure_len(data, offset, 4)?;
        if data[offset..offset + 4] == OHDR_SIGNATURE {
            Self::parse_v2(data, context, offset, filter)
        } else {
            Self::parse_v1(data, context, offset, &mut filter)
        }
    }

    /// Parses an object header from a [`MetadataSource`] using bounded reads for each
    /// header and continuation chunk.
    ///
    /// `base_address` is added to v1 continuation offsets as in
    /// [`Self::parse_with_base`]. The parser holds at most one chunk at a time, so
    /// it supports files larger than the address space on a 32-bit host. Every
    /// message record is tested against `access_mode` as in [`parse`](Self::parse).
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
    pub fn parse_from_source<S: MetadataSource + ?Sized>(
        source: &S,
        access_mode: AccessMode,
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_from_source_filtered(
            source,
            access_mode,
            address,
            offset_size,
            length_size,
            base_address,
            MessageFilter::All,
        )
    }

    /// Parses an object header from a [`MetadataSource`], keeping only the messages `filter` names.
    ///
    /// The streaming counterpart of [`parse_filtered`](Self::parse_filtered).
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
    pub fn parse_from_source_filtered<S: MetadataSource + ?Sized>(
        source: &S,
        access_mode: AccessMode,
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
        mut filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        let context = ParseContext {
            access_mode,
            offset_size,
            length_size,
            base_address,
        };
        let mut sig = [0u8; 4];
        source.read_at(address, &mut sig)?;
        if sig == OHDR_SIGNATURE {
            Self::parse_v2_from_source(source, context, address, filter)
        } else {
            Self::parse_v1_from_source(source, context, address, &mut filter)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use test_util::image::Image;
    use test_util::object_header::v1 as v1_bytes;
    use test_util::object_header::v2 as v2_bytes;
    use test_util::object_header::{
        Message, MessageFlags as RecordFlags, MessageType as RecordType,
    };
    use test_util::widths::Widths;

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
        let header = parse(&data).unwrap();

        assert!(header.messages.is_empty(), "Nil messages are not kept");
        assert_eq!(
            header.messages.capacity(),
            0,
            "a chunk the parse keeps nothing from must not reserve for it"
        );
    }

    #[test]
    fn parse_v1_zero_messages() {
        let header = parse(&v1_bytes::Header::new().build()).unwrap();

        assert_eq!(header.version, 1);
        assert!(header.messages.is_empty());
        assert_eq!(header.reference_count, Some(1));
        assert_eq!(header.flags, 0);
    }

    #[test]
    fn parse_v1_two_messages() {
        const DATASPACE: [u8; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
        const DATA_LAYOUT: [u8; 8] = [5, 6, 0, 0, 0, 0, 0, 0];

        let data = v1_bytes::Header::new()
            .message(Message::new(RecordType::DATASPACE, &DATASPACE))
            .message(Message::new(RecordType::DATA_LAYOUT, &DATA_LAYOUT))
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 2);
        assert_message(&header.messages[0], MessageType::DATASPACE, &DATASPACE);
        assert_message(&header.messages[1], MessageType::DATA_LAYOUT, &DATA_LAYOUT);
    }

    #[test]
    fn parse_v1_unknown_message_ok() {
        let data = v1_bytes::Header::new()
            .message(Message::new(RecordType::UNKNOWN, &UNKNOWN_BODY))
            .build();
        let header = parse(&data).unwrap();

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
        let header = parse_for_mode(&data, AccessMode::ReadWrite).unwrap();

        assert_eq!(header.messages.len(), 1);
        assert_eq!(
            header.messages[0].msg_type,
            MessageType::EXTERNAL_DATA_FILES
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

    #[rstest]
    #[case::v1_header(UnknownMessageLocation::V1Header)]
    #[case::v1_continuation(UnknownMessageLocation::V1Continuation)]
    #[case::v2_header(UnknownMessageLocation::V2Header)]
    fn unknown_message_that_must_always_be_understood_is_rejected(
        #[case] location: UnknownMessageLocation,
    ) {
        let data = unknown_message_data(location, RecordFlags::FAIL_IF_UNKNOWN_ALWAYS);

        assert_eq!(
            parse(&data).unwrap_err(),
            FormatError::UnsupportedMessage(RecordType::UNKNOWN.0),
            "buffered {location:?}"
        );
        assert_eq!(
            parse_from_source_for_mode(&data, AccessMode::ReadOnly).unwrap_err(),
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

        let buffered = parse_for_mode(&data, AccessMode::ReadOnly).unwrap();
        let streamed = parse_from_source_for_mode(&data, AccessMode::ReadOnly).unwrap();

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
            parse_for_mode(&data, AccessMode::ReadWrite).unwrap_err(),
            FormatError::UnsupportedMessage(RecordType::UNKNOWN.0),
            "buffered"
        );
        assert_eq!(
            parse_from_source_for_mode(&data, AccessMode::ReadWrite).unwrap_err(),
            FormatError::UnsupportedMessage(RecordType::UNKNOWN.0),
            "streamed"
        );
    }

    #[test]
    fn parse_v2_no_timestamps_one_message() {
        let data = v2_bytes::Header::new()
            .message(Message::new(RecordType::DATASPACE, &[10, 20]))
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.version, 2);
        assert_eq!(header.flags, 0);
        assert_eq!(header.messages.len(), 1);
        assert_message(&header.messages[0], MessageType::DATASPACE, &[10, 20]);
        assert!(header.access_time.is_none());
    }

    #[test]
    fn parse_v2_with_timestamps() {
        let data = v2_bytes::Header::new()
            .timestamps(v2_bytes::Timestamps {
                access: 100,
                modification: 200,
                change: 300,
                birth: 400,
            })
            .message(Message::new(RecordType::DATASPACE, &[1]))
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.access_time, Some(100));
        assert_eq!(header.modification_time, Some(200));
        assert_eq!(header.change_time, Some(300));
        assert_eq!(header.birth_time, Some(400));
        assert_eq!(header.messages.len(), 1);
        assert!(header.messages[0].creation_order.is_none());
    }

    #[test]
    fn parse_v2_creation_order() {
        let data = v2_bytes::Header::new()
            .flags(v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER)
            .timestamps(v2_bytes::Timestamps::default())
            .message(Message::new(RecordType::DATATYPE, &[9]))
            .message(Message::new(RecordType::FILL_VALUE, &[8]))
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 2);
        assert!(header.messages[0].creation_order.is_some());
        assert!(header.messages[1].creation_order.is_some());
        assert_eq!(header.access_time, Some(0));
    }

    #[test]
    fn parse_v2_checksum_is_validated() {
        let mut data = v2_bytes::Header::new()
            .message(Message::new(RecordType::DATASPACE, &[1, 2, 3]))
            .build();

        assert_eq!(parse(&data).unwrap().messages.len(), 1);

        *data.last_mut().unwrap() ^= 0xFF;
        assert!(matches!(
            parse(&data),
            Err(FormatError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn parse_v2_nil_padding_skipped() {
        let data = v2_bytes::Header::new()
            .message(Message::nil(4))
            .message(Message::new(RecordType::DATASPACE, &[42]))
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 1);
        assert_eq!(header.messages[0].msg_type, MessageType::DATASPACE);
    }

    #[rstest]
    #[case::one_byte(0x00)]
    #[case::two_bytes(0x01)]
    #[case::four_bytes(0x02)]
    fn parse_v2_chunk_size_widths(#[case] flags: u8) {
        let data = v2_bytes::Header::new()
            .flags(v2_bytes::HeaderFlags(flags))
            .message(Message::new(RecordType::DATASPACE, &[1]))
            .build();

        assert_eq!(parse(&data).unwrap().messages.len(), 1);
    }

    #[test]
    fn parse_v2_continuation() {
        let continuation = v2_continuation(&[0xDE, 0xAD]);
        let mut image = Image::starting_with(
            &v2_bytes::Header::new()
                .message(Message::new(RecordType::DATASPACE, &[42]))
                .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
                .build(),
        );
        image.place(CONTINUATION_OFFSET, &continuation);
        let data = image.build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 2);
        assert_eq!(header.messages[0].msg_type, MessageType::DATASPACE);
        assert_message(&header.messages[1], MessageType::DATATYPE, &[0xDE, 0xAD]);
    }

    #[rstest]
    #[case::empty(&[], FormatError::UnexpectedEof { expected: 8, available: 0 })]
    #[case::the_address_alone(&[0; 8], FormatError::UnexpectedEof { expected: 16, available: 8 })]
    #[case::one_byte_short(&[0; 15], FormatError::UnexpectedEof { expected: 16, available: 15 })]
    fn a_v2_continuation_message_too_short_for_its_fields_is_rejected(
        #[case] body: &[u8],
        #[case] expected: FormatError,
    ) {
        let data = v2_bytes::Header::new()
            .message(Message::new(RecordType::OBJECT_HEADER_CONTINUATION, body))
            .build();

        assert_eq!(parse(&data).unwrap_err(), expected);
        assert_eq!(
            parse_from_source_for_mode(&data, AccessMode::ReadOnly).unwrap_err(),
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
            ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, offset_size, LENGTH_SIZE)
                .unwrap_err(),
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

        assert_eq!(bodies(&header), expected);
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

        assert_eq!(bodies(&header), [b"b".as_slice(), b"a1", b"b1"]);
    }

    #[rstest]
    fn a_short_version_2_continuation_is_rejected_under_a_filter_that_keeps_nothing(
        #[values(Driver::Buffered, Driver::Source)] driver: Driver,
    ) {
        // The body of a continuation message is 16 bytes at 8-byte offsets and lengths.
        let data = v2_bytes::Header::new()
            .message(named(b"a"))
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

    #[rstest]
    #[case::at_the_limit(256, Ok(256))]
    #[case::past_the_limit(257, Err(FormatError::NestingDepthExceeded))]
    fn a_version_2_header_parse_accepts_at_most_256_continuation_messages(
        #[values(Driver::Buffered, Driver::Source)] driver: Driver,
        #[case] count: usize,
        #[case] expected: Result<usize, FormatError>,
    ) {
        // Every continuation message refers to the same block, which the parse reads once per message.
        let block = continuation_block(&[named(b"a")]);
        let data = continuation_image(
            v2_bytes::Header::new()
                .flags(v2_bytes::HeaderFlags(0x01))
                .messages((0..count).map(|_| continuation_to(BLOCK_PAST_THE_HEADER, &block))),
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
    fn truncated_headers_are_rejected() {
        for (version, data) in [("v1", vec![1, 0]), ("v2", vec![b'O', b'H', b'D', b'R', 2])] {
            assert!(
                matches!(parse(&data), Err(FormatError::UnexpectedEof { .. })),
                "{version}"
            );
        }
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

        assert_all_parse_paths_match(&data);
    }

    #[test]
    fn streaming_v2_with_continuation_matches_buffered() {
        let continuation = v2_continuation(&[0xDE, 0xAD]);
        let mut image = Image::starting_with(
            &v2_bytes::Header::new()
                .message(Message::new(RecordType::DATASPACE, &[42]))
                .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
                .build(),
        );
        image.place(CONTINUATION_OFFSET, &continuation);

        assert_all_parse_paths_match(image.as_bytes());
    }

    #[test]
    fn streaming_v1_with_continuation_matches_buffered() {
        let continuation = v1_datatype_chunk(&[0xBE, 0xEF, 0, 0, 0, 0, 0, 0]);
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

        assert_all_parse_paths_match(image.as_bytes());
    }

    #[test]
    fn version_1_message_overrunning_chunk0_is_rejected() {
        // Regression for #140: a v1 chunk 0 message whose data overruns the declared
        // object header size (`header_data_size`) is malformed. All parser paths
        // reject the message at the chunk boundary, so the continuation is not
        // followed.
        let continuation = v1_datatype_chunk(&[0xBE, 0xEF, 0, 0, 0, 0, 0, 0]);
        let mut image = Image::starting_with(
            &v1_bytes::Header::new()
                .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
                .declared_data_size(16)
                .build(),
        );
        image.place(CONTINUATION_OFFSET, &continuation);

        assert_unexpected_eof_all_parse_paths(image.as_bytes());
    }

    #[test]
    fn version_1_continuation_with_partial_message_prefix_is_rejected() {
        let mut continuation = v1_datatype_chunk(&[0xAB; 8]);

        // One trailing byte cannot form the eight-byte prefix of another v1 message.
        continuation.push(0);

        let mut image = Image::starting_with(
            &v1_bytes::Header::new()
                .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
                .build(),
        );
        image.place(CONTINUATION_OFFSET, &continuation);

        assert_unexpected_eof_all_parse_paths(image.as_bytes());
    }

    #[test]
    fn a_filtered_version_1_parse_follows_unretained_continuations() {
        let nested_offset = 512;
        let nested_data = [0xDE, 0xAD, 0, 0, 0, 0, 0, 0];
        let nested_chunk = v1_datatype_chunk(&nested_data);
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
            assert_message(&header.messages[0], MessageType::DATATYPE, &nested_data);
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

    #[test]
    fn version_1_chunk0_message_with_unaligned_size_is_rejected() {
        let data = v1_bytes::Header::new()
            .message(Message::new(RecordType::DATATYPE, &[0xAB; 7]))
            .build();

        assert_invalid_v1_message_size_all_parse_paths(&data, 7);
    }

    #[test]
    fn version_1_continuation_message_with_unaligned_size_is_rejected() {
        let continuation = v1_datatype_chunk(&[0xAB; 7]);

        let mut image = Image::starting_with(
            &v1_bytes::Header::new()
                .continuation(CONTINUATION_OFFSET, &continuation, WIDTHS)
                .build(),
        );
        image.place(CONTINUATION_OFFSET, &continuation);

        assert_invalid_v1_message_size_all_parse_paths(image.as_bytes(), 7);
    }

    /// A version 1 continuation chunk of one Datatype message carrying `data`.
    fn v1_datatype_chunk(data: &[u8]) -> Vec<u8> {
        v1_bytes::message_record(&Message::new(RecordType::DATATYPE, data))
    }

    /// A version 2 continuation chunk of one Datatype message carrying `data`.
    fn v2_continuation(data: &[u8]) -> Vec<u8> {
        v2_bytes::continuation_chunk(
            &[Message::new(RecordType::DATATYPE, data)],
            v2_bytes::HeaderFlags::default(),
        )
    }

    /// The two entry points of the parser: a buffer that holds the whole file, and a
    /// [`MetadataSource`].
    #[derive(Clone, Copy, Debug)]
    enum Driver {
        Buffered,
        Source,
    }

    impl Driver {
        fn parse(
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
    enum ContinuationTree {
        /// Block `a` refers to block `a1`, and block `b` to block `b1`.
        BothNested,
        /// Block `a` refers to block `a1`.
        Nested,
        /// Neither block refers to a further block.
        Siblings,
    }

    impl ContinuationTree {
        fn image(self) -> Vec<u8> {
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
    fn continuation_image(header: v2_bytes::Header, blocks: &[(usize, &[u8])]) -> Vec<u8> {
        let mut image = Image::starting_with(&header.build());
        for &(at, block) in blocks {
            image.place(at, block);
        }
        image.build()
    }

    fn continuation_block(messages: &[Message]) -> Vec<u8> {
        v2_bytes::continuation_chunk(messages, v2_bytes::HeaderFlags::default())
    }

    fn continuation_to(at: usize, block: &[u8]) -> Message {
        Message::continuation(at as u64, block.len() as u64, WIDTHS)
    }

    /// Returns a Datatype message whose body is `body`, the name [`bodies`] lists it by.
    fn named(body: &[u8]) -> Message {
        Message::new(RecordType::DATATYPE, body)
    }

    fn bodies(header: &ObjectHeader) -> Vec<&[u8]> {
        header
            .messages
            .iter()
            .map(|message| message.data.as_slice())
            .collect()
    }

    fn assert_message(message: &HeaderMessage, msg_type: MessageType, data: &[u8]) {
        assert_eq!(message.msg_type, msg_type);
        assert_eq!(message.data.as_slice(), data);
    }

    fn assert_same_header(actual: &ObjectHeader, expected: &ObjectHeader) {
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

        for (index, (actual, expected)) in
            actual.messages.iter().zip(&expected.messages).enumerate()
        {
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

    fn assert_all_parse_paths_match(data: &[u8]) {
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

    fn assert_unexpected_eof_all_parse_paths(data: &[u8]) {
        let buffered = ObjectHeader::parse_with_base(
            data,
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        );
        let memory = parse_from_source_for_mode(data, AccessMode::ReadOnly);

        assert!(matches!(buffered, Err(FormatError::UnexpectedEof { .. })));
        assert!(matches!(memory, Err(FormatError::UnexpectedEof { .. })));
    }

    fn assert_invalid_v1_message_size_all_parse_paths(data: &[u8], size: u16) {
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

    fn parse(data: &[u8]) -> Result<ObjectHeader, FormatError> {
        parse_for_mode(data, AccessMode::ReadOnly)
    }

    fn parse_for_mode(data: &[u8], access_mode: AccessMode) -> Result<ObjectHeader, FormatError> {
        ObjectHeader::parse(data, access_mode, 0, OFFSET_SIZE, LENGTH_SIZE)
    }

    fn parse_from_source_for_mode(
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

    const WIDTHS: Widths = Widths::EIGHT;
    const OFFSET_SIZE: u8 = 8;
    const LENGTH_SIZE: u8 = 8;
    const CONTINUATION_OFFSET: usize = 256;
    const UNKNOWN_BODY: [u8; 8] = [0xAA, 0, 0, 0, 0, 0, 0, 0];
    const BLOCK_A: usize = 256;
    const BLOCK_B: usize = 320;
    const BLOCK_A1: usize = 384;
    const BLOCK_B1: usize = 448;
    const BLOCK_PAST_THE_HEADER: usize = 8192;
}
