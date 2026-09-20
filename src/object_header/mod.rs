//! HDF5 Object Header parsing (v1 and v2).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::bytes::ensure_len;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::Source;

mod v1;
mod v2;

#[derive(Clone, Copy)]
struct ParseContext {
    access_mode: AccessMode,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
}

/// OHDR signature for v2 object headers.
const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

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
pub(crate) enum MessageFilter<'f> {
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
    pub(crate) fn parse_filtered(
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
            Self::parse_v2(data, context, offset, &mut filter)
        } else {
            Self::parse_v1(data, context, offset, &mut filter)
        }
    }

    /// Parses an object header from a [`Source`] using bounded reads for each
    /// header and continuation chunk.
    ///
    /// `base_address` is added to v1 continuation offsets as in
    /// [`Self::parse_with_base`]. The parser holds at most one chunk at a time, so
    /// it supports files larger than the address space on a 32-bit host. Every
    /// message record is tested against `access_mode` as in [`parse`](Self::parse).
    pub fn parse_from_source<S: Source + ?Sized>(
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

    /// Streaming counterpart of [`parse_filtered`](Self::parse_filtered).
    pub(crate) fn parse_from_source_filtered<S: Source + ?Sized>(
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
            Self::parse_v2_from_source(source, context, address, &mut filter)
        } else {
            Self::parse_v1_from_source(source, context, address, &mut filter)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{source::BytesSource, width::UintWidth};
    use rstest::rstest;

    /// A chunk of only Nil padding reserves nothing.
    ///
    /// The reservation is sized from a walk of the chunk. A `HeaderMessage` is
    /// much wider than the four-byte message prefix, so skipped Nil messages are
    /// excluded from the reservation count.
    #[test]
    fn a_chunk_of_padding_reserves_no_messages() {
        let data = V2HeaderBuilder::new()
            .flags(0x01)
            .repeat_raw_message(256, 0x00, &[])
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
        let header = parse(&V1HeaderBuilder::new().build()).unwrap();

        assert_eq!(header.version, 1);
        assert!(header.messages.is_empty());
        assert_eq!(header.reference_count, Some(1));
        assert_eq!(header.flags, 0);
    }

    #[test]
    fn parse_v1_two_messages() {
        const DATASPACE: [u8; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
        const DATA_LAYOUT: [u8; 8] = [5, 6, 0, 0, 0, 0, 0, 0];

        let data = V1HeaderBuilder::new()
            .message(MessageType::DATASPACE, &DATASPACE)
            .message(MessageType::DATA_LAYOUT, &DATA_LAYOUT)
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 2);
        assert_message(&header.messages[0], MessageType::DATASPACE, &DATASPACE);
        assert_message(&header.messages[1], MessageType::DATA_LAYOUT, &DATA_LAYOUT);
    }

    #[test]
    fn parse_v1_unknown_message_ok() {
        let data = V1HeaderBuilder::new()
            .raw_message(UNKNOWN_TYPE, &[0xAA, 0, 0, 0, 0, 0, 0, 0])
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 1);
        assert_eq!(header.messages[0].msg_type.unknown_id(), Some(UNKNOWN_TYPE));
    }

    /// The must-understand guard applies only to message types the parser cannot
    /// name. External Data Files (`0x0007`) is a known type, so the header parses
    /// and the reader reports external storage when the message is used.
    ///
    /// The reference library writes flags `0x01` on this message. A crafted
    /// `0x08` flag exercises the guard.
    #[test]
    fn parse_v1_named_message_survives_must_understand() {
        let data = V1HeaderBuilder::new()
            .message_with_flags(
                MessageType::EXTERNAL_DATA_FILES,
                &[0xAA, 0, 0, 0, 0, 0, 0, 0],
                MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
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

    fn unknown_message_data(location: UnknownMessageLocation, flags: MessageFlags) -> Vec<u8> {
        match location {
            UnknownMessageLocation::V1Header => V1HeaderBuilder::new()
                .raw_message_with_flags(UNKNOWN_TYPE, &[0xAA, 0, 0, 0, 0, 0, 0, 0], flags)
                .build(),
            UnknownMessageLocation::V1Continuation => {
                let continuation =
                    v1_message_record(UNKNOWN_TYPE, &[0xAA, 0, 0, 0, 0, 0, 0, 0], flags);
                TestFileBuilder::new(
                    V1HeaderBuilder::new()
                        .continuation(CONTINUATION_OFFSET, &continuation)
                        .build(),
                )
                .chunk(CONTINUATION_OFFSET, &continuation)
                .build()
            }
            UnknownMessageLocation::V2Header => V2HeaderBuilder::new()
                .raw_message_with_flags(0xFF, &[0xAA], flags)
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
        let data = unknown_message_data(location, MessageFlags::FAIL_IF_UNKNOWN_ALWAYS);

        assert_eq!(
            parse(&data).unwrap_err(),
            FormatError::UnsupportedMessage(UNKNOWN_TYPE),
            "buffered {location:?}"
        );
        assert_eq!(
            parse_from_source_for_mode(&data, AccessMode::ReadOnly).unwrap_err(),
            FormatError::UnsupportedMessage(UNKNOWN_TYPE),
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
        let data = unknown_message_data(location, MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE);

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
                        Some(UNKNOWN_TYPE),
                        "{parser}"
                    );
                    assert_eq!(
                        header.messages[1].data.as_slice(),
                        &[0xAA, 0, 0, 0, 0, 0, 0, 0],
                        "{parser}"
                    );
                }
                UnknownMessageLocation::V1Header => {
                    assert_eq!(header.messages.len(), 1, "{parser}");
                    assert_eq!(
                        header.messages[0].msg_type.unknown_id(),
                        Some(UNKNOWN_TYPE),
                        "{parser}"
                    );
                    assert_eq!(
                        header.messages[0].data.as_slice(),
                        &[0xAA, 0, 0, 0, 0, 0, 0, 0],
                        "{parser}"
                    );
                }
                UnknownMessageLocation::V2Header => {
                    assert_eq!(header.messages.len(), 1, "{parser}");
                    assert_eq!(
                        header.messages[0].msg_type.unknown_id(),
                        Some(UNKNOWN_TYPE),
                        "{parser}"
                    );
                    assert_eq!(header.messages[0].data.as_slice(), &[0xAA], "{parser}");
                }
            }
        }

        assert_eq!(
            parse_for_mode(&data, AccessMode::ReadWrite).unwrap_err(),
            FormatError::UnsupportedMessage(UNKNOWN_TYPE),
            "buffered"
        );
        assert_eq!(
            parse_from_source_for_mode(&data, AccessMode::ReadWrite).unwrap_err(),
            FormatError::UnsupportedMessage(UNKNOWN_TYPE),
            "streamed"
        );
    }

    #[test]
    fn parse_v2_no_timestamps_one_message() {
        let data = V2HeaderBuilder::new()
            .message(MessageType::DATASPACE, &[10, 20])
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
        let data = V2HeaderBuilder::new()
            .flags(V2_STORES_TIMES)
            .timestamps((100, 200, 300, 400))
            .message(MessageType::DATASPACE, &[1])
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
        let data = V2HeaderBuilder::new()
            .flags(V2_STORES_TIMES | V2_TRACKS_CREATION_ORDER)
            .timestamps((0, 0, 0, 0))
            .message(MessageType::DATATYPE, &[9])
            .raw_message(0x05, &[8])
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 2);
        assert!(header.messages[0].creation_order.is_some());
        assert!(header.messages[1].creation_order.is_some());
        assert_eq!(header.access_time, Some(0));
    }

    #[test]
    fn parse_v2_checksum_is_validated() {
        let mut data = V2HeaderBuilder::new()
            .message(MessageType::DATASPACE, &[1, 2, 3])
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
        let data = V2HeaderBuilder::new()
            .raw_message(0x00, &[0, 0, 0, 0])
            .message(MessageType::DATASPACE, &[42])
            .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 1);
        assert_eq!(header.messages[0].msg_type, MessageType::DATASPACE);
    }

    #[test]
    fn parse_v2_chunk_size_widths() {
        for flags in [0x00, 0x01, 0x02] {
            let data = V2HeaderBuilder::new()
                .flags(flags)
                .message(MessageType::DATASPACE, &[1])
                .build();

            assert_eq!(
                parse(&data).unwrap().messages.len(),
                1,
                "flags={flags:#04x}"
            );
        }
    }

    #[test]
    fn parse_v2_continuation() {
        let continuation = v2_continuation_chunk(MessageType::DATATYPE, &[0xDE, 0xAD]);
        let data = TestFileBuilder::new(
            V2HeaderBuilder::new()
                .message(MessageType::DATASPACE, &[42])
                .continuation(CONTINUATION_OFFSET, &continuation)
                .build(),
        )
        .chunk(CONTINUATION_OFFSET, &continuation)
        .build();
        let header = parse(&data).unwrap();

        assert_eq!(header.messages.len(), 2);
        assert_eq!(header.messages[0].msg_type, MessageType::DATASPACE);
        assert_message(&header.messages[1], MessageType::DATATYPE, &[0xDE, 0xAD]);
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

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v2_simple_matches_buffered() {
        let data = V2HeaderBuilder::new()
            .flags(V2_STORES_TIMES)
            .timestamps((1, 2, 3, 4))
            .message(MessageType::DATASPACE, &[1, 2, 3])
            .build();

        assert_all_parse_paths_match(&data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v2_with_continuation_matches_buffered() {
        let continuation = v2_continuation_chunk(MessageType::DATATYPE, &[0xDE, 0xAD]);
        let data = TestFileBuilder::new(
            V2HeaderBuilder::new()
                .message(MessageType::DATASPACE, &[42])
                .continuation(CONTINUATION_OFFSET, &continuation)
                .build(),
        )
        .chunk(CONTINUATION_OFFSET, &continuation)
        .build();

        assert_all_parse_paths_match(&data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v1_with_continuation_matches_buffered() {
        let continuation = v1_message_record(
            MessageType::DATATYPE.to_u16(),
            &[0xBE, 0xEF, 0, 0, 0, 0, 0, 0],
            MessageFlags::NONE,
        );
        let data = TestFileBuilder::new(
            V1HeaderBuilder::new()
                .message(MessageType::DATASPACE, &[42, 0, 0, 0, 0, 0, 0, 0])
                .continuation(CONTINUATION_OFFSET, &continuation)
                .build(),
        )
        .chunk(CONTINUATION_OFFSET, &continuation)
        .build();

        assert_all_parse_paths_match(&data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn version_1_message_overrunning_chunk0_is_rejected() {
        // Regression for #140: a v1 chunk 0 message whose data overruns the declared
        // object header size (`header_data_size`) is malformed. All parser paths
        // reject the message at the chunk boundary, so the continuation is not
        // followed.
        let continuation = v1_message_record(
            MessageType::DATATYPE.to_u16(),
            &[0xBE, 0xEF, 0, 0, 0, 0, 0, 0],
            MessageFlags::NONE,
        );
        let data = TestFileBuilder::new(
            V1HeaderBuilder::new()
                .continuation(CONTINUATION_OFFSET, &continuation)
                .declared_data_size(16)
                .build(),
        )
        .chunk(CONTINUATION_OFFSET, &continuation)
        .build();

        assert_unexpected_eof_all_parse_paths(&data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn version_1_continuation_with_partial_message_prefix_is_rejected() {
        let mut continuation = v1_message_record(
            MessageType::DATATYPE.to_u16(),
            &[0xAB; 8],
            MessageFlags::NONE,
        );

        // One trailing byte cannot form the eight-byte prefix of another v1 message.
        continuation.push(0);

        let data = TestFileBuilder::new(
            V1HeaderBuilder::new()
                .continuation(CONTINUATION_OFFSET, &continuation)
                .build(),
        )
        .chunk(CONTINUATION_OFFSET, &continuation)
        .build();

        assert_unexpected_eof_all_parse_paths(&data);
    }

    #[test]
    fn a_filtered_version_1_parse_follows_unretained_continuations() {
        let nested_offset = 512;
        let nested_data = [0xDE, 0xAD, 0, 0, 0, 0, 0, 0];
        let nested_chunk = v1_message_record(
            MessageType::DATATYPE.to_u16(),
            &nested_data,
            MessageFlags::NONE,
        );
        let continuation_chunk = v1_message_record(
            MessageType::OBJECT_HEADER_CONTINUATION.to_u16(),
            &continuation_pointer(nested_offset, nested_chunk.len()),
            MessageFlags::NONE,
        );
        let data = TestFileBuilder::new(
            V1HeaderBuilder::new()
                .continuation(CONTINUATION_OFFSET, &continuation_chunk)
                .build(),
        )
        .chunk(CONTINUATION_OFFSET, &continuation_chunk)
        .chunk(nested_offset, &nested_chunk)
        .build();

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
            &BytesSource::new(&data),
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

    #[test]
    fn a_filtered_version_1_parse_checks_must_understand_before_filtering() {
        let data = V1HeaderBuilder::new()
            .raw_message_with_flags(
                UNKNOWN_TYPE,
                &[0xAA, 0, 0, 0, 0, 0, 0, 0],
                MessageFlags::FAIL_IF_UNKNOWN_ALWAYS,
            )
            .build();

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
                &BytesSource::new(&data),
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
            FormatError::UnsupportedMessage(UNKNOWN_TYPE)
        );
        assert_eq!(
            streamed_error,
            FormatError::UnsupportedMessage(UNKNOWN_TYPE)
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

    #[cfg(feature = "std")]
    #[test]
    fn version_1_chunk0_message_with_unaligned_size_is_rejected() {
        let data = V1HeaderBuilder::new()
            .message(MessageType::DATATYPE, &[0xAB; 7])
            .build();

        assert_invalid_v1_message_size_all_parse_paths(&data, 7);
    }

    #[cfg(feature = "std")]
    #[test]
    fn version_1_continuation_message_with_unaligned_size_is_rejected() {
        let continuation = v1_message_record(
            MessageType::DATATYPE.to_u16(),
            &[0xAB; 7],
            MessageFlags::NONE,
        );

        let data = TestFileBuilder::new(
            V1HeaderBuilder::new()
                .continuation(CONTINUATION_OFFSET, &continuation)
                .build(),
        )
        .chunk(CONTINUATION_OFFSET, &continuation)
        .build();

        assert_invalid_v1_message_size_all_parse_paths(&data, 7);
    }

    fn assert_message(message: &HeaderMessage, msg_type: MessageType, data: &[u8]) {
        assert_eq!(message.msg_type, msg_type);
        assert_eq!(message.data.as_slice(), data);
    }

    #[cfg(feature = "std")]
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

    #[cfg(feature = "std")]
    fn assert_all_parse_paths_match(data: &[u8]) {
        use crate::source::ReadSeekSource;

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
        let seek = ObjectHeader::parse_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(data.to_vec())).unwrap(),
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        )
        .unwrap();

        assert_same_header(&memory, &buffered);
        assert_same_header(&seek, &buffered);
    }

    #[cfg(feature = "std")]
    fn assert_unexpected_eof_all_parse_paths(data: &[u8]) {
        use crate::source::ReadSeekSource;

        let buffered = ObjectHeader::parse_with_base(
            data,
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        );
        let memory = parse_from_source_for_mode(data, AccessMode::ReadOnly);
        let seek = ObjectHeader::parse_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(data.to_vec())).unwrap(),
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        );

        assert!(matches!(buffered, Err(FormatError::UnexpectedEof { .. })));
        assert!(matches!(memory, Err(FormatError::UnexpectedEof { .. })));
        assert!(matches!(seek, Err(FormatError::UnexpectedEof { .. })));
    }

    #[cfg(feature = "std")]
    fn assert_invalid_v1_message_size_all_parse_paths(data: &[u8], size: u16) {
        use crate::source::ReadSeekSource;

        let buffered = ObjectHeader::parse_with_base(
            data,
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        );
        let memory = parse_from_source_for_mode(data, AccessMode::ReadOnly);
        let seek = ObjectHeader::parse_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(data.to_vec())).unwrap(),
            AccessMode::ReadOnly,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        );

        for (parser, result) in [
            ("buffered", buffered),
            ("memory source", memory),
            ("seek source", seek),
        ] {
            assert_eq!(
                result.unwrap_err(),
                FormatError::InvalidObjectHeaderMessageSize(size),
                "{parser}"
            );
        }
    }

    #[derive(Clone)]
    struct TestMessage<T> {
        msg_type: T,
        data: Vec<u8>,
        flags: MessageFlags,
    }

    impl<T> TestMessage<T> {
        fn new(msg_type: T, data: &[u8], flags: MessageFlags) -> Self {
            Self {
                msg_type,
                data: data.to_vec(),
                flags,
            }
        }
    }

    struct V1HeaderBuilder {
        messages: Vec<TestMessage<u16>>,
        reference_count: u32,
        declared_data_size: Option<usize>,
    }

    impl V1HeaderBuilder {
        fn new() -> Self {
            Self {
                messages: Vec::new(),
                reference_count: 1,
                declared_data_size: None,
            }
        }

        fn message(self, msg_type: MessageType, data: &[u8]) -> Self {
            self.raw_message(msg_type.to_u16(), data)
        }

        fn message_with_flags(
            self,
            msg_type: MessageType,
            data: &[u8],
            flags: MessageFlags,
        ) -> Self {
            self.raw_message_with_flags(msg_type.to_u16(), data, flags)
        }

        fn raw_message(self, msg_type: u16, data: &[u8]) -> Self {
            self.raw_message_with_flags(msg_type, data, MessageFlags::NONE)
        }

        fn raw_message_with_flags(
            mut self,
            msg_type: u16,
            data: &[u8],
            flags: MessageFlags,
        ) -> Self {
            self.messages.push(TestMessage::new(msg_type, data, flags));
            self
        }

        fn continuation(self, offset: usize, chunk: &[u8]) -> Self {
            self.message(
                MessageType::OBJECT_HEADER_CONTINUATION,
                &continuation_pointer(offset, chunk.len()),
            )
        }

        fn declared_data_size(mut self, size: usize) -> Self {
            self.declared_data_size = Some(size);
            self
        }

        fn build(self) -> Vec<u8> {
            let records = v1_message_records(&self.messages);
            let data_size = self.declared_data_size.unwrap_or(records.len());

            let mut buf = Vec::new();
            buf.push(1);
            buf.push(0);
            buf.extend_from_slice(&(self.messages.len() as u16).to_le_bytes());
            buf.extend_from_slice(&self.reference_count.to_le_bytes());
            buf.extend_from_slice(&(data_size as u32).to_le_bytes());
            buf.extend_from_slice(&[0u8; 4]);
            buf.extend_from_slice(&records);
            buf
        }
    }

    struct V2HeaderBuilder {
        flags: u8,
        messages: Vec<TestMessage<u8>>,
        timestamps: Option<(u32, u32, u32, u32)>,
    }

    impl V2HeaderBuilder {
        fn new() -> Self {
            Self {
                flags: 0,
                messages: Vec::new(),
                timestamps: None,
            }
        }

        fn flags(mut self, flags: u8) -> Self {
            self.flags = flags;
            self
        }

        fn timestamps(mut self, timestamps: (u32, u32, u32, u32)) -> Self {
            self.timestamps = Some(timestamps);
            self
        }

        fn message(self, msg_type: MessageType, data: &[u8]) -> Self {
            self.raw_message(msg_type.to_u16() as u8, data)
        }

        fn raw_message(self, msg_type: u8, data: &[u8]) -> Self {
            self.raw_message_with_flags(msg_type, data, MessageFlags::NONE)
        }

        fn raw_message_with_flags(
            mut self,
            msg_type: u8,
            data: &[u8],
            flags: MessageFlags,
        ) -> Self {
            self.messages.push(TestMessage::new(msg_type, data, flags));
            self
        }

        fn repeat_raw_message(mut self, count: usize, msg_type: u8, data: &[u8]) -> Self {
            for _ in 0..count {
                self.messages
                    .push(TestMessage::new(msg_type, data, MessageFlags::NONE));
            }
            self
        }

        fn continuation(self, offset: usize, chunk: &[u8]) -> Self {
            self.message(
                MessageType::OBJECT_HEADER_CONTINUATION,
                &continuation_pointer(offset, chunk.len()),
            )
        }

        fn build(self) -> Vec<u8> {
            let header_flags = v2::HeaderFlags::new(self.flags);
            let has_creation_order = header_flags.tracks_creation_order();

            let mut buf = Vec::new();
            buf.extend_from_slice(&OHDR_SIGNATURE);
            buf.push(2);
            buf.push(self.flags);

            if header_flags.stores_times()
                && let Some((access, modification, change, birth)) = self.timestamps
            {
                buf.extend_from_slice(&access.to_le_bytes());
                buf.extend_from_slice(&modification.to_le_bytes());
                buf.extend_from_slice(&change.to_le_bytes());
                buf.extend_from_slice(&birth.to_le_bytes());
            }

            if header_flags.stores_attribute_phase_change() {
                buf.extend_from_slice(&8u16.to_le_bytes());
                buf.extend_from_slice(&6u16.to_le_bytes());
            }

            let mut records = Vec::new();
            for message in &self.messages {
                records.push(message.msg_type);
                records.extend_from_slice(&(message.data.len() as u16).to_le_bytes());
                records.push(message.flags.get());
                if has_creation_order {
                    records.extend_from_slice(&0u16.to_le_bytes());
                }
                records.extend_from_slice(&message.data);
            }

            match header_flags.chunk_size_width() {
                UintWidth::One => buf.push(records.len() as u8),
                UintWidth::Two => buf.extend_from_slice(&(records.len() as u16).to_le_bytes()),
                UintWidth::Four => buf.extend_from_slice(&(records.len() as u32).to_le_bytes()),
                UintWidth::Eight => buf.extend_from_slice(&(records.len() as u64).to_le_bytes()),
            }

            buf.extend_from_slice(&records);
            append_checksum(&mut buf);
            buf
        }
    }

    struct TestFileBuilder {
        data: Vec<u8>,
    }

    impl TestFileBuilder {
        fn new(header: Vec<u8>) -> Self {
            Self { data: header }
        }

        fn chunk(mut self, offset: usize, chunk: &[u8]) -> Self {
            self.data
                .resize(self.data.len().max(offset + chunk.len()), 0);
            self.data[offset..offset + chunk.len()].copy_from_slice(chunk);
            self
        }

        fn build(self) -> Vec<u8> {
            self.data
        }
    }

    fn v1_message_records(messages: &[TestMessage<u16>]) -> Vec<u8> {
        let mut records = Vec::new();
        for message in messages {
            records.extend_from_slice(&message.msg_type.to_le_bytes());
            records.extend_from_slice(&(message.data.len() as u16).to_le_bytes());
            records.push(message.flags.get());
            records.extend_from_slice(&[0u8; 3]);
            records.extend_from_slice(&message.data);
        }
        records
    }

    fn v1_message_record(msg_type: u16, data: &[u8], flags: MessageFlags) -> Vec<u8> {
        v1_message_records(&[TestMessage::new(msg_type, data, flags)])
    }

    fn v2_continuation_chunk(msg_type: MessageType, data: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&v2::OCHK_SIGNATURE);
        chunk.push(msg_type.to_u16() as u8);
        chunk.extend_from_slice(&(data.len() as u16).to_le_bytes());
        chunk.push(MessageFlags::NONE.get());
        chunk.extend_from_slice(data);
        append_checksum(&mut chunk);
        chunk
    }

    fn continuation_pointer(offset: usize, length: usize) -> Vec<u8> {
        let mut pointer = Vec::with_capacity(16);
        pointer.extend_from_slice(&(offset as u64).to_le_bytes());
        pointer.extend_from_slice(&(length as u64).to_le_bytes());
        pointer
    }

    fn append_checksum(data: &mut Vec<u8>) {
        let checksum = crate::checksum::jenkins_lookup3(data);
        data.extend_from_slice(&checksum.to_le_bytes());
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
            &BytesSource::new(data),
            access_mode,
            0,
            OFFSET_SIZE,
            LENGTH_SIZE,
            BaseAddress::ZERO,
        )
    }

    const OFFSET_SIZE: u8 = 8;
    const LENGTH_SIZE: u8 = 8;
    const CONTINUATION_OFFSET: usize = 256;
    const UNKNOWN_TYPE: u16 = 0x00FF;
    const V2_TRACKS_CREATION_ORDER: u8 = 0x04;
    const V2_STORES_TIMES: u8 = 0x20;
}
