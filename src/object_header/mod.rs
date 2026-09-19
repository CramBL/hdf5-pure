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

    // Helper: build a v1 object header with given messages
    fn build_v1_header(
        messages: &[(u16, &[u8], MessageFlags)], // (type, data, flags)
        offset_size: u8,
        length_size: u8,
    ) -> Vec<u8> {
        let _ = (offset_size, length_size);
        let msg_bytes = v1_message_records(messages);

        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(messages.len() as u16).to_le_bytes()); // `num_messages`
        buf.extend_from_slice(&1u32.to_le_bytes()); // `reference_count`
        buf.extend_from_slice(&(msg_bytes.len() as u32).to_le_bytes()); // `header_data_size`
        // Pad to 8-byte alignment (12 bytes so far, pad 4)
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&msg_bytes);
        buf
    }

    /// Builds the message records of one version 1 object header chunk: for each message a type,
    /// a size, a flags byte, three reserved bytes, and the body.
    fn v1_message_records(messages: &[(u16, &[u8], MessageFlags)]) -> Vec<u8> {
        let mut records = Vec::new();
        for (msg_type, msg_data, msg_flags) in messages {
            records.extend_from_slice(&msg_type.to_le_bytes());
            records.extend_from_slice(&(msg_data.len() as u16).to_le_bytes());
            records.push(msg_flags.get());
            records.extend_from_slice(&[0u8; 3]);
            records.extend_from_slice(msg_data);
        }
        records
    }

    // Helper: build a v2 object header chunk0 with given messages
    fn build_v2_header(
        flags: u8,
        messages: &[(u8, &[u8], MessageFlags)],
        timestamps: Option<(u32, u32, u32, u32)>,
    ) -> Vec<u8> {
        let header_flags = v2::HeaderFlags::new(flags);
        let has_creation_order = header_flags.tracks_creation_order();
        let has_timestamps = header_flags.stores_times();
        let mut buf = Vec::new();
        buf.extend_from_slice(&OHDR_SIGNATURE); // 4
        buf.push(2); // version
        buf.push(flags);

        if has_timestamps && let Some((at, mt, ct, bt)) = timestamps {
            buf.extend_from_slice(&at.to_le_bytes());
            buf.extend_from_slice(&mt.to_le_bytes());
            buf.extend_from_slice(&ct.to_le_bytes());
            buf.extend_from_slice(&bt.to_le_bytes());
        }

        if header_flags.stores_attribute_phase_change() {
            buf.extend_from_slice(&8u16.to_le_bytes()); // `max_compact`
            buf.extend_from_slice(&6u16.to_le_bytes()); // `min_dense`
        }

        // Build message bytes to get chunk size
        let mut msg_bytes = Vec::new();
        for (mtype, mdata, mflags) in messages {
            msg_bytes.push(*mtype); // type(1)
            msg_bytes.extend_from_slice(&(mdata.len() as u16).to_le_bytes()); // size(2)
            msg_bytes.push(mflags.get()); // flags(1)
            if has_creation_order {
                msg_bytes.extend_from_slice(&0u16.to_le_bytes()); // creation_order(2)
            }
            msg_bytes.extend_from_slice(mdata);
        }

        let chunk_size = msg_bytes.len();
        match header_flags.chunk_size_width() {
            UintWidth::One => buf.push(chunk_size as u8),
            UintWidth::Two => buf.extend_from_slice(&(chunk_size as u16).to_le_bytes()),
            UintWidth::Four => buf.extend_from_slice(&(chunk_size as u32).to_le_bytes()),
            UintWidth::Eight => buf.extend_from_slice(&(chunk_size as u64).to_le_bytes()),
        }

        buf.extend_from_slice(&msg_bytes);

        // Checksum (CRC32C of everything from OHDR to here)
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// A chunk of only Nil padding reserves nothing.
    ///
    /// The reservation is sized from a walk of the chunk. A `HeaderMessage` is
    /// much wider than the four-byte message prefix, so skipped Nil messages are
    /// excluded from the reservation count.
    #[test]
    fn a_chunk_of_padding_reserves_no_messages() {
        // 256 empty Nil messages: 1 KiB of chunk, none of it kept.
        let nils: Vec<(u8, &[u8], MessageFlags)> = vec![(0u8, &[][..], MessageFlags::NONE); 256];
        let data = build_v2_header(0x01, &nils, None);
        let header = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();

        assert!(header.messages.is_empty(), "Nil messages are not kept");
        assert_eq!(
            header.messages.capacity(),
            0,
            "a chunk the parse keeps nothing from must not reserve for it"
        );
    }

    #[test]
    fn parse_v1_zero_messages() {
        let data = build_v1_header(&[], 8, 8);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.messages.len(), 0);
        assert_eq!(hdr.reference_count, Some(1));
        assert_eq!(hdr.flags, 0);
    }

    #[test]
    fn parse_v1_two_messages() {
        let messages = [
            (0x0001u16, &[1u8, 2, 3, 4][..], MessageFlags::NONE), // Dataspace
            (0x0008, &[5u8, 6][..], MessageFlags::NONE),          // DataLayout
        ];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::DATASPACE);
        assert_eq!(hdr.messages[0].data, vec![1, 2, 3, 4]);
        assert_eq!(hdr.messages[1].msg_type, MessageType::DATA_LAYOUT);
        assert_eq!(hdr.messages[1].data, vec![5, 6]);
    }

    #[test]
    fn parse_v1_unknown_message_ok() {
        const UNKNOWN_TYPE: u16 = 0x00FF;
        let messages = [(UNKNOWN_TYPE, &[0xAA, 0xBB][..], MessageFlags::NONE)];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();

        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type.unknown_id(), Some(UNKNOWN_TYPE));
    }

    /// The must-understand guard applies only to message types the parser cannot
    /// name. External Data Files (`0x0007`) is a known type, so the header parses
    /// and the reader reports external storage when the message is used.
    ///
    /// The reference library writes flags `0x01` on this message. A crafted
    /// `0x08` flag exercises the guard. The same check applies to both header
    /// versions and their continuations. This test covers the version 1 header
    /// path.
    #[test]
    fn parse_v1_named_message_survives_must_understand() {
        let messages = [(
            0x0007u16,
            &[0xAA][..],
            MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
        )];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadWrite, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::EXTERNAL_DATA_FILES);
    }

    #[test]
    fn a_version_1_header_rejects_an_unknown_message_that_must_always_be_understood() {
        let messages = [(0x00FFu16, &[0xAA][..], MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)];
        let data = build_v1_header(&messages, 8, 8);
        let err = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    #[test]
    fn a_version_1_continuation_rejects_an_unknown_message_that_must_always_be_understood() {
        let cont_chunk =
            v1_message_records(&[(0x00FFu16, &[0xAA][..], MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)]);

        let cont_offset = 256usize;
        let mut cont_ptr = Vec::new();
        cont_ptr.extend_from_slice(&(cont_offset as u64).to_le_bytes());
        cont_ptr.extend_from_slice(&(cont_chunk.len() as u64).to_le_bytes());

        let header = build_v1_header(
            &[(
                MessageType::OBJECT_HEADER_CONTINUATION.to_u16(),
                &cont_ptr[..],
                MessageFlags::NONE,
            )],
            8,
            8,
        );
        let mut file_data = vec![0u8; cont_offset + cont_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[cont_offset..cont_offset + cont_chunk.len()].copy_from_slice(&cont_chunk);

        let err = ObjectHeader::parse(&file_data, AccessMode::ReadOnly, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    #[test]
    fn a_version_2_header_rejects_an_unknown_message_that_must_always_be_understood() {
        let data = build_v2_header(
            0x00,
            &[(0xFF, &[0xAA][..], MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)],
            None,
        );
        let err = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    #[test]
    fn a_streamed_version_1_header_rejects_an_unknown_message_that_must_always_be_understood() {
        let messages = [(0x00FFu16, &[0xAA][..], MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)];
        let data = build_v1_header(&messages, 8, 8);
        let err = ObjectHeader::parse_from_source(
            &BytesSource::new(&data),
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        )
        .unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    #[test]
    fn a_version_1_header_rejects_an_unknown_message_a_writer_must_understand_only_for_write() {
        const UNKNOWN_TYPE: u16 = 0x00FF;

        let messages = [(
            UNKNOWN_TYPE,
            &[0xAA][..],
            MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
        )];
        let data = build_v1_header(&messages, 8, 8);

        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();

        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type.unknown_id(), Some(UNKNOWN_TYPE));
        assert_eq!(hdr.messages[0].data, vec![0xAA]);

        let err = ObjectHeader::parse(&data, AccessMode::ReadWrite, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(UNKNOWN_TYPE));
    }

    #[test]
    fn a_version_1_continuation_rejects_an_unknown_message_a_writer_must_understand_only_for_write()
    {
        const UNKNOWN_TYPE: u16 = 0x00FF;

        let cont_chunk = v1_message_records(&[(
            UNKNOWN_TYPE,
            &[0xAA][..],
            MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
        )]);

        let cont_offset = 256usize;
        let mut cont_ptr = Vec::new();
        cont_ptr.extend_from_slice(&(cont_offset as u64).to_le_bytes());
        cont_ptr.extend_from_slice(&(cont_chunk.len() as u64).to_le_bytes());

        let header = build_v1_header(
            &[(
                MessageType::OBJECT_HEADER_CONTINUATION.to_u16(),
                &cont_ptr[..],
                MessageFlags::NONE,
            )],
            8,
            8,
        );
        let mut file_data = vec![0u8; cont_offset + cont_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[cont_offset..cont_offset + cont_chunk.len()].copy_from_slice(&cont_chunk);

        let hdr = ObjectHeader::parse(&file_data, AccessMode::ReadOnly, 0, 8, 8).unwrap();

        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(
            hdr.messages[0].msg_type,
            MessageType::OBJECT_HEADER_CONTINUATION
        );
        assert_eq!(hdr.messages[1].msg_type.unknown_id(), Some(UNKNOWN_TYPE));

        let err = ObjectHeader::parse(&file_data, AccessMode::ReadWrite, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(UNKNOWN_TYPE));
    }

    #[test]
    fn a_version_2_header_rejects_an_unknown_message_a_writer_must_understand_only_for_write() {
        let data = build_v2_header(
            0x00,
            &[(
                0xFF,
                &[0xAA][..],
                MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
            )],
            None,
        );

        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::from(0x00FF));
        assert_eq!(hdr.messages[0].data, vec![0xAA]);

        let err = ObjectHeader::parse(&data, AccessMode::ReadWrite, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    #[test]
    fn a_streamed_version_1_header_rejects_an_unknown_message_a_writer_must_understand_only_for_write()
     {
        let messages = [(
            0x00FFu16,
            &[0xAA][..],
            MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
        )];
        let data = build_v1_header(&messages, 8, 8);
        let source = BytesSource::new(&data);

        let hdr = ObjectHeader::parse_from_source(
            &source,
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        )
        .unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::from(0x00FF));
        assert_eq!(hdr.messages[0].data, vec![0xAA]);

        let err = ObjectHeader::parse_from_source(
            &source,
            AccessMode::ReadWrite,
            0,
            8,
            8,
            BaseAddress::ZERO,
        )
        .unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    #[test]
    fn parse_v2_no_timestamps_one_message() {
        let data = build_v2_header(0x00, &[(0x01, &[10, 20], MessageFlags::NONE)], None);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.flags, 0);
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::DATASPACE);
        assert_eq!(hdr.messages[0].data, vec![10, 20]);
        assert!(hdr.access_time.is_none());
    }

    #[test]
    fn parse_v2_with_timestamps() {
        let data = build_v2_header(
            0x20,
            &[(0x01, &[1], MessageFlags::NONE)],
            Some((100, 200, 300, 400)),
        );
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.access_time, Some(100));
        assert_eq!(hdr.modification_time, Some(200));
        assert_eq!(hdr.change_time, Some(300));
        assert_eq!(hdr.birth_time, Some(400));
        assert_eq!(hdr.messages.len(), 1);
        // flags bit 5 = timestamps, but bit 2 not set → no creation order in messages
        assert!(hdr.messages[0].creation_order.is_none());
    }

    #[test]
    fn parse_v2_creation_order() {
        // flags bit 2 enables attribute/message creation order tracking
        // flags bit 5 enables timestamps
        // Use 0x24 = bit 2 + bit 5
        let data = build_v2_header(
            0x24,
            &[
                (0x03, &[9], MessageFlags::NONE),
                (0x05, &[8], MessageFlags::NONE),
            ],
            Some((0, 0, 0, 0)),
        );
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert!(hdr.messages[0].creation_order.is_some());
        assert!(hdr.messages[1].creation_order.is_some());
        assert_eq!(hdr.access_time, Some(0));
    }

    #[test]
    fn parse_v2_checksum_valid() {
        let data = build_v2_header(0x00, &[(0x01, &[1, 2, 3], MessageFlags::NONE)], None);
        // The valid checksum allows parsing to succeed.
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_checksum_invalid() {
        let mut data = build_v2_header(0x00, &[(0x01, &[1, 2, 3], MessageFlags::NONE)], None);
        // Corrupt checksum
        let len = data.len();
        data[len - 1] ^= 0xFF;
        let err = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn parse_v2_nil_padding_skipped() {
        let data = build_v2_header(
            0x00,
            &[
                (0x00, &[0, 0, 0, 0], MessageFlags::NONE), // NIL
                (0x01, &[42], MessageFlags::NONE),         // Dataspace
            ],
            None,
        );
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::DATASPACE);
    }

    #[test]
    fn parse_v2_chunk_size_1byte() {
        // flags bits 0-1 = 0 → 1-byte chunk size
        let data = build_v2_header(0x00, &[(0x01, &[1], MessageFlags::NONE)], None);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_chunk_size_2byte() {
        let data = build_v2_header(0x01, &[(0x01, &[1], MessageFlags::NONE)], None);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_chunk_size_4byte() {
        let data = build_v2_header(0x02, &[(0x01, &[1], MessageFlags::NONE)], None);
        let hdr = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_continuation() {
        // Build a continuation chunk (OCHK) at a known offset
        let ochk_offset = 256usize;
        let ochk_msg_type = 0x03u8; // Datatype
        let ochk_msg_data = [0xDE, 0xAD];

        // Build the OCHK chunk
        let mut ochk_buf = Vec::new();
        ochk_buf.extend_from_slice(&v2::OCHK_SIGNATURE);
        ochk_buf.push(ochk_msg_type);
        ochk_buf.extend_from_slice(&(ochk_msg_data.len() as u16).to_le_bytes());
        ochk_buf.push(MessageFlags::NONE.get());
        ochk_buf.extend_from_slice(&ochk_msg_data);
        let checksum = crate::checksum::jenkins_lookup3(&ochk_buf);
        ochk_buf.extend_from_slice(&checksum.to_le_bytes());

        let ochk_length = ochk_buf.len();

        // Build continuation message data: offset(8 LE) + length(8 LE)
        let mut cont_data = Vec::new();
        cont_data.extend_from_slice(&(ochk_offset as u64).to_le_bytes());
        cont_data.extend_from_slice(&(ochk_length as u64).to_le_bytes());

        // Build main header with continuation message + a regular message
        let header = build_v2_header(
            0x00,
            &[
                (0x01, &[42], MessageFlags::NONE),      // Dataspace
                (0x10, &cont_data, MessageFlags::NONE), // Continuation
            ],
            None,
        );

        // Assemble full "file"
        let total_size = ochk_offset + ochk_buf.len();
        let mut file_data = vec![0u8; total_size];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[ochk_offset..ochk_offset + ochk_buf.len()].copy_from_slice(&ochk_buf);

        let hdr = ObjectHeader::parse(&file_data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::DATASPACE);
        assert_eq!(hdr.messages[1].msg_type, MessageType::DATATYPE);
        assert_eq!(hdr.messages[1].data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn truncated_v1_header() {
        let data = vec![1u8, 0]; // version 1, but too short
        let err = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn truncated_v2_header() {
        let data = [b'O', b'H', b'D', b'R', 2]; // signature + version, but no flags
        let err = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[cfg(feature = "std")]
    fn assert_same_header(a: &ObjectHeader, b: &ObjectHeader) {
        assert_eq!(a.version, b.version);
        assert_eq!(a.reference_count, b.reference_count);
        assert_eq!(a.flags, b.flags);
        assert_eq!(a.access_time, b.access_time);
        assert_eq!(a.modification_time, b.modification_time);
        assert_eq!(a.messages.len(), b.messages.len(), "message count");
        for (i, (x, y)) in a.messages.iter().zip(&b.messages).enumerate() {
            assert_eq!(x.msg_type, y.msg_type, "msg {i} type");
            assert_eq!(x.size, y.size, "msg {i} size");
            assert_eq!(x.flags, y.flags, "msg {i} flags");
            assert_eq!(x.creation_order, y.creation_order, "msg {i} creation_order");
            assert_eq!(x.data, y.data, "msg {i} data");
        }
    }

    #[cfg(feature = "std")]
    fn parse_three_ways(file_data: Vec<u8>, os: u8, ls: u8, base: BaseAddress) {
        use crate::source::{BytesSource, ReadSeekSource};
        let buffered =
            ObjectHeader::parse_with_base(&file_data, AccessMode::ReadOnly, 0, os, ls, base)
                .unwrap();
        let from_mem = ObjectHeader::parse_from_source(
            &BytesSource::new(&file_data),
            AccessMode::ReadOnly,
            0,
            os,
            ls,
            base,
        )
        .unwrap();
        let from_seek = ObjectHeader::parse_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap(),
            AccessMode::ReadOnly,
            0,
            os,
            ls,
            base,
        )
        .unwrap();
        assert_same_header(&buffered, &from_mem);
        assert_same_header(&buffered, &from_seek);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v2_simple_matches_buffered() {
        let header = build_v2_header(
            0x20,
            &[(0x01, &[1, 2, 3], MessageFlags::NONE)],
            Some((1, 2, 3, 4)),
        );
        parse_three_ways(header, 8, 8, BaseAddress::ZERO);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v2_with_continuation_matches_buffered() {
        // A v2 header at offset 0 has a continuation that points to an OCHK chunk at
        // offset 256. The streaming parser reads that chunk from the source and
        // produces the same messages in the same order.
        let ochk_msg_data = [0xDE, 0xAD];
        let mut ochk_buf = Vec::new();
        ochk_buf.extend_from_slice(&v2::OCHK_SIGNATURE);
        ochk_buf.push(0x03); // Datatype
        ochk_buf.extend_from_slice(&(ochk_msg_data.len() as u16).to_le_bytes());
        ochk_buf.push(MessageFlags::NONE.get());
        ochk_buf.extend_from_slice(&ochk_msg_data);
        let cks = crate::checksum::jenkins_lookup3(&ochk_buf);
        ochk_buf.extend_from_slice(&cks.to_le_bytes());

        let ochk_offset = 256usize;
        let mut cont_data = Vec::new();
        cont_data.extend_from_slice(&(ochk_offset as u64).to_le_bytes());
        cont_data.extend_from_slice(&(ochk_buf.len() as u64).to_le_bytes());

        let header = build_v2_header(
            0x00,
            &[
                (0x01, &[42], MessageFlags::NONE),
                (0x10, &cont_data, MessageFlags::NONE),
            ],
            None,
        );
        let mut file_data = vec![0u8; ochk_offset + ochk_buf.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[ochk_offset..ochk_offset + ochk_buf.len()].copy_from_slice(&ochk_buf);

        parse_three_ways(file_data, 8, 8, BaseAddress::ZERO);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v1_with_continuation_matches_buffered() {
        // A v1 header whose continuation points to a raw-message chunk at offset
        // 256 (v1 continuations have no signature). The buffered parser keeps the
        // continuation message in the list and follows it depth-first. The
        // streaming parser does the same.
        let cont_msg_data = [0xBE, 0xEF];
        let mut cont_chunk = Vec::new();
        cont_chunk.extend_from_slice(&0x03u16.to_le_bytes()); // Datatype
        cont_chunk.extend_from_slice(&(cont_msg_data.len() as u16).to_le_bytes());
        cont_chunk.push(MessageFlags::NONE.get());
        cont_chunk.extend_from_slice(&[0u8; 3]); // reserved
        cont_chunk.extend_from_slice(&cont_msg_data);

        let cont_offset = 256usize;
        let mut cont_ptr = Vec::new();
        cont_ptr.extend_from_slice(&(cont_offset as u64).to_le_bytes());
        cont_ptr.extend_from_slice(&(cont_chunk.len() as u64).to_le_bytes());

        let header = build_v1_header(
            &[
                (0x01, &[42][..], MessageFlags::NONE),
                (0x10, &cont_ptr[..], MessageFlags::NONE),
            ],
            8,
            8,
        );
        let mut file_data = vec![0u8; cont_offset + cont_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[cont_offset..cont_offset + cont_chunk.len()].copy_from_slice(&cont_chunk);

        parse_three_ways(file_data, 8, 8, BaseAddress::ZERO);
    }

    #[cfg(feature = "std")]
    #[test]
    fn version_1_message_overrunning_chunk0_is_rejected() {
        // Regression for #140: a v1 chunk 0 message whose data overruns the declared
        // object header size (`header_data_size`) is malformed. All parser paths
        // reject the message at the chunk boundary, so the continuation is not
        // followed.
        let cont_msg_data = [0xBE, 0xEF];
        let mut cont_chunk = Vec::new();
        cont_chunk.extend_from_slice(&0x03u16.to_le_bytes()); // Datatype
        cont_chunk.extend_from_slice(&(cont_msg_data.len() as u16).to_le_bytes());
        cont_chunk.push(MessageFlags::NONE.get());
        cont_chunk.extend_from_slice(&[0u8; 3]); // reserved
        cont_chunk.extend_from_slice(&cont_msg_data);

        let cont_offset = 256usize;
        let mut cont_ptr = Vec::new();
        cont_ptr.extend_from_slice(&(cont_offset as u64).to_le_bytes());
        cont_ptr.extend_from_slice(&(cont_chunk.len() as u64).to_le_bytes());

        // Build the v1 prefix by hand so `header_data_size` can be understated:
        // the sole continuation message occupies 8 (prefix) + 16 (pointer) = 24
        // bytes, but we declare only 16, so its data overruns chunk 0 by 8 bytes.
        let mut header = Vec::new();
        header.push(1); // version
        header.push(0); // reserved
        header.extend_from_slice(&1u16.to_le_bytes()); // `num_messages`
        header.extend_from_slice(&1u32.to_le_bytes()); // `reference_count`
        header.extend_from_slice(&16u32.to_le_bytes()); // `header_data_size` (understated)
        header.extend_from_slice(&[0u8; 4]); // pad prefix to 16 bytes
        header.extend_from_slice(&0x0010u16.to_le_bytes()); // Continuation
        header.extend_from_slice(&(cont_ptr.len() as u16).to_le_bytes()); // size = 16
        header.push(MessageFlags::NONE.get());
        header.extend_from_slice(&[0u8; 3]); // reserved
        header.extend_from_slice(&cont_ptr); // pointer (overruns chunk 0)

        let mut file_data = vec![0u8; cont_offset + cont_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[cont_offset..cont_offset + cont_chunk.len()].copy_from_slice(&cont_chunk);

        let buffered = ObjectHeader::parse_with_base(
            &file_data,
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        );
        assert!(
            matches!(buffered, Err(FormatError::UnexpectedEof { .. })),
            "buffered parser accepted a v1 message crossing the chunk boundary"
        );

        let from_mem = ObjectHeader::parse_from_source(
            &BytesSource::new(&file_data),
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        );
        assert!(
            matches!(from_mem, Err(FormatError::UnexpectedEof { .. })),
            "memory source parser accepted a v1 message crossing the chunk boundary"
        );

        let from_seek = ObjectHeader::parse_from_source(
            &crate::ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap(),
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        );
        assert!(
            matches!(from_seek, Err(FormatError::UnexpectedEof { .. })),
            "seek source parser accepted a v1 message crossing the chunk boundary"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn version_1_continuation_with_partial_message_prefix_is_rejected() {
        let cont_msg_data = [0xAB; 8];
        let mut cont_chunk = v1_message_records(&[(
            MessageType::DATATYPE.to_u16(),
            &cont_msg_data,
            MessageFlags::NONE,
        )]);

        // The complete Datatype record occupies 16 bytes. One additional byte
        // cannot form the eight-byte prefix of another version 1 message.
        cont_chunk.push(0);

        let cont_offset = 256usize;
        let mut cont_ptr = Vec::new();
        cont_ptr.extend_from_slice(&(cont_offset as u64).to_le_bytes());
        cont_ptr.extend_from_slice(&(cont_chunk.len() as u64).to_le_bytes());

        let header = build_v1_header(
            &[(
                MessageType::OBJECT_HEADER_CONTINUATION.to_u16(),
                &cont_ptr,
                MessageFlags::NONE,
            )],
            8,
            8,
        );

        let mut file_data = vec![0u8; cont_offset + cont_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[cont_offset..].copy_from_slice(&cont_chunk);

        let buffered = ObjectHeader::parse_with_base(
            &file_data,
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        );
        assert!(
            matches!(buffered, Err(FormatError::UnexpectedEof { .. })),
            "buffered parser accepted a partial v1 continuation prefix"
        );

        let from_mem = ObjectHeader::parse_from_source(
            &BytesSource::new(&file_data),
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        );
        assert!(
            matches!(from_mem, Err(FormatError::UnexpectedEof { .. })),
            "memory source parser accepted a partial v1 continuation prefix"
        );

        let from_seek = ObjectHeader::parse_from_source(
            &crate::ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap(),
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        );
        assert!(
            matches!(from_seek, Err(FormatError::UnexpectedEof { .. })),
            "seek source parser accepted a partial v1 continuation prefix"
        );
    }

    #[test]
    fn a_filtered_version_1_parse_follows_unretained_continuations() {
        let nested_data = [0xDE, 0xAD];
        let nested_chunk = v1_message_records(&[(
            MessageType::DATATYPE.to_u16(),
            &nested_data[..],
            MessageFlags::NONE,
        )]);

        let nested_offset = 512usize;
        let mut nested_ptr = Vec::new();
        nested_ptr.extend_from_slice(&(nested_offset as u64).to_le_bytes());
        nested_ptr.extend_from_slice(&(nested_chunk.len() as u64).to_le_bytes());

        let continuation_chunk = v1_message_records(&[(
            MessageType::OBJECT_HEADER_CONTINUATION.to_u16(),
            &nested_ptr[..],
            MessageFlags::NONE,
        )]);

        let continuation_offset = 256usize;
        let mut continuation_ptr = Vec::new();
        continuation_ptr.extend_from_slice(&(continuation_offset as u64).to_le_bytes());
        continuation_ptr.extend_from_slice(&(continuation_chunk.len() as u64).to_le_bytes());

        let header = build_v1_header(
            &[(
                MessageType::OBJECT_HEADER_CONTINUATION.to_u16(),
                &continuation_ptr[..],
                MessageFlags::NONE,
            )],
            8,
            8,
        );

        let mut file_data = vec![0u8; nested_offset + nested_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[continuation_offset..continuation_offset + continuation_chunk.len()]
            .copy_from_slice(&continuation_chunk);
        file_data[nested_offset..nested_offset + nested_chunk.len()].copy_from_slice(&nested_chunk);

        let mut keep_datatype = |msg_type: MessageType, _: &[u8]| msg_type == MessageType::DATATYPE;
        let buffered = ObjectHeader::parse_filtered(
            &file_data,
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
            MessageFilter::Only(&mut keep_datatype),
        )
        .unwrap();

        assert_eq!(buffered.messages.len(), 1);
        assert_eq!(buffered.messages[0].msg_type, MessageType::DATATYPE);
        assert_eq!(buffered.messages[0].data, nested_data);

        let source = BytesSource::new(&file_data);
        let mut keep_datatype = |msg_type: MessageType, _: &[u8]| msg_type == MessageType::DATATYPE;
        let streamed = ObjectHeader::parse_from_source_filtered(
            &source,
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
            MessageFilter::Only(&mut keep_datatype),
        )
        .unwrap();

        assert_eq!(streamed.messages.len(), 1);
        assert_eq!(streamed.messages[0].msg_type, MessageType::DATATYPE);
        assert_eq!(streamed.messages[0].data, nested_data);
    }

    #[test]
    fn a_filtered_version_1_parse_checks_must_understand_before_filtering() {
        const UNKNOWN_TYPE: u16 = 0x00FF;

        let data = build_v1_header(
            &[(
                UNKNOWN_TYPE,
                &[0xAA][..],
                MessageFlags::FAIL_IF_UNKNOWN_ALWAYS,
            )],
            8,
            8,
        );

        let mut buffered_filter_called = false;
        let err = {
            let mut drop_all = |_: MessageType, _: &[u8]| {
                buffered_filter_called = true;
                false
            };

            ObjectHeader::parse_filtered(
                &data,
                AccessMode::ReadOnly,
                0,
                8,
                8,
                BaseAddress::ZERO,
                MessageFilter::Only(&mut drop_all),
            )
        }
        .unwrap_err();

        assert_eq!(err, FormatError::UnsupportedMessage(UNKNOWN_TYPE));
        assert!(
            !buffered_filter_called,
            "must-understand validation must run before retained-message filtering"
        );

        let source = BytesSource::new(&data);
        let mut streamed_filter_called = false;
        let err = {
            let mut drop_all = |_: MessageType, _: &[u8]| {
                streamed_filter_called = true;
                false
            };

            ObjectHeader::parse_from_source_filtered(
                &source,
                AccessMode::ReadOnly,
                0,
                8,
                8,
                BaseAddress::ZERO,
                MessageFilter::Only(&mut drop_all),
            )
        }
        .unwrap_err();

        assert_eq!(err, FormatError::UnsupportedMessage(UNKNOWN_TYPE));
        assert!(
            !streamed_filter_called,
            "must-understand validation must run before retained-message filtering"
        );
    }
}
