//! HDF5 Object Header parsing (v1 and v2).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use crate::access_mode::AccessMode;
use crate::address::{BaseAddress, StoredAddress};
use crate::bytes::{ensure_len, read_length, read_offset, read_uint_width};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::Source;

mod v2;
use self::v2::{
    Continuation, HeaderFlags as V2HeaderFlags, MAX_CONTINUATIONS as MAX_V2_CONTINUATIONS,
};

const MAX_V1_CONTINUATION_DEPTH: u16 = 32;

#[derive(Clone, Copy)]
struct ParseContext {
    access_mode: AccessMode,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
}

/// OHDR signature for v2 object headers.
const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// OCHK signature for v2 continuation chunks.
const OCHK_SIGNATURE: [u8; 4] = *b"OCHK";

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

    fn parse_v1(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        // version(1) + reserved(1) + num_messages(2) + ref_count(4) + header_size(4) = 12
        // then pad to 8-byte alignment from start of header
        ensure_len(data, offset, 12)?;

        let version = data[offset];
        if version != 1 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }

        let num_messages = LittleEndian::read_u16(&data[offset + 2..offset + 4]);
        let reference_count = LittleEndian::read_u32(&data[offset + 4..offset + 8]);
        let header_data_size = LittleEndian::read_u32(&data[offset + 8..offset + 12]).to_usize()?;

        // Pad to 8-byte alignment: header prefix is 12 bytes, pad to 16
        let padding = 4; // pad 12-byte prefix to 16-byte alignment
        let msg_start = offset
            .checked_add(12 + padding)
            .ok_or(FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: data.len(),
            })?;

        ensure_len(data, msg_start, header_data_size)?;

        // The v1 header states its own message count, so the vector is sized
        // once. The count is capped by what the chunk can physically hold.
        // Each message requires an eight-byte header at minimum, so a larger
        // count cannot describe records contained by the chunk.
        let mut messages = Vec::with_capacity(if filter.keeps_all() {
            (num_messages as usize).min(header_data_size / 8)
        } else {
            0
        });
        let mut pos = msg_start;
        let msg_end =
            msg_start
                .checked_add(header_data_size)
                .ok_or(FormatError::UnexpectedEof {
                    expected: usize::MAX,
                    available: data.len(),
                })?;

        for _ in 0..num_messages {
            if pos + 8 > msg_end {
                break;
            }
            let msg_type_raw = LittleEndian::read_u16(&data[pos..pos + 2]);
            let msg_data_size = LittleEndian::read_u16(&data[pos + 2..pos + 4]) as usize;
            let msg_flags = MessageFlags::new(data[pos + 4]);
            // reserved(3) at pos+5..pos+8
            pos += 8;

            // A message must lie entirely within chunk 0 (`header_data_size`).
            // All parser paths enforce the chunk boundary so malformed headers
            // produce consistent results.
            if pos + msg_data_size > msg_end {
                break;
            }

            ensure_len(data, pos, msg_data_size)?;
            let msg_type = MessageType::from_u16(msg_type_raw);

            if let Some(id) = msg_type.unknown_id()
                && msg_flags.must_be_understood(context.access_mode)
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_body = &data[pos..pos + msg_data_size];
            if msg_type != MessageType::NIL && filter.keeps(msg_type, msg_body) {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order: None,
                    data: msg_body.to_vec(),
                });
            }

            pos += msg_data_size;

            // Follow continuations. The pointer is read directly from the message
            // body because a filtered parse may not have kept the message.
            // Continuations are followed independently of the filter.
            if msg_type == MessageType::OBJECT_HEADER_CONTINUATION
                && msg_body.len() >= (context.offset_size as usize + context.length_size as usize)
            {
                let cont_offset_raw =
                    StoredAddress::new(read_offset(msg_body, 0, context.offset_size)?);
                let cont_offset = context.base_address.absolute(cont_offset_raw)?.to_usize()?;
                let cont_length =
                    read_length(msg_body, context.offset_size as usize, context.length_size)?
                        .to_usize()?;
                // Parse continuation block (v1: just raw messages, no signature)
                let cont_msgs = Self::parse_v1_continuation(
                    data,
                    context,
                    cont_offset,
                    cont_length,
                    32, // max continuation depth
                    filter,
                )?;
                messages.extend(cont_msgs);
            }
        }

        Ok(ObjectHeader {
            version: 1,
            messages,
            reference_count: Some(reference_count),
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        })
    }

    fn parse_v1_continuation(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        length: usize,
        depth_remaining: u16,
        filter: &mut MessageFilter<'_>,
    ) -> Result<Vec<HeaderMessage>, FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::NestingDepthExceeded);
        }
        ensure_len(data, offset, length)?;
        let mut messages = Vec::new();
        let mut pos = offset;
        let end = offset.saturating_add(length);

        while pos + 8 <= end {
            let msg_type_raw = LittleEndian::read_u16(&data[pos..pos + 2]);
            let msg_data_size = LittleEndian::read_u16(&data[pos + 2..pos + 4]) as usize;
            let msg_flags = MessageFlags::new(data[pos + 4]);
            pos += 8;

            if pos + msg_data_size > end {
                break;
            }

            let msg_type = MessageType::from_u16(msg_type_raw);

            if let Some(id) = msg_type.unknown_id()
                && msg_flags.must_be_understood(context.access_mode)
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_body = &data[pos..pos + msg_data_size];
            if msg_type != MessageType::NIL && filter.keeps(msg_type, msg_body) {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order: None,
                    data: msg_body.to_vec(),
                });
            }

            pos += msg_data_size;

            // Recursive continuations, read from the body where it lies for the
            // reason [`parse_v1`] gives.
            if msg_type == MessageType::OBJECT_HEADER_CONTINUATION
                && msg_body.len() >= (context.offset_size as usize + context.length_size as usize)
            {
                let cont_offset_raw =
                    StoredAddress::new(read_offset(msg_body, 0, context.offset_size)?);
                let cont_offset = context.base_address.absolute(cont_offset_raw)?.to_usize()?;
                let cont_length =
                    read_length(msg_body, context.offset_size as usize, context.length_size)?
                        .to_usize()?;
                let cont_msgs = Self::parse_v1_continuation(
                    data,
                    context,
                    cont_offset,
                    cont_length,
                    depth_remaining - 1,
                    filter,
                )?;
                messages.extend(cont_msgs);
            }
        }

        Ok(messages)
    }

    fn parse_v2(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        // signature(4) + version(1) + flags(1) = 6
        ensure_len(data, offset, 6)?;

        let version = data[offset + 4];
        if version != 2 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }
        let flags = V2HeaderFlags::new(data[offset + 5]);

        let mut pos = offset + 6;

        let (access_time, modification_time, change_time, birth_time) = if flags.stores_times() {
            ensure_len(data, pos, 16)?;
            let at = LittleEndian::read_u32(&data[pos..pos + 4]);
            let mt = LittleEndian::read_u32(&data[pos + 4..pos + 8]);
            let ct = LittleEndian::read_u32(&data[pos + 8..pos + 12]);
            let bt = LittleEndian::read_u32(&data[pos + 12..pos + 16]);
            pos += 16;
            (Some(at), Some(mt), Some(ct), Some(bt))
        } else {
            (None, None, None, None)
        };

        // Optional attribute storage thresholds (flags bit 4)
        if flags.stores_attribute_phase_change() {
            ensure_len(data, pos, 4)?;
            // The attribute phase change values occupy four bytes.
            pos += 4;
        }

        let chunk_size_width = flags.chunk_size_width();
        let chunk0_size = read_uint_width(data, pos, chunk_size_width)?.to_usize()?;
        pos += usize::from(chunk_size_width.get());

        let chunk0_msg_start = pos;
        let chunk0_msg_end = pos
            .checked_add(chunk0_size)
            .ok_or(FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: data.len(),
            })?;

        // Validate checksum: from OHDR signature through all messages (before checksum)
        ensure_len(data, chunk0_msg_end, 4)?;
        #[cfg(feature = "checksum")]
        {
            let stored = LittleEndian::read_u32(&data[chunk0_msg_end..chunk0_msg_end + 4]);
            let computed = crate::checksum::jenkins_lookup3(&data[offset..chunk0_msg_end]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        let has_creation_order = flags.tracks_creation_order();

        // Parse messages from chunk0
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        Self::parse_v2_messages(
            data,
            context,
            chunk0_msg_start,
            chunk0_msg_end,
            has_creation_order,
            &mut messages,
            &mut continuations,
            filter,
        )?;

        // Follow continuations (limit to prevent cycles in malformed data). In
        // this buffered path the absolute position indexes the in-memory image,
        // so each address is narrowed (checked) to usize here.
        let mut cont_remaining = MAX_V2_CONTINUATIONS;
        while let Some(continuation) = continuations.pop() {
            if cont_remaining == 0 {
                return Err(FormatError::NestingDepthExceeded);
            }

            cont_remaining -= 1;
            let cont_offset = context.base_address.absolute(continuation.address)?;

            Self::parse_v2_continuation(
                data,
                context,
                cont_offset.to_usize()?,
                continuation.length.to_usize()?,
                has_creation_order,
                &mut messages,
                &mut continuations,
                filter,
            )?;
        }

        Ok(ObjectHeader {
            version: 2,
            messages,
            reference_count: None,
            flags: flags.raw(),
            access_time,
            modification_time,
            change_time,
            birth_time,
        })
    }

    /// Counting first allows one capacity reservation for the messages retained
    /// by an unfiltered parse. A group stores one Link message per child, so this
    /// avoids repeated vector growth during path resolution (issue #228).
    ///
    /// Nil and continuation messages are excluded because the message vector
    /// stores neither. A `HeaderMessage` is much wider than the four-byte message
    /// prefix, so counting skipped padding could reserve substantially more memory
    /// than the parser retains. Both loops stop when the remaining chunk tail
    /// cannot contain a complete message.
    ///
    /// Filtered parsing does not use this count because the filter determines how
    /// many messages are retained.
    fn count_v2_messages(data: &[u8], start: usize, end: usize, msg_header_size: usize) -> usize {
        let mut pos = start;
        let mut count = 0;
        while pos + msg_header_size <= end {
            let msg_type = MessageType::from_u16(data[pos] as u16);
            let msg_data_size = LittleEndian::read_u16(&data[pos + 1..pos + 3]) as usize;
            pos += msg_header_size;
            if pos + msg_data_size > end {
                break;
            }
            pos += msg_data_size;
            if msg_type != MessageType::NIL && msg_type != MessageType::OBJECT_HEADER_CONTINUATION {
                count += 1;
            }
        }
        count
    }

    fn parse_v2_messages(
        data: &[u8],
        context: ParseContext,
        start: usize,
        end: usize,
        has_creation_order: bool,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<Continuation>,
        filter: &mut MessageFilter<'_>,
    ) -> Result<(), FormatError> {
        let msg_header_size = if has_creation_order { 6 } else { 4 };
        if filter.keeps_all() {
            messages.reserve(Self::count_v2_messages(data, start, end, msg_header_size));
        }
        let mut pos = start;

        while pos + msg_header_size <= end {
            let msg_type_raw = data[pos] as u16;
            let msg_data_size = LittleEndian::read_u16(&data[pos + 1..pos + 3]) as usize;
            let msg_flags = MessageFlags::new(data[pos + 3]);
            let creation_order = if has_creation_order {
                Some(LittleEndian::read_u16(&data[pos + 4..pos + 6]))
            } else {
                None
            };
            pos += msg_header_size;

            if pos + msg_data_size > end {
                // Could be padding at end of chunk
                break;
            }

            let msg_type = MessageType::from_u16(msg_type_raw);

            if let Some(id) = msg_type.unknown_id()
                && msg_flags.must_be_understood(context.access_mode)
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_data = &data[pos..pos + msg_data_size];

            if msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                // Neither the offset nor the length is narrowed here, so that
                // the driver, buffered or streaming, can fetch a region a
                // 32-bit `usize` does not reach: a streaming reader follows a
                // continuation past 4 GiB on a 32-bit host.
                if msg_data.len() >= (context.offset_size as usize + context.length_size as usize) {
                    let address =
                        StoredAddress::new(read_offset(msg_data, 0, context.offset_size)?);
                    let length =
                        read_length(msg_data, context.offset_size as usize, context.length_size)?;

                    continuations.push(Continuation { address, length });
                }
            } else if msg_type != MessageType::NIL && filter.keeps(msg_type, msg_data) {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order,
                    data: msg_data.to_vec(),
                });
            }

            pos += msg_data_size;
        }

        Ok(())
    }

    fn parse_v2_continuation(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        length: usize,
        has_creation_order: bool,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<Continuation>,
        filter: &mut MessageFilter<'_>,
    ) -> Result<(), FormatError> {
        // OCHK signature(4) + messages + checksum(4)
        ensure_len(data, offset, length)?;
        if length < 8 {
            return Err(FormatError::UnexpectedEof {
                expected: 8,
                available: length,
            });
        }

        ensure_len(data, offset, 4)?;
        if data[offset..offset + 4] != OCHK_SIGNATURE {
            return Err(FormatError::InvalidObjectHeaderSignature);
        }

        let msg_start = offset + 4;
        let checksum_pos = offset + length - 4;

        #[cfg(feature = "checksum")]
        {
            let stored = LittleEndian::read_u32(&data[checksum_pos..checksum_pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&data[offset..checksum_pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        Self::parse_v2_messages(
            data,
            context,
            msg_start,
            checksum_pos,
            has_creation_order,
            messages,
            continuations,
            filter,
        )
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

    fn parse_v2_from_source<S: Source + ?Sized>(
        source: &S,
        context: ParseContext,
        address: u64,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        // The v2 prefix is bounded: sig(4) + ver(1) + flags(1) + optional
        // timestamps(16) + optional attribute thresholds(4) + chunk0-size
        // field(<=8) = at most 34 bytes. Read that window, then the chunk0 body.
        const MAX_PREFIX: u64 = 4 + 1 + 1 + 16 + 4 + 8;
        let head_len = MAX_PREFIX
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let head = source.read_metadata_at(address, head_len)?;
        if head.len() < 6 {
            return Err(FormatError::UnexpectedEof {
                expected: 6,
                available: head.len(),
            });
        }
        let version = head[4];
        if version != 2 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }
        let flags = V2HeaderFlags::new(head[5]);
        let mut pos = 6usize;

        let (access_time, modification_time, change_time, birth_time) = if flags.stores_times() {
            ensure_len(&head, pos, 16)?;
            let at = LittleEndian::read_u32(&head[pos..pos + 4]);
            let mt = LittleEndian::read_u32(&head[pos + 4..pos + 8]);
            let ct = LittleEndian::read_u32(&head[pos + 8..pos + 12]);
            let bt = LittleEndian::read_u32(&head[pos + 12..pos + 16]);
            pos += 16;
            (Some(at), Some(mt), Some(ct), Some(bt))
        } else {
            (None, None, None, None)
        };

        if flags.stores_attribute_phase_change() {
            ensure_len(&head, pos, 4)?;
            pos += 4;
        }

        let chunk_size_width = flags.chunk_size_width();
        let chunk0_size = read_uint_width(&head, pos, chunk_size_width)?.to_usize()?;
        pos += usize::from(chunk_size_width.get());
        let prefix_len = pos;

        // The chunk 0 body, including the prefix, messages, and checksum, is
        // contiguous from `address`. Reading the complete region gives the checksum
        // the same byte range as the buffered parser.
        let chunk0_end = prefix_len
            .checked_add(chunk0_size)
            .ok_or(FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: head.len(),
            })?;
        let chunk0_total =
            (chunk0_end as u64)
                .checked_add(4)
                .ok_or(FormatError::OffsetOverflow {
                    offset: chunk0_end as u64,
                    length: 4,
                })?;
        let chunk0 = source.read_metadata_at(address, chunk0_total.to_usize()?)?;

        #[cfg(feature = "checksum")]
        {
            let stored = LittleEndian::read_u32(&chunk0[chunk0_end..chunk0_end + 4]);
            let computed = crate::checksum::jenkins_lookup3(&chunk0[..chunk0_end]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        let has_creation_order = flags.tracks_creation_order();
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        Self::parse_v2_messages(
            &chunk0,
            context,
            prefix_len,
            chunk0_end,
            has_creation_order,
            &mut messages,
            &mut continuations,
            filter,
        )?;

        // Follow continuations by reading each (bounded) chunk from the source.
        let mut cont_remaining = MAX_V2_CONTINUATIONS;
        while let Some(continuation) = continuations.pop() {
            if cont_remaining == 0 {
                return Err(FormatError::NestingDepthExceeded);
            }

            cont_remaining -= 1;
            let length = continuation.length.to_usize()?;
            let address = context.base_address.absolute(continuation.address)?;

            let region = source.read_metadata_at(address, length)?;

            Self::parse_v2_continuation(
                &region,
                context,
                0,
                length,
                has_creation_order,
                &mut messages,
                &mut continuations,
                filter,
            )?;
        }

        Ok(ObjectHeader {
            version: 2,
            messages,
            reference_count: None,
            flags: flags.raw(),
            access_time,
            modification_time,
            change_time,
            birth_time,
        })
    }

    fn parse_v1_from_source<S: Source + ?Sized>(
        source: &S,
        context: ParseContext,
        address: u64,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        // version(1) + reserved(1) + num_messages(2) + ref_count(4) +
        // header_size(4) = 12, padded to 16 before the first message.
        let prefix = source.read_metadata_at(address, 16)?;
        let version = prefix[0];
        if version != 1 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }
        let num_messages = LittleEndian::read_u16(&prefix[2..4]);
        let reference_count = LittleEndian::read_u32(&prefix[4..8]);
        let header_data_size = u64::from(LittleEndian::read_u32(&prefix[8..12]));

        // Capacity is bounded by the header's message count and the number of
        // minimum-size records the chunk can hold. [`parse_v1`] uses the same bound.
        let mut messages = Vec::with_capacity(if filter.keeps_all() {
            (num_messages as usize).min((header_data_size / 8).to_usize()?)
        } else {
            0
        });
        Self::parse_v1_chunk_from_source(
            source,
            context,
            address + 16,
            header_data_size,
            num_messages,
            MAX_V1_CONTINUATION_DEPTH,
            &mut messages,
            filter,
        )?;

        Ok(ObjectHeader {
            version: 1,
            messages,
            reference_count: Some(reference_count),
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        })
    }

    /// Parse the messages of one v1 header chunk read from the source, following
    /// each continuation depth-first (as the buffered v1 parser does) so the
    /// resulting message order is identical.
    fn parse_v1_chunk_from_source<S: Source + ?Sized>(
        source: &S,
        context: ParseContext,
        region_addr: u64,
        region_len: u64,
        max_messages: u16,
        depth_remaining: u16,
        messages: &mut Vec<HeaderMessage>,
        filter: &mut MessageFilter<'_>,
    ) -> Result<(), FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::NestingDepthExceeded);
        }
        let region = source.read_metadata_at(region_addr, region_len.to_usize()?)?;
        let end = region.len();
        let mut pos = 0usize;
        let mut count = 0u16;

        while count < max_messages && pos + 8 <= end {
            let msg_type_raw = LittleEndian::read_u16(&region[pos..pos + 2]);
            let msg_data_size = LittleEndian::read_u16(&region[pos + 2..pos + 4]) as usize;
            let msg_flags = MessageFlags::new(region[pos + 4]);
            // reserved(3) at pos+5..pos+8
            pos += 8;

            if pos + msg_data_size > end {
                break;
            }
            count += 1;

            let msg_type = MessageType::from_u16(msg_type_raw);
            if let Some(id) = msg_type.unknown_id()
                && msg_flags.must_be_understood(context.access_mode)
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_data = &region[pos..pos + msg_data_size];
            pos += msg_data_size;

            // Decode the continuation pointer (if any) from the body where it
            // lies: it is followed whatever the filter kept, as in [`parse_v1`].
            let cont = if msg_type == MessageType::OBJECT_HEADER_CONTINUATION
                && msg_data.len() >= (context.offset_size as usize + context.length_size as usize)
            {
                let off_raw = StoredAddress::new(read_offset(msg_data, 0, context.offset_size)?);
                let len = read_length(msg_data, context.offset_size as usize, context.length_size)?;
                Some((off_raw, len))
            } else {
                None
            };

            if msg_type != MessageType::NIL && filter.keeps(msg_type, msg_data) {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order: None,
                    data: msg_data.to_vec(),
                });
            }

            if let Some((off_raw, len)) = cont {
                let cont_off = context.base_address.absolute(off_raw)?;
                Self::parse_v1_chunk_from_source(
                    source,
                    context,
                    cont_off,
                    len,
                    u16::MAX,
                    depth_remaining - 1,
                    messages,
                    filter,
                )?;
            }
        }

        Ok(())
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
        let header_flags = V2HeaderFlags::new(flags);
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
        ochk_buf.extend_from_slice(&OCHK_SIGNATURE);
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

    // ---- Streaming parser equivalence -------------------------------------

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
        ochk_buf.extend_from_slice(&OCHK_SIGNATURE);
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
    fn streaming_v1_message_overrunning_chunk0_matches_buffered() {
        // Regression for #140: a v1 chunk 0 message whose data overruns the declared
        // object header size (`header_data_size`). All parser paths stop at the chunk
        // boundary, so the malformed continuation is not read or followed.
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

        // All three backends agree on an empty message list because the overrunning
        // continuation is dropped and its Datatype message is unreachable.
        parse_three_ways(file_data.clone(), 8, 8, BaseAddress::ZERO);
        let buffered = ObjectHeader::parse_with_base(
            &file_data,
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        )
        .unwrap();
        assert_eq!(buffered.messages.len(), 0);
    }
}
