//! HDF5 Object Header parsing (v1 and v2).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use crate::address::BaseAddress;
use crate::bytes::{ensure_len, read_length, read_offset, read_uint_width};
use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::source::Source;

/// OHDR signature for v2 object headers.
const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// OCHK signature for v2 continuation chunks.
const OCHK_SIGNATURE: [u8; 4] = *b"OCHK";

/// Which of a header's messages a parse keeps.
///
/// A parse owns a copy of every message it keeps, so a caller looking for one
/// message in a header that holds a thousand pays for a thousand. Path
/// resolution is that caller: a group of *n* children is *n* Link messages in
/// its object header, and it needs the one link it was asked for. Naming that
/// up front costs the header walk (which reads the bytes either way) and one
/// message body, instead of one allocation per child on every lookup — the
/// difference between opening each child of a group being linear and quadratic
/// in the group's size (issue #228).
///
/// A filter is an optimization, never a decision: dropping a message must not
/// change what the parse *means*, so a filter that cannot tell keeps the
/// message and lets the reader that understands it decide. Continuation
/// messages are followed whatever the filter says.
pub(crate) enum MessageFilter<'f> {
    /// Keep every message — what every caller but a targeted lookup wants.
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
    /// Message flags byte.
    pub flags: u8,
    /// Creation order (v2 only, when tracking is enabled).
    pub creation_order: Option<u16>,
    /// Raw message data bytes.
    pub data: Vec<u8>,
}

/// Parsed HDF5 object header.
///
/// The version/refcount/flags and the four v2 timestamp fields are decoded from
/// the header for on-disk-format completeness but are not consulted by the
/// current reader; kept to document the format.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Header version (1 or 2).
    pub version: u8,
    /// All non-NIL messages collected from all chunks.
    pub messages: Vec<HeaderMessage>,
    /// Object reference count (v1 only).
    pub reference_count: Option<u32>,
    /// Object header flags (v2 only; 0 for v1).
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
    /// Parse an object header at the given offset in the data buffer.
    ///
    /// `offset_size` and `length_size` come from the superblock.
    pub fn parse(
        data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_with_base(data, offset, offset_size, length_size, BaseAddress::ZERO)
    }

    /// Parse an object header, applying `base_address` to continuation offsets.
    ///
    /// A continuation message stores a *file address*, and the format specification
    /// says of the superblock's base address that "unless otherwise noted, all
    /// other file addresses are relative to this base address". So for a file with
    /// a userblock (a `.mat` carries a 512-byte one) the stored offset is short of
    /// the real position by exactly the base, and this method adds it back before
    /// reading the block. Both header versions store the address the same way.
    ///
    /// `offset` itself is already absolute: every caller resolves an address to a
    /// file position before parsing there.
    pub fn parse_with_base(
        data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_filtered(
            data,
            offset,
            offset_size,
            length_size,
            base_address,
            MessageFilter::All,
        )
    }

    /// Parse an object header, keeping only the messages `filter` names.
    ///
    /// The header is read and validated exactly as [`parse_with_base`](Self::parse_with_base)
    /// reads it — same checksums, same refusals, same continuations followed —
    /// and the result differs only in which messages it carries. See
    /// [`MessageFilter`] for when that is worth asking for.
    pub(crate) fn parse_filtered(
        data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
        mut filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        ensure_len(data, offset, 4)?;
        if data[offset..offset + 4] == OHDR_SIGNATURE {
            Self::parse_v2(
                data,
                offset,
                offset_size,
                length_size,
                base_address,
                &mut filter,
            )
        } else {
            Self::parse_v1(
                data,
                offset,
                offset_size,
                length_size,
                base_address,
                &mut filter,
            )
        }
    }

    fn parse_v1(
        data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
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
        // once rather than grown — see [`count_v2_messages`] for why that is
        // worth doing. The count comes from the file, so it is capped by what
        // the chunk can physically hold: a message is a header (8 bytes) at
        // minimum, so a claim beyond that is a malformed file's, not a size.
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
            let msg_flags = data[pos + 4];
            // reserved(3) at pos+5..pos+8
            pos += 8;

            // A message must lie entirely within chunk 0 (`header_data_size`).
            // The buffered continuation parser and both streaming parsers already
            // enforce this; without the same check here, a message that overruns
            // `msg_end` would be read from the whole-file buffer and followed,
            // while the streaming backend stops at the chunk boundary — the two
            // backends would then disagree on a malformed header (issue #140).
            if pos + msg_data_size > msg_end {
                break;
            }

            ensure_len(data, pos, msg_data_size)?;
            let msg_type = MessageType::from_u16(msg_type_raw);

            // Check if unknown + must-understand (bit 3 of msg_flags)
            if let MessageType::Unknown(id) = msg_type
                && msg_flags & 0x08 != 0
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_body = &data[pos..pos + msg_data_size];
            if msg_type != MessageType::Nil && filter.keeps(msg_type, msg_body) {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order: None,
                    data: msg_body.to_vec(),
                });
            }

            pos += msg_data_size;

            // Follow continuations. The pointer is read from the message body
            // where it lies rather than from the message just pushed: a filtered
            // parse may not have kept it, and a continuation is followed whatever
            // the filter says.
            if msg_type == MessageType::ObjectHeaderContinuation
                && msg_body.len() >= (offset_size as usize + length_size as usize)
            {
                let cont_offset_raw = read_offset(msg_body, 0, offset_size)?;
                let cont_offset = base_address.absolute(cont_offset_raw)?.to_usize()?;
                let cont_length =
                    read_length(msg_body, offset_size as usize, length_size)?.to_usize()?;
                // Parse continuation block (v1: just raw messages, no signature)
                let cont_msgs = Self::parse_v1_continuation(
                    data,
                    cont_offset,
                    cont_length,
                    offset_size,
                    length_size,
                    base_address,
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

    #[allow(clippy::too_many_arguments)]
    fn parse_v1_continuation(
        data: &[u8],
        offset: usize,
        length: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
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
            let msg_flags = data[pos + 4];
            pos += 8;

            if pos + msg_data_size > end {
                break;
            }

            let msg_type = MessageType::from_u16(msg_type_raw);

            if let MessageType::Unknown(id) = msg_type
                && msg_flags & 0x08 != 0
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_body = &data[pos..pos + msg_data_size];
            if msg_type != MessageType::Nil && filter.keeps(msg_type, msg_body) {
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
            if msg_type == MessageType::ObjectHeaderContinuation
                && msg_body.len() >= (offset_size as usize + length_size as usize)
            {
                let cont_offset_raw = read_offset(msg_body, 0, offset_size)?;
                let cont_offset = base_address.absolute(cont_offset_raw)?.to_usize()?;
                let cont_length =
                    read_length(msg_body, offset_size as usize, length_size)?.to_usize()?;
                let cont_msgs = Self::parse_v1_continuation(
                    data,
                    cont_offset,
                    cont_length,
                    offset_size,
                    length_size,
                    base_address,
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
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        // signature(4) + version(1) + flags(1) = 6
        ensure_len(data, offset, 6)?;

        let version = data[offset + 4];
        if version != 2 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }
        let flags = data[offset + 5];

        let mut pos = offset + 6;

        // Optional timestamps (flags bit 5)
        let (access_time, modification_time, change_time, birth_time) = if flags & 0x20 != 0 {
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
        if flags & 0x10 != 0 {
            ensure_len(data, pos, 4)?;
            // max_compact_attrs(2) + min_dense_attrs(2) — read but don't store for now
            pos += 4;
        }

        // chunk0 size: width depends on flags bits 0-1
        let chunk_size_width = match flags & 0x03 {
            0 => 1u8,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        };
        ensure_len(data, pos, chunk_size_width as usize)?;
        let chunk0_size = read_uint_width(data, pos, chunk_size_width)?.to_usize()?;
        pos += chunk_size_width as usize;

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

        // Bit 2: attribute creation order tracked → messages include creation order field
        let has_creation_order = flags & 0x04 != 0;

        // Parse messages from chunk0
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        Self::parse_v2_messages(
            data,
            chunk0_msg_start,
            chunk0_msg_end,
            has_creation_order,
            offset_size,
            length_size,
            &mut messages,
            &mut continuations,
            filter,
        )?;

        // Follow continuations (limit to prevent cycles in malformed data).
        // Offsets are u64 file addresses relative to the superblock base; in this
        // buffered path the absolute position indexes the in-memory image, so add
        // the base and narrow (checked) to usize here.
        let mut cont_remaining = 256u16;
        while let Some((cont_offset, cont_length)) = continuations.pop() {
            if cont_remaining == 0 {
                return Err(FormatError::NestingDepthExceeded);
            }
            cont_remaining -= 1;
            let cont_offset = base_address.absolute(cont_offset)?;
            Self::parse_v2_continuation(
                data,
                cont_offset.to_usize()?,
                cont_length.to_usize()?,
                has_creation_order,
                offset_size,
                length_size,
                &mut messages,
                &mut continuations,
                filter,
            )?;
        }

        Ok(ObjectHeader {
            version: 2,
            messages,
            reference_count: None,
            flags,
            access_time,
            modification_time,
            change_time,
            birth_time,
        })
    }

    /// How many messages the chunk `[start, end)` holds, by the same stepping
    /// [`parse_v2_messages`](Self::parse_v2_messages) does.
    ///
    /// Walking the chunk a second time buys the parse one allocation instead of
    /// a doubling sequence: a group stores one Link message per child, so a
    /// 1,024-child group's header grew that vector ten times over and copied
    /// 187 KiB doing it — on *every* path resolution, since each one parses the
    /// group header afresh (issue #228).
    ///
    /// It counts only the messages the parse *keeps*, which matters for more than
    /// accuracy: a `HeaderMessage` is an order of magnitude wider than the 4-byte
    /// message prefix a chunk can be filled with, so counting the ones that are
    /// skipped would let a chunk of nothing but Nil padding — which the parse
    /// stores none of — reserve about twelve bytes for every byte of a file that
    /// need not be well formed. A chunk whose tail is padding rather than a
    /// message stops both loops at the same place.
    ///
    /// It is still a reservation, so being off by a few is harmless: what a
    /// filter drops is not known here, which is why the caller only reserves when
    /// the filter keeps everything.
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
            if msg_type != MessageType::Nil && msg_type != MessageType::ObjectHeaderContinuation {
                count += 1;
            }
        }
        count
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_v2_messages(
        data: &[u8],
        start: usize,
        end: usize,
        has_creation_order: bool,
        offset_size: u8,
        length_size: u8,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<(u64, u64)>,
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
            let msg_flags = data[pos + 3];
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

            if let MessageType::Unknown(id) = msg_type
                && msg_flags & 0x08 != 0
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_data = &data[pos..pos + msg_data_size];

            if msg_type == MessageType::ObjectHeaderContinuation {
                // The continuation offset/length are file offsets; keep them as
                // u64 so the driver (buffered or streaming) can fetch that
                // region — a streaming reader can then follow a continuation
                // past 4 GiB on a 32-bit host.
                if msg_data.len() >= (offset_size as usize + length_size as usize) {
                    let cont_off = read_offset(msg_data, 0, offset_size)?;
                    let cont_len = read_length(msg_data, offset_size as usize, length_size)?;
                    continuations.push((cont_off, cont_len));
                }
            } else if msg_type != MessageType::Nil && filter.keeps(msg_type, msg_data) {
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

    #[allow(clippy::too_many_arguments)]
    fn parse_v2_continuation(
        data: &[u8],
        offset: usize,
        length: usize,
        has_creation_order: bool,
        offset_size: u8,
        length_size: u8,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<(u64, u64)>,
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
            msg_start,
            checksum_pos,
            has_creation_order,
            offset_size,
            length_size,
            messages,
            continuations,
            filter,
        )
    }

    // -----------------------------------------------------------------------
    // Streaming parsers (read each header chunk from a `Source` on demand)
    // -----------------------------------------------------------------------

    /// Parse an object header from a [`Source`], reading each header chunk
    /// (and continuation chunk) as a small bounded window via
    /// [`Source::read_at`] rather than indexing a whole-file buffer.
    ///
    /// `base_address` is added to v1 continuation offsets (as in
    /// [`Self::parse_with_base`]). The result is identical to the buffered
    /// parser; this path simply never holds more than one chunk at a time, so it
    /// works against a file larger than the address space on a 32-bit host.
    pub fn parse_from_source<S: Source + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_from_source_filtered(
            source,
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
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
        mut filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        let mut sig = [0u8; 4];
        source.read_at(address, &mut sig)?;
        if sig == OHDR_SIGNATURE {
            Self::parse_v2_from_source(
                source,
                address,
                offset_size,
                length_size,
                base_address,
                &mut filter,
            )
        } else {
            Self::parse_v1_from_source(
                source,
                address,
                offset_size,
                length_size,
                base_address,
                &mut filter,
            )
        }
    }

    fn parse_v2_from_source<S: Source + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
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
        let flags = head[5];
        let mut pos = 6usize;

        let (access_time, modification_time, change_time, birth_time) = if flags & 0x20 != 0 {
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

        if flags & 0x10 != 0 {
            ensure_len(&head, pos, 4)?;
            pos += 4;
        }

        let chunk_size_width = match flags & 0x03 {
            0 => 1u8,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        };
        ensure_len(&head, pos, chunk_size_width as usize)?;
        let chunk0_size = read_uint_width(&head, pos, chunk_size_width)?.to_usize()?;
        pos += chunk_size_width as usize;
        let prefix_len = pos;

        // The chunk0 body (prefix + messages + 4-byte checksum) is contiguous
        // from `address`; read it all so the checksum covers the same bytes the
        // buffered parser hashes.
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

        let has_creation_order = flags & 0x04 != 0;
        let mut messages = Vec::new();
        let mut continuations: Vec<(u64, u64)> = Vec::new();
        Self::parse_v2_messages(
            &chunk0,
            prefix_len,
            chunk0_end,
            has_creation_order,
            offset_size,
            length_size,
            &mut messages,
            &mut continuations,
            filter,
        )?;

        // Follow continuations by reading each (bounded) chunk from the source.
        let mut cont_remaining = 256u16;
        while let Some((cont_off, cont_len)) = continuations.pop() {
            if cont_remaining == 0 {
                return Err(FormatError::NestingDepthExceeded);
            }
            cont_remaining -= 1;
            let cont_len = cont_len.to_usize()?;
            let region = source.read_metadata_at(base_address.absolute(cont_off)?, cont_len)?;
            Self::parse_v2_continuation(
                &region,
                0,
                cont_len,
                has_creation_order,
                offset_size,
                length_size,
                &mut messages,
                &mut continuations,
                filter,
            )?;
        }

        Ok(ObjectHeader {
            version: 2,
            messages,
            reference_count: None,
            flags,
            access_time,
            modification_time,
            change_time,
            birth_time,
        })
    }

    fn parse_v1_from_source<S: Source + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
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

        // Sized from the header's own message count, bounded by what the chunk
        // can hold — see the same reservation in [`parse_v1`].
        let mut messages = Vec::with_capacity(if filter.keeps_all() {
            (num_messages as usize).min((header_data_size / 8).to_usize()?)
        } else {
            0
        });
        Self::parse_v1_chunk_from_source(
            source,
            address + 16,
            header_data_size,
            num_messages,
            offset_size,
            length_size,
            base_address,
            32,
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
    #[allow(clippy::too_many_arguments)]
    fn parse_v1_chunk_from_source<S: Source + ?Sized>(
        source: &S,
        region_addr: u64,
        region_len: u64,
        max_messages: u16,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
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
            let msg_flags = region[pos + 4];
            // reserved(3) at pos+5..pos+8
            pos += 8;

            if pos + msg_data_size > end {
                break;
            }
            count += 1;

            let msg_type = MessageType::from_u16(msg_type_raw);
            if let MessageType::Unknown(id) = msg_type
                && msg_flags & 0x08 != 0
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            let msg_data = &region[pos..pos + msg_data_size];
            pos += msg_data_size;

            // Decode the continuation pointer (if any) from the body where it
            // lies: it is followed whatever the filter kept, as in [`parse_v1`].
            let cont = if msg_type == MessageType::ObjectHeaderContinuation
                && msg_data.len() >= (offset_size as usize + length_size as usize)
            {
                let off_raw = read_offset(msg_data, 0, offset_size)?;
                let len = read_length(msg_data, offset_size as usize, length_size)?;
                Some((off_raw, len))
            } else {
                None
            };

            if msg_type != MessageType::Nil && filter.keeps(msg_type, msg_data) {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data_size,
                    flags: msg_flags,
                    creation_order: None,
                    data: msg_data.to_vec(),
                });
            }

            if let Some((off_raw, len)) = cont {
                let cont_off = base_address.absolute(off_raw)?;
                Self::parse_v1_chunk_from_source(
                    source,
                    cont_off,
                    len,
                    u16::MAX,
                    offset_size,
                    length_size,
                    base_address,
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

    // Helper: build a v1 object header with given messages
    fn build_v1_header(
        messages: &[(u16, &[u8], u8)], // (type, data, flags)
        offset_size: u8,
        length_size: u8,
    ) -> Vec<u8> {
        let _ = (offset_size, length_size);
        // Calculate total header message data size
        let mut msg_bytes = Vec::new();
        for (mtype, mdata, mflags) in messages {
            msg_bytes.extend_from_slice(&mtype.to_le_bytes()); // type(2)
            msg_bytes.extend_from_slice(&(mdata.len() as u16).to_le_bytes()); // size(2)
            msg_bytes.push(*mflags); // flags(1)
            msg_bytes.extend_from_slice(&[0u8; 3]); // reserved(3)
            msg_bytes.extend_from_slice(mdata); // data
        }

        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(messages.len() as u16).to_le_bytes()); // num_messages
        buf.extend_from_slice(&1u32.to_le_bytes()); // reference_count
        buf.extend_from_slice(&(msg_bytes.len() as u32).to_le_bytes()); // header_data_size
        // Pad to 8-byte alignment (12 bytes so far, pad 4)
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&msg_bytes);
        buf
    }

    // Helper: build a v2 object header chunk0 with given messages
    fn build_v2_header(
        flags: u8,
        messages: &[(u8, &[u8], u8)], // (type, data, msg_flags)
        timestamps: Option<(u32, u32, u32, u32)>,
    ) -> Vec<u8> {
        let has_creation_order = flags & 0x04 != 0;
        let has_timestamps = flags & 0x20 != 0;
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

        if flags & 0x10 != 0 {
            buf.extend_from_slice(&8u16.to_le_bytes()); // max_compact
            buf.extend_from_slice(&6u16.to_le_bytes()); // min_dense
        }

        // Build message bytes to get chunk size
        let mut msg_bytes = Vec::new();
        for (mtype, mdata, mflags) in messages {
            msg_bytes.push(*mtype); // type(1)
            msg_bytes.extend_from_slice(&(mdata.len() as u16).to_le_bytes()); // size(2)
            msg_bytes.push(*mflags); // flags(1)
            if has_creation_order {
                msg_bytes.extend_from_slice(&0u16.to_le_bytes()); // creation_order(2)
            }
            msg_bytes.extend_from_slice(mdata);
        }

        let chunk_size = msg_bytes.len();
        // Write chunk size based on flags bits 0-1
        match flags & 0x03 {
            0 => buf.push(chunk_size as u8),
            1 => buf.extend_from_slice(&(chunk_size as u16).to_le_bytes()),
            2 => buf.extend_from_slice(&(chunk_size as u32).to_le_bytes()),
            3 => buf.extend_from_slice(&(chunk_size as u64).to_le_bytes()),
            _ => unreachable!(),
        }

        buf.extend_from_slice(&msg_bytes);

        // Checksum (CRC32C of everything from OHDR to here)
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// A chunk of nothing but Nil padding reserves nothing.
    ///
    /// The reservation is sized from a walk of the chunk, and a `HeaderMessage`
    /// is an order of magnitude wider than the 4-byte prefix a chunk can be
    /// filled with — so counting the messages the parse *skips* would let a file
    /// of padding make the parser reserve about twelve bytes per byte of it.
    #[test]
    fn a_chunk_of_padding_reserves_no_messages() {
        // 256 empty Nil messages: 1 KiB of chunk, none of it kept.
        let nils: Vec<(u8, &[u8], u8)> = vec![(0u8, &[][..], 0u8); 256];
        let data = build_v2_header(0x01, &nils, None);
        let header = ObjectHeader::parse(&data, 0, 8, 8).unwrap();

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
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.messages.len(), 0);
        assert_eq!(hdr.reference_count, Some(1));
        assert_eq!(hdr.flags, 0);
    }

    #[test]
    fn parse_v1_two_messages() {
        let messages = [
            (0x0001u16, &[1u8, 2, 3, 4][..], 0u8), // Dataspace
            (0x0008, &[5u8, 6][..], 0),            // DataLayout
        ];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[0].data, vec![1, 2, 3, 4]);
        assert_eq!(hdr.messages[1].msg_type, MessageType::DataLayout);
        assert_eq!(hdr.messages[1].data, vec![5, 6]);
    }

    #[test]
    fn parse_v1_unknown_message_ok() {
        let messages = [(0x00FFu16, &[0xAA, 0xBB][..], 0u8)];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Unknown(0x00FF));
    }

    #[test]
    fn parse_v1_unknown_must_understand_errors() {
        // Bit 3 of msg_flags = must understand
        let messages = [(0x00FFu16, &[0xAA][..], 0x08u8)];
        let data = build_v1_header(&messages, 8, 8);
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::UnsupportedMessage(0x00FF));
    }

    /// The must-understand guard fires on a message this parser cannot *name*,
    /// not on every message it declines to act on. External Data Files (0x0007)
    /// is named as of #331, so a header carrying it parses and the refusal moves
    /// to the read, which reports external storage rather than an unsupported
    /// message id.
    ///
    /// The reference library writes flags `0x01` (constant) on this message, not
    /// `0x08`, so only a crafted file reaches the guard at all. The same guard is
    /// written four times over — version 1 and version 2 headers, each with its
    /// continuation form — and loosens identically in all four; this pins the
    /// version 1 one.
    #[test]
    fn parse_v1_named_message_survives_must_understand() {
        let messages = [(0x0007u16, &[0xAA][..], 0x08u8)];
        let data = build_v1_header(&messages, 8, 8);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::ExternalDataFiles);
    }

    #[test]
    fn parse_v2_no_timestamps_one_message() {
        let data = build_v2_header(0x00, &[(0x01, &[10, 20], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.flags, 0);
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[0].data, vec![10, 20]);
        assert!(hdr.access_time.is_none());
    }

    #[test]
    fn parse_v2_with_timestamps() {
        let data = build_v2_header(0x20, &[(0x01, &[1], 0)], Some((100, 200, 300, 400)));
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
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
            &[(0x03, &[9], 0), (0x05, &[8], 0)],
            Some((0, 0, 0, 0)),
        );
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert!(hdr.messages[0].creation_order.is_some());
        assert!(hdr.messages[1].creation_order.is_some());
        assert_eq!(hdr.access_time, Some(0));
    }

    #[test]
    fn parse_v2_checksum_valid() {
        let data = build_v2_header(0x00, &[(0x01, &[1, 2, 3], 0)], None);
        // Should succeed — checksum is valid
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_checksum_invalid() {
        let mut data = build_v2_header(0x00, &[(0x01, &[1, 2, 3], 0)], None);
        // Corrupt checksum
        let len = data.len();
        data[len - 1] ^= 0xFF;
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn parse_v2_nil_padding_skipped() {
        let data = build_v2_header(
            0x00,
            &[
                (0x00, &[0, 0, 0, 0], 0), // NIL
                (0x01, &[42], 0),         // Dataspace
            ],
            None,
        );
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
    }

    #[test]
    fn parse_v2_chunk_size_1byte() {
        // flags bits 0-1 = 0 → 1-byte chunk size
        let data = build_v2_header(0x00, &[(0x01, &[1], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_chunk_size_2byte() {
        let data = build_v2_header(0x01, &[(0x01, &[1], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
    }

    #[test]
    fn parse_v2_chunk_size_4byte() {
        let data = build_v2_header(0x02, &[(0x01, &[1], 0)], None);
        let hdr = ObjectHeader::parse(&data, 0, 8, 8).unwrap();
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
        ochk_buf.push(0); // msg flags
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
                (0x01, &[42], 0),      // Dataspace
                (0x10, &cont_data, 0), // Continuation
            ],
            None,
        );

        // Assemble full "file"
        let total_size = ochk_offset + ochk_buf.len();
        let mut file_data = vec![0u8; total_size];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[ochk_offset..ochk_offset + ochk_buf.len()].copy_from_slice(&ochk_buf);

        let hdr = ObjectHeader::parse(&file_data, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[1].msg_type, MessageType::Datatype);
        assert_eq!(hdr.messages[1].data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn truncated_v1_header() {
        let data = vec![1u8, 0]; // version 1, but too short
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn truncated_v2_header() {
        let data = [b'O', b'H', b'D', b'R', 2]; // signature + version, but no flags
        let err = ObjectHeader::parse(&data, 0, 8, 8).unwrap_err();
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
        let buffered = ObjectHeader::parse_with_base(&file_data, 0, os, ls, base).unwrap();
        let from_mem =
            ObjectHeader::parse_from_source(&BytesSource::new(&file_data), 0, os, ls, base)
                .unwrap();
        let from_seek = ObjectHeader::parse_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap(),
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
        let header = build_v2_header(0x20, &[(0x01, &[1, 2, 3], 0)], Some((1, 2, 3, 4)));
        parse_three_ways(header, 8, 8, BaseAddress::ZERO);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v2_with_continuation_matches_buffered() {
        // A v2 header at offset 0 whose continuation points to an OCHK chunk at
        // offset 256 — the streaming parser must read that second chunk from the
        // source and produce the same messages in the same order.
        let ochk_msg_data = [0xDE, 0xAD];
        let mut ochk_buf = Vec::new();
        ochk_buf.extend_from_slice(&OCHK_SIGNATURE);
        ochk_buf.push(0x03); // Datatype
        ochk_buf.extend_from_slice(&(ochk_msg_data.len() as u16).to_le_bytes());
        ochk_buf.push(0);
        ochk_buf.extend_from_slice(&ochk_msg_data);
        let cks = crate::checksum::jenkins_lookup3(&ochk_buf);
        ochk_buf.extend_from_slice(&cks.to_le_bytes());

        let ochk_offset = 256usize;
        let mut cont_data = Vec::new();
        cont_data.extend_from_slice(&(ochk_offset as u64).to_le_bytes());
        cont_data.extend_from_slice(&(ochk_buf.len() as u64).to_le_bytes());

        let header = build_v2_header(0x00, &[(0x01, &[42], 0), (0x10, &cont_data, 0)], None);
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
        // continuation message in the list and follows it depth-first; the
        // streaming parser must do the same.
        let cont_msg_data = [0xBE, 0xEF];
        let mut cont_chunk = Vec::new();
        cont_chunk.extend_from_slice(&0x03u16.to_le_bytes()); // Datatype
        cont_chunk.extend_from_slice(&(cont_msg_data.len() as u16).to_le_bytes());
        cont_chunk.push(0);
        cont_chunk.extend_from_slice(&[0u8; 3]); // reserved
        cont_chunk.extend_from_slice(&cont_msg_data);

        let cont_offset = 256usize;
        let mut cont_ptr = Vec::new();
        cont_ptr.extend_from_slice(&(cont_offset as u64).to_le_bytes());
        cont_ptr.extend_from_slice(&(cont_chunk.len() as u64).to_le_bytes());

        let header = build_v1_header(&[(0x01, &[42][..], 0), (0x10, &cont_ptr[..], 0)], 8, 8);
        let mut file_data = vec![0u8; cont_offset + cont_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[cont_offset..cont_offset + cont_chunk.len()].copy_from_slice(&cont_chunk);

        parse_three_ways(file_data, 8, 8, BaseAddress::ZERO);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_v1_message_overrunning_chunk0_matches_buffered() {
        // Regression for #140: a v1 chunk-0 message whose data overruns the
        // declared object-header size (`header_data_size`). The buffered chunk-0
        // parser must stop at the chunk boundary exactly as the buffered
        // continuation parser and the streaming parser already do — otherwise it
        // reads (and follows) a continuation message the streaming backend drops,
        // so the two readers disagree on a malformed header.
        let cont_msg_data = [0xBE, 0xEF];
        let mut cont_chunk = Vec::new();
        cont_chunk.extend_from_slice(&0x03u16.to_le_bytes()); // Datatype
        cont_chunk.extend_from_slice(&(cont_msg_data.len() as u16).to_le_bytes());
        cont_chunk.push(0);
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
        header.extend_from_slice(&1u16.to_le_bytes()); // num_messages
        header.extend_from_slice(&1u32.to_le_bytes()); // reference_count
        header.extend_from_slice(&16u32.to_le_bytes()); // header_data_size (understated)
        header.extend_from_slice(&[0u8; 4]); // pad prefix to 16 bytes
        header.extend_from_slice(&0x0010u16.to_le_bytes()); // Continuation
        header.extend_from_slice(&(cont_ptr.len() as u16).to_le_bytes()); // size = 16
        header.push(0); // flags
        header.extend_from_slice(&[0u8; 3]); // reserved
        header.extend_from_slice(&cont_ptr); // pointer (overruns chunk 0)

        let mut file_data = vec![0u8; cont_offset + cont_chunk.len()];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[cont_offset..cont_offset + cont_chunk.len()].copy_from_slice(&cont_chunk);

        // All three backends must agree — and, with the overrunning message
        // dropped, agree on an empty message list (the continuation is never
        // followed, so its Datatype message is unreachable too).
        parse_three_ways(file_data.clone(), 8, 8, BaseAddress::ZERO);
        let buffered =
            ObjectHeader::parse_with_base(&file_data, 0, 8, 8, BaseAddress::ZERO).unwrap();
        assert_eq!(buffered.messages.len(), 0);
    }
}
