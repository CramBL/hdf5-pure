#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use super::{HeaderMessage, MessageFilter, ObjectHeader, ParseContext};
use crate::address::StoredAddress;
use crate::bytes::{ensure_len, read_length, read_offset};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::Source;

const MAX_V1_CONTINUATION_DEPTH: u16 = 32;

impl ObjectHeader {
    pub(super) fn parse_v1(
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

    pub(super) fn parse_v1_from_source<S: Source + ?Sized>(
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
