#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use super::{HeaderMessage, MessageFilter, ObjectHeader, ParseContext};
use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::Source;

/// Defines the fixed prefix width of a version 1 object header message.
///
/// The prefix contains a two-byte type, two-byte data size, one-byte flags
/// field, and three reserved bytes. The layout is defined in "Version 1 Data
/// Object Header Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
const MESSAGE_PREFIX_LEN: usize = 8;

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
        bytes::ensure_len(data, offset, 12)?;

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

        bytes::ensure_len(data, msg_start, header_data_size)?;

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
            let Some((record, record_len)) =
                parse_message_record(&data[pos..msg_end], context.access_mode)?
            else {
                break;
            };

            if record.msg_type != MessageType::NIL && filter.keeps(record.msg_type, record.body) {
                messages.push(HeaderMessage {
                    msg_type: record.msg_type,
                    size: record.body.len(),
                    flags: record.flags,
                    creation_order: None,
                    data: record.body.to_vec(),
                });
            }

            pos += record_len;

            if record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION
                && let Some(continuation) = parse_continuation(record.body, context)?
            {
                let offset = context
                    .base_address
                    .absolute(continuation.address)?
                    .to_usize()?;

                let cont_msgs = Self::parse_v1_continuation(
                    data,
                    context,
                    offset,
                    continuation.length.to_usize()?,
                    MAX_V1_CONTINUATION_DEPTH,
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
        bytes::ensure_len(data, offset, length)?;
        let mut messages = Vec::new();
        let mut pos = offset;
        let end = offset.saturating_add(length);

        while pos < end {
            let Some((record, record_len)) =
                parse_message_record(&data[pos..end], context.access_mode)?
            else {
                break;
            };

            if record.msg_type != MessageType::NIL && filter.keeps(record.msg_type, record.body) {
                messages.push(HeaderMessage {
                    msg_type: record.msg_type,
                    size: record.body.len(),
                    flags: record.flags,
                    creation_order: None,
                    data: record.body.to_vec(),
                });
            }

            pos += record_len;

            if record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION
                && let Some(continuation) = parse_continuation(record.body, context)?
            {
                let offset = context
                    .base_address
                    .absolute(continuation.address)?
                    .to_usize()?;

                let cont_msgs = Self::parse_v1_continuation(
                    data,
                    context,
                    offset,
                    continuation.length.to_usize()?,
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

        while count < max_messages && pos < end {
            let Some((record, record_len)) =
                parse_message_record(&region[pos..end], context.access_mode)?
            else {
                break;
            };

            count += 1;
            pos += record_len;

            let continuation = if record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                parse_continuation(record.body, context)?
            } else {
                None
            };

            if record.msg_type != MessageType::NIL && filter.keeps(record.msg_type, record.body) {
                messages.push(HeaderMessage {
                    msg_type: record.msg_type,
                    size: record.body.len(),
                    flags: record.flags,
                    creation_order: None,
                    data: record.body.to_vec(),
                });
            }

            if let Some(continuation) = continuation {
                let address = context.base_address.absolute(continuation.address)?;

                Self::parse_v1_chunk_from_source(
                    source,
                    context,
                    address,
                    continuation.length,
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

/// Represents one decoded message record from a version 1 object header.
///
/// The body borrows from the current header chunk. The type, size, flags,
/// reserved bytes, and data layout are defined in "Version 1 Data Object
/// Header Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
#[derive(Clone, Copy)]
struct MessageRecord<'a> {
    msg_type: MessageType,
    flags: MessageFlags,
    body: &'a [u8],
}

/// Decodes one version 1 message record from a bounded chunk slice.
///
/// Fewer than [`MESSAGE_PREFIX_LEN`] bytes produces `None`. A complete prefix
/// whose declared body extends past the bounded slice produces
/// [`FormatError::UnexpectedEof`]. Unknown message types whose flags require
/// understanding under `access_mode` produce
/// [`FormatError::UnsupportedMessage`].
///
/// The record layout is defined in "Version 1 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
fn parse_message_record(
    data: &[u8],
    access_mode: AccessMode,
) -> Result<Option<(MessageRecord<'_>, usize)>, FormatError> {
    if data.len() < MESSAGE_PREFIX_LEN {
        return Ok(None);
    }

    let msg_type = MessageType::from_u16(LittleEndian::read_u16(&data[..2]));
    let body_len = usize::from(LittleEndian::read_u16(&data[2..4]));
    let flags = MessageFlags::new(data[4]);

    bytes::ensure_len(data, MESSAGE_PREFIX_LEN, body_len)?;
    let record_len = MESSAGE_PREFIX_LEN + body_len;

    if let Some(id) = msg_type.unknown_id()
        && flags.must_be_understood(access_mode)
    {
        return Err(FormatError::UnsupportedMessage(id));
    }

    Ok(Some((
        MessageRecord {
            msg_type,
            flags,
            body: &data[MESSAGE_PREFIX_LEN..record_len],
        },
        record_len,
    )))
}

fn parse_continuation(
    data: &[u8],
    context: ParseContext,
) -> Result<Option<Continuation>, FormatError> {
    let required_len = usize::from(context.offset_size) + usize::from(context.length_size);

    if data.len() < required_len {
        return Ok(None);
    }

    let address = StoredAddress::new(bytes::read_offset(data, 0, context.offset_size)?);
    let length = bytes::read_length(data, usize::from(context.offset_size), context.length_size)?;

    Ok(Some(Continuation { address, length }))
}

#[derive(Clone, Copy)]
struct Continuation {
    address: StoredAddress,
    length: u64,
}
