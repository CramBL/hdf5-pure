#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use byteorder::{ByteOrder, LittleEndian};

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::{HeaderMessage, MessageFilter, ObjectHeader, ParseContext};
use crate::source::Source;
use crate::width::UintWidth;

/// Interprets the flags byte in a version 2 object header prefix.
///
/// The byte encodes the chunk size field width and the presence of optional
/// creation order, attribute phase change, and timestamp fields. Unused and
/// reserved bits are preserved in the raw value. The layout is defined in
/// "Version 2 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(super) struct HeaderFlags(u8);

impl HeaderFlags {
    const TRACKS_CREATION_ORDER: u8 = 0x04;
    const STORES_ATTRIBUTE_PHASE_CHANGE: u8 = 0x10;
    const STORES_TIMES: u8 = 0x20;

    /// Preserves every bit from the on-disk flags byte.
    pub(super) const fn new(flags: u8) -> Self {
        Self(flags)
    }

    /// Returns the on-disk flags byte unchanged.
    pub(super) const fn raw(self) -> u8 {
        self.0
    }

    /// Reports whether messages carry creation order values.
    pub(super) const fn tracks_creation_order(self) -> bool {
        self.0 & Self::TRACKS_CREATION_ORDER != 0
    }

    /// Reports whether non-default attribute phase change values are present.
    pub(super) const fn stores_attribute_phase_change(self) -> bool {
        self.0 & Self::STORES_ATTRIBUTE_PHASE_CHANGE != 0
    }

    /// Reports whether the four object timestamps are present.
    pub(super) const fn stores_times(self) -> bool {
        self.0 & Self::STORES_TIMES != 0
    }

    /// Returns the width encoded for the chunk size field.
    pub(super) fn chunk_size_width(self) -> UintWidth {
        UintWidth::from_flags(self.0)
    }
}

impl ObjectHeader {
    pub(super) fn parse_v2(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        // signature(4) + version(1) + flags(1) = 6
        bytes::ensure_len(data, offset, 6)?;

        let version = data[offset + 4];
        if version != 2 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }
        let flags = HeaderFlags::new(data[offset + 5]);

        let mut pos = offset + 6;

        let (access_time, modification_time, change_time, birth_time) = if flags.stores_times() {
            bytes::ensure_len(data, pos, 16)?;
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
            bytes::ensure_len(data, pos, 4)?;
            // The attribute phase change values occupy four bytes.
            pos += 4;
        }

        let chunk_size_width = flags.chunk_size_width();
        let chunk0_size = bytes::read_uint_width(data, pos, chunk_size_width)?.to_usize()?;
        pos += usize::from(chunk_size_width.get());

        let chunk0_msg_start = pos;
        let chunk0_msg_end = pos
            .checked_add(chunk0_size)
            .ok_or(FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: data.len(),
            })?;

        // Validate checksum: from OHDR signature through all messages (before checksum)
        bytes::ensure_len(data, chunk0_msg_end, 4)?;
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
        let mut cont_remaining = MAX_CONTINUATIONS;
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
        bytes::ensure_len(data, offset, length)?;
        if length < 8 {
            return Err(FormatError::UnexpectedEof {
                expected: 8,
                available: length,
            });
        }

        bytes::ensure_len(data, offset, 4)?;
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
                        StoredAddress::new(bytes::read_offset(msg_data, 0, context.offset_size)?);
                    let length = bytes::read_length(
                        msg_data,
                        context.offset_size as usize,
                        context.length_size,
                    )?;

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

    pub(super) fn parse_v2_from_source<S: Source + ?Sized>(
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
        let flags = HeaderFlags::new(head[5]);
        let mut pos = 6usize;

        let (access_time, modification_time, change_time, birth_time) = if flags.stores_times() {
            bytes::ensure_len(&head, pos, 16)?;
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
            bytes::ensure_len(&head, pos, 4)?;
            pos += 4;
        }

        let chunk_size_width = flags.chunk_size_width();
        let chunk0_size = bytes::read_uint_width(&head, pos, chunk_size_width)?.to_usize()?;
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
        let mut cont_remaining = MAX_CONTINUATIONS;
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
}

/// Describes the location and size of an object header continuation block.
///
/// The Object Header Continuation message stores a file address and a byte
/// length for a block containing additional header messages. The fields are
/// defined in "The Object Header Continuation Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_continuation
#[derive(Clone, Copy)]
struct Continuation {
    address: StoredAddress,
    length: u64,
}

/// Limits continuation traversal to protect the parser from cycles.
const MAX_CONTINUATIONS: u16 = 256;

/// OCHK signature for v2 continuation chunks.
pub(super) const OCHK_SIGNATURE: [u8; 4] = *b"OCHK";
