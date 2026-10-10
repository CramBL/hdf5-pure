//! The version 2 object header: its prefix, its message records, and its continuation blocks.

use alloc::vec::Vec;
use core::ops::Range;

#[cfg(feature = "checksum")]
use byteorder::ByteOrder;
#[cfg(feature = "checksum")]
use byteorder::LittleEndian;

use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::error::OBJECT_HEADER_MESSAGE_MAX;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;
use crate::object_header::HeaderMessage;
use crate::object_header::MessageFilter;
use crate::object_header::OHDR_SIGNATURE;
use crate::object_header::ObjectHeader;
use crate::object_header::ParseContext;
use crate::width::FormatWidths;
use crate::width::UintWidth;

impl ObjectHeader {
    pub(super) fn parse_v2(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        bytes::ensure_len(data, offset, 6)?;
        let prefix = ObjectHeaderPrefix::parse(&data[offset..])?;

        let chunk0_msg_start =
            offset
                .checked_add(prefix.len)
                .ok_or(FormatError::UnexpectedEof {
                    expected: usize::MAX,
                    available: data.len(),
                })?;
        let chunk0_msg_end = chunk0_msg_start
            .checked_add(prefix.chunk0_size.to_usize()?)
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

        let layout = prefix.prefix.layout;

        // Parse messages from chunk0
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        Self::parse_v2_messages(
            data,
            context,
            chunk0_msg_start,
            chunk0_msg_end,
            layout,
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
            let cont_offset = context.base_address.absolute(continuation.address())?;

            Self::parse_v2_continuation(
                data,
                context,
                cont_offset.to_usize()?,
                continuation.length().to_usize()?,
                layout,
                &mut messages,
                &mut continuations,
                filter,
            )?;
        }

        Ok(prefix.object_header(messages))
    }

    fn parse_v2_continuation(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        length: usize,
        layout: MessageRecordLayout,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<ObjectHeaderContinuation>,
        filter: &mut MessageFilter<'_>,
    ) -> Result<(), FormatError> {
        bytes::ensure_len(data, offset, length)?;
        let block = &data[offset..offset + length];
        let block_messages = continuation_block_messages(block)?;

        #[cfg(feature = "checksum")]
        {
            let stored = LittleEndian::read_u32(&block[block_messages.end..]);
            let computed = crate::checksum::jenkins_lookup3(&block[..block_messages.end]);
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
            offset + block_messages.start,
            offset + block_messages.end,
            layout,
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
    fn count_v2_messages(region: &[u8], start: usize, layout: MessageRecordLayout) -> usize {
        let mut pos = start;
        let mut count = 0;
        while let Some(record) = layout.next_message(region, pos).ok().flatten() {
            if record.msg_type != MessageType::NIL
                && record.msg_type != MessageType::OBJECT_HEADER_CONTINUATION
            {
                count += 1;
            }
            pos = record.body_range.end;
        }
        count
    }

    fn parse_v2_messages(
        data: &[u8],
        context: ParseContext,
        start: usize,
        end: usize,
        layout: MessageRecordLayout,
        messages: &mut Vec<HeaderMessage>,
        continuations: &mut Vec<ObjectHeaderContinuation>,
        filter: &mut MessageFilter<'_>,
    ) -> Result<(), FormatError> {
        let region = data.get(..end).ok_or(FormatError::UnexpectedEof {
            expected: end,
            available: data.len(),
        })?;
        if filter.keeps_all() {
            messages.reserve(Self::count_v2_messages(region, start, layout));
        }
        let mut pos = start;

        // A record whose body runs past the region ends the walk.
        while let Some(record) = layout.next_message(region, pos).ok().flatten() {
            let MessageRecord {
                msg_type,
                flags: msg_flags,
                creation_index: creation_order,
                body: msg_data,
                body_range,
            } = record;

            if let Some(id) = msg_type.unknown_id()
                && msg_flags.must_be_understood(context.access_mode)
            {
                return Err(FormatError::UnsupportedMessage(id));
            }

            if msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                // Neither the offset nor the length is narrowed here, so that
                // the driver, buffered or streaming, can fetch a region a
                // 32-bit `usize` does not reach: a streaming reader follows a
                // continuation past 4 GiB on a 32-bit host.
                continuations.push(ObjectHeaderContinuation::parse(
                    msg_data,
                    context.offset_size,
                    context.length_size,
                )?);
            } else if msg_type != MessageType::NIL && filter.keeps(msg_type, msg_data) {
                messages.push(HeaderMessage {
                    msg_type,
                    size: msg_data.len(),
                    flags: msg_flags,
                    creation_order,
                    data: msg_data.to_vec(),
                });
            }

            pos = body_range.end;
        }

        Ok(())
    }

    pub(super) fn parse_v2_from_source<S: MetadataSource + ?Sized>(
        source: &S,
        context: ParseContext,
        address: u64,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        let head_len = OBJECT_HEADER_PREFIX_MAX_LEN
            .to_u64()
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let head = source.read_metadata_at(address, head_len)?;
        let prefix = ObjectHeaderPrefix::parse(&head)?;
        let prefix_len = prefix.len;

        // The chunk 0 body, including the prefix, messages, and checksum, is
        // contiguous from `address`. Reading the complete region gives the checksum
        // the same byte range as the buffered parser.
        let chunk0_end = prefix_len
            .checked_add(prefix.chunk0_size.to_usize()?)
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

        let layout = prefix.prefix.layout;
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        Self::parse_v2_messages(
            &chunk0,
            context,
            prefix_len,
            chunk0_end,
            layout,
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
            let length = continuation.length().to_usize()?;
            let address = context.base_address.absolute(continuation.address())?;

            let region = source.read_metadata_at(address, length)?;

            Self::parse_v2_continuation(
                &region,
                context,
                0,
                length,
                layout,
                &mut messages,
                &mut continuations,
                filter,
            )?;
        }

        Ok(prefix.object_header(messages))
    }
}

/// The length in bytes of the longest version 2 object header prefix: the signature (4), the
/// version (1), the flags (1), the four times (16), the attribute phase change values (4), and an
/// 8-byte Size of Chunk #0 field.
///
/// A reader that reads this many bytes at the start of a header has its whole prefix. The fields
/// are defined in "Version 2 Data Object Header Prefix" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub const OBJECT_HEADER_PREFIX_MAX_LEN: usize = 34;

/// A version 2 object header prefix as [`ObjectHeaderPrefix::parse`] reads it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsedObjectHeaderPrefix {
    /// The record layout and the optional blocks the prefix declares.
    pub prefix: ObjectHeaderPrefix,
    /// The flags byte, with every bit as the header stores it.
    pub flags: u8,
    /// The Size of Chunk #0 field: the length in bytes of chunk 0 from the end of the prefix to
    /// its checksum.
    pub chunk0_size: u64,
    /// The length of the prefix in bytes, from the signature to the end of the Size of Chunk #0
    /// field.
    pub len: usize,
}

impl ParsedObjectHeaderPrefix {
    fn object_header(&self, messages: Vec<HeaderMessage>) -> ObjectHeader {
        let times = self.prefix.times;
        ObjectHeader {
            version: OBJECT_HEADER_VERSION,
            messages,
            reference_count: None,
            flags: self.flags,
            access_time: times.map(|times| times.access),
            modification_time: times.map(|times| times.modification),
            change_time: times.map(|times| times.change),
            birth_time: times.map(|times| times.birth),
        }
    }
}

/// The layout of the message records of a version 2 object header, from bits 2 and 3 of its flags.
///
/// Every record opens with a type byte, a 2-byte body size, and a flags byte. In a header that
/// tracks attribute creation order, bit 2, every record follows them with a 2-byte creation index,
/// for a 6-byte record prefix. A header that indexes attribute creation order as well, bit 3,
/// keeps a creation-order index beside the name index once its attributes are in dense storage.
/// `H5Pset_attr_creation_order` sets bit 2 for `H5P_CRT_ORDER_TRACKED` and bit 3 for
/// `H5P_CRT_ORDER_INDEXED`.
///
/// The bits apply to the whole header, so chunk 0 and every continuation block of a header share
/// one layout. The layout is defined in "Version 2 Data Object Header Prefix" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MessageRecordLayout {
    /// Every record stores a 2-byte creation index (object header flags bit 2).
    tracked: bool,
    /// The header indexes attribute creation order (object header flags bit 3).
    indexed: bool,
}

/// Bit 2 of the flags of a version 2 object header, set where the header tracks attribute
/// creation order and every message record stores a creation index.
///
/// The bit is defined in "Version 2 Data Object Header Prefix" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
const OH_FLAG_CREATION_ORDER_TRACKED: u8 = 0x04;

/// Bit 3 of the flags, set where the header indexes attribute creation order as well as tracking
/// it, from the same section as [`OH_FLAG_CREATION_ORDER_TRACKED`].
const OH_FLAG_CREATION_ORDER_INDEXED: u8 = 0x08;

/// Bit 4 of the flags, set where the prefix stores the attribute phase change values, from the
/// same section as [`OH_FLAG_CREATION_ORDER_TRACKED`]. The C library calls it
/// `H5O_HDR_ATTR_STORE_PHASE_CHANGE`.
const OH_FLAG_ATTRIBUTE_PHASE_CHANGE: u8 = 0x10;

/// Bit 5 of the flags, set where the prefix stores the four times, from the same section as
/// [`OH_FLAG_CREATION_ORDER_TRACKED`]. The C library calls it `H5O_HDR_STORE_TIMES`.
const OH_FLAG_STORE_TIMES: u8 = 0x20;

/// The access, modification, change, and birth times a version 2 object header prefix stores,
/// each in seconds since the Unix epoch.
///
/// The prefix stores them where bit 5 of its flags is set. The C library sets that bit on every
/// version 2 header it writes by default: `H5O_CRT_OHDR_FLAGS_DEF` is `H5O_HDR_STORE_TIMES`
/// (`H5Opkg.h`, HDF5 2.2.0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectTimes {
    /// The time the raw data of the object was last read or written.
    pub access: u32,
    /// The time the raw data of the object was last written.
    pub modification: u32,
    /// The time the metadata of the object was last changed.
    pub change: u32,
    /// The time the object was created.
    pub birth: u32,
}

impl ObjectTimes {
    /// The length in bytes of the four times in a prefix.
    const LEN: usize = 16;

    fn parse(block: [u8; Self::LEN]) -> Self {
        let [access, modification, change, birth] = [0, 1, 2, 3].map(|field| {
            let at = 4 * field;
            u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
        });
        Self {
            access,
            modification,
            change,
            birth,
        }
    }

    /// Returns the four times as the prefix stores them.
    fn to_bytes(self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[0..4].copy_from_slice(&self.access.to_le_bytes());
        out[4..8].copy_from_slice(&self.modification.to_le_bytes());
        out[8..12].copy_from_slice(&self.change.to_le_bytes());
        out[12..16].copy_from_slice(&self.birth.to_le_bytes());
        out
    }

    /// Returns the times with the modification and change times set to `now`, for a rewrite of
    /// the header.
    ///
    /// The access and birth times keep their values. On a version 2 header, `H5O_touch_oh` sets
    /// the access and change times, with a source comment that the modification time needs code to
    /// update it (`H5Oint.c`, HDF5 2.2.0).
    pub fn touched(self, now: u32) -> Self {
        Self {
            modification: now,
            change: now,
            ..self
        }
    }
}

/// The attribute phase change values a version 2 object header prefix stores, which
/// `H5Pset_attr_phase_change` sets.
///
/// The prefix stores them where bit 4 of its flags is set. The C library sets that bit only where
/// the values differ from its defaults of 8 and 6 (`H5O_apply_ohdr` in `H5Oint.c`, HDF5 2.2.0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttributePhaseChange {
    /// The most attributes the object stores compactly, as messages in its header.
    pub max_compact: u16,
    /// The fewest attributes the object stores densely, in a fractal heap.
    pub min_dense: u16,
}

impl AttributePhaseChange {
    /// The length in bytes of the two values in a prefix.
    const LEN: usize = 4;

    fn parse(block: [u8; Self::LEN]) -> Self {
        let [max_low, max_high, min_low, min_high] = block;
        Self {
            max_compact: u16::from_le_bytes([max_low, max_high]),
            min_dense: u16::from_le_bytes([min_low, min_high]),
        }
    }

    /// Returns the two values as the prefix stores them.
    fn to_bytes(self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[0..2].copy_from_slice(&self.max_compact.to_le_bytes());
        out[2..4].copy_from_slice(&self.min_dense.to_le_bytes());
        out
    }
}

/// The properties of a version 2 object header that its prefix stores: the layout of its message
/// records, the four times, and the attribute phase change values.
///
/// The message records do not store them, so a caller that rewrites a header keeps them beside
/// its messages. [`parse`](Self::parse) reads them from a header, and
/// [`encode_header`](Self::encode_header) writes them back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjectHeaderPrefix {
    /// The layout of the message records.
    pub layout: MessageRecordLayout,
    /// The four times, where the prefix stores them.
    pub times: Option<ObjectTimes>,
    /// The attribute phase change values, where the prefix stores them.
    pub attribute_phase_change: Option<AttributePhaseChange>,
}

impl ObjectHeaderPrefix {
    /// The prefix of a header with [`MessageRecordLayout::PLAIN`] records and neither optional
    /// block.
    pub const PLAIN: Self = Self::with_layout(MessageRecordLayout::PLAIN);

    /// Returns the prefix of a header whose records are in `layout`, with neither optional block.
    pub const fn with_layout(layout: MessageRecordLayout) -> Self {
        Self {
            layout,
            times: None,
            attribute_phase_change: None,
        }
    }

    /// Parses the prefix at the start of `data`, from the `OHDR` signature to the end of the Size
    /// of Chunk #0 field.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidObjectHeaderSignature`] if `data` does not begin with
    /// `OHDR`, [`FormatError::InvalidObjectHeaderVersion`] if the version is not 2, and
    /// [`FormatError::UnexpectedEof`] if `data` ends inside the prefix.
    pub fn parse(data: &[u8]) -> Result<ParsedObjectHeaderPrefix, FormatError> {
        bytes::ensure_len(data, 0, 6)?;
        if data[..4] != OHDR_SIGNATURE {
            return Err(FormatError::InvalidObjectHeaderSignature);
        }
        if data[4] != OBJECT_HEADER_VERSION {
            return Err(FormatError::InvalidObjectHeaderVersion(data[4]));
        }
        let flags = data[5];
        let mut pos = 6;
        let times = optional_block(data, &mut pos, flags & OH_FLAG_STORE_TIMES != 0)?
            .map(ObjectTimes::parse);
        let attribute_phase_change =
            optional_block(data, &mut pos, flags & OH_FLAG_ATTRIBUTE_PHASE_CHANGE != 0)?
                .map(AttributePhaseChange::parse);
        let size_width = UintWidth::from_flags(flags);
        let chunk0_size = bytes::read_uint_width(data, pos, size_width)?;
        Ok(ParsedObjectHeaderPrefix {
            prefix: Self {
                layout: MessageRecordLayout::from_header_flags(flags),
                times,
                attribute_phase_change,
            },
            flags,
            chunk0_size,
            len: pos + usize::from(size_width.get()),
        })
    }

    /// Returns a version 2 object header of one chunk: this prefix, the message records
    /// `messages`, and the checksum.
    ///
    /// The Size of Chunk #0 field takes the fewest bytes that hold the length of `messages`.
    pub fn encode_header(self, messages: &[u8]) -> Vec<u8> {
        self.encode_header_with(messages.len(), |buf| buf.extend_from_slice(messages))
    }

    /// Returns a version 2 object header of one chunk, and calls `write_messages` to append its
    /// `messages_len` bytes of message records.
    ///
    /// # Panics
    ///
    /// Panics if `write_messages` appends a length other than `messages_len`.
    pub fn encode_header_with(
        self,
        messages_len: usize,
        write_messages: impl FnOnce(&mut Vec<u8>),
    ) -> Vec<u8> {
        let width = UintWidth::smallest_for_len(messages_len);
        let prefix_len = OHDR_SIGNATURE.len() + 2 + self.optional_len() + usize::from(width.get());
        let mut buf = Vec::with_capacity(prefix_len + messages_len + CHECKSUM_LEN);
        buf.extend_from_slice(&OHDR_SIGNATURE);
        buf.push(OBJECT_HEADER_VERSION);
        buf.push(width.flag_bits() | self.header_flags());
        if let Some(times) = self.times {
            buf.extend_from_slice(&times.to_bytes());
        }
        if let Some(phase) = self.attribute_phase_change {
            buf.extend_from_slice(&phase.to_bytes());
        }
        width.write(&mut buf, messages_len);
        write_messages(&mut buf);
        assert_eq!(
            buf.len(),
            prefix_len + messages_len,
            "the messages written differ from the length the header declares"
        );
        let checksum = checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Returns the flag bits of these properties, apart from the two bits of the width of the
    /// Size of Chunk #0 field.
    const fn header_flags(self) -> u8 {
        let times = if self.times.is_some() {
            OH_FLAG_STORE_TIMES
        } else {
            0
        };
        let phase = if self.attribute_phase_change.is_some() {
            OH_FLAG_ATTRIBUTE_PHASE_CHANGE
        } else {
            0
        };
        self.layout.header_flags() | times | phase
    }

    /// Returns the length in bytes the optional blocks add to the prefix.
    const fn optional_len(self) -> usize {
        let times = if self.times.is_some() {
            ObjectTimes::LEN
        } else {
            0
        };
        let phase = if self.attribute_phase_change.is_some() {
            AttributePhaseChange::LEN
        } else {
            0
        };
        times + phase
    }
}

impl MessageRecordLayout {
    /// The layout of a header that does not track creation order, whose record prefixes are 4
    /// bytes long.
    pub const PLAIN: Self = Self {
        tracked: false,
        indexed: false,
    };

    /// Returns the layout that the flags byte of a version 2 object header declares.
    pub const fn from_header_flags(flags: u8) -> Self {
        Self {
            tracked: flags & OH_FLAG_CREATION_ORDER_TRACKED != 0,
            indexed: flags & OH_FLAG_CREATION_ORDER_INDEXED != 0,
        }
    }

    /// Returns the length in bytes of a message record before its body: 4, or 6 with a creation
    /// index.
    pub const fn prefix_len(self) -> usize {
        if self.tracked { 6 } else { 4 }
    }

    /// Returns the flag bits of this layout.
    const fn header_flags(self) -> u8 {
        let tracked = if self.tracked {
            OH_FLAG_CREATION_ORDER_TRACKED
        } else {
            0
        };
        let indexed = if self.indexed {
            OH_FLAG_CREATION_ORDER_INDEXED
        } else {
            0
        };
        tracked | indexed
    }

    /// Returns `true` if every message record stores a creation index.
    pub const fn tracks_creation_order(self) -> bool {
        self.tracked
    }

    /// Returns `true` if the header indexes attribute creation order, with a creation-order index
    /// beside the name index once its attributes are in dense storage.
    pub const fn indexes_creation_order(self) -> bool {
        self.indexed
    }

    /// Returns the message record at `msg_start` in the message region `region`, or `None` where
    /// fewer bytes remain than a record prefix.
    ///
    /// A chunk may end in a gap shorter than a record prefix, as the specification allows.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the body of the record runs past the end of
    /// `region`.
    pub fn next_message<'a>(
        self,
        region: &'a [u8],
        msg_start: usize,
    ) -> Result<Option<MessageRecord<'a>>, FormatError> {
        let Some((&[type_byte, size_low, size_high, flags], rest)) = region
            .get(msg_start..)
            .and_then(|record| record.split_first_chunk::<4>())
        else {
            return Ok(None);
        };
        let (creation_index, rest) = if self.tracked {
            let Some((&index, rest)) = rest.split_first_chunk::<2>() else {
                return Ok(None);
            };
            (Some(u16::from_le_bytes(index)), rest)
        } else {
            (None, rest)
        };
        let body_start = msg_start + self.prefix_len();
        let body_len = usize::from(u16::from_le_bytes([size_low, size_high]));
        let Some(body) = rest.get(..body_len) else {
            return Err(FormatError::UnexpectedEof {
                expected: body_start + body_len,
                available: region.len(),
            });
        };
        Ok(Some(MessageRecord {
            msg_type: MessageType::from_u16(u16::from(type_byte)),
            flags: MessageFlags::new(flags),
            creation_index,
            body,
            body_range: body_start..body_start + body_len,
        }))
    }

    /// Returns the creation index of the record at `msg_start` in `region`, or `None` where the
    /// layout has none or `region` ends inside it.
    pub fn creation_index(self, region: &[u8], msg_start: usize) -> Option<u16> {
        self.tracked
            .then(|| region.get(msg_start + 4..)?.first_chunk().copied())
            .flatten()
            .map(u16::from_le_bytes)
    }

    /// Returns a message record of `msg_type` with `flags`, `creation_index`, and `body`.
    ///
    /// The record stores `creation_index` only where the layout tracks creation order. The
    /// creation index of an Attribute message is the creation order of the attribute, and the C
    /// library writes 0 for every other message type (`H5Omessage.c`, HDF5 2.2.0), which
    /// [`record`](Self::record) passes.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte message type field, or if `body` is longer
    /// than [`OBJECT_HEADER_MESSAGE_MAX`].
    pub fn record_with_creation_index(
        self,
        msg_type: MessageType,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
    ) -> Vec<u8> {
        let mut record = Vec::with_capacity(self.prefix_len() + body.len());
        self.write_record(&mut record, msg_type, flags, creation_index, body);
        record
    }

    /// Appends a message record of `msg_type` with `flags`, `creation_index`, and `body` to
    /// `buf`, as [`record_with_creation_index`](Self::record_with_creation_index) returns it.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte message type field, or if `body` is longer
    /// than [`OBJECT_HEADER_MESSAGE_MAX`].
    pub fn write_record(
        self,
        buf: &mut Vec<u8>,
        msg_type: MessageType,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
    ) {
        assert!(
            u8::try_from(msg_type.to_u16()).is_ok(),
            "message type {:#06x} does not fit the 1-byte type field of a version 2 object header",
            msg_type.to_u16()
        );
        assert!(
            body.len() <= OBJECT_HEADER_MESSAGE_MAX,
            "a {}-byte message body does not fit the 2-byte message size field",
            body.len()
        );
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the first assertion above admits only a type that fits the 1-byte field"
        )]
        buf.push(msg_type.to_u16() as u8);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the second assertion above bounds the body to the 2-byte size field"
        )]
        buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
        buf.push(flags.get());
        if self.tracks_creation_order() {
            buf.extend_from_slice(&creation_index.to_le_bytes());
        }
        buf.extend_from_slice(body);
    }

    /// Returns a message record of `msg_type` and `body` with no flag set, and with a creation
    /// index of 0 where the layout tracks creation order.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte message type field, or if `body` is longer
    /// than [`OBJECT_HEADER_MESSAGE_MAX`].
    pub fn record(self, msg_type: MessageType, body: &[u8]) -> Vec<u8> {
        self.record_with_creation_index(msg_type, MessageFlags::NONE, 0, body)
    }
}

/// A message record of a version 2 object header, as [`MessageRecordLayout::next_message`] reads
/// it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageRecord<'a> {
    /// The type of the message.
    pub msg_type: MessageType,
    /// The message flags.
    pub flags: MessageFlags,
    /// The creation index, where the layout tracks creation order.
    pub creation_index: Option<u16>,
    /// The body of the message.
    pub body: &'a [u8],
    /// The range of the body in the region, whose end is where the next record begins.
    pub body_range: Range<usize>,
}

/// Returns the `N` bytes at `*pos` and moves `pos` past them where `present`, and `None`
/// otherwise.
fn optional_block<const N: usize>(
    data: &[u8],
    pos: &mut usize,
    present: bool,
) -> Result<Option<[u8; N]>, FormatError> {
    if !present {
        return Ok(None);
    }
    let block = data
        .get(*pos..)
        .and_then(|rest| rest.first_chunk::<N>())
        .copied()
        .ok_or(FormatError::UnexpectedEof {
            expected: pos.saturating_add(N),
            available: data.len(),
        })?;
    *pos += N;
    Ok(Some(block))
}

/// The location and length of an object header continuation block, the body of an Object
/// Header Continuation message.
///
/// The fields are defined in "The Object Header Continuation Message" of the [format
/// specification, version 4.0][spec]. The address is never the undefined address, which
/// [`parse`](Self::parse) rejects.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_continuation
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectHeaderContinuation {
    /// The address of the continuation block.
    address: StoredAddress,
    /// The length of the continuation block in bytes.
    length: u64,
}

impl ObjectHeaderContinuation {
    /// Parses the body of an Object Header Continuation message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `body` ends inside a field,
    /// [`FormatError::UndefinedContinuationAddress`] if the address is the undefined address, and
    /// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not
    /// 2, 4, or 8.
    pub fn parse(body: &[u8], offset_size: u8, length_size: u8) -> Result<Self, FormatError> {
        let FormatWidths { offsets, lengths } = FormatWidths::from_sizes(offset_size, length_size)?;
        Ok(Self {
            address: bytes::read_optional_offset_width(body, 0, offsets)?
                .map(StoredAddress::new)
                .ok_or(FormatError::UndefinedContinuationAddress)?,
            length: bytes::read_length_width(body, usize::from(offsets.get()), lengths)?,
        })
    }

    /// Returns the address of the continuation block, relative to the base address.
    pub fn address(&self) -> StoredAddress {
        self.address
    }

    /// Returns the length of the continuation block in bytes.
    pub fn length(&self) -> u64 {
        self.length
    }
}

/// Returns the range of the message records in the continuation block `block`, between its `OCHK`
/// signature and its checksum.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `block` is shorter than the signature and the
/// checksum, and [`FormatError::InvalidObjectHeaderSignature`] if it does not begin with `OCHK`.
pub fn continuation_block_messages(block: &[u8]) -> Result<Range<usize>, FormatError> {
    let end = block
        .len()
        .checked_sub(CHECKSUM_LEN)
        .filter(|&end| end >= OCHK_SIGNATURE.len())
        .ok_or(FormatError::UnexpectedEof {
            expected: OCHK_SIGNATURE.len() + CHECKSUM_LEN,
            available: block.len(),
        })?;
    if block[..OCHK_SIGNATURE.len()] != OCHK_SIGNATURE {
        return Err(FormatError::InvalidObjectHeaderSignature);
    }
    Ok(OCHK_SIGNATURE.len()..end)
}

/// Limits continuation traversal to protect the parser from cycles.
const MAX_CONTINUATIONS: u16 = 256;

/// OCHK signature for v2 continuation chunks.
const OCHK_SIGNATURE: [u8; 4] = *b"OCHK";

/// The length in bytes of the checksum that ends a chunk.
const CHECKSUM_LEN: usize = 4;

/// The version of a version 2 object header, from the same section as
/// [`OH_FLAG_CREATION_ORDER_TRACKED`].
const OBJECT_HEADER_VERSION: u8 = 2;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::object_header::Message;
    use test_util::object_header::MessageType as RecordType;
    use test_util::object_header::v2 as v2_bytes;

    use super::*;

    #[test]
    fn a_prefix_parses_to_the_fields_its_flags_select() {
        let flags = v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER
            | v2_bytes::HeaderFlags::INDEXES_CREATION_ORDER
            | v2_bytes::HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE;
        let header = v2_bytes::Header::new()
            .flags(flags)
            .timestamps(v2_bytes::Timestamps {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4,
            })
            .message(Message::new(RecordType::DATASPACE, &[9]))
            .build();

        assert_eq!(
            ObjectHeaderPrefix::parse(&header),
            Ok(ParsedObjectHeaderPrefix {
                prefix: ObjectHeaderPrefix {
                    layout: TRACKED,
                    times: Some(TIMES),
                    attribute_phase_change: Some(AttributePhaseChange {
                        max_compact: 8,
                        min_dense: 6,
                    }),
                },
                flags: (flags | v2_bytes::HeaderFlags::STORES_TIMES).0,
                chunk0_size: 7,
                len: 27,
            })
        );
    }

    #[rstest]
    fn an_encoded_header_parses_back_to_its_prefix_and_messages(
        #[values(
            MessageRecordLayout::PLAIN,
            MessageRecordLayout::from_header_flags(0x0C)
        )]
        layout: MessageRecordLayout,
        #[values(None, Some(TIMES))] times: Option<ObjectTimes>,
        #[values(None, Some(PHASE))] attribute_phase_change: Option<AttributePhaseChange>,
    ) {
        let prefix = ObjectHeaderPrefix {
            layout,
            times,
            attribute_phase_change,
        };
        let records = layout.record(MessageType::DATASPACE, &[1, 2, 3]);
        let header = prefix.encode_header(&records);

        let parsed = ObjectHeaderPrefix::parse(&header).unwrap();
        assert_eq!(parsed.prefix, prefix);
        assert_eq!(parsed.chunk0_size, records.len().to_u64());
        let parsed_header =
            ObjectHeader::parse(&header, crate::access_mode::AccessMode::ReadOnly, 0, 8, 8)
                .unwrap();
        assert_eq!(parsed_header.messages.len(), 1);
        assert_eq!(parsed_header.messages[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn an_encoded_header_matches_the_bytes_of_an_independent_builder() {
        let message = Message::new(RecordType::DATATYPE, &[5, 6]);
        let expected = v2_bytes::Header::new()
            .timestamps(v2_bytes::Timestamps {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4,
            })
            .message(message.clone())
            .build();
        let prefix = ObjectHeaderPrefix {
            times: Some(TIMES),
            ..ObjectHeaderPrefix::PLAIN
        };
        let records = v2_bytes::message_records(&[message], v2_bytes::HeaderFlags::default());

        assert_eq!(prefix.encode_header(&records), expected);
    }

    #[rstest]
    #[case::inside_the_signature(3, FormatError::UnexpectedEof { expected: 6, available: 3 })]
    #[case::inside_the_times(10, FormatError::UnexpectedEof { expected: 22, available: 10 })]
    #[case::inside_the_phase_change(24, FormatError::UnexpectedEof { expected: 26, available: 24 })]
    #[case::before_the_chunk_size(26, FormatError::UnexpectedEof { expected: 27, available: 26 })]
    fn a_truncated_prefix_is_unexpected_eof(#[case] cut: usize, #[case] expected: FormatError) {
        let header = ObjectHeaderPrefix {
            times: Some(TIMES),
            attribute_phase_change: Some(PHASE),
            ..ObjectHeaderPrefix::PLAIN
        }
        .encode_header(&[]);
        assert_eq!(ObjectHeaderPrefix::parse(&header[..cut]), Err(expected));
    }

    #[rstest]
    #[case::another_signature(0, b'X', FormatError::InvalidObjectHeaderSignature)]
    #[case::version_1(4, 1, FormatError::InvalidObjectHeaderVersion(1))]
    fn a_prefix_that_is_not_version_2_is_rejected(
        #[case] at: usize,
        #[case] byte: u8,
        #[case] expected: FormatError,
    ) {
        let mut header = ObjectHeaderPrefix::PLAIN.encode_header(&[]);
        header[at] = byte;
        assert_eq!(ObjectHeaderPrefix::parse(&header), Err(expected));
    }

    #[rstest]
    #[case::a_record_that_ends_the_region(
        MessageRecordLayout::PLAIN,
        &[0x01, 2, 0, 0x01, 7, 8],
        Ok(Some(MessageRecord {
            msg_type: MessageType::DATASPACE,
            flags: MessageFlags::CONSTANT,
            creation_index: None,
            body: &[7, 8],
            body_range: 4..6,
        }))
    )]
    #[case::a_tracked_record(
        TRACKED,
        &[0x01, 1, 0, 0, 0x34, 0x12, 7],
        Ok(Some(MessageRecord {
            msg_type: MessageType::DATASPACE,
            flags: MessageFlags::NONE,
            creation_index: Some(0x1234),
            body: &[7],
            body_range: 6..7,
        }))
    )]
    #[case::a_tail_shorter_than_a_record_prefix(MessageRecordLayout::PLAIN, &[0x01, 2, 0], Ok(None))]
    #[case::a_tail_shorter_than_a_tracked_prefix(TRACKED, &[0x01, 0, 0, 0, 0x34], Ok(None))]
    #[case::a_body_past_the_region(
        MessageRecordLayout::PLAIN,
        &[0x01, 3, 0, 0, 7, 8],
        Err(FormatError::UnexpectedEof { expected: 7, available: 6 })
    )]
    fn a_record_is_read_within_its_region(
        #[case] layout: MessageRecordLayout,
        #[case] region: &[u8],
        #[case] expected: Result<Option<MessageRecord<'_>>, FormatError>,
    ) {
        assert_eq!(layout.next_message(region, 0), expected);
    }

    #[rstest]
    #[case::a_plain_record(MessageRecordLayout::PLAIN, &[0x01, 0, 0, 0][..], None)]
    #[case::a_tracked_record(TRACKED, &[0x01, 0, 0, 0, 0x34, 0x12][..], Some(0x1234))]
    #[case::a_tracked_record_cut_short(TRACKED, &[0x01, 0, 0, 0, 0x34][..], None)]
    fn a_creation_index_is_read_where_the_layout_has_one(
        #[case] layout: MessageRecordLayout,
        #[case] region: &[u8],
        #[case] expected: Option<u16>,
    ) {
        assert_eq!(layout.creation_index(region, 0), expected);
    }

    #[rstest]
    #[case::plain(MessageRecordLayout::PLAIN, &[0x0C, 1, 0, 0x02, 0xAA][..])]
    #[case::tracked(TRACKED, &[0x0C, 1, 0, 0x02, 0x02, 0x01, 0xAA][..])]
    fn a_record_stores_a_creation_index_only_where_the_header_tracks_one(
        #[case] layout: MessageRecordLayout,
        #[case] expected: &[u8],
    ) {
        assert_eq!(
            layout.record_with_creation_index(
                MessageType::ATTRIBUTE,
                MessageFlags::SHARED,
                0x0102,
                &[0xAA]
            ),
            expected
        );
    }

    #[rstest]
    #[case::plain(MessageRecordLayout::PLAIN, &[0x0A, 1, 0, 0, 0xAA][..])]
    #[case::tracked(TRACKED, &[0x0A, 1, 0, 0, 0, 0, 0xAA][..])]
    fn a_record_of_another_message_type_stores_a_zero_creation_index(
        #[case] layout: MessageRecordLayout,
        #[case] expected: &[u8],
    ) {
        assert_eq!(layout.record(MessageType::GROUP_INFO, &[0xAA]), expected);
    }

    #[rstest]
    #[case::neither_block(ObjectHeaderPrefix::PLAIN, 0, 11)]
    #[case::the_times(
        ObjectHeaderPrefix { times: Some(TIMES), ..ObjectHeaderPrefix::PLAIN },
        v2_bytes::HeaderFlags::STORES_TIMES.0,
        11 + 16
    )]
    #[case::the_phase_change(
        ObjectHeaderPrefix { attribute_phase_change: Some(PHASE), ..ObjectHeaderPrefix::PLAIN },
        v2_bytes::HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE.0,
        11 + 4
    )]
    fn an_optional_block_sets_its_own_flag_bit(
        #[case] prefix: ObjectHeaderPrefix,
        #[case] flags: u8,
        #[case] header_len: usize,
    ) {
        let header = prefix.encode_header(&[]);
        assert_eq!((header[5], header.len()), (flags, header_len));
    }

    #[test]
    #[should_panic(
        expected = "message type 0x0117 does not fit the 1-byte type field of a version 2 object header"
    )]
    fn a_record_of_a_type_wider_than_its_field_panics() {
        MessageRecordLayout::PLAIN.record(MessageType::from_u16(0x0117), &[]);
    }

    #[test]
    #[should_panic(
        expected = "a 65536-byte message body does not fit the 2-byte message size field"
    )]
    fn a_record_of_a_body_wider_than_its_size_field_panics() {
        MessageRecordLayout::PLAIN.record(
            MessageType::ATTRIBUTE,
            &vec![0; OBJECT_HEADER_MESSAGE_MAX + 1],
        );
    }

    #[test]
    fn a_continuation_message_parses_to_its_address_and_length() {
        let mut body = 0x800u32.to_le_bytes().to_vec();
        body.extend_from_slice(&0x40u64.to_le_bytes());
        assert_eq!(
            ObjectHeaderContinuation::parse(&body, 4, 8),
            Ok(ObjectHeaderContinuation {
                address: StoredAddress::new(0x800),
                length: 0x40,
            })
        );
    }

    #[rstest]
    #[case::a_block_of_messages(b"OCHK\x01\x00\x00\x00cksm", Ok(4..8))]
    #[case::an_empty_block(b"OCHKcksm", Ok(4..4))]
    #[case::a_block_shorter_than_its_signature_and_checksum(
        b"OCHKcks",
        Err(FormatError::UnexpectedEof { expected: 8, available: 7 })
    )]
    #[case::another_signature(b"OHDRcksm", Err(FormatError::InvalidObjectHeaderSignature))]
    fn a_continuation_block_holds_its_messages_between_signature_and_checksum(
        #[case] block: &[u8],
        #[case] expected: Result<Range<usize>, FormatError>,
    ) {
        assert_eq!(continuation_block_messages(block), expected);
    }

    const TRACKED: MessageRecordLayout = MessageRecordLayout::from_header_flags(
        v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0
            | v2_bytes::HeaderFlags::INDEXES_CREATION_ORDER.0,
    );

    const TIMES: ObjectTimes = ObjectTimes {
        access: 1,
        modification: 2,
        change: 3,
        birth: 4,
    };

    const PHASE: AttributePhaseChange = AttributePhaseChange {
        max_compact: 32,
        min_dense: 24,
    };
}
