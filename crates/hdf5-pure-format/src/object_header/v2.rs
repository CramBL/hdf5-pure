use alloc::vec::Vec;
use byteorder::{ByteOrder, LittleEndian};

use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum::jenkins_lookup3;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::error::OBJECT_HEADER_MESSAGE_MAX;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;
use crate::object_header::{HeaderMessage, MessageFilter, ObjectHeader, ParseContext};
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
        bytes::ensure_len(data, offset, 6)?;
        let prefix = Self::parse_prefix(&data[offset..])?;

        let chunk0_msg_start =
            offset
                .checked_add(prefix.len)
                .ok_or(FormatError::UnexpectedEof {
                    expected: usize::MAX,
                    available: data.len(),
                })?;
        let chunk0_msg_end =
            chunk0_msg_start
                .checked_add(prefix.chunk0_size)
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

        let has_creation_order = prefix.tracks_creation_order();

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
            flags: prefix.flags.raw(),
            access_time: prefix.timestamps.map(|timestamps| timestamps.access),
            modification_time: prefix.timestamps.map(|timestamps| timestamps.modification),
            change_time: prefix.timestamps.map(|timestamps| timestamps.change),
            birth_time: prefix.timestamps.map(|timestamps| timestamps.birth),
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

    pub(super) fn parse_v2_from_source<S: MetadataSource + ?Sized>(
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
        let prefix = Self::parse_prefix(&head)?;
        let prefix_len = prefix.len;

        // The chunk 0 body, including the prefix, messages, and checksum, is
        // contiguous from `address`. Reading the complete region gives the checksum
        // the same byte range as the buffered parser.
        let chunk0_end =
            prefix_len
                .checked_add(prefix.chunk0_size)
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

        let has_creation_order = prefix.tracks_creation_order();
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
            flags: prefix.flags.raw(),
            access_time: prefix.timestamps.map(|timestamps| timestamps.access),
            modification_time: prefix.timestamps.map(|timestamps| timestamps.modification),
            change_time: prefix.timestamps.map(|timestamps| timestamps.change),
            birth_time: prefix.timestamps.map(|timestamps| timestamps.birth),
        })
    }

    fn parse_prefix(data: &[u8]) -> Result<Prefix, FormatError> {
        bytes::ensure_len(data, 0, 6)?;

        let version = data[4];
        if version != 2 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }

        let flags = HeaderFlags::new(data[5]);
        let mut pos = 6;

        let timestamps = if flags.stores_times() {
            bytes::ensure_len(data, pos, 16)?;

            let timestamps = Timestamps {
                access: LittleEndian::read_u32(&data[pos..pos + 4]),
                modification: LittleEndian::read_u32(&data[pos + 4..pos + 8]),
                change: LittleEndian::read_u32(&data[pos + 8..pos + 12]),
                birth: LittleEndian::read_u32(&data[pos + 12..pos + 16]),
            };

            pos += 16;
            Some(timestamps)
        } else {
            None
        };

        if flags.stores_attribute_phase_change() {
            bytes::ensure_len(data, pos, 4)?;
            pos += 4;
        }

        let chunk_size_width = flags.chunk_size_width();
        let chunk0_size = bytes::read_uint_width(data, pos, chunk_size_width)?.to_usize()?;
        pos += usize::from(chunk_size_width.get());

        Ok(Prefix {
            flags,
            timestamps,
            chunk0_size,
            len: pos,
        })
    }
}

/// Maximum length of a version 2 object header's fixed prefix: signature (4) +
/// version (1) + flags (1) + optional access/modification/change/birth times
/// (16) + optional attribute phase-change thresholds (4) + the chunk-0 size
/// field (up to 8). Reading this many bytes always covers the prefix, so a
/// caller reads one bounded window of the file.
pub const OBJECT_HEADER_PREFIX_MAX_LEN: usize = 34;

pub struct ParsedObjectHeaderPrefix {
    pub prefix: ObjectHeaderPrefix,
    pub chunk0_size: u64,
    pub len: usize,
}

/// How a version 2 object header's message records are laid out, and what its
/// flags say about attribute creation order.
///
/// Every record opens with a type byte, a 2-byte body size and a flags byte. A
/// header that *tracks* attribute creation order - bit 2 of the object header's
/// own flags, what `H5Pset_attr_creation_order` and h5py's `track_order=True`
/// turn on, and what netCDF-4 sets on every object it writes - follows those
/// with a 2-byte creation index, so each of its records is 6 bytes wide.
/// A header that also *indexes* that order (bit 3) carries a creation-order
/// B-tree beside the name index once its attributes go dense. The reference C
/// library reads that bit back out of the header when it builds an Attribute
/// Info message, so a rewrite that dropped it would quietly stop indexing.
///
/// Both bits are properties of the whole header, so chunk 0 and every
/// continuation block of one header share a layout. Carrying it beside the bytes
/// (`OhRegion`) is what keeps the two dozen walkers and the emitters in this
/// module from having to agree about it one by one.
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

    /// Parse the block at `at`. The caller must have checked that 16 bytes are
    /// available there.
    fn parse(prefix: &[u8], at: usize) -> Self {
        let field =
            |i: usize| u32::from_le_bytes(prefix[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
        Self {
            access: field(0),
            modification: field(1),
            change: field(2),
            birth: field(3),
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

    /// Parse the block at `at`. The caller must have checked that 4 bytes are
    /// available there.
    fn parse(prefix: &[u8], at: usize) -> Self {
        Self {
            max_compact: u16::from_le_bytes(prefix[at..at + 2].try_into().unwrap()),
            min_dense: u16::from_le_bytes(prefix[at + 2..at + 4].try_into().unwrap()),
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
        crate::bytes::ensure_len(data, 0, 6)?;
        if data[..4] != *b"OHDR" {
            return Err(FormatError::InvalidObjectHeaderSignature);
        }
        if data[4] != 2 {
            return Err(FormatError::InvalidObjectHeaderVersion(data[4]));
        }
        let flags = data[5];
        let mut pos = 6usize;
        let times = if flags & OH_FLAG_STORE_TIMES != 0 {
            crate::bytes::ensure_len(data, pos, ObjectTimes::LEN)?;
            let times = ObjectTimes::parse(data, pos);
            pos += ObjectTimes::LEN;
            Some(times)
        } else {
            None
        };
        let attribute_phase_change = if flags & OH_FLAG_ATTRIBUTE_PHASE_CHANGE != 0 {
            crate::bytes::ensure_len(data, pos, AttributePhaseChange::LEN)?;
            let phase = AttributePhaseChange::parse(data, pos);
            pos += AttributePhaseChange::LEN;
            Some(phase)
        } else {
            None
        };
        let size_width = usize::from(UintWidth::from_flags(flags).get());
        crate::bytes::ensure_len(data, pos, size_width)?;
        let chunk0_size = read_le(&data[pos..pos + size_width]) as u64;
        pos += size_width;
        Ok(ParsedObjectHeaderPrefix {
            prefix: Self {
                layout: MessageRecordLayout::from_header_flags(flags),
                times,
                attribute_phase_change,
            },
            chunk0_size,
            len: pos,
        })
    }

    /// Returns a version 2 object header of one chunk: this prefix, the message records
    /// `messages`, and the checksum.
    ///
    /// The Size of Chunk #0 field takes the fewest bytes that hold the length of `messages`.
    pub fn encode_header(self, messages: &[u8]) -> Vec<u8> {
        let total = messages.len();
        let width = UintWidth::smallest_for_len(total);
        let mut buf = Vec::with_capacity(8 + self.optional_len() + total + 4);
        buf.extend_from_slice(b"OHDR");
        buf.push(2); // version
        buf.push(width.flag_bits() | self.header_flags());
        if let Some(times) = self.times {
            buf.extend_from_slice(&times.to_bytes());
        }
        if let Some(phase) = self.attribute_phase_change {
            buf.extend_from_slice(&phase.to_bytes());
        }
        buf.extend_from_slice(&(total as u64).to_le_bytes()[..usize::from(width.get())]);
        buf.extend_from_slice(messages);
        let checksum = jenkins_lookup3(&buf);
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

    /// Parse the message record at `p` within a chunk's message region,
    /// returning `(message type, body start, body end)`. The next record begins
    /// at `body end`. Returns `Ok(None)` once fewer bytes remain than a record
    /// prefix takes (a clean end of the region, or the gap the reference C
    /// library leaves when a chunk's free space is too small to hold a message),
    /// and `Err` if a record's declared body runs past the region. Centralizes
    /// the bounds check shared by every walker.
    pub fn next_message(
        self,
        region: &[u8],
        p: usize,
    ) -> Result<Option<(MessageType, usize, usize)>, FormatError> {
        if p + self.prefix_len() > region.len() {
            return Ok(None);
        }
        let msg_type = MessageType::from_u16(region[p] as u16);
        let msg_size = u16::from_le_bytes([region[p + 1], region[p + 2]]) as usize;
        let body = p + self.prefix_len();
        let body_end = body + msg_size;
        if body_end > region.len() {
            return Err(FormatError::UnexpectedEof {
                expected: body_end,
                available: region.len(),
            });
        }
        Ok(Some((msg_type, body, body_end)))
    }

    /// The creation index the record at `msg_start` carries, or `None` where the
    /// layout has no such field. The caller must have located `msg_start` with
    /// [`next_message`](Self::next_message), which bounds the read.
    pub fn creation_index(self, region: &[u8], msg_start: usize) -> Option<u16> {
        self.tracked
            .then(|| u16::from_le_bytes([region[msg_start + 4], region[msg_start + 5]]))
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
        let mut m = Vec::with_capacity(self.prefix_len() + body.len());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the first assertion above admits only a type that fits the 1-byte field"
        )]
        m.push(msg_type.to_u16() as u8);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the second assertion above bounds the body to the 2-byte size field"
        )]
        m.extend_from_slice(&(body.len() as u16).to_le_bytes());
        m.push(flags.get());
        if self.tracks_creation_order() {
            m.extend_from_slice(&creation_index.to_le_bytes());
        }
        m.extend_from_slice(body);
        m
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

/// Read a little-endian unsigned integer of `bytes.len()` (≤ 8) bytes.
#[expect(
    clippy::cast_possible_truncation,
    reason = "callers parse in-file sizes/offsets bounded by the in-memory image; downstream \
              slicing is length-checked, so a malformed oversized field errors rather than reads OOB"
)]
fn read_le(bytes: &[u8]) -> usize {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v as usize
}

/// Groups the four timestamps stored together in a version 2 object header.
///
/// The fields are present when the timestamp flag is set in the version 2
/// object header prefix. Their order is defined in "Version 2 Data Object
/// Header Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
#[derive(Clone, Copy)]
struct Timestamps {
    access: u32,
    modification: u32,
    change: u32,
    birth: u32,
}

/// Represents the decoded variable-length prefix of a version 2 object header.
///
/// The prefix contains the flags byte, optional timestamp and attribute phase
/// change fields, and the encoded size of chunk 0. `len` is the number of bytes
/// from the `OHDR` signature through the chunk size field. The layout is
/// defined in "Version 2 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
struct Prefix {
    flags: HeaderFlags,
    timestamps: Option<Timestamps>,
    chunk0_size: usize,
    len: usize,
}

impl Prefix {
    fn tracks_creation_order(&self) -> bool {
        self.flags.tracks_creation_order()
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
