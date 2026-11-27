use alloc::vec::Vec;

use super::{CHECKSUM_LEN, MessageRecordLayout};
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::object_header::HeaderMessage;
use crate::object_header::ObjectHeader;
use crate::object_header::header::OHDR_SIGNATURE;
use crate::width::UintWidth;

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
    /// Returns the length in bytes of chunk 0, from the signature to the end of its checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the length exceeds `u64`, and
    /// [`FormatError::ValueTooLargeForPlatform`] if it exceeds `usize` on this host.
    pub(super) fn chunk0_len(&self) -> Result<usize, FormatError> {
        self.len
            .to_u64()
            .checked_add(self.chunk0_size)
            .and_then(|len| len.checked_add(CHECKSUM_LEN.to_u64()))
            .ok_or(FormatError::OffsetOverflow {
                offset: self.len.to_u64(),
                length: self.chunk0_size,
            })?
            .to_usize()
    }

    pub(super) fn object_header(&self, messages: Vec<HeaderMessage>) -> ObjectHeader {
        let times = self.prefix.times;
        ObjectHeader {
            version: OBJECT_HEADER_VERSION,
            prefix: Some(self.prefix),
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

/// Bit 4 of the flags, set where the prefix stores the attribute phase change values, from the
/// same section as [`MessageRecordLayout`]. libhdf5 calls it
/// `H5O_HDR_ATTR_STORE_PHASE_CHANGE`.
const OH_FLAG_ATTRIBUTE_PHASE_CHANGE: u8 = 0x10;

/// Bit 5 of the flags, set where the prefix stores the four times, from the same section as
/// [`MessageRecordLayout`]. libhdf5 calls it `H5O_HDR_STORE_TIMES`.
const OH_FLAG_STORE_TIMES: u8 = 0x20;

/// The access, modification, change, and birth times a version 2 object header prefix stores,
/// each in seconds since the Unix epoch.
///
/// The prefix stores them where bit 5 of its flags is set. libhdf5 sets that bit on every
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
/// The prefix stores them where bit 4 of its flags is set. libhdf5 sets that bit only where
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
        let messages_len = messages.len();
        let width = UintWidth::smallest_for_len(messages_len);
        let mut buf = Vec::with_capacity(self.prefix_len(width) + messages_len + CHECKSUM_LEN);
        self.write_prefix(&mut buf, width, messages_len);
        buf.extend_from_slice(messages);
        append_checksum(&mut buf);
        buf
    }

    /// Returns the shortest prefix length that can encode `messages_len` as its chunk size.
    pub(crate) fn prefix_len_for_messages(self, messages_len: usize) -> usize {
        self.prefix_len(UintWidth::smallest_for_len(messages_len))
    }

    /// Finishes a header whose messages start after `start` bytes.
    ///
    /// Chooses the smallest chunk length width, shifts the messages to that prefix length,
    /// and writes the prefix and checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `start` exceeds the buffer length, or
    /// [`FormatError::AllocationFailed`] if reserving space for the final prefix and checksum
    /// fails.
    pub(crate) fn finish_header(
        self,
        mut buf: Vec<u8>,
        start: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let end = buf.len();
        let messages_len = end.checked_sub(start).ok_or_else(|| {
            FormatError::Internal("object header buffer lacks its prefix reservation".into())
        })?;
        let width = UintWidth::smallest_for_len(messages_len);
        let prefix_len = self.prefix_len(width);
        let header_len = prefix_len + messages_len;
        let additional = (header_len + CHECKSUM_LEN).saturating_sub(end);
        buf.try_reserve_exact(additional)
            .map_err(|source| FormatError::AllocationFailed {
                values: additional,
                source,
            })?;
        if prefix_len != start {
            buf.resize(end.max(header_len), 0);
            buf.copy_within(start..end, prefix_len);
        }
        buf.truncate(header_len);
        self.write_prefix_into(&mut buf[..prefix_len], width, messages_len);
        append_checksum(&mut buf);
        buf.shrink_to_fit();
        Ok(buf)
    }

    fn prefix_len(self, width: UintWidth) -> usize {
        OHDR_SIGNATURE.len() + 2 + self.optional_len() + usize::from(width.get())
    }

    /// Appends the signature, flags, optional fields, and chunk-size field.
    ///
    /// The fields are defined in "Version 2 Data Object Header Prefix" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
    pub(super) fn write_prefix(self, buf: &mut Vec<u8>, width: UintWidth, messages_len: usize) {
        let start = buf.len();
        buf.resize(start + self.prefix_len(width), 0);
        self.write_prefix_into(&mut buf[start..], width, messages_len);
    }

    /// Writes the signature, version, flags, optional fields, and chunk size into `buf`.
    ///
    /// # Panics
    ///
    /// Panics if `buf.len()` differs from [`Self::prefix_len`] for `width`, or if
    /// `messages_len` exceeds the largest value representable at `width`.
    fn write_prefix_into(self, buf: &mut [u8], width: UintWidth, messages_len: usize) {
        let (signature, buf) = buf.split_at_mut(OHDR_SIGNATURE.len());
        signature.copy_from_slice(&OHDR_SIGNATURE);
        let (version_flags, mut buf) = buf.split_at_mut(2);
        version_flags.copy_from_slice(&[
            OBJECT_HEADER_VERSION,
            width.flag_bits() | self.header_flags(),
        ]);
        if let Some(times) = self.times {
            let (field, rest) = buf.split_at_mut(ObjectTimes::LEN);
            field.copy_from_slice(&times.to_bytes());
            buf = rest;
        }
        if let Some(phase) = self.attribute_phase_change {
            let (field, rest) = buf.split_at_mut(AttributePhaseChange::LEN);
            field.copy_from_slice(&phase.to_bytes());
            buf = rest;
        }
        width.write(buf, messages_len);
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
    pub(super) const fn optional_len(self) -> usize {
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

fn append_checksum(buf: &mut Vec<u8>) {
    let checksum = checksum::jenkins_lookup3(buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
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

/// The version of a version 2 object header, from the same section as
/// [`MessageRecordLayout`].
const OBJECT_HEADER_VERSION: u8 = 2;

#[cfg(test)]
#[path = "tests/prefix.rs"]
mod tests;
