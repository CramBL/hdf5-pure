use core::ops::Range;

use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;

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
    pub(super) const fn header_flags(self) -> u8 {
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

    /// Returns the creation index, or `None` if the layout omits it or the field is truncated.
    ///
    /// Pass the starting offset of a message record in `region` as `msg_start`.
    ///
    /// # Panics
    ///
    /// With overflow checks enabled, panics if the layout tracks creation order and
    /// `msg_start + 4` exceeds [`usize::MAX`].
    pub fn creation_index(self, region: &[u8], msg_start: usize) -> Option<u16> {
        self.tracked
            .then(|| region.get(msg_start + 4..)?.first_chunk().copied())
            .flatten()
            .map(u16::from_le_bytes)
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

#[cfg(test)]
#[path = "tests/record.rs"]
mod tests;
