use hdf5_pure_core::__private::FileRange;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::object_header::{MessageRecord, ObjectHeaderChunk};

/// An observed object-reference-count encoding, including implicit and unusable forms.
///
/// An observation establishes encoding and physical associations. The caller determines whether
/// the value is authoritative for an editing operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectReferenceCount {
    /// Multiple version 2 count messages, or a version 1 header with a count message.
    Ambiguous,
    /// A version 2 header without an observed count message.
    ImplicitOne,
    /// An encoding or association that cannot supply a supported count observation.
    Unsupported(ObjectReferenceCountIssue),
    /// A four-byte count in a version 1 object-header prefix.
    Version1(ObjectReferenceCountField),
    /// A version 0 count message in a version 2 object header.
    Version2(ObjectReferenceCountField),
}

impl ObjectReferenceCount {
    /// Associates a version 1 prefix count with its four-byte field in the initial chunk.
    ///
    /// Returns an unsupported observation for a missing count or a field outside the chunk.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the field address or its end exceeds
    /// [`u64::MAX`].
    pub(super) fn from_prefix(chunk: FileRange, count: Option<u32>) -> Result<Self, FormatError> {
        let Some(value) = count else {
            return Ok(Self::Unsupported(
                ObjectReferenceCountIssue::MissingPrefixCount,
            ));
        };
        let range = FileRange::new(
            chunk.start().checked_add(V1_COUNT_OFFSET)?,
            COUNT_LENGTH.to_u64(),
        )?;
        if !chunk.contains(range) {
            return Ok(Self::Unsupported(ObjectReferenceCountIssue::InitialChunk));
        }
        Ok(Self::Version1(ObjectReferenceCountField {
            value,
            range,
            chunk,
            flags: None,
        }))
    }

    pub(super) fn from_record(
        chunk: ObjectHeaderChunk<'_>,
        record: MessageRecord<'_>,
    ) -> Result<Self, FormatError> {
        let MessageRecord {
            msg_type: _,
            flags,
            creation_index: _,
            body,
            body_range,
        } = record;
        if flags.is_shared() {
            return Ok(Self::Unsupported(ObjectReferenceCountIssue::Shared));
        }
        let message = match ObjectReferenceCountMessage::parse(body) {
            Ok(message) => message,
            Err(issue) => return Ok(Self::Unsupported(issue)),
        };
        let records = chunk.records();
        let start = records
            .range()
            .start()
            .checked_add(body_range.start.to_u64())?
            .checked_add(MESSAGE_COUNT_OFFSET)?;
        let range = FileRange::new(start, COUNT_LENGTH.to_u64())?;
        records.subregion(range)?;
        Ok(Self::Version2(ObjectReferenceCountField {
            value: message.count(),
            range,
            chunk: chunk.allocation(),
            flags: Some(flags),
        }))
    }
}

/// The unsigned count in a version 0 object-reference-count message.
///
/// The parser reads the version byte and four-byte count defined in "The Object Reference Count
/// Message" of the [format specification, version 4.0][spec], accepting trailing body bytes. The
/// encoder writes only the five-byte message body. The caller checks record flags, message
/// uniqueness, and allocation ownership separately.
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::ObjectReferenceCountMessage;
///
/// let message = ObjectReferenceCountMessage::new(0x1234_5678);
/// let bytes = message.encode();
/// assert_eq!(bytes, [0, 0x78, 0x56, 0x34, 0x12]);
/// let mut padded = bytes.to_vec();
/// padded.extend_from_slice(&[0xA5, 0x5A]);
/// let parsed = ObjectReferenceCountMessage::parse(&padded).unwrap();
/// assert_eq!(parsed, message);
/// assert_eq!(parsed.encode(), bytes);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_refcount
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReferenceCountMessage {
    count: u32,
}

impl ObjectReferenceCountMessage {
    /// Creates a version 0 message with any unsigned 32-bit count, including zero.
    pub const fn new(count: u32) -> Self {
        Self { count }
    }

    /// Parses the version byte and count, ignoring any trailing body bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ObjectReferenceCountIssue::BodyLength`] if `body` is empty or a version 0 body
    /// is shorter than five bytes. Returns [`ObjectReferenceCountIssue::MessageVersion`] if the
    /// first byte is not zero.
    pub fn parse(body: &[u8]) -> Result<Self, ObjectReferenceCountIssue> {
        let Some((&version, rest)) = body.split_first() else {
            return Err(ObjectReferenceCountIssue::BodyLength(0));
        };
        if version != MESSAGE_VERSION {
            return Err(ObjectReferenceCountIssue::MessageVersion(version));
        }
        let Some(&count) = rest.first_chunk::<COUNT_LENGTH>() else {
            return Err(ObjectReferenceCountIssue::BodyLength(
                body.len()
                    .narrow_or_else(|| ObjectReferenceCountIssue::BodyLength(u16::MAX))?,
            ));
        };
        Ok(Self::new(u32::from_le_bytes(count)))
    }

    /// Returns the unsigned count independently of any physical field association.
    pub const fn count(self) -> u32 {
        self.count
    }

    /// Serializes the version byte and little-endian count into a five-byte body.
    pub const fn encode(self) -> [u8; MESSAGE_LENGTH] {
        let count = self.count.to_le_bytes();
        [MESSAGE_VERSION, count[0], count[1], count[2], count[3]]
    }

    /// The count assumed for a version 2 object header without a count message.
    pub const IMPLICIT_ONE: Self = Self::new(1);
}

/// A decoded count and its four-byte physical field within a complete header chunk.
///
/// The chunk association includes framing and, for version 2, the checksum. The caller
/// establishes source freshness and allocation ownership separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectReferenceCountField {
    value: u32,
    range: FileRange,
    chunk: FileRange,
    flags: Option<MessageFlags>,
}

impl ObjectReferenceCountField {
    /// Returns the decoded unsigned count, independently of any retirement policy.
    pub const fn value(self) -> u32 {
        self.value
    }

    /// Returns the four-byte physical count field without surrounding prefix or record bytes.
    pub const fn range(self) -> FileRange {
        self.range
    }

    /// Returns the complete containing physical chunk as reported by the parser.
    pub const fn chunk(self) -> FileRange {
        self.chunk
    }

    /// Returns the original version 2 record flags, or `None` for a version 1 prefix field.
    pub const fn flags(self) -> Option<MessageFlags> {
        self.flags
    }
}

/// The reason an observation cannot supply a supported count field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectReferenceCountIssue {
    /// A count-message body shorter than five bytes, with its observed length.
    BodyLength(u16),
    /// An object-header version other than 1 or 2.
    HeaderVersion(u8),
    /// A missing or mismatched initial chunk, or one too short for the version 1 count.
    InitialChunk,
    /// A count-message version other than 0.
    MessageVersion(u8),
    /// A parsed version 1 header without its prefix count.
    MissingPrefixCount,
    /// A shared count message whose body is not an inline count.
    Shared,
}

// "The Object Reference Count Message" of the format specification, version 4.0, defines a
// version byte of 0 followed by an unsigned 32-bit reference count.
const MESSAGE_VERSION: u8 = 0;
const MESSAGE_LENGTH: usize = 5;
const MESSAGE_COUNT_OFFSET: u64 = 1;
pub(super) const COUNT_LENGTH: usize = 4;

// "Version 1 Data Object Header Prefix" of the format specification, version 4.0, places the
// unsigned 32-bit reference count after four prefix bytes.
const V1_COUNT_OFFSET: u64 = 4;

#[cfg(test)]
#[path = "tests/update.rs"]
mod tests;
