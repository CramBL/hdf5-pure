//! Count observations, physical field associations, and prepared count updates.
//!
//! Version 1 headers use the prefix count defined in "Version 1 Data Object Header Prefix" of the
//! [format specification, version 4.0][v1]. Version 2 headers are defined in "Version 2 Data Object
//! Header Prefix" of [the same document][v2], and their version 0 count message in "The Object
//! Reference Count Message" of [the same document][count].
//!
//! [v1]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
//! [v2]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
//! [count]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_refcount

use core::cell::Cell;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;

use super::MessageRecord;
use super::ObjectHeader;
use super::ObjectHeaderChunk;

mod update;

pub use update::ObjectReferenceCountUpdate;
pub use update::ObjectReferenceCountUpdateError;

/// Provisional count observations from one object-header parse.
///
/// The caller passes the initial chunk first, then each continuation chunk once to
/// [`Self::observe`], and reports every message type through [`Self::observe_message_type`].
/// Finish only after the complete parser and every observation succeed. Retains fixed-size
/// scratch. The caller establishes source freshness, allocation ownership, and count authority
/// separately before retirement.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::{BaseAddressExt, FileRegion};
/// use hdf5_pure_core::PhysicalAddress;
/// use hdf5_pure_format::{BaseAddress, FormatError, MessageType};
/// use hdf5_pure_format::__private::{
///     AccessMode, FormatWidths, MessageFilter, ObjectHeader, ObjectHeaderPrefix,
///     ObjectReferenceCount, ObjectReferenceCountObserver,
/// };
///
/// let prefix = ObjectHeaderPrefix::PLAIN;
/// let records = prefix.layout.record(MessageType::OBJECT_REFERENCE_COUNT, &[0, 2, 0, 0, 0]);
/// let bytes = prefix.encode_header(&records);
/// let observer = ObjectReferenceCountObserver::new(PhysicalAddress::new(0));
/// let mut observation_error = None;
/// let header = ObjectHeader::parse_from_source_with_chunk_observer(
///     bytes.as_slice(),
///     AccessMode::ReadOnly,
///     PhysicalAddress::new(0),
///     FormatWidths::from_sizes(8, 8)?,
///     BaseAddress::ZERO,
///     MessageFilter::Only(&mut |kind, _| {
///         observer.observe_message_type(kind);
///         false
///     }),
///     &mut |chunk| {
///         if observation_error.is_none() {
///             observation_error = observer.observe(chunk).err();
///         }
///     },
/// )?;
/// if let Some(error) = observation_error {
///     return Err(error);
/// }
/// let ObjectReferenceCount::Version2(field) = observer.finish(&header)? else {
///     panic!("expected an explicit version 2 count");
/// };
/// assert_eq!(field.value(), 2);
/// assert_eq!(field.range().length(), 4);
/// assert!(field.chunk().contains(field.range()));
///
/// let current = FileRegion::new(field.chunk(), &bytes)?;
/// let update = field.update(current, 1).unwrap();
/// let mut writes = update.writes();
/// assert_eq!(writes.next().unwrap(), (field.range(), [1, 0, 0, 0].as_slice()));
/// let (checksum_range, checksum_bytes) = writes.next().unwrap();
/// let expected_records =
///     prefix.layout.record(MessageType::OBJECT_REFERENCE_COUNT, &[0, 1, 0, 0, 0]);
/// let expected = prefix.encode_header(&expected_records);
/// assert_eq!(checksum_range.length(), 4);
/// assert_eq!(checksum_range.end(), field.chunk().end());
/// assert_eq!(checksum_bytes, &expected[expected.len() - 4..]);
/// assert!(writes.next().is_none());
/// # Ok::<(), FormatError>(())
/// ```
pub struct ObjectReferenceCountObserver {
    address: PhysicalAddress,
    initial: Cell<Option<FileRange>>,
    explicit: Cell<Option<ObjectReferenceCount>>,
    count_message: Cell<bool>,
}

impl ObjectReferenceCountObserver {
    /// Starts an observation for an already resolved physical object-header address.
    pub const fn new(address: PhysicalAddress) -> Self {
        Self {
            address,
            initial: Cell::new(None),
            explicit: Cell::new(None),
            count_message: Cell::new(false),
        }
    }

    /// Reads version 2 count records and retains their physical field and chunk ranges.
    ///
    /// Records the first chunk's allocation for the initial-prefix association. Unsupported
    /// count bodies and repeated counts are classified in the result of [`Self::finish`].
    /// Permits trailing body bytes after the version byte and four-byte count.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a record body exceeds the borrowed message
    /// region, [`FormatError::OffsetOverflow`] if field-address calculations overflow, or
    /// [`FormatError::InsufficientRegion`] if a field lies outside that region.
    pub fn observe(&self, chunk: ObjectHeaderChunk<'_>) -> Result<(), FormatError> {
        if self.initial.get().is_none() {
            self.initial.set(Some(chunk.allocation()));
        }
        let Some(layout) = chunk.v2_layout() else {
            return Ok(());
        };
        let records = chunk.records();
        let mut position = 0;
        while let Some(record) = layout.next_message(records.bytes(), position)? {
            position = record.body_range.end;
            if record.msg_type != MessageType::OBJECT_REFERENCE_COUNT {
                continue;
            }
            let observed = if self.explicit.get().is_some() {
                ObjectReferenceCount::Ambiguous
            } else {
                ObjectReferenceCount::from_record(chunk, record)?
            };
            self.explicit.set(Some(observed));
        }
        Ok(())
    }

    /// Records count-message presence, including when message filtering discards the body.
    ///
    /// The caller reports every message type. A count message in a version 1 header is ambiguous
    /// with that header's prefix count.
    pub fn observe_message_type(&self, kind: MessageType) {
        if kind == MessageType::OBJECT_REFERENCE_COUNT {
            self.count_message.set(true);
        }
    }

    /// Combines the provisional observations with the matching successfully parsed header.
    ///
    /// The caller first completes the full parse and checks every [`Self::observe`] result.
    /// Returns an unsupported observation if the initial chunk association is missing or
    /// inconsistent. Uses the parsed prefix count for version 1 and observed records for
    /// version 2.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the version 1 count range cannot be represented.
    pub fn finish(self, header: &ObjectHeader) -> Result<ObjectReferenceCount, FormatError> {
        let Self {
            address,
            initial,
            explicit,
            count_message,
        } = self;
        let Some(chunk) = initial.get().filter(|chunk| chunk.start() == address) else {
            return Ok(ObjectReferenceCount::Unsupported(
                ObjectReferenceCountIssue::InitialChunk,
            ));
        };
        Ok(match header.version {
            V1_HEADER_VERSION if count_message.get() => ObjectReferenceCount::Ambiguous,
            V1_HEADER_VERSION => match header.reference_count {
                Some(value) => {
                    let range = FileRange::new(
                        address.checked_add(V1_COUNT_OFFSET)?,
                        COUNT_LENGTH.to_u64(),
                    )?;
                    if !chunk.contains(range) {
                        return Ok(ObjectReferenceCount::Unsupported(
                            ObjectReferenceCountIssue::InitialChunk,
                        ));
                    }
                    ObjectReferenceCount::Version1(ObjectReferenceCountField {
                        value,
                        range,
                        chunk,
                        flags: None,
                    })
                }
                None => {
                    ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::MissingPrefixCount)
                }
            },
            V2_HEADER_VERSION => explicit.get().unwrap_or(ObjectReferenceCount::ImplicitOne),
            version => {
                ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::HeaderVersion(version))
            }
        })
    }
}

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
    fn from_record(
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
const COUNT_LENGTH: usize = 4;

// "Version 1 Data Object Header Prefix" of the format specification, version 4.0, places the
// unsigned 32-bit reference count after four prefix bytes.
const V1_COUNT_OFFSET: u64 = 4;
const V1_HEADER_VERSION: u8 = 1;

// "Version 2 Data Object Header Prefix" of the format specification, version 4.0, defines
// version 2 object headers.
const V2_HEADER_VERSION: u8 = 2;

#[cfg(test)]
mod tests;
