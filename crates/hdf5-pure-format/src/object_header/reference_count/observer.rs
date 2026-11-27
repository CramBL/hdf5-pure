use core::cell::Cell;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

use super::{ObjectReferenceCount, ObjectReferenceCountIssue};
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::{ObjectHeader, ObjectHeaderChunk};

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
///     AccessMode, FormatWidths, MessageFilter, ObjectHeader, ObjectHeaderWriter,
///     ObjectReferenceCount, ObjectReferenceCountMessage, ObjectReferenceCountObserver,
/// };
///
/// let header_bytes = |count| {
///     let mut writer = ObjectHeaderWriter::new();
///     let body = ObjectReferenceCountMessage::new(count).encode();
///     writer.append_raw_message(MessageType::OBJECT_REFERENCE_COUNT, &body)?;
///     writer.serialize()
/// };
/// let bytes = header_bytes(2)?;
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
/// let expected = header_bytes(1)?;
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
            V1_HEADER_VERSION => ObjectReferenceCount::from_prefix(chunk, header.reference_count)?,
            V2_HEADER_VERSION => explicit.get().unwrap_or(ObjectReferenceCount::ImplicitOne),
            version => {
                ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::HeaderVersion(version))
            }
        })
    }
}
const V1_HEADER_VERSION: u8 = 1;

// "Version 2 Data Object Header Prefix" of the format specification, version 4.0, defines
// version 2 object headers.
const V2_HEADER_VERSION: u8 = 2;
