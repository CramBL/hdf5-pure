use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use crate::checksum;
use crate::checksum::JenkinsLookup3;
use crate::convert::Narrow;
use crate::error::FormatError;

use super::super::v2::CHECKSUM_LEN;
use super::COUNT_LENGTH;
use super::ObjectReferenceCountField;

impl ObjectReferenceCountField {
    /// Prepares count and checksum writes for the complete associated chunk in `current`.
    ///
    /// Checks the exact chunk range and original count bytes. For a version 2 header, verifies
    /// the input checksum when the `checksum` feature is enabled and always computes a new
    /// checksum over the current chunk with the replacement count. Leaves `current` unchanged.
    ///
    /// The caller keeps the observed framing and field association valid while acquiring
    /// `current`. Source freshness, object identity, allocation ownership, count authority, and
    /// publication permissions require separate evidence. This operation checks existing field
    /// geometry without parsing the header again.
    ///
    /// # Errors
    ///
    /// Returns [`ObjectReferenceCountUpdateError::Chunk`] if `current` has a different range,
    /// [`ObjectReferenceCountUpdateError::Value`] if the current count differs from the observed
    /// value, or [`ObjectReferenceCountUpdateError::Format`] for range or checksum errors,
    /// including inconsistent checksum input lengths. The count field must fit before the
    /// checksum in a version 2 chunk.
    ///
    /// # Examples
    ///
    /// See [`ObjectReferenceCountObserver`](super::ObjectReferenceCountObserver) for observing a
    /// field and preparing its count and checksum writes.
    pub fn update(
        self,
        current: FileRegion<'_>,
        count: u32,
    ) -> Result<ObjectReferenceCountUpdate, ObjectReferenceCountUpdateError> {
        let Self {
            value,
            range,
            chunk,
            flags,
        } = self;
        if current.range() != chunk {
            return Err(ObjectReferenceCountUpdateError::Chunk {
                expected: chunk,
                provided: current.range(),
            });
        }
        let observed = u32::from_le_bytes(current.subregion(range)?.bytes().try_into().map_err(
            |_error| {
                FormatError::Internal("an observed count field does not contain four bytes".into())
            },
        )?);
        if observed != value {
            return Err(ObjectReferenceCountUpdateError::Value {
                field: range.start(),
                expected: value,
                actual: observed,
            });
        }
        let bytes = count.to_le_bytes();
        let checksum = if flags.is_some() {
            let Some((body, _)) = current.bytes().split_last_chunk::<CHECKSUM_LEN>() else {
                return Err(FormatError::UnexpectedEof {
                    expected: CHECKSUM_LEN,
                    available: current.bytes().len(),
                }
                .into());
            };
            let body_range = FileRange::new(chunk.start(), body.len().to_u64())?;
            let body = FileRegion::new(body_range, body)?;
            body.subregion(range)?;
            checksum::verify_trailing(current.bytes())?;
            let before = FileRange::new(chunk.start(), range.start().get() - chunk.start().get())?;
            let after = FileRange::new(range.end(), body_range.end().get() - range.end().get())?;
            let mut checksum = JenkinsLookup3::new(body_range.length());
            checksum.update(body.subregion(before)?.bytes())?;
            checksum.update(&bytes)?;
            checksum.update(body.subregion(after)?.bytes())?;
            let at = FileRange::new(body_range.end(), CHECKSUM_LEN.to_u64())?;
            Some((at, checksum.finish()?.to_le_bytes()))
        } else {
            None
        };
        Ok(ObjectReferenceCountUpdate {
            count: (range, bytes),
            checksum,
        })
    }
}

/// Encoded count bytes and any containing-chunk checksum, with their physical write ranges.
///
/// Retains four count bytes and, for a version 2 header, four checksum bytes.
/// Applying the writes requires separate source and publication permissions.
#[derive(Debug, Eq, PartialEq)]
pub struct ObjectReferenceCountUpdate {
    count: (FileRange, [u8; COUNT_LENGTH]),
    checksum: Option<(FileRange, [u8; CHECKSUM_LEN])>,
}

impl ObjectReferenceCountUpdate {
    /// Returns the four-byte count write followed by any four-byte chunk-checksum write.
    ///
    /// Version 1 prefix updates have one write. Version 2 updates have two, including for a count
    /// in a continuation chunk. Every range uses physical coordinates.
    pub fn writes(&self) -> impl Iterator<Item = (FileRange, &[u8])> + '_ {
        let Self { count, checksum } = self;
        core::iter::once((count.0, count.1.as_slice())).chain(
            checksum
                .as_ref()
                .map(|(range, bytes)| (*range, bytes.as_slice())),
        )
    }
}

/// A mismatched chunk or count, or an encoding failure during count preparation.
#[derive(Debug, Eq, PartialEq)]
pub enum ObjectReferenceCountUpdateError {
    /// A supplied region with a range different from the observed complete chunk.
    Chunk {
        /// The complete chunk associated with the observed field.
        expected: FileRange,
        /// The range associated with the supplied current bytes.
        provided: FileRange,
    },
    /// A range or checksum error, including inconsistent checksum input lengths.
    Format(FormatError),
    /// Current field bytes that decode to a count different from the observed value.
    Value {
        /// The physical start of the count field.
        field: PhysicalAddress,
        /// The original observed count.
        expected: u32,
        /// The count decoded from the supplied current bytes.
        actual: u32,
    },
}

impl From<FormatError> for ObjectReferenceCountUpdateError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}
