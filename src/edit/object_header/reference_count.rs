//! Checked count updates within an owned object-header record region.

use core::ops::Range;

use hdf5_pure_format::__private::MessageRecord;
use hdf5_pure_format::__private::ObjectReferenceCountIssue;
use hdf5_pure_format::__private::ObjectReferenceCountMessage;

use crate::error::FormatError;
use crate::message_type::MessageType;

use super::OhRegion;

impl OhRegion {
    /// Checks local record framing and returns an exclusive count projection.
    ///
    /// Uses the implicit count of one when the region has no count message. A successful
    /// projection covers the complete record region and has at most one supported count message.
    ///
    /// # Errors
    ///
    /// Returns [`HeaderCountError::Ambiguous`] for repeated count messages,
    /// [`HeaderCountError::Unsupported`] for a shared or unsupported count body, or
    /// [`HeaderCountError::Framing`] for invalid record framing.
    pub(in crate::edit) fn reference_count_mut(
        &mut self,
    ) -> Result<HeaderCount<'_>, HeaderCountError> {
        let mut found = None;
        let mut position = 0;
        while let Some(record) = self.layout().next_message(self, position)? {
            let MessageRecord {
                msg_type,
                flags,
                creation_index: _,
                body,
                body_range,
            } = record;
            position = body_range.end;
            if msg_type != MessageType::OBJECT_REFERENCE_COUNT {
                continue;
            }
            let observed = if flags.is_shared() {
                Err(HeaderCountError::Unsupported(
                    ObjectReferenceCountIssue::Shared,
                ))
            } else {
                ObjectReferenceCountMessage::parse(body)
                    .map(|message| (message, CountPlace::Record(body_range)))
                    .map_err(HeaderCountError::Unsupported)
            };
            if found.replace(observed).is_some() {
                return Err(HeaderCountError::Ambiguous);
            }
        }
        let (message, place) = found.transpose()?.unwrap_or((
            ObjectReferenceCountMessage::IMPLICIT_ONE,
            CountPlace::Missing(position),
        ));
        Ok(HeaderCount {
            region: self,
            message,
            place,
        })
    }
}

/// An exclusive count projection with checked local record geometry.
///
/// Retains the parsed or implicit count and its location within the borrowed [`OhRegion`].
/// Physical source associations and authority to change a published count are separate.
pub(in crate::edit) struct HeaderCount<'region> {
    region: &'region mut OhRegion,
    message: ObjectReferenceCountMessage,
    place: CountPlace,
}

impl HeaderCount<'_> {
    /// Updates the owned count body or inserts a missing count message before the trailing gap.
    ///
    /// Leaves the region unchanged when `count` equals the parsed or implicit value. Updating an
    /// existing record preserves its flags, creation index, and trailing body bytes. Insertion
    /// uses the region's record layout and may allocate.
    ///
    /// # Errors
    ///
    /// Returns [`HeaderCountError::Framing`] with [`FormatError::Internal`] if the checked count
    /// location lies outside the borrowed region.
    pub(in crate::edit) fn set(self, count: u32) -> Result<(), HeaderCountError> {
        let Self {
            region,
            message,
            place,
        } = self;
        if count == message.count() {
            return Ok(());
        }
        let body = ObjectReferenceCountMessage::new(count).encode();
        match place {
            CountPlace::Missing(position) => {
                if region.get(..position).is_none() {
                    return Err(FormatError::Internal(
                        "an owned count insertion lies outside its record region".into(),
                    )
                    .into());
                }
                let record = region
                    .layout()
                    .record(MessageType::OBJECT_REFERENCE_COUNT, &body);
                region.bytes.splice(position..position, record);
            }
            CountPlace::Record(range) => {
                region
                    .bytes
                    .get_mut(range)
                    .and_then(|record| record.get_mut(..body.len()))
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "an owned count field lies outside its record body".into(),
                        )
                    })?
                    .copy_from_slice(&body);
            }
        }
        Ok(())
    }

    /// Returns the parsed local count, or one when the region has no count message.
    pub(in crate::edit) fn count(&self) -> u32 {
        self.message.count()
    }
}

/// A count-message ambiguity, unsupported encoding, or local framing failure.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum HeaderCountError {
    /// Multiple count messages in the record region.
    Ambiguous,
    /// Invalid record framing or an inconsistent checked local count location.
    Framing(FormatError),
    /// A count record that cannot be decoded as a supported inline message.
    Unsupported(ObjectReferenceCountIssue),
}

impl From<FormatError> for HeaderCountError {
    fn from(error: FormatError) -> Self {
        Self::Framing(error)
    }
}

enum CountPlace {
    Missing(usize),
    Record(Range<usize>),
}

#[cfg(test)]
mod tests;
