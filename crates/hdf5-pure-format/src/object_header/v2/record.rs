//! Composition of version 2 object header records from typed or raw bodies.
//!
//! The record prefix is defined in "Version 2 Data Object Header Prefix" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two

use alloc::vec::Vec;
use core::ops::Range;

use crate::attribute_info::AttributeInfoMessage;
use crate::bytes::UnexpectedEof;
use crate::codec::Encode;
use crate::error::FormatError;
use crate::filter_pipeline::PreparedFilterPipeline;
use crate::group_info::GroupInfo;
use crate::link_info::LinkInfoMessage;
use crate::link_message_write::PreparedLink;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::v2::MessageRecordLayout;

/// An inline body whose type determines the object header message type.
///
/// This trait is sealed. A caller can pass a supported body by value or by reference to
/// [`MessageRecordLayout::prepare_record`].
#[expect(
    private_bounds,
    reason = "the private encoding contract seals the workspace-facing message identity"
)]
pub trait MessageBody: Encode<Error: Into<FormatError>> {
    /// The message type to write in the record prefix.
    const MESSAGE_TYPE: MessageType;
}

impl<T: MessageBody> MessageBody for &T {
    const MESSAGE_TYPE: MessageType = T::MESSAGE_TYPE;
}

impl MessageBody for LinkInfoMessage {
    const MESSAGE_TYPE: MessageType = MessageType::LINK_INFO;
}

impl MessageBody for PreparedLink<'_> {
    const MESSAGE_TYPE: MessageType = MessageType::LINK;
}

impl MessageBody for GroupInfo {
    const MESSAGE_TYPE: MessageType = MessageType::GROUP_INFO;
}

impl MessageBody for PreparedFilterPipeline<'_> {
    const MESSAGE_TYPE: MessageType = MessageType::FILTER_PIPELINE;
}

impl MessageBody for AttributeInfoMessage {
    const MESSAGE_TYPE: MessageType = MessageType::ATTRIBUTE_INFO;
}

impl MessageRecordLayout {
    /// Prepares an inline record whose type comes from its body.
    ///
    /// The writer retains `flags` and includes `creation_index` when the layout tracks creation
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if the body exceeds [`u16::MAX`]
    /// bytes, or [`FormatError::SharedFlagForInlineMessage`] if [`MessageFlags::SHARED`] is set.
    /// Checks the body length before the shared flag.
    ///
    /// # Examples
    ///
    /// Writes a Group Info record into a slice with room for one extra byte.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{GroupInfo, MessageFlags, MessageRecordLayout};
    /// use hdf5_pure_format::FormatError;
    ///
    /// let record = MessageRecordLayout::PLAIN.prepare_record(
    ///     MessageFlags::CONSTANT,
    ///     0,
    ///     GroupInfo::DEFAULT,
    /// )?;
    /// let mut output = [0xa5; 7];
    /// assert_eq!(record.write_into(&mut output)?, 6);
    /// assert_eq!(output, [0x0a, 2, 0, 0x01, 0, 0, 0xa5]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn prepare_record<B: MessageBody>(
        self,
        flags: MessageFlags,
        creation_index: u16,
        body: B,
    ) -> Result<PreparedMessageRecord<B>, FormatError> {
        let message_type = const { RecordPrefix::type_byte(B::MESSAGE_TYPE) };
        let prefix = RecordPrefix::new(
            self,
            message_type,
            flags,
            creation_index,
            body.encoded_len(),
        )?;
        prefix.require_inline()?;
        Ok(PreparedMessageRecord { prefix, body })
    }

    /// Prepares a record that borrows its already encoded body bytes.
    ///
    /// The writer retains the supplied message type, flags, and body, including unknown types
    /// and shared message references. It includes the creation index when the layout tracks
    /// creation order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTypeTooLarge`] if the message type exceeds
    /// [`u8::MAX`], or [`FormatError::ObjectHeaderMessageTooLarge`] if the body exceeds
    /// [`u16::MAX`] bytes. Checks the type before the body length.
    ///
    /// # Examples
    ///
    /// Appends an existing record with an unknown message type.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{MessageFlags, MessageRecordLayout};
    /// use hdf5_pure_format::{FormatError, MessageType};
    ///
    /// let body = [0xaa, 0xbb];
    /// let record = MessageRecordLayout::PLAIN.prepare_raw_record(
    ///     MessageType::from_u16(0xff),
    ///     MessageFlags::MARK_IF_UNKNOWN | MessageFlags::WAS_UNKNOWN,
    ///     0,
    ///     &body,
    /// )?;
    /// let mut output = vec![0xa5];
    /// record.append_to(&mut output)?;
    /// assert_eq!(output, [0xa5, 0xff, 2, 0, 0x30, 0xaa, 0xbb]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn prepare_raw_record(
        self,
        message_type: MessageType,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
    ) -> Result<PreparedMessageRecord<&[u8]>, FormatError> {
        let type_code = message_type.to_u16();
        let type_byte = u8::try_from(type_code)
            .map_err(|_error| FormatError::ObjectHeaderMessageTypeTooLarge(type_code))?;
        let prefix = RecordPrefix::new(self, type_byte, flags, creation_index, body.len())?;
        Ok(PreparedMessageRecord { prefix, body })
    }
}

/// A version 2 object header record with a checked prefix and body length.
///
/// Constructed by [`MessageRecordLayout::prepare_record`] or
/// [`MessageRecordLayout::prepare_raw_record`]. The record retains the selected prefix layout
/// for sizing and writing.
#[derive(Debug)]
pub struct PreparedMessageRecord<B> {
    prefix: RecordPrefix,
    body: B,
}

impl<B> PreparedMessageRecord<B> {
    /// Returns the exact byte count of the selected record prefix and body.
    pub fn encoded_size(&self) -> usize {
        self.prefix.encoded_size()
    }
}

#[expect(
    private_bounds,
    reason = "the record dispatches through the private encoding contract of its admitted body"
)]
impl<B: Encode<Error: Into<FormatError>>> PreparedMessageRecord<B> {
    /// Encodes the complete record into an owned buffer.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving space fails. Propagates encoding
    /// errors from the body.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let mut bytes = self.prefix.allocate_output()?;
        self.write_record(&mut bytes)?;
        Ok(bytes)
    }

    /// Writes the record at the start of `output` and returns the number of bytes written.
    ///
    /// Leaves bytes after the record unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is shorter than [`Self::encoded_size`].
    /// The error reports the encoded length as `expected` and `output.len()` as `available`.
    /// Propagates encoding errors from the body. Leaves the destination unchanged on error.
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        self.encode_into(output)
    }

    /// Appends the record prefix and body to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving space fails. Propagates encoding
    /// errors from the body. Leaves the previous bytes and length unchanged on error.
    pub fn append_to(&self, output: &mut Vec<u8>) -> Result<(), FormatError> {
        let start = self.prefix.reserve_append(output)?;
        let result = self.write_record(&mut output[start..]);
        finish_append(output, start, result)
    }

    /// Writes the body before the prefix.
    ///
    /// # Errors
    ///
    /// Propagates encoding errors from the body, leaving `output` unchanged on error.
    ///
    /// # Panics
    ///
    /// Panics if `output` is shorter than the record prefix.
    fn write_record(&self, output: &mut [u8]) -> Result<(), FormatError> {
        let (prefix, body) = output.split_at_mut(self.prefix.layout.prefix_len());
        self.body.encode_into(body).map_err(Into::into)?;
        self.prefix.write_into(prefix);
        Ok(())
    }
}

impl PreparedMessageRecord<PreparedLink<'_>> {
    /// Returns the hard-link address range relative to the start of the record.
    ///
    /// The address offset includes the retained record prefix length. Returns [`None`] for a
    /// symbolic or user-defined target.
    pub fn hard_link_address_range(&self) -> Option<Range<usize>> {
        let prefix = self.prefix.layout.prefix_len();
        self.body
            .hard_link_address_range()
            .map(|field| prefix + field.start..prefix + field.end)
    }
}

impl<B: Encode<Error: Into<FormatError>>> Encode for PreparedMessageRecord<B> {
    type Error = FormatError;

    fn encoded_len(&self) -> usize {
        self.encoded_size()
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let len = self.encoded_len();
        self.write_record(destination(output, len)?)?;
        Ok(len)
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_prefix_two`, version 4.0, defines the one-byte type,
// two-byte size, and optional two-byte creation index.
#[derive(Debug)]
pub(super) struct RecordPrefix {
    message_type: u8,
    body_len: u16,
    flags: MessageFlags,
    creation_index: u16,
    layout: MessageRecordLayout,
}

impl RecordPrefix {
    /// Returns the one-byte type code stored in a version 2 record prefix.
    ///
    /// # Panics
    ///
    /// Panics if the message type exceeds [`u8::MAX`].
    pub(super) const fn type_byte(message_type: MessageType) -> u8 {
        let [low, high] = message_type.to_u16().to_le_bytes();
        assert!(high == 0);
        low
    }

    pub(super) const fn from_fields(
        layout: MessageRecordLayout,
        message_type: u8,
        flags: MessageFlags,
        creation_index: u16,
        body_len: u16,
    ) -> Self {
        Self {
            message_type,
            body_len,
            flags,
            creation_index,
            layout,
        }
    }

    fn new(
        layout: MessageRecordLayout,
        message_type: u8,
        flags: MessageFlags,
        creation_index: u16,
        body_len: usize,
    ) -> Result<Self, FormatError> {
        let body_len =
            u16::try_from(body_len).map_err(|_error| FormatError::ObjectHeaderMessageTooLarge {
                message_type: u16::from(message_type),
                size: body_len,
            })?;
        Ok(Self::from_fields(
            layout,
            message_type,
            flags,
            creation_index,
            body_len,
        ))
    }

    /// Appends the prefix for a body that the caller writes separately.
    pub(super) fn append(&self, output: &mut Vec<u8>) {
        let start = output.len();
        output.resize(start + self.layout.prefix_len(), 0);
        self.write_into(&mut output[start..]);
    }

    fn require_inline(&self) -> Result<(), FormatError> {
        if self.flags.is_shared() {
            return Err(FormatError::SharedFlagForInlineMessage {
                message_type: u16::from(self.message_type),
                flags: self.flags.get(),
            });
        }
        Ok(())
    }

    fn encoded_size(&self) -> usize {
        const { assert!(usize::BITS >= 32) };
        self.layout.prefix_len() + usize::from(self.body_len)
    }

    /// Returns a zeroed buffer the length of the complete record.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving the complete record fails.
    fn allocate_output(&self) -> Result<Vec<u8>, FormatError> {
        let len = self.encoded_size();
        let mut output = Vec::new();
        output
            .try_reserve_exact(len)
            .map_err(|source| FormatError::AllocationFailed {
                values: len,
                source,
            })?;
        output.resize(len, 0);
        Ok(output)
    }

    /// Appends zeroed space for the complete record and returns its starting offset in `output`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving space fails, leaving the buffer's
    /// bytes and length unchanged.
    fn reserve_append(&self, output: &mut Vec<u8>) -> Result<usize, FormatError> {
        let len = self.encoded_size();
        output
            .try_reserve(len)
            .map_err(|source| FormatError::AllocationFailed {
                values: len,
                source,
            })?;
        let start = output.len();
        output.resize(start + len, 0);
        Ok(start)
    }

    fn write_into(&self, output: &mut [u8]) {
        let [size_low, size_high] = self.body_len.to_le_bytes();
        let fields = [self.message_type, size_low, size_high, self.flags.get()];
        let (prefix, creation_index) = output.split_at_mut(fields.len());
        prefix.copy_from_slice(&fields);
        if self.layout.tracks_creation_order() {
            creation_index.copy_from_slice(&self.creation_index.to_le_bytes());
        }
    }
}

impl Encode for [u8] {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        self.len()
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        destination(output, self.len())?.copy_from_slice(self);
        Ok(self.len())
    }
}

fn destination(output: &mut [u8], len: usize) -> Result<&mut [u8], UnexpectedEof> {
    let available = output.len();
    output
        .get_mut(..len)
        .ok_or_else(|| UnexpectedEof::new(len, available))
}

fn finish_append(
    output: &mut Vec<u8>,
    start: usize,
    result: Result<(), FormatError>,
) -> Result<(), FormatError> {
    if result.is_err() {
        output.truncate(start);
    }
    result
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::cell::Cell;

    use super::*;

    #[test]
    fn a_short_destination_keeps_the_encoder_from_writing() {
        let seen_len = Cell::new(None);
        let record = failing_record(&seen_len);
        let mut output = [0x5a; 7];
        assert_eq!(
            record.write_into(&mut output),
            Err(FormatError::UnexpectedEof {
                expected: 8,
                available: 7
            })
        );
        assert_eq!(seen_len.get(), None);
        assert_eq!(output, [0x5a; 7]);
    }

    #[test]
    fn a_body_error_leaves_the_complete_destination_unchanged() {
        let seen_len = Cell::new(None);
        let record = failing_record(&seen_len);
        let mut output = [0x5a; 13];
        assert_eq!(record.write_into(&mut output[1..]), Err(INJECTED_ERROR));
        assert_eq!(seen_len.get(), Some(2));
        assert_eq!(output, [0x5a; 13]);
    }

    #[test]
    fn a_body_error_restores_the_original_append_length_and_bytes() {
        let seen_len = Cell::new(None);
        let record = failing_record(&seen_len);
        let mut output = vec![0x5a, 0x4b, 0x3c];
        assert_eq!(record.append_to(&mut output), Err(INJECTED_ERROR));
        assert_eq!(seen_len.get(), Some(2));
        assert_eq!(output, [0x5a, 0x4b, 0x3c]);
    }

    fn failing_record(seen_len: &Cell<Option<usize>>) -> PreparedMessageRecord<FailingBody<'_>> {
        let layout = MessageRecordLayout::from_header_flags(TRACKED);
        layout
            .prepare_record(MessageFlags::CONSTANT, 0x1234, FailingBody(seen_len))
            .unwrap()
    }

    struct FailingBody<'a>(&'a Cell<Option<usize>>);

    impl MessageBody for FailingBody<'_> {
        const MESSAGE_TYPE: MessageType = MessageType::GROUP_INFO;
    }

    impl Encode for FailingBody<'_> {
        type Error = FormatError;

        fn encoded_len(&self) -> usize {
            2
        }

        fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
            self.0.set(Some(output.len()));
            Err(INJECTED_ERROR)
        }
    }

    const TRACKED: u8 = test_util::object_header::v2::HeaderFlags::TRACKS_CREATION_ORDER.0;
    const INJECTED_ERROR: FormatError = FormatError::InvalidGroupInfoVersion(u8::MAX);
}
