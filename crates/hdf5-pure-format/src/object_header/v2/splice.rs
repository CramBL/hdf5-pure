//! The replacement of one message record in the message region of a version 2 object header.

use alloc::borrow::ToOwned;
use alloc::vec::Vec;
use core::ops::Range;

use crate::codec::Encode;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::v2::MessageBody;
use crate::object_header::v2::MessageRecord;
use crate::object_header::v2::MessageRecordLayout;
use crate::object_header::v2::PreparedMessageRecord;

/// The first message record of one type in the message region of a version 2 object header, with
/// the bytes before and after it.
///
/// [`first`](Self::first) selects the record. [`replace_body`](Self::replace_body) and
/// [`replace_record`](Self::replace_record) return a copy of the region in which that record is
/// rewritten and every other byte is unchanged, a later record of the same type and a gap at the
/// end of the region included.
///
/// The message records are defined in "Version 2 Data Object Header Prefix" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordSplice<'a> {
    layout: MessageRecordLayout,
    head: &'a [u8],
    record: MessageRecord<'a>,
    tail: &'a [u8],
}

impl<'a> RecordSplice<'a> {
    /// Returns the splice of the first message record of `msg_type` in `region`, whose records
    /// have `layout`, or `None` where every record is of another type.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the body of a record it reads runs past the end
    /// of `region`.
    pub fn first(
        layout: MessageRecordLayout,
        region: &'a [u8],
        msg_type: MessageType,
    ) -> Result<Option<Self>, FormatError> {
        let mut msg_start = 0;
        while let Some(record) = layout.next_message(region, msg_start)? {
            if record.msg_type == msg_type {
                let (Some(head), Some(tail)) =
                    (region.get(..msg_start), region.get(record.body_range.end..))
                else {
                    return Err(FormatError::Internal(
                        "a message record read from the region lies outside it".to_owned(),
                    ));
                };
                return Ok(Some(Self {
                    layout,
                    head,
                    record,
                    tail,
                }));
            }
            msg_start = record.body_range.end;
        }
        Ok(None)
    }

    /// Returns the selected record, with its body range in the original region.
    pub fn record(&self) -> &MessageRecord<'a> {
        &self.record
    }

    /// Returns a copy of the region in which the selected record has `body` and keeps its flags
    /// and its creation index.
    ///
    /// # Errors
    ///
    /// Propagates message-size and allocation errors from [`Self::replace_record`].
    pub fn replace_body(&self, body: &[u8]) -> Result<Vec<u8>, FormatError> {
        self.replace_record(
            self.record.flags,
            self.record.creation_index.unwrap_or(0),
            body,
        )
    }

    /// Appends a copy of the source region with `body` in the selected record to `region`.
    ///
    /// Preserves the record's flags and creation index, unrelated source bytes, and the
    /// existing contents of `region`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `body` exceeds
    /// [`crate::error::OBJECT_HEADER_MESSAGE_MAX`], or [`FormatError::AllocationFailed`] if
    /// reserving space fails. Leaves the existing contents and length of `region` unchanged
    /// on error.
    pub fn replace_body_into(&self, region: &mut Vec<u8>, body: &[u8]) -> Result<(), FormatError> {
        let record = self.raw_record(
            self.record.flags,
            self.record.creation_index.unwrap_or(0),
            body,
        )?;
        self.append_replacement(region, &record)
    }

    /// Appends a copy of the source region with a replacement inline body.
    ///
    /// The replacement body must have the selected record's message type. Preserves the flags,
    /// creation index, and surrounding bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTypeMismatch`] if the selected record's type
    /// differs from the replacement body's type, or [`FormatError::AllocationFailed`] if
    /// reserving space fails. Propagates errors from [`MessageRecordLayout::prepare_record`] and
    /// the body encoder. Preserves the contents and length of `region` on error. Reserving space
    /// may increase its capacity.
    ///
    /// # Examples
    ///
    /// Updates the next creation index for tracked compact attributes. The tracking flag is
    /// defined in "Version 2 Data Object Header Prefix" of the [format specification,
    /// version 4.0][spec].
    ///
    /// ```
    /// use hdf5_pure_format::{FormatError, MessageType, OffsetWidth};
    /// use hdf5_pure_format::__private::{
    ///     AttributeInfoMessage, MessageFlags, MessageRecordLayout, RecordSplice,
    /// };
    ///
    /// const TRACK_CREATION_ORDER: u8 = 0x04;
    /// let layout = MessageRecordLayout::from_header_flags(TRACK_CREATION_ORDER);
    /// let mut info = AttributeInfoMessage::compact(OffsetWidth::Two, Some(3), false);
    /// let source = layout.prepare_record(MessageFlags::NONE, 0, &info)?.to_bytes()?;
    /// let splice = RecordSplice::first(layout, &source, MessageType::ATTRIBUTE_INFO)?.unwrap();
    ///
    /// info.set_max_creation_index(Some(4));
    /// let mut output = Vec::new();
    /// splice.replace_typed_body_into(&mut output, &info)?;
    /// let record = layout.next_message(&output, 0)?.unwrap();
    /// let updated = AttributeInfoMessage::parse_with_offset_width(record.body, OffsetWidth::Two)?;
    /// assert_eq!(updated, info);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
    pub fn replace_typed_body_into<B: MessageBody>(
        &self,
        region: &mut Vec<u8>,
        body: B,
    ) -> Result<(), FormatError> {
        let actual = self.record.msg_type;
        if actual != B::MESSAGE_TYPE {
            return Err(FormatError::ObjectHeaderMessageTypeMismatch {
                expected: B::MESSAGE_TYPE.to_u16(),
                actual: actual.to_u16(),
            });
        }
        let record = self.layout.prepare_record(
            self.record.flags,
            self.record.creation_index.unwrap_or(0),
            body,
        )?;
        self.append_replacement(region, &record)
    }

    /// Returns a copy of the region in which the selected record has `flags`, `creation_index`,
    /// and `body`.
    ///
    /// The record stores `creation_index` only where the layout tracks creation order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `body` exceeds
    /// [`crate::error::OBJECT_HEADER_MESSAGE_MAX`], or [`FormatError::AllocationFailed`] if
    /// reserving space fails.
    pub fn replace_record(
        &self,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
    ) -> Result<Vec<u8>, FormatError> {
        let record = self.raw_record(flags, creation_index, body)?;
        let mut region = Vec::new();
        let len = self.replacement_len(record.encoded_size());
        region
            .try_reserve_exact(len)
            .map_err(|source| FormatError::AllocationFailed {
                values: len,
                source,
            })?;
        self.append_replacement(&mut region, &record)?;
        Ok(region)
    }

    fn raw_record<'body>(
        &self,
        flags: MessageFlags,
        creation_index: u16,
        body: &'body [u8],
    ) -> Result<PreparedMessageRecord<&'body [u8]>, FormatError> {
        self.layout
            .prepare_raw_record(self.record.msg_type, flags, creation_index, body)
    }

    fn append_replacement<B: Encode<Error: Into<FormatError>>>(
        &self,
        region: &mut Vec<u8>,
        record: &PreparedMessageRecord<B>,
    ) -> Result<(), FormatError> {
        let original_len = region.len();
        let replacement = self.prepare_replacement(region, record.encoded_size())?;
        let result = record.write_into(&mut region[replacement]).map(|_| ());
        self.finish_replacement(region, original_len, result.is_ok());
        result
    }

    /// Reserves space for the entire splice and copies the bytes before the selected record.
    ///
    /// Returns the range of the zero-filled replacement record.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving the splice extent fails, leaving
    /// `region` unchanged.
    fn prepare_replacement(
        &self,
        region: &mut Vec<u8>,
        record_len: usize,
    ) -> Result<Range<usize>, FormatError> {
        let len = self.replacement_len(record_len);
        region
            .try_reserve(len)
            .map_err(|source| FormatError::AllocationFailed {
                values: len,
                source,
            })?;

        region.extend_from_slice(self.head);
        let start = region.len();
        let end = start + record_len;
        region.resize(end, 0);
        Ok(start..end)
    }

    fn finish_replacement(&self, region: &mut Vec<u8>, original_len: usize, record_written: bool) {
        if record_written {
            region.extend_from_slice(self.tail);
        } else {
            region.truncate(original_len);
        }
    }

    fn replacement_len(&self, record_len: usize) -> usize {
        self.head.len() + record_len + self.tail.len()
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use rstest::rstest;
    use test_util::object_header::v2 as v2_bytes;
    use test_util::object_header::{
        Message, MessageFlags as FixtureFlags, MessageType as FixtureType,
    };

    use super::*;
    use crate::error::OBJECT_HEADER_MESSAGE_MAX;

    #[rstest]
    fn a_replaced_record_takes_the_flags_and_the_creation_index_passed(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    ) {
        let region = records_around(layout, first_data_layout_record(layout));

        let region = RecordSplice::first(layout, &region, MessageType::DATA_LAYOUT)
            .unwrap()
            .expect("a Data Layout record")
            .replace_record(MessageFlags::NONE, 0, &[0x77; 6])
            .unwrap();

        assert_eq!(
            region,
            records_around(
                layout,
                fixture_record(
                    layout,
                    MessageType::DATA_LAYOUT,
                    MessageFlags::NONE,
                    0,
                    &[0x77; 6]
                )
            )
        );
    }

    #[rstest]
    fn a_replaced_body_keeps_the_flags_and_the_creation_index(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    ) {
        let region = records_around(layout, first_data_layout_record(layout));

        let region = RecordSplice::first(layout, &region, MessageType::DATA_LAYOUT)
            .unwrap()
            .expect("a Data Layout record")
            .replace_body(&[0x77; 6])
            .unwrap();

        assert_eq!(
            region,
            records_around(
                layout,
                fixture_record(
                    layout,
                    MessageType::DATA_LAYOUT,
                    MessageFlags::CONSTANT,
                    7,
                    &[0x77; 6],
                ),
            )
        );
    }

    #[rstest]
    fn the_first_record_of_the_type_is_selected(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    ) {
        let region = records_around(layout, first_data_layout_record(layout));
        // Its own prefix follows two records with 3- and 8-byte bodies.
        let body_start = 3 * layout.prefix_len() + 3 + 8;

        let splice = RecordSplice::first(layout, &region, MessageType::DATA_LAYOUT)
            .unwrap()
            .expect("a Data Layout record");

        assert_eq!(
            splice.record(),
            &MessageRecord {
                msg_type: MessageType::DATA_LAYOUT,
                flags: MessageFlags::CONSTANT,
                creation_index: layout.tracks_creation_order().then_some(7),
                body: &[0x55; 4],
                body_range: body_start..body_start + 4,
            }
        );
    }

    #[rstest]
    fn a_region_without_the_type_has_no_record_to_replace(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    ) {
        let region = records_around(layout, first_data_layout_record(layout));

        assert_eq!(
            RecordSplice::first(layout, &region, MessageType::FILTER_PIPELINE),
            Ok(None)
        );
    }

    #[rstest]
    #[case::a_record_of_another_type(MessageType::DATASPACE, MessageType::DATA_LAYOUT)]
    #[case::a_record_of_the_type_sought(MessageType::DATA_LAYOUT, MessageType::DATA_LAYOUT)]
    fn a_record_past_the_region_is_an_error(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
        #[case] cut_type: MessageType,
        #[case] msg_type: MessageType,
    ) {
        let mut region = fixture_record(layout, cut_type, MessageFlags::NONE, 0, &[0x11; 8]);
        region.pop();
        let len = region.len();

        assert_eq!(
            RecordSplice::first(layout, &region, msg_type),
            Err(FormatError::UnexpectedEof {
                expected: len + 1,
                available: len,
            })
        );
    }

    #[rstest]
    fn a_body_wider_than_the_size_field_is_an_error(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    ) {
        let region = records_around(layout, first_data_layout_record(layout));
        let splice = RecordSplice::first(layout, &region, MessageType::DATA_LAYOUT)
            .unwrap()
            .expect("a Data Layout record");

        assert_eq!(
            splice.replace_body(&vec![0; OBJECT_HEADER_MESSAGE_MAX + 1]),
            Err(FormatError::ObjectHeaderMessageTooLarge {
                message_type: MessageType::DATA_LAYOUT.to_u16(),
                size: OBJECT_HEADER_MESSAGE_MAX + 1,
            })
        );
    }

    #[rstest]
    #[case::empty_body(Vec::new())]
    #[case::short_body(vec![0x77; 6])]
    #[case::largest_body(vec![0x77; OBJECT_HEADER_MESSAGE_MAX])]
    fn a_replacement_can_be_appended_to_an_existing_buffer(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
        #[case] body: Vec<u8>,
    ) {
        let region = records_around(layout, first_data_layout_record(layout));
        let splice = RecordSplice::first(layout, &region, MessageType::DATA_LAYOUT)
            .unwrap()
            .unwrap();
        let mut out = vec![0x99; 5];

        splice.replace_body_into(&mut out, &body).unwrap();

        let mut expected = vec![0x99; 5];
        expected.extend_from_slice(&records_around(
            layout,
            fixture_record(
                layout,
                MessageType::DATA_LAYOUT,
                MessageFlags::CONSTANT,
                7,
                &body,
            ),
        ));
        assert_eq!(out, expected);
    }

    #[rstest]
    fn an_oversized_replacement_leaves_the_output_buffer_unchanged(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    ) {
        let region = records_around(layout, first_data_layout_record(layout));
        let splice = RecordSplice::first(layout, &region, MessageType::DATA_LAYOUT)
            .unwrap()
            .unwrap();
        let mut out = vec![0x99; 5];

        assert_eq!(
            splice.replace_body_into(&mut out, &vec![0; OBJECT_HEADER_MESSAGE_MAX + 1]),
            Err(FormatError::ObjectHeaderMessageTooLarge {
                message_type: MessageType::DATA_LAYOUT.to_u16(),
                size: OBJECT_HEADER_MESSAGE_MAX + 1,
            })
        );
        assert_eq!(out, vec![0x99; 5]);
    }

    #[rstest]
    #[case::body_error(9, Some(9), ENCODING_ERROR)]
    #[case::oversized_body(
        OBJECT_HEADER_MESSAGE_MAX + 1,
        None,
        FormatError::ObjectHeaderMessageTooLarge {
            message_type: MessageType::DATA_LAYOUT.to_u16(),
            size: OBJECT_HEADER_MESSAGE_MAX + 1,
        },
    )]
    fn splice_writer_preserves_the_destination_on_preparation_and_encoding_errors(
        #[case] body_len: usize,
        #[case] seen_len: Option<usize>,
        #[case] expected: FormatError,
    ) {
        let source = records_around(TRACKED, first_data_layout_record(TRACKED));
        let splice = RecordSplice::first(TRACKED, &source, MessageType::DATA_LAYOUT)
            .unwrap()
            .unwrap();
        let original = [0xa5, 0x5a, 0xaa];
        let mut output = original.to_vec();
        let destination_len = Cell::new(None);
        let body = FailingBody {
            body_len,
            destination_len: &destination_len,
        };

        assert_eq!(
            splice.replace_typed_body_into(&mut output, body),
            Err(expected)
        );
        assert_eq!(destination_len.get(), seen_len);
        assert_eq!(output, original);
    }

    struct FailingBody<'a> {
        body_len: usize,
        destination_len: &'a Cell<Option<usize>>,
    }

    impl MessageBody for FailingBody<'_> {
        const MESSAGE_TYPE: MessageType = MessageType::DATA_LAYOUT;
    }

    impl Encode for FailingBody<'_> {
        type Error = FormatError;

        fn encoded_len(&self) -> usize {
            self.body_len
        }

        fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
            self.destination_len.set(Some(output.len()));
            Err(ENCODING_ERROR)
        }
    }

    fn fixture_record(
        layout: MessageRecordLayout,
        message_type: MessageType,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
    ) -> Vec<u8> {
        let header_flags = if layout.tracks_creation_order() {
            v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER
        } else {
            v2_bytes::HeaderFlags::default()
        };
        v2_bytes::message_record(
            &Message::new(FixtureType(message_type.to_u16()), body)
                .with_flags(FixtureFlags(flags.get()))
                .with_creation_order(creation_index),
            header_flags,
        )
    }

    const ENCODING_ERROR: FormatError = FormatError::UnexpectedEof {
        expected: 12,
        available: 9,
    };

    fn records_around(layout: MessageRecordLayout, data_layout: Vec<u8>) -> Vec<u8> {
        [
            fixture_record(
                layout,
                UNKNOWN_MESSAGE_TYPE,
                MessageFlags::SHARED,
                1,
                &[0xAB; 3],
            ),
            fixture_record(
                layout,
                MessageType::DATASPACE,
                MessageFlags::NONE,
                0,
                &[0x11; 8],
            ),
            data_layout,
            fixture_record(
                layout,
                MessageType::ATTRIBUTE,
                MessageFlags::NONE,
                2,
                &[0xCD; 5],
            ),
            fixture_record(
                layout,
                MessageType::DATA_LAYOUT,
                MessageFlags::NONE,
                0,
                &[0xEE; 4],
            ),
            TRAILING_GAP.to_vec(),
        ]
        .concat()
    }

    fn first_data_layout_record(layout: MessageRecordLayout) -> Vec<u8> {
        fixture_record(
            layout,
            MessageType::DATA_LAYOUT,
            MessageFlags::CONSTANT,
            7,
            &[0x55; 4],
        )
    }

    const TRACKED: MessageRecordLayout =
        MessageRecordLayout::from_header_flags(v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0);

    const UNKNOWN_MESSAGE_TYPE: MessageType = MessageType::from_u16(0x00C8);

    const TRAILING_GAP: [u8; 3] = [0x01, 0x02, 0x03];
}
