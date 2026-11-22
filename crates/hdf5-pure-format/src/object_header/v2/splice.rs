//! The replacement of one message record in the message region of a version 2 object header.

use alloc::borrow::ToOwned;
use alloc::vec::Vec;

use crate::error::FormatError;
use crate::error::OBJECT_HEADER_MESSAGE_MAX;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::v2::MessageRecord;
use crate::object_header::v2::MessageRecordLayout;

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
    type_byte: u8,
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
        while let Some((type_byte, record)) =
            layout.next_message_with_type_byte(region, msg_start)?
        {
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
                    type_byte,
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
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `body` is longer than
    /// [`OBJECT_HEADER_MESSAGE_MAX`], the largest size the 2-byte size field of a record holds.
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
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `body` is longer than
    /// [`OBJECT_HEADER_MESSAGE_MAX`], leaving `region` unchanged.
    pub fn replace_body_into(&self, region: &mut Vec<u8>, body: &[u8]) -> Result<(), FormatError> {
        self.replace_record_into(
            region,
            self.record.flags,
            self.record.creation_index.unwrap_or(0),
            body,
        )
    }

    /// Appends the source region with a new body for the selected record.
    ///
    /// Calls `write_body` with a zero-filled slice of `body_len` bytes for the replacement body.
    /// Preserves the record's type, flags, and creation index and copies the surrounding bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `body_len` exceeds
    /// [`OBJECT_HEADER_MESSAGE_MAX`] or [`FormatError::AllocationFailed`] if reserving space fails.
    /// Propagates an error from `write_body`. Preserves the contents and length of `region` on
    /// error. Reserving space may increase its capacity.
    ///
    /// # Examples
    ///
    /// Updates the next creation index for tracked compact attributes. The tracking flag is
    /// defined in "Version 2 Data Object Header Prefix" of the [format specification,
    /// version 4.0][spec].
    ///
    /// ```
    /// use hdf5_pure_format::{FormatError, MessageType, OffsetWidth};
    /// use hdf5_pure_format::__private::{AttributeInfoMessage, MessageRecordLayout, RecordSplice};
    ///
    /// const TRACK_CREATION_ORDER: u8 = 0x04;
    /// let layout = MessageRecordLayout::from_header_flags(TRACK_CREATION_ORDER);
    /// let mut info = AttributeInfoMessage::compact(OffsetWidth::Two, Some(3), false);
    /// let source = layout.record(MessageType::ATTRIBUTE_INFO, &info.to_bytes());
    /// let splice = RecordSplice::first(layout, &source, MessageType::ATTRIBUTE_INFO)?.unwrap();
    ///
    /// info.set_max_creation_index(Some(4));
    /// let mut output = Vec::new();
    /// splice.replace_body_with(&mut output, info.encoded_size(), |body| {
    ///     assert_eq!(info.write_into(body).unwrap(), body.len());
    ///     Ok(())
    /// })?;
    /// let record = layout.next_message(&output, 0)?.unwrap();
    /// let updated = AttributeInfoMessage::parse_with_offset_width(record.body, OffsetWidth::Two)?;
    /// assert_eq!(updated, info);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
    pub fn replace_body_with(
        &self,
        region: &mut Vec<u8>,
        body_len: usize,
        write_body: impl FnOnce(&mut [u8]) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        let original_len = region.len();
        let body = self.prepare_replacement_body(region, body_len)?;
        let result = write_body(body);
        self.finish_replacement(region, original_len, result.is_ok());
        result
    }

    /// Returns a copy of the region in which the selected record has `flags`, `creation_index`,
    /// and `body`.
    ///
    /// The record stores `creation_index` only where the layout tracks creation order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `body` is longer than
    /// [`OBJECT_HEADER_MESSAGE_MAX`], the largest size the 2-byte size field of a record holds.
    pub fn replace_record(
        &self,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
    ) -> Result<Vec<u8>, FormatError> {
        let replacement = self.replacement(body.len())?;
        let mut region = Vec::with_capacity(replacement.len);
        self.append_record(
            &mut region,
            flags,
            creation_index,
            body,
            replacement.body_len,
        );
        Ok(region)
    }

    /// Reserves the full replacement region and returns a zero-filled body slice.
    ///
    /// Appends the source bytes preceding the selected record and writes the replacement
    /// record's prefix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `body_len` exceeds
    /// [`OBJECT_HEADER_MESSAGE_MAX`] or [`FormatError::AllocationFailed`] if reserving space fails.
    fn prepare_replacement_body<'region>(
        &self,
        region: &'region mut Vec<u8>,
        body_len: usize,
    ) -> Result<&'region mut [u8], FormatError> {
        let replacement = self.replacement(body_len)?;
        region
            .try_reserve(replacement.len)
            .map_err(|source| FormatError::AllocationFailed {
                values: replacement.len,
                source,
            })?;

        self.append_prefix(
            region,
            self.record.flags,
            self.record.creation_index.unwrap_or(0),
            replacement.body_len,
        );
        let body_start = region.len();
        region.resize(body_start + body_len, 0);
        Ok(&mut region[body_start..])
    }

    /// Appends the source tail after a successful write or truncates `region` to `original_len`
    /// after an error.
    fn finish_replacement(&self, region: &mut Vec<u8>, original_len: usize, body_written: bool) {
        if body_written {
            region.extend_from_slice(self.tail);
        } else {
            region.truncate(original_len);
        }
    }

    fn replace_record_into(
        &self,
        region: &mut Vec<u8>,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
    ) -> Result<(), FormatError> {
        let replacement = self.replacement(body.len())?;
        region.reserve(replacement.len);
        self.append_record(region, flags, creation_index, body, replacement.body_len);
        Ok(())
    }

    fn append_record(
        &self,
        region: &mut Vec<u8>,
        flags: MessageFlags,
        creation_index: u16,
        body: &[u8],
        body_len: u16,
    ) {
        self.append_prefix(region, flags, creation_index, body_len);
        region.extend_from_slice(body);
        region.extend_from_slice(self.tail);
    }

    /// Appends the source bytes before the selected record and the replacement record prefix.
    fn append_prefix(
        &self,
        region: &mut Vec<u8>,
        flags: MessageFlags,
        creation_index: u16,
        body_len: u16,
    ) {
        region.extend_from_slice(self.head);
        self.layout
            .write_record_prefix(region, self.type_byte, flags, creation_index, body_len);
    }

    /// Checks that the body length fits the record's size field and computes the append length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `size` exceeds
    /// [`OBJECT_HEADER_MESSAGE_MAX`].
    fn replacement(&self, size: usize) -> Result<ReplacementExtent, FormatError> {
        const { assert!(OBJECT_HEADER_MESSAGE_MAX == u16::MAX as usize) };
        let body_len =
            u16::try_from(size).map_err(|_error| FormatError::ObjectHeaderMessageTooLarge {
                message_type: self.record.msg_type.to_u16(),
                size,
            })?;
        let len =
            self.head.len() + self.layout.prefix_len() + usize::from(body_len) + self.tail.len();
        Ok(ReplacementExtent { len, body_len })
    }
}

/// The region length to append and the body length for the record's size field.
struct ReplacementExtent {
    len: usize,
    body_len: u16,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::object_header::v2 as v2_bytes;

    use super::*;

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
            records_around(layout, layout.record(MessageType::DATA_LAYOUT, &[0x77; 6]))
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
                layout.record_with_creation_index(
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
        let mut region = layout.record(cut_type, &[0x11; 8]);
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
            layout.record_with_creation_index(
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

    fn records_around(layout: MessageRecordLayout, data_layout: Vec<u8>) -> Vec<u8> {
        [
            layout.record_with_creation_index(
                UNKNOWN_MESSAGE_TYPE,
                MessageFlags::SHARED,
                1,
                &[0xAB; 3],
            ),
            layout.record(MessageType::DATASPACE, &[0x11; 8]),
            data_layout,
            layout.record_with_creation_index(
                MessageType::ATTRIBUTE,
                MessageFlags::NONE,
                2,
                &[0xCD; 5],
            ),
            layout.record(MessageType::DATA_LAYOUT, &[0xEE; 4]),
            TRAILING_GAP.to_vec(),
        ]
        .concat()
    }

    fn first_data_layout_record(layout: MessageRecordLayout) -> Vec<u8> {
        layout.record_with_creation_index(
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
