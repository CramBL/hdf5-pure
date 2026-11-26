//! Version 2 object-header serialization.

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::group_info::GroupInfo;
use crate::link::PreparedLink;
use crate::link_info::LinkInfoMessage;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::{
    MessageBody, ObjectHeaderPrefix, PreparedMessageRecord, PreparedRecordRef,
};
use crate::width::OffsetWidth;

/// A writer for checksummed version 2 object headers.
///
/// Appends message records to a single buffer. Writes the supplied prefix properties and
/// record flags. Includes each record's creation index when the layout tracks creation order.
/// These fields are defined in "Version 2 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
#[derive(Debug)]
pub struct ObjectHeaderWriter {
    prefix: ObjectHeaderPrefix,
    buffer: Vec<u8>,
    messages_start: usize,
    records: RecordSet,
}

impl ObjectHeaderWriter {
    /// Creates an object header writer with no messages.
    ///
    /// Uses [`ObjectHeaderPrefix::PLAIN`].
    pub fn new() -> Self {
        Self::with_prefix(ObjectHeaderPrefix::PLAIN)
    }

    /// Creates an empty writer that retains `prefix` for serialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::__private::{
    ///     AttributePhaseChange, ObjectHeaderPrefix, ObjectHeaderWriter,
    /// };
    ///
    /// let prefix = ObjectHeaderPrefix {
    ///     attribute_phase_change: Some(AttributePhaseChange {
    ///         max_compact: 32,
    ///         min_dense: 24,
    ///     }),
    ///     ..ObjectHeaderPrefix::PLAIN
    /// };
    /// let bytes = ObjectHeaderWriter::with_prefix(prefix).serialize()?;
    /// assert_eq!(ObjectHeaderPrefix::parse(&bytes)?.prefix, prefix);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn with_prefix(prefix: ObjectHeaderPrefix) -> Self {
        Self {
            prefix,
            buffer: Vec::new(),
            messages_start: prefix.prefix_len_for_messages(0),
            records: RecordSet::All,
        }
    }

    /// Starts a header with a batch of typed or raw prepared records.
    ///
    /// Skips [`None`] entries and writes the remaining records in slice order. Reserves space
    /// for the complete batch before writing any record.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a record and the header disagree on whether to write
    /// a creation index, [`FormatError::OffsetOverflow`] if the header extent exceeds
    /// [`usize::MAX`], or [`FormatError::AllocationFailed`] if reserving space fails.
    /// Propagates body encoding errors.
    ///
    /// # Examples
    ///
    /// Writes a typed Group Info record and an already encoded Object Reference Count record.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{
    ///     AccessMode, GroupInfo, MessageFlags, ObjectHeader, ObjectHeaderPrefix, ObjectHeaderWriter,
    ///     ObjectReferenceCountMessage,
    /// };
    /// use hdf5_pure_format::{FormatError, MessageType};
    ///
    /// let prefix = ObjectHeaderPrefix::PLAIN;
    /// let group_info = prefix.layout.prepare_record(MessageFlags::NONE, 0, GroupInfo::DEFAULT)?;
    /// let count_body = ObjectReferenceCountMessage::new(2).encode();
    /// let count = prefix.layout.prepare_raw_record(
    ///     MessageType::OBJECT_REFERENCE_COUNT,
    ///     MessageFlags::FORBID_SHARING,
    ///     0,
    ///     &count_body,
    /// )?;
    /// let records = [Some(group_info.as_record()), None, Some(count.as_record())];
    /// let bytes = ObjectHeaderWriter::with_prepared_records(prefix, &records)?.serialize()?;
    /// let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8)?;
    /// assert_eq!(header.messages.len(), 2);
    /// assert_eq!(header.messages[0].msg_type, MessageType::GROUP_INFO);
    /// assert_eq!(header.messages[0].data, [0, 0]);
    /// assert_eq!(header.messages[1].msg_type, MessageType::OBJECT_REFERENCE_COUNT);
    /// assert_eq!(header.messages[1].data, count_body);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn with_prepared_records(
        prefix: ObjectHeaderPrefix,
        records: &[Option<PreparedRecordRef<'_>>],
    ) -> Result<Self, FormatError> {
        let records_len = records.iter().flatten().try_fold(0, |len, record| {
            record.require_layout(prefix.layout)?;
            add_extent(len, record.encoded_size())
        })?;
        let mut writer = Self::with_prefix(prefix);
        writer.reserve_prefix_for(records_len)?;
        let start = writer.messages_start;
        writer.buffer.resize(start + records_len, 0);
        let mut remaining = &mut writer.buffer[start..];
        for record in records.iter().flatten() {
            let (destination, rest) = remaining.split_at_mut(record.encoded_size());
            record.write_into(destination)?;
            remaining = rest;
        }
        Ok(writer)
    }

    /// Starts a group header with Link Info, Group Info, and inline Link records in that order.
    ///
    /// A defined heap address in `link_info` selects dense storage and requires an empty
    /// `links` iterator. Inline hard links use the offset width retained by `link_info`.
    /// Uses the length of `links` to reserve space for the link record prefixes.
    /// Subsequent appends reject Link, Link Info, Group Info, and Symbol Table records.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if an inline Link body exceeds
    /// [`crate::error::OBJECT_HEADER_MESSAGE_MAX`], [`FormatError::InlineLinksWithDenseStorage`]
    /// for an inline link with dense storage, or [`FormatError::LinkOffsetWidthMismatch`] for
    /// a hard link with a different offset width. Checks the body length, storage mode, and
    /// offset width in that order for each link.
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the estimated extent exceeds [`usize::MAX`],
    /// or [`FormatError::AllocationFailed`] if reserving output space fails. Propagates errors
    /// from `links`.
    ///
    /// # Examples
    ///
    /// Writes a compact group header with one hard link.
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{
    ///     AccessMode, GroupInfo, LinkInfoMessage, ObjectHeader, ObjectHeaderPrefix,
    ///     ObjectHeaderWriter, PreparedLink,
    /// };
    /// use hdf5_pure_format::{CharacterSet, MessageType, OffsetWidth};
    ///
    /// let info = LinkInfoMessage::compact(OffsetWidth::Two, None, false);
    /// let link = PreparedLink::hard(
    ///     OffsetWidth::Two,
    ///     "alias",
    ///     StoredAddress::new(0x1234),
    ///     None,
    ///     CharacterSet::Ascii,
    /// )?;
    /// let writer = ObjectHeaderWriter::for_group(
    ///     ObjectHeaderPrefix::PLAIN,
    ///     &info,
    ///     &GroupInfo::DEFAULT,
    ///     core::iter::once(Ok(link)),
    /// )?;
    /// let bytes = writer.serialize()?;
    /// let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 2, 2)?;
    /// assert_eq!(header.messages[2].msg_type, MessageType::LINK);
    /// assert_eq!(header.messages[2].data, b"\x01\x00\x05alias\x34\x12");
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn for_group<'a>(
        prefix: ObjectHeaderPrefix,
        link_info: &LinkInfoMessage,
        group_info: &GroupInfo,
        links: impl ExactSizeIterator<Item = Result<PreparedLink<'a>, FormatError>>,
    ) -> Result<Self, FormatError> {
        let mut writer = Self::start_group(prefix, link_info, group_info, links.len())?;
        let offset_width = link_info.offset_width();
        let heap_address = link_info.fractal_heap_address();
        for link in links {
            writer.append_link(offset_width, heap_address, &link?)?;
        }
        writer.records = RecordSet::Supplemental;
        Ok(writer)
    }

    /// Appends an encoded body with empty flags and a creation index of zero when tracked.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::append_raw_record`].
    pub fn append_raw_message(
        &mut self,
        message_type: MessageType,
        body: &[u8],
    ) -> Result<(), FormatError> {
        self.append_raw_message_with_flags(message_type, body, MessageFlags::NONE)
    }

    /// Appends an encoded body with `flags` and a creation index of zero when tracked.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::append_raw_record`].
    pub fn append_raw_message_with_flags(
        &mut self,
        message_type: MessageType,
        body: &[u8],
        flags: MessageFlags,
    ) -> Result<(), FormatError> {
        self.append_raw_record(message_type, body, flags, 0)
    }

    /// Appends an encoded body with the supplied message type and flags.
    ///
    /// Writes `creation_index` when the layout tracks creation order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::QueuedGroupLinkMessage`] for a type rejected by [`Self::for_group`],
    /// or [`FormatError::AllocationFailed`] if reserving output space fails. Propagates errors
    /// from [raw record preparation].
    /// Leaves previous records unchanged on error.
    ///
    /// [raw record preparation]: crate::object_header::MessageRecordLayout::prepare_raw_record
    pub fn append_raw_record(
        &mut self,
        message_type: MessageType,
        body: &[u8],
        flags: MessageFlags,
        creation_index: u16,
    ) -> Result<(), FormatError> {
        self.check_message_type(message_type)?;
        let record =
            self.prefix
                .layout
                .prepare_raw_record(message_type, flags, creation_index, body)?;
        self.append_prepared(&record)
    }

    /// Encodes a typed inline body into the header with its associated message type.
    ///
    /// Writes `creation_index` when the layout tracks creation order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::QueuedGroupLinkMessage`] for a type rejected by [`Self::for_group`],
    /// or [`FormatError::AllocationFailed`] if reserving output space fails. Propagates errors
    /// from [typed record preparation] and the body encoder. Leaves previous records unchanged
    /// on error.
    ///
    /// [typed record preparation]: crate::object_header::MessageRecordLayout::prepare_record
    pub fn append_message<B: MessageBody>(
        &mut self,
        body: B,
        flags: MessageFlags,
        creation_index: u16,
    ) -> Result<(), FormatError> {
        self.check_message_type(B::MESSAGE_TYPE)?;
        let record = self
            .prefix
            .layout
            .prepare_record(flags, creation_index, body)?;
        self.append_prepared(&record)
    }

    /// Finishes the single-chunk header with its prefix and checksum.
    ///
    /// Uses the smallest chunk-size field that holds the length of the message records.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving final output space fails.
    pub fn serialize(mut self) -> Result<Vec<u8>, FormatError> {
        self.reserve_prefix_for(0)?;
        self.prefix.finish_header(self.buffer, self.messages_start)
    }

    /// Reserves space for Link Info, Group Info, and `link_count` Link record prefixes.
    ///
    /// Appends Link Info followed by Group Info.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the reserved extent exceeds [`usize::MAX`],
    /// or [`FormatError::AllocationFailed`] if reserving output space fails.
    /// Propagates errors preparing and appending the initial records.
    fn start_group(
        prefix: ObjectHeaderPrefix,
        link_info: &LinkInfoMessage,
        group_info: &GroupInfo,
        link_count: usize,
    ) -> Result<Self, FormatError> {
        let layout = prefix.layout;
        let link_info_record = layout.prepare_record(MessageFlags::NONE, 0, link_info)?;
        let group_info_record = layout.prepare_record(MessageFlags::NONE, 0, group_info)?;
        let link_prefixes = link_count.saturating_mul(layout.prefix_len());
        let mut writer = Self::with_prefix(prefix);
        writer.reserve_prefix_for(add_extent(
            link_info_record.encoded_size() + group_info_record.encoded_size(),
            link_prefixes,
        )?)?;
        writer.append_prepared(&link_info_record)?;
        writer.append_prepared(&group_info_record)?;
        Ok(writer)
    }

    fn append_link(
        &mut self,
        group_width: OffsetWidth,
        heap_address: Option<StoredAddress>,
        link: &PreparedLink<'_>,
    ) -> Result<(), FormatError> {
        let record = self
            .prefix
            .layout
            .prepare_record(MessageFlags::NONE, 0, link)?;
        if let Some(heap_address) = heap_address {
            return Err(FormatError::InlineLinksWithDenseStorage {
                heap_address: heap_address.get(),
            });
        }
        if let Some(link_width) = link
            .hard_offset_width()
            .filter(|&width| width != group_width)
        {
            return Err(FormatError::LinkOffsetWidthMismatch {
                group_width: group_width.get(),
                link_width: link_width.get(),
            });
        }
        self.append_prepared(&record)
    }

    fn append_prepared<B: Encode<Error: Into<FormatError>>>(
        &mut self,
        record: &PreparedMessageRecord<B>,
    ) -> Result<(), FormatError> {
        self.reserve_prefix_for(record.encoded_size())?;
        record.append_to(&mut self.buffer)
    }

    /// Reserves the prefix, `records_len` bytes, and the checksum for an empty header buffer.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the extent exceeds [`usize::MAX`], or
    /// [`FormatError::AllocationFailed`] if reserving output space fails.
    fn reserve_prefix_for(&mut self, records_len: usize) -> Result<(), FormatError> {
        if self.buffer.is_empty() {
            let prefix = self.prefix.prefix_len_for_messages(records_len);
            let capacity = add_extent(prefix + size_of::<u32>(), records_len)?;
            self.buffer.try_reserve_exact(capacity).map_err(|source| {
                FormatError::AllocationFailed {
                    values: capacity,
                    source,
                }
            })?;
            self.buffer.resize(prefix, 0);
            self.messages_start = prefix;
        }
        Ok(())
    }

    fn check_message_type(&self, message_type: MessageType) -> Result<(), FormatError> {
        if matches!(self.records, RecordSet::Supplemental)
            && matches!(
                message_type,
                MessageType::LINK
                    | MessageType::LINK_INFO
                    | MessageType::GROUP_INFO
                    | MessageType::SYMBOL_TABLE
            )
        {
            return Err(FormatError::QueuedGroupLinkMessage(message_type.to_u16()));
        }
        Ok(())
    }
}

impl Default for ObjectHeaderWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum RecordSet {
    All,
    Supplemental,
}

fn add_extent(offset: usize, length: usize) -> Result<usize, FormatError> {
    offset
        .checked_add(length)
        .ok_or_else(|| FormatError::OffsetOverflow {
            offset: offset.to_u64(),
            length: length.to_u64(),
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::object_header::Message;
    use test_util::object_header::MessageFlags as FixtureFlags;
    use test_util::object_header::MessageType as FixtureType;
    use test_util::object_header::v2 as v2_bytes;

    use super::*;
    use crate::access_mode::AccessMode;
    use crate::error::OBJECT_HEADER_MESSAGE_MAX;
    use crate::object_header::AttributePhaseChange;
    use crate::object_header::MessageRecordLayout;
    use crate::object_header::ObjectHeader;
    use crate::object_header::ObjectTimes;

    #[rstest]
    fn supplied_prefix_and_record_metadata_match_independent_bytes(
        #[values(
            v2_bytes::HeaderFlags::default(),
            v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER,
            v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER | v2_bytes::HeaderFlags::INDEXES_CREATION_ORDER
        )]
        flags: v2_bytes::HeaderFlags,
        #[values(false, true)] times: bool,
        #[values(false, true)] phase_change: bool,
    ) {
        let prefix = ObjectHeaderPrefix {
            layout: MessageRecordLayout::from_header_flags(flags.0),
            times: times.then_some(ObjectTimes {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4,
            }),
            attribute_phase_change: phase_change.then_some(AttributePhaseChange {
                max_compact: 32,
                min_dense: 24,
            }),
        };
        let mut expected = v2_bytes::Header::new().flags(flags);
        if times {
            expected = expected.timestamps(v2_bytes::Timestamps {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4,
            });
        }
        if phase_change {
            expected = expected.phase_change(32, 24);
        }
        let expected = expected
            .message(
                Message::new(FixtureType::ATTRIBUTE, &[7, 11])
                    .with_flags(FixtureFlags::CONSTANT)
                    .with_creation_order(17),
            )
            .message(Message::new(FixtureType::ATTRIBUTE, &[19]).with_creation_order(31))
            .build();
        let first = prefix
            .layout
            .prepare_raw_record(MessageType::ATTRIBUTE, MessageFlags::CONSTANT, 17, &[7, 11])
            .unwrap();
        let second = prefix
            .layout
            .prepare_raw_record(MessageType::ATTRIBUTE, MessageFlags::NONE, 31, &[19])
            .unwrap();
        let writer = ObjectHeaderWriter::with_prepared_records(
            prefix,
            &[Some(first.as_record()), None, Some(second.as_record())],
        )
        .unwrap();
        let bytes = writer.serialize().unwrap();
        assert_eq!(bytes, expected);
        let parsed = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(parsed.prefix, Some(prefix));
        assert_eq!(parsed.messages.len(), 2);
        assert_eq!(
            parsed.messages[0].creation_order,
            flags.tracks_creation_order().then_some(17)
        );
        assert_eq!(
            parsed.messages[1].creation_order,
            flags.tracks_creation_order().then_some(31)
        );
    }

    #[test]
    fn prepared_records_reject_different_creation_index_presence() {
        let tracked = ObjectHeaderPrefix {
            layout: MessageRecordLayout::from_header_flags(
                v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0,
            ),
            ..ObjectHeaderPrefix::PLAIN
        };
        let record = tracked
            .layout
            .prepare_raw_record(MessageType::ATTRIBUTE, MessageFlags::NONE, 1, &[7])
            .unwrap();
        let error = ObjectHeaderWriter::with_prepared_records(
            ObjectHeaderPrefix::PLAIN,
            &[Some(record.as_record())],
        )
        .unwrap_err();
        assert_eq!(
            error,
            FormatError::Internal(
                "prepared record and header disagree on creation index presence".into()
            )
        );
    }

    #[test]
    fn empty_header_roundtrip() {
        let writer = ObjectHeaderWriter::new();
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.messages.len(), 0);
    }

    #[test]
    fn two_messages_roundtrip() {
        let mut writer = ObjectHeaderWriter::new();
        writer
            .append_raw_message(MessageType::DATASPACE, &[1, 2, 3, 4])
            .unwrap();
        writer
            .append_raw_message(MessageType::DATATYPE, &[5, 6])
            .unwrap();
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::DATASPACE);
        assert_eq!(hdr.messages[0].data, vec![1, 2, 3, 4]);
        assert_eq!(hdr.messages[1].msg_type, MessageType::DATATYPE);
        assert_eq!(hdr.messages[1].data, vec![5, 6]);
    }

    #[test]
    fn a_serialized_record_stores_the_flags_the_message_was_added_with() {
        let mut writer = ObjectHeaderWriter::new();
        writer
            .append_raw_message_with_flags(
                MessageType::DATATYPE,
                &[5, 6],
                MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
            )
            .unwrap();
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(
            hdr.messages[0].flags,
            MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING
        );
    }

    #[test]
    fn a_message_added_without_flags_sets_no_flag_in_its_record() {
        let mut writer = ObjectHeaderWriter::new();
        writer
            .append_raw_message(MessageType::DATATYPE, &[5, 6])
            .unwrap();
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages[0].flags, MessageFlags::NONE);
    }

    #[test]
    fn large_header_uses_2byte_chunk_size() {
        let mut writer = ObjectHeaderWriter::new();
        // Add a message with >255 bytes of payload
        writer
            .append_raw_message(MessageType::DATATYPE, &[0xAA; 300])
            .unwrap();
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].data.len(), 300);
    }

    #[rstest]
    fn message_at_the_size_field_limit_serializes(
        #[values(
            ObjectHeaderPrefix::PLAIN,
            ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0))
        )]
        prefix: ObjectHeaderPrefix,
    ) {
        let mut writer = ObjectHeaderWriter::with_prefix(prefix);
        writer
            .append_raw_message(
                MessageType::ATTRIBUTE,
                &vec![0xAA; OBJECT_HEADER_MESSAGE_MAX],
            )
            .unwrap();
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].data.len(), OBJECT_HEADER_MESSAGE_MAX);
    }

    #[rstest]
    #[case::oversized_body(MessageType::ATTRIBUTE, FormatError::ObjectHeaderMessageTooLarge {
        message_type: MessageType::ATTRIBUTE.to_u16(),
        size: OBJECT_HEADER_MESSAGE_MAX + 1,
    })]
    #[case::wide_type(
        MessageType::from_u16(u16::MAX),
        FormatError::ObjectHeaderMessageTypeTooLarge(u16::MAX)
    )]
    fn writer_preserves_existing_records_after_a_failed_append(
        #[case] message_type: MessageType,
        #[case] error: FormatError,
    ) {
        let pad = [0xa5];
        let mut writer = ObjectHeaderWriter::new();
        writer.append_raw_message(MessageType::NIL, &pad).unwrap();
        assert_eq!(
            writer.append_raw_message(message_type, &vec![0xaa; OBJECT_HEADER_MESSAGE_MAX + 1]),
            Err(error)
        );
        writer.append_raw_message(MessageType::NIL, &pad).unwrap();
        let expected = v2_bytes::Header::new()
            .message(Message::new(FixtureType::NIL, &pad))
            .message(Message::new(FixtureType::NIL, &pad))
            .build();
        assert_eq!(writer.serialize().unwrap(), expected);
    }
}
