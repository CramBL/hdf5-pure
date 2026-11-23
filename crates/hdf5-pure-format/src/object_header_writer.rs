//! Version 2 object-header serialization.

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::group_info::GroupInfo;
use crate::link_info::LinkInfoMessage;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::{
    MessageRecordLayout, OBJECT_HEADER_PREFIX_MAX_LEN, ObjectHeaderPrefix, PreparedLinkRecord,
};
use crate::width::OffsetWidth;

/// A writer for checksummed version 2 object headers.
///
/// Serializes supplied prefix properties and record flags. Encodes record creation indexes
/// when the prefix's layout tracks creation order. These fields are defined in "Version 2
/// Data Object Header Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub struct ObjectHeaderWriter {
    prefix: ObjectHeaderPrefix,
    /// Message bodies with record flags and creation indexes.
    messages: Vec<(MessageType, Vec<u8>, MessageFlags, u16)>,
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
            messages: Vec::new(),
        }
    }

    /// Adds a message with flags and creation index set to zero.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte message type field of a version 2 object
    /// header.
    pub fn add_message(&mut self, msg_type: MessageType, data: Vec<u8>) {
        self.add_message_with_flags(msg_type, data, MessageFlags::NONE);
    }

    /// Adds a message with `flags` and creation index zero.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte message type field of a version 2 object
    /// header.
    pub fn add_message_with_flags(
        &mut self,
        msg_type: MessageType,
        data: Vec<u8>,
        flags: MessageFlags,
    ) {
        self.add_message_record(msg_type, data, flags, 0);
    }

    /// Adds a message whose creation index is encoded in tracked layouts.
    ///
    /// Stores `flags` verbatim. [`serialize`](Self::serialize) encodes `creation_index` when
    /// [`ObjectHeaderPrefix::layout`] tracks creation order.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte message type field of a version 2 object
    /// header.
    pub fn add_message_record(
        &mut self,
        msg_type: MessageType,
        data: Vec<u8>,
        flags: MessageFlags,
        creation_index: u16,
    ) {
        assert!(
            u8::try_from(msg_type.to_u16()).is_ok(),
            "message type {:#06x} does not fit the 1-byte type field of a version 2 object header",
            msg_type.to_u16()
        );
        self.messages.push((msg_type, data, flags, creation_index));
    }

    /// Serializes the version 2 object header: the prefix, the messages, and the checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if a message body exceeds
    /// [`OBJECT_HEADER_MESSAGE_MAX`], or [`FormatError::OffsetOverflow`] if the combined extent
    /// exceeds [`usize::MAX`]. The optional record creation index is outside the message body.
    pub fn serialize(&self) -> Result<Vec<u8>, FormatError> {
        let messages_len = self.messages_len(RecordSet::All)?;
        add_extent(
            messages_len,
            OBJECT_HEADER_PREFIX_MAX_LEN + size_of::<u32>(),
        )?;
        Ok(self
            .prefix
            .encode_header_with(messages_len, |buf| self.append_messages(buf)))
    }

    /// Writes Link Info, Group Info, inline Link records, and supplemental records in order.
    ///
    /// A defined heap address in `link_info` selects dense storage and requires an empty
    /// `links` iterator. Inline hard links use the offset width retained by `link_info`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::QueuedGroupLinkMessage`] for a queued Link, Link Info, Group Info,
    /// or Symbol Table record, or [`FormatError::ObjectHeaderMessageTooLarge`] if a queued body
    /// exceeds [`OBJECT_HEADER_MESSAGE_MAX`]. Returns [`FormatError::InlineLinksWithDenseStorage`]
    /// for an inline link with dense storage or [`FormatError::LinkOffsetWidthMismatch`] for a
    /// hard link with a different offset width. Returns [`FormatError::OffsetOverflow`] if the
    /// combined extent exceeds [`usize::MAX`], or [`FormatError::AllocationFailed`] if reserving
    /// output space fails. Propagates errors from `links`.
    ///
    /// # Examples
    ///
    /// Writes a compact group header with one hard link.
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{
    ///     AccessMode, GroupInfo, LinkInfoMessage, ObjectHeader, ObjectHeaderWriter,
    ///     PreparedLink, PreparedLinkRecord,
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
    /// let bytes = ObjectHeaderWriter::new().serialize_group(
    ///     &info,
    ///     &GroupInfo::DEFAULT,
    ///     core::iter::once(PreparedLinkRecord::new(link)),
    /// )?;
    /// let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 2, 2)?;
    /// assert_eq!(header.messages[2].msg_type, MessageType::LINK);
    /// assert_eq!(header.messages[2].data, b"\x01\x00\x05alias\x34\x12");
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn serialize_group<'a>(
        &self,
        link_info: &LinkInfoMessage,
        group_info: &GroupInfo,
        links: impl ExactSizeIterator<Item = Result<PreparedLinkRecord<'a>, FormatError>>,
    ) -> Result<Vec<u8>, FormatError> {
        let supplemental_len = self.messages_len(RecordSet::Supplemental)?;
        let mut group = self.begin_group(link_info, group_info, links.len())?;
        for link in links {
            group.append_link(&link?)?;
        }
        self.finish_group(group, supplemental_len)
    }

    fn begin_group(
        &self,
        link_info: &LinkInfoMessage,
        group_info: &GroupInfo,
        link_count: usize,
    ) -> Result<GroupRecordWriter, FormatError> {
        let layout = self.prefix.layout;
        let group_info = group_info.serialize();
        let mut buffer = Vec::new();
        let prefix = self.prefix.reserved_prefix_len();
        let capacity = add_extent(
            prefix + 2 * layout.prefix_len() + link_info.encoded_size() + group_info.len(),
            link_count.saturating_mul(layout.prefix_len()),
        )?;
        buffer
            .try_reserve_exact(capacity)
            .map_err(|source| FormatError::AllocationFailed {
                values: capacity,
                source,
            })?;
        buffer.resize(prefix, 0);
        layout.append_link_info(&mut buffer, link_info);
        layout.write_record(
            &mut buffer,
            MessageType::GROUP_INFO,
            MessageFlags::NONE,
            0,
            &group_info,
        );
        Ok(GroupRecordWriter {
            layout,
            offset_width: link_info.offset_width(),
            heap_address: link_info.fractal_heap_address(),
            buffer,
        })
    }

    fn finish_group(
        &self,
        group: GroupRecordWriter,
        supplemental_len: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let mut buffer = group.buffer;
        buffer
            .try_reserve(supplemental_len)
            .map_err(|source| FormatError::AllocationFailed {
                values: supplemental_len,
                source,
            })?;
        self.append_messages(&mut buffer);
        self.prefix.finish_header(buffer)
    }

    fn messages_len(&self, records: RecordSet) -> Result<usize, FormatError> {
        let record_prefix = self.prefix.layout.prefix_len();
        let mut messages_len = 0;
        for (msg_type, data, _, _) in &self.messages {
            if matches!(records, RecordSet::Supplemental)
                && matches!(
                    *msg_type,
                    MessageType::LINK
                        | MessageType::LINK_INFO
                        | MessageType::GROUP_INFO
                        | MessageType::SYMBOL_TABLE
                )
            {
                return Err(FormatError::QueuedGroupLinkMessage(msg_type.to_u16()));
            }
            if data.len() > OBJECT_HEADER_MESSAGE_MAX {
                return Err(FormatError::ObjectHeaderMessageTooLarge {
                    message_type: msg_type.to_u16(),
                    size: data.len(),
                });
            }
            messages_len = add_extent(messages_len, record_prefix + data.len())?;
        }
        Ok(messages_len)
    }

    fn append_messages(&self, output: &mut Vec<u8>) {
        let layout = self.prefix.layout;
        for (msg_type, data, msg_flags, creation_index) in &self.messages {
            layout.write_record(output, *msg_type, *msg_flags, *creation_index, data);
        }
    }
}

impl Default for ObjectHeaderWriter {
    fn default() -> Self {
        Self::new()
    }
}

enum RecordSet {
    All,
    Supplemental,
}

struct GroupRecordWriter {
    layout: MessageRecordLayout,
    offset_width: OffsetWidth,
    heap_address: Option<StoredAddress>,
    buffer: Vec<u8>,
}

impl GroupRecordWriter {
    fn append_link(&mut self, link: &PreparedLinkRecord<'_>) -> Result<(), FormatError> {
        if let Some(heap_address) = self.heap_address {
            return Err(FormatError::InlineLinksWithDenseStorage {
                heap_address: heap_address.get(),
            });
        }
        if let Some(link_width) = link
            .hard_offset_width()
            .filter(|&width| width != self.offset_width)
        {
            return Err(FormatError::LinkOffsetWidthMismatch {
                group_width: self.offset_width.get(),
                link_width: link_width.get(),
            });
        }
        link.append_to(self.layout, &mut self.buffer)
    }
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
        let mut writer = ObjectHeaderWriter::with_prefix(prefix);
        writer.add_message_record(
            MessageType::ATTRIBUTE,
            vec![7, 11],
            MessageFlags::CONSTANT,
            17,
        );
        writer.add_message_record(MessageType::ATTRIBUTE, vec![19], MessageFlags::NONE, 31);
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
        writer.add_message(MessageType::DATASPACE, vec![1, 2, 3, 4]);
        writer.add_message(MessageType::DATATYPE, vec![5, 6]);
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
        writer.add_message_with_flags(
            MessageType::DATATYPE,
            vec![5, 6],
            MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
        );
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
        writer.add_message(MessageType::DATATYPE, vec![5, 6]);
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages[0].flags, MessageFlags::NONE);
    }

    #[test]
    fn large_header_uses_2byte_chunk_size() {
        let mut writer = ObjectHeaderWriter::new();
        // Add a message with >255 bytes of payload
        writer.add_message(MessageType::DATATYPE, vec![0xAA; 300]);
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
        writer.add_message(
            MessageType::ATTRIBUTE,
            vec![0xAA; OBJECT_HEADER_MESSAGE_MAX],
        );
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].data.len(), OBJECT_HEADER_MESSAGE_MAX);
    }

    #[rstest]
    fn message_past_the_size_field_limit_is_rejected(
        #[values(
            ObjectHeaderPrefix::PLAIN,
            ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(v2_bytes::HeaderFlags::TRACKS_CREATION_ORDER.0))
        )]
        prefix: ObjectHeaderPrefix,
    ) {
        let mut writer = ObjectHeaderWriter::with_prefix(prefix);
        writer.add_message(MessageType::DATASPACE, vec![0u8; 8]);
        writer.add_message(
            MessageType::ATTRIBUTE,
            vec![0xAA; OBJECT_HEADER_MESSAGE_MAX + 1],
        );
        assert_eq!(
            writer.serialize(),
            Err(FormatError::ObjectHeaderMessageTooLarge {
                message_type: MessageType::ATTRIBUTE.to_u16(),
                size: OBJECT_HEADER_MESSAGE_MAX + 1,
            })
        );
    }

    #[test]
    #[should_panic(
        expected = "message type 0x0117 does not fit the 1-byte type field of a version 2 object header"
    )]
    fn a_message_type_wider_than_the_type_field_panics() {
        ObjectHeaderWriter::new().add_message(MessageType::from_u16(0x0117), vec![0]);
    }
}
