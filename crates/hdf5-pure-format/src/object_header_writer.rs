//! Object header writer for v2 format.

use alloc::vec::Vec;

use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::MessageRecordLayout;
use crate::object_header::ObjectHeaderPrefix;

/// Writer for v2 object headers with proper checksums.
pub struct ObjectHeaderWriter {
    messages: Vec<(MessageType, Vec<u8>, MessageFlags)>, // (type, data, flags)
}

impl ObjectHeaderWriter {
    /// Creates an object header writer with no messages.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Adds a message whose record sets no flag.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte message type field of a version 2 object
    /// header.
    pub fn add_message(&mut self, msg_type: MessageType, data: Vec<u8>) {
        self.add_message_with_flags(msg_type, data, MessageFlags::NONE);
    }

    /// Adds a message whose record stores `flags`.
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
        assert!(
            u8::try_from(msg_type.to_u16()).is_ok(),
            "message type {:#06x} does not fit the 1-byte type field of a version 2 object header",
            msg_type.to_u16()
        );
        self.messages.push((msg_type, data, flags));
    }

    /// Serializes the version 2 object header: the prefix, the messages, and the checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if any message is
    /// larger than [`OBJECT_HEADER_MESSAGE_MAX`]. The per-message size field is
    /// 2 bytes wide, so a larger message could only be written by truncating
    /// its own length, which shifts every message that follows it.
    /// Callers that can identify the offending object (the whole-file writer identifies
    /// the attribute) check first and report a more specific error. This is the
    /// backstop for every other message the writer emits.
    pub fn serialize(&self) -> Result<Vec<u8>, FormatError> {
        for (msg_type, data, _) in &self.messages {
            if data.len() > OBJECT_HEADER_MESSAGE_MAX {
                return Err(FormatError::ObjectHeaderMessageTooLarge {
                    message_type: msg_type.to_u16(),
                    size: data.len(),
                });
            }
        }

        let layout = MessageRecordLayout::PLAIN;
        let messages_len = self
            .messages
            .iter()
            .map(|(_, data, _)| layout.prefix_len() + data.len())
            .sum();
        Ok(
            ObjectHeaderPrefix::PLAIN.encode_header_with(messages_len, |buf| {
                for (msg_type, data, msg_flags) in &self.messages {
                    layout.write_record(buf, *msg_type, *msg_flags, 0, data);
                }
            }),
        )
    }
}

impl Default for ObjectHeaderWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_mode::AccessMode;
    use crate::object_header::ObjectHeader;

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

    #[test]
    fn message_at_the_size_field_limit_still_serializes() {
        let mut writer = ObjectHeaderWriter::new();
        writer.add_message(
            MessageType::ATTRIBUTE,
            vec![0xAA; OBJECT_HEADER_MESSAGE_MAX],
        );
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].data.len(), OBJECT_HEADER_MESSAGE_MAX);
    }

    #[test]
    fn message_past_the_size_field_limit_is_refused() {
        let mut writer = ObjectHeaderWriter::new();
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
