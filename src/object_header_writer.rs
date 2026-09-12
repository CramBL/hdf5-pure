//! Object header writer for v2 format.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::checksum::jenkins_lookup3;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;

/// Writer for v2 object headers with proper checksums.
pub struct ObjectHeaderWriter {
    messages: Vec<(MessageType, Vec<u8>, MessageFlags)>, // (type, data, flags)
}

impl ObjectHeaderWriter {
    /// Create a new empty object header writer.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Adds a message whose record sets no flag.
    pub fn add_message(&mut self, msg_type: MessageType, data: Vec<u8>) {
        self.messages.push((msg_type, data, MessageFlags::NONE));
    }

    /// Adds a message whose record stores `flags`.
    pub fn add_message_with_flags(
        &mut self,
        msg_type: MessageType,
        data: Vec<u8>,
        flags: MessageFlags,
    ) {
        self.messages.push((msg_type, data, flags));
    }

    /// Serialize the complete v2 object header (OHDR + messages + checksum).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if any message is
    /// larger than [`OBJECT_HEADER_MESSAGE_MAX`]. The per-message size field is
    /// 2 bytes wide, so a larger message could only be written by truncating
    /// its own length, which desynchronizes every message that follows it.
    /// Callers that can name the offending object (the whole-file writer names
    /// the attribute) check first and report a more specific error; this is the
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

        // Calculate total message bytes: each message has type(1) + size(2) + flags(1) + data
        let msg_bytes_total: usize = self
            .messages
            .iter()
            .map(|(_, data, _)| 4 + data.len())
            .sum();

        // Determine chunk size field width based on msg_bytes_total
        let (flags, chunk_size_width) = if msg_bytes_total <= 255 {
            (0x00u8, 1usize)
        } else if msg_bytes_total <= 65535 {
            (0x01u8, 2)
        } else {
            (0x02u8, 4)
        };

        let mut buf = Vec::new();

        // OHDR signature
        buf.extend_from_slice(b"OHDR");
        // version
        buf.push(2);
        // flags
        buf.push(flags);
        // chunk0 size
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk_size_width is the byte width chosen to hold msg_bytes_total, so each arm casts to a width that fits by construction"
        )]
        match chunk_size_width {
            1 => buf.push(msg_bytes_total as u8),
            2 => buf.extend_from_slice(&(msg_bytes_total as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(msg_bytes_total as u32).to_le_bytes()),
            _ => {}
        }

        // Messages
        for (msg_type, data, msg_flags) in &self.messages {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "message type id is a small enum value written into the 1-byte message-type field of the v2 object header"
            )]
            buf.push(msg_type.to_u16() as u8); // type (1 byte in v2)
            #[expect(
                clippy::cast_possible_truncation,
                reason = "message data length is bounded by OBJECT_HEADER_MESSAGE_MAX above, so it fits the 2-byte message-size field of the v2 object header"
            )]
            buf.extend_from_slice(&(data.len() as u16).to_le_bytes()); // size (2 bytes)
            buf.push(msg_flags.get()); // flags
            buf.extend_from_slice(data);
        }

        // Checksum
        let checksum = jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());

        Ok(buf)
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
    use crate::object_header::ObjectHeader;

    #[test]
    fn empty_header_roundtrip() {
        let writer = ObjectHeaderWriter::new();
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.messages.len(), 0);
    }

    #[test]
    fn two_messages_roundtrip() {
        let mut writer = ObjectHeaderWriter::new();
        writer.add_message(MessageType::Dataspace, vec![1, 2, 3, 4]);
        writer.add_message(MessageType::Datatype, vec![5, 6]);
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 2);
        assert_eq!(hdr.messages[0].msg_type, MessageType::Dataspace);
        assert_eq!(hdr.messages[0].data, vec![1, 2, 3, 4]);
        assert_eq!(hdr.messages[1].msg_type, MessageType::Datatype);
        assert_eq!(hdr.messages[1].data, vec![5, 6]);
    }

    #[test]
    fn a_serialized_record_stores_the_flags_the_message_was_added_with() {
        let mut writer = ObjectHeaderWriter::new();
        writer.add_message_with_flags(
            MessageType::Datatype,
            vec![5, 6],
            MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
        );
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(
            hdr.messages[0].flags,
            MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING
        );
    }

    #[test]
    fn a_message_added_without_flags_sets_no_flag_in_its_record() {
        let mut writer = ObjectHeaderWriter::new();
        writer.add_message(MessageType::Datatype, vec![5, 6]);
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages[0].flags, MessageFlags::NONE);
    }

    #[test]
    fn large_header_uses_2byte_chunk_size() {
        let mut writer = ObjectHeaderWriter::new();
        // Add a message with >255 bytes of payload
        writer.add_message(MessageType::Datatype, vec![0xAA; 300]);
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].data.len(), 300);
    }

    #[test]
    fn message_at_the_size_field_limit_still_serializes() {
        let mut writer = ObjectHeaderWriter::new();
        writer.add_message(
            MessageType::Attribute,
            vec![0xAA; OBJECT_HEADER_MESSAGE_MAX],
        );
        let bytes = writer.serialize().unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, 8, 8).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].data.len(), OBJECT_HEADER_MESSAGE_MAX);
    }

    #[test]
    fn message_past_the_size_field_limit_is_refused() {
        let mut writer = ObjectHeaderWriter::new();
        writer.add_message(MessageType::Dataspace, vec![0u8; 8]);
        writer.add_message(
            MessageType::Attribute,
            vec![0xAA; OBJECT_HEADER_MESSAGE_MAX + 1],
        );
        assert_eq!(
            writer.serialize(),
            Err(FormatError::ObjectHeaderMessageTooLarge {
                message_type: MessageType::Attribute.to_u16(),
                size: OBJECT_HEADER_MESSAGE_MAX + 1,
            })
        );
    }
}
