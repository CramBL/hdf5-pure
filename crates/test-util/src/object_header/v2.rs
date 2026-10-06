//! Version 2 object headers: section
//! `subsubsec_fmt4_dataobject_hdr_prefix_two`, version 4.0.

use crate::bytes;
use crate::checksum;
use crate::object_header::Message;
use crate::object_header::MessageFlags;
use crate::object_header::MessageType;
use crate::widths::Widths;

/// The bytes of a version 2 object header's chunk zero, checksum included.
#[derive(Clone, Debug, Default)]
pub struct Header {
    flags: HeaderFlags,
    timestamps: Timestamps,
    messages: Vec<Message>,
}

impl Header {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the whole flags byte, including bits the builder does not otherwise
    /// write, so that a fixture can declare a chunk-size width or a flag a
    /// reader is meant to reject.
    pub fn flags(mut self, flags: HeaderFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Writes the four timestamps and sets the flag bit that says they are
    /// there.
    pub fn timestamps(mut self, timestamps: Timestamps) -> Self {
        self.flags = self.flags | HeaderFlags::STORES_TIMES;
        self.timestamps = timestamps;
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// An Object Header Continuation message naming `chunk` at `at`.
    pub fn continuation(self, at: usize, chunk: &[u8], widths: Widths) -> Self {
        self.message(Message::continuation(at as u64, chunk.len() as u64, widths))
    }

    pub fn build(&self) -> Vec<u8> {
        let records = message_records(&self.messages, self.flags);

        let mut header = Vec::new();
        header.extend_from_slice(SIGNATURE);
        header.push(VERSION);
        header.push(self.flags.0);
        if self.flags.stores_times() {
            let Timestamps {
                access,
                modification,
                change,
                birth,
            } = self.timestamps;
            for time in [access, modification, change, birth] {
                header.extend_from_slice(&time.to_le_bytes());
            }
        }
        if self.flags.stores_attribute_phase_change() {
            header.extend_from_slice(&DEFAULT_MAX_COMPACT.to_le_bytes());
            header.extend_from_slice(&DEFAULT_MIN_DENSE.to_le_bytes());
        }
        bytes::push_uint(
            &mut header,
            records.len() as u64,
            self.flags.chunk_size_width(),
        );
        header.extend_from_slice(&records);
        checksum::append(&mut header);
        header
    }
}

/// The bytes of a continuation chunk holding `messages`: its signature, the
/// records, and the checksum over both.
pub fn continuation_chunk(messages: &[Message], flags: HeaderFlags) -> Vec<u8> {
    let mut chunk = CONTINUATION_SIGNATURE.to_vec();
    chunk.extend_from_slice(&message_records(messages, flags));
    checksum::append(&mut chunk);
    chunk
}

/// The bytes of `messages` as records laid back to back.
///
/// `flags` decides only whether each record carries a creation order, which
/// belongs to the header the chunk is part of.
pub fn message_records(messages: &[Message], flags: HeaderFlags) -> Vec<u8> {
    messages
        .iter()
        .flat_map(|message| message_record(message, flags))
        .collect()
}

/// One record: type(1) + size(2) + flags(1) + creation order(2, optional),
/// then the body.
pub fn message_record(message: &Message, flags: HeaderFlags) -> Vec<u8> {
    let mut record = Vec::new();
    record
        .push(u8::try_from(message.msg_type.0).expect("a version 2 record stores a one-byte type"));
    let size = u16::try_from(message.data.len()).expect("a message body size");
    record.extend_from_slice(&size.to_le_bytes());
    record.push(message.flags.0);
    if flags.tracks_creation_order() {
        record.extend_from_slice(&message.creation_order.to_le_bytes());
    }
    record.extend_from_slice(&message.data);
    record
}

/// Replaces the body of the first `msg_type` message in chunk zero of the version 2 object header
/// at `header_at`, and recomputes the checksum of the chunk.
///
/// The record keeps its flags and its creation order. A Nil message fills the bytes a shorter body
/// leaves, so the chunk keeps its size.
///
/// # Panics
///
/// Panics if `header_at` is not the start of a version 2 object header, if chunk zero holds no
/// `msg_type` message, or if `body` is longer than the body it replaces or shorter by less than a
/// record prefix.
pub fn replace_message(file: &mut [u8], header_at: usize, msg_type: MessageType, body: &[u8]) {
    assert_eq!(bytes::slice_at(file, header_at, SIGNATURE.len()), SIGNATURE);
    let flags = HeaderFlags(bytes::u8_at(file, header_at + FLAGS_AT));
    let chunk_size_at = header_at + flags.prefix_len();
    let records_at = chunk_size_at + flags.chunk_size_width();
    let chunk_size = bytes::uint_at(file, chunk_size_at, flags.chunk_size_width());
    let records_end = records_at + usize::try_from(chunk_size).unwrap();

    let mut at = records_at;
    let old = loop {
        assert!(at < records_end, "chunk zero holds no {msg_type:?} message");
        let message = read_record(file, at, flags);
        if message.msg_type == msg_type {
            break message;
        }
        at += flags.record_prefix_len() + message.data.len();
    };
    let record_len = flags.record_prefix_len() + old.data.len();
    let mut records = message_record(
        &Message {
            data: body.to_vec(),
            ..old
        },
        flags,
    );
    if records.len() < record_len {
        let nil_len = (record_len - records.len())
            .checked_sub(flags.record_prefix_len())
            .expect("the space a shorter body leaves holds a Nil message");
        records.extend(message_record(&Message::nil(nil_len), flags));
    }
    assert_eq!(
        records.len(),
        record_len,
        "a body no longer than the one it replaces"
    );

    bytes::set_slice_at(file, at, &records);
    checksum::restamp(
        file,
        header_at,
        records_end + checksum::CHECKSUM - header_at,
    );
}

fn read_record(file: &[u8], at: usize, flags: HeaderFlags) -> Message {
    let size = bytes::u16_at(file, at + RECORD_SIZE_AT);
    Message {
        msg_type: MessageType(u16::from(bytes::u8_at(file, at))),
        flags: MessageFlags(bytes::u8_at(file, at + RECORD_FLAGS_AT)),
        creation_order: if flags.tracks_creation_order() {
            bytes::u16_at(file, at + RECORD_PREFIX_LEN)
        } else {
            0
        },
        data: bytes::slice_at(file, at + flags.record_prefix_len(), usize::from(size)).to_vec(),
    }
}

/// The flags byte of a version 2 object header.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeaderFlags(pub u8);

impl HeaderFlags {
    /// The width in bytes of the chunk size field, which bits 0 and 1 give as
    /// its base-two logarithm.
    pub fn chunk_size_width(self) -> usize {
        1 << (self.0 & Self::CHUNK_SIZE_WIDTH.0)
    }

    /// Returns the length of the header prefix before the chunk size field.
    fn prefix_len(self) -> usize {
        let times = if self.stores_times() {
            4 * size_of::<u32>()
        } else {
            0
        };
        let phase_change = if self.stores_attribute_phase_change() {
            2 * size_of::<u16>()
        } else {
            0
        };
        FLAGS_AT + 1 + times + phase_change
    }

    /// Returns the length of a message record before its body.
    fn record_prefix_len(self) -> usize {
        if self.tracks_creation_order() {
            RECORD_PREFIX_LEN + size_of::<u16>()
        } else {
            RECORD_PREFIX_LEN
        }
    }

    pub fn tracks_creation_order(self) -> bool {
        self.0 & Self::TRACKS_CREATION_ORDER.0 != 0
    }

    pub fn stores_attribute_phase_change(self) -> bool {
        self.0 & Self::STORES_ATTRIBUTE_PHASE_CHANGE.0 != 0
    }

    pub fn stores_times(self) -> bool {
        self.0 & Self::STORES_TIMES.0 != 0
    }

    /// Bits 0 and 1 encode the width of the chunk size field, as one byte
    /// shifted left by the value they hold.
    pub const CHUNK_SIZE_WIDTH: Self = Self(0x03);
    pub const TRACKS_CREATION_ORDER: Self = Self(0x04);
    pub const INDEXES_CREATION_ORDER: Self = Self(0x08);
    pub const STORES_ATTRIBUTE_PHASE_CHANGE: Self = Self(0x10);
    pub const STORES_TIMES: Self = Self(0x20);
}

impl core::ops::BitOr for HeaderFlags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// The four times a version 2 header stores, in the order it stores them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Timestamps {
    pub access: u32,
    pub modification: u32,
    pub change: u32,
    pub birth: u32,
}

pub const SIGNATURE: &[u8; 4] = b"OHDR";

pub const CONTINUATION_SIGNATURE: &[u8; 4] = b"OCHK";

const VERSION: u8 = 2;

// The signature, then the version byte.
const FLAGS_AT: usize = SIGNATURE.len() + 1;

// A record's type (1 byte), size (2) and flags (1), before its optional creation order.
const RECORD_SIZE_AT: usize = 1;
const RECORD_FLAGS_AT: usize = 3;
const RECORD_PREFIX_LEN: usize = 4;

/// How many attributes an object keeps in its header before they move to a
/// fractal heap, under a default creation property list.
const DEFAULT_MAX_COMPACT: u16 = 8;

/// How few bring them back, under the same defaults.
const DEFAULT_MIN_DENSE: u16 = 6;

#[cfg(test)]
mod tests {
    use crate::bytes;
    use crate::checksum;
    use crate::object_header::MessageFlags;
    use crate::object_header::v2::{self, HeaderFlags, Timestamps};
    use crate::object_header::{Message, MessageType};

    #[test]
    fn writes_the_prefix_the_flags_call_for() {
        let header = v2::Header::new()
            .timestamps(Timestamps {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4,
            })
            .flags(HeaderFlags::STORES_TIMES | HeaderFlags::TRACKS_CREATION_ORDER)
            .message(Message::new(MessageType::DATASPACE, &[9]).with_creation_order(5))
            .build();

        assert_eq!(&header[..4], v2::SIGNATURE);
        assert_eq!(bytes::u8_at(&header, 4), 2, "the version");
        assert_eq!(bytes::u32_at(&header, 6), 1, "the access time");
        assert_eq!(bytes::u32_at(&header, 18), 4, "the birth time");
        // One byte of chunk size, then type(1) + size(2) + flags(1) +
        // creation order(2) + one byte of body.
        assert_eq!(bytes::u8_at(&header, 22), 7, "the chunk size");
        assert_eq!(bytes::u16_at(&header, 27), 5, "the creation order");
        assert_eq!(header.len(), 23 + 7 + 4);
        assert_eq!(
            checksum::lookup3(&header[..header.len() - 4]).to_le_bytes(),
            header[header.len() - 4..]
        );
    }

    #[test]
    fn replacing_a_message_keeps_its_record_prefix_and_pads_the_rest_with_nil() {
        let header = |messages: Vec<Message>| {
            v2::Header::new()
                .flags(
                    HeaderFlags::STORES_TIMES
                        | HeaderFlags::TRACKS_CREATION_ORDER
                        | HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE,
                )
                .timestamps(Timestamps {
                    access: 1,
                    modification: 2,
                    change: 3,
                    birth: 4,
                })
                .messages(messages)
                .build()
        };
        let first = Message::new(MessageType::DATASPACE, &[9]).with_creation_order(1);
        let second = Message::new(MessageType::FILE_SPACE_INFO, &[7; 10])
            .with_flags(MessageFlags::CONSTANT)
            .with_creation_order(5);
        let mut file = header(vec![first.clone(), second.clone()]);

        v2::replace_message(&mut file, 0, MessageType::FILE_SPACE_INFO, &[3, 4]);

        assert_eq!(
            file,
            header(vec![
                first,
                Message {
                    data: vec![3, 4],
                    ..second
                },
                Message::nil(2),
            ])
        );
    }
}
