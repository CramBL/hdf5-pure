//! Version 1 object headers: section
//! `subsubsec_fmt4_dataobject_hdr_prefix_one`, version 4.0.

use std::collections::{BTreeSet, VecDeque};
use std::ops::Range;

use crate::bytes;
use crate::object_header::{Message, MessageFlags, MessageType};
use crate::superblock::v0;
use crate::widths::Widths;

/// The bytes of a version 1 object header's chunk zero, prefix included.
#[derive(Clone, Debug)]
pub struct Header {
    reference_count: u32,
    declared_data_size: Option<u32>,
    messages: Vec<Message>,
}

impl Header {
    pub fn new() -> Self {
        Self {
            reference_count: 1,
            declared_data_size: None,
            messages: Vec::new(),
        }
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

    /// Declares a chunk zero size other than the one the messages occupy, for
    /// a fixture about what a reader does when the two disagree.
    pub fn declared_data_size(mut self, size: u32) -> Self {
        self.declared_data_size = Some(size);
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let records = message_records(&self.messages);
        let message_count = u16::try_from(self.messages.len()).expect("a message count");
        let data_size = self
            .declared_data_size
            .unwrap_or_else(|| u32::try_from(records.len()).expect("a chunk zero size"));

        let mut header = Vec::new();
        header.push(VERSION);
        header.push(0);
        header.extend_from_slice(&message_count.to_le_bytes());
        header.extend_from_slice(&self.reference_count.to_le_bytes());
        header.extend_from_slice(&data_size.to_le_bytes());
        header.resize(PREFIX, 0);
        header.extend_from_slice(&records);
        header
    }
}

impl Default for Header {
    fn default() -> Self {
        Self::new()
    }
}

/// The bytes of `messages` as records laid back to back, which is what a
/// continuation chunk holds and what follows a header's prefix.
pub fn message_records(messages: &[Message]) -> Vec<u8> {
    messages.iter().flat_map(message_record).collect()
}

/// One record: type(2) + size(2) + flags(1) + reserved(3), then the body.
pub fn message_record(message: &Message) -> Vec<u8> {
    let mut record = Vec::new();
    record.extend_from_slice(&message.msg_type.0.to_le_bytes());
    let size = u16::try_from(message.data.len()).expect("a message body size");
    record.extend_from_slice(&size.to_le_bytes());
    record.push(message.flags.0);
    record.extend_from_slice(&[0; 3]);
    record.extend_from_slice(&message.data);
    record
}

/// Every chunk of the header at `at`: chunk zero first, then the chunk each
/// Object Header Continuation message names, in the order the walk meets them.
///
/// Panics on a chunk reached twice, since a fixture with a continuation cycle
/// would otherwise walk forever.
#[track_caller]
pub fn chunks(file: &[u8], at: usize, widths: Widths) -> Vec<Chunk> {
    let version = bytes::u8_at(file, at);
    assert_eq!(
        version, VERSION,
        "object header at {at:#x} has version {version}, not 1"
    );

    let mut walked = Vec::new();
    let mut seen = BTreeSet::new();
    // The prefix declares chunk zero's size where a continuation message
    // declares every other chunk's.
    let mut pending = VecDeque::from([Field {
        at: at + DATA_SIZE_FIELD,
        width: 4,
        start: at + PREFIX,
    }]);

    while let Some(length_field) = pending.pop_front() {
        let chunk = read_chunk(file, length_field);
        assert!(
            seen.insert(chunk.range.start),
            "the chunk at {:#x} is reached twice",
            chunk.range.start
        );
        pending.extend(
            chunk
                .records
                .iter()
                .filter(|record| record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION)
                .map(|record| {
                    let start = bytes::uint_at(file, record.body.start, widths.offset);
                    Field {
                        at: record.body.start + widths.offset,
                        width: widths.length,
                        start: usize::try_from(start).expect("a continuation address"),
                    }
                }),
        );
        walked.push(chunk);
    }
    walked
}

/// Every chunk of the root group's header, in a file whose version 0 or 1 superblock begins it
/// and whose base address is zero, so that every address is a file offset.
#[track_caller]
pub fn root_group_chunks(file: &[u8]) -> Vec<Chunk> {
    let superblock = v0::Fields::read(file, 0);
    assert_eq!(
        superblock.base_address, 0,
        "the superblock's base address is not zero, so its addresses are not file offsets"
    );
    let at = usize::try_from(superblock.root_header_address).expect("a root header address");
    chunks(file, at, superblock.widths)
}

/// One chunk of a version 1 object header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Chunk {
    /// The bytes the chunk's records occupy, prefix excluded for chunk zero.
    pub range: Range<usize>,
    /// Where the file declares how long this chunk is: the header prefix for
    /// chunk zero, and the continuation message pointing at it for any other.
    pub length_field: Field,
    pub records: Vec<Record>,
}

/// One record physically stored in a chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Record {
    /// Where the record begins, which is where its type field sits.
    pub at: usize,
    pub msg_type: MessageType,
    pub flags: MessageFlags,
    /// The bytes the record's body occupies, its eight-byte prefix excluded.
    pub body: Range<usize>,
}

impl Record {
    /// Where the record states its body's size.
    pub fn size_at(&self) -> usize {
        self.at + 2
    }

    pub fn set_type(&self, file: &mut [u8], msg_type: MessageType) {
        bytes::set_u16_at(file, self.at, msg_type.0);
    }

    pub fn set_body_size(&self, file: &mut [u8], size: u16) {
        bytes::set_u16_at(file, self.size_at(), size);
    }

    pub fn set_flags(&self, file: &mut [u8], flags: MessageFlags) {
        bytes::set_u8_at(file, self.at + 4, flags.0);
    }
}

/// Where a file states an offset or a length, and how wide that statement is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Field {
    pub at: usize,
    pub width: usize,
    /// The offset the length is measured from.
    pub start: usize,
}

impl Field {
    #[track_caller]
    pub fn value(&self, file: &[u8]) -> u64 {
        bytes::uint_at(file, self.at, self.width)
    }

    #[track_caller]
    pub fn set(&self, file: &mut [u8], value: u64) {
        bytes::set_uint_at(file, self.at, self.width, value);
    }
}

#[track_caller]
fn read_chunk(file: &[u8], length_field: Field) -> Chunk {
    let length = length_field.value(file);
    let end = length_field.start + usize::try_from(length).expect("a chunk length");
    assert!(
        end <= file.len(),
        "the chunk at {:#x} ends at {end:#x}, past the file's {} bytes",
        length_field.start,
        file.len()
    );

    let mut records = Vec::new();
    let mut at = length_field.start;
    while at < end {
        assert!(
            at + RECORD_PREFIX <= end,
            "the chunk ending at {end:#x} has {} bytes left at {at:#x}, too few for a record",
            end - at
        );
        let body_size = usize::from(bytes::u16_at(file, at + 2));
        let body = at + RECORD_PREFIX..at + RECORD_PREFIX + body_size;
        assert!(
            body.end <= end,
            "the record at {at:#x} has a body ending at {:#x}, past its chunk's {end:#x}",
            body.end
        );
        records.push(Record {
            at,
            msg_type: MessageType(bytes::u16_at(file, at)),
            flags: MessageFlags(bytes::u8_at(file, at + 4)),
            body: body.clone(),
        });
        at = body.end;
    }

    Chunk {
        range: length_field.start..end,
        length_field,
        records,
    }
}

/// Where the header prefix declares chunk zero's size: version(1) +
/// reserved(1) + message count(2) + reference count(4).
const DATA_SIZE_FIELD: usize = 8;

/// The prefix a version 1 header pads to eight bytes before its first record.
pub const PREFIX: usize = 16;

/// type(2) + size(2) + flags(1) + reserved(3).
pub const RECORD_PREFIX: usize = 8;

const VERSION: u8 = 1;

#[cfg(test)]
mod tests {
    use crate::image::Image;
    use crate::object_header::{Message, MessageFlags, MessageType, v1};
    use crate::superblock::v0;
    use crate::widths::Widths;

    #[test]
    fn walks_a_header_back_into_the_messages_it_was_built_from() {
        let continuation = v1::message_records(&[Message::new(MessageType::DATATYPE, &[7; 8])]);
        let mut image = Image::starting_with(
            &v1::Header::new()
                .message(Message::new(MessageType::DATASPACE, &[1; 8]))
                .continuation(256, &continuation, Widths::EIGHT)
                .build(),
        );
        image.place(256, &continuation);
        let file = image.build();

        let chunks = v1::chunks(&file, 0, Widths::EIGHT);
        let types: Vec<_> = chunks
            .iter()
            .flat_map(|chunk| chunk.records.iter().map(|record| record.msg_type))
            .collect();
        assert_eq!(
            types,
            vec![
                MessageType::DATASPACE,
                MessageType::OBJECT_HEADER_CONTINUATION,
                MessageType::DATATYPE
            ]
        );
        // An eight-byte dataspace record and a sixteen-byte continuation one,
        // each behind an eight-byte prefix.
        assert_eq!(chunks[0].range, v1::PREFIX..v1::PREFIX + 16 + 24);
        assert_eq!(chunks[1].range, 256..256 + 16);
        assert_eq!(
            file[chunks[1].records[0].body.clone()],
            [7; 8],
            "the continuation's message body"
        );
    }

    #[test]
    fn edits_a_record_of_the_root_group_header_in_place() {
        let mut image = Image::starting_with(
            &v0::Superblock::new(Widths::EIGHT)
                .root_group(0, 512)
                .build(),
        );
        image.place(
            512,
            &v1::Header::new()
                .message(Message::new(MessageType::ATTRIBUTE, &[1; 8]))
                .message(Message::nil(8))
                .build(),
        );
        let mut file = image.build();

        // The first record loses its body and chunk zero shrinks to that record, which drops the
        // Nil message after it.
        let chunk_zero = v1::root_group_chunks(&file)[0].clone();
        let record = &chunk_zero.records[0];
        record.set_type(&mut file, MessageType::UNKNOWN);
        record.set_flags(&mut file, MessageFlags::FAIL_IF_UNKNOWN_ALWAYS);
        record.set_body_size(&mut file, 0);
        chunk_zero
            .length_field
            .set(&mut file, v1::RECORD_PREFIX as u64);

        let start = 512 + v1::PREFIX;
        assert_eq!(record.size_at(), start + 2);
        assert_eq!(
            v1::root_group_chunks(&file),
            vec![v1::Chunk {
                range: start..start + v1::RECORD_PREFIX,
                length_field: v1::Field {
                    at: 512 + 8,
                    width: 4,
                    start,
                },
                records: vec![v1::Record {
                    at: start,
                    msg_type: MessageType::UNKNOWN,
                    flags: MessageFlags::FAIL_IF_UNKNOWN_ALWAYS,
                    body: start + v1::RECORD_PREFIX..start + v1::RECORD_PREFIX,
                }],
            }]
        );
    }
}
