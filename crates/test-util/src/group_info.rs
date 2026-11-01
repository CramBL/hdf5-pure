//! Independent files with root and child groups for Group Info inspection.

use crate::bytes;
use crate::image::Image;
use crate::link_message::HardLink;
use crate::object_header::{Message, MessageType};
use crate::object_header::{v1, v2};
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// The object header encoding for both groups.
#[derive(Clone, Copy, Debug)]
pub enum HeaderVersion {
    /// Version 1 with eight-byte message alignment.
    V1,
    /// Version 2 with a checksummed header.
    V2,
}

/// The prefix before the superblock signature.
#[derive(Clone, Copy, Debug)]
pub enum Userblock {
    /// Places the signature at byte zero.
    Absent,
    /// Places the signature at byte 512.
    Bytes512,
}

impl Userblock {
    fn offset(self) -> usize {
        match self {
            Self::Absent => 0,
            Self::Bytes512 => USERBLOCK_LENGTH,
        }
    }
}

/// Builds a file whose root and `child` group contain `messages`.
///
/// The file uses eight-byte addresses and lengths. The root also contains a hard link to `child`.
pub fn file(messages: &[Message], version: HeaderVersion, userblock: Userblock) -> Vec<u8> {
    let userblock = userblock.offset();
    let superblock_size = Superblock::new(Widths::EIGHT).build().len();
    let mut image = Image::starting_with(&vec![0; userblock + superblock_size]);
    let child = image.append(&header(messages, version, None));
    let root = image.append(&header(
        messages,
        version,
        Some(u64::try_from(child - userblock).unwrap()),
    ));
    image.place(
        userblock,
        &Superblock::new(Widths::EIGHT)
            .version(SUPERBLOCK_VERSION)
            .base_address(u64::try_from(userblock).unwrap())
            .root_header_address(u64::try_from(root - userblock).unwrap())
            .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
            .build(),
    );
    image.build()
}

fn header(messages: &[Message], version: HeaderVersion, child: Option<u64>) -> Vec<u8> {
    let mut link_info = vec![LINK_INFO_VERSION, NO_FLAGS];
    bytes::push_undefined_address(&mut link_info, Widths::EIGHT.offset);
    bytes::push_undefined_address(&mut link_info, Widths::EIGHT.offset);
    let mut records = vec![Message::new(MessageType::LINK_INFO, &link_info)];
    records.extend_from_slice(messages);
    if let Some(child) = child {
        records.push(Message::new(
            MessageType::LINK,
            &HardLink::new("child", child).build(Widths::EIGHT),
        ));
    }
    match version {
        HeaderVersion::V1 => {
            for record in &mut records {
                record
                    .data
                    .resize(record.data.len().next_multiple_of(MESSAGE_ALIGNMENT), 0);
            }
            v1::Header::new().messages(records).build()
        }
        HeaderVersion::V2 => v2::Header::new().messages(records).build(),
    }
}

// "Format Signature and Superblock" in the format specification, version 4.0, defines
// superblock version three.
const SUPERBLOCK_VERSION: u8 = 3;
// "The Link Info Message" in the format specification, version 4.0, defines version zero.
const LINK_INFO_VERSION: u8 = 0;
// "The Link Info Message" in the format specification, version 4.0, defines optional fields
// by set flags.
const NO_FLAGS: u8 = 0;
// "Version 1 Data Object Header Prefix" in the format specification, version 4.0, aligns
// messages to eight bytes.
const MESSAGE_ALIGNMENT: usize = 8;
// "Format Signature and Superblock" in the format specification, version 4.0, permits the
// signature at byte 512.
const USERBLOCK_LENGTH: usize = 512;
