//! Independent group metadata files for policy inspection and link reconstruction.

use crate::bytes;
use crate::image::Image;
use crate::link_message::HardLink;
use crate::link_message::SymbolicLink;
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

/// The link creation-order metadata of a repack fixture.
#[derive(Clone, Copy, Debug)]
pub enum LinkOrder {
    /// Tracks creation indexes and enables creation-order indexing.
    Indexed,
    /// Tracks creation indexes with indexing disabled.
    Tracked,
    /// Omits creation indexes and their counter.
    Untracked,
}

/// A malformed link or Link Info field in an otherwise reproducible fixture.
#[derive(Clone, Copy, Debug)]
pub enum LinkDamage {
    /// A next creation index below a surviving link's index.
    Counter,
    /// A next creation index above [`i64::MAX`].
    CounterRange,
    /// Two links with the same creation index.
    DuplicateCreation,
    /// An external link value with version one.
    ExternalVersion,
    /// Creation-order indexing with tracking disabled.
    IndexingWithoutTracking,
    /// Two Link Info messages in one group header.
    RepeatedInfo,
    /// An extra byte after a soft Link message.
    TrailingLink,
}

/// Builds a version 2 group fixture with malformed link metadata.
pub fn damaged_repack_link_file(damage: LinkDamage) -> Vec<u8> {
    repack_link_image(LinkOrder::Indexed, HeaderVersion::V2, Some(damage)).0
}

/// Two encoded names with the same lossy UTF-8 display spelling.
#[derive(Clone, Copy, Debug)]
pub enum NameCollision {
    /// Two distinct invalid UTF-8 names.
    RawAndRaw,
    /// A valid UTF-8 name and an invalid UTF-8 name.
    Utf8AndRaw,
}

/// Builds two child objects whose encoded names have the same lossy UTF-8 display spelling.
pub fn colliding_link_file(collision: NameCollision) -> Vec<u8> {
    let mut image = Image::starting_with(&Superblock::new(Widths::EIGHT).build());
    let first = image.append(&header(&[], HeaderVersion::V2, None));
    let second = image.append(&header(&[], HeaderVersion::V2, None));
    let second_name = match collision {
        NameCollision::RawAndRaw => b"x\xfe".as_slice(),
        NameCollision::Utf8AndRaw => "x�".as_bytes(),
    };
    let records = [
        Message::new(
            MessageType::LINK,
            &HardLink::from_bytes(b"x\xff", u64::try_from(first).unwrap()).build(Widths::EIGHT),
        ),
        Message::new(
            MessageType::LINK,
            &HardLink::from_bytes(second_name, u64::try_from(second).unwrap()).build(Widths::EIGHT),
        ),
    ];
    let root = image.append(&header(&records, HeaderVersion::V2, None));
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .root_header_address(u64::try_from(root).unwrap())
            .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
            .build(),
    );
    image.build()
}

/// Builds a compact group with hard, soft, and external links containing non-UTF-8 bytes.
///
/// Returns file bytes, the expected Link Info body, and sorted Link bodies with hard target
/// addresses zeroed. Tracked fixtures have creation indexes 2, 7, and 11, with next index 17.
pub fn repack_link_file(
    order: LinkOrder,
    version: HeaderVersion,
) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    repack_link_image(order, version, None)
}

fn repack_link_image(
    order: LinkOrder,
    version: HeaderVersion,
    damage: Option<LinkDamage>,
) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let mut image = Image::starting_with(&Superblock::new(Widths::EIGHT).build());
    let child = image.append(&header(&[], version, None));
    let mut info = vec![
        LINK_INFO_VERSION,
        match order {
            LinkOrder::Untracked => NO_FLAGS,
            LinkOrder::Tracked => TRACKS_ORDER,
            LinkOrder::Indexed => TRACKS_ORDER | INDEXES_ORDER,
        },
    ];
    if !matches!(order, LinkOrder::Untracked) {
        info.extend_from_slice(&17_u64.to_le_bytes());
    }
    bytes::push_undefined_address(&mut info, Widths::EIGHT.offset);
    bytes::push_undefined_address(&mut info, Widths::EIGHT.offset);
    if matches!(order, LinkOrder::Indexed) {
        bytes::push_undefined_address(&mut info, Widths::EIGHT.offset);
    }
    let hard = HardLink::from_bytes(b"raw_\xff", u64::try_from(child).unwrap());
    let soft = SymbolicLink::soft_bytes("soft", b"raw_\xff");
    let external = SymbolicLink::external_bytes("external", b"file_\xff.h5", b"/target_\xfe");
    let (hard, soft, external) = if matches!(order, LinkOrder::Untracked) {
        (hard, soft, external)
    } else {
        (
            hard.creation_order(2),
            soft.creation_order(7),
            external.creation_order(11),
        )
    };
    let mut bodies = vec![hard.build(Widths::EIGHT), soft.build(), external.build()];
    match damage {
        Some(LinkDamage::Counter) => info[2..10].copy_from_slice(&7_u64.to_le_bytes()),
        Some(LinkDamage::CounterRange) => info[2..10].copy_from_slice(&u64::MAX.to_le_bytes()),
        Some(LinkDamage::DuplicateCreation) => {
            bodies[0][2..10].copy_from_slice(&7_u64.to_le_bytes())
        }
        Some(LinkDamage::ExternalVersion) => {
            let version = bodies[2].len() - (b"file_\xff.h5".len() + b"/target_\xfe".len() + 3);
            bodies[2][version] = 0x10;
        }
        Some(LinkDamage::IndexingWithoutTracking) => {
            info[1] = INDEXES_ORDER;
            info.drain(2..10);
        }
        Some(LinkDamage::TrailingLink) => bodies[1].push(0),
        Some(LinkDamage::RepeatedInfo) | None => {}
    }
    let mut records = vec![
        Message::new(MessageType::LINK_INFO, &info),
        Message::new(
            MessageType::GROUP_INFO,
            &[GROUP_INFO_VERSION, STORES_PHASE_CHANGE, 12, 0, 8, 0],
        ),
    ];
    if matches!(damage, Some(LinkDamage::RepeatedInfo)) {
        records.push(Message::new(MessageType::LINK_INFO, &info));
    }
    records.extend(
        bodies
            .iter()
            .map(|body| Message::new(MessageType::LINK, body)),
    );
    let root = image.append(&messages_header(records, version));
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .root_header_address(u64::try_from(root).unwrap())
            .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
            .build(),
    );
    let end = bodies[0].len();
    bodies[0][end - Widths::EIGHT.offset..].fill(0);
    bodies.sort();
    (image.build(), info, bodies)
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
    messages_header(records, version)
}

fn messages_header(mut records: Vec<Message>, version: HeaderVersion) -> Vec<u8> {
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

// "The Group Info Message" in the format specification, version 4.0, defines version zero.
const GROUP_INFO_VERSION: u8 = 0;
// "The Group Info Message" in the format specification, version 4.0, defines bit zero for
// thresholds.
const STORES_PHASE_CHANGE: u8 = 0x01;
// "The Link Info Message" in the format specification, version 4.0, defines the tracking flag bit.
const TRACKS_ORDER: u8 = 0x01;
// "The Link Info Message" in the format specification, version 4.0, defines the creation-index
// flag bit.
const INDEXES_ORDER: u8 = 0x02;
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
