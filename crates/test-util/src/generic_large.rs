//! Independent images with a continued attribute header and generic-large free-space fragments.

use crate::attribute::Attribute;
use crate::checksum;
use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::dataspace::Kind;
use crate::datatype;
use crate::datatype::Class;
use crate::free_space;
use crate::image::Image;
use crate::link_message::HardLink;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v2;
use crate::object_header::v2::Header;
use crate::object_header::v2::HeaderFlags;
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// Builds `d` with a continued attribute header in a persisted 512-byte paged file.
///
/// The contiguous dataset contains 64 [`f64`] values equal to 1.0. Its `big_attr`
/// attribute contains 512 [`i64`] values equal to 7. Both use little-endian storage.
/// The header has a 268-byte initial chunk and a 4,158-byte continuation. The file
/// has a version 3 superblock, eight-byte addresses and lengths, and a sub-page
/// fragment in the generic-large free-space manager.
pub fn generic_large() -> Vec<u8> {
    let mut image = Image::starting_with(&vec![0; EOF]);
    image.place(
        DATA,
        &vec![1.0f64; 64]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let continuation = attribute_chunk();
    image.place(CONTINUATION, &continuation);
    let mut messages = vec![
        Message::new(
            MessageType::DATASPACE,
            &dataspace::v2(1, Flags::NONE, Kind::SIMPLE, &[64], None),
        ),
        Message::new(MessageType::DATATYPE, &datatype::f64_le()),
        Message::new(
            MessageType::DATA_LAYOUT,
            &data_layout::contiguous(DATA as u64, PAGE as u64),
        ),
        Message::continuation(
            CONTINUATION as u64,
            continuation.len() as u64,
            Widths::EIGHT,
        ),
    ];
    let used: usize = messages
        .iter()
        .map(|message| RECORD_PREFIX + message.data.len())
        .sum();
    messages.push(Message::nil(CHUNK_ZERO_RECORDS - used - RECORD_PREFIX));
    let header = Header::new().flags(HEADER_FLAGS).messages(messages).build();
    assert_eq!(header.len(), HEADER_LEN);
    image.place(HEADER, &header);
    let mut info = vec![LINK_INFO_VERSION, NO_FLAGS];
    info.extend_from_slice(&u64::MAX.to_le_bytes());
    info.extend_from_slice(&u64::MAX.to_le_bytes());
    image.place(
        ROOT,
        &Header::new()
            .messages([
                Message::new(MessageType::LINK_INFO, &info),
                Message::new(MessageType::GROUP_INFO, &[GROUP_INFO_VERSION, NO_FLAGS]),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("d", HEADER as u64).build(Widths::EIGHT),
                ),
            ])
            .build(),
    );
    Manager::Metadata.append(&mut image);
    Manager::Large.append(&mut image);
    image.place(
        EXTENSION,
        &Header::new()
            .message(Message::new(MessageType::FILE_SPACE_INFO, &space_info()))
            .build(),
    );
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .version(SUPERBLOCK_VERSION)
            .extension_address(EXTENSION as u64)
            .root_header_address(ROOT as u64)
            .eof_address(EOF as u64)
            .build(),
    );
    image.build()
}

fn attribute_chunk() -> Vec<u8> {
    let mut datatype = datatype::header(Class::FIXED_POINT, DATATYPE_VERSION, [SIGNED, 0, 0], 8);
    datatype.extend_from_slice(&0u16.to_le_bytes());
    datatype.extend_from_slice(&64u16.to_le_bytes());
    let space = dataspace::v2(1, Flags::NONE, Kind::SIMPLE, &[512], None);
    let raw: Vec<_> = vec![7i64; 512]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect();
    let attribute = Message::new(
        MessageType::ATTRIBUTE,
        &Attribute::new("big_attr", &datatype, &space, &raw)
            .character_set(ASCII)
            .build(),
    );
    let padding = CONTINUATION_LEN
        - v2::CONTINUATION_SIGNATURE.len()
        - checksum::CHECKSUM
        - 2 * RECORD_PREFIX
        - attribute.data.len();
    let chunk = v2::continuation_chunk(&[attribute, Message::nil(padding)], HEADER_FLAGS);
    assert_eq!(chunk.len(), CONTINUATION_LEN);
    chunk
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_fsinfo`, version 4.0.
fn space_info() -> Vec<u8> {
    let mut info = vec![SPACE_INFO_VERSION, PAGE_STRATEGY, PERSIST];
    info.extend_from_slice(&1u64.to_le_bytes());
    info.extend_from_slice(&(PAGE as u64).to_le_bytes());
    info.extend_from_slice(&0u16.to_le_bytes());
    info.extend_from_slice(&(MANAGERS as u64).to_le_bytes());
    for address in [
        MANAGERS as u64,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        LARGE_MANAGER as u64,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    ] {
        info.extend_from_slice(&address.to_le_bytes());
    }
    info
}

enum Manager {
    Large,
    Metadata,
}

impl Manager {
    // Section `subsec_fmt4_infra_freespaceindex`, version 4.0, defines both structures.
    fn append(self, image: &mut Image) {
        let (header_address, sections, class): (_, &[(u64, u16)], _) = match self {
            Self::Large => (LARGE_MANAGER, &[(5_694, 450)], SECTION_LARGE),
            Self::Metadata => (MANAGERS, &[(400, 112), (780, 244)], SECTION_SMALL),
        };
        let list_address = header_address + LIST_DISTANCE;
        let mut list = free_space::SECTIONS_SIGNATURE.to_vec();
        list.push(VERSION_ZERO);
        list.extend_from_slice(&(header_address as u64).to_le_bytes());
        for &(address, size) in sections {
            list.push(1);
            list.extend_from_slice(&size.to_le_bytes());
            list.extend_from_slice(&address.to_le_bytes());
            list.push(class);
        }
        checksum::append(&mut list);
        let mut header = free_space::SIGNATURE.to_vec();
        header.extend_from_slice(&[VERSION_ZERO, CLIENT_FILE]);
        for field in [
            sections.iter().map(|(_, size)| u64::from(*size)).sum(),
            sections.len() as u64,
            sections.len() as u64,
            0,
        ] {
            header.extend_from_slice(&field.to_le_bytes());
        }
        for field in [SECTION_CLASSES, 80, 120, ADDRESS_BITS] {
            header.extend_from_slice(&field.to_le_bytes());
        }
        for field in [
            PAGE as u64,
            list_address as u64,
            list.len() as u64,
            list.len() as u64,
        ] {
            header.extend_from_slice(&field.to_le_bytes());
        }
        checksum::append(&mut header);
        image.place(header_address, &header);
        image.place(list_address, &list);
    }
}

const PAGE: usize = 512;
const EXTENSION: usize = 48;
const ROOT: usize = 256;
const HEADER: usize = 512;
const HEADER_LEN: usize = 268;
const DATA: usize = 1_024;
const CONTINUATION: usize = 1_536;
const CONTINUATION_LEN: usize = 4_158;
const MANAGERS: usize = 6_144;
const LARGE_MANAGER: usize = MANAGERS + 256;
const LIST_DISTANCE: usize = 96;
const EOF: usize = MANAGERS + PAGE;
const CHUNK_ZERO_RECORDS: usize = 256;

// Section `subsubsec_fmt4_dataobject_hdr_prefix_two`, version 4.0, defines the
// two-byte chunk-size field and four-byte record prefix without creation-order tracking.
const HEADER_FLAGS: HeaderFlags = HeaderFlags(0x01);
const RECORD_PREFIX: usize = 4;

// Sections `subsec_fmt4_boot_super`, `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`
// and `subsubsec_fmt4_dataobject_hdr_msg_attribute`, version 4.0.
const SUPERBLOCK_VERSION: u8 = 3;
const DATATYPE_VERSION: u8 = 1;
const SIGNED: u8 = 0x08;
const ASCII: u8 = 0;

// Sections `subsubsec_fmt4_dataobject_hdr_msg_linkinfo` and
// `subsubsec_fmt4_dataobject_hdr_msg_groupinfo`, version 4.0.
const LINK_INFO_VERSION: u8 = 0;
const GROUP_INFO_VERSION: u8 = 0;
const NO_FLAGS: u8 = 0;

// Section `subsubsec_fmt4_dataobject_hdr_msg_fsinfo`, version 4.0.
const SPACE_INFO_VERSION: u8 = 1;
const PAGE_STRATEGY: u8 = 1;
const PERSIST: u8 = 1;

// Section `subsec_fmt4_infra_freespaceindex`, version 4.0.
const VERSION_ZERO: u8 = 0;
const CLIENT_FILE: u8 = 1;
const ADDRESS_BITS: u16 = 64;

// H5MF_FSPACE_SECT_SMALL/LARGE in H5MFpkg.h and the three H5MF_FSPACE_SECT_CLS
// descriptors in H5MFsection.c, HDF5 2.2.0, define the paged file-client classes.
const SECTION_SMALL: u8 = 1;
const SECTION_LARGE: u8 = 2;
const SECTION_CLASSES: u16 = 3;
