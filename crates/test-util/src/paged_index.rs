//! Independent images with an extensible-array chunk index and persisted paged free space.

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
use crate::object_header::v2::Header;
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// Builds `keep` and `victim` datasets with separate metadata and raw-data pages.
///
/// `victim` contains [`i32`] values 0 through 8,191 in sixteen 512-element chunks,
/// with an unlimited maximum dimension. `keep` contains `[1, 2, 3, 4]` contiguously.
/// Both use little-endian storage. The file has a version 3 superblock, eight-byte
/// addresses and lengths, and persisted free-space managers with 4,096-byte pages.
pub fn paged_index() -> Vec<u8> {
    let mut image = Image::starting_with(&vec![0; EOF]);
    for chunk in 0..16 {
        let raw: Vec<_> = (chunk * 512..(chunk + 1) * 512)
            .flat_map(i32::to_le_bytes)
            .collect();
        image.place(RAW + chunk as usize * 2048, &raw);
    }
    image.place(
        KEEP_DATA,
        &(1..=4).flat_map(i32::to_le_bytes).collect::<Vec<_>>(),
    );
    image.place(EA_HEADER, &array_header());
    image.place(EA_INDEX, &array_index());
    image.place(VICTIM, &Dataset::Victim.build());
    image.place(KEEP, &Dataset::Keep.build());
    image.place(
        ROOT,
        &Header::new()
            .messages([
                Message::new(
                    MessageType::LINK_INFO,
                    &[LINK_INFO_VERSION, NO_FLAGS]
                        .into_iter()
                        .chain(u64::MAX.to_le_bytes())
                        .chain(u64::MAX.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
                Message::new(MessageType::GROUP_INFO, &[GROUP_INFO_VERSION, NO_FLAGS]),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("keep", KEEP as u64).build(Widths::EIGHT),
                ),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("victim", VICTIM as u64).build(Widths::EIGHT),
                ),
            ])
            .build(),
    );
    Manager::Metadata.append(&mut image);
    Manager::Raw.append(&mut image);
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

enum Dataset {
    Keep,
    Victim,
}

impl Dataset {
    fn build(self) -> Vec<u8> {
        let mut datatype =
            datatype::header(Class::FIXED_POINT, DATATYPE_VERSION, [SIGNED, 0, 0], 4);
        datatype.extend_from_slice(&0u16.to_le_bytes());
        datatype.extend_from_slice(&32u16.to_le_bytes());
        let (space, layout) = if matches!(self, Self::Victim) {
            let mut layout = vec![
                data_layout::VERSION_FOUR,
                data_layout::CHUNKED_CLASS,
                data_layout::NO_CHUNKED_FLAGS,
                2,
                DIMENSION_WIDTH,
            ];
            layout.extend_from_slice(&512u32.to_le_bytes());
            layout.extend_from_slice(&4u32.to_le_bytes());
            layout.extend_from_slice(&[
                EXTENSIBLE_ARRAY,
                MAX_ELEMENTS_BITS,
                INDEX_ELEMENTS,
                MIN_SECONDARY_POINTERS,
                MIN_DATA_ELEMENTS,
                PAGE_ELEMENTS_BITS,
            ]);
            layout.extend_from_slice(&(EA_HEADER as u64).to_le_bytes());
            (
                dataspace::v2(1, Flags::MAX_DIMS, Kind::SIMPLE, &[8192], Some(&[u64::MAX])),
                layout,
            )
        } else {
            (
                dataspace::v2(1, Flags::NONE, Kind::SIMPLE, &[4], None),
                data_layout::contiguous(KEEP_DATA as u64, 16),
            )
        };
        Header::new()
            .messages([
                Message::new(MessageType::DATASPACE, &space),
                Message::new(MessageType::DATATYPE, &datatype),
                Message::new(MessageType::DATA_LAYOUT, &layout),
            ])
            .build()
    }
}

// Section `subsec_fmt4_appendixc_extarr`, version 4.0, defines the header and statistics.
fn array_header() -> Vec<u8> {
    let mut header = EA_SIGNATURE.to_vec();
    header.extend_from_slice(&[
        VERSION_ZERO,
        CLIENT_UNFILTERED,
        8,
        MAX_ELEMENTS_BITS,
        INDEX_ELEMENTS,
        MIN_DATA_ELEMENTS,
        MIN_SECONDARY_POINTERS,
        PAGE_ELEMENTS_BITS,
    ]);
    for field in [0, 0, 0, 0, 16, 16, EA_INDEX as u64] {
        header.extend_from_slice(&field.to_le_bytes());
    }
    checksum::append(&mut header);
    header
}

// Section `subsec_fmt4_appendixc_extarr`, version 4.0, defines the index block.
fn array_index() -> Vec<u8> {
    let mut index = INDEX_SIGNATURE.to_vec();
    index.extend_from_slice(&[VERSION_ZERO, CLIENT_UNFILTERED]);
    index.extend_from_slice(&(EA_HEADER as u64).to_le_bytes());
    for chunk in 0..16u64 {
        index.extend_from_slice(&(RAW as u64 + chunk * 2048).to_le_bytes());
    }
    for _ in 0..DIRECT_POINTERS + SECONDARY_POINTERS {
        index.extend_from_slice(&u64::MAX.to_le_bytes());
    }
    checksum::append(&mut index);
    index
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_fsinfo`, version 4.0, defines the
// persistent version 1 message and its twelve manager slots.
fn space_info() -> Vec<u8> {
    let mut info = vec![SPACE_INFO_VERSION, PAGE_STRATEGY, PERSIST];
    info.extend_from_slice(&1u64.to_le_bytes());
    info.extend_from_slice(&(PAGE as u64).to_le_bytes());
    info.extend_from_slice(&0u16.to_le_bytes());
    info.extend_from_slice(&(EOF as u64).to_le_bytes());
    for address in [
        META_MANAGER as u64,
        u64::MAX,
        RAW_MANAGER as u64,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
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

// Section `subsec_fmt4_infra_freespaceindex`, version 4.0, defines both structures.
enum Manager {
    Metadata,
    Raw,
}

impl Manager {
    fn append(self, image: &mut Image) {
        let (header_address, list_address, sections): (_, _, &[(u64, u16)]) = match self {
            Self::Metadata => (META_MANAGER, META_LIST, &META_FREE),
            Self::Raw => (RAW_MANAGER, RAW_LIST, &RAW_FREE),
        };
        let mut list = free_space::SECTIONS_SIGNATURE.to_vec();
        list.push(VERSION_ZERO);
        list.extend_from_slice(&(header_address as u64).to_le_bytes());
        for &(address, size) in sections {
            list.push(1);
            list.extend_from_slice(&size.to_le_bytes());
            list.extend_from_slice(&address.to_le_bytes());
            list.push(SECTION_SMALL);
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

const META_FREE: [(u64, u16); 2] = [(2048, 2048), (4608, 3584)];
const RAW_FREE: [(u64, u16); 1] = [(KEEP_DATA as u64 + 16, 4080)];
const PAGE: usize = 4096;
const EXTENSION: usize = 48;
const ROOT: usize = 256;
const VICTIM: usize = 512;
const KEEP: usize = 768;
const EA_HEADER: usize = 1024;
const EA_INDEX: usize = 1152;
const META_MANAGER: usize = PAGE;
const META_LIST: usize = PAGE + MANAGER_LIST_DISTANCE;
const RAW_MANAGER: usize = PAGE + 256;
const RAW_LIST: usize = RAW_MANAGER + MANAGER_LIST_DISTANCE;
const MANAGER_LIST_DISTANCE: usize = 96;
const RAW: usize = 2 * PAGE;
const KEEP_DATA: usize = RAW + 8 * PAGE;
const EOF: usize = KEEP_DATA + PAGE;

// Sections `subsec_fmt4_boot_super`, `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`
// and `subsubsec_fmt4_dataobject_hdr_msg_layout`, version 4.0.
const SUPERBLOCK_VERSION: u8 = 3;
const DATATYPE_VERSION: u8 = 1;
const SIGNED: u8 = 0x08;
const DIMENSION_WIDTH: u8 = 4;
const EXTENSIBLE_ARRAY: u8 = 4;

// Section `subsec_fmt4_appendixc_extarr`, version 4.0.
const EA_SIGNATURE: &[u8; 4] = b"EAHD";
const INDEX_SIGNATURE: &[u8; 4] = b"EAIB";
const VERSION_ZERO: u8 = 0;
const CLIENT_UNFILTERED: u8 = 0;
const MAX_ELEMENTS_BITS: u8 = 32;
const INDEX_ELEMENTS: u8 = 16;
const MIN_DATA_ELEMENTS: u8 = 16;
const MIN_SECONDARY_POINTERS: u8 = 4;
const PAGE_ELEMENTS_BITS: u8 = 10;
// H5EA__hdr_init in H5EAhdr.c and H5EA__iblock_alloc in H5EAiblock.c,
// HDF5 2.2.0, derive six direct and twenty-five secondary pointers for this geometry.
const DIRECT_POINTERS: usize = 6;
const SECONDARY_POINTERS: usize = 25;

// Section `subsubsec_fmt4_dataobject_hdr_msg_fsinfo`, version 4.0.
const SPACE_INFO_VERSION: u8 = 1;
const PAGE_STRATEGY: u8 = 1;
const PERSIST: u8 = 1;

// Section `subsec_fmt4_infra_freespaceindex`, version 4.0.
const CLIENT_FILE: u8 = 1;
const ADDRESS_BITS: u16 = 64;

// H5MF_FSPACE_SECT_SMALL in H5MFpkg.h and H5MF_FSPACE_SECT_CLS_SMALL in
// H5MFsection.c, HDF5 2.2.0, define the paged class and three file-client classes.
const SECTION_SMALL: u8 = 1;
const SECTION_CLASSES: u16 = 3;

// Sections `subsubsec_fmt4_dataobject_hdr_msg_linkinfo` and
// `subsubsec_fmt4_dataobject_hdr_msg_groupinfo`, version 4.0.
const LINK_INFO_VERSION: u8 = 0;
const GROUP_INFO_VERSION: u8 = 0;
const NO_FLAGS: u8 = 0;
