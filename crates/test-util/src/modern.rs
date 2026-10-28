//! Independent file images with compact or dense link-message groups.
//!
//! Each image has a version 3 superblock, version 2 object headers, eight-byte
//! addresses and lengths, and little-endian contiguous numeric datasets.

use crate::btree_v2;
use crate::bytes;
use crate::checksum;
use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::dataspace::Kind;
use crate::datatype;
use crate::datatype::Class;
use crate::fractal_heap;
use crate::image::Image;
use crate::link_message::HardLink;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v2;
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// A fixed hierarchy with compact or dense link storage.
#[derive(Clone, Copy, Debug)]
pub enum Fixture {
    /// The compact group `sensors` with temperature and humidity datasets.
    Groups,
    /// A dense root group with twenty datasets.
    ManyLinks,
}

impl Fixture {
    /// Encodes the hierarchy as an independent file image.
    pub fn build(self) -> Vec<u8> {
        match self {
            Self::Groups => v2_groups(),
            Self::ManyLinks => v2_many_links(),
        }
    }
}

/// Builds compact groups containing `sensors/temperature` and `sensors/humidity`.
///
/// The temperature dataset stores `[22.5, 23.1, 21.8]` as [`f64`] values, and
/// the humidity dataset stores `[45, 50, 55]` as [`i32`] values.
pub fn v2_groups() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; SUPERBLOCK_SIZE]);
    let temperature = dataset(&mut image, Values::F64(&[22.5, 23.1, 21.8]));
    let humidity = dataset(&mut image, Values::I32(&[45, 50, 55]));
    let sensors = compact_group(
        &mut image,
        &[("temperature", temperature), ("humidity", humidity)],
    );
    let root = compact_group(&mut image, &[("sensors", sensors)]);
    finish(image, root)
}

/// Builds a dense root group with datasets `dataset_000` through `dataset_019`.
///
/// Each dataset contains its index as one [`f64`] value. The links are stored in
/// a fractal heap with a version 2 B-tree name index.
pub fn v2_many_links() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; SUPERBLOCK_SIZE]);
    let links: Vec<_> = (0..20)
        .map(|index| {
            (
                format!("dataset_{index:03}"),
                dataset(&mut image, Values::F64(&[f64::from(index)])),
            )
        })
        .collect();
    let mut objects = Vec::new();
    let mut records: Vec<_> = links
        .iter()
        .map(|(name, address)| {
            let link = HardLink::new(name, *address).build(Widths::EIGHT);
            let mut record = checksum::lookup3(name.as_bytes()).to_le_bytes().to_vec();
            record.push(MANAGED_ID);
            // Twenty fixed-size links occupy less than the 1024-byte block.
            record.extend_from_slice(
                &u32::try_from(DIRECT_BLOCK_PREFIX + objects.len())
                    .unwrap()
                    .to_le_bytes(),
            );
            // Each ASCII name has eleven bytes, so the encoded link fits u16.
            record.extend_from_slice(&u16::try_from(link.len()).unwrap().to_le_bytes());
            objects.extend_from_slice(&link);
            record
        })
        .collect();
    records.sort_by_key(|record| bytes::u32_at(record, 0));
    let heap = image.as_bytes().len() as u64;
    let placeholder = heap_header(0, objects.len());
    image.append(&placeholder);
    let mut block = fractal_heap::DIRECT_BLOCK_SIGNATURE.to_vec();
    block.push(HEAP_VERSION);
    block.extend_from_slice(&heap.to_le_bytes());
    block.extend_from_slice(&0u32.to_le_bytes());
    block.extend_from_slice(&objects);
    block.resize(DIRECT_BLOCK_SIZE, 0);
    let block = image.append(&block) as u64;
    image.place(heap as usize, &heap_header(block, objects.len()));
    let mut leaf = btree_v2::leaf(LINK_NAME_INDEX, &records);
    leaf.resize(BTREE_NODE_SIZE, 0);
    let leaf = image.append(&leaf) as u64;
    let index = image.append(
        &btree_v2::Header::new(LINK_NAME_INDEX, LINK_RECORD_SIZE, leaf, 20).build(Widths::EIGHT),
    ) as u64;
    let root = image.append(&group_header(Some((heap, index))).build()) as u64;
    finish(image, root)
}

fn finish(mut image: Image, root: u64) -> Vec<u8> {
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .version(SUPERBLOCK_VERSION)
            .root_header_address(root)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    image.build()
}

fn compact_group(image: &mut Image, links: &[(&str, u64)]) -> u64 {
    image.append(
        &group_header(None)
            .messages(links.iter().map(|(name, address)| {
                Message::new(
                    MessageType::LINK,
                    &HardLink::new(name, *address).build(Widths::EIGHT),
                )
            }))
            .build(),
    ) as u64
}

fn group_header(dense: Option<(u64, u64)>) -> v2::Header {
    let (heap, index) = dense.unwrap_or((u64::MAX, u64::MAX));
    let mut info = vec![LINK_INFO_VERSION, NO_FLAGS];
    info.extend_from_slice(&heap.to_le_bytes());
    info.extend_from_slice(&index.to_le_bytes());
    v2::Header::new().messages([
        Message::new(MessageType::LINK_INFO, &info),
        Message::new(MessageType::GROUP_INFO, &[GROUP_INFO_VERSION, NO_FLAGS]),
    ])
}

fn dataset(image: &mut Image, values: Values<'_>) -> u64 {
    let (datatype, raw, count) = match values {
        Values::F64(values) => (
            datatype::f64_le(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            values.len(),
        ),
        Values::I32(values) => {
            let mut datatype =
                datatype::header(Class::FIXED_POINT, DATATYPE_VERSION, SIGNED_I32, 4);
            datatype.extend_from_slice(&0u16.to_le_bytes());
            datatype.extend_from_slice(&32u16.to_le_bytes());
            (
                datatype,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                values.len(),
            )
        }
    };
    let address = image.append(&raw) as u64;
    image.append(
        &v2::Header::new()
            .messages([
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::v2(1, Flags::NONE, Kind::SIMPLE, &[count as u64], None),
                ),
                Message::new(MessageType::DATATYPE, &datatype),
                Message::new(
                    MessageType::DATA_LAYOUT,
                    &data_layout::contiguous(address, raw.len() as u64),
                ),
            ])
            .build(),
    ) as u64
}

// Section `subsec_fmt4_infra_fractalheap`, version 4.0, defines this single direct
// block's accounting, four-byte heap offsets, and two-byte managed lengths.
fn heap_header(block: u64, used: usize) -> Vec<u8> {
    let mut header = fractal_heap::SIGNATURE.to_vec();
    header.push(HEAP_VERSION);
    header.extend_from_slice(&HEAP_ID_LENGTH.to_le_bytes());
    header.extend_from_slice(&0u16.to_le_bytes());
    header.push(NO_FLAGS);
    header.extend_from_slice(&MAX_MANAGED_SIZE.to_le_bytes());
    for field in [
        0,
        u64::MAX,
        (DIRECT_BLOCK_SIZE - DIRECT_BLOCK_PREFIX - used) as u64,
        u64::MAX,
        DIRECT_BLOCK_SIZE as u64,
        DIRECT_BLOCK_SIZE as u64,
        DIRECT_BLOCK_SIZE as u64,
        20,
        0,
        0,
        0,
        0,
    ] {
        header.extend_from_slice(&field.to_le_bytes());
    }
    header.extend_from_slice(&4u16.to_le_bytes());
    header.extend_from_slice(&(DIRECT_BLOCK_SIZE as u64).to_le_bytes());
    header.extend_from_slice(&MAX_DIRECT_BLOCK_SIZE.to_le_bytes());
    header.extend_from_slice(&HEAP_OFFSET_BITS.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&block.to_le_bytes());
    header.extend_from_slice(&0u16.to_le_bytes());
    checksum::append(&mut header);
    header
}

enum Values<'a> {
    F64(&'a [f64]),
    I32(&'a [i32]),
}

// Section `subsec_fmt4_boot_super`, version 4.0, defines the version 3 superblock
// and its length with eight-byte addresses and lengths.
const SUPERBLOCK_VERSION: u8 = 3;
const SUPERBLOCK_SIZE: usize = 48;

// Section `subsubsec_fmt4_dataobject_hdr_msg_linkinfo`, version 4.0.
const LINK_INFO_VERSION: u8 = 0;
const NO_FLAGS: u8 = 0;

// Section `subsubsec_fmt4_dataobject_hdr_msg_groupinfo`, version 4.0.
const GROUP_INFO_VERSION: u8 = 0;

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, defines
// the signed flag, little-endian byte order, and datatype version.
const SIGNED_I32: [u8; 3] = [0x08, 0, 0];
const DATATYPE_VERSION: u8 = 1;

// Section `subsec_fmt4_infra_fractalheap`, version 4.0, defines the version,
// managed ID, and prefix with eight-byte addresses and four-byte heap offsets.
const HEAP_VERSION: u8 = 0;
const MANAGED_ID: u8 = 0;
const HEAP_ID_LENGTH: u16 = 7;
const HEAP_OFFSET_BITS: u16 = 32;
const DIRECT_BLOCK_PREFIX: usize = 4 + 1 + 8 + 4;
const DIRECT_BLOCK_SIZE: usize = 1024;
const MAX_MANAGED_SIZE: u32 = 512;
const MAX_DIRECT_BLOCK_SIZE: u64 = 65_536;

// Section `subsubsec_fmt4_infra_btrees_v2`, version 4.0, defines type 5 records
// as a four-byte name hash and seven-byte heap ID.
const LINK_NAME_INDEX: u8 = 5;
const LINK_RECORD_SIZE: u16 = 11;
const BTREE_NODE_SIZE: usize = 512;
