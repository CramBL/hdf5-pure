//! Independent file images with dense attributes on a dataset or the root group.
//!
//! Each image has a version 2 superblock, version 2 object headers, eight-byte
//! addresses and lengths, and little-endian numeric values. Attributes are stored
//! in a fractal heap with a version 2 B-tree name index.

use crate::attribute;
use crate::attribute::Attribute;
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
use crate::link_message::CharacterSet;
use crate::link_message::HardLink;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v2;
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// A fixed hierarchy with dense scalar [`f64`] attributes.
#[derive(Clone, Copy, Debug)]
pub enum Fixture {
    /// Fifty attributes on the dataset `data`.
    Dataset,
    /// Twenty attributes on the root group, beside the dataset `values`.
    Root,
}

impl Fixture {
    /// Encodes the hierarchy and its dense attributes as an independent file image.
    pub fn build(self) -> Vec<u8> {
        let mut image = Image::starting_with(&[0; SUPERBLOCK_SIZE]);
        let (dataset_attributes, root_attributes) = match self {
            Fixture::Dataset => (Some(dense_storage(&mut image, self)), None),
            Fixture::Root => (None, Some(dense_storage(&mut image, self))),
        };
        let (link_name, datatype, raw) = match self {
            Fixture::Dataset => (
                "data",
                datatype::f64_le(),
                [1.0f64, 2.0, 3.0]
                    .into_iter()
                    .flat_map(f64::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            Fixture::Root => {
                let mut datatype =
                    datatype::header(Class::FIXED_POINT, DATATYPE_VERSION, SIGNED_I32, 4);
                datatype.extend_from_slice(&0u16.to_le_bytes());
                datatype.extend_from_slice(&32u16.to_le_bytes());
                (
                    "values",
                    datatype,
                    [10i32, 20, 30]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect(),
                )
            }
        };
        let address = image.append(&raw) as u64;
        let dataset = v2::Header::new()
            .messages([
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::v2(1, Flags::NONE, Kind::SIMPLE, &[3], None),
                ),
                Message::new(MessageType::DATATYPE, &datatype),
                Message::new(
                    MessageType::DATA_LAYOUT,
                    &data_layout::contiguous(address, raw.len() as u64),
                ),
            ])
            .messages(dataset_attributes);
        let dataset = image.append(&dataset.build()) as u64;
        let mut link_info = vec![LINK_INFO_VERSION, NO_FLAGS];
        link_info.extend_from_slice(&u64::MAX.to_le_bytes());
        link_info.extend_from_slice(&u64::MAX.to_le_bytes());
        let root = v2::Header::new()
            .messages([
                Message::new(MessageType::LINK_INFO, &link_info),
                Message::new(MessageType::GROUP_INFO, &[GROUP_INFO_VERSION, NO_FLAGS]),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new(link_name, dataset).build(Widths::EIGHT),
                ),
            ])
            .messages(root_attributes);
        let root = image.append(&root.build()) as u64;
        image.place(
            0,
            &Superblock::new(Widths::EIGHT)
                .root_header_address(root)
                .eof_address(image.as_bytes().len() as u64)
                .build(),
        );
        image.build()
    }

    /// Returns the number of attributes on the selected owner.
    pub fn attribute_count(self) -> u16 {
        match self {
            Self::Dataset => 50,
            Self::Root => 20,
        }
    }

    /// Returns the attribute names and scalar values in increasing index order.
    pub fn attributes(self) -> impl Iterator<Item = (String, f64)> {
        (0..self.attribute_count()).map(move |index| match self {
            Self::Dataset => (format!("attr_{index:03}"), f64::from(index) * 1.5),
            Self::Root => (format!("root_attr_{index:02}"), f64::from(index) * 2.0),
        })
    }
}

/// Builds `data = [1.0, 2.0, 3.0]` as [`f64`] values with fifty dense attributes.
///
/// Attributes `attr_000` through `attr_049` store their index multiplied by 1.5
/// as scalar [`f64`] values.
pub fn dense_attrs() -> Vec<u8> {
    Fixture::Dataset.build()
}

/// Builds twenty dense root attributes beside `values = [10, 20, 30]` as [`i32`] values.
///
/// Attributes `root_attr_00` through `root_attr_19` store their index multiplied
/// by 2.0 as scalar [`f64`] values.
pub fn dense_attrs_root() -> Vec<u8> {
    Fixture::Root.build()
}

fn dense_storage(image: &mut Image, fixture: Fixture) -> Message {
    let datatype = datatype::f64_le();
    let dataspace = dataspace::scalar();
    let mut objects = Vec::new();
    let mut records: Vec<_> = fixture
        .attributes()
        .enumerate()
        .map(|(index, (name, value))| {
            let attribute = Attribute::new(&name, &datatype, &dataspace, &value.to_le_bytes())
                .character_set(CharacterSet::ASCII.0)
                .build();
            let mut record = vec![MANAGED_ID];
            record.extend_from_slice(
                &((DIRECT_BLOCK_PREFIX + objects.len()) as u64).to_le_bytes()[..HEAP_OFFSET_WIDTH],
            );
            // Each scalar attribute has fewer than 64 bytes.
            record.extend_from_slice(&u16::try_from(attribute.len()).unwrap().to_le_bytes());
            record.push(NO_FLAGS);
            // The fixture has at most fifty attributes.
            record.extend_from_slice(&u32::try_from(index).unwrap().to_le_bytes());
            record.extend_from_slice(&checksum::lookup3(name.as_bytes()).to_le_bytes());
            objects.extend_from_slice(&attribute);
            record
        })
        .collect();
    records.sort_by_key(|record| bytes::u32_at(record, HASH_AT));
    let heap_at = image.append(&heap_header(0, objects.len(), fixture.attribute_count()));
    let mut block = fractal_heap::DIRECT_BLOCK_SIGNATURE.to_vec();
    block.push(HEAP_VERSION);
    block.extend_from_slice(&(heap_at as u64).to_le_bytes());
    block.extend_from_slice(&[0; HEAP_OFFSET_WIDTH]);
    block.extend_from_slice(&objects);
    assert!(block.len() <= DIRECT_BLOCK_SIZE);
    block.resize(DIRECT_BLOCK_SIZE, 0);
    let block = image.append(&block) as u64;
    image.place(
        heap_at,
        &heap_header(block, objects.len(), fixture.attribute_count()),
    );
    let mut leaf = btree_v2::leaf(ATTRIBUTE_NAME_INDEX, &records);
    leaf.resize(BTREE_NODE_SIZE, 0);
    let leaf = image.append(&leaf) as u64;
    let index = image.append(
        &btree_v2::Header::new(
            ATTRIBUTE_NAME_INDEX,
            ATTRIBUTE_RECORD_SIZE,
            leaf,
            fixture.attribute_count(),
        )
        .node_size(BTREE_NODE_SIZE as u32)
        .build(Widths::EIGHT),
    ) as u64;
    Message::new(
        MessageType::ATTRIBUTE_INFO,
        &attribute::info(Some(heap_at as u64), Some(index), Widths::EIGHT),
    )
}

// Section `subsec_fmt4_infra_fractalheap`, version 4.0, defines the header's
// field order and accounting for a single unfiltered managed direct block.
fn heap_header(block: u64, used: usize, count: u16) -> Vec<u8> {
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
        u64::from(count),
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

// Section `subsec_fmt4_boot_super`, version 4.0, defines the length of a
// version 2 superblock with eight-byte addresses and lengths.
const SUPERBLOCK_SIZE: usize = 48;

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.
const DATATYPE_VERSION: u8 = 1;
const SIGNED_I32: [u8; 3] = [0x08, 0, 0];

// Section `subsec_fmt4_infra_fractalheap`, version 4.0, defines the version,
// flags and managed ID encoding. These heap offsets occupy five bytes.
const HEAP_VERSION: u8 = 0;
const NO_FLAGS: u8 = 0;
const MANAGED_ID: u8 = 0;
const HEAP_ID_LENGTH: u16 = 8;
const HEAP_OFFSET_BITS: u16 = 40;
const HEAP_OFFSET_WIDTH: usize = 5;
const DIRECT_BLOCK_PREFIX: usize = 4 + 1 + 8 + HEAP_OFFSET_WIDTH;
const DIRECT_BLOCK_SIZE: usize = 4096;
const MAX_MANAGED_SIZE: u32 = 512;
const MAX_DIRECT_BLOCK_SIZE: u64 = 65_536;

// Section `subsubsec_fmt4_infra_btrees_v2`, version 4.0, defines type 8 records:
// heap ID, flags, creation order, and name hash.
const ATTRIBUTE_NAME_INDEX: u8 = 8;
const ATTRIBUTE_RECORD_SIZE: u16 = 17;
const HASH_AT: usize = 8 + 1 + 4;
const BTREE_NODE_SIZE: usize = 1024;

// Section `subsubsec_fmt4_dataobject_hdr_msg_linkinfo`, version 4.0.
const LINK_INFO_VERSION: u8 = 0;

// Section `subsubsec_fmt4_dataobject_hdr_msg_groupinfo`, version 4.0.
const GROUP_INFO_VERSION: u8 = 0;
