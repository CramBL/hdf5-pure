//! Independent file images with symbol-table groups and contiguous numeric datasets.
//!
//! Each image has a version 0 superblock, version 1 object headers, eight-byte
//! addresses and lengths, and little-endian numeric payloads.

use crate::btree_v1;
use crate::btree_v1::NodeType;
use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::datatype;
use crate::datatype::Class;
use crate::image::Image;
use crate::local_heap::Segment;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v1;
use crate::superblock::v0::Superblock;
use crate::symbol_table;
use crate::symbol_table::Entry;
use crate::widths::Widths;

/// A fixed legacy group and dataset hierarchy.
#[derive(Clone, Copy, Debug)]
pub enum Fixture {
    /// The dataset `a/b/c/deep`.
    NestedGroups,
    /// The root dataset `data`.
    SimpleDataset,
    /// The datasets `group1/values` and `group2/temps`.
    TwoGroups,
}

impl Fixture {
    /// Encodes the hierarchy as an independent file image.
    pub fn build(self) -> Vec<u8> {
        match self {
            Self::NestedGroups => nested_groups(),
            Self::SimpleDataset => simple_dataset(),
            Self::TwoGroups => two_groups(),
        }
    }
}

/// Builds nested symbol-table groups with `a/b/c/deep = [42.0]` as [`f64`] values.
pub fn nested_groups() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let deep = dataset(&mut image, Values::F64(&[42.0]));
    let c = group(&mut image, &[("deep", deep)]);
    let b = group(&mut image, &[("c", c)]);
    let a = group(&mut image, &[("b", b)]);
    let root = group(&mut image, &[("a", a)]);
    finish(image, root)
}

/// Builds a contiguous [`f64`] dataset `data = [1.0, 2.0, 3.0]` in the root group.
pub fn simple_dataset() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let data = dataset(&mut image, Values::F64(&[1.0, 2.0, 3.0]));
    let root = group(&mut image, &[("data", data)]);
    finish(image, root)
}

/// Builds `group1/values = [10, 20, 30]` and `group2/temps = [98.6, 37.0]`.
///
/// The first dataset stores [`i32`] values and the second stores [`f32`] values.
pub fn two_groups() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let values = dataset(&mut image, Values::I32(&[10, 20, 30]));
    let temps = dataset(&mut image, Values::F32(&[98.6, 37.0]));
    let group1 = group(&mut image, &[("values", values)]);
    let group2 = group(&mut image, &[("temps", temps)]);
    let root = group(&mut image, &[("group1", group1), ("group2", group2)]);
    finish(image, root)
}

pub(crate) fn finish(mut image: Image, root: u64) -> Vec<u8> {
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .root_group(0, root)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    image.build()
}

fn group(image: &mut Image, links: &[(&str, u64)]) -> u64 {
    group_with_messages(image, links, &[])
}

/// Writes a symbol-table group with caller-supplied header messages.
///
/// Pass links in name order. Symbol entries are distributed across leaves with
/// at most eight entries each and at least four when more than one leaf is used.
///
/// # Panics
///
/// Panics if there are more than 256 links.
pub(crate) fn group_with_messages(
    image: &mut Image,
    links: &[(&str, u64)],
    messages: &[Message],
) -> u64 {
    let names: Vec<_> = std::iter::once("")
        .chain(links.iter().map(|(name, _)| *name))
        .collect();
    let mut segment = Segment::of_names(0, &names);
    segment
        .bytes
        .resize(segment.bytes.len().next_multiple_of(8), 0);
    segment.at = image.append_aligned(&segment.bytes, 8) as u64;
    let heap = image.append_aligned(&segment.header(Widths::EIGHT), 8) as u64;
    let entries: Vec<_> = links
        .iter()
        .enumerate()
        .map(|(index, (_, address))| Entry::new(segment.offset_of(index + 1), *address))
        .collect();
    let mut tree = if entries.is_empty() {
        btree_v1::node_with_group_keys(NodeType::GROUP, 0, &[0], &[], Widths::EIGHT)
    } else if entries.len() <= SYMBOL_NODE_CAPACITY {
        let mut symbols = symbol_table::node(&entries, Widths::EIGHT);
        symbols.resize(SYMBOL_NODE_SIZE, 0);
        btree_v1::node_with_group_keys(
            NodeType::GROUP,
            0,
            &[0, segment.offset_of(links.len())],
            &[image.append_aligned(&symbols, 8) as u64],
            Widths::EIGHT,
        )
    } else {
        assert!(entries.len() <= SYMBOL_NODE_CAPACITY * BTREE_CHILD_CAPACITY);
        let leaves = entries.len().div_ceil(SYMBOL_NODE_CAPACITY);
        let entries_per_leaf = entries.len() / leaves;
        let longer_leaves = entries.len() % leaves;
        let mut keys = vec![0];
        let children: Vec<_> = (0..leaves)
            .map(|leaf| {
                let start = leaf * entries_per_leaf + leaf.min(longer_leaves);
                let end = start + entries_per_leaf + usize::from(leaf < longer_leaves);
                let entries = &entries[start..end];
                let mut symbols = symbol_table::node(entries, Widths::EIGHT);
                symbols.resize(SYMBOL_NODE_SIZE, 0);
                keys.push(entries.last().unwrap().link_name_offset);
                image.append_aligned(&symbols, 8) as u64
            })
            .collect();
        btree_v1::node_with_group_keys(NodeType::GROUP, 0, &keys, &children, Widths::EIGHT)
    };
    tree.resize(BTREE_NODE_SIZE, 0);
    let tree = image.append_aligned(&tree, 8) as u64;
    let mut table = tree.to_le_bytes().to_vec();
    table.extend_from_slice(&heap.to_le_bytes());
    image.append_aligned(
        &v1::Header::new()
            .message(Message::new(MessageType::SYMBOL_TABLE, &table))
            .messages(messages.iter().cloned())
            .build(),
        8,
    ) as u64
}

fn dataset(image: &mut Image, values: Values<'_>) -> u64 {
    let (datatype, raw, count) = match values {
        Values::F64(values) => (
            datatype::f64_le(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            values.len(),
        ),
        Values::F32(values) => (
            f32_datatype(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            values.len(),
        ),
        Values::I32(values) => (
            i32_datatype(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            values.len(),
        ),
    };
    let raw: Vec<u8> = raw;
    dataset_with_messages(image, datatype, &raw, count as u64, &[])
}

pub(crate) fn dataset_with_messages(
    image: &mut Image,
    datatype: Vec<u8>,
    raw: &[u8],
    count: u64,
    messages: &[Message],
) -> u64 {
    let address = image.append_aligned(raw, 8) as u64;
    image.append_aligned(
        &v1::Header::new()
            .messages([
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::v1(1, Flags::NONE, &[count], None),
                ),
                padded_message(MessageType::DATATYPE, datatype),
                padded_message(
                    MessageType::DATA_LAYOUT,
                    data_layout::contiguous(address, raw.len() as u64),
                ),
            ])
            .messages(messages.iter().cloned())
            .build(),
        8,
    ) as u64
}

fn padded_message(kind: MessageType, mut data: Vec<u8>) -> Message {
    data.resize(data.len().next_multiple_of(MESSAGE_ALIGNMENT), 0);
    Message::new(kind, &data)
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, defines the
// class fields, bit precision and IEEE floating-point fields encoded here.
fn f32_datatype() -> Vec<u8> {
    let mut datatype = datatype::header(
        Class::FLOATING_POINT,
        DATATYPE_VERSION,
        IEEE754_BINARY32_CLASS_FIELDS,
        4,
    );
    datatype.extend_from_slice(&0u16.to_le_bytes());
    datatype.extend_from_slice(&32u16.to_le_bytes());
    datatype.extend_from_slice(&[23, 8, 0, 23]);
    datatype.extend_from_slice(&127u32.to_le_bytes());
    datatype
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, defines
// the signed flag and bit precision encoded here.
/// Encodes a signed little-endian 32-bit integer datatype.
pub(crate) fn i32_datatype() -> Vec<u8> {
    let mut datatype = datatype::header(
        Class::FIXED_POINT,
        DATATYPE_VERSION,
        SIGNED_I32_CLASS_FIELDS,
        4,
    );
    datatype.extend_from_slice(&0u16.to_le_bytes());
    datatype.extend_from_slice(&32u16.to_le_bytes());
    datatype
}

enum Values<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    I32(&'a [i32]),
}

// Section `subsec_fmt4_infra_symboltable`, version 4.0, reserves 2K entries
// after the eight-byte prefix. The superblock's default leaf K is four.
const SYMBOL_NODE_CAPACITY: usize = 8;
const SYMBOL_NODE_SIZE: usize = 8 + SYMBOL_NODE_CAPACITY * 40;

// Section `subsubsec_fmt4_infra_btrees_v1`, version 4.0, reserves 2K children
// and 2K+1 keys. The superblock's default internal K is sixteen.
const BTREE_CHILD_CAPACITY: usize = 32;
const BTREE_NODE_SIZE: usize = 24 + BTREE_CHILD_CAPACITY * 8 + (BTREE_CHILD_CAPACITY + 1) * 8;

// Section `subsubsec_fmt4_dataobject_hdr_prefix_one`, version 4.0, aligns
// version 1 message bodies to eight bytes.
pub(crate) const MESSAGE_ALIGNMENT: usize = 8;

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.
const DATATYPE_VERSION: u8 = 1;

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, defines
// the implied-leading-one normalization and sign position encoded here.
const IEEE754_BINARY32_CLASS_FIELDS: [u8; 3] = [0x20, 31, 0];

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, defines
// the signed flag and little-endian byte order encoded here.
const SIGNED_I32_CLASS_FIELDS: [u8; 3] = [0x08, 0, 0];

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::bytes;

    use super::*;

    #[rstest]
    #[case::single_leaf(8, 1)]
    #[case::first_split(9, 2)]
    #[case::tail_below_minimum(11, 2)]
    #[case::tail_at_minimum(12, 2)]
    #[case::mat_reference_group(54, 7)]
    #[case::full_tree(256, 32)]
    fn symbol_table_leaves_preserve_occupancy_and_boundaries(
        #[case] count: usize,
        #[case] expected_leaves: usize,
    ) {
        let names = (0..count)
            .map(|i| format!("entry_{i:03}"))
            .collect::<Vec<_>>();
        let links = names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i as u64))
            .collect::<Vec<_>>();
        let mut image = Image::new();
        let address = group_with_messages(&mut image, &links, &[]) as usize;
        let bytes = image.as_bytes();
        let chunks = v1::chunks(bytes, address, Widths::EIGHT);
        let table = chunks[0]
            .records
            .iter()
            .find(|record| record.msg_type == MessageType::SYMBOL_TABLE)
            .unwrap();
        let tree = bytes::u64_at(bytes, table.body.start) as usize;
        assert_eq!(
            &bytes[tree..tree + btree_v1::SIGNATURE.len()],
            btree_v1::SIGNATURE
        );
        let leaves = usize::from(bytes::u16_at(bytes, tree + NODE_COUNT_AT));
        assert_eq!(leaves, expected_leaves);
        let names = Segment::of_names(
            0,
            &std::iter::once("")
                .chain(names.iter().map(String::as_str))
                .collect::<Vec<_>>(),
        );
        let mut seen = 0;
        for leaf in 0..leaves {
            let key = tree + BTREE_PREFIX + leaf * 2 * Widths::EIGHT.offset;
            assert_eq!(bytes::u64_at(bytes, key), names.offset_of(seen));
            let symbols = bytes::u64_at(bytes, key + Widths::EIGHT.length) as usize;
            assert_eq!(
                &bytes[symbols..symbols + symbol_table::SIGNATURE.len()],
                symbol_table::SIGNATURE
            );
            let size = usize::from(bytes::u16_at(bytes, symbols + NODE_COUNT_AT));
            assert!((MIN_SYMBOLS..=SYMBOL_NODE_CAPACITY).contains(&size));
            for entry in 0..size {
                let at = symbols + SYMBOL_PREFIX + entry * SYMBOL_ENTRY_SIZE;
                assert_eq!(bytes::u64_at(bytes, at), names.offset_of(seen + entry + 1));
                assert_eq!(
                    bytes::u64_at(bytes, at + Widths::EIGHT.length),
                    (seen + entry) as u64
                );
            }
            seen += size;
            assert_eq!(
                bytes::u64_at(bytes, key + 2 * Widths::EIGHT.offset),
                names.offset_of(seen)
            );
        }
        assert_eq!(seen, count);
    }

    // Sections `subsec_fmt4_boot_super`, `subsubsec_fmt4_infra_btrees_v1`,
    // `subsec_fmt4_infra_symboltable` and `subsec_fmt4_infra_symboltableentry`, version 4.0.
    const MIN_SYMBOLS: usize = 4;
    const NODE_COUNT_AT: usize = 6;
    const BTREE_PREFIX: usize = 24;
    const SYMBOL_PREFIX: usize = 8;
    const SYMBOL_ENTRY_SIZE: usize = 40;
}
