//! Malformed-file construction and fixture loading for regression tests.

use crate::btree::v1 as btree_v1;
use crate::bytes;
use crate::image::Image;
use crate::legacy;
use crate::local_heap::Segment;
use crate::object_header::{Message, MessageType, v1};
use crate::superblock::v0::Fields;
use crate::superblock::v0::Superblock;
use crate::widths::Widths;

/// Loads a regression fixture and extends it to the end-of-file address stored in its superblock.
///
/// The fixture is read from `tests/data/fuzz/` relative to the repository root.
///
/// # Panics
///
/// Panics if the fixture cannot be read, its superblock cannot be parsed, or its recorded extent
/// is smaller than the fixture or cannot be represented as a buffer length.
pub fn fixture_with_recorded_extent(filename: &str) -> Vec<u8> {
    let mut image = std::fs::read(crate::data(&format!("fuzz/{filename}"))).unwrap();
    let Fields {
        widths,
        base_address,
        eof_address_at,
        root_header_address: _,
    } = Fields::read(&image, 0);
    assert_eq!(base_address, 0);
    let stored_eof = bytes::uint_at(&image, eof_address_at, widths.offset);
    assert!(stored_eof >= image.len() as u64);
    image.resize(usize::try_from(stored_eof).unwrap(), 0);
    image
}

/// Constructs a file with an incomplete group B-tree header and invalid signature.
///
/// Uses eight-byte addresses and leaves only 18 bytes at the referenced node.
pub fn truncated_group_btree_header() -> Vec<u8> {
    let widths = Widths::EIGHT;
    let superblock = Superblock::new(widths)
        .root_group(0, 0x60)
        .eof_address(0x89a)
        .build();
    let mut image = Image::starting_with(&superblock);
    let mut table = 0x888_u64.to_le_bytes().to_vec();
    table.extend_from_slice(&0x100_u64.to_le_bytes());
    image.place(
        0x60,
        &v1::Header::new()
            .message(Message::new(MessageType::SYMBOL_TABLE, &table))
            .build(),
    );
    image.place(
        0x88,
        &btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0], &[], widths),
    );
    let names = Segment::of_names(0x120, &[""]);
    image.place(0x100, &names.header(widths));
    image.place(0x120, &names.bytes);
    let mut tail = [0; 18];
    tail[..4].copy_from_slice(&[0xf0, 0x3f, 0, 0]);
    image.place(0x888, &tail);
    image.build()
}

/// Constructs a dataset file whose group B-tree root points twice to one leaf.
///
/// Uses the dataset from [`legacy::simple_dataset`] and duplicates its B-tree leaf reference.
pub fn repeated_group_btree_leaf() -> Vec<u8> {
    let mut image = legacy::simple_dataset();
    let Fields {
        widths,
        base_address: _,
        eof_address_at,
        root_header_address,
    } = Fields::read(&image, 0);
    let table = v1::root_group_chunks(&image)
        .into_iter()
        .flat_map(|chunk| chunk.records)
        .find(|record| record.msg_type == MessageType::SYMBOL_TABLE)
        .unwrap();
    let leaf = bytes::uint_at(&image, table.body.start, widths.offset);
    let root = image.len().next_multiple_of(8);
    let mut node = btree_v1::node_with_group_keys(
        btree_v1::NodeType::GROUP,
        1,
        &[0, 1, 1],
        &[leaf, leaf],
        widths,
    );
    let capacity = usize::try_from(root_header_address - leaf).unwrap();
    assert!(node.len() <= capacity);
    node.resize(capacity, 0);
    image.resize(root, 0);
    image.extend_from_slice(&node);
    bytes::set_uint_at(
        &mut image,
        table.body.start,
        widths.offset,
        u64::try_from(root).unwrap(),
    );
    let eof = u64::try_from(image.len()).unwrap();
    bytes::set_uint_at(&mut image, eof_address_at, widths.offset, eof);
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_group_fixture_has_two_references_to_one_leaf() {
        let image = repeated_group_btree_leaf();
        let fields = Fields::read(&image, 0);
        let table = v1::root_group_chunks(&image)
            .into_iter()
            .flat_map(|chunk| chunk.records)
            .find(|record| record.msg_type == MessageType::SYMBOL_TABLE)
            .unwrap();
        let root = usize::try_from(bytes::uint_at(
            &image,
            table.body.start,
            fields.widths.offset,
        ))
        .unwrap();
        assert_eq!(
            &image[root..root + btree_v1::SIGNATURE.len()],
            btree_v1::SIGNATURE
        );
        assert_eq!(
            &image[root + btree_v1::SIGNATURE.len()..root + btree_v1::SIGNATURE.len() + 4],
            &[btree_v1::NodeType::GROUP.0, 1, 2, 0]
        );
        let first_child =
            root + btree_v1::SIGNATURE.len() + 4 + 2 * fields.widths.offset + fields.widths.length;
        let leaf = bytes::uint_at(&image, first_child, fields.widths.offset);
        assert_eq!(
            bytes::uint_at(
                &image,
                first_child + fields.widths.offset + fields.widths.length,
                fields.widths.offset
            ),
            leaf
        );
        let leaf = usize::try_from(leaf).unwrap();
        assert_eq!(
            &image[leaf..leaf + btree_v1::SIGNATURE.len()],
            btree_v1::SIGNATURE
        );
        assert_eq!(
            &image[leaf + btree_v1::SIGNATURE.len()..leaf + btree_v1::SIGNATURE.len() + 4],
            &[btree_v1::NodeType::GROUP.0, 0, 1, 0]
        );
        assert_eq!(
            bytes::uint_at(&image, fields.eof_address_at, fields.widths.offset),
            u64::try_from(image.len()).unwrap()
        );
    }
}
