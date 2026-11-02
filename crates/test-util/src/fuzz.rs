//! Malformed-file construction and fixture loading for regression tests.

use crate::btree_v1;
use crate::bytes;
use crate::image::Image;
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
