use test_util::btree::v1 as btree_v1;
use test_util::widths::Widths;

pub(super) fn node(widths: Widths, level: u8) -> Vec<u8> {
    btree_v1::node(
        btree_v1::NodeType::CHUNK,
        level,
        &[
            btree_v1::chunk_key(160, 0, &[0, 0]),
            btree_v1::chunk_key(0, 0, &[20, 0]),
        ],
        &[0x1000],
        widths,
    )
}

pub(super) fn header_len(widths: Widths) -> usize {
    CHUNK_HEADER_PREFIX_LEN + 2 * widths.offset
}

// "Version 1 B-trees" in the format specification, version 4.0, places the sibling
// addresses after the signature, type, level, and count. Each chunk key stores a size,
// a filter mask, and offsets.
const CHUNK_HEADER_PREFIX_LEN: usize = 8;
pub(super) const KEY_LEN: usize = 2 * size_of::<u32>() + 2 * size_of::<u64>();
