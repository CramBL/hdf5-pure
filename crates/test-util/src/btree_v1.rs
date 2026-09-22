//! Version 1 B-tree nodes: section `subsec_fmt4_infra_btrees`, version 4.0.
//!
//! Both node types share this layout and differ only in what a key holds: a
//! type 0 node indexes a group's symbol table nodes by a link name's heap
//! offset, and a type 1 node indexes a dataset's chunks by their coordinates.

use crate::bytes;
use crate::widths::Widths;

/// The bytes of one node, whose `keys` are the raw key bytes of each entry and
/// which holds one more key than it has children.
///
/// A node of level zero is a leaf, whose children are the structures the tree
/// indexes. A node above it has other nodes as children.
#[track_caller]
pub fn node(
    node_type: NodeType,
    level: u8,
    keys: &[Vec<u8>],
    children: &[u64],
    widths: Widths,
) -> Vec<u8> {
    assert_eq!(
        keys.len(),
        children.len() + 1,
        "a node holds one more key than it has children"
    );

    let mut node = SIGNATURE.to_vec();
    node.push(node_type.0);
    node.push(level);
    node.extend_from_slice(&(children.len() as u16).to_le_bytes());
    // The siblings at this level, both undefined in every tree these fixtures
    // build, which is one node wide per level.
    bytes::push_undefined_address(&mut node, widths.offset);
    bytes::push_undefined_address(&mut node, widths.offset);
    for (key, &child) in keys.iter().zip(children) {
        node.extend_from_slice(key);
        bytes::push_uint(&mut node, child, widths.offset);
    }
    node.extend_from_slice(keys.last().expect("one more key than children"));
    node
}

/// The key of a type 0 node: the offset in the group's local heap of the first
/// link name in the subtree below it.
pub fn group_key(link_name_offset: u64, widths: Widths) -> Vec<u8> {
    let mut key = Vec::new();
    bytes::push_uint(&mut key, link_name_offset, widths.length);
    key
}

/// The key of a type 1 node: the chunk's stored size, the filters skipped for
/// it, and where it sits in the dataset.
///
/// `offsets` is the dataset's own dimensionality plus the trailing offset
/// within an element that the format's chunk keys carry, so a rank 1 dataset
/// gives two.
pub fn chunk_key(storage_size: u32, filter_mask: u32, offsets: &[u64]) -> Vec<u8> {
    let mut key = Vec::new();
    key.extend_from_slice(&storage_size.to_le_bytes());
    key.extend_from_slice(&filter_mask.to_le_bytes());
    key.extend(offsets.iter().flat_map(|offset| offset.to_le_bytes()));
    key
}

/// What a node's keys index, which the specification calls its node type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeType(pub u8);

impl NodeType {
    /// A group's symbol table nodes, keyed by a link name's heap offset.
    pub const GROUP: Self = Self(0);
    /// A dataset's chunks, keyed by their coordinates in the dataset.
    pub const CHUNK: Self = Self(1);
}

pub const SIGNATURE: &[u8; 4] = b"TREE";
