//! Version 2 B-trees: section `subsec_fmt4_infra_btrees`, version 4.0.
//!
//! A version 2 tree keeps its shape in a header of its own, and its nodes
//! carry a checksum.

use crate::bytes;
use crate::checksum;
use crate::widths::Widths;

/// The bytes of a tree's header, checksum included.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    tree_type: u8,
    node_size: u32,
    record_size: u16,
    depth: u16,
    split_percent: u8,
    merge_percent: u8,
    root_address: u64,
    records_in_root: u16,
    total_records: u64,
}

impl Header {
    /// A header for a tree whose root holds `records_in_root` records of
    /// `record_size` bytes and nothing below it.
    pub fn new(tree_type: u8, record_size: u16, root_address: u64, records_in_root: u16) -> Self {
        Self {
            tree_type,
            node_size: DEFAULT_NODE_SIZE,
            record_size,
            depth: 0,
            split_percent: DEFAULT_SPLIT_PERCENT,
            merge_percent: DEFAULT_MERGE_PERCENT,
            root_address,
            records_in_root,
            total_records: u64::from(records_in_root),
        }
    }

    pub fn node_size(mut self, node_size: u32) -> Self {
        self.node_size = node_size;
        self
    }

    /// How many levels of internal nodes sit above the leaves, which decides
    /// the width of the record counts a child pointer carries.
    pub fn depth(mut self, depth: u16) -> Self {
        self.depth = depth;
        self
    }

    pub fn total_records(mut self, total_records: u64) -> Self {
        self.total_records = total_records;
        self
    }

    pub fn build(&self, widths: Widths) -> Vec<u8> {
        let mut header = SIGNATURE.to_vec();
        header.push(VERSION);
        header.push(self.tree_type);
        header.extend_from_slice(&self.node_size.to_le_bytes());
        header.extend_from_slice(&self.record_size.to_le_bytes());
        header.extend_from_slice(&self.depth.to_le_bytes());
        header.push(self.split_percent);
        header.push(self.merge_percent);
        bytes::push_uint(&mut header, self.root_address, widths.offset);
        header.extend_from_slice(&self.records_in_root.to_le_bytes());
        bytes::push_uint(&mut header, self.total_records, widths.length);
        checksum::append(&mut header);
        header
    }
}

/// The bytes of a leaf node holding `records`, checksum included.
pub fn leaf(tree_type: u8, records: &[Vec<u8>]) -> Vec<u8> {
    let mut node = LEAF_SIGNATURE.to_vec();
    node.push(VERSION);
    node.push(tree_type);
    node.extend(records.iter().flatten());
    checksum::append(&mut node);
    node
}

/// The bytes of an internal node holding `records` above `children`, checksum
/// included.
pub fn internal(tree_type: u8, records: &[Vec<u8>], children: &[Child], widths: Widths) -> Vec<u8> {
    let mut node = INTERNAL_SIGNATURE.to_vec();
    node.push(VERSION);
    node.push(tree_type);
    node.extend(records.iter().flatten());
    for child in children {
        bytes::push_uint(&mut node, child.address, widths.offset);
        bytes::push_uint(&mut node, u64::from(child.records), child.records_width);
        if let Some(subtree) = child.subtree {
            bytes::push_uint(&mut node, subtree.records, subtree.width);
        }
    }
    checksum::append(&mut node);
    node
}

/// A pointer from an internal node to the node below it.
///
/// The two counts are stored in as few bytes as the node size and the depth
/// allow, so their widths belong to the level the pointer sits at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Child {
    pub address: u64,
    /// Records in the child node itself.
    pub records: u16,
    pub records_width: usize,
    /// `None` for a pointer to a leaf, whose own count is already the
    /// subtree's, so that the node leaves the field out.
    pub subtree: Option<Subtree>,
}

/// How many records a subtree holds, and the width the node above it stores
/// that count in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Subtree {
    pub records: u64,
    pub width: usize,
}

/// Depth of the file's only v2 B-tree, from its header: 0 when the root is a
/// leaf, and one more per level of internal nodes above it.
///
/// A test that means to exercise an internal node has to prove it built one,
/// since the same assertions pass vacuously on a single-leaf tree. Panics unless
/// the file holds exactly one B-tree header, so a caller cannot silently read the
/// depth of a heap's huge-objects index when it meant the attribute name index.
///
/// Header layout: signature(4) + version(1) + type(1) + node size(4) + record
/// size(2) + depth(2).
pub fn sole_btree_depth(bytes: &[u8]) -> u16 {
    bytes::u16_at(bytes, sole_btree_header(bytes) + 4 + 1 + 1 + 4 + 2)
}

/// Offset of the file's only v2 B-tree header.
///
/// Sole-ness is the point, not a convenience: a heap with a huge object carries a
/// second B-tree whose records are 24 bytes where the first index's are 17, and
/// decoding one at the other's stride yields plausible numbers. This reports
/// the moment the file stops holding the attribute name index alone.
#[track_caller]
fn sole_btree_header(bytes: &[u8]) -> usize {
    bytes::sole_signature(bytes, SIGNATURE)
}

/// The `(creation order, hash)` of `count` dense-attribute name-index records
/// laid out back to back at `first`.
///
/// Record layout, for the 8-byte heap IDs that index uses: heap ID(8) + message
/// flags(1) + creation order(4) + name hash(4). Shared by the leaf and root
/// readers so the two cannot drift on it.
fn name_index_records(bytes: &[u8], first: usize, count: usize) -> Vec<(u32, u32)> {
    const RECORD: usize = 8 + 1 + 4 + 4;
    (0..count)
        .map(|i| {
            let at = first + i * RECORD;
            (bytes::u32_at(bytes, at + 9), bytes::u32_at(bytes, at + 13))
        })
        .collect()
}

/// The `(creation order, name hash)` of each record in the file's first v2
/// B-tree leaf node, in the order the node stores them.
///
/// That order is the one a reader binary-searches, so it is the thing a test
/// about record ordering has to look at: reading the attributes back walks the
/// node start to finish and is satisfied by any order at all. Creation order is
/// the attribute's index in the order it was set, which is what identifies
/// *which* attribute a record belongs to without decoding the heap.
///
/// Only meaningful for an index small enough to be a single leaf, and for the
/// 8-byte heap IDs a dense attribute name index uses (record layout: heap ID(8) +
/// message flags(1) + creation order(4) + name hash(4)).
pub fn name_index_leaf_records(bytes: &[u8], count: usize) -> Vec<(u32, u32)> {
    let leaf = bytes::find_signature(bytes, LEAF_SIGNATURE)
        .expect("a dense attribute name index has a leaf node");
    name_index_records(bytes, leaf + NODE_PREFIX, count)
}

/// The `(creation order, name hash)` of each record the file's sole v2 B-tree
/// keeps in its *root* node, in the order the node stores them.
///
/// A record in an internal node is one the tree promoted out of the level below:
/// it lives only there, and a search compares against it while choosing which
/// child to descend into. That makes it the record a wrong ordering hurts most,
/// and the reason a test about ordering reports whether a colliding name
/// reached one: on a single-leaf index there is nothing to descend and the same
/// assertions pass without testing anything.
///
/// The root is the one node reachable without decoding a child pointer, whose
/// width varies by depth: the header carries both its address and its own record
/// count. That works at any depth: on a depth-0 tree the root is the single
/// leaf and this returns the whole index, and only a deeper tree makes the
/// result mean "promoted".
///
/// Header layout past [`sole_btree_depth`]: split %(1) + merge %(1) + root
/// address(8) + records in root(2).
pub fn root_records(bytes: &[u8]) -> Vec<(u32, u32)> {
    let at = sole_btree_header(bytes) + 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1;
    let root = bytes::u64_at(bytes, at) as usize;
    let count = bytes::u16_at(bytes, at + LENGTH) as usize;
    name_index_records(bytes, root + NODE_PREFIX, count)
}

pub const SIGNATURE: &[u8; 4] = b"BTHD";

pub const LEAF_SIGNATURE: &[u8; 4] = b"BTLF";

pub const INTERNAL_SIGNATURE: &[u8; 4] = b"BTIN";

/// signature(4) + version(1) + type(1), which every node carries before its
/// first record.
pub const NODE_PREFIX: usize = 6;

const VERSION: u8 = 0;

/// The node size the reference library gives a tree unless a creation property
/// says otherwise.
const DEFAULT_NODE_SIZE: u32 = 512;

/// How full a node is allowed to get before it splits, as a percentage.
const DEFAULT_SPLIT_PERCENT: u8 = 85;

/// How empty a node is allowed to get before it merges, as a percentage.
const DEFAULT_MERGE_PERCENT: u8 = 40;

/// Byte offsets below are for the eight-byte lengths every file these readers
/// are pointed at carries.
const LENGTH: usize = 8;
