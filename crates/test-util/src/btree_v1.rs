//! Version 1 B-tree nodes: section `subsec_fmt4_infra_btrees`, version 4.0.
//!
//! Both node types share this layout and differ only in what a key holds: a
//! type 0 node indexes a group's symbol table nodes by a link name's heap
//! offset, and a type 1 node indexes a dataset's chunks by their coordinates.

use crate::bytes;
use crate::image::Image;
use crate::widths::Widths;

/// An in-memory traversal fixture with expected node and payload addresses.
pub struct TraversalTree {
    /// The node image, including chunk payload bytes for a chunk fixture.
    pub bytes: Vec<u8>,
    /// Each node's stored address and in-use byte length, in address order.
    pub nodes: Vec<(u64, usize)>,
    /// One stored payload address per leaf, in leaf order.
    pub payloads: Vec<u64>,
}

impl TraversalTree {
    /// Builds group nodes whose leaf pointers name synthetic symbol table addresses.
    ///
    /// The image contains node bytes only.
    ///
    /// # Panics
    ///
    /// Panics if a binary tree's root level exceeds eight.
    pub fn group(shape: TraversalShape) -> Self {
        Self::build(shape, TreeKind::Group)
    }

    /// Builds chunk nodes with one four-byte payload per leaf.
    ///
    /// # Panics
    ///
    /// Panics if a binary tree's root level exceeds eight.
    pub fn chunk(shape: TraversalShape) -> Self {
        Self::build(shape, TreeKind::Chunk)
    }

    fn build(shape: TraversalShape, kind: TreeKind) -> Self {
        let (depth, count) = match shape {
            TraversalShape::Binary(depth) => {
                assert!(depth <= 8);
                (depth, (1usize << (depth + 1)) - 1)
            }
            TraversalShape::Chain(depth) | TraversalShape::SharedChain(depth) => {
                (depth, usize::from(depth) + 1)
            }
        };
        let mut image = Image::new();
        let mut nodes = Vec::new();
        let mut payloads = Vec::new();
        for index in 0..count {
            let level = match shape {
                TraversalShape::Binary(_) => depth - u8::try_from((index + 1).ilog2()).unwrap(),
                TraversalShape::Chain(_) | TraversalShape::SharedChain(_) => {
                    depth - u8::try_from(index).unwrap()
                }
            };
            let children = if level == 0 {
                let address = PAYLOAD_BASE + u64::try_from(payloads.len()).unwrap() * 4;
                payloads.push(address);
                vec![address]
            } else {
                match shape {
                    TraversalShape::Binary(_) => vec![
                        u64::try_from(2 * index + 1).unwrap() * NODE_STRIDE,
                        u64::try_from(2 * index + 2).unwrap() * NODE_STRIDE,
                    ],
                    TraversalShape::Chain(_) => {
                        vec![u64::try_from(index + 1).unwrap() * NODE_STRIDE]
                    }
                    TraversalShape::SharedChain(_) => {
                        vec![u64::try_from(index + 1).unwrap() * NODE_STRIDE; 2]
                    }
                }
            };
            let keys = match kind {
                TreeKind::Group => (0..=children.len())
                    .map(|key| group_key(u64::try_from(key).unwrap(), Widths::EIGHT))
                    .collect::<Vec<_>>(),
                TreeKind::Chunk => (0..=children.len())
                    .map(|key| {
                        let offset = if level == 0 {
                            u64::try_from(payloads.len() - 1 + key).unwrap()
                        } else {
                            u64::try_from(key).unwrap()
                        };
                        chunk_key(4, 0, &[offset, 0])
                    })
                    .collect(),
            };
            let node_type = match kind {
                TreeKind::Group => NodeType::GROUP,
                TreeKind::Chunk => NodeType::CHUNK,
            };
            let bytes = node(node_type, level, &keys, &children, Widths::EIGHT);
            assert!(bytes.len() <= usize::try_from(NODE_STRIDE).unwrap());
            let address = u64::try_from(index).unwrap() * NODE_STRIDE;
            image.place(usize::try_from(address).unwrap(), &bytes);
            if level == 0 && matches!(kind, TreeKind::Chunk) {
                image.place(usize::try_from(children[0]).unwrap(), &[1, 0, 0, 0]);
            }
            nodes.push((address, bytes.len()));
        }
        Self {
            bytes: image.build(),
            nodes,
            payloads,
        }
    }
}

/// The child-pointer arrangement and root level of a traversal fixture.
#[derive(Clone, Copy, Debug)]
pub enum TraversalShape {
    /// A complete binary tree with the supplied root level.
    Binary(u8),
    /// A chain with one node per level.
    Chain(u8),
    /// A chain with two pointers to the same child at each internal node.
    SharedChain(u8),
}

/// The bytes of one node, whose `keys` are the raw key bytes of each entry and
/// which holds one more key than it has children.
///
/// A node of level zero is a leaf, whose children are the structures the tree
/// indexes. A node above it has other nodes as children.
///
/// Both sibling addresses are written as the undefined address, as in a tree one node wide per
/// level. [`node_with_siblings`] writes a node with its sibling addresses.
#[track_caller]
pub fn node(
    node_type: NodeType,
    level: u8,
    keys: &[Vec<u8>],
    children: &[u64],
    widths: Widths,
) -> Vec<u8> {
    node_with_siblings(node_type, level, keys, children, None, None, widths)
}

/// The bytes of one node, as [`node`] builds them, with the addresses of its left and right
/// siblings at its level.
///
/// A sibling of `None` is written as the undefined address, the value "Version 1 B-trees" of the
/// [format specification, version 4.0][spec] defines for the left sibling of the leftmost node at
/// a level and the right sibling of the rightmost one.
///
/// # Panics
///
/// Panics if `keys` does not hold one more key than `children` has addresses.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
#[track_caller]
pub fn node_with_siblings(
    node_type: NodeType,
    level: u8,
    keys: &[Vec<u8>],
    children: &[u64],
    left_sibling: Option<u64>,
    right_sibling: Option<u64>,
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
    for sibling in [left_sibling, right_sibling] {
        match sibling {
            Some(address) => bytes::push_uint(&mut node, address, widths.offset),
            None => bytes::push_undefined_address(&mut node, widths.offset),
        }
    }
    for (key, &child) in keys.iter().zip(children) {
        node.extend_from_slice(key);
        bytes::push_uint(&mut node, child, widths.offset);
    }
    node.extend_from_slice(keys.last().expect("one more key than children"));
    node
}

/// The bytes of one node of `node_type` whose keys are group keys, the offsets of link names in the
/// group's local heap.
#[track_caller]
pub fn node_with_group_keys(
    node_type: NodeType,
    level: u8,
    link_name_offsets: &[u64],
    children: &[u64],
    widths: Widths,
) -> Vec<u8> {
    let keys: Vec<_> = link_name_offsets
        .iter()
        .map(|&offset| group_key(offset, widths))
        .collect();
    node(node_type, level, &keys, children, widths)
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

#[derive(Clone, Copy)]
enum TreeKind {
    Group,
    Chunk,
}

const NODE_STRIDE: u64 = 128;
const PAYLOAD_BASE: u64 = 0x10_0000;
