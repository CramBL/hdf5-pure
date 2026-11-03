//! Version 2 B-trees: section `subsec_fmt4_infra_btrees`, version 4.0.
//!
//! A version 2 tree keeps its shape in a header of its own, and its nodes
//! carry a checksum.

use core::mem;
use core::num::NonZeroU16;

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

/// Rewrites a leaf root and its header with checksum-valid replacement records.
///
/// Uses eight-byte addresses and lengths, preserving the existing record count and node size.
/// The header and leaf layouts are defined in "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// # Panics
///
/// Panics if the image or root address is out of bounds, the root is not a leaf, the record count
/// differs, or `records` is empty. Also panics if record widths differ or exceed [`u16::MAX`], or
/// the encoded node exceeds its allocation.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub fn replace_leaf_records(
    image: &mut [u8],
    header_at: usize,
    tree_type: u8,
    records: &[Vec<u8>],
) {
    let widths = Widths::EIGHT;
    let type_at = SIGNATURE.len() + mem::size_of::<u8>();
    let record_size_at = type_at + mem::size_of::<u8>() + mem::size_of::<u32>();
    let depth_at = record_size_at + mem::size_of::<u16>();
    let root_at = depth_at + mem::size_of::<u16>() + 2 * mem::size_of::<u8>();
    assert_eq!(bytes::u16_at(image, header_at + depth_at), 0);
    let root = usize::try_from(bytes::u64_at(image, header_at + root_at)).unwrap();
    assert_eq!(
        usize::from(bytes::u16_at(image, header_at + root_at + widths.offset)),
        records.len()
    );
    let record_size = u16::try_from(records.first().unwrap().len()).unwrap();
    assert!(
        records
            .iter()
            .all(|record| record.len() == usize::from(record_size))
    );
    bytes::set_u8_at(image, header_at + type_at, tree_type);
    bytes::set_u16_at(image, header_at + record_size_at, record_size);
    let header_len = SIGNATURE.len()
        + 4 * mem::size_of::<u8>()
        + mem::size_of::<u32>()
        + 3 * mem::size_of::<u16>()
        + widths.offset
        + widths.length
        + checksum::CHECKSUM;
    checksum::restamp(image, header_at, header_len);
    let node = leaf(tree_type, records);
    let node_size = usize::try_from(bytes::u32_at(
        image,
        header_at + type_at + mem::size_of::<u8>(),
    ))
    .unwrap();
    assert!(node.len() <= node_size);
    bytes::set_slice_at(image, root, &node);
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

/// The message flags, creation order, and hash from a type 8 attribute index record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttributeNameRecord {
    pub flags: u8,
    pub creation_order: u32,
    pub name_hash: u32,
}

/// Reads the records of a file's sole type 8 index, independently of implementation parsers.
///
/// Requires eight-byte addresses and lengths, eight-byte heap IDs, and a leaf root.
///
/// # Panics
///
/// Panics if the index is absent, ambiguous, truncated, or has a different layout.
pub fn attribute_name_root_records(bytes: &[u8]) -> Vec<AttributeNameRecord> {
    let headers: Vec<_> = bytes
        .windows(SIGNATURE.len())
        .enumerate()
        .filter_map(|(at, signature)| {
            (signature == SIGNATURE && bytes::u8_at(bytes, at + 5) == ATTRIBUTE_NAME_TYPE)
                .then_some(at)
        })
        .collect();
    assert_eq!(headers.len(), 1, "expected one attribute name index");
    let at = headers[0];
    assert_eq!(bytes::u16_at(bytes, at + 10), ATTRIBUTE_NAME_RECORD_SIZE);
    assert_eq!(bytes::u16_at(bytes, at + 12), 0, "expected a leaf root");
    let root = usize::try_from(bytes::u64_at(bytes, at + 16)).unwrap();
    assert_eq!(bytes::slice_at(bytes, root, 4), LEAF_SIGNATURE);
    assert_eq!(bytes::u8_at(bytes, root + 5), ATTRIBUTE_NAME_TYPE);
    let count = usize::from(bytes::u16_at(bytes, at + 24));
    assert_eq!(bytes::u64_at(bytes, at + 26), u64::try_from(count).unwrap());
    (0..count)
        .map(|i| {
            let at = root + NODE_PREFIX + i * usize::from(ATTRIBUTE_NAME_RECORD_SIZE);
            AttributeNameRecord {
                flags: bytes::u8_at(bytes, at + 8),
                creation_order: bytes::u32_at(bytes, at + 9),
                name_hash: bytes::u32_at(bytes, at + 13),
            }
        })
        .collect()
}

/// Writes a type 8 record with the supplied heap ID, message flags, creation order, and hash.
pub fn attribute_name_record(heap_id: &[u8], flags: u8, creation_order: u32, hash: u32) -> Vec<u8> {
    let mut record = heap_id.to_vec();
    record.push(flags);
    record.extend_from_slice(&creation_order.to_le_bytes());
    record.extend_from_slice(&hash.to_le_bytes());
    record
}

/// A tree with one root record above two chains of zero-record nodes.
pub struct Chain {
    /// The header followed by complete 64-byte node allocations.
    pub bytes: Vec<u8>,
    /// The number of nodes, excluding the header.
    pub nodes: usize,
}

impl Chain {
    /// Builds a type-5 tree with `depth` internal levels and one record in its root.
    ///
    /// Uses eight-byte addresses and lengths. Every byte of the root record is 42.
    pub fn new(depth: NonZeroU16) -> Self {
        let depth = depth.get();
        let node_size = 64;
        let widths = Widths::EIGHT;
        let mut image = vec![0; node_size];
        let mut append = |node: Vec<u8>| {
            let address = u64::try_from(image.len()).unwrap();
            assert!(node.len() <= node_size);
            image.extend_from_slice(&node);
            image.resize(image.len() + node_size - node.len(), 0);
            address
        };
        let mut child = append(leaf(LINK_NAME_TYPE, &[]));
        let mut last = append(leaf(LINK_NAME_TYPE, &[]));
        let record_size = usize::from(LINK_NAME_RECORD_SIZE);
        let overhead = NODE_PREFIX + checksum::CHECKSUM;
        let mut capacity = u64::try_from((node_size - overhead) / record_size).unwrap();
        let direct_width =
            usize::try_from((u64::BITS - capacity.leading_zeros()).div_ceil(u8::BITS)).unwrap();
        let mut count_width = direct_width;
        for level in 1..=depth {
            let children = vec![
                Child {
                    address: child,
                    records: 0,
                    records_width: direct_width,
                    subtree: (level > 1).then_some(Subtree {
                        records: 0,
                        width: count_width,
                    }),
                };
                if level == depth { 2 } else { 1 }
            ];
            let mut children = children;
            if level == depth {
                children[1].address = last;
            }
            let records = if level == depth {
                vec![vec![42; record_size]]
            } else {
                Vec::new()
            };
            if level < depth {
                let mut right = children[0];
                right.address = last;
                last = append(internal(LINK_NAME_TYPE, &[], &[right], widths));
            }
            child = append(internal(LINK_NAME_TYPE, &records, &children, widths));
            let pointer_width =
                widths.offset + direct_width + if level == 1 { 0 } else { count_width };
            let direct = u64::try_from(
                (node_size - overhead - pointer_width) / (record_size + pointer_width),
            )
            .unwrap();
            capacity = (direct + 1).saturating_mul(capacity).saturating_add(direct);
            count_width =
                usize::try_from((u64::BITS - capacity.leading_zeros()).div_ceil(u8::BITS)).unwrap();
        }
        let header = Header::new(LINK_NAME_TYPE, LINK_NAME_RECORD_SIZE, child, 1)
            .node_size(u32::try_from(node_size).unwrap())
            .depth(depth)
            .total_records(1)
            .build(widths);
        image[..header.len()].copy_from_slice(&header);
        Self {
            bytes: image,
            nodes: 2 * usize::from(depth) + 1,
        }
    }
}

// "Version 2 B-trees" of the format specification, version 4.0, defines type 8 and its
// 17-byte record.
const ATTRIBUTE_NAME_TYPE: u8 = 8;
const ATTRIBUTE_NAME_RECORD_SIZE: u16 = 17;

/// The client type for a dense group's link-name index.
///
/// Defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub const LINK_NAME_TYPE: u8 = 5;
/// The size of a link-name record's four-byte hash and seven-byte heap ID.
///
/// Defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub const LINK_NAME_RECORD_SIZE: u16 = 11;

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
