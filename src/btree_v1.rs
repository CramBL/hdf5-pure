//! HDF5 B-tree v1 parsing (type 0 for groups).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::address::BaseAddress;
use crate::bytes::{read_offset, read_optional_offset};
use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::source::Source;

/// The size in bytes of the prefix of a version 1 B-tree node that does not depend on the
/// offset width: the signature (4), the node type (1), the node level (1), and the entries
/// used (2). The two sibling addresses that follow are each `offset_size` bytes wide, see
/// [`btree_v1_node_header_size`]. The node is defined in "Version 1 B-trees" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
pub(crate) const BTREE_V1_NODE_PREFIX_LEN: usize = 8;

/// Total size of a version 1 B-tree node header for a file whose size-of-offsets
/// is `offset_size`: the fixed [`BTREE_V1_NODE_PREFIX_LEN`] prefix plus the left
/// and right sibling addresses (`offset_size` bytes each). This is the offset
/// from the start of a node to its first key.
pub(crate) const fn btree_v1_node_header_size(offset_size: u8) -> usize {
    BTREE_V1_NODE_PREFIX_LEN + (offset_size as usize) * 2
}

/// A parsed B-tree v1 node.
///
/// Some fields are decoded from the on-disk node for completeness but are not
/// consulted by the current traversal (which walks `children`); they are kept
/// to document the format and allow future use.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BTreeV1Node {
    /// Node type: 0=group, 1=raw data chunks.
    pub node_type: u8,
    /// Node level: 0=leaf, >0=internal.
    pub node_level: u8,
    /// Number of entries used.
    pub entries_used: u16,
    /// Left sibling address, or None if undefined.
    pub left_sibling: Option<u64>,
    /// Right sibling address, or None if undefined.
    pub right_sibling: Option<u64>,
    /// Keys (entries_used + 1 values).
    pub keys: Vec<u64>,
    /// Child addresses (entries_used values).
    pub children: Vec<u64>,
}

impl BTreeV1Node {
    /// Parse a B-tree v1 node at the given offset in the file data.
    ///
    /// For type 0 (group) nodes, keys are offset_size bytes each (heap name offsets).
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        _length_size: u8,
    ) -> Result<BTreeV1Node, FormatError> {
        // signature(4) + node_type(1) + node_level(1) + entries_used(2),
        // then left_sibling(offset_size) + right_sibling(offset_size).
        let os = offset_size as usize;
        let header_size = btree_v1_node_header_size(offset_size);
        if header_size > file_data.len() || offset > file_data.len() - header_size {
            return Err(FormatError::UnexpectedEof {
                expected: offset.saturating_add(header_size),
                available: file_data.len(),
            });
        }

        if &file_data[offset..offset + 4] != b"TREE" {
            return Err(FormatError::InvalidBTreeSignature);
        }

        let node_type = file_data[offset + 4];
        let node_level = file_data[offset + 5];
        let entries_used = u16::from_le_bytes([file_data[offset + 6], file_data[offset + 7]]);

        let mut pos = offset + BTREE_V1_NODE_PREFIX_LEN;
        let left_sibling = read_optional_offset(file_data, pos, offset_size)?;
        pos += os;
        let right_sibling = read_optional_offset(file_data, pos, offset_size)?;
        pos += os;

        // For type 0: keys are offset_size bytes, children are offset_size bytes
        // Layout: key[0], child[0], key[1], child[1], ..., key[N-1], child[N-1], key[N]
        let eu = entries_used as usize;
        let key_size = os; // For type 0, key = offset_size
        let needed = eu * (key_size + os) + key_size; // eu children + (eu+1) keys
        if needed > file_data.len() || pos > file_data.len() - needed {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(needed),
                available: file_data.len(),
            });
        }

        let mut keys = Vec::with_capacity(eu + 1);
        let mut children = Vec::with_capacity(eu);

        for i in 0..eu {
            // key[i]
            let key = read_offset(file_data, pos, offset_size)?;
            keys.push(key);
            pos += key_size;
            // child[i]
            let child = read_offset(file_data, pos, offset_size)?;
            children.push(child);
            pos += os;
            let _ = i;
        }
        // final key
        let key = read_offset(file_data, pos, offset_size)?;
        keys.push(key);

        Ok(BTreeV1Node {
            node_type,
            node_level,
            entries_used,
            left_sibling,
            right_sibling,
            keys,
            children,
        })
    }

    /// Parse a B-tree v1 (type 0, group) node from a [`Source`] on demand.
    ///
    /// Reads the fixed node prefix to learn `entries_used`, then the exact node
    /// body, so no more than one node is resident at a time.
    pub fn parse_from_source<S: Source + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV1Node, FormatError> {
        let prefix = source.read_metadata_at(address, BTREE_V1_NODE_PREFIX_LEN)?;
        if &prefix[0..4] != b"TREE" {
            return Err(FormatError::InvalidBTreeSignature);
        }
        let entries_used = u16::from_le_bytes([prefix[6], prefix[7]]) as usize;

        // For type 0: header + entries_used*(key + child) + a trailing key,
        // with key and child each `offset_size` bytes wide.
        let os = offset_size as usize;
        let total = btree_v1_node_header_size(offset_size) + entries_used * (os + os) + os;
        let buf = source.read_metadata_at(address, total)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// Recursion-depth cap for the symbol-table (group) B-tree walk, guarding
/// against a stack overflow on a cyclic or pathological internal node in a
/// foreign file. A real group B-tree is only a few levels deep, so this is far
/// beyond any valid tree. Mirrors `MAX_CHUNK_BTREE_DEPTH` in `chunked_read`.
const MAX_SYMBOL_TABLE_BTREE_DEPTH: u32 = 64;

/// Collect all leaf-level child addresses (SNOD addresses) by traversing the B-tree.
///
/// `base_address` is the superblock base address. All addresses stored in the
/// B-tree are relative to this value. The returned SNOD addresses are also
/// relative (the caller must add `base_address` to obtain absolute file offsets).
pub fn collect_symbol_table_nodes(
    file_data: &[u8],
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<u64>, FormatError> {
    collect_symbol_table_nodes_inner(
        file_data,
        btree_address,
        offset_size,
        length_size,
        base_address,
        0,
    )
}

/// Depth-tracking core of [`collect_symbol_table_nodes`]. The `depth` guard
/// stops a cyclic or pathologically deep internal node from recursing until the
/// stack overflows (an uncatchable process abort) when listing a v1 group.
fn collect_symbol_table_nodes_inner(
    file_data: &[u8],
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    depth: u32,
) -> Result<Vec<u64>, FormatError> {
    if depth > MAX_SYMBOL_TABLE_BTREE_DEPTH {
        return Err(FormatError::NestingDepthExceeded);
    }
    let node_offset = base_address.absolute(btree_address)?.to_usize()?;
    let node = BTreeV1Node::parse(file_data, node_offset, offset_size, length_size)?;

    if node.node_type != 0 {
        return Err(FormatError::InvalidBTreeNodeType(node.node_type));
    }

    if node.node_level == 0 {
        // Leaf: children are SNOD addresses (relative to base_address)
        Ok(node.children)
    } else {
        // Internal: recurse into children (child addresses are relative to base_address)
        let mut result = Vec::new();
        for &child_addr in &node.children {
            let child_snods = collect_symbol_table_nodes_inner(
                file_data,
                child_addr,
                offset_size,
                length_size,
                base_address,
                depth + 1,
            )?;
            result.extend(child_snods);
        }
        Ok(result)
    }
}

/// Streaming counterpart of [`collect_symbol_table_nodes`]: walks the v1 B-tree
/// through a [`Source`], reading one node at a time.
///
/// `base_address` is the superblock base address; all B-tree node and SNOD
/// addresses are stored relative to it. The returned SNOD addresses are also
/// relative (the caller adds `base_address` to obtain absolute file offsets).
pub fn collect_symbol_table_nodes_from_source<S: Source + ?Sized>(
    source: &S,
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<u64>, FormatError> {
    collect_symbol_table_nodes_from_source_inner(
        source,
        btree_address,
        offset_size,
        length_size,
        base_address,
        0,
    )
}

/// Depth-tracking core of [`collect_symbol_table_nodes_from_source`]; see
/// [`collect_symbol_table_nodes_inner`] for why the bound is required.
fn collect_symbol_table_nodes_from_source_inner<S: Source + ?Sized>(
    source: &S,
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    depth: u32,
) -> Result<Vec<u64>, FormatError> {
    if depth > MAX_SYMBOL_TABLE_BTREE_DEPTH {
        return Err(FormatError::NestingDepthExceeded);
    }
    let node_offset = base_address.absolute(btree_address)?;
    let node = BTreeV1Node::parse_from_source(source, node_offset, offset_size, length_size)?;

    if node.node_type != 0 {
        return Err(FormatError::InvalidBTreeNodeType(node.node_type));
    }

    if node.node_level == 0 {
        Ok(node.children)
    } else {
        let mut result = Vec::new();
        for &child_addr in &node.children {
            let child_snods = collect_symbol_table_nodes_from_source_inner(
                source,
                child_addr,
                offset_size,
                length_size,
                base_address,
                depth + 1,
            )?;
            result.extend(child_snods);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_offset(buf: &mut Vec<u8>, val: u64, size: u8) {
        match size {
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => panic!("test"),
        }
    }

    fn build_btree_node(
        node_type: u8,
        level: u8,
        keys: &[u64],
        children: &[u64],
        left: Option<u64>,
        right: Option<u64>,
        offset_size: u8,
    ) -> Vec<u8> {
        assert_eq!(keys.len(), children.len() + 1);
        let entries_used = children.len() as u16;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TREE");
        buf.push(node_type);
        buf.push(level);
        buf.extend_from_slice(&entries_used.to_le_bytes());
        let undef: u64 = if offset_size == 4 {
            0xFFFFFFFF
        } else {
            0xFFFFFFFFFFFFFFFF
        };
        write_offset(&mut buf, left.unwrap_or(undef), offset_size);
        write_offset(&mut buf, right.unwrap_or(undef), offset_size);
        for i in 0..children.len() {
            write_offset(&mut buf, keys[i], offset_size);
            write_offset(&mut buf, children[i], offset_size);
        }
        write_offset(&mut buf, *keys.last().unwrap(), offset_size);
        buf
    }

    #[test]
    fn parse_leaf_node() {
        let data = build_btree_node(0, 0, &[0, 5, 10], &[0x100, 0x200], None, None, 8);
        let node = BTreeV1Node::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(node.node_type, 0);
        assert_eq!(node.node_level, 0);
        assert_eq!(node.entries_used, 2);
        assert_eq!(node.keys, vec![0, 5, 10]);
        assert_eq!(node.children, vec![0x100, 0x200]);
        assert_eq!(node.left_sibling, None);
        assert_eq!(node.right_sibling, None);
    }

    #[test]
    fn parse_with_siblings_none() {
        let data = build_btree_node(0, 0, &[0, 8], &[0x300], None, None, 8);
        let node = BTreeV1Node::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(node.left_sibling, None);
        assert_eq!(node.right_sibling, None);
    }

    #[test]
    fn parse_internal_node_and_collect() {
        // Build a 2-level tree: one internal node pointing to two leaf nodes
        let os: u8 = 8;
        let leaf1_offset: usize = 0;
        let leaf2_offset: usize = 256;
        let internal_offset: usize = 512;

        let leaf1 = build_btree_node(0, 0, &[0, 5], &[0xA00], None, None, os);
        let leaf2 = build_btree_node(0, 0, &[5, 10], &[0xB00], None, None, os);
        let internal = build_btree_node(
            0,
            1,
            &[0, 5, 10],
            &[leaf1_offset as u64, leaf2_offset as u64],
            None,
            None,
            os,
        );

        let mut file = vec![0u8; 1024];
        file[leaf1_offset..leaf1_offset + leaf1.len()].copy_from_slice(&leaf1);
        file[leaf2_offset..leaf2_offset + leaf2.len()].copy_from_slice(&leaf2);
        file[internal_offset..internal_offset + internal.len()].copy_from_slice(&internal);

        let snods =
            collect_symbol_table_nodes(&file, internal_offset as u64, os, os, BaseAddress::ZERO)
                .unwrap();
        assert_eq!(snods, vec![0xA00, 0xB00]);
    }

    #[test]
    fn invalid_signature() {
        let mut data = build_btree_node(0, 0, &[0, 1], &[0x100], None, None, 8);
        data[0] = b'X';
        let err = BTreeV1Node::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeSignature);
    }

    #[test]
    fn collect_wrong_node_type() {
        let data = build_btree_node(1, 0, &[0, 1], &[0x100], None, None, 8);
        let mut file = vec![0u8; 512];
        file[..data.len()].copy_from_slice(&data);
        let err = collect_symbol_table_nodes(&file, 0, 8, 8, BaseAddress::ZERO).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeNodeType(1));
    }

    #[test]
    fn parse_4byte_offsets() {
        let data = build_btree_node(0, 0, &[0, 4], &[0x50], None, None, 4);
        let node = BTreeV1Node::parse(&data, 0, 4, 4).unwrap();
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children, vec![0x50]);
    }

    #[test]
    fn collect_symbol_table_nodes_rejects_cyclic_btree() {
        // An internal node (level 1) whose only child points back at itself.
        // Listing a malicious v1 group must error via the depth guard rather
        // than recurse until the stack overflows (an uncatchable abort).
        let os: u8 = 8;
        let node = build_btree_node(0, 1, &[0, 0], &[0], None, None, os);
        let mut file = vec![0u8; 1024];
        file[..node.len()].copy_from_slice(&node);

        let err = collect_symbol_table_nodes(&file, 0, os, os, BaseAddress::ZERO).unwrap_err();
        assert_eq!(err, FormatError::NestingDepthExceeded);
    }
}
