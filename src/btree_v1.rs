//! The walk of a group's version 1 B-tree to its symbol table nodes.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::BTreeV1Node;

use crate::address::BaseAddressExt;
use crate::address::{BaseAddress, StoredAddress};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;
use crate::source::SourceMetadata;

/// Recursion-depth cap for the symbol-table (group) B-tree walk, guarding
/// against a stack overflow on a cyclic or pathological internal node in a
/// foreign file. A real group B-tree is only a few levels deep, so this is far
/// beyond any valid tree. Mirrors `MAX_CHUNK_BTREE_DEPTH` in `chunked_read`.
const MAX_SYMBOL_TABLE_BTREE_DEPTH: u32 = 64;

/// Returns the address of every symbol table node the B-tree at `btree_address` leads to.
///
/// The leaves of a group's B-tree name the symbol table nodes that hold its entries, and the walk
/// descends to them through the internal nodes.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if a node on the walk is not a group node,
/// [`FormatError::NestingDepthExceeded`] if the walk descends past
/// [`MAX_SYMBOL_TABLE_BTREE_DEPTH`] levels, and the [`FormatError`] of the first node that does
/// not parse.
pub fn collect_symbol_table_nodes(
    file_data: &[u8],
    btree_address: StoredAddress,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<StoredAddress>, FormatError> {
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
    btree_address: StoredAddress,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    depth: u32,
) -> Result<Vec<StoredAddress>, FormatError> {
    if depth > MAX_SYMBOL_TABLE_BTREE_DEPTH {
        return Err(FormatError::NestingDepthExceeded);
    }
    let node_offset = base_address.absolute(btree_address)?.to_usize()?;
    let node = BTreeV1Node::parse(file_data, node_offset, offset_size, length_size)?;

    if node.node_type != 0 {
        return Err(FormatError::InvalidBTreeNodeType(node.node_type));
    }

    if node.node_level == 0 {
        // A leaf's children are the symbol table nodes themselves.
        Ok(node.children)
    } else {
        // An internal node's children are B-tree nodes one level down.
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
/// # Errors
///
/// Returns the errors of [`collect_symbol_table_nodes`], and the error `source` reports for a node
/// it cannot read.
pub fn collect_symbol_table_nodes_from_source<S: Source + ?Sized>(
    source: &S,
    btree_address: StoredAddress,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<StoredAddress>, FormatError> {
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
    btree_address: StoredAddress,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    depth: u32,
) -> Result<Vec<StoredAddress>, FormatError> {
    if depth > MAX_SYMBOL_TABLE_BTREE_DEPTH {
        return Err(FormatError::NestingDepthExceeded);
    }
    let node_offset = base_address.absolute(btree_address)?;
    let node = BTreeV1Node::parse_from_source(
        &SourceMetadata(source),
        node_offset,
        offset_size,
        length_size,
    )?;

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
    use test_util::btree_v1;
    use test_util::widths::Widths;

    use super::*;

    #[test]
    fn parse_internal_node_and_collect() {
        // Build a 2-level tree: one internal node pointing to two leaf nodes
        let os: u8 = 8;
        let ls: u8 = 8;
        let leaf1_offset: usize = 0;
        let leaf2_offset: usize = 256;
        let internal_offset: usize = 512;

        let leaf1 =
            btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 5], &[0xA00], WIDTHS);
        let leaf2 = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            0,
            &[5, 10],
            &[0xB00],
            WIDTHS,
        );
        let internal = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            1,
            &[0, 5, 10],
            &[leaf1_offset as u64, leaf2_offset as u64],
            WIDTHS,
        );

        let mut file = vec![0u8; 1024];
        file[leaf1_offset..leaf1_offset + leaf1.len()].copy_from_slice(&leaf1);
        file[leaf2_offset..leaf2_offset + leaf2.len()].copy_from_slice(&leaf2);
        file[internal_offset..internal_offset + internal.len()].copy_from_slice(&internal);

        let snods = collect_symbol_table_nodes(
            &file,
            StoredAddress::new(internal_offset as u64),
            os,
            ls,
            BaseAddress::ZERO,
        )
        .unwrap();
        assert_eq!(
            snods,
            vec![StoredAddress::new(0xA00), StoredAddress::new(0xB00)]
        );
    }

    #[test]
    fn collect_wrong_node_type() {
        let data =
            btree_v1::node_with_group_keys(btree_v1::NodeType::CHUNK, 0, &[0, 1], &[0x100], WIDTHS);
        let mut file = vec![0u8; 512];
        file[..data.len()].copy_from_slice(&data);
        let err = collect_symbol_table_nodes(&file, StoredAddress::new(0), 8, 8, BaseAddress::ZERO)
            .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeNodeType(1));
    }

    #[test]
    fn collect_symbol_table_nodes_rejects_cyclic_btree() {
        // An internal node (level 1) whose only child points back at itself.
        // Listing a malicious v1 group must error via the depth guard rather
        // than recurse until the stack overflows (an uncatchable abort).
        let os: u8 = 8;
        let ls: u8 = 8;
        let node =
            btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 1, &[0, 0], &[0], WIDTHS);
        let mut file = vec![0u8; 1024];
        file[..node.len()].copy_from_slice(&node);

        let err =
            collect_symbol_table_nodes(&file, StoredAddress::new(0), os, ls, BaseAddress::ZERO)
                .unwrap_err();
        assert_eq!(err, FormatError::NestingDepthExceeded);
    }

    const WIDTHS: Widths = Widths::EIGHT;
}
