use alloc::vec::Vec;

use super::header::{self, BTREE_V1_NODE_PREFIX_LEN, NodePrefix};
use crate::address::{StoredAddress, StoredAddressExt};
use crate::bytes;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;
use crate::width::FormatWidths;

/// A version 1 B-tree node, parsed with the keys of node type 0.
///
/// A group that stores its links in a symbol table indexes its symbol table nodes with a tree of
/// node type 0. A leaf, at level 0, points to symbol table nodes, and a node at a higher level to
/// nodes one level down.
///
/// The node is defined in "Version 1 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeV1Node {
    /// What the tree indexes: 0 for the symbol table nodes of a group, and 1 for the chunks of a
    /// dataset.
    pub node_type: u8,
    /// The level of the node in the tree, 0 for a leaf.
    pub node_level: u8,
    /// The number of children the node points to.
    pub entries_used: u16,
    /// Address of the left sibling, or `None` where the node stores the undefined address.
    pub left_sibling: Option<StoredAddress>,
    /// Address of the right sibling, or `None` where the node stores the undefined address.
    pub right_sibling: Option<StoredAddress>,
    /// The `entries_used + 1` local heap offsets that delimit the child name ranges.
    /// For child `i`, key `i + 1` points to the greatest link name in its subtree.
    pub keys: Vec<u64>,
    /// The `entries_used` child addresses: symbol table nodes where the node is a leaf, and B-tree
    /// nodes one level down where it is not.
    pub children: Vec<StoredAddress>,
}

impl BTreeV1Node {
    /// Parses the version 1 B-tree node at `offset` in `file_data`.
    ///
    /// Reads each key as a type 0 key, a length of `length_size` bytes, whatever node type the node
    /// stores. A caller checks [`node_type`](Self::node_type).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the node header or the keys and children in use
    /// run past the end of `file_data`, [`FormatError::InvalidBTreeSignature`] if the node does
    /// not begin with `TREE`, and [`FormatError::InvalidOffsetSize`] or
    /// [`FormatError::InvalidLengthSize`] if a width is not 2, 4, or 8.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV1Node, FormatError> {
        Self::parse_with_widths(
            file_data,
            offset,
            FormatWidths::from_sizes(offset_size, length_size)?,
        )
    }

    /// Parses the version 1 B-tree node at `offset` with validated file widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the node header or the keys and children in use
    /// run past the end of `file_data`, or [`FormatError::InvalidBTreeSignature`] if the node does
    /// not begin with `TREE`.
    pub fn parse_with_widths(
        file_data: &[u8],
        offset: usize,
        widths: FormatWidths,
    ) -> Result<BTreeV1Node, FormatError> {
        // signature(4) + node_type(1) + node_level(1) + entries_used(2),
        // then left_sibling(offset_width) + right_sibling(offset_width).
        let os = usize::from(widths.offsets.get());
        let ls = usize::from(widths.lengths.get());
        let header_size = header::btree_v1_node_header_size(widths.offsets.get());
        if header_size > file_data.len() || offset > file_data.len() - header_size {
            return Err(FormatError::UnexpectedEof {
                expected: offset.saturating_add(header_size),
                available: file_data.len(),
            });
        }

        let NodePrefix {
            node_type,
            node_level,
            entries_used,
        } = NodePrefix::parse(&file_data[offset..offset + header_size])?;

        let mut pos = offset + BTREE_V1_NODE_PREFIX_LEN;
        let left_sibling = bytes::read_optional_offset_width(file_data, pos, widths.offsets)?
            .map(StoredAddress::new);
        pos += os;
        let right_sibling = bytes::read_optional_offset_width(file_data, pos, widths.offsets)?
            .map(StoredAddress::new);
        pos += os;

        // For type 0: keys are `length_size` bytes, children are `offset_size` bytes
        // Layout: key[0], child[0], key[1], child[1], ..., key[N-1], child[N-1], key[N]
        let eu = entries_used as usize;
        let key_size = ls; // For type 0, key = `length_size`
        let needed = eu * (key_size + os) + key_size; // `eu` children and `eu + 1` keys
        if needed > file_data.len() || pos > file_data.len() - needed {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(needed),
                available: file_data.len(),
            });
        }

        let mut keys = Vec::with_capacity(eu + 1);
        let mut children = Vec::with_capacity(eu);

        for _ in 0..eu {
            // key[i]
            let key = bytes::read_length_width(file_data, pos, widths.lengths)?;
            keys.push(key);
            pos += key_size;
            // child[i]
            children.push(StoredAddress::new(bytes::read_offset_width(
                file_data,
                pos,
                widths.offsets,
            )?));
            pos += os;
        }
        // final key
        let key = bytes::read_length_width(file_data, pos, widths.lengths)?;
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

    /// Parses the version 1 B-tree node at `address` in `source`.
    ///
    /// Reads the complete header, including both sibling addresses, before checking the signature.
    /// Then reads the node through its last key in use and parses it as [`parse`](Self::parse) does.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
    pub fn parse_from_source(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV1Node, FormatError> {
        Self::parse_from_source_with_widths(
            source,
            address,
            FormatWidths::from_sizes(offset_size, length_size)?,
        )
    }

    /// Parses the version 1 B-tree node at `address` with validated file widths.
    ///
    /// Reads the complete header before checking the signature, then the node through its last
    /// key in use.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse_with_widths`](Self::parse_with_widths) returns, and the error
    /// `source` returns if a read fails.
    pub fn parse_from_source_with_widths(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        widths: FormatWidths,
    ) -> Result<BTreeV1Node, FormatError> {
        let header_size = header::btree_v1_node_header_size(widths.offsets.get());
        let header = source.read_metadata(address, header_size)?;
        let entries_used = usize::from(NodePrefix::parse(&header)?.entries_used);

        // For type 0: header + entries_used*(key + child) + a trailing key,
        // with child `offset_size` bytes wide and key `length_size` bytes wide.
        let os = usize::from(widths.offsets.get());
        let ls = usize::from(widths.lengths.get());
        let total = header_size + entries_used * (ls + os) + ls;
        let buf = source.read_metadata(address, total)?;
        Self::parse_with_widths(&buf, 0, widths)
    }
}

#[cfg(test)]
#[path = "tests/group.rs"]
mod tests;
