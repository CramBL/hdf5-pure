//! The version 1 B-tree node: the header layout both node types share, and the parser of a
//! group's node, node type 0.

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes::read_length;
use crate::bytes::read_offset;
use crate::bytes::read_optional_offset;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;

/// The size in bytes of the prefix of a version 1 B-tree node that does not depend on the
/// offset width: the signature (4), the node type (1), the node level (1), and the entries
/// used (2). The two sibling addresses that follow are each `offset_size` bytes wide, see
/// [`btree_v1_node_header_size`]. The node is defined in "Version 1 B-trees" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
pub const BTREE_V1_NODE_PREFIX_LEN: usize = 8;

/// Returns the size in bytes of a version 1 B-tree node header, the offset of the node's first key.
///
/// The header is the [`BTREE_V1_NODE_PREFIX_LEN`] bytes of the prefix and the left and right
/// sibling addresses, `offset_size` bytes each.
pub const fn btree_v1_node_header_size(offset_size: u8) -> usize {
    BTREE_V1_NODE_PREFIX_LEN + (offset_size as usize) * 2
}

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
    /// The `entries_used + 1` keys, each the offset in the group's local heap of the first link
    /// name in the subtree the key describes.
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
        // signature(4) + node_type(1) + node_level(1) + entries_used(2),
        // then left_sibling(offset_size) + right_sibling(offset_size).
        let os = offset_size as usize;
        let ls = length_size as usize;
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
        let left_sibling =
            read_optional_offset(file_data, pos, offset_size)?.map(StoredAddress::new);
        pos += os;
        let right_sibling =
            read_optional_offset(file_data, pos, offset_size)?.map(StoredAddress::new);
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
            let key = read_length(file_data, pos, length_size)?;
            keys.push(key);
            pos += key_size;
            // child[i]
            children.push(StoredAddress::new(read_offset(
                file_data,
                pos,
                offset_size,
            )?));
            pos += os;
        }
        // final key
        let key = read_length(file_data, pos, length_size)?;
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
    /// Reads the [`BTREE_V1_NODE_PREFIX_LEN`] bytes of the prefix for the number of entries in use,
    /// then the node up to its last key in use, and parses it as [`parse`](Self::parse) does.
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
        let prefix = source.read_metadata_at(address, BTREE_V1_NODE_PREFIX_LEN)?;
        if &prefix[0..4] != b"TREE" {
            return Err(FormatError::InvalidBTreeSignature);
        }
        let entries_used = u16::from_le_bytes([prefix[6], prefix[7]]) as usize;

        // For type 0: header + entries_used*(key + child) + a trailing key,
        // with child `offset_size` bytes wide and key `length_size` bytes wide.
        let os = offset_size as usize;
        let ls = length_size as usize;
        let total = btree_v1_node_header_size(offset_size) + entries_used * (ls + os) + ls;
        let buf = source.read_metadata_at(address, total)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

#[cfg(test)]
mod tests {
    use test_util::btree_v1;
    use test_util::widths::Widths;

    use super::*;

    #[test]
    fn parse_leaf_node() {
        let data = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            0,
            &[0, 5, 10],
            &[0x100, 0x200],
            WIDTHS,
        );
        let node = BTreeV1Node::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(node.node_type, 0);
        assert_eq!(node.node_level, 0);
        assert_eq!(node.entries_used, 2);
        assert_eq!(node.keys, vec![0, 5, 10]);
        assert_eq!(
            node.children,
            vec![StoredAddress::new(0x100), StoredAddress::new(0x200)]
        );
        assert_eq!(node.left_sibling, None);
        assert_eq!(node.right_sibling, None);
    }

    #[test]
    fn parse_with_siblings_none() {
        let data =
            btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 8], &[0x300], WIDTHS);
        let node = BTreeV1Node::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(node.left_sibling, None);
        assert_eq!(node.right_sibling, None);
    }

    #[test]
    fn parse_leaf_node_differing_offset_and_length_sizes() {
        let data = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            0,
            &[0x10, 0x20],
            &[0x300],
            Widths::new(4, 8),
        );
        let node = BTreeV1Node::parse(&data, 0, 4, 8).unwrap();
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.keys, vec![0x10, 0x20]);
        assert_eq!(node.children, vec![StoredAddress::new(0x300)]);
    }

    #[test]
    fn invalid_signature() {
        let mut data =
            btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 1], &[0x100], WIDTHS);
        data[0] = b'X';
        let err = BTreeV1Node::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeSignature);
    }

    #[test]
    fn parse_4byte_offsets() {
        let data = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            0,
            &[0, 4],
            &[0x50],
            Widths::new(4, 4),
        );
        let node = BTreeV1Node::parse(&data, 0, 4, 4).unwrap();
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children, vec![StoredAddress::new(0x50)]);
    }

    const WIDTHS: Widths = Widths::EIGHT;

    #[test]
    fn a_group_node_parses_from_bytes_and_from_a_source() {
        let widths = Widths::new(4, 8);
        let keys = [
            btree_v1::group_key(0, widths),
            btree_v1::group_key(6, widths),
        ];
        let node = btree_v1::node(btree_v1::NodeType::GROUP, 0, &keys, &[0x300], widths);
        let expected = BTreeV1Node {
            node_type: 0,
            node_level: 0,
            entries_used: 1,
            left_sibling: None,
            right_sibling: None,
            keys: vec![0, 6],
            children: vec![StoredAddress::new(0x300)],
        };

        assert_eq!(btree_v1_node_header_size(4), 16);
        assert_eq!(BTreeV1Node::parse(&node, 0, 4, 8), Ok(expected.clone()));
        assert_eq!(
            BTreeV1Node::parse_from_source(node.as_slice(), 0, 4, 8),
            Ok(expected)
        );
    }
}
