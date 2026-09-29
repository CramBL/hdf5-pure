//! The version 1 B-tree node: the header layout both node types share, and the parsers of a
//! group's node, node type 0, and of a chunk node, node type 1.

use core::ops::Range;

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;
use crate::width::OffsetWidth;

/// The size in bytes of the prefix of a version 1 B-tree node that does not depend on the
/// offset width: the signature (4), the node type (1), the node level (1), and the entries
/// used (2). The two sibling addresses that follow are each `offset_size` bytes wide, see
/// [`btree_v1_node_header_size`]. The node is defined in "Version 1 B-trees" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
pub(crate) const BTREE_V1_NODE_PREFIX_LEN: usize = 8;

/// Returns the size in bytes of a version 1 B-tree node header, the offset of the node's first key.
///
/// The header is the 8-byte prefix and the left and right sibling addresses, `offset_size` bytes
/// each.
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

        if &file_data[offset..offset + 4] != BTREE_V1_SIGNATURE {
            return Err(FormatError::InvalidBTreeSignature);
        }

        let node_type = file_data[offset + 4];
        let node_level = file_data[offset + 5];
        let entries_used = u16::from_le_bytes([file_data[offset + 6], file_data[offset + 7]]);

        let mut pos = offset + BTREE_V1_NODE_PREFIX_LEN;
        let left_sibling =
            bytes::read_optional_offset(file_data, pos, offset_size)?.map(StoredAddress::new);
        pos += os;
        let right_sibling =
            bytes::read_optional_offset(file_data, pos, offset_size)?.map(StoredAddress::new);
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
            let key = bytes::read_length(file_data, pos, length_size)?;
            keys.push(key);
            pos += key_size;
            // child[i]
            children.push(StoredAddress::new(bytes::read_offset(
                file_data,
                pos,
                offset_size,
            )?));
            pos += os;
        }
        // final key
        let key = bytes::read_length(file_data, pos, length_size)?;
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
    /// Reads the 8-byte prefix for the number of entries in use, then the node up to its last key
    /// in use, and parses it as [`parse`](Self::parse) does.
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
        if &prefix[0..4] != BTREE_V1_SIGNATURE {
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

/// The bytes of a version 1 B-tree chunk node after its 8-byte prefix, read from a
/// [`MetadataSource`].
///
/// [`node`](Self::node) returns a [`BTreeV1ChunkNode`] that borrows them.
#[derive(Clone, Debug)]
pub struct BTreeV1ChunkNodeBytes {
    layout: ChunkNodeLayout,
    entries: Range<usize>,
    body: Vec<u8>,
}

impl BTreeV1ChunkNodeBytes {
    /// Reads the chunk node at `address` in `source`.
    ///
    /// Reads the 8-byte prefix for the number of entries in use, then the rest of the node.
    ///
    /// # Errors
    ///
    /// Returns the errors [`BTreeV1ChunkNode::parse`] returns, and the error `source` returns if
    /// a read fails.
    pub fn read_from_source(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        offset_size: u8,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        let header = source.read_metadata_at(address, BTREE_V1_NODE_PREFIX_LEN)?;
        let prefix = header.first_chunk().ok_or(FormatError::UnexpectedEof {
            expected: BTREE_V1_NODE_PREFIX_LEN,
            available: header.len(),
        })?;
        let layout = ChunkNodeLayout::parse(prefix, offset_size, ndims)?;
        let body_address = address
            .checked_add(BTREE_V1_NODE_PREFIX_LEN.to_u64())
            .ok_or(FormatError::OffsetOverflow {
                offset: address,
                length: BTREE_V1_NODE_PREFIX_LEN.to_u64(),
            })?;
        let body = source.read_metadata_at(body_address, layout.body_len())?;
        let entries = layout.entries();
        if body.len() < entries.end {
            return Err(FormatError::UnexpectedEof {
                expected: layout.node_len,
                available: BTREE_V1_NODE_PREFIX_LEN + body.len(),
            });
        }
        Ok(Self {
            layout,
            entries,
            body,
        })
    }

    /// Returns the node the bytes hold.
    pub fn node(&self) -> BTreeV1ChunkNode<'_> {
        BTreeV1ChunkNode {
            layout: self.layout,
            // `read_from_source`, the only constructor, checked that `body` holds `entries`.
            entries: &self.body[self.entries.clone()],
        }
    }
}

/// A version 1 B-tree node of node type 1, which indexes the chunks of a dataset.
///
/// A leaf, at level 0, points to chunks, and a node at a higher level to nodes one level down. The
/// node borrows its bytes, and the iterators of [`entries`](Self::entries) and
/// [`children`](Self::children) parse each key and child address as the caller advances them.
///
/// The node is defined in "Version 1 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
#[derive(Clone, Copy, Debug)]
pub struct BTreeV1ChunkNode<'a> {
    layout: ChunkNodeLayout,
    entries: &'a [u8],
}

impl<'a> BTreeV1ChunkNode<'a> {
    /// Parses the chunk node at `offset` in `file_data`, whose keys hold `ndims` offsets each.
    ///
    /// `ndims` is the rank of the dataset plus one, for the offset in the datatype that each key
    /// stores after the chunk's offsets.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the node runs past the end of `file_data`,
    /// [`FormatError::InvalidBTreeSignature`] if the node does not begin with `TREE`,
    /// [`FormatError::InvalidBTreeNodeType`] if its node type is not 1,
    /// [`FormatError::OffsetOverflow`] if the length of the node does not fit a `usize`, and
    /// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8.
    pub fn parse(
        file_data: &'a [u8],
        offset: usize,
        offset_size: u8,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        let rest = file_data.get(offset..).unwrap_or_default();
        let eof = |len: usize| FormatError::UnexpectedEof {
            expected: offset.saturating_add(len),
            available: file_data.len(),
        };
        let prefix = rest
            .first_chunk()
            .ok_or_else(|| eof(btree_v1_node_header_size(offset_size)))?;
        let layout = ChunkNodeLayout::parse(prefix, offset_size, ndims)?;
        let entries = rest
            .get(BTREE_V1_NODE_PREFIX_LEN..)
            .and_then(|body| body.get(layout.entries()))
            .filter(|_| rest.len() >= layout.node_len)
            .ok_or_else(|| eof(layout.node_len))?;
        Ok(Self { layout, entries })
    }

    /// Returns the level of the node in the tree, 0 for a leaf.
    pub fn node_level(&self) -> u8 {
        self.layout.node_level
    }

    /// Returns the number of children the node points to.
    pub fn entries_used(&self) -> u16 {
        self.layout.entries_used
    }

    /// Returns the length of the node in bytes: the header, the keys and children in use, and the
    /// key after the last child.
    pub fn stored_len(&self) -> usize {
        self.layout.node_len
    }

    /// Returns each child address with the key before it.
    ///
    /// Key `i` describes the first chunk under child `i`, and the key after the last child is not
    /// returned.
    ///
    /// An item is an error only where its entry is shorter than a key and an address, which
    /// [`parse`](Self::parse) and [`BTreeV1ChunkNodeBytes::read_from_source`] rule out when they
    /// check the length of the node.
    pub fn entries(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<(BTreeV1ChunkKey<'a>, StoredAddress), FormatError>> + 'a
    {
        let layout = self.layout;
        self.entries
            .chunks_exact(layout.entry_len())
            .map(move |entry| layout.split_entry(entry))
    }

    /// Returns the child addresses of the node.
    ///
    /// An item is an error where the item of [`entries`](Self::entries) is.
    pub fn children(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<StoredAddress, FormatError>> + 'a {
        self.entries().map(|entry| entry.map(|(_, child)| child))
    }
}

/// The level and the number of entries of a chunk node, and the sizes of its parts.
#[derive(Clone, Copy, Debug)]
struct ChunkNodeLayout {
    node_level: u8,
    entries_used: u16,
    key_size: usize,
    offset_width: OffsetWidth,
    /// The length of the node in bytes, the prefix included.
    node_len: usize,
}

impl ChunkNodeLayout {
    /// Parses the prefix of a chunk node whose keys hold `ndims` offsets each.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeSignature`] if the prefix does not begin with `TREE`,
    /// [`FormatError::InvalidBTreeNodeType`] if its node type is not 1,
    /// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and
    /// [`FormatError::OffsetOverflow`] if the length of the node does not fit a `usize`.
    fn parse(
        prefix: &[u8; BTREE_V1_NODE_PREFIX_LEN],
        offset_size: u8,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        let [s0, s1, s2, s3, node_type, node_level, used0, used1] = *prefix;
        if [s0, s1, s2, s3] != *BTREE_V1_SIGNATURE {
            return Err(FormatError::InvalidBTreeSignature);
        }
        if node_type != BTREE_V1_CHUNK_NODE_TYPE {
            return Err(FormatError::InvalidBTreeNodeType(node_type));
        }
        let offset_width = OffsetWidth::try_from(offset_size)?;
        let entries_used = u16::from_le_bytes([used0, used1]);
        let key_size = chunk_key_size(ndims)?;
        Ok(Self {
            node_level,
            entries_used,
            key_size,
            offset_width,
            node_len: btree_v1_chunk_node_len(key_size, offset_width, entries_used)?,
        })
    }

    /// Returns the length of the node after its 8-byte prefix.
    fn body_len(&self) -> usize {
        // `node_len` counts the prefix.
        self.node_len - BTREE_V1_NODE_PREFIX_LEN
    }

    /// Returns the range of the keys and children in use in the bytes after the prefix, between
    /// the sibling addresses and the key after the last child.
    fn entries(&self) -> Range<usize> {
        // `node_len` counts the prefix, both siblings and the trailing key, so neither bound
        // underflows.
        2 * usize::from(self.offset_width.get())..self.body_len() - self.key_size
    }

    /// Returns the length of a key and the child address after it.
    fn entry_len(&self) -> usize {
        self.key_size + usize::from(self.offset_width.get())
    }

    /// Splits `entry` into its key and its child address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `entry` is shorter than a key, and
    /// [`FormatError::UnexpectedEof`] if it is shorter than a key and an address.
    fn split_entry<'a>(
        &self,
        entry: &'a [u8],
    ) -> Result<(BTreeV1ChunkKey<'a>, StoredAddress), FormatError> {
        let internal =
            || FormatError::Internal("a chunk node entry is shorter than its key".into());
        let (key, child) = entry.split_at_checked(self.key_size).ok_or_else(internal)?;
        let (chunk_size, key) = key.split_first_chunk().ok_or_else(internal)?;
        let (filter_mask, offsets) = key.split_first_chunk().ok_or_else(internal)?;
        let address = bytes::read_offset_width(child, 0, self.offset_width)?;
        Ok((
            BTreeV1ChunkKey {
                chunk_size: u32::from_le_bytes(*chunk_size),
                filter_mask: u32::from_le_bytes(*filter_mask),
                offsets: offsets.as_chunks::<8>().0,
            },
            StoredAddress::new(address),
        ))
    }
}

/// The key of a version 1 B-tree chunk node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BTreeV1ChunkKey<'a> {
    /// The size of the chunk in the file in bytes.
    pub chunk_size: u32,
    /// The filters skipped for the chunk: bit `i` is set where filter `i` of the pipeline was not
    /// applied.
    pub filter_mask: u32,
    offsets: &'a [[u8; 8]],
}

impl<'a> BTreeV1ChunkKey<'a> {
    /// Returns the offset of the chunk in each dimension of the dataset, in elements, then the
    /// offset in the datatype, which is 0.
    pub fn offsets(&self) -> impl ExactSizeIterator<Item = u64> + 'a {
        self.offsets
            .iter()
            .map(|&offset| u64::from_le_bytes(offset))
    }
}

/// Returns the length in bytes of a chunk node of `entries_used` children whose keys are
/// `key_size` bytes: the header, the keys and children in use, and the key after the last child.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the length does not fit a `usize`.
fn btree_v1_chunk_node_len(
    key_size: usize,
    offset_width: OffsetWidth,
    entries_used: u16,
) -> Result<usize, FormatError> {
    let entry_len = key_size + usize::from(offset_width.get());
    usize::from(entries_used)
        .checked_mul(entry_len)
        .and_then(|entries| entries.checked_add(key_size))
        .and_then(|body| body.checked_add(btree_v1_node_header_size(offset_width.get())))
        .ok_or(FormatError::OffsetOverflow {
            offset: u64::from(entries_used),
            length: entry_len.to_u64(),
        })
}

/// Returns the size in bytes of a key in a version 1 B-tree of type 1 (raw data chunks): the chunk
/// byte size (4), the filter mask (4), and `ndims` 64-bit offsets (8 bytes each). The key is
/// defined in "Version 1 B-trees" of the [format specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the size does not fit a `usize`.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
fn chunk_key_size(ndims: usize) -> Result<usize, FormatError> {
    ndims
        .checked_mul(8)
        .and_then(|offsets| offsets.checked_add(4 + 4))
        .ok_or(FormatError::OffsetOverflow {
            offset: ndims.to_u64(),
            length: 8,
        })
}

/// The signature of a version 1 B-tree node, defined in "Version 1 B-trees" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
const BTREE_V1_SIGNATURE: &[u8; 4] = b"TREE";

/// The node type of a version 1 B-tree that indexes the chunks of a dataset.
const BTREE_V1_CHUNK_NODE_TYPE: u8 = 1;

#[cfg(test)]
mod tests {
    use core::cell::RefCell;

    use rstest::rstest;
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

    /// Returns the size, the filter mask, and the offsets of each key of `node`, with the child
    /// after it.
    fn entries(node: BTreeV1ChunkNode<'_>) -> Vec<(u32, u32, Vec<u64>, StoredAddress)> {
        node.entries()
            .map(|entry| {
                let (key, child) = entry.unwrap();
                (
                    key.chunk_size,
                    key.filter_mask,
                    key.offsets().collect(),
                    child,
                )
            })
            .collect()
    }

    /// A source over `bytes` that records the offset and the length of each read.
    struct RecordingSource<'a> {
        bytes: &'a [u8],
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl MetadataSource for RecordingSource<'_> {
        fn len(&self) -> u64 {
            MetadataSource::len(self.bytes)
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push((offset, buf.len()));
            self.bytes.read_at(offset, buf)
        }

        fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            self.reads.borrow_mut().push((offset, len));
            self.bytes.read_metadata_at(offset, len)
        }
    }

    #[rstest]
    #[case::a_leaf(0, Widths::EIGHT)]
    #[case::a_leaf_with_4_byte_addresses(0, Widths::new(4, 8))]
    #[case::an_internal_node(1, Widths::EIGHT)]
    fn a_chunk_node_parses_from_bytes_and_from_a_source(#[case] level: u8, #[case] widths: Widths) {
        let keys = [
            btree_v1::chunk_key(160, 0, &[0, 0]),
            btree_v1::chunk_key(96, 2, &[20, 0]),
            btree_v1::chunk_key(0, 0, &[40, 0]),
        ];
        let node = btree_v1::node(
            btree_v1::NodeType::CHUNK,
            level,
            &keys,
            &[0x1000, 0x2000],
            widths,
        );
        let offset_size = u8::try_from(widths.offset).unwrap();
        let expected = vec![
            (160, 0, vec![0, 0], StoredAddress::new(0x1000)),
            (96, 2, vec![20, 0], StoredAddress::new(0x2000)),
        ];

        let parsed = BTreeV1ChunkNode::parse(&node, 0, offset_size, 2).unwrap();
        assert_eq!(
            (
                parsed.node_level(),
                parsed.entries_used(),
                parsed.stored_len()
            ),
            (level, 2, node.len())
        );
        assert_eq!(entries(parsed), expected);
        assert_eq!(
            parsed.children().map(Result::unwrap).collect::<Vec<_>>(),
            vec![StoredAddress::new(0x1000), StoredAddress::new(0x2000)]
        );

        let source = RecordingSource {
            bytes: &node,
            reads: RefCell::new(Vec::new()),
        };
        let read = BTreeV1ChunkNodeBytes::read_from_source(&source, 0, offset_size, 2).unwrap();
        let read = read.node();
        assert_eq!(
            (read.node_level(), read.entries_used(), read.stored_len()),
            (level, 2, node.len())
        );
        assert_eq!(entries(read), expected);
        assert_eq!(
            source.reads.into_inner(),
            vec![
                (0, BTREE_V1_NODE_PREFIX_LEN),
                (8, node.len() - BTREE_V1_NODE_PREFIX_LEN)
            ],
            "the node is read once, the prefix and then the rest"
        );
    }

    #[rstest]
    #[case::a_group_node(
        4,
        btree_v1::NodeType::GROUP.0,
        FormatError::InvalidBTreeNodeType(btree_v1::NodeType::GROUP.0)
    )]
    #[case::a_corrupted_signature(0, b'X', FormatError::InvalidBTreeSignature)]
    fn a_node_of_another_type_fails_to_parse(
        #[case] offset: usize,
        #[case] byte: u8,
        #[case] expected: FormatError,
    ) {
        let keys = [
            btree_v1::chunk_key(160, 0, &[0, 0]),
            btree_v1::chunk_key(0, 0, &[20, 0]),
        ];
        let mut node = btree_v1::node(btree_v1::NodeType::CHUNK, 0, &keys, &[0x1000], WIDTHS);
        node[offset] = byte;
        assert_eq!(
            BTreeV1ChunkNode::parse(&node, 0, 8, 2).unwrap_err(),
            expected
        );
        assert_eq!(
            BTreeV1ChunkNodeBytes::read_from_source(node.as_slice(), 0, 8, 2).unwrap_err(),
            expected
        );
    }

    #[test]
    fn a_chunk_node_cut_short_fails_to_parse() {
        let keys = [
            btree_v1::chunk_key(160, 0, &[0, 0]),
            btree_v1::chunk_key(0, 0, &[20, 0]),
        ];
        let mut node = btree_v1::node(btree_v1::NodeType::CHUNK, 0, &keys, &[0x1000], WIDTHS);
        let full = node.len();
        node.pop();
        let expected = FormatError::UnexpectedEof {
            expected: full,
            available: full - 1,
        };
        assert_eq!(
            BTreeV1ChunkNode::parse(&node, 0, 8, 2).unwrap_err(),
            expected
        );
        assert_eq!(
            BTreeV1ChunkNodeBytes::read_from_source(node.as_slice(), 0, 8, 2).unwrap_err(),
            expected
        );
    }
}
