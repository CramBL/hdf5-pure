use core::ops::Deref;

use super::header::{self, NodePrefix};
use crate::address::{StoredAddress, StoredAddressExt};
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;
use crate::width::OffsetWidth;

/// The checked layout and body of a version 1 B-tree chunk node read from a [`MetadataSource`].
///
/// Retains the source's byte owner, which may borrow resident bytes or share an immutable
/// allocation. The body begins after the sibling addresses and includes the entries in use and
/// the key after the last child.
/// [`node`](Self::node) returns a [`BTreeV1ChunkNode`] that borrows these bytes.
#[derive(Clone, Debug)]
pub struct BTreeV1ChunkNodeBytes<B> {
    layout: ChunkNodeLayout,
    body: B,
}

impl<B: Deref<Target = [u8]>> BTreeV1ChunkNodeBytes<B> {
    /// Reads the chunk node at `address` in `source`.
    ///
    /// Acquires the complete header and body as described in
    /// [`Self::read_from_source_with_offset_width`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8. Propagates
    /// the errors from [`Self::read_from_source_with_offset_width`].
    pub fn read_from_source<'a>(
        source: &'a (impl MetadataSource<Bytes<'a> = B> + ?Sized),
        address: u64,
        offset_size: u8,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        Self::read_from_source_with_offset_width(
            source,
            address,
            OffsetWidth::try_from(offset_size)?,
            ndims,
        )
    }

    /// Reads the chunk node at `address` in `source` using a validated address width.
    ///
    /// Reads the complete header, including both sibling addresses, before checking the
    /// signature and node type. Then reads through the key after the last child and retains
    /// the body bytes returned by [`MetadataSource::read_metadata`].
    ///
    /// `ndims` is the dataset rank plus one, including the offset within the datatype.
    ///
    /// # Errors
    ///
    /// Propagates the source's read errors, including [`FormatError::UnexpectedSourceEof`]
    /// when the source reports truncation. Returns [`FormatError::UnexpectedEof`] if a
    /// successful read supplies fewer bytes than requested, with extents from the start of
    /// the node. Returns the header and length errors of [`BTreeV1ChunkNode::parse_with_offset_width`]
    /// and [`FormatError::OffsetOverflow`] if the body address exceeds [`u64::MAX`].
    pub fn read_from_source_with_offset_width<'a>(
        source: &'a (impl MetadataSource<Bytes<'a> = B> + ?Sized),
        address: u64,
        offset_width: OffsetWidth,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        let header = source.read_metadata(
            address,
            header::btree_v1_node_header_size(offset_width.get()),
        )?;
        let layout =
            ChunkNodeLayout::parse(&mut bytes::ByteCursor::new(&header), offset_width, ndims)?;
        let header_len = layout.header_len();
        let body_address =
            address
                .checked_add(header_len.to_u64())
                .ok_or(FormatError::OffsetOverflow {
                    offset: address,
                    length: header_len.to_u64(),
                })?;
        let body = source.read_metadata(body_address, layout.body_len())?;
        if body.len() < layout.body_len() {
            return Err(FormatError::UnexpectedEof {
                expected: layout.node_len,
                available: header_len + body.len(),
            });
        }
        Ok(Self { layout, body })
    }

    /// Borrows the entries in use from the retained body.
    pub fn node(&self) -> BTreeV1ChunkNode<'_> {
        BTreeV1ChunkNode {
            layout: self.layout,
            // The constructor acquires the full body, including the key after the entries.
            entries: &self.body[..self.layout.entries_len()],
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
    /// stores after the chunk's offsets. See [`Self::parse_with_offset_width`] for the acquisition
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8. Propagates
    /// the errors from [`Self::parse_with_offset_width`].
    pub fn parse(
        file_data: &'a [u8],
        offset: usize,
        offset_size: u8,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        Self::parse_with_offset_width(
            file_data,
            offset,
            OffsetWidth::try_from(offset_size)?,
            ndims,
        )
    }

    /// Parses the chunk node at `offset` in `file_data` using a validated address width.
    ///
    /// `ndims` is the rank of the dataset plus one, for the offset in the datatype that each key
    /// stores after the chunk's offsets.
    ///
    /// Acquires the complete header, including both sibling addresses, before checking the
    /// signature and node type. Then acquires the entries in use and the key after the last
    /// child. Ignores unused node capacity.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the header or the body runs past the end of
    /// `file_data`, with extents from the start of `file_data`. After acquiring the header,
    /// returns [`FormatError::InvalidBTreeSignature`] if it does not begin with `TREE`, or
    /// [`FormatError::InvalidBTreeNodeType`] if its node type is not 1. Returns
    /// [`FormatError::OffsetOverflow`] if the length of the node does not fit [`usize`].
    pub fn parse_with_offset_width(
        file_data: &'a [u8],
        offset: usize,
        offset_width: OffsetWidth,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        let mut cursor = bytes::ByteCursor::with_position(file_data, offset).map_err(|error| {
            bytes::UnexpectedEof::new(
                error
                    .expected
                    .saturating_add(header::btree_v1_node_header_size(offset_width.get())),
                error.available,
            )
        })?;
        let layout = ChunkNodeLayout::parse(&mut cursor, offset_width, ndims)?;
        let body = cursor.take(layout.body_len())?;
        Ok(Self {
            layout,
            entries: &body[..layout.entries_len()],
        })
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
            .chunks_exact(layout.entry_len)
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

/// A chunk node's level, entry count, and checked lengths.
#[derive(Clone, Copy, Debug)]
struct ChunkNodeLayout {
    node_level: u8,
    entries_used: u16,
    key_size: usize,
    entry_len: usize,
    offset_width: OffsetWidth,
    /// The length through the key after the last child, including the complete header.
    node_len: usize,
}

impl ChunkNodeLayout {
    /// Acquires the complete header and derives the body length for keys of `ndims` offsets.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the cursor cannot supply the complete header.
    /// Returns [`FormatError::InvalidBTreeSignature`] if the header does not begin with `TREE`,
    /// [`FormatError::InvalidBTreeNodeType`] if its node type is not 1, or
    /// [`FormatError::OffsetOverflow`] if the length of the node does not fit [`usize`].
    fn parse(
        cursor: &mut bytes::ByteCursor<'_>,
        offset_width: OffsetWidth,
        ndims: usize,
    ) -> Result<Self, FormatError> {
        let header_len = header::btree_v1_node_header_size(offset_width.get());
        let header = cursor.take(header_len)?;
        let NodePrefix {
            node_type,
            node_level,
            entries_used,
        } = NodePrefix::parse(header)?;
        if node_type != BTREE_V1_CHUNK_NODE_TYPE {
            return Err(FormatError::InvalidBTreeNodeType(node_type));
        }
        let key_size = chunk_key_size(ndims)?;
        let child_len = usize::from(offset_width.get());
        let entry_len = key_size
            .checked_add(child_len)
            .ok_or(FormatError::OffsetOverflow {
                offset: key_size.to_u64(),
                length: child_len.to_u64(),
            })?;
        let node_len = usize::from(entries_used)
            .checked_mul(entry_len)
            .and_then(|entries| entries.checked_add(key_size))
            .and_then(|body| body.checked_add(header_len))
            .ok_or(FormatError::OffsetOverflow {
                offset: u64::from(entries_used),
                length: entry_len.to_u64(),
            })?;
        Ok(Self {
            node_level,
            entries_used,
            key_size,
            entry_len,
            offset_width,
            node_len,
        })
    }

    fn header_len(&self) -> usize {
        header::btree_v1_node_header_size(self.offset_width.get())
    }

    fn body_len(&self) -> usize {
        self.node_len - self.header_len()
    }

    fn entries_len(&self) -> usize {
        self.body_len() - self.key_size
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
    /// stored offset in the datatype. The format specifies zero for the last offset.
    pub fn offsets(&self) -> impl ExactSizeIterator<Item = u64> + 'a {
        self.offsets
            .iter()
            .map(|&offset| u64::from_le_bytes(offset))
    }
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

/// The node type of a version 1 B-tree that indexes the chunks of a dataset.
const BTREE_V1_CHUNK_NODE_TYPE: u8 = 1;

#[cfg(test)]
#[path = "tests/chunk.rs"]
mod tests;
