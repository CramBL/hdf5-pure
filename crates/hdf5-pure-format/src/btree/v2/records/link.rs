use alloc::vec::Vec;
use core::ops::Range;

use super::raw::NAME_HASH_LEN;
use crate::btree::v2::BTreeV2Header;
use crate::btree::v2::BTreeV2Layout;
use crate::btree::v2::BTreeV2Record;
use crate::error::FormatError;
use crate::fractal_heap::PreparedHeapId;
use crate::width::OffsetWidth;

/// The creation order or name hash preceding the heap ID in a dense link index record.
pub(crate) enum LinkIndexKey {
    CreationOrder(u64),
    NameHash(u32),
}

impl LinkIndexKey {
    /// Appends the little-endian key followed by the original encoded heap ID.
    pub(crate) fn append_record(self, output: &mut Vec<u8>, id: PreparedHeapId<'_>) {
        match self {
            Self::NameHash(hash) => output.extend_from_slice(&hash.to_le_bytes()),
            Self::CreationOrder(order) => output.extend_from_slice(&order.to_le_bytes()),
        }
        output.extend_from_slice(id.bytes());
    }
}

impl BTreeV2Record {
    /// Returns a borrowed heap ID after checking the link-index type and complete record prefix.
    ///
    /// The heap ID follows the four-byte name hash in a type 5 record and the eight-byte creation
    /// order in a type 6 record. Bytes after the heap ID are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeNodeType`] if `tree_type` is neither 5 nor 6, and
    /// [`FormatError::InvalidBTreeV2Signature`] if the heap ID length is zero, the combined key and
    /// ID length exceeds [`u16::MAX`], or the record ends before the complete heap ID.
    pub fn link_heap_id(&self, tree_type: u8, heap_id_length: u16) -> Result<&[u8], FormatError> {
        BTreeV2LinkRecordContext::new(tree_type, heap_id_length)?
            .record(&self.data)
            .map(|record| record.heap_id)
    }
}

/// A checked heap-ID range for type 5 and type 6 link-index records.
///
/// A nonempty heap ID and its preceding key fit in a record of at most [`u16::MAX`] bytes. The
/// caller supplies individual record bytes to [`record`](Self::record), then parses the ID for
/// its fractal heap.
///
/// The record layouts are defined in "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::{BTREE_V2_LINK_NAME, BTreeV2LinkRecordContext};
///
/// let context = BTreeV2LinkRecordContext::new(BTREE_V2_LINK_NAME, 7)?;
/// let bytes = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7];
/// let record = context.record(&bytes)?;
/// assert_eq!(record.heap_id(), &[1, 2, 3, 4, 5, 6, 7]);
/// # Ok::<(), hdf5_pure_format::FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Debug)]
pub struct BTreeV2LinkRecordContext {
    heap_id: Range<usize>,
}

impl BTreeV2LinkRecordContext {
    /// Checks the key width and heap-ID length for a name or creation-order index.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeNodeType`] if `tree_type` is neither 5 nor 6, and
    /// [`FormatError::InvalidBTreeV2Signature`] if `heap_id_length` is zero or the key and ID
    /// together exceed [`u16::MAX`] bytes.
    pub fn new(tree_type: u8, heap_id_length: u16) -> Result<Self, FormatError> {
        let kind = match tree_type {
            BTREE_V2_LINK_NAME => LinkIndexKind::Name,
            BTREE_V2_LINK_CREATION_ORDER => LinkIndexKind::CreationOrder,
            other => return Err(FormatError::InvalidBTreeNodeType(other)),
        };
        if heap_id_length == 0 {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let start = match kind {
            LinkIndexKind::Name => usize::from(NAME_HASH_LEN),
            LinkIndexKind::CreationOrder => LINK_CREATION_ORDER_LEN,
        };
        let end = start
            .checked_add(usize::from(heap_id_length))
            .filter(|&end| end <= usize::from(u16::MAX))
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        Ok(Self {
            heap_id: start..end,
        })
    }

    /// Checks a name-index header's client type and minimum record width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeNodeType`] if the header's type is not 5, and
    /// [`FormatError::InvalidBTreeV2Signature`] if the heap ID length is zero or the declared
    /// record size is too small for the name hash and heap ID.
    pub fn for_name_index(
        header: &BTreeV2Header,
        heap_id_length: u16,
    ) -> Result<Self, FormatError> {
        if header.tree_type != BTREE_V2_LINK_NAME {
            return Err(FormatError::InvalidBTreeNodeType(header.tree_type));
        }
        let context = Self::new(header.tree_type, heap_id_length)?;
        context.require_record_size(header.record_size)?;
        Ok(context)
    }

    /// Checks that the declared record size includes the complete key and heap ID.
    ///
    /// Additional trailing fields are accepted.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if `record_size` is too small.
    pub fn require_record_size(&self, record_size: u16) -> Result<(), FormatError> {
        if usize::from(record_size) < self.heap_id.end {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        Ok(())
    }

    /// Borrows the complete heap ID from `bytes` using the checked key width.
    ///
    /// Bytes after the heap ID are ignored. The caller parses the ID for its fractal heap.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if `bytes` ends before the heap ID.
    pub fn record<'a>(&self, bytes: &'a [u8]) -> Result<BTreeV2LinkRecord<'a>, FormatError> {
        let heap_id = bytes
            .get(self.heap_id.clone())
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        Ok(BTreeV2LinkRecord { heap_id })
    }
}

#[derive(Debug)]
enum LinkIndexKind {
    Name,
    CreationOrder,
}

/// A borrowed view of a link-index record's complete heap ID.
///
/// Obtained from [`BTreeV2LinkRecordContext::record`]. The caller parses the ID for the
/// owning fractal heap.
#[derive(Clone, Copy, Debug)]
pub struct BTreeV2LinkRecord<'a> {
    heap_id: &'a [u8],
}

impl<'a> BTreeV2LinkRecord<'a> {
    /// Returns the heap ID without copying its bytes.
    pub fn heap_id(self) -> &'a [u8] {
        self.heap_id
    }
}

/// The validated node geometry and record fields of a type 5 link-name index.
///
/// Each record contains a name hash followed by the link's fractal-heap ID. The heap ID has a
/// positive length, and both fields fit within the declared record size.
///
/// The record is defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub struct BTreeV2LinkNameIndex {
    nodes: BTreeV2Layout,
    records: BTreeV2LinkRecordContext,
}

impl BTreeV2LinkNameIndex {
    /// Creates a link-name index context from the tree header and the heap's ID length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeNodeType`] if the tree type is not 5, and
    /// [`FormatError::InvalidBTreeV2Signature`] if the heap ID length is zero, the record is too
    /// short for its hash and heap ID, or the node geometry is invalid.
    pub fn new(
        header: &BTreeV2Header,
        offset_width: OffsetWidth,
        heap_id_length: u16,
    ) -> Result<Self, FormatError> {
        let records = BTreeV2LinkRecordContext::for_name_index(header, heap_id_length)?;
        Ok(Self {
            nodes: BTreeV2Layout::new(header, offset_width)?,
            records,
        })
    }

    /// Returns the node layout with the tree's validated record-count widths.
    pub fn nodes(&self) -> &BTreeV2Layout {
        &self.nodes
    }

    /// Returns the name hash and a borrowed heap ID from the record bytes.
    ///
    /// Bytes after the heap ID are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if `bytes` is too short for the hash and
    /// heap ID.
    pub fn record<'a>(&self, bytes: &'a [u8]) -> Result<(u32, &'a [u8]), FormatError> {
        let hash = bytes
            .get(..usize::from(NAME_HASH_LEN))
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        let heap_id = self.records.record(bytes)?.heap_id();
        Ok((hash, heap_id))
    }
}

/// The type of a version 2 B-tree that indexes the link names of a group, from the same table as
/// [`BTREE_V2_HUGE_OBJECT`](super::BTREE_V2_HUGE_OBJECT).
pub const BTREE_V2_LINK_NAME: u8 = 5;

/// The type of a version 2 B-tree that indexes the link creation order of a group, from the same
/// table as [`BTREE_V2_HUGE_OBJECT`](super::BTREE_V2_HUGE_OBJECT).
pub const BTREE_V2_LINK_CREATION_ORDER: u8 = 6;

/// The width of the creation order that opens a type 6 record, from the same section as
/// [`BTREE_V2_HUGE_OBJECT`](super::BTREE_V2_HUGE_OBJECT).
const LINK_CREATION_ORDER_LEN: usize = 8;

#[cfg(test)]
#[path = "tests/link.rs"]
mod tests;
