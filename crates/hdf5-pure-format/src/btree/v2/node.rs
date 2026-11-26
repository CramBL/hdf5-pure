use core::num::NonZeroU16;

use super::geometry::{self, BTreeV2CollectionInfo, BTreeV2Geometry};
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2::BTreeV2Header;
use crate::btree::v2::BTreeV2NodeInfo;
use crate::btree::v2::integer;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The node geometry and child-pointer count widths of a version 2 B-tree.
///
/// The caller obtains the root from [`root`](Self::root), supplies node bytes to
/// [`BTreeV2Node::parse`], and follows children through [`BTreeV2NodeView::child`].
///
/// The version 0 node layouts are defined in "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub struct BTreeV2Layout {
    tree_type: u8,
    node_size: usize,
    record_size: usize,
    offset_width: OffsetWidth,
    geometry: BTreeV2Geometry,
    root: BTreeV2RootState,
}

impl BTreeV2Layout {
    /// Creates a node layout from the header and the file's address width.
    ///
    /// Uses [`BTreeV2NodeInfo`] to compute record-count widths for every depth.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if the sizes, depth, root address, or
    /// record counts are inconsistent or exceed the supported node geometry.
    pub fn new(header: &BTreeV2Header, offset_width: OffsetWidth) -> Result<Self, FormatError> {
        let BTreeV2Header {
            tree_type,
            node_size,
            record_size,
            depth,
            root_node_address: _,
            num_records_in_root: _,
            total_records: _,
        } = *header;
        if node_size < NODE_OVERHEAD
            || record_size == 0
            || u32::from(depth) >= u64::BITS
            || geometry::max_records_leaf(node_size, record_size) > u64::from(u16::MAX)
        {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let info = BTreeV2NodeInfo::compute(node_size, record_size, offset_width, depth);
        if (0..=depth).any(|depth| info.max_nrec(depth) == 0) {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let root = BTreeV2RootState::parse(header, offset_width, LegacyEmptyRoot::Reject)?;
        let layout = Self {
            tree_type,
            node_size: u64::from(node_size).to_usize()?,
            record_size: usize::from(record_size),
            offset_width,
            geometry: BTreeV2Geometry::Indexed(info),
            root,
        };
        if let Some(root) = root.frame() {
            layout.node_geometry(root)?;
        }
        Ok(layout)
    }

    /// Creates a collection layout with checked root fields and bounded count geometry.
    ///
    /// Accepts undefined empty roots and, for compatibility, defined zero-record roots at depth
    /// zero. A nonempty root must have positive root and total counts. Positive subtree totals
    /// are retained as supplied, and records and child pointers must fit the node allocation.
    /// [`new`](Self::new) applies the stricter count and depth constraints used for indexed lookup.
    ///
    /// Count geometry is retained through the declared depth or the first stable capacity and
    /// width state. The reader uses the final stable widths for deeper levels.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_format::__private::{
    /// #     BTreeV2CollectionRoot, BTreeV2Header, BTreeV2Layout, BTREE_V2_LINK_NAME, OffsetWidth,
    /// # };
    /// # use hdf5_pure_format::FormatError;
    /// # use test_util::{btree::v2 as btree_v2, widths::Widths};
    /// # let header_bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 512, 0)
    /// #     .build(Widths::EIGHT);
    /// # let header = BTreeV2Header::parse(&header_bytes, 0, 8, 8)?;
    /// # let node_bytes = btree_v2::leaf(BTREE_V2_LINK_NAME, &[]);
    /// let layout = BTreeV2Layout::for_collection(&header, OffsetWidth::Eight)?;
    /// assert!(matches!(layout.collection_root(), BTreeV2CollectionRoot::LegacyEmpty));
    /// let root = layout.root().unwrap();
    /// let node = root.parse(&node_bytes)?;
    /// assert_eq!(node.record_count(), 0);
    /// assert_eq!(node.child(0)?, None);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if the record size is zero, the node
    /// allocation is too small, the root address exceeds the offset width, the root fields
    /// disagree about emptiness, or the records and child pointers exceed the node allocation.
    pub fn for_collection(
        header: &BTreeV2Header,
        offset_width: OffsetWidth,
    ) -> Result<Self, FormatError> {
        Self::for_collection_with_policy(header, offset_width, LegacyEmptyRoot::Allow)
    }

    fn for_collection_with_policy(
        header: &BTreeV2Header,
        offset_width: OffsetWidth,
        empty_policy: LegacyEmptyRoot,
    ) -> Result<Self, FormatError> {
        let BTreeV2Header {
            tree_type,
            node_size,
            record_size,
            depth,
            root_node_address: _,
            num_records_in_root: _,
            total_records: _,
        } = *header;
        if node_size < NODE_OVERHEAD || record_size == 0 {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let root = BTreeV2RootState::parse(header, offset_width, empty_policy)?;
        let layout = Self {
            tree_type,
            node_size: u64::from(node_size).to_usize()?,
            record_size: usize::from(record_size),
            offset_width,
            geometry: BTreeV2Geometry::Collection((depth != 0).then(|| {
                BTreeV2CollectionInfo::compute(node_size, record_size, offset_width, depth)
            })),
            root,
        };
        if let Some(frame) = root.frame() {
            layout.node_geometry(frame)?;
        }
        Ok(layout)
    }

    /// Returns the root's inherited counts and stored address.
    ///
    /// Returns `None` for an undefined empty root. A layout constructed with
    /// [`for_collection`](Self::for_collection) may return a defined zero-record root accepted
    /// for compatibility.
    pub fn root(&self) -> Option<BTreeV2Node<'_>> {
        self.root.frame().map(|frame| BTreeV2Node {
            layout: self,
            frame,
        })
    }

    /// Returns the empty-root classification or nonempty root for record collection.
    ///
    /// Distinguishes undefined empty roots from the defined zero-record roots accepted by
    /// [`for_collection`](Self::for_collection).
    pub fn collection_root(&self) -> BTreeV2CollectionRoot<'_> {
        match self.root {
            BTreeV2RootState::Empty => BTreeV2CollectionRoot::Empty,
            BTreeV2RootState::LegacyEmpty(_) => BTreeV2CollectionRoot::LegacyEmpty,
            BTreeV2RootState::NonEmpty(frame) => BTreeV2CollectionRoot::NonEmpty(BTreeV2Node {
                layout: self,
                frame,
            }),
        }
    }

    /// Returns the end offsets of the records and child pointers, respectively.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if the node's inherited counts or address
    /// are inconsistent with this layout.
    fn node_geometry(&self, node: BTreeV2NodeFrame) -> Result<(usize, usize), FormatError> {
        let Self {
            tree_type: _,
            node_size,
            record_size,
            offset_width,
            geometry,
            root: _,
        } = self;
        let BTreeV2NodeFrame {
            address,
            kind,
            records,
            total_records,
        } = node;
        let depth = kind.depth();
        if let BTreeV2Geometry::Indexed(info) = geometry {
            let (capacity, subtree_capacity) = info.capacities(depth)?;
            let minimum = u64::from(records)
                .checked_add(1)
                .and_then(|children| children.checked_mul((1u64 << depth) - 1))
                .and_then(|descendants| descendants.checked_add(u64::from(records)))
                .ok_or(FormatError::InvalidBTreeV2Signature)?;
            if records == 0
                || total_records == 0
                || address.get() >= StoredAddress::undefined(*offset_width).get()
                || u64::from(records) > capacity
                || total_records < minimum
                || total_records > subtree_capacity
                || (depth == 0 && total_records != u64::from(records))
            {
                return Err(FormatError::InvalidBTreeV2Signature);
            }
        }
        let records_end = usize::from(records)
            .checked_mul(*record_size)
            .and_then(|length| NODE_PREFIX_LEN.checked_add(length))
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        let pointer_size = if depth == 0 {
            0
        } else {
            geometry.count_widths(depth)?.pointer_size(*offset_width)
        };
        let children_end = (usize::from(records) + 1)
            .checked_mul(pointer_size)
            .and_then(|length| records_end.checked_add(length))
            .filter(|end| *end <= *node_size - NODE_CHECKSUM_LEN)
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        Ok((records_end, children_end))
    }
}

/// A version 2 B-tree root classified for record collection.
///
/// Returned by [`BTreeV2Layout::collection_root`].
#[derive(Debug)]
pub enum BTreeV2CollectionRoot<'a> {
    /// An undefined root with zero depth and record counts.
    Empty,
    /// A defined root with zero depth and record counts, accepted for compatibility.
    LegacyEmpty,
    /// A defined root with positive declared record counts.
    NonEmpty(BTreeV2Node<'a>),
}

/// A reference to a version 2 B-tree node and its owning layout.
///
/// Obtained from [`BTreeV2Layout::root`] or [`BTreeV2NodeView::child`]. The stored address, depth,
/// and record counts come from the header or a parsed parent. Records and child-pointer fields
/// fit the allocation size of the borrowed [`BTreeV2Layout`].
#[derive(Clone, Copy)]
pub struct BTreeV2Node<'a> {
    layout: &'a BTreeV2Layout,
    frame: BTreeV2NodeFrame,
}

impl<'a> BTreeV2Node<'a> {
    /// Parses the node using its owner's geometry and inherited record counts.
    ///
    /// `bytes` starts at the node's signature and includes its checksum field. Unused allocation
    /// bytes may follow it. With the `checksum` feature, verifies the checksum over the prefix,
    /// records, and child-pointer fields after checking the signature, version, and owner type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_format::__private::{
    /// #     BTreeV2Header, BTreeV2Layout, BTREE_V2_ATTRIBUTE_NAME, OffsetWidth,
    /// # };
    /// # use hdf5_pure_format::FormatError;
    /// # use test_util::{btree::v2 as btree_v2, widths::Widths};
    /// # let header_bytes = btree_v2::Header::new(BTREE_V2_ATTRIBUTE_NAME, 17, 512, 1)
    /// #     .build(Widths::EIGHT);
    /// # let header = BTreeV2Header::parse(&header_bytes, 0, 8, 8)?;
    /// # let node_bytes = btree_v2::leaf(BTREE_V2_ATTRIBUTE_NAME, &[vec![0; 17]]);
    /// let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight)?;
    /// let root = layout.root().unwrap();
    /// let node = root.parse(&node_bytes)?;
    /// assert_eq!(node.record_count(), 1);
    /// assert_eq!(node.record(0)?, &[0; 17]);
    /// assert_eq!(node.child(0)?, None);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] for an invalid signature or node geometry,
    /// [`FormatError::InvalidBTreeV2Version`] for a version other than 0,
    /// [`FormatError::InvalidBTreeNodeType`] for a type that differs from the header, and
    /// [`FormatError::UnexpectedEof`] for truncated node contents. With the `checksum` feature,
    /// returns [`FormatError::ChecksumMismatch`] if verification fails.
    pub fn parse<'b>(self, bytes: &'b [u8]) -> Result<BTreeV2NodeView<'a, 'b>, FormatError> {
        let (records_end, children_end) = self.layout.node_geometry(self.frame)?;
        let encoded_len = children_end
            .checked_add(NODE_CHECKSUM_LEN)
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        bytes::ensure_len(bytes, 0, encoded_len)?;
        let signature = match self.frame.kind {
            BTreeV2NodeKind::Leaf => LEAF_SIGNATURE,
            BTreeV2NodeKind::Internal(_) => INTERNAL_SIGNATURE,
        };
        if bytes.get(..SIGNATURE_LEN) != Some(signature.as_slice()) {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        if bytes[SIGNATURE_LEN] != NODE_VERSION {
            return Err(FormatError::InvalidBTreeV2Version(bytes[SIGNATURE_LEN]));
        }
        if bytes[SIGNATURE_LEN + 1] != self.layout.tree_type {
            return Err(FormatError::InvalidBTreeNodeType(bytes[SIGNATURE_LEN + 1]));
        }
        crate::checksum::verify_trailing(&bytes[..encoded_len])?;
        Ok(BTreeV2NodeView {
            node: self,
            records: &bytes[NODE_PREFIX_LEN..records_end],
            children: &bytes[records_end..children_end],
        })
    }

    /// Returns the length through the node's checksum, excluding unused allocation bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if the node's inherited counts or address
    /// are inconsistent with its owning layout.
    pub fn encoded_len(self) -> Result<usize, FormatError> {
        self.layout.node_geometry(self.frame).and_then(|(_, end)| {
            end.checked_add(NODE_CHECKSUM_LEN)
                .ok_or(FormatError::InvalidBTreeV2Signature)
        })
    }

    /// Returns the base-relative address for the caller to resolve before reading the node.
    pub fn address(self) -> StoredAddress {
        self.frame.address
    }

    /// Returns the record count for this node and all its descendants.
    pub fn total_records(self) -> u64 {
        self.frame.total_records
    }
}

impl PartialEq for BTreeV2Node<'_> {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.layout, other.layout) && self.frame == other.frame
    }
}

impl Eq for BTreeV2Node<'_> {}

impl core::fmt::Debug for BTreeV2Node<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.frame.fmt(f)
    }
}

/// The borrowed records and child-pointer bytes of a validated version 2 B-tree node.
///
/// Obtained from [`BTreeV2Node::parse`]. The signature, version, owner type, and complete encoded
/// extent have been checked. [`record`](Self::record) returns slices of the supplied node bytes.
/// [`child`](Self::child) decodes and validates one selected pointer using the same owner.
pub struct BTreeV2NodeView<'a, 'b> {
    node: BTreeV2Node<'a>,
    records: &'b [u8],
    children: &'b [u8],
}

impl<'a, 'b> BTreeV2NodeView<'a, 'b> {
    /// Returns the number of records inherited from the node's parent or header.
    pub fn record_count(&self) -> usize {
        usize::from(self.node.frame.records)
    }

    /// Returns the encoded record at the zero-based `index` without copying it.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if `index` is outside the records.
    pub fn record(&self, index: usize) -> Result<&'b [u8], FormatError> {
        let start = index
            .checked_mul(self.node.layout.record_size)
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        start
            .checked_add(self.node.layout.record_size)
            .and_then(|end| self.records.get(start..end))
            .ok_or(FormatError::InvalidBTreeV2Signature)
    }

    /// Decodes the selected child pointer, or returns `None` for a leaf.
    ///
    /// In an internal node, child `index` precedes record `index`. The last child follows the
    /// last record, so `index` may equal [`record_count`](Self::record_count).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if `index` is outside an internal node's
    /// children, the child's encoded extent exceeds its allocation, or its address or counts
    /// violate the selected layout's constraints.
    pub fn child(&self, index: usize) -> Result<Option<BTreeV2Node<'a>>, FormatError> {
        let Self {
            node,
            records: _,
            children,
        } = self;
        let BTreeV2Node { layout, frame } = *node;
        let BTreeV2NodeFrame {
            address: _,
            kind,
            records: parent_records,
            total_records: parent_total,
        } = frame;
        let BTreeV2NodeKind::Internal(depth) = kind else {
            return Ok(None);
        };
        let depth = depth.get();
        if index > usize::from(parent_records) {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let widths = layout.geometry.count_widths(depth)?;
        let width = widths.pointer_size(layout.offset_width);
        let start = index
            .checked_mul(width)
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        let address = StoredAddress::new(bytes::read_offset_width(
            children,
            start,
            layout.offset_width,
        )?);
        let count_at = start + usize::from(layout.offset_width.get());
        let records = u16::try_from(integer::read_var_uint(children, count_at, widths.direct())?)
            .map_err(|_error| FormatError::InvalidBTreeV2Signature)?;
        let total_width = widths.subtree();
        let total_records = if total_width == 0 {
            u64::from(records)
        } else {
            integer::read_var_uint(children, count_at + widths.direct(), total_width)?
        };
        let frame = BTreeV2NodeFrame {
            address,
            kind: BTreeV2NodeKind::from_depth(depth - 1),
            records,
            total_records,
        };
        layout.node_geometry(frame)?;
        if matches!(layout.geometry, BTreeV2Geometry::Indexed(_))
            && total_records > parent_total - u64::from(parent_records)
        {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        Ok(Some(BTreeV2Node { layout, frame }))
    }
}

/// Header root fields checked for address width and agreement on zero counts.
#[derive(Clone, Copy)]
enum BTreeV2RootState {
    Empty,
    LegacyEmpty(BTreeV2NodeFrame),
    NonEmpty(BTreeV2NodeFrame),
}

impl BTreeV2RootState {
    fn parse(
        header: &BTreeV2Header,
        offset_width: OffsetWidth,
        policy: LegacyEmptyRoot,
    ) -> Result<Self, FormatError> {
        let BTreeV2Header {
            tree_type: _,
            node_size: _,
            record_size: _,
            depth,
            root_node_address,
            num_records_in_root,
            total_records,
        } = *header;
        let maximum = StoredAddress::undefined(offset_width).get();
        if root_node_address.get() > maximum {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let frame = BTreeV2NodeFrame {
            address: root_node_address,
            kind: BTreeV2NodeKind::from_depth(depth),
            records: num_records_in_root,
            total_records,
        };
        let undefined = root_node_address.is_undefined(offset_width);
        Ok(match (total_records, num_records_in_root, undefined) {
            (0, 0, true) if depth == 0 => Self::Empty,
            (0, 0, false) if depth == 0 && matches!(policy, LegacyEmptyRoot::Allow) => {
                Self::LegacyEmpty(frame)
            }
            (1.., 1.., false) => Self::NonEmpty(frame),
            _ => return Err(FormatError::InvalidBTreeV2Signature),
        })
    }

    fn frame(self) -> Option<BTreeV2NodeFrame> {
        match self {
            Self::Empty => None,
            Self::LegacyEmpty(frame) | Self::NonEmpty(frame) => Some(frame),
        }
    }
}

/// Whether collection accepts a defined zero-record root at depth zero.
#[derive(Clone, Copy, Debug)]
enum LegacyEmptyRoot {
    Allow,
    Reject,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BTreeV2NodeFrame {
    address: StoredAddress,
    kind: BTreeV2NodeKind,
    records: u16,
    total_records: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BTreeV2NodeKind {
    Internal(NonZeroU16),
    Leaf,
}

impl BTreeV2NodeKind {
    fn from_depth(depth: u16) -> Self {
        match NonZeroU16::new(depth) {
            None => Self::Leaf,
            Some(depth) => Self::Internal(depth),
        }
    }

    fn depth(self) -> u16 {
        match self {
            Self::Leaf => 0,
            Self::Internal(depth) => depth.get(),
        }
    }
}

// "Version 2 B-trees" in the format specification, version 4.0, defines the version 0 node
// signatures, six-byte prefix, and four-byte trailing checksum.
const SIGNATURE_LEN: usize = 4;
pub(super) const NODE_PREFIX_LEN: usize = 6;
pub(super) const NODE_CHECKSUM_LEN: usize = 4;
pub(super) const NODE_OVERHEAD: u32 = 10;
const NODE_VERSION: u8 = 0;
const LEAF_SIGNATURE: &[u8; 4] = b"BTLF";
const INTERNAL_SIGNATURE: &[u8; 4] = b"BTIN";

#[cfg(test)]
#[path = "tests/node.rs"]
mod tests;
