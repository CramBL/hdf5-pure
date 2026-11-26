use alloc::vec::Vec;
use core::num::NonZeroU16;

use crate::btree::v2::BTreeV2NodeInfo;
use crate::btree::v2::node::NODE_CHECKSUM_LEN;
use crate::btree::v2::node::NODE_PREFIX_LEN;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The node size in bytes libhdf5 gives the name index of an object's dense attributes and
/// the huge-object index of a fractal heap.
///
/// `H5A_NAME_BT2_NODE_SIZE` in `H5Adense.c` and `H5HF_HUGE_BT2_NODE_SIZE` in `H5HFhuge.c` (HDF5
/// 2.2.0) are both 512.
pub const BTREE_V2_NODE_SIZE: u32 = 512;

/// The encoding of direct record counts in version 2 B-tree child pointers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BTreeV2WriteDialect {
    /// Uses one width, derived from leaf capacity, for direct record counts at every depth.
    ///
    /// libhdf5 computes this width in `H5B2__hdr_init` in `H5B2hdr.c` (HDF5 2.2.0).
    Libhdf5Global,
}

/// One node of a planned tree, before any address is known.
pub(super) struct PlannedNode {
    /// The depth of the node, 0 for a leaf.
    depth: u16,
    /// The indices of the node's records in the caller's record list, in order. The records of an
    /// internal node are apart in that list, with a whole subtree between two of them.
    records: Vec<usize>,
    /// The indices of the node's children in [`BTreeV2Plan::nodes`], in order. Empty for a leaf,
    /// and `records.len() + 1` long for an internal node.
    children: Vec<usize>,
}

impl PlannedNode {
    pub(super) fn depth(&self) -> u16 {
        self.depth
    }

    pub(super) fn records(&self) -> &[usize] {
        &self.records
    }

    pub(super) fn children(&self) -> &[usize] {
        &self.children
    }

    /// Returns the used bytes through the checksum, or `None` if the size exceeds [`usize::MAX`].
    pub(super) fn encoded_size(
        &self,
        record_size: NonZeroU16,
        pointer_width: usize,
    ) -> Option<usize> {
        self.records
            .len()
            .checked_mul(usize::from(record_size.get()))?
            .checked_add(self.children.len().checked_mul(pointer_width)?)?
            .checked_add(NODE_PREFIX_LEN)?
            .checked_add(NODE_CHECKSUM_LEN)
    }
}

/// The layout of a version 2 B-tree: its nodes and the records each holds, before the tree has an
/// address.
///
/// A caller allocates one range per length returned by [`block_lengths`](Self::block_lengths),
/// then passes the stored addresses to [`place`](Self::place). For contiguous nodes,
/// [`serialize`](Self::serialize) writes the tree at one starting address. The tree is defined
/// in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub struct BTreeV2Plan {
    tree_type: u8,
    node_size: u32,
    record_size: NonZeroU16,
    depth: u16,
    total_records: u64,
    offset_width: OffsetWidth,
    /// Every node, each child before its parent, so the root is the last.
    nodes: Vec<PlannedNode>,
    info: BTreeV2NodeInfo,
}

/// Spreads `total` items over `parts` groups, the first `total % parts` groups one larger than the
/// rest.
///
/// The groups differ by one at most, so each is within the capacity the caller sized `parts`
/// against.
fn distribute(total: usize, parts: usize) -> Vec<usize> {
    debug_assert!(parts > 0, "a node always has at least one child");
    let base = total / parts;
    let remainder = total % parts;
    (0..parts)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}

/// Returns `capacity` as a `usize`, saturated at `usize::MAX`.
///
/// A capacity is a `u64`, and a record count is at most the length of the caller's slice. On a
/// 32-bit host, a capacity above `usize::MAX` is larger than every record count, and so is
/// `usize::MAX`.
fn capacity_as_usize(capacity: u64) -> usize {
    usize::try_from(capacity).unwrap_or(usize::MAX)
}

impl BTreeV2Plan {
    /// Lays out a tree with nonzero record width and checked node capacity.
    ///
    /// For zero records, plans one empty leaf with a defined address.
    /// [`Self::new_with_undefined_empty_root`] plans no nodes for an empty tree.
    ///
    /// Returns `None` if `record_size` is zero, a leaf cannot fit its prefix, checksum, and one
    /// complete record, or a leaf fits more than [`u16::MAX`] records. For nonzero `record_count`,
    /// also returns `None` if the tree cannot be laid out without an empty node.
    pub fn new(
        tree_type: u8,
        record_count: usize,
        record_size: u16,
        node_size: u32,
        offset_width: OffsetWidth,
    ) -> Option<BTreeV2Plan> {
        let record_size = NonZeroU16::new(record_size)?;
        let (info, depth) = BTreeV2NodeInfo::for_record_count(
            node_size,
            record_size.get(),
            offset_width,
            record_count as u64,
        )?;
        if info.max_nrec(0) == 0 {
            return None;
        }
        u16::try_from(info.max_nrec(0)).ok()?;

        let mut nodes = Vec::new();
        if record_count == 0 {
            // This constructor represents an empty tree with one empty leaf.
            nodes.push(PlannedNode {
                depth: 0,
                records: Vec::new(),
                children: Vec::new(),
            });
        } else {
            let mut next_record = 0usize;
            plan_subtree(record_count, depth, &info, &mut next_record, &mut nodes)?;
            debug_assert_eq!(next_record, record_count, "every record is placed once");
        }

        Some(BTreeV2Plan {
            tree_type,
            node_size,
            record_size,
            depth,
            total_records: record_count as u64,
            offset_width,
            nodes,
            info,
        })
    }

    /// Lays out a tree with libhdf5's representation of an empty tree.
    ///
    /// Uses the geometry and return conditions of [`Self::new`]. For zero records, serializes
    /// an undefined root address and no nodes, as defined in "Version 2 B-trees" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
    pub fn new_with_undefined_empty_root(
        dialect: BTreeV2WriteDialect,
        tree_type: u8,
        record_count: usize,
        record_size: u16,
        node_size: u32,
        offset_width: OffsetWidth,
    ) -> Option<Self> {
        // H5B2__hdr_init in H5B2hdr.c, HDF5 2.2.0, derives one direct-count width
        // from leaf capacity. An empty tree has an undefined root and no nodes.
        let mut plan = match dialect {
            BTreeV2WriteDialect::Libhdf5Global => Self::new(
                tree_type,
                record_count,
                record_size,
                node_size,
                offset_width,
            ),
        }?;
        if record_count == 0 {
            plan.nodes.clear();
        }
        Some(plan)
    }

    /// Returns the total node size in bytes, or `None` if it exceeds [`u64::MAX`].
    pub fn checked_nodes_size(&self) -> Option<u64> {
        u64::try_from(self.nodes.len())
            .ok()?
            .checked_mul(u64::from(self.node_size))
    }

    /// Returns the size in bytes of all the nodes of the tree together.
    pub fn nodes_size(&self) -> u64 {
        self.nodes.len() as u64 * self.node_size as u64
    }

    /// Returns each complete node's serialized length, with each child before its parent.
    ///
    /// The root is last. Pass addresses in this order to [`Self::place`]. Each length includes
    /// the node prefix, records, child pointers, checksum, and trailing zero padding.
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> + DoubleEndedIterator + '_ {
        core::iter::repeat_n(u64::from(self.node_size), self.nodes.len())
    }

    pub(super) fn nodes(&self) -> &[PlannedNode] {
        &self.nodes
    }

    pub(super) fn node_size(&self) -> u32 {
        self.node_size
    }

    pub(super) fn tree_type(&self) -> u8 {
        self.tree_type
    }

    pub(super) fn record_size(&self) -> NonZeroU16 {
        self.record_size
    }

    pub(super) fn depth(&self) -> u16 {
        self.depth
    }

    pub(super) fn total_records(&self) -> u64 {
        self.total_records
    }

    pub(super) fn offset_width(&self) -> OffsetWidth {
        self.offset_width
    }

    pub(super) fn info(&self) -> &BTreeV2NodeInfo {
        &self.info
    }

    pub(super) fn prepare_records<'a>(
        &self,
        records: &'a [u8],
    ) -> Result<CheckedRecords<'a>, FormatError> {
        let expected = self
            .total_records
            .checked_mul(u64::from(self.record_size.get()))
            .ok_or_else(|| {
                FormatError::Internal("planned record buffer length exceeds u64::MAX".into())
            })?;
        if records.len().to_u64() != expected {
            return Err(FormatError::Internal(
                "record buffer length differs from the planned records".into(),
            ));
        }
        Ok(CheckedRecords { bytes: records })
    }
}

/// Record bytes whose length matches the plan's checked count and nonzero record width.
pub(super) struct CheckedRecords<'a> {
    bytes: &'a [u8],
}

impl<'a> CheckedRecords<'a> {
    pub(super) fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Returns the fewest records a subtree of `depth` holds with no node empty, `2^(depth + 1) - 1`
/// for one record and two children at every depth.
///
/// The count saturates at [`u64::MAX`] for depths of 63 or more.
fn min_records(depth: u16) -> u64 {
    1u64.checked_shl(depth as u32 + 1)
        .map_or(u64::MAX, |v| v - 1)
}

/// Lays out a subtree of `count` records rooted at `depth`, appends its nodes to `out`, and
/// returns the index of its root.
///
/// The walk takes the records in order as it visits their positions, so an in-order walk of the
/// tree reads the caller's records in their order.
///
/// Returns `None` if a subtree of `depth` over `count` records needs an empty node, which depends
/// on the node and record sizes the caller chose.
fn plan_subtree(
    count: usize,
    depth: u16,
    info: &BTreeV2NodeInfo,
    next_record: &mut usize,
    out: &mut Vec<PlannedNode>,
) -> Option<usize> {
    debug_assert!(
        count as u64 <= info.cum_max_nrec(depth),
        "a subtree was handed more records than its depth can hold"
    );
    if (count as u64) < min_records(depth) {
        return None;
    }

    if depth == 0 {
        let records = (*next_record..*next_record + count).collect();
        *next_record += count;
        out.push(PlannedNode {
            depth,
            records,
            children: Vec::new(),
        });
        return Some(out.len() - 1);
    }

    // The fewest records this node keeps with its children holding the rest. Each of the `k + 1`
    // subtrees below takes `child_capacity` records at most, so `k` satisfies
    // `(k + 1) * child_capacity >= count - k`. The smallest such `k` fills the subtrees as full as
    // they go.
    let child_capacity = capacity_as_usize(info.cum_max_nrec(depth - 1));
    let k = count
        .saturating_sub(child_capacity)
        .div_ceil(child_capacity + 1)
        .max(1);
    debug_assert!(
        k as u64 <= info.max_nrec(depth),
        "a node was given more records than its level can hold"
    );

    let group_sizes = distribute(count - k, k + 1);
    let mut records = Vec::with_capacity(k);
    let mut children = Vec::with_capacity(k + 1);
    for (i, &size) in group_sizes.iter().enumerate() {
        children.push(plan_subtree(size, depth - 1, info, next_record, out)?);
        if i < k {
            records.push(*next_record);
            *next_record += 1;
        }
    }

    out.push(PlannedNode {
        depth,
        records,
        children,
    });
    Some(out.len() - 1)
}

#[cfg(test)]
#[path = "tests/plan.rs"]
mod tests;
