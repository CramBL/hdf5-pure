use alloc::vec;
use alloc::vec::Vec;

use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The count-geometry policy selected by a layout constructor.
pub(super) enum BTreeV2Geometry {
    Collection(Option<BTreeV2CollectionInfo>),
    Indexed(BTreeV2NodeInfo),
}

impl BTreeV2Geometry {
    pub(super) fn count_widths(&self, depth: u16) -> Result<BTreeV2CountWidths, FormatError> {
        Ok(match self {
            Self::Collection(Some(info)) => info.count_widths(depth),
            Self::Collection(None) => return Err(FormatError::InvalidBTreeV2Signature),
            Self::Indexed(info) => BTreeV2CountWidths {
                direct: info.max_nrec_size,
                subtree: info.total_nrec_size(depth),
            },
        })
    }
}

/// The count widths in bytes of an internal-node child pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BTreeV2CountWidths {
    /// The width of the count of records in the child node.
    direct: usize,
    /// The subtree-count width, or zero when that field is absent.
    subtree: usize,
}

impl BTreeV2CountWidths {
    pub(super) fn direct(self) -> usize {
        self.direct
    }

    pub(super) fn subtree(self) -> usize {
        self.subtree
    }

    pub(super) fn pointer_size(self, offsets: OffsetWidth) -> usize {
        usize::from(offsets.get()) + self.direct + self.subtree
    }
}

/// Collection count geometry with stable widths for deeper levels.
pub(super) struct BTreeV2CollectionInfo {
    info: BTreeV2NodeInfo,
}

impl BTreeV2CollectionInfo {
    pub(super) fn compute(
        node_size: u32,
        record_size: u16,
        offsets: OffsetWidth,
        depth: u16,
    ) -> Self {
        let mut info = BTreeV2NodeInfo::leaf_only(max_records_leaf(node_size, record_size));
        for _ in 1..=depth {
            let previous = info.max_nrec.len() - 1;
            info.push_level(node_size, record_size, offsets);
            let current = previous + 1;
            // `H5B2__hdr_init` in `H5B2hdr.c` (HDF5 2.2.0) uses one direct-count width.
            // Equal capacity, cumulative capacity and width states therefore have equal successors.
            if info.max_nrec[previous] == info.max_nrec[current]
                && info.cum_max_nrec[previous] == info.cum_max_nrec[current]
                && info.cum_max_nrec_size[previous] == info.cum_max_nrec_size[current]
            {
                break;
            }
        }
        Self { info }
    }

    fn count_widths(&self, depth: u16) -> BTreeV2CountWidths {
        let below = depth.saturating_sub(1);
        let at = usize::from(below).min(self.info.cum_max_nrec_size.len() - 1);
        BTreeV2CountWidths {
            direct: self.info.max_nrec_size,
            subtree: self.info.cum_max_nrec_size[at],
        }
    }
}

/// Returns the fewest bytes that hold `max_nrec`, and 1 for 0.
///
/// A child pointer stores each of its record counts in the width this returns for the largest
/// count the field can hold.
fn bytes_for_max_records(max_nrec: u64) -> usize {
    if max_nrec == 0 {
        return 1;
    }
    let bits = 64 - max_nrec.leading_zeros() as usize;
    bits.div_ceil(8)
}

/// Returns the number of records a leaf of `node_size` bytes holds, or 0 where the leaf's prefix
/// and checksum fill the node or `record_size` is 0.
pub(super) fn max_records_leaf(node_size: u32, record_size: u16) -> u64 {
    // Leaf overhead: signature(4) + version(1) + type(1) + checksum(4) = 10
    let overhead = 10u32;
    if node_size <= overhead || record_size == 0 {
        return 0;
    }
    ((node_size - overhead) / record_size as u32) as u64
}

/// The capacities of the nodes of a version 2 B-tree at each depth, and the widths of the record
/// counts in its child pointers.
///
/// A child pointer holds the address of a child node, the number of records in the child, and,
/// above depth 1, the number of records in the child's subtree. The file does not store the widths
/// of the two counts, and a reader computes them from the node size, the record size, the width of
/// an address, and the depth. The specification describes the subtree capacity as the product of
/// the capacities below it. `H5B2__hdr_init` in `H5B2hdr.c` (HDF5 2.2.0) computes it as
/// `cum_max_nrec[u] = (max_nrec[u] + 1) * cum_max_nrec[u - 1] + max_nrec[u]`, and the table follows
/// the C library.
///
/// [`BTreeV2Plan`](crate::btree::v2::write::BTreeV2Plan) lays out a tree from the same table, so a
/// written tree has the widths a reader computes for it.
///
/// The child pointers are defined in "Version 2 B-trees" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub struct BTreeV2NodeInfo {
    /// The width in bytes of the "Number of Records in Child Node" field of a child pointer. The C
    /// library sizes it for the leaf capacity, the largest of any depth, and uses the one width at
    /// every depth.
    max_nrec_size: usize,
    /// The width in bytes of the "Total Number of Records in Child Node" field of a child pointer,
    /// by the depth of the child. Entry 0 is 0, since a pointer to a leaf has no such field.
    cum_max_nrec_size: Vec<usize>,
    /// The number of records a full node holds, by depth. Entry 0 is the leaf capacity, and no
    /// entry is larger than the one before it, which `H5B2__hdr_init` asserts.
    max_nrec: Vec<u64>,
    /// The number of records a full subtree holds, by the depth of its root.
    cum_max_nrec: Vec<u64>,
}

impl BTreeV2NodeInfo {
    /// Returns the record capacities of a node and its entire subtree at `depth`, in that order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if the table has no entry at `depth`.
    pub(super) fn capacities(&self, depth: u16) -> Result<(u64, u64), FormatError> {
        self.max_nrec
            .get(usize::from(depth))
            .copied()
            .zip(self.cum_max_nrec.get(usize::from(depth)).copied())
            .ok_or(FormatError::InvalidBTreeV2Signature)
    }

    /// Returns the table of a tree whose root is a leaf with room for `max_nrec0` records.
    fn leaf_only(max_nrec0: u64) -> Self {
        BTreeV2NodeInfo {
            max_nrec_size: bytes_for_max_records(max_nrec0),
            cum_max_nrec_size: vec![0],
            max_nrec: vec![max_nrec0],
            cum_max_nrec: vec![max_nrec0],
        }
    }

    /// Adds the capacities and the widths of one internal depth above the deepest in the table.
    ///
    /// Returns the record capacities of one node and its subtree at the added depth, in that
    /// order.
    fn push_level(
        &mut self,
        node_size: u32,
        record_size: u16,
        offset_width: OffsetWidth,
    ) -> (u64, u64) {
        let u = self.max_nrec.len();
        // Internal-pointer size at this level uses the *previous* level's
        // subtree-total width (H5B2_INT_POINTER_SIZE).
        let int_ptr =
            usize::from(offset_width.get()) + self.max_nrec_size + self.cum_max_nrec_size[u - 1];
        // Records that fit an internal node at this level (H5B2_NUM_INT_REC).
        let avail = (node_size as usize).saturating_sub(10 + int_ptr);
        let denom = record_size as usize + int_ptr;
        let max_nrec_u = avail.checked_div(denom).unwrap_or(0) as u64;
        let cum = max_nrec_u
            .saturating_add(1)
            .saturating_mul(self.cum_max_nrec[u - 1])
            .saturating_add(max_nrec_u);
        self.cum_max_nrec_size.push(bytes_for_max_records(cum));
        self.max_nrec.push(max_nrec_u);
        self.cum_max_nrec.push(cum);
        (max_nrec_u, cum)
    }

    /// Computes the table for a tree of `depth`, with nodes of `node_size` bytes and records of
    /// `record_size` bytes.
    ///
    /// `offset_width` is the validated width of an address in the file.
    pub fn compute(
        node_size: u32,
        record_size: u16,
        offset_width: OffsetWidth,
        depth: u16,
    ) -> BTreeV2NodeInfo {
        let mut info = BTreeV2NodeInfo::leaf_only(max_records_leaf(node_size, record_size));
        for _ in 1..=depth {
            info.push_level(node_size, record_size, offset_width);
        }
        info
    }

    /// Computes the table for the shallowest tree that holds `records`, and returns it with the
    /// depth of that tree.
    ///
    /// Returns `None` if no tree of these node and record sizes holds `records`, which happens
    /// where an internal node fits 0 records.
    pub fn for_record_count(
        node_size: u32,
        record_size: u16,
        offset_width: OffsetWidth,
        records: u64,
    ) -> Option<(BTreeV2NodeInfo, u16)> {
        let mut capacity = max_records_leaf(node_size, record_size);
        let mut info = Self::leaf_only(capacity);
        let mut depth = 0u16;
        while capacity < records {
            let (node_capacity, subtree_capacity) =
                info.push_level(node_size, record_size, offset_width);
            depth += 1;
            if node_capacity == 0 {
                return None;
            }
            capacity = subtree_capacity;
        }
        Some((info, depth))
    }

    /// Returns the number of records a full node at `depth` holds.
    ///
    /// # Panics
    ///
    /// Panics if `depth` is deeper than the tree the table was computed for.
    pub fn max_nrec(&self, depth: u16) -> u64 {
        self.max_nrec[depth as usize]
    }

    /// Returns the number of records a full subtree rooted at `depth` holds.
    pub(crate) fn cum_max_nrec(&self, depth: u16) -> u64 {
        self.cum_max_nrec[depth as usize]
    }

    /// Returns the width in bytes of the count of records in the child, one width at every depth.
    pub(crate) fn max_nrec_size(&self) -> usize {
        self.max_nrec_size
    }

    /// Returns the width in bytes of the count of records in the child's subtree, in the child
    /// pointers of a node at `depth`.
    ///
    /// The children are one depth below, so the width is `cum_max_nrec_size[depth - 1]`, and 0 at
    /// depth 1, where the children are leaves and the pointers have no such field.
    pub(crate) fn total_nrec_size(&self, depth: u16) -> usize {
        depth
            .checked_sub(1)
            .and_then(|below| self.cum_max_nrec_size.get(usize::from(below)))
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
#[path = "tests/geometry.rs"]
mod tests;
