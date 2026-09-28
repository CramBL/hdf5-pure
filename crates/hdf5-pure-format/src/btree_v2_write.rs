//! The version 2 B-tree writer: [`BTreeV2Plan`] lays out a tree over records in their sort order,
//! and serializes it.
//!
//! The plan builds the tree from the leaves up in one pass. Every node is `node_size` bytes, so a
//! node begins its index times the node size past the first. The tree keeps the properties the C
//! library keeps as it inserts records one at a time: every leaf is at one depth, and no node is
//! over its capacity.

use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::btree_v2::BTreeV2NodeInfo;
use crate::bytes;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The node size in bytes the C library gives the name index of an object's dense attributes and
/// the huge-object index of a fractal heap.
///
/// `H5A_NAME_BT2_NODE_SIZE` in `H5Adense.c` and `H5HF_HUGE_BT2_NODE_SIZE` in `H5HFhuge.c` (HDF5
/// 2.2.0) are both 512.
pub const BTREE_V2_NODE_SIZE: u32 = 512;

/// The split percentage of the header, 100 in `H5A_NAME_BT2_SPLIT_PERC` and
/// `H5HF_HUGE_BT2_SPLIT_PERC` (HDF5 2.2.0).
const SPLIT_PERCENT: u8 = 100;

/// The merge percentage of the header, 40 in `H5A_NAME_BT2_MERGE_PERC` and
/// `H5HF_HUGE_BT2_MERGE_PERC` (HDF5 2.2.0).
const MERGE_PERCENT: u8 = 40;

/// One node of a planned tree, before any address is known.
struct PlannedNode {
    /// The depth of the node, 0 for a leaf.
    depth: u16,
    /// The indices of the node's records in the caller's record list, in order. The records of an
    /// internal node are apart in that list, with a whole subtree between two of them.
    records: Vec<usize>,
    /// The indices of the node's children in [`BTreeV2Plan::nodes`], in order. Empty for a leaf,
    /// and `records.len() + 1` long for an internal node.
    children: Vec<usize>,
}

/// The layout of a version 2 B-tree: its nodes and the records each holds, before the tree has an
/// address.
///
/// [`nodes_size`](Self::nodes_size) returns the space the nodes take, which a caller allocates
/// before [`serialize`](Self::serialize) writes the tree at the address the caller chose. The tree
/// is defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::BTreeV2Header;
/// use hdf5_pure_format::BTreeV2Plan;
/// use hdf5_pure_format::BTREE_V2_NODE_SIZE;
/// use hdf5_pure_format::LengthWidth;
/// use hdf5_pure_format::OffsetWidth;
/// use hdf5_pure_format::StoredAddress;
///
/// // Two 17-byte records of a tree of type 8, in their sort order.
/// let records = [[1u8; 17], [2u8; 17]].concat();
/// let plan = BTreeV2Plan::new(8, 2, 17, BTREE_V2_NODE_SIZE, OffsetWidth::Eight).unwrap();
/// assert_eq!(plan.nodes_size(), 512);
///
/// // The caller allocates `nodes_size` bytes at an address of its choice.
/// let nodes_address = StoredAddress::new(0x1000);
/// let image = plan.serialize(&records, nodes_address, OffsetWidth::Eight, LengthWidth::Eight);
///
/// let header = BTreeV2Header::parse(&image.header, 0, 8, 8).unwrap();
/// assert_eq!(header.root_node_address, nodes_address);
/// assert_eq!(header.total_records, 2);
/// assert_eq!(image.nodes.len(), 512);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub struct BTreeV2Plan {
    tree_type: u8,
    node_size: u32,
    record_size: u16,
    depth: u16,
    total_records: u64,
    /// Every node, each child before its parent, so the root is the last.
    nodes: Vec<PlannedNode>,
    info: BTreeV2NodeInfo,
}

/// A serialized version 2 B-tree: the header, and the nodes it points to.
pub struct BTreeV2Image {
    /// The header, signature `BTHD`.
    pub header: Vec<u8>,
    /// The nodes back to back, each `node_size` bytes, with the root last.
    pub nodes: Vec<u8>,
}

/// Returns the size in bytes of a version 2 B-tree header at these widths.
pub const fn btree_v2_header_size(offset_width: OffsetWidth, length_width: LengthWidth) -> usize {
    // signature(4) + version(1) + type(1) + node size(4) + record size(2) +
    // depth(2) + split %(1) + merge %(1) + root address + records in root(2) +
    // total records + checksum(4)
    4 + 1
        + 1
        + 4
        + 2
        + 2
        + 1
        + 1
        + offset_width.get() as usize
        + 2
        + length_width.get() as usize
        + 4
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
    /// Lays out a tree of type `tree_type` with `record_count` records of `record_size` bytes in
    /// nodes of `node_size` bytes.
    ///
    /// Returns `None` if every tree of these sizes over `record_count` records has an empty node,
    /// or if a leaf of `node_size` bytes fits more than `u16::MAX` records, the largest count the
    /// 2-byte "Number of Records in Root Node" field of the header stores.
    pub fn new(
        tree_type: u8,
        record_count: usize,
        record_size: u16,
        node_size: u32,
        offset_width: OffsetWidth,
    ) -> Option<BTreeV2Plan> {
        let (info, depth) = BTreeV2NodeInfo::for_record_count(
            node_size,
            record_size,
            offset_width.get(),
            record_count as u64,
        )?;
        u16::try_from(info.max_nrec(0)).ok()?;

        let mut nodes = Vec::new();
        if record_count == 0 {
            // The one place an empty node is right: a tree with no records is
            // still a header pointing at a root, and the root is a bare leaf.
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
            nodes,
            info,
        })
    }

    /// Returns the size in bytes of all the nodes of the tree together.
    pub fn nodes_size(&self) -> u64 {
        self.nodes.len() as u64 * self.node_size as u64
    }

    /// Serializes the tree, with its nodes from `nodes_address` on.
    ///
    /// `records` holds the records back to back in their sort order, `record_count * record_size`
    /// bytes. The header points to the root, the last of the nodes.
    ///
    /// # Panics
    ///
    /// Panics if `records` is not `record_count * record_size` bytes long, if the address of a node
    /// does not fit `offset_width`, or if `record_count` does not fit `length_width`.
    pub fn serialize(
        &self,
        records: &[u8],
        nodes_address: StoredAddress,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> BTreeV2Image {
        let rs = self.record_size as usize;
        assert_eq!(
            records.len() as u64,
            self.total_records * rs as u64,
            "record buffer must hold exactly the records the plan placed"
        );

        // The records in the subtree of each node, which a child pointer above depth 1 holds.
        // Each child is before its parent, so one pass in order computes every count.
        let mut subtree_total = vec![0u64; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            let below: u64 = node.children.iter().map(|&c| subtree_total[c]).sum();
            subtree_total[i] = node.records.len() as u64 + below;
        }

        let address_of = |index: usize| nodes_address.offset(index as u64 * self.node_size as u64);

        let mut nodes = Vec::with_capacity(self.nodes.len() * self.node_size as usize);
        for (i, node) in self.nodes.iter().enumerate() {
            let start = nodes.len();
            nodes.extend_from_slice(if node.depth == 0 { b"BTLF" } else { b"BTIN" });
            nodes.push(0); // version
            nodes.push(self.tree_type);
            for &r in &node.records {
                nodes.extend_from_slice(&records[r * rs..(r + 1) * rs]);
            }
            if node.depth > 0 {
                let nrec_width = self.info.max_nrec_size();
                let total_width = self.info.total_nrec_size(node.depth);
                for &child in &node.children {
                    bytes::write_offset(&mut nodes, address_of(child).get(), offset_width);
                    write_uint(
                        &mut nodes,
                        self.nodes[child].records.len() as u64,
                        nrec_width,
                    );
                    write_uint(&mut nodes, subtree_total[child], total_width);
                }
            }
            let checksum = crate::checksum::jenkins_lookup3(&nodes[start..]);
            nodes.extend_from_slice(&checksum.to_le_bytes());
            debug_assert!(
                nodes.len() - start <= self.node_size as usize,
                "a planned node overflows its node size"
            );
            nodes.resize(start + self.node_size as usize, 0);
            debug_assert_eq!(address_of(i).get() - nodes_address.get(), start as u64);
        }

        let root = self.nodes.last().expect("a plan always has a root");
        let mut header = Vec::with_capacity(btree_v2_header_size(offset_width, length_width));
        header.extend_from_slice(b"BTHD");
        header.push(0); // version
        header.push(self.tree_type);
        header.extend_from_slice(&self.node_size.to_le_bytes());
        header.extend_from_slice(&self.record_size.to_le_bytes());
        header.extend_from_slice(&self.depth.to_le_bytes());
        header.push(SPLIT_PERCENT);
        header.push(MERGE_PERCENT);
        bytes::write_offset(
            &mut header,
            address_of(self.nodes.len() - 1).get(),
            offset_width,
        );
        // `new` rejects a geometry whose leaf capacity exceeds `u16::MAX`, and no node holds more
        // records than a leaf.
        let root_nrec = u16::try_from(root.records.len())
            .expect("`BTreeV2Plan::new` bounds every node's record count by `u16::MAX`");
        header.extend_from_slice(&root_nrec.to_le_bytes());
        bytes::write_length(&mut header, self.total_records, length_width);
        let checksum = crate::checksum::jenkins_lookup3(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(
            header.len(),
            btree_v2_header_size(offset_width, length_width)
        );

        BTreeV2Image { header, nodes }
    }
}

/// Returns the fewest records a subtree of `depth` holds with no node empty, `2^(depth + 1) - 1`
/// for one record and two children at every depth.
///
/// The count saturates at `u64::MAX` past depth 62, which exceeds every record count.
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

/// Appends `value` to `buf` as a little-endian integer of `width` bytes.
///
/// A width of 0 appends no bytes, for a field the child pointers of a node do not have.
fn write_uint(buf: &mut Vec<u8>, value: u64, width: usize) {
    for i in 0..width {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "masked to one byte by the shift-and-truncate"
        )]
        buf.push((value >> (i * 8)) as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
    const LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;

    #[test]
    fn no_node_exceeds_its_capacity_or_sits_empty() {
        for count in [1usize, 30, 569, 570, 10_260, 40_000] {
            let plan = BTreeV2Plan::new(8, count, 17, BTREE_V2_NODE_SIZE, OFFSET_WIDTH)
                .expect("plannable");
            for node in &plan.nodes {
                assert!(
                    !node.records.is_empty(),
                    "empty node at depth {} for {count} records",
                    node.depth
                );
                assert!(
                    node.records.len() as u64 <= plan.info.max_nrec(node.depth),
                    "node at depth {} holds {} records, capacity {}",
                    node.depth,
                    node.records.len(),
                    plan.info.max_nrec(node.depth)
                );
                assert_eq!(
                    node.children.len(),
                    if node.depth == 0 {
                        0
                    } else {
                        node.records.len() + 1
                    },
                    "a node's children must interleave with its records"
                );
            }
        }
    }

    #[test]
    fn nodes_are_all_one_node_size_long() {
        let plan =
            BTreeV2Plan::new(8, 5_000, 17, BTREE_V2_NODE_SIZE, OFFSET_WIDTH).expect("plannable");
        let image = plan.serialize(
            &vec![0; 5_000 * 17],
            StoredAddress::new(4_096),
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        );
        assert_eq!(image.nodes.len() as u64, plan.nodes_size());
        assert_eq!(image.nodes.len() % BTREE_V2_NODE_SIZE as usize, 0);
    }

    #[test]
    fn a_shape_needing_an_empty_node_is_refused() {
        // A 256-byte node holds one 200-byte record at every depth, so the counts are 1, 3, 7, 15,
        // and so on.
        assert!(BTreeV2Plan::new(8, 1_000, 200, 256, OFFSET_WIDTH).is_none());
        assert!(BTreeV2Plan::new(8, 4, 200, 256, OFFSET_WIDTH).is_none());
        assert!(BTreeV2Plan::new(8, 7, 200, 256, OFFSET_WIDTH).is_some());
    }

    #[test]
    #[should_panic(expected = "record buffer must hold exactly the records the plan placed")]
    fn serializing_a_record_buffer_of_another_length_panics() {
        let plan = BTreeV2Plan::new(8, 2, 17, BTREE_V2_NODE_SIZE, OFFSET_WIDTH).expect("plannable");
        plan.serialize(&[0; 35], StoredAddress::new(0), OFFSET_WIDTH, LENGTH_WIDTH);
    }
}
