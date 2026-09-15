//! HDF5 B-tree v2 emission.
//!
//! The counterpart to [`btree_v2`](crate::btree_v2): given records already in
//! the tree's sort order, lay out a balanced tree of fixed-size nodes and
//! serialize it.
//!
//! Trees are built bottom-up in one pass rather than by repeated insertion.
//! Every node is exactly `node_size` bytes, so a node's address is its index
//! times that size, and a parent can be serialized as soon as its children have
//! been placed. The result is the same *shape* the reference C library reaches
//! by inserting records one at a time — balanced, all leaves at one depth, no
//! node over capacity — without reproducing its split-and-promote machinery.
//!
//! Node capacities come from [`NodeInfo`], the table
//! [`btree_v2`](crate::btree_v2) already computes to decode a tree's
//! variable-width child pointers, so an emitted tree cannot declare a geometry
//! this crate would then read it back with differently.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::address::StoredAddress;
use crate::btree_v2::NodeInfo;

/// The node size the reference C library uses for every B-tree v2 this crate
/// emits: `H5A_NAME_BT2_NODE_SIZE` for the attribute name index and
/// `H5HF_HUGE_BT2_NODE_SIZE` for a fractal heap's huge-object index are both
/// 512 bytes.
pub(crate) const NODE_SIZE: u32 = 512;

/// Split threshold, as a percentage of a node's capacity
/// (`H5A_NAME_BT2_SPLIT_PERC` and its siblings, all 100).
const SPLIT_PERCENT: u8 = 100;

/// Merge threshold, as a percentage of a node's capacity
/// (`H5A_NAME_BT2_MERGE_PERC` and its siblings, all 40).
const MERGE_PERCENT: u8 = 40;

/// One node of a planned tree, before any address is known.
struct PlannedNode {
    /// Depth of this node; 0 is a leaf.
    depth: u16,
    /// Indices into the caller's record list of the records this node holds, in
    /// order. An internal node's records are *not* contiguous in that list:
    /// whole subtrees sit between them.
    records: Vec<usize>,
    /// Indices into [`BTreeV2Plan::nodes`] of this node's children, in order.
    /// Empty for a leaf, and `records.len() + 1` long otherwise.
    children: Vec<usize>,
}

/// The shape of a tree over some number of records: how many nodes it takes and
/// which records land where, with no addresses committed yet.
///
/// Split from serialization because a caller has to reserve the tree's space
/// before it can know where the tree goes. [`Self::nodes_size`] answers that
/// without materializing a byte.
pub(crate) struct BTreeV2Plan {
    tree_type: u8,
    node_size: u32,
    record_size: u16,
    depth: u16,
    total_records: u64,
    /// Every node, children before parents, so the root is last and a parent's
    /// children always have smaller indices than it does.
    nodes: Vec<PlannedNode>,
    info: NodeInfo,
}

/// A serialized tree: a header and the block of nodes it points into.
pub(crate) struct BTreeV2Image {
    /// The "BTHD" header bytes.
    pub(crate) header: Vec<u8>,
    /// Every node, back to back, each exactly `node_size` bytes.
    pub(crate) nodes: Vec<u8>,
}

/// On-disk size of a v2 B-tree header.
pub(crate) const fn header_size(offset_size: u8, length_size: u8) -> usize {
    // signature(4) + version(1) + type(1) + node size(4) + record size(2) +
    // depth(2) + split %(1) + merge %(1) + root address + records in root(2) +
    // total records + checksum(4)
    4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + offset_size as usize + 2 + length_size as usize + 4
}

/// Spread `total` items over `parts` groups as evenly as possible, largest
/// first. The groups differ by at most one, so no group can exceed the capacity
/// the caller sized `parts` against.
fn distribute(total: usize, parts: usize) -> Vec<usize> {
    debug_assert!(parts > 0, "a node always has at least one child");
    let base = total / parts;
    let remainder = total % parts;
    (0..parts)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}

/// A per-level capacity as a count of records this host could actually hold.
///
/// Capacities are `u64` because the format's fields are, but a record count is
/// bounded by the caller's slice and so by `usize`. On a 64-bit host these are
/// the same number; on a 32-bit one a capacity past `usize::MAX` means "more
/// than can exist here", and saturating says exactly that.
fn capacity_as_usize(capacity: u64) -> usize {
    usize::try_from(capacity).unwrap_or(usize::MAX)
}

impl BTreeV2Plan {
    /// Plan a tree holding `record_count` records of `record_size` bytes.
    ///
    /// `None` when this node and record size cannot express a tree of that many
    /// records without an empty node — a shape the reference C library never
    /// produces, and one this crate would rather refuse than emit. That takes a
    /// node so small it fits barely one record beside an internal node's
    /// pointers; the 512-byte nodes and 17- and 24-byte records this crate emits
    /// are nowhere near it.
    pub(crate) fn new(
        tree_type: u8,
        record_count: usize,
        record_size: u16,
        node_size: u32,
        offset_size: u8,
    ) -> Option<BTreeV2Plan> {
        let (info, depth) =
            NodeInfo::for_record_count(node_size, record_size, offset_size, record_count as u64)?;

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

    /// Bytes the tree's nodes occupy, all of them together.
    pub(crate) fn nodes_size(&self) -> u64 {
        self.nodes.len() as u64 * self.node_size as u64
    }

    /// Serializes the tree, with its nodes starting at `nodes_address`.
    ///
    /// `records` holds every record back to back in the tree's sort order, so it
    /// is `total_records * record_size` bytes long.
    pub(crate) fn serialize(
        &self,
        records: &[u8],
        nodes_address: StoredAddress,
        offset_size: u8,
        length_size: u8,
    ) -> BTreeV2Image {
        let rs = self.record_size as usize;
        debug_assert_eq!(
            records.len() as u64,
            self.total_records * rs as u64,
            "record buffer must hold exactly the records the plan placed"
        );

        // Records in a node's whole subtree, needed by every child pointer at
        // depth > 1. Children always precede their parent, so one forward pass
        // fills this in.
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
                    write_uint(&mut nodes, address_of(child).get(), offset_size as usize);
                    write_uint(
                        &mut nodes,
                        self.nodes[child].records.len() as u64,
                        nrec_width,
                    );
                    // A child that is a leaf carries no subtree total, and
                    // `total_nrec_size` is 0 there, so this writes nothing.
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
        let mut header = Vec::with_capacity(header_size(offset_size, length_size));
        header.extend_from_slice(b"BTHD");
        header.push(0); // version
        header.push(self.tree_type);
        header.extend_from_slice(&self.node_size.to_le_bytes());
        header.extend_from_slice(&self.record_size.to_le_bytes());
        header.extend_from_slice(&self.depth.to_le_bytes());
        header.push(SPLIT_PERCENT);
        header.push(MERGE_PERCENT);
        write_uint(
            &mut header,
            address_of(self.nodes.len() - 1).get(),
            offset_size as usize,
        );
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the root's own record count is bounded by its level's capacity, which \
                      `NodeInfo` derives from the node size — far below u16::MAX for any node \
                      size this crate emits"
        )]
        let root_nrec = root.records.len() as u16;
        header.extend_from_slice(&root_nrec.to_le_bytes());
        write_uint(&mut header, self.total_records, length_size as usize);
        let checksum = crate::checksum::jenkins_lookup3(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(header.len(), header_size(offset_size, length_size));

        BTreeV2Image { header, nodes }
    }

    /// Depth of the planned tree; 0 means the root is a leaf.
    #[cfg(test)]
    fn depth(&self) -> u16 {
        self.depth
    }
}

/// The fewest records a subtree of `depth` can hold with no node left empty:
/// one record and two children at every level, so `2^(depth + 1) - 1`.
///
/// Saturating, because a depth past 62 is unreachable for any real geometry and
/// the answer there is "more records than exist" either way.
fn min_records(depth: u16) -> u64 {
    1u64.checked_shl(depth as u32 + 1)
        .map_or(u64::MAX, |v| v - 1)
}

/// Plan one subtree of `count` records rooted at `depth`, appending its nodes to
/// `out` and returning the root's index.
///
/// Records are handed out in order as the walk visits their positions, so the
/// in-order traversal of the finished tree yields the caller's record list
/// unchanged.
///
/// `None` when `count` is too small for `depth` to be filled without an empty
/// node. Reported rather than asserted because it is a property of the node and
/// record sizes the caller chose, not of this function.
fn plan_subtree(
    count: usize,
    depth: u16,
    info: &NodeInfo,
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

    // The fewest records this node can keep while its children still hold the
    // rest: each of the `k + 1` subtrees below takes at most `child_capacity`,
    // so `k` must satisfy `(k + 1) * child_capacity >= count - k`. Taking the
    // smallest such `k` fills the subtrees as full as possible, which keeps the
    // tree the same shape sequential insertion would reach.
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

/// Append `value` as a little-endian integer `width` bytes wide. A width of 0
/// writes nothing, which is how an absent child-pointer field is expressed.
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
    use crate::btree_v2::{BTreeV2Header, collect_btree_v2_records};

    const OFFSET_SIZE: u8 = 8;
    const LENGTH_SIZE: u8 = 8;

    /// Records that are just their own index, so a round trip proves both that
    /// every record survived and that the in-order traversal preserved order.
    fn numbered_records(count: usize, record_size: u16) -> Vec<u8> {
        let mut buf = Vec::with_capacity(count * record_size as usize);
        for i in 0..count as u64 {
            let mut rec = vec![0u8; record_size as usize];
            rec[..8].copy_from_slice(&i.to_le_bytes());
            buf.extend_from_slice(&rec);
        }
        buf
    }

    /// Build a tree, then read it back through the parser this crate ships,
    /// returning the record indices in traversal order alongside the depth.
    fn round_trip(count: usize, record_size: u16) -> (u16, Vec<u64>) {
        let plan =
            BTreeV2Plan::new(8, count, record_size, NODE_SIZE, OFFSET_SIZE).expect("plannable");
        let records = numbered_records(count, record_size);

        // Put the header at 0 and the nodes right after it, then parse the
        // whole thing back out of one buffer.
        let nodes_address = StoredAddress::new(header_size(OFFSET_SIZE, LENGTH_SIZE) as u64);
        let image = plan.serialize(&records, nodes_address, OFFSET_SIZE, LENGTH_SIZE);
        let mut file = image.header.clone();
        file.extend_from_slice(&image.nodes);

        let header = BTreeV2Header::parse(&file, 0, OFFSET_SIZE, LENGTH_SIZE).expect("header");
        assert_eq!(header.node_size, NODE_SIZE);
        assert_eq!(header.total_records, count as u64);
        let read =
            collect_btree_v2_records(&file, &header, OFFSET_SIZE, LENGTH_SIZE).expect("read");
        let ids = read
            .iter()
            .map(|r| u64::from_le_bytes(r.data[..8].try_into().expect("8 bytes")))
            .collect();
        (plan.depth(), ids)
    }

    /// The property that matters: whatever the depth, reading the tree back
    /// yields every record exactly once, in the order it was given.
    #[test]
    fn every_record_survives_a_round_trip_in_order() {
        // 29 records fill one 512-byte leaf of 17-byte records, 569 fill a
        // depth-1 tree and 10,259 a depth-2 one, so this crosses both
        // boundaries and lands just inside and just outside each.
        for count in [0, 1, 29, 30, 568, 569, 570, 10_259, 10_260, 40_000] {
            let (_, ids) = round_trip(count, 17);
            assert_eq!(
                ids,
                (0..count as u64).collect::<Vec<_>>(),
                "round trip of {count} records"
            );
        }
    }

    /// The tree grows a level exactly when the level below is full, which is
    /// what keeps a small attribute set byte-identical to the single-leaf trees
    /// this crate emitted before internal nodes existed.
    #[test]
    fn depth_grows_only_when_the_level_below_is_full() {
        assert_eq!(round_trip(29, 17).0, 0);
        assert_eq!(round_trip(30, 17).0, 1);
        assert_eq!(round_trip(569, 17).0, 1);
        assert_eq!(round_trip(570, 17).0, 2);
        assert_eq!(round_trip(10_259, 17).0, 2);
        assert_eq!(round_trip(10_260, 17).0, 3);
    }

    /// A 24-byte huge-object record fits fewer per node than a 17-byte name
    /// record, so the same count reaches a different depth. Both indexes go
    /// through this planner, so both are covered.
    #[test]
    fn a_wider_record_reaches_depth_sooner() {
        let (depth, ids) = round_trip(1_000, 24);
        assert_eq!(ids, (0..1_000u64).collect::<Vec<_>>());
        assert_eq!(depth, 2, "20 records per leaf, 380 per depth-1 subtree");
    }

    /// No node may claim more records than its level's capacity, and no node
    /// may be empty: both would be read back wrong, and an assertion-enabled
    /// libhdf5 aborts on the first.
    #[test]
    fn no_node_exceeds_its_capacity_or_sits_empty() {
        for count in [1usize, 30, 569, 570, 10_260, 40_000] {
            let plan = BTreeV2Plan::new(8, count, 17, NODE_SIZE, OFFSET_SIZE).expect("plannable");
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

    /// Every node is one `node_size` block, which is what lets an address be
    /// computed from an index rather than tracked while serializing.
    #[test]
    fn nodes_are_all_one_node_size_long() {
        let plan = BTreeV2Plan::new(8, 5_000, 17, NODE_SIZE, OFFSET_SIZE).expect("plannable");
        let image = plan.serialize(
            &numbered_records(5_000, 17),
            StoredAddress::new(4_096),
            OFFSET_SIZE,
            LENGTH_SIZE,
        );
        assert_eq!(image.nodes.len() as u64, plan.nodes_size());
        assert_eq!(image.nodes.len() % NODE_SIZE as usize, 0);
    }

    /// A node that fits only one record beside an internal node's pointers has
    /// exactly one valid record count per depth, so almost every count would
    /// need an empty node somewhere. That is reported, not emitted, and not
    /// looped on.
    #[test]
    fn a_shape_needing_an_empty_node_is_refused() {
        // A 256-byte node holds one 200-byte record, at every depth, so the
        // only counts it can express are 1, 3, 7, 15, ...
        assert!(BTreeV2Plan::new(8, 1_000, 200, 256, OFFSET_SIZE).is_none());
        assert!(BTreeV2Plan::new(8, 4, 200, 256, OFFSET_SIZE).is_none());
        assert!(BTreeV2Plan::new(8, 7, 200, 256, OFFSET_SIZE).is_some());
    }
}
