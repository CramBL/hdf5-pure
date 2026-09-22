//! HDF5 B-tree v2 parsing.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "checksum")]
use byteorder::{ByteOrder, LittleEndian};

use crate::address::StoredAddress;
use crate::bytes::{ensure_len, read_length, read_offset};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;

/// Parsed B-tree v2 header (signature "BTHD").
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    /// B-tree type: 5=links indexed by name, 6=links indexed by creation order, etc.
    pub tree_type: u8,
    /// Node size in bytes.
    pub node_size: u32,
    /// Record size in bytes.
    pub record_size: u16,
    /// Depth of the tree (0 = root is a leaf).
    pub depth: u16,
    /// Address of root node.
    pub root_node_address: StoredAddress,
    /// Number of records in the root node.
    pub num_records_in_root: u16,
    /// Total number of records in all nodes.
    pub total_records: u64,
}

/// A single record from a B-tree v2 node.
#[derive(Debug, Clone)]
pub struct BTreeV2Record {
    /// Raw record bytes (record_size bytes).
    pub data: Vec<u8>,
}

/// Compute the number of bytes needed to represent a count, using variable-width encoding.
/// B-tree v2 uses this for the number of records fields in internal nodes.
fn bytes_for_max_records(max_nrec: u64) -> usize {
    if max_nrec == 0 {
        return 1;
    }
    let bits = 64 - max_nrec.leading_zeros() as usize;
    bits.div_ceil(8)
}

/// Read a variable-width unsigned integer (1-8 bytes, LE).
fn read_var_uint(data: &[u8], pos: usize, width: usize) -> Result<u64, FormatError> {
    ensure_len(data, pos, width)?;
    let mut val = 0u64;
    for i in 0..width {
        val |= (data[pos + i] as u64) << (i * 8);
    }
    Ok(val)
}

impl BTreeV2Header {
    /// Parse a B-tree v2 header at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        ensure_len(file_data, offset, 4)?;
        if &file_data[offset..offset + 4] != b"BTHD" {
            return Err(FormatError::InvalidBTreeV2Signature);
        }

        ensure_len(file_data, offset, 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1)?;
        let version = file_data[offset + 4];
        if version != 0 {
            return Err(FormatError::InvalidBTreeV2Version(version));
        }

        let tree_type = file_data[offset + 5];
        let node_size = u32::from_le_bytes([
            file_data[offset + 6],
            file_data[offset + 7],
            file_data[offset + 8],
            file_data[offset + 9],
        ]);
        let record_size = u16::from_le_bytes([file_data[offset + 10], file_data[offset + 11]]);
        let depth = u16::from_le_bytes([file_data[offset + 12], file_data[offset + 13]]);
        let _split_percent = file_data[offset + 14];
        let _merge_percent = file_data[offset + 15];

        let mut pos = offset + 16;
        let root_node_address = StoredAddress::new(read_offset(file_data, pos, offset_size)?);
        pos += offset_size as usize;

        ensure_len(file_data, pos, 2)?;
        let num_records_in_root = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        let total_records = read_length(file_data, pos, length_size)?;
        #[allow(unused_assignments)]
        {
            pos += length_size as usize;
        }

        // Validate header checksum
        #[cfg(feature = "checksum")]
        {
            ensure_len(file_data, pos, 4)?;
            let stored = LittleEndian::read_u32(&file_data[pos..pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&file_data[offset..pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        Ok(BTreeV2Header {
            tree_type,
            node_size,
            record_size,
            depth,
            root_node_address,
            num_records_in_root,
            total_records,
        })
    }

    /// Parse a B-tree v2 header from a [`Source`].
    ///
    /// The header is fully self-contained (signature + fixed fields + a root
    /// pointer + checksum), so only a small bounded window is read.
    pub fn parse_from_source<S: Source + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        // 16 fixed prefix bytes + root address + 2 (num records) + total-records
        // field + 4 checksum; <= 64 with 8-byte offsets/lengths.
        const MAX_HEADER: u64 = 64;
        let window = MAX_HEADER
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let buf = source.read_metadata_at(address, window)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// Compute maximum records per node for a given depth level.
/// leaf: (node_size - overhead) / record_size
/// internal: depends on pointers
fn max_records_leaf(node_size: u32, record_size: u16) -> u64 {
    // Leaf overhead: signature(4) + version(1) + type(1) + checksum(4) = 10
    let overhead = 10u32;
    if node_size <= overhead || record_size == 0 {
        return 0;
    }
    ((node_size - overhead) / record_size as u32) as u64
}

/// The per-level capacities and child-pointer field widths of a v2 B-tree's
/// doubling table, computed exactly as the HDF5 C library does (`H5B2hdr.c`).
///
/// An internal node's child pointer is `address + records-in-child +
/// total-records-in-subtree`. The last two are variable-width integers whose
/// sizes the on-disk format does not store; a reader must recompute them from
/// the node size, record size, and tree depth, or it mis-reads every pointer.
/// The widths are *not* a simple function of the leaf capacity — the
/// per-subtree-total width follows the recurrence
/// `cum_max_nrec[u] = (max_nrec[u] + 1) * cum_max_nrec[u-1] + max_nrec[u]`,
/// and an earlier conservative estimate of it disagreed with the C library at
/// depth 3 and beyond, leaving large groups (tens of thousands of links)
/// unreadable.
///
/// [`btree_v2_write`](crate::btree_v2_write) builds trees from this same table
/// rather than a second copy of the recurrence, so an emitted tree and the
/// widths this module decodes it with cannot drift apart.
pub(crate) struct NodeInfo {
    /// Bytes encoding a child pointer's "number of records in the child node".
    /// HDF5 uses one width at every level, taken from the leaf maximum (the
    /// largest, since `max_nrec` shrinks with depth).
    max_nrec_size: usize,
    /// Bytes encoding a child pointer's "total records in the child's subtree",
    /// indexed by the child node's depth. `[0]` is 0 (a leaf has no subtree
    /// total); `[u]` sizes the field for a child at depth `u`.
    cum_max_nrec_size: Vec<usize>,
    /// Records one node at each depth holds when full. `[0]` is the leaf
    /// capacity, and the sequence is non-increasing (the C library asserts as
    /// much in `H5B2__hdr_init`).
    max_nrec: Vec<u64>,
    /// Records a whole subtree rooted at each depth holds when full.
    cum_max_nrec: Vec<u64>,
}

impl NodeInfo {
    /// The leaf level, before any internal level has been added.
    fn leaf_only(node_size: u32, record_size: u16) -> NodeInfo {
        let max_nrec0 = max_records_leaf(node_size, record_size);
        NodeInfo {
            max_nrec_size: bytes_for_max_records(max_nrec0),
            cum_max_nrec_size: vec![0], // a leaf's pointer carries no subtree total
            max_nrec: vec![max_nrec0],
            cum_max_nrec: vec![max_nrec0],
        }
    }

    /// Extend the table by one internal level above the current top.
    fn push_level(&mut self, node_size: u32, record_size: u16, offset_size: u8) {
        let u = self.max_nrec.len();
        // Internal-pointer size at this level uses the *previous* level's
        // subtree-total width (H5B2_INT_POINTER_SIZE).
        let int_ptr = offset_size as usize + self.max_nrec_size + self.cum_max_nrec_size[u - 1];
        // Records that fit an internal node at this level (H5B2_NUM_INT_REC).
        let avail = (node_size as usize).saturating_sub(10 + int_ptr);
        let denom = record_size as usize + int_ptr;
        let max_nrec_u = avail.checked_div(denom).unwrap_or(0) as u64;
        // cum_max_nrec[u] = (max_nrec[u] + 1) * cum_max_nrec[u-1] + max_nrec[u]
        let cum = max_nrec_u
            .saturating_add(1)
            .saturating_mul(self.cum_max_nrec[u - 1])
            .saturating_add(max_nrec_u);
        self.cum_max_nrec_size.push(bytes_for_max_records(cum));
        self.max_nrec.push(max_nrec_u);
        self.cum_max_nrec.push(cum);
    }

    /// Build the doubling table for a tree of the given root `depth`.
    fn compute(node_size: u32, record_size: u16, offset_size: u8, depth: u16) -> NodeInfo {
        let mut info = NodeInfo::leaf_only(node_size, record_size);
        for _ in 1..=depth {
            info.push_level(node_size, record_size, offset_size);
        }
        info
    }

    /// Build the doubling table deep enough to hold `records`, returning it with
    /// the depth a tree of that many records needs.
    ///
    /// `None` when no tree of this node and record size can hold them: an
    /// internal level that fits no records at all cannot make the tree taller,
    /// so growing further would loop forever.
    pub(crate) fn for_record_count(
        node_size: u32,
        record_size: u16,
        offset_size: u8,
        records: u64,
    ) -> Option<(NodeInfo, u16)> {
        let mut info = NodeInfo::leaf_only(node_size, record_size);
        let mut depth = 0u16;
        while *info.cum_max_nrec.last().expect("table is never empty") < records {
            info.push_level(node_size, record_size, offset_size);
            depth += 1;
            if *info.max_nrec.last().expect("just pushed") == 0 {
                return None;
            }
        }
        Some((info, depth))
    }

    /// Records one node at `depth` holds when full.
    pub(crate) fn max_nrec(&self, depth: u16) -> u64 {
        self.max_nrec[depth as usize]
    }

    /// Records a whole subtree rooted at `depth` holds when full.
    pub(crate) fn cum_max_nrec(&self, depth: u16) -> u64 {
        self.cum_max_nrec[depth as usize]
    }

    /// Width of a child pointer's "number of records in the child node" field,
    /// one width at every level.
    pub(crate) fn max_nrec_size(&self) -> usize {
        self.max_nrec_size
    }

    /// Width of a child pointer's "total records in subtree" field for a node at
    /// `depth` (its children sit one level below, so the field has width
    /// `cum_max_nrec_size[depth - 1]`; for `depth == 1` the children are leaves
    /// and the field is absent).
    pub(crate) fn total_nrec_size(&self, depth: u16) -> usize {
        self.cum_max_nrec_size
            .get((depth - 1) as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Full on-disk width of one child pointer for a node at `depth`.
    fn child_ptr_size(&self, depth: u16, offset_size: u8) -> usize {
        offset_size as usize + self.max_nrec_size + self.total_nrec_size(depth)
    }
}

/// Collect all records from a B-tree v2 by traversing from the root.
pub fn collect_btree_v2_records(
    file_data: &[u8],
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }

    if header.depth == 0 {
        // Root is a leaf
        parse_leaf_records(
            file_data,
            header.root_node_address.get().to_usize()?,
            header.num_records_in_root,
            header.record_size,
        )
    } else {
        // Root is internal; traverse recursively
        let node_info = NodeInfo::compute(
            header.node_size,
            header.record_size,
            offset_size,
            header.depth,
        );
        let mut records = Vec::new();
        collect_internal_records(
            file_data,
            header.root_node_address.get().to_usize()?,
            header.num_records_in_root,
            header.depth,
            header.record_size,
            header.node_size,
            offset_size,
            length_size,
            &node_info,
            &mut records,
        )?;
        Ok(records)
    }
}

/// Parse records from a leaf node (signature "BTLF").
fn parse_leaf_records(
    file_data: &[u8],
    offset: usize,
    num_records: u16,
    record_size: u16,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    // signature(4) + version(1) + type(1) = 6 bytes header
    ensure_len(file_data, offset, 6)?;
    if &file_data[offset..offset + 4] != b"BTLF" {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let pos = offset + 6;
    let rs = record_size as usize;
    let total = num_records as usize * rs;
    ensure_len(file_data, pos, total)?;

    // Validate checksum: 4 bytes after records + padding
    #[cfg(feature = "checksum")]
    {
        let checksum_pos = pos + total;
        if file_data.len() >= checksum_pos + 4 {
            let stored = LittleEndian::read_u32(&file_data[checksum_pos..checksum_pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&file_data[offset..checksum_pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }
    }

    let mut records = Vec::with_capacity(num_records as usize);
    for i in 0..num_records as usize {
        let start = pos + i * rs;
        records.push(BTreeV2Record {
            data: file_data[start..start + rs].to_vec(),
        });
    }
    Ok(records)
}

/// Parse an internal node's child pointers from its node bytes (offset 0 = the
/// "BTIN" signature), returning the `(child_address, child_num_records)` list.
///
/// The node's own records sit at `node[6 + i * record_size ..]`; the caller
/// reads them while interleaving child traversals. Shared by the buffered and
/// streaming collectors so the child-pointer-width logic lives in one place.
fn parse_internal_child_pointers(
    node: &[u8],
    num_records: u16,
    depth: u16,
    record_size: u16,
    offset_size: u8,
    node_info: &NodeInfo,
) -> Result<Vec<(StoredAddress, u16)>, FormatError> {
    // signature(4) + version(1) + type(1) = 6
    ensure_len(node, 0, 6)?;
    if &node[0..4] != b"BTIN" {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let nr = num_records as usize;
    let rs = record_size as usize;
    // Records come first, then the child pointers.
    let mut pos = 6;
    ensure_len(node, pos, nr * rs)?;
    pos += nr * rs;

    // Child-pointer field widths, computed exactly from the doubling table:
    // the records-in-child field is one width at every level, and the
    // subtree-total field's width is that of the child's depth (`depth - 1`).
    let nrec_width = node_info.max_nrec_size;
    let total_nrec_width = node_info.total_nrec_size(depth);

    let num_children = nr + 1;
    let child_ptr_size = node_info.child_ptr_size(depth, offset_size);
    ensure_len(node, pos, num_children * child_ptr_size)?;

    let mut children = Vec::with_capacity(num_children);
    for _ in 0..num_children {
        let addr = StoredAddress::new(read_offset(node, pos, offset_size)?);
        pos += offset_size as usize;
        // `read_var_uint` returns a value spanning `nrec_width` bytes, and
        // `max_nrec_size` is sized to the node's record capacity, which the v2
        // b-tree format keeps within a 2-byte field — so the count fits `u16`.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "records-per-node fits max_nrec_size (<= 2 bytes for valid files)"
        )]
        let child_nrec = read_var_uint(node, pos, nrec_width)? as u16;
        pos += nrec_width;
        pos += total_nrec_width; // skip total-records-in-subtree
        children.push((addr, child_nrec));
    }

    Ok(children)
}

/// Recursively collect records from an internal node (buffered path).
#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn collect_internal_records(
    file_data: &[u8],
    offset: usize,
    num_records: u16,
    depth: u16,
    record_size: u16,
    node_size: u32,
    offset_size: u8,
    length_size: u8,
    node_info: &NodeInfo,
    out: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    ensure_len(file_data, offset, 6)?;
    let node = &file_data[offset..];
    let children = parse_internal_child_pointers(
        node,
        num_records,
        depth,
        record_size,
        offset_size,
        node_info,
    )?;

    let nr = num_records as usize;
    let rs = record_size as usize;
    let child_depth = depth - 1;

    // Interleave: child[0], record[0], child[1], record[1], ..., child[nr].
    for (i, &(child_addr, child_nrec)) in children.iter().enumerate() {
        if child_depth == 0 {
            out.extend(parse_leaf_records(
                file_data,
                child_addr.get().to_usize()?,
                child_nrec,
                record_size,
            )?);
        } else {
            collect_internal_records(
                file_data,
                child_addr.get().to_usize()?,
                child_nrec,
                child_depth,
                record_size,
                node_size,
                offset_size,
                length_size,
                node_info,
                out,
            )?;
        }

        if i < nr {
            let start = 6 + i * rs;
            out.push(BTreeV2Record {
                data: node[start..start + rs].to_vec(),
            });
        }
    }

    Ok(())
}

/// Collect all records from a B-tree v2 by traversing from the root, reading
/// each fixed-size node from a [`Source`] on demand rather than indexing a
/// whole-file buffer.
pub fn collect_btree_v2_records_from_source<S: Source + ?Sized>(
    source: &S,
    header: &BTreeV2Header,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }
    let node_info = NodeInfo::compute(
        header.node_size,
        header.record_size,
        offset_size,
        header.depth,
    );
    let mut records = Vec::new();
    collect_node_from_source(
        source,
        header.root_node_address.get(),
        header.num_records_in_root,
        header.depth,
        header.record_size,
        header.node_size,
        offset_size,
        &node_info,
        &mut records,
    )?;
    Ok(records)
}

/// Read and collect one node (leaf or internal) from the source, recursing into
/// children. Mirrors the buffered [`collect_btree_v2_records`] traversal and
/// produces records in the same order.
#[allow(clippy::too_many_arguments)]
fn collect_node_from_source<S: Source + ?Sized>(
    source: &S,
    address: u64,
    num_records: u16,
    depth: u16,
    record_size: u16,
    node_size: u32,
    offset_size: u8,
    node_info: &NodeInfo,
    out: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    // Every node occupies `node_size` bytes; read that window (clamped to the
    // bytes available, in case the final node abuts EOF).
    let node_len = u64::from(node_size)
        .min(source.len().saturating_sub(address))
        .to_usize()?;
    let node = source.read_metadata_at(address, node_len)?;

    if depth == 0 {
        out.extend(parse_leaf_records(&node, 0, num_records, record_size)?);
        return Ok(());
    }

    let children = parse_internal_child_pointers(
        &node,
        num_records,
        depth,
        record_size,
        offset_size,
        node_info,
    )?;

    let nr = num_records as usize;
    let rs = record_size as usize;
    let child_depth = depth - 1;
    for (i, &(child_addr, child_nrec)) in children.iter().enumerate() {
        collect_node_from_source(
            source,
            child_addr.get(),
            child_nrec,
            child_depth,
            record_size,
            node_size,
            offset_size,
            node_info,
            out,
        )?;
        if i < nr {
            let start = 6 + i * rs;
            out.push(BTreeV2Record {
                data: node[start..start + rs].to_vec(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_util::btree_v2;
    use test_util::image::Image;
    use test_util::widths::Widths;

    fn build_btree_v2_header(
        tree_type: u8,
        node_size: u32,
        record_size: u16,
        depth: u16,
        root_addr: u64,
        num_records_root: u16,
        total_records: u64,
        widths: Widths,
    ) -> Vec<u8> {
        btree_v2::Header::new(tree_type, record_size, root_addr, num_records_root)
            .node_size(node_size)
            .depth(depth)
            .total_records(total_records)
            .build(widths)
    }

    fn build_leaf_node(tree_type: u8, records: &[&[u8]]) -> Vec<u8> {
        let records: Vec<_> = records.iter().map(|record| record.to_vec()).collect();
        btree_v2::leaf(tree_type, &records)
    }

    /// An 11-byte record carrying its in-order id in byte 0.
    fn rec(id: u8) -> Vec<u8> {
        let mut record = vec![0u8; 11];
        record[0] = id;
        record
    }

    /// Append a leaf node, returning its address.
    fn put_leaf(file: &mut Image, tree_type: u8, records: &[Vec<u8>]) -> u64 {
        file.append(&btree_v2::leaf(tree_type, records)) as u64
    }

    /// Append an internal node. `children` is `(addr, records-in-child,
    /// total-records-in-subtree)`, and `max_nrec_size` and `total_width` are
    /// the doubling-table field widths for the node's level. A node directly
    /// above the leaves has no subtree field, so `total_width` is `None`.
    fn put_internal(
        file: &mut Image,
        tree_type: u8,
        records: &[Vec<u8>],
        children: &[(u64, u16, u64)],
        max_nrec_size: usize,
        total_width: Option<usize>,
    ) -> u64 {
        let children: Vec<_> = children
            .iter()
            .map(|&(address, records, subtree_records)| btree_v2::Child {
                address,
                records,
                records_width: max_nrec_size,
                subtree: total_width.map(|width| btree_v2::Subtree {
                    records: subtree_records,
                    width,
                }),
            })
            .collect();
        file.append(&btree_v2::internal(tree_type, records, &children, WIDTHS)) as u64
    }

    /// A hand-built depth-3 B-tree (the same node/record sizes the C library
    /// uses for a dense group's name index) must be traversed in record order.
    /// This is the regression the field-width fix targets: at depth 3 the
    /// subtree-total field is 2 bytes (`cum_max_nrec_size[2]`), and the earlier
    /// over-estimate read it as 3, misaligning every root child pointer.
    #[test]
    fn reads_depth_3_btree() {
        // For node_size=512, record_size=11, 8-byte offsets: max_nrec_size = 1,
        // cum_max_nrec_size = [0, 2, 2]; so depth-1 child pointers carry no
        // subtree total, while depth-2 and depth-3 pointers carry a 2-byte one.
        let mut image = Image::new();
        // The header occupies the front, and the tree follows it.
        image.place(0, &[0u8; 64]);

        // Eight leaves, each one record; four depth-1 nodes; two depth-2 nodes;
        // one depth-3 root. In-order traversal yields ids 0..15.
        let leaf = |f: &mut Image, id: u8| put_leaf(f, 5, &[rec(id)]);
        let l0 = leaf(&mut image, 0);
        let l2 = leaf(&mut image, 2);
        let l4 = leaf(&mut image, 4);
        let l6 = leaf(&mut image, 6);
        let l8 = leaf(&mut image, 8);
        let l10 = leaf(&mut image, 10);
        let l12 = leaf(&mut image, 12);
        let l14 = leaf(&mut image, 14);

        // Depth-1 internal nodes (children are leaves: no subtree total).
        let n1 = put_internal(&mut image, 5, &[rec(1)], &[(l0, 1, 1), (l2, 1, 1)], 1, None);
        let n2 = put_internal(&mut image, 5, &[rec(5)], &[(l4, 1, 1), (l6, 1, 1)], 1, None);
        let n3 = put_internal(
            &mut image,
            5,
            &[rec(9)],
            &[(l8, 1, 1), (l10, 1, 1)],
            1,
            None,
        );
        let n4 = put_internal(
            &mut image,
            5,
            &[rec(13)],
            &[(l12, 1, 1), (l14, 1, 1)],
            1,
            None,
        );

        // Depth-2 internal nodes (children are depth-1: total_width = 2).
        let m1 = put_internal(
            &mut image,
            5,
            &[rec(3)],
            &[(n1, 1, 3), (n2, 1, 3)],
            1,
            Some(2),
        );
        let m2 = put_internal(
            &mut image,
            5,
            &[rec(11)],
            &[(n3, 1, 3), (n4, 1, 3)],
            1,
            Some(2),
        );

        // Depth-3 root (children are depth-2: total_width = 2).
        let root = put_internal(
            &mut image,
            5,
            &[rec(7)],
            &[(m1, 1, 7), (m2, 1, 7)],
            1,
            Some(2),
        );

        // Lay the header (root address, depth 3, 15 total records) at the front.
        let header = build_btree_v2_header(5, 512, 11, 3, root, 1, 15, WIDTHS);
        image.place(0, &header);
        let file = image.build();

        let hdr = BTreeV2Header::parse(&file, 0, 8, 8).unwrap();
        let ids: Vec<u8> = collect_btree_v2_records(&file, &hdr, 8, 8)
            .unwrap()
            .iter()
            .map(|r| r.data[0])
            .collect();
        assert_eq!(ids, (0u8..15).collect::<Vec<_>>());

        // The streaming collector must agree.
        #[cfg(feature = "std")]
        {
            use crate::source::BytesSource;
            let src = BytesSource::new(&file);
            let hdr_s = BTreeV2Header::parse_from_source(&src, 0, 8, 8).unwrap();
            let ids_s: Vec<u8> = collect_btree_v2_records_from_source(&src, &hdr_s, 8, 8)
                .unwrap()
                .iter()
                .map(|r| r.data[0])
                .collect();
            assert_eq!(ids_s, (0u8..15).collect::<Vec<_>>());
        }
    }

    #[test]
    fn parse_header() {
        let data = build_btree_v2_header(5, 512, 11, 0, 0x1000, 3, 3, WIDTHS);
        let hdr = BTreeV2Header::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.tree_type, 5);
        assert_eq!(hdr.node_size, 512);
        assert_eq!(hdr.record_size, 11);
        assert_eq!(hdr.depth, 0);
        assert_eq!(hdr.root_node_address, StoredAddress::new(0x1000));
        assert_eq!(hdr.num_records_in_root, 3);
        assert_eq!(hdr.total_records, 3);
    }

    #[test]
    fn node_info_matches_hdf5_widths() {
        // Real name-index B-tree parameters (node 512, record 11, 8-byte
        // offsets), hand-verified against H5B2hdr.c. The depth-3 regression:
        // a depth-3 root's child pointer is 11 bytes (8 + max_nrec_size 1 +
        // cum_max_nrec_size[2] 2), not 12. The earlier estimate produced a
        // 3-byte subtree-total field (from 45^3 = 91125) instead of the exact
        // 2 (from cum_max_nrec[2] = 26449), misaligning every pointer and making
        // groups of ~26k+ links unreadable.
        let ni = NodeInfo::compute(512, 11, 8, 3);
        assert_eq!(ni.max_nrec_size, 1);
        assert_eq!(ni.cum_max_nrec_size, vec![0, 2, 2, 3]);
        assert_eq!(ni.child_ptr_size(1, 8), 9); // depth-1 children are leaves
        assert_eq!(ni.child_ptr_size(2, 8), 11);
        assert_eq!(ni.child_ptr_size(3, 8), 11);

        // A larger leaf capacity needs a 2-byte per-node record count:
        // (4096 - 10) / 8 = 510 records, and enc(510) = 2. This also guards the
        // old `enc(max_leaf * 2)` mistake for the records-in-child field.
        let big = NodeInfo::compute(4096, 8, 8, 1);
        assert_eq!(big.max_nrec_size, 2);
    }

    #[test]
    fn parse_leaf_with_2_records() {
        let rec1 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let rec2 = [11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let leaf = build_leaf_node(5, &[&rec1, &rec2]);

        let leaf_offset = 256usize;
        let header = build_btree_v2_header(5, 512, 11, 0, leaf_offset as u64, 2, 2, WIDTHS);

        let mut file_data = vec![0u8; 512];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[leaf_offset..leaf_offset + leaf.len()].copy_from_slice(&leaf);

        let hdr = BTreeV2Header::parse(&file_data, 0, 8, 8).unwrap();
        let records = collect_btree_v2_records(&file_data, &hdr, 8, 8).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data, rec1.to_vec());
        assert_eq!(records[1].data, rec2.to_vec());
    }

    #[test]
    fn invalid_signature() {
        let mut data = build_btree_v2_header(5, 512, 11, 0, 0, 0, 0, WIDTHS);
        data[0] = b'X';
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn invalid_version() {
        let mut data = build_btree_v2_header(5, 512, 11, 0, 0, 0, 0, WIDTHS);
        data[4] = 1; // bad version
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Version(1));
    }

    #[test]
    fn empty_tree() {
        let header = build_btree_v2_header(5, 512, 11, 0, 0, 0, 0, WIDTHS);
        let hdr = BTreeV2Header::parse(&header, 0, 8, 8).unwrap();
        let records = collect_btree_v2_records(&header, &hdr, 8, 8).unwrap();
        assert!(records.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_btree_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let rec1 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let rec2 = [11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let leaf = build_leaf_node(5, &[&rec1, &rec2]);
        let leaf_offset = 256usize;
        let header = build_btree_v2_header(5, 512, 11, 0, leaf_offset as u64, 2, 2, WIDTHS);
        let mut file_data = vec![0u8; 512];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[leaf_offset..leaf_offset + leaf.len()].copy_from_slice(&leaf);

        let hdr = BTreeV2Header::parse(&file_data, 0, 8, 8).unwrap();
        let buffered: Vec<_> = collect_btree_v2_records(&file_data, &hdr, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        let mem = BytesSource::new(&file_data);
        let hdr_mem = BTreeV2Header::parse_from_source(&mem, 0, 8, 8).unwrap();
        assert_eq!(hdr_mem.root_node_address, hdr.root_node_address);
        let from_mem: Vec<_> = collect_btree_v2_records_from_source(&mem, &hdr_mem, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
        let hdr_seek = BTreeV2Header::parse_from_source(&seek, 0, 8, 8).unwrap();
        let from_seek: Vec<_> = collect_btree_v2_records_from_source(&seek, &hdr_seek, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(from_seek.len(), 2);
    }

    const WIDTHS: Widths = Widths::EIGHT;
}
