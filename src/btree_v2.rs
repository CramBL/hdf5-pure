//! Traverses version 2 B-trees to read their records or enumerate their allocated storage.
//!
//! "Version 2 B-trees" of the [format specification, version 4.0][spec] defines separate `BTHD`
//! header, `BTIN` internal-node, and `BTLF` leaf-node layouts. The header's "Node Size" field is
//! the size in bytes of every B-tree node, not of the header. Its "Root Node Address" is undefined
//! for a tree with no records. Otherwise, "Depth" determines whether that root is a leaf or an
//! internal node. Record walks preserve tree order. [`collect_btree_v2_storage_extents`] returns
//! the header allocation and every reachable full "Node Size" allocation in file-address order.
//!
//! [spec]: https://support.hdfgroup.org/releases/hdf5/2.1.0/documentation/hdf5-2.1.0.doxygen/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2

use core::num::NonZeroU16;

use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2NodeInfo;
pub use hdf5_pure_format::__private::BTreeV2Record;
use hdf5_pure_space::__private::Extent;

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;
use crate::width::{LengthWidth, OffsetWidth};

/// Returns the allocations reachable from a version 2 B-tree header in file-address order.
///
/// "Version 2 B-trees" of the [format specification, version 4.0][spec] defines the `BTHD` header
/// separately from `BTIN` internal nodes and `BTLF` leaf nodes. Its "Node Size" field is the size
/// in bytes of every B-tree node. The returned [`Extent`] values therefore contain the exact
/// encoded header and one full [`BTreeV2Header::node_size`] allocation for every reachable node,
/// regardless of that node's current record count.
///
/// "Depth" is 0 where the root is a leaf. A `BTIN` node with R records stores R + 1 child-pointer
/// groups. Each group contains a "Child Node Pointer" and "Number of Records in Child Node". Where
/// that child is itself an internal node, the group also contains "Total Number of Records in Child
/// Node". The walk decrements the declared depth and uses the child record count at the next node.
/// A tree with zero total records has an undefined "Root Node Address" and contributes only its
/// header extent.
///
/// Repeated node addresses and overlapping allocations are errors. The walk also proves each
/// complete allocation lies within [`Source::len`] before reading it, so a truncated node cannot
/// become a shorter owned extent.
///
/// # Errors
///
/// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a file width
/// is not 2, 4, or 8, [`FormatError::InvalidBTreeV2Signature`] if a required `BTHD`, `BTIN`, or
/// `BTLF` signature is absent, and [`FormatError::InvalidBTreeV2Version`] if the header version is
/// not 0. With the `checksum` feature, returns [`FormatError::ChecksumMismatch`] for an invalid
/// header checksum.
///
/// Returns [`FormatError::UnexpectedEof`] if a complete header or node allocation, or the declared
/// contents of a node, exceeds the bytes available, and [`FormatError::OffsetOverflow`] if an
/// allocation crosses the `u64` address-space end. Repeated node addresses and overlapping owned
/// allocations also return [`FormatError::InvalidBTreeV2Signature`]. The error from `source` is
/// returned if a metadata read fails.
///
/// [spec]: https://support.hdfgroup.org/releases/hdf5/2.1.0/documentation/hdf5-2.1.0.doxygen/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[allow(
    dead_code,
    reason = "storage ownership is an internal structural capability without a production caller"
)]
pub(crate) fn collect_btree_v2_storage_extents<S: Source + ?Sized>(
    source: &S,
    header_address: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<Extent>, FormatError> {
    let offset_width = OffsetWidth::try_from(offset_size)?;
    let length_width = LengthWidth::try_from(length_size)?;
    // "Node Size" applies to BTIN and BTLF nodes. BTHD has its own layout and encoded length.
    let header_len = hdf5_pure_format::__private::btree_v2_header_size(offset_width, length_width);
    let header_extent = extent_in_source(source, header_address.get(), header_len.to_u64())?;
    let header_bytes = source.read_metadata_at(header_address.get(), header_len)?;
    let header = BTreeV2Header::parse(&header_bytes, 0, offset_size, length_size)?;
    // BTIN and BTLF both have a 6-byte signature/version/type prefix and a 4-byte checksum.
    if header.node_size < 10 {
        return Err(FormatError::UnexpectedEof {
            expected: 10,
            available: header.node_size as usize,
        });
    }

    let mut extents = Vec::with_capacity(1);
    extents.push(header_extent);
    if header.total_records == 0 {
        return Ok(extents);
    }

    let node_info = BTreeV2NodeInfo::compute(
        header.node_size,
        header.record_size,
        offset_size,
        header.depth,
    );
    let mut visited = BTreeSet::new();
    collect_storage_node(
        source,
        header.root_node_address,
        header.num_records_in_root,
        header.depth,
        &header,
        offset_size,
        &node_info,
        &mut visited,
        &mut extents,
    )?;

    extents.sort_unstable();
    for pair in extents.windows(2) {
        if pair[0].end() > pair[1].start() {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
    }
    Ok(extents)
}

/// Adds one complete node allocation and recursively follows the children its depth permits.
///
/// "Version 2 B-trees" gives an internal node with R records R + 1 child-pointer groups. Each
/// group contains "Child Node Pointer" and "Number of Records in Child Node". A pointer whose child
/// is also an internal node additionally contains "Total Number of Records in Child Node". "Depth"
/// 0 is a leaf, so recursion stops there after validating the leaf layout.
///
/// [spec]: https://support.hdfgroup.org/releases/hdf5/2.1.0/documentation/hdf5-2.1.0.doxygen/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[allow(clippy::too_many_arguments)]
fn collect_storage_node<S: Source + ?Sized>(
    source: &S,
    address: StoredAddress,
    num_records: u16,
    depth: u16,
    header: &BTreeV2Header,
    offset_size: u8,
    node_info: &BTreeV2NodeInfo,
    visited: &mut BTreeSet<StoredAddress>,
    extents: &mut Vec<Extent>,
) -> Result<(), FormatError> {
    if !visited.insert(address) {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let extent = extent_in_source(source, address.get(), u64::from(header.node_size))?;
    let node = source.read_metadata_at(address.get(), u64::from(header.node_size).to_usize()?)?;
    extents.push(extent);

    let Some(depth) = NonZeroU16::new(depth) else {
        return validate_storage_leaf(&node, num_records, header.record_size);
    };
    let children = hdf5_pure_format::__private::parse_btree_v2_internal_child_pointers(
        &node,
        num_records,
        depth,
        header.record_size,
        offset_size,
        node_info,
    )?;
    for (child_address, child_records) in children {
        collect_storage_node(
            source,
            child_address,
            child_records,
            depth.get() - 1,
            header,
            offset_size,
            node_info,
            visited,
            extents,
        )?;
    }
    Ok(())
}

/// Proves the declared leaf contents fit inside the node allocation.
///
/// "Version 2 B-trees" lays out a leaf as the `BTLF` signature, version and type, R records of the
/// header's "Record Size", and the checksum. This check requires the records and checksum of that
/// declared layout to fit inside the already bounded "Node Size" allocation.
///
/// [spec]: https://support.hdfgroup.org/releases/hdf5/2.1.0/documentation/hdf5-2.1.0.doxygen/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
fn validate_storage_leaf(
    node: &[u8],
    num_records: u16,
    record_size: u16,
) -> Result<(), FormatError> {
    bytes::ensure_len(node, 0, 6)?;
    if &node[..4] != b"BTLF" {
        return Err(FormatError::InvalidBTreeV2Signature);
    }
    let record_bytes = u64::from(num_records) * u64::from(record_size);
    bytes::ensure_len(node, 6, (record_bytes + 4).to_usize()?)
}

/// Returns a checked extent after proving its complete byte range lies within `source`.
fn extent_in_source<S: Source + ?Sized>(
    source: &S,
    address: u64,
    length: u64,
) -> Result<Extent, FormatError> {
    debug_assert!(length > 0, "storage extents are non-empty");
    let extent = Extent::new(address, length).ok_or(FormatError::OffsetOverflow {
        offset: address,
        length,
    })?;
    if extent.end() > source.len() {
        return Err(FormatError::UnexpectedEof {
            expected: extent.end().to_usize().unwrap_or(usize::MAX),
            available: source.len().to_usize().unwrap_or(usize::MAX),
        });
    }
    Ok(extent)
}

/// Collects all records from a version 2 B-tree by traversing from the root.
pub fn collect_btree_v2_records(
    file_data: &[u8],
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }

    let Some(depth) = NonZeroU16::new(header.depth) else {
        // Root is a leaf
        return hdf5_pure_format::__private::parse_btree_v2_leaf_records(
            file_data,
            header.root_node_address.get().to_usize()?,
            header.num_records_in_root,
            header.record_size,
        );
    };
    // Root is internal: traverse recursively
    let node_info = BTreeV2NodeInfo::compute(
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
        depth,
        header.record_size,
        header.node_size,
        offset_size,
        length_size,
        &node_info,
        &mut records,
    )?;
    Ok(records)
}

/// Recursively collects records from an internal node in the buffered path.
#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn collect_internal_records(
    file_data: &[u8],
    offset: usize,
    num_records: u16,
    depth: NonZeroU16,
    record_size: u16,
    node_size: u32,
    offset_size: u8,
    length_size: u8,
    node_info: &BTreeV2NodeInfo,
    out: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    bytes::ensure_len(file_data, offset, 6)?;
    let node = &file_data[offset..];
    let children = hdf5_pure_format::__private::parse_btree_v2_internal_child_pointers(
        node,
        num_records,
        depth,
        record_size,
        offset_size,
        node_info,
    )?;

    let nr = num_records as usize;
    let rs = record_size as usize;
    let child_depth = NonZeroU16::new(depth.get() - 1);

    // Interleave: child[0], record[0], child[1], record[1], ..., child[nr].
    for (i, &(child_addr, child_nrec)) in children.iter().enumerate() {
        if let Some(child_depth) = child_depth {
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
        } else {
            out.extend(hdf5_pure_format::__private::parse_btree_v2_leaf_records(
                file_data,
                child_addr.get().to_usize()?,
                child_nrec,
                record_size,
            )?);
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

/// Collects all records from a version 2 B-tree through a [`Source`].
///
/// Each fixed-size node is read on demand from the source.
pub fn collect_btree_v2_records_from_source<S: Source + ?Sized>(
    source: &S,
    header: &BTreeV2Header,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }
    let node_info = BTreeV2NodeInfo::compute(
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

/// Reads one node from the source and recursively collects its records in tree order.
///
/// This mirrors the buffered [`collect_btree_v2_records`] traversal.
#[allow(clippy::too_many_arguments)]
fn collect_node_from_source<S: Source + ?Sized>(
    source: &S,
    address: u64,
    num_records: u16,
    depth: u16,
    record_size: u16,
    node_size: u32,
    offset_size: u8,
    node_info: &BTreeV2NodeInfo,
    out: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    // Every node occupies `node_size` bytes; read that window (clamped to the
    // bytes available, in case the final node abuts EOF).
    let node_len = u64::from(node_size)
        .min(source.len().saturating_sub(address))
        .to_usize()?;
    let node = source.read_metadata_at(address, node_len)?;

    let Some(depth) = NonZeroU16::new(depth) else {
        out.extend(hdf5_pure_format::__private::parse_btree_v2_leaf_records(
            &node,
            0,
            num_records,
            record_size,
        )?);
        return Ok(());
    };

    let children = hdf5_pure_format::__private::parse_btree_v2_internal_child_pointers(
        &node,
        num_records,
        depth,
        record_size,
        offset_size,
        node_info,
    )?;

    let nr = num_records as usize;
    let rs = record_size as usize;
    let child_depth = depth.get() - 1;
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
    #[cfg(feature = "std")]
    use std::io::Cursor;

    use hdf5_pure_format::__private::BTREE_V2_NODE_SIZE;
    use hdf5_pure_format::__private::BTreeV2Plan;
    use test_util::btree_v2;
    use test_util::image::Image;
    use test_util::widths::Widths;

    use super::*;
    use crate::source::BytesSource;
    #[cfg(feature = "std")]
    use crate::source::ReadSeekSource;

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

    /// Pads one encoded node to its full allocation size.
    fn node_allocation(node: Vec<u8>) -> Vec<u8> {
        assert!(node.len() <= BTREE_V2_NODE_SIZE as usize);
        let mut allocation = vec![0u8; BTREE_V2_NODE_SIZE as usize];
        allocation[..node.len()].copy_from_slice(&node);
        allocation
    }

    /// Appends one full leaf-node allocation and returns its address.
    fn put_leaf(file: &mut Image, tree_type: u8, records: &[Vec<u8>]) -> u64 {
        file.append(&node_allocation(btree_v2::leaf(tree_type, records)))
            .to_u64()
    }

    /// Appends one full internal-node allocation and returns its address.
    ///
    /// `children` holds `(address, records in child, records in subtree)`. `max_nrec_size` and
    /// `total_width` are the record-count widths for this level. A node directly above the leaves
    /// has no subtree-total field, so its `total_width` is `None`.
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
        file.append(&node_allocation(btree_v2::internal(
            tree_type, records, &children, WIDTHS,
        )))
        .to_u64()
    }

    struct Depth3Tree {
        file: Vec<u8>,
        nodes: [u64; 15],
    }

    /// Builds the depth-3 dense-name-index geometry used by the record and storage walks.
    fn depth_3_tree() -> Depth3Tree {
        // For 512-byte nodes with 11-byte records and 8-byte offsets, child record counts use 1
        // byte. Cumulative child counts use [0, 2, 2] bytes. Depth-1 child pointers have no subtree
        // total, while depth-2 and depth-3 pointers have a 2-byte one.
        let mut image = Image::new();
        image.place(0, &[0u8; 64]);

        // Eight leaves, four depth-1 nodes, two depth-2 nodes, and one depth-3 root. In-order
        // traversal yields record ids 0..15.
        let leaf = |f: &mut Image, id: u8| put_leaf(f, 5, &[rec(id)]);
        let l0 = leaf(&mut image, 0);
        let l2 = leaf(&mut image, 2);
        let l4 = leaf(&mut image, 4);
        let l6 = leaf(&mut image, 6);
        let l8 = leaf(&mut image, 8);
        let l10 = leaf(&mut image, 10);
        let l12 = leaf(&mut image, 12);
        let l14 = leaf(&mut image, 14);

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
        let root = put_internal(
            &mut image,
            5,
            &[rec(7)],
            &[(m1, 1, 7), (m2, 1, 7)],
            1,
            Some(2),
        );

        let header = btree_v2::Header::new(5, 11, root, 1)
            .depth(3)
            .total_records(15)
            .build(WIDTHS);
        image.place(0, &header);

        Depth3Tree {
            file: image.build(),
            nodes: [
                l0, l2, l4, l6, l8, l10, l12, l14, n1, n2, n3, n4, m1, m2, root,
            ],
        }
    }

    /// A hand-built depth-3 B-tree uses the subtree-count widths the C library computes for a
    /// dense group's name index and traverses in record order.
    #[test]
    fn reads_depth_3_btree() {
        let tree = depth_3_tree();
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let ids: Vec<u8> = collect_btree_v2_records(&tree.file, &header, OFFSET_SIZE, LENGTH_SIZE)
            .unwrap()
            .iter()
            .map(|record| record.data[0])
            .collect();
        assert_eq!(ids, (0u8..15).collect::<Vec<_>>());

        #[cfg(feature = "std")]
        {
            use crate::source::SourceMetadata;

            let source = BytesSource::new(&tree.file);
            let header = BTreeV2Header::parse_from_source(
                &SourceMetadata(&source),
                0,
                OFFSET_SIZE,
                LENGTH_SIZE,
            )
            .unwrap();
            let ids: Vec<u8> =
                collect_btree_v2_records_from_source(&source, &header, OFFSET_SIZE, LENGTH_SIZE)
                    .unwrap()
                    .iter()
                    .map(|record| record.data[0])
                    .collect();
            assert_eq!(ids, (0u8..15).collect::<Vec<_>>());
        }
    }

    #[test]
    fn storage_walk_of_a_leaf_returns_header_and_full_node() {
        let (file, root) = single_leaf_tree(WIDTHS);
        let source = BytesSource::new(&file);
        let extents = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap();
        let header_len = hdf5_pure_format::__private::btree_v2_header_size(
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .to_u64();
        assert_eq!(
            extents,
            vec![
                Extent::new(0, header_len).unwrap(),
                Extent::new(root, u64::from(BTREE_V2_NODE_SIZE)).unwrap(),
            ]
        );
    }

    #[test]
    fn storage_walk_enumerates_every_depth_3_node_once() {
        let tree = depth_3_tree();
        let source = BytesSource::new(&tree.file);
        let extents = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap();

        assert_eq!(extents.len(), tree.nodes.len() + 1);
        let mut expected_nodes = tree.nodes.to_vec();
        expected_nodes.sort_unstable();
        assert_eq!(
            extents[1..]
                .iter()
                .map(|extent| extent.start())
                .collect::<Vec<_>>(),
            expected_nodes
        );
        assert!(
            extents[1..]
                .iter()
                .all(|extent| extent.len() == u64::from(BTREE_V2_NODE_SIZE))
        );
        assert!(
            extents
                .windows(2)
                .all(|pair| pair[0].end() <= pair[1].start())
        );
    }

    #[test]
    fn storage_walk_of_an_empty_tree_returns_only_the_header() {
        let root = StoredAddress::undefined(OFFSET_SIZE).get();
        let header = btree_v2::Header::new(5, 11, root, 0).build(WIDTHS);
        let source = BytesSource::new(&header);
        let extents = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap();
        assert_eq!(extents, [Extent::new(0, header.len().to_u64()).unwrap()]);
    }

    #[test]
    fn storage_walk_rejects_a_truncated_header() {
        let mut header = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        header.pop();
        let source = BytesSource::new(&header);
        let err = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: header.len() + 1,
                available: header.len(),
            }
        );
    }

    #[test]
    fn storage_walk_rejects_a_truncated_node() {
        let (mut file, root) = single_leaf_tree(WIDTHS);
        file.pop();
        let source = BytesSource::new(&file);
        let err = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: (root + u64::from(BTREE_V2_NODE_SIZE)).to_usize().unwrap(),
                available: file.len(),
            }
        );
    }

    #[test]
    fn storage_walk_rejects_a_duplicate_child_pointer() {
        let mut image = Image::new();
        image.place(0, &[0u8; 64]);
        let leaf = put_leaf(&mut image, 5, &[rec(0)]);
        let root = put_internal(
            &mut image,
            5,
            &[rec(1)],
            &[(leaf, 1, 1), (leaf, 1, 1)],
            1,
            None,
        );
        let header = btree_v2::Header::new(5, 11, root, 1)
            .depth(1)
            .total_records(3)
            .build(WIDTHS);
        image.place(0, &header);
        let source = BytesSource::new(image.build());

        let err = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_rejects_overlapping_node_allocations() {
        let mut image = Image::new();
        image.place(0, &[0u8; 64]);
        let first = 64u64;
        let second = 320u64;
        image.place(
            first.to_usize().unwrap(),
            &node_allocation(btree_v2::leaf(5, &[rec(0)])),
        );
        image.place(
            second.to_usize().unwrap(),
            &node_allocation(btree_v2::leaf(5, &[rec(2)])),
        );
        let root = put_internal(
            &mut image,
            5,
            &[rec(1)],
            &[(first, 1, 1), (second, 1, 1)],
            1,
            None,
        );
        let header = btree_v2::Header::new(5, 11, root, 1)
            .depth(1)
            .total_records(3)
            .build(WIDTHS);
        image.place(0, &header);
        let source = BytesSource::new(image.build());

        let err = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_rejects_a_child_past_eof() {
        let (file, bad_address) = tree_with_one_invalid_child(4096);
        let source = BytesSource::new(&file);
        let err = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: (bad_address + u64::from(BTREE_V2_NODE_SIZE))
                    .to_usize()
                    .unwrap(),
                available: file.len(),
            }
        );
    }

    #[test]
    fn storage_walk_rejects_a_child_range_that_overflows() {
        let bad_address = u64::MAX - 255;
        let (file, _) = tree_with_one_invalid_child(bad_address);
        let source = BytesSource::new(file);
        let err = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::OffsetOverflow {
                offset: bad_address,
                length: u64::from(BTREE_V2_NODE_SIZE),
            }
        );
    }

    #[test]
    fn storage_walk_supports_four_and_eight_byte_widths() {
        for (widths, offset_width, length_width) in [
            (Widths::FOUR, OffsetWidth::Four, LengthWidth::Four),
            (Widths::EIGHT, OffsetWidth::Eight, LengthWidth::Eight),
        ] {
            let (file, root) = single_leaf_tree(widths);
            let source = BytesSource::new(file);
            let extents = collect_btree_v2_storage_extents(
                &source,
                StoredAddress::new(0),
                offset_width.get(),
                length_width.get(),
            )
            .unwrap();
            assert_eq!(
                extents,
                [
                    Extent::new(
                        0,
                        hdf5_pure_format::__private::btree_v2_header_size(
                            offset_width,
                            length_width,
                        )
                        .to_u64(),
                    )
                    .unwrap(),
                    Extent::new(root, u64::from(BTREE_V2_NODE_SIZE)).unwrap(),
                ]
            );
        }
    }

    #[test]
    fn storage_walk_rejects_zero_node_size() {
        let header = btree_v2::Header::new(5, 11, 0, 0)
            .node_size(0)
            .build(WIDTHS);
        let source = BytesSource::new(header);
        let err = collect_btree_v2_storage_extents(
            &source,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: 10,
                available: 0,
            }
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn storage_walk_matches_memory_and_seek_sources() {
        let tree = depth_3_tree();
        let memory = BytesSource::new(&tree.file);
        let seek = ReadSeekSource::new(Cursor::new(tree.file.clone())).unwrap();
        let memory_extents = collect_btree_v2_storage_extents(
            &memory,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap();
        let seek_extents = collect_btree_v2_storage_extents(
            &seek,
            StoredAddress::new(0),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap();
        assert_eq!(memory_extents, seek_extents);
    }

    fn single_leaf_tree(widths: Widths) -> (Vec<u8>, u64) {
        let mut image = Image::new();
        image.place(0, &[0u8; 64]);
        let root = put_leaf(&mut image, 5, &[rec(0)]);
        let header = btree_v2::Header::new(5, 11, root, 1).build(widths);
        image.place(0, &header);
        (image.build(), root)
    }

    fn tree_with_one_invalid_child(child_address: u64) -> (Vec<u8>, u64) {
        let mut image = Image::new();
        image.place(0, &[0u8; 64]);
        let root = put_internal(&mut image, 5, &[], &[(child_address, 0, 0)], 1, None);
        let header = btree_v2::Header::new(5, 11, root, 0)
            .depth(1)
            .total_records(1)
            .build(WIDTHS);
        image.place(0, &header);
        (image.build(), child_address)
    }

    #[test]
    fn parse_leaf_with_2_records() {
        let rec1 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let rec2 = [11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let leaf = build_leaf_node(5, &[&rec1, &rec2]);

        let leaf_offset = 256usize;
        let header = btree_v2::Header::new(5, 11, leaf_offset as u64, 2).build(WIDTHS);

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
    fn empty_tree() {
        let header = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        let hdr = BTreeV2Header::parse(&header, 0, 8, 8).unwrap();
        let records = collect_btree_v2_records(&header, &hdr, 8, 8).unwrap();
        assert!(records.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_btree_matches_buffered() {
        use crate::source::SourceMetadata;
        let rec1 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let rec2 = [11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let leaf = build_leaf_node(5, &[&rec1, &rec2]);
        let leaf_offset = 256usize;
        let header = btree_v2::Header::new(5, 11, leaf_offset as u64, 2).build(WIDTHS);
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
        let hdr_mem = BTreeV2Header::parse_from_source(&SourceMetadata(&mem), 0, 8, 8).unwrap();
        assert_eq!(hdr_mem.root_node_address, hdr.root_node_address);
        let from_mem: Vec<_> = collect_btree_v2_records_from_source(&mem, &hdr_mem, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        let seek = ReadSeekSource::new(Cursor::new(file_data)).unwrap();
        let hdr_seek = BTreeV2Header::parse_from_source(&SourceMetadata(&seek), 0, 8, 8).unwrap();
        let from_seek: Vec<_> = collect_btree_v2_records_from_source(&seek, &hdr_seek, 8, 8)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(from_seek.len(), 2);
    }

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

    /// Writes a tree of `count` records, reads it back with [`collect_btree_v2_records`], and
    /// returns its depth and the record indices in the order the walk reads them.
    fn round_trip(count: usize, record_size: u16) -> (u16, Vec<u64>) {
        let plan = BTreeV2Plan::new(
            8,
            count,
            record_size,
            BTREE_V2_NODE_SIZE,
            OffsetWidth::Eight,
        )
        .expect("plannable");
        let records = numbered_records(count, record_size);

        // Put the header at 0 and the nodes right after it, then parse the
        // whole thing back out of one buffer.
        let nodes_address = StoredAddress::new(hdf5_pure_format::__private::btree_v2_header_size(
            OffsetWidth::Eight,
            LengthWidth::Eight,
        ) as u64);
        let image = plan.serialize(
            &records,
            nodes_address,
            OffsetWidth::Eight,
            LengthWidth::Eight,
        );
        let mut file = image.header.clone();
        file.extend_from_slice(&image.nodes);

        let header = BTreeV2Header::parse(&file, 0, OFFSET_SIZE, LENGTH_SIZE).expect("header");
        assert_eq!(header.node_size, BTREE_V2_NODE_SIZE);
        assert_eq!(header.total_records, count as u64);
        let read =
            collect_btree_v2_records(&file, &header, OFFSET_SIZE, LENGTH_SIZE).expect("read");
        let ids = read
            .iter()
            .map(|r| u64::from_le_bytes(r.data[..8].try_into().expect("8 bytes")))
            .collect();
        (header.depth, ids)
    }

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

    #[test]
    fn depth_grows_only_when_the_level_below_is_full() {
        assert_eq!(round_trip(29, 17).0, 0);
        assert_eq!(round_trip(30, 17).0, 1);
        assert_eq!(round_trip(569, 17).0, 1);
        assert_eq!(round_trip(570, 17).0, 2);
        assert_eq!(round_trip(10_259, 17).0, 2);
        assert_eq!(round_trip(10_260, 17).0, 3);
    }

    /// A node holds fewer 24-byte huge-object records than 17-byte name records.
    #[test]
    fn a_wider_record_reaches_depth_sooner() {
        let (depth, ids) = round_trip(1_000, 24);
        assert_eq!(ids, (0..1_000u64).collect::<Vec<_>>());
        assert_eq!(depth, 2, "20 records per leaf, 314 per depth-1 subtree");
    }

    const WIDTHS: Widths = Widths::EIGHT;
    const OFFSET_SIZE: u8 = 8;
    const LENGTH_SIZE: u8 = 8;
}
