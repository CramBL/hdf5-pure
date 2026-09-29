//! The walks of a version 2 B-tree, which read every record in order from a file image or from a
//! [`Source`].

use core::num::NonZeroU16;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2NodeInfo;
pub use hdf5_pure_format::__private::BTreeV2Record;

use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;

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

/// Recursively collect records from an internal node (buffered path).
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
    use hdf5_pure_format::__private::BTREE_V2_NODE_SIZE;
    use hdf5_pure_format::__private::BTreeV2Plan;
    use test_util::btree_v2;
    use test_util::image::Image;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::StoredAddress;
    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

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
        let header = btree_v2::Header::new(5, 11, root, 1)
            .depth(3)
            .total_records(15)
            .build(WIDTHS);
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
            use crate::source::SourceMetadata;
            let src = BytesSource::new(&file);
            let hdr_s = BTreeV2Header::parse_from_source(&SourceMetadata(&src), 0, 8, 8).unwrap();
            let ids_s: Vec<u8> = collect_btree_v2_records_from_source(&src, &hdr_s, 8, 8)
                .unwrap()
                .iter()
                .map(|r| r.data[0])
                .collect();
            assert_eq!(ids_s, (0u8..15).collect::<Vec<_>>());
        }
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
        use crate::source::BytesSource;
        use crate::source::ReadSeekSource;
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

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
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
