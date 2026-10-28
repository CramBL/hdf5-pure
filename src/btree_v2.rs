//! Traverses version 2 B-trees to read their records or enumerate their allocated storage.
//!
//! "Version 2 B-trees" of the [format specification, version 4.0][spec] defines separate `BTHD`
//! header, `BTIN` internal-node, and `BTLF` leaf-node layouts. The header's "Node Size" field is
//! the size in bytes of every B-tree node, not of the header. Its "Root Node Address" is undefined
//! for a tree with no records. Otherwise, "Depth" determines whether that root is a leaf or an
//! internal node. Record walks preserve tree order.
//! [`collect_btree_v2_storage_extents`] returns the header allocation and every
//! reachable full "Node Size" allocation in file-address order.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2

use core::cmp::Ordering;
use core::num::NonZeroU16;

use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
pub use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::BTreeV2NodeInfo;
pub use hdf5_pure_format::__private::BTreeV2Record;
use hdf5_pure_space::__private::Extent;

use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::{RangeReader, Source};
#[cfg(test)]
use crate::width::LengthWidth;
use crate::width::{FormatWidths, OffsetWidth};

/// Returns a copy of the matching record by binary-searching each node along one search path.
///
/// `compare` orders each borrowed record relative to the sought key. The tree's records must be
/// ordered consistently with `compare`. An equal record in an internal node is a match. The
/// reader resolves stored addresses through `addresses` and reads the encoded contents of each
/// visited node before calling `compare`.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeV2Signature`] for a repeated node address, and the first
/// error from address resolution, a node read, node parsing, or `compare`.
pub(crate) fn search_btree_v2(
    reader: &(impl RangeReader + ?Sized),
    layout: &BTreeV2Layout,
    addresses: AddressContext,
    mut compare: impl FnMut(&[u8]) -> Result<Ordering, FormatError>,
) -> Result<Option<Vec<u8>>, FormatError> {
    let Some(mut node) = layout.root() else {
        return Ok(None);
    };
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(node.address()) {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let range = FileRange::new(
            addresses.resolve(node.address())?,
            layout.encoded_len(node)?.to_u64(),
        )?;
        let bytes = reader.read_range(range)?;
        let view = layout.parse(&bytes, node)?;
        let mut left = 0;
        let mut right = view.record_count();
        while left < right {
            let middle = left + (right - left) / 2;
            match compare(view.record(middle)?)? {
                Ordering::Less => left = middle + 1,
                Ordering::Equal => return Ok(Some(view.record(middle)?.to_vec())),
                Ordering::Greater => right = middle,
            }
        }
        let Some(child) = view.child(left)? else {
            return Ok(None);
        };
        node = child;
    }
}

/// Returns the allocations reachable from a version 2 B-tree header using validated file widths.
///
/// # Errors
///
/// Returns the structural error of the first B-tree header or node that does not parse.
pub(crate) fn collect_btree_v2_storage_extents<S: Source + ?Sized>(
    source: &S,
    header_address: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<Extent>, FormatError> {
    // "Node Size" applies to BTIN and BTLF nodes. BTHD has its own layout and encoded length.
    let header_len =
        hdf5_pure_format::__private::btree_v2_header_size(widths.offsets, widths.lengths);
    let header_extent = extent_in_source(source, header_address.get(), header_len.to_u64())?;
    let header_bytes = source.read_metadata(header_address.get(), header_len)?;
    let header = BTreeV2Header::parse_with_widths(&header_bytes, 0, widths)?;
    // BTIN and BTLF both have a 6-byte signature/version/type prefix and a 4-byte checksum.
    if header.node_size < 10 {
        return Err(FormatError::UnexpectedEof {
            expected: 10,
            available: header.node_size as usize,
        });
    }

    let mut extents = Vec::with_capacity(1);
    extents.push(header_extent);
    if !storage_header_has_root(&header, widths.offsets)? {
        return Ok(extents);
    }

    let node_info = BTreeV2NodeInfo::compute(
        header.node_size,
        header.record_size,
        widths.offsets,
        header.depth,
    );
    let mut visited = BTreeSet::new();
    collect_storage_node(
        source,
        header.root_node_address,
        header.num_records_in_root,
        header.depth,
        &header,
        widths.offsets,
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

/// Proves the root fields agree with whether a B-tree has records.
///
/// "Version 2 B-trees" requires the root address to be undefined for a tree with no records. A
/// non-empty tree must name the root allocation the structural walk starts from. Requiring the
/// root's record count to be zero for an empty tree also prevents contradictory header state from
/// being accepted merely because there is no node to visit.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
fn storage_header_has_root(
    header: &BTreeV2Header,
    offset_width: OffsetWidth,
) -> Result<bool, FormatError> {
    let root_is_undefined = header.root_node_address.is_undefined(offset_width);
    let valid = if header.total_records == 0 {
        root_is_undefined && header.num_records_in_root == 0
    } else {
        !root_is_undefined
    };
    if !valid {
        return Err(FormatError::InvalidBTreeV2Signature);
    }
    Ok(!root_is_undefined)
}

/// Adds one complete node allocation and recursively follows the children its depth permits.
///
/// "Version 2 B-trees" gives an internal node with R records R + 1 child-pointer groups. Each
/// group contains "Child Node Pointer" and "Number of Records in Child Node". A pointer whose child
/// is also an internal node additionally contains "Total Number of Records in Child Node". "Depth"
/// 0 is a leaf, so recursion stops there after validating the leaf layout.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[allow(clippy::too_many_arguments)]
fn collect_storage_node<S: Source + ?Sized>(
    source: &S,
    address: StoredAddress,
    num_records: u16,
    depth: u16,
    header: &BTreeV2Header,
    offset_width: OffsetWidth,
    node_info: &BTreeV2NodeInfo,
    visited: &mut BTreeSet<StoredAddress>,
    extents: &mut Vec<Extent>,
) -> Result<(), FormatError> {
    if !visited.insert(address) {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let extent = extent_in_source(source, address.get(), u64::from(header.node_size))?;
    let node = source.read_metadata(address.get(), u64::from(header.node_size).to_usize()?)?;
    extents.push(extent);

    let signature = if depth == 0 { b"BTLF" } else { b"BTIN" };
    validate_node_prefix(&node, signature, header.tree_type)?;
    let Some(depth) = NonZeroU16::new(depth) else {
        return validate_storage_leaf(&node, num_records, header.record_size);
    };
    let children = hdf5_pure_format::__private::parse_btree_v2_internal_child_pointers(
        &node,
        num_records,
        depth,
        header.record_size,
        offset_width,
        node_info,
    )?;
    for (child_address, child_records) in children {
        if child_address.is_undefined(offset_width) {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        collect_storage_node(
            source,
            child_address,
            child_records,
            depth.get() - 1,
            header,
            offset_width,
            node_info,
            visited,
            extents,
        )?;
    }
    Ok(())
}

/// Proves a node has the version and type required by its B-tree header.
///
/// "Version 2 B-trees" defines version 0 for both internal and leaf nodes, and requires each
/// node's "Type" field to equal the tree type stored in `BTHD`.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
fn validate_node_prefix(
    node: &[u8],
    signature: &[u8; 4],
    tree_type: u8,
) -> Result<(), FormatError> {
    bytes::ensure_len(node, 0, 6)?;
    if &node[..4] != signature {
        return Err(FormatError::InvalidBTreeV2Signature);
    }
    let version = node[4];
    if version != 0 {
        return Err(FormatError::InvalidBTreeV2Version(version));
    }
    let node_type = node[5];
    if node_type != tree_type {
        return Err(FormatError::InvalidBTreeNodeType(node_type));
    }
    Ok(())
}

/// Proves the declared leaf contents fit inside the node allocation.
///
/// "Version 2 B-trees" lays out a leaf as the `BTLF` signature, version and type, R records of the
/// header's "Record Size", and the checksum. This check requires the records and checksum of that
/// declared layout to fit inside the already bounded "Node Size" allocation.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
fn validate_storage_leaf(
    node: &[u8],
    num_records: u16,
    record_size: u16,
) -> Result<(), FormatError> {
    let record_bytes = u64::from(num_records) * u64::from(record_size);
    let encoded_len = 6u64
        .checked_add(record_bytes)
        .and_then(|length| length.checked_add(4))
        .ok_or(FormatError::InvalidBTreeV2Signature)?
        .to_usize()?;
    bytes::ensure_len(node, 0, encoded_len)?;
    hdf5_pure_format::__private::verify_trailing(&node[..encoded_len])
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
        return Err(FormatError::UnexpectedSourceEof {
            expected: extent.end(),
            available: source.len(),
        });
    }
    Ok(extent)
}

/// Returns all records from a version 2 B-tree using a validated address width.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first B-tree node that does not parse.
pub fn collect_btree_v2_records(
    file_data: &[u8],
    header: &BTreeV2Header,
    offset_width: OffsetWidth,
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
        offset_width,
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
        offset_width,
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
    offset_width: OffsetWidth,
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
        offset_width,
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
                offset_width,
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

/// Returns all records from a version 2 B-tree source using a validated address width.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first B-tree node that does not parse, or the error `source`
/// reports for a node it cannot read.
pub fn collect_btree_v2_records_from_source<S: Source + ?Sized>(
    source: &S,
    header: &BTreeV2Header,
    offset_width: OffsetWidth,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }
    let node_info = BTreeV2NodeInfo::compute(
        header.node_size,
        header.record_size,
        offset_width,
        header.depth,
    );
    let mut visited = BTreeSet::new();
    let mut records = Vec::new();
    collect_node_from_source(
        source,
        header.root_node_address.get(),
        header.num_records_in_root,
        header.depth,
        header.tree_type,
        header.record_size,
        header.node_size,
        offset_width,
        &node_info,
        &mut visited,
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
    tree_type: u8,
    record_size: u16,
    node_size: u32,
    offset_width: OffsetWidth,
    node_info: &BTreeV2NodeInfo,
    visited: &mut BTreeSet<StoredAddress>,
    out: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    if !visited.insert(StoredAddress::new(address)) {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    // Every node occupies `node_size` bytes; read that window (clamped to the
    // bytes available, in case the final node abuts EOF).
    let node_len = u64::from(node_size)
        .min(source.len().saturating_sub(address))
        .to_usize()?;
    let node = source.read_metadata(address, node_len)?;

    let Some(depth) = NonZeroU16::new(depth) else {
        validate_node_prefix(&node, b"BTLF", tree_type)?;
        out.extend(hdf5_pure_format::__private::parse_btree_v2_leaf_records(
            &node,
            0,
            num_records,
            record_size,
        )?);
        return Ok(());
    };

    validate_node_prefix(&node, b"BTIN", tree_type)?;
    let children = hdf5_pure_format::__private::parse_btree_v2_internal_child_pointers(
        &node,
        num_records,
        depth,
        record_size,
        offset_width,
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
            tree_type,
            record_size,
            node_size,
            offset_width,
            node_info,
            visited,
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

    use hdf5_pure_format::__private::BTreeV2Plan;
    use hdf5_pure_format::__private::{BTREE_V2_LINK_NAME, BTREE_V2_NODE_SIZE};
    use test_util::btree_v2;
    use test_util::image::Image;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::{BaseAddress, BaseAddressExt};
    use crate::source::BytesSource;
    #[cfg(feature = "std")]
    use crate::source::ReadSeekSource;

    #[rstest::rstest]
    #[case::root(7, Some(7), 1)]
    #[case::internal(3, Some(3), 2)]
    #[case::left_leaf(0, Some(0), 4)]
    #[case::right_leaf(14, Some(14), 4)]
    #[case::absent(15, None, 4)]
    fn indexed_search_reads_only_the_selected_path(
        #[case] wanted: u8,
        #[case] expected: Option<u8>,
        #[case] reads: usize,
    ) {
        let tree = depth_3_tree();
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let source = CountingSource {
            bytes: &tree.file,
            reads: core::cell::Cell::new(0),
        };
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        assert_eq!(
            search_btree_v2(&source, &layout, addresses, |record| Ok(
                record[0].cmp(&wanted)
            ))
            .unwrap()
            .map(|record| record[0]),
            expected
        );
        assert_eq!(source.reads.get(), reads);
        assert_eq!(
            search_btree_v2(tree.file.as_slice(), &layout, addresses, |record| Ok(
                record[0].cmp(&wanted)
            ))
            .unwrap()
            .map(|record| record[0]),
            expected
        );
    }

    #[test]
    fn indexed_search_of_an_empty_tree_returns_none_with_zero_reads() {
        let bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, u64::MAX, 0).build(WIDTHS);
        let header = BTreeV2Header::parse(&bytes, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let source = CountingSource {
            bytes: &bytes,
            reads: core::cell::Cell::new(0),
        };
        assert_eq!(
            search_btree_v2(
                &source,
                &layout,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                |_| { panic!("an empty tree has no record to compare") }
            )
            .unwrap(),
            None
        );
        assert_eq!(source.reads.get(), 0);
    }

    #[rstest::rstest]
    #[case::first(0)]
    #[case::between_records(1)]
    #[case::middle(40)]
    #[case::later(400)]
    #[case::absent(5001)]
    fn indexed_search_binary_searches_nodes_with_multiple_records(
        #[case] wanted: u64,
        #[values(1, 45, 46, 2500)] count: usize,
    ) {
        let plan = BTreeV2Plan::new(
            BTREE_V2_LINK_NAME,
            count,
            11,
            BTREE_V2_NODE_SIZE,
            OffsetWidth::Eight,
        )
        .unwrap();
        let image = plan.serialize(
            &numbered_records(count, 11),
            StoredAddress::new(64),
            OffsetWidth::Eight,
            LengthWidth::Eight,
        );
        let mut bytes = image.header;
        bytes.resize(64, 0);
        bytes.extend_from_slice(&image.nodes);
        let header = BTreeV2Header::parse(&bytes, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let mut comparisons = 0;
        let result = search_btree_v2(
            bytes.as_slice(),
            &layout,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            |record| {
                comparisons += 1;
                Ok((u64::from_le_bytes(record[..8].try_into().unwrap()) * 2).cmp(&wanted))
            },
        )
        .unwrap()
        .map(|record| u64::from_le_bytes(record[..8].try_into().unwrap()));
        let expected =
            (wanted.is_multiple_of(2) && wanted / 2 < count as u64).then_some(wanted / 2);
        assert_eq!(result, expected);
        assert!(comparisons <= 6 * (usize::from(header.depth) + 1));
    }

    struct CountingSource<'a> {
        bytes: &'a [u8],
        reads: core::cell::Cell<usize>,
    }

    impl Source for CountingSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }
        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
            self.reads.set(self.reads.get() + 1);
            BytesSource::new(self.bytes).read_at(offset, buffer)
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn indexed_comparison_can_read_through_the_same_metadata_cache() {
        use crate::source::{MetadataCacheConfig, MetadataCachingSource};

        let tree = depth_3_tree();
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let source = MetadataCachingSource::new(
            BytesSource::new(&tree.file),
            MetadataCacheConfig::default(),
        );
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        assert_eq!(
            search_btree_v2(&source, &layout, addresses, |record| {
                assert_eq!(
                    source.read_metadata_at(0, btree_v2::SIGNATURE.len())?,
                    btree_v2::SIGNATURE
                );
                Ok(record[0].cmp(&0))
            })
            .unwrap(),
            Some(rec(0))
        );
        assert_eq!(
            search_btree_v2(&source, &layout, addresses, |_| {
                Err(FormatError::InvalidLinkType(99))
            })
            .unwrap_err(),
            FormatError::InvalidLinkType(99)
        );
    }

    #[rstest::rstest]
    #[case::userblock(512, false)]
    #[case::relocated(1024, false)]
    #[case::framed(512, true)]
    fn indexed_search_resolves_the_effective_base_once(#[case] base: u64, #[case] framed: bool) {
        use crate::source::BaseOffsetSource;

        let tree = depth_3_tree();
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let mut bytes = vec![0; usize::try_from(base).unwrap()];
        bytes.extend_from_slice(&tree.file);
        let source = BytesSource::new(&bytes);
        let mut addresses = AddressContext::from_effective_base(BaseAddress::new(base));
        let expected = search_btree_v2(bytes.as_slice(), &layout, addresses, |record| {
            Ok(record[0].cmp(&14))
        })
        .unwrap();
        let framed_source = BaseOffsetSource {
            inner: &source,
            base: BaseAddress::new(base),
        };
        let reader: &dyn Source = if framed {
            addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
            &framed_source
        } else {
            &source
        };
        assert_eq!(
            search_btree_v2(reader, &layout, addresses, |record| Ok(record[0].cmp(&14))).unwrap(),
            expected
        );
        assert_eq!(expected, Some(rec(14)));
    }

    #[test]
    fn indexed_search_decodes_only_the_selected_pointer() {
        let mut tree = depth_3_tree();
        let root = tree.nodes[14] as usize;
        tree.file[root + 17..root + 25].fill(0xff);
        let checksum_at = root + 17 + 2 * 11;
        let checksum = hdf5_pure_format::__private::jenkins_lookup3(&tree.file[root..checksum_at]);
        tree.file[checksum_at..checksum_at + 4].copy_from_slice(&checksum.to_le_bytes());
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        assert_eq!(
            search_btree_v2(tree.file.as_slice(), &layout, addresses, |record| Ok(
                record[0].cmp(&14)
            ))
            .unwrap(),
            Some(rec(14))
        );
        assert_eq!(
            search_btree_v2(tree.file.as_slice(), &layout, addresses, |record| Ok(
                record[0].cmp(&0)
            ))
            .unwrap_err(),
            FormatError::InvalidBTreeV2Signature
        );
    }

    #[test]
    fn indexed_search_reports_a_truncated_visited_node() {
        let tree = depth_3_tree();
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let expected = tree.nodes[14] + layout.encoded_len(layout.root().unwrap()).unwrap() as u64;
        let bytes = &tree.file[..usize::try_from(expected - 1).unwrap()];
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        assert_eq!(
            search_btree_v2(bytes, &layout, addresses, |_| Ok(Ordering::Equal)).unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected,
                available: expected - 1
            }
        );
        assert_eq!(
            search_btree_v2(&BytesSource::new(bytes), &layout, addresses, |_| Ok(
                Ordering::Equal
            ))
            .unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected,
                available: expected - 1
            }
        );
    }

    #[test]
    fn indexed_search_rejects_a_cycle() {
        let mut tree = depth_3_tree();
        let root = tree.nodes[14] as usize;
        tree.file[root + 17..root + 25].copy_from_slice(&(root as u64).to_le_bytes());
        let checksum_at = root + 17 + 2 * 11;
        let checksum = hdf5_pure_format::__private::jenkins_lookup3(&tree.file[root..checksum_at]);
        tree.file[checksum_at..checksum_at + 4].copy_from_slice(&checksum.to_le_bytes());
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        assert_eq!(
            search_btree_v2(tree.file.as_slice(), &layout, addresses, |record| Ok(
                record[0].cmp(&0)
            ))
            .unwrap_err(),
            FormatError::InvalidBTreeV2Signature
        );
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
        let ids: Vec<u8> = collect_btree_v2_records(&tree.file, &header, OffsetWidth::Eight)
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
                collect_btree_v2_records_from_source(&source, &header, OffsetWidth::Eight)
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
        let extents =
            collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
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
        let extents =
            collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
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
        let root = StoredAddress::undefined(OFFSET_WIDTH).get();
        let header = btree_v2::Header::new(5, 11, root, 0).build(WIDTHS);
        let source = BytesSource::new(&header);
        let extents =
            collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
                .unwrap();
        assert_eq!(extents, [Extent::new(0, header.len().to_u64()).unwrap()]);
    }

    #[test]
    fn storage_walk_rejects_an_empty_tree_with_a_defined_root() {
        let header = btree_v2::Header::new(5, 11, 0x100, 0).build(WIDTHS);
        let source = BytesSource::new(header);

        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();

        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_rejects_an_empty_tree_with_root_records() {
        let root = StoredAddress::undefined(OFFSET_WIDTH).get();
        let header = btree_v2::Header::new(5, 11, root, 1)
            .total_records(0)
            .build(WIDTHS);
        let source = BytesSource::new(header);

        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();

        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_rejects_a_nonempty_tree_without_a_root() {
        let root = StoredAddress::undefined(OFFSET_WIDTH).get();
        let header = btree_v2::Header::new(5, 11, root, 1).build(WIDTHS);
        let source = BytesSource::new(header);

        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();

        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_reports_a_header_extent_above_four_gibibytes() {
        let source = BytesSource::new([]);
        let address = 4_294_967_536;
        assert_eq!(
            collect_btree_v2_storage_extents(&source, StoredAddress::new(address), FORMAT_WIDTHS)
                .unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: address
                    + hdf5_pure_format::__private::btree_v2_header_size(
                        FORMAT_WIDTHS.offsets,
                        FORMAT_WIDTHS.lengths
                    )
                    .to_u64(),
                available: 0,
            }
        );
    }

    #[test]
    fn storage_walk_rejects_a_truncated_header() {
        let mut header = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        header.pop();
        let source = BytesSource::new(&header);
        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedSourceEof {
                expected: (header.len() + 1).to_u64(),
                available: header.len().to_u64(),
            }
        );
    }

    #[test]
    fn storage_walk_rejects_a_truncated_node() {
        let (mut file, root) = single_leaf_tree(WIDTHS);
        file.pop();
        let source = BytesSource::new(&file);
        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedSourceEof {
                expected: root + u64::from(BTREE_V2_NODE_SIZE),
                available: file.len().to_u64(),
            }
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn storage_walk_rejects_a_leaf_with_a_bad_checksum() {
        let (mut file, root) = single_leaf_tree(WIDTHS);
        let checksum = root.to_usize().unwrap() + 6 + 11;
        let stored = u32::from_le_bytes(file[checksum..checksum + 4].try_into().unwrap());
        file[checksum] ^= 1;
        let source = BytesSource::new(file);

        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::ChecksumMismatch {
                expected: stored ^ 1,
                computed: stored,
            }
        );
    }

    #[test]
    fn storage_walk_rejects_a_node_with_the_wrong_version() {
        let (mut file, root) = single_leaf_tree(WIDTHS);
        file[root.to_usize().unwrap() + 4] = 1;
        let source = BytesSource::new(file);
        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Version(1));
    }

    #[test]
    fn storage_walk_rejects_a_node_with_a_different_tree_type() {
        let (mut file, root) = single_leaf_tree(WIDTHS);
        file[root.to_usize().unwrap() + 5] = 6;
        let source = BytesSource::new(file);
        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeNodeType(6));
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn storage_walk_rejects_an_internal_node_with_a_bad_checksum() {
        let mut image = Image::new();
        image.place(0, &[0u8; 64]);
        let left = put_leaf(&mut image, 5, &[rec(0)]);
        let right = put_leaf(&mut image, 5, &[rec(2)]);
        let root = put_internal(
            &mut image,
            5,
            &[rec(1)],
            &[(left, 1, 1), (right, 1, 1)],
            1,
            None,
        );
        let header = btree_v2::Header::new(5, 11, root, 1)
            .depth(1)
            .total_records(3)
            .build(WIDTHS);
        image.place(0, &header);
        let mut file = image.build();
        let checksum = root.to_usize().unwrap() + 6 + 11 + 2 * (8 + 1);
        let stored = u32::from_le_bytes(file[checksum..checksum + 4].try_into().unwrap());
        file[checksum] ^= 1;
        let source = BytesSource::new(file);

        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::ChecksumMismatch {
                expected: stored ^ 1,
                computed: stored,
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

        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
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

        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_rejects_a_child_past_eof() {
        let (file, bad_address) = tree_with_one_invalid_child(4096);
        let source = BytesSource::new(&file);
        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedSourceEof {
                expected: bad_address + u64::from(BTREE_V2_NODE_SIZE),
                available: file.len().to_u64(),
            }
        );
    }

    #[test]
    fn storage_walk_rejects_a_child_range_that_overflows() {
        let bad_address = u64::MAX - 255;
        let (file, _) = tree_with_one_invalid_child(bad_address);
        let source = BytesSource::new(file);
        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
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
                FormatWidths {
                    offsets: offset_width,
                    lengths: length_width,
                },
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
        let err = collect_btree_v2_storage_extents(&source, StoredAddress::new(0), FORMAT_WIDTHS)
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
        let memory_extents =
            collect_btree_v2_storage_extents(&memory, StoredAddress::new(0), FORMAT_WIDTHS)
                .unwrap();
        let seek_extents =
            collect_btree_v2_storage_extents(&seek, StoredAddress::new(0), FORMAT_WIDTHS).unwrap();
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
        let records = collect_btree_v2_records(&file_data, &hdr, OffsetWidth::Eight).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data, rec1.to_vec());
        assert_eq!(records[1].data, rec2.to_vec());
    }

    #[test]
    fn empty_tree() {
        let header = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        let hdr = BTreeV2Header::parse(&header, 0, 8, 8).unwrap();
        let records = collect_btree_v2_records(&header, &hdr, OffsetWidth::Eight).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn source_walk_rejects_node_type_mismatching_header() {
        use crate::source::SourceMetadata;

        let record = rec(1);
        let leaf = btree_v2::leaf(11, &[record]);
        let leaf_offset = 256usize;
        let header =
            btree_v2::Header::new(10, 11, u64::try_from(leaf_offset).unwrap(), 1).build(WIDTHS);
        let mut file_data = vec![0u8; 512];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[leaf_offset..leaf_offset + leaf.len()].copy_from_slice(&leaf);

        let source = BytesSource::new(&file_data);
        let header = BTreeV2Header::parse_from_source(&SourceMetadata(&source), 0, 8, 8).unwrap();
        assert_eq!(
            collect_btree_v2_records_from_source(&source, &header, OffsetWidth::Eight,)
                .unwrap_err(),
            FormatError::InvalidBTreeNodeType(11)
        );
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
        let buffered: Vec<_> = collect_btree_v2_records(&file_data, &hdr, OffsetWidth::Eight)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect();

        let mem = BytesSource::new(&file_data);
        let hdr_mem = BTreeV2Header::parse_from_source(&SourceMetadata(&mem), 0, 8, 8).unwrap();
        assert_eq!(hdr_mem.root_node_address, hdr.root_node_address);
        let from_mem: Vec<_> =
            collect_btree_v2_records_from_source(&mem, &hdr_mem, OffsetWidth::Eight)
                .unwrap()
                .into_iter()
                .map(|r| r.data)
                .collect();

        let seek = ReadSeekSource::new(Cursor::new(file_data)).unwrap();
        let hdr_seek = BTreeV2Header::parse_from_source(&SourceMetadata(&seek), 0, 8, 8).unwrap();
        let from_seek: Vec<_> =
            collect_btree_v2_records_from_source(&seek, &hdr_seek, OffsetWidth::Eight)
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
        let read = collect_btree_v2_records(&file, &header, OffsetWidth::Eight).expect("read");
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
    const FORMAT_WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    const OFFSET_SIZE: u8 = 8;
    const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
    const LENGTH_SIZE: u8 = 8;
}
