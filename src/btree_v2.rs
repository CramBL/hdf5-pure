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

use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::MetadataFrame;
use hdf5_pure_core::MetadataBytes;
pub use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::BTreeV2Node;
pub use hdf5_pure_format::__private::BTreeV2Record;
use hdf5_pure_format::__private::MetadataSource;
use hdf5_pure_space::__private::Extent;

#[cfg(any(test, doc))]
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;
use crate::source::StoredBytes;
use crate::source::StoredMetadata;
use crate::source::StoredRead;
#[cfg(test)]
use crate::width::LengthWidth;
use crate::width::{FormatWidths, OffsetWidth};

/// A metadata source for version 2 B-trees with stored-coordinate offsets.
///
/// With [`Self::new`], the reader delegates metadata reads to the stored view. With
/// [`Self::resident`], it borrows the resident image and reports EOF bounds in stored coordinates.
/// The reader preserves the physical ranges of returned [`MetadataBytes`].
pub(crate) struct BTreeV2Source<'a, R: StoredRead + ?Sized> {
    backing: BTreeV2Backing<'a, R>,
}

impl<'a, R: StoredRead + ?Sized> BTreeV2Source<'a, R> {
    /// Borrows a stored view and preserves its metadata read operations.
    pub(crate) fn new(source: &'a R) -> Self {
        Self {
            backing: BTreeV2Backing::Stored(StoredMetadata(source)),
        }
    }
}

impl<'a> BTreeV2Source<'a, StoredBytes<'a>> {
    /// Creates a resident reader with stored-coordinate EOF bounds.
    pub(crate) fn resident(source: StoredBytes<'a>) -> Self {
        Self {
            backing: BTreeV2Backing::Resident(source.metadata_frame()),
        }
    }
}

impl<R: StoredRead + ?Sized> MetadataSource for BTreeV2Source<'_, R> {
    type Bytes<'a>
        = MetadataBytes<'a>
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, length: usize) -> Result<Self::Bytes<'_>, FormatError> {
        match &self.backing {
            BTreeV2Backing::Stored(source) => source.read_metadata(offset, length),
            BTreeV2Backing::Resident(frame) => Source::read_metadata(frame, offset, length),
        }
    }

    fn len(&self) -> u64 {
        match &self.backing {
            BTreeV2Backing::Stored(source) => source.len(),
            BTreeV2Backing::Resident(frame) => frame.len(),
        }
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        match &self.backing {
            BTreeV2Backing::Stored(source) => source.read_at(offset, buffer),
            BTreeV2Backing::Resident(frame) => frame.read_at(offset, buffer),
        }
    }

    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        match &self.backing {
            BTreeV2Backing::Stored(source) => source.read_metadata_at(offset, length),
            BTreeV2Backing::Resident(frame) => frame.read_metadata_at(offset, length),
        }
    }
}

enum BTreeV2Backing<'a, R: StoredRead + ?Sized> {
    Resident(MetadataFrame<'a>),
    Stored(StoredMetadata<'a, R>),
}

/// Returns a copy of the matching record along one binary-search path.
///
/// `compare` orders each borrowed record relative to the sought key. Records must be ordered
/// consistently with `compare`. An equal record in an internal node is a match. Returns [`None`]
/// for an empty tree or an absent key. Reads each visited node through
/// [`StoredRead::read_metadata`].
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeV2Signature`] for a repeated node address, and errors from
/// address resolution, metadata reads, node parsing, or `compare`.
pub(crate) fn search_btree_v2(
    reader: &(impl StoredRead + ?Sized),
    layout: &BTreeV2Layout,
    compare: impl FnMut(&[u8]) -> Result<Ordering, FormatError>,
) -> Result<Option<Vec<u8>>, FormatError> {
    search_nodes(
        layout,
        |node| {
            let start = reader.source_address(node.address())?;
            let length = node.encoded_len()?;
            FileRange::new(start, length.to_u64())?;
            reader.read_metadata(node.address(), length)
        },
        compare,
    )
}

fn search_nodes<'layout, 'reader>(
    layout: &'layout BTreeV2Layout,
    mut read_node: impl FnMut(BTreeV2Node<'layout>) -> Result<MetadataBytes<'reader>, FormatError>,
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
        let bytes = read_node(node)?;
        let view = node.parse(&bytes)?;
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

/// Returns the header and full node allocations in stored-address order.
///
/// Reads metadata through [`StoredRead::read_metadata`]. The returned addresses are relative to
/// the opened image's effective base.
///
/// # Errors
///
/// Returns errors from metadata reads, checked allocation bounds, and parsing the header or a node.
pub(crate) fn collect_btree_v2_storage_extents(
    source: &(impl StoredRead + ?Sized),
    header_address: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<Extent>, FormatError> {
    // "Node Size" applies to BTIN and BTLF nodes. BTHD has its own layout and encoded length.
    let header_len =
        hdf5_pure_format::__private::btree_v2_header_size(widths.offsets, widths.lengths);
    let header_extent = extent_in_source(source, header_address, header_len.to_u64())?;
    let header_bytes = source.read_metadata(header_address, header_len)?;
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

    let layout = BTreeV2Layout::for_collection(&header, widths.offsets)?;
    let root = layout.root().ok_or(FormatError::InvalidBTreeV2Signature)?;
    let mut visited = BTreeSet::new();
    collect_storage_node(
        source,
        root,
        header.node_size,
        widths.offsets,
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
fn collect_storage_node(
    source: &(impl StoredRead + ?Sized),
    node: BTreeV2Node<'_>,
    node_size: u32,
    offset_width: OffsetWidth,
    visited: &mut BTreeSet<StoredAddress>,
    extents: &mut Vec<Extent>,
) -> Result<(), FormatError> {
    if !visited.insert(node.address()) {
        return Err(FormatError::InvalidBTreeV2Signature);
    }
    let extent = extent_in_source(source, node.address(), u64::from(node_size))?;
    let bytes = source.read_metadata(node.address(), u64::from(node_size).to_usize()?)?;
    let view = node.parse(&bytes)?;
    extents.push(extent);
    for index in 0..=view.record_count() {
        if let Some(child) = view.child(index)? {
            if child.address().is_undefined(offset_width) {
                return Err(FormatError::InvalidBTreeV2Signature);
            }
            collect_storage_node(source, child, node_size, offset_width, visited, extents)?;
        }
    }
    Ok(())
}

/// Returns a checked extent after proving its complete byte range lies within `source`.
fn extent_in_source(
    source: &(impl StoredRead + ?Sized),
    address: StoredAddress,
    length: u64,
) -> Result<Extent, FormatError> {
    debug_assert!(length > 0, "storage extents are non-empty");
    let extent = Extent::new(address, length).ok_or(FormatError::OffsetOverflow {
        offset: address.get(),
        length,
    })?;
    if extent.end() > StoredAddress::new(source.len()) {
        return Err(FormatError::UnexpectedSourceEof {
            expected: extent.end().get(),
            available: source.len(),
        });
    }
    Ok(extent)
}

/// Reads all records in tree order from the resident image at stored addresses.
///
/// Returns an empty vector when either record count in `header` is zero. Parses each visited
/// node with the header's type and geometry before reading its records and children.
///
/// # Errors
///
/// Returns errors from address conversion or B-tree parsing.
pub(crate) fn collect_btree_v2_records(
    source: StoredBytes<'_>,
    header: &BTreeV2Header,
    offset_width: OffsetWidth,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }

    let layout = BTreeV2Layout::for_collection(header, offset_width)?;
    let root = layout.root().ok_or(FormatError::InvalidBTreeV2Signature)?;
    let mut records = if header.depth == 0 {
        Vec::with_capacity(usize::from(header.num_records_in_root))
    } else {
        Vec::new()
    };
    collect_node_records(source, root, &mut records)?;
    Ok(records)
}

fn collect_node_records(
    source: StoredBytes<'_>,
    node: BTreeV2Node<'_>,
    records: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    let file_data = source.metadata_frame().bytes();
    let start = node.address().get().to_usize()?;
    bytes::ensure_len(file_data, start, 0)?;
    let view = node.parse(&file_data[start..])?;
    for index in 0..=view.record_count() {
        if let Some(child) = view.child(index)? {
            collect_node_records(source, child, records)?;
        }
        if index < view.record_count() {
            records.push(BTreeV2Record {
                data: view.record(index)?.to_vec(),
            });
        }
    }
    Ok(())
}

/// Reads all records in tree order through a stored-coordinate view.
///
/// Returns an empty vector when either record count in `header` is zero. Each node is parsed with
/// the header's type and geometry. The reader preserves the source's metadata ownership and
/// physical ranges.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeV2Signature`] for a repeated node address, and errors from
/// metadata reads or B-tree parsing.
pub(crate) fn collect_btree_v2_records_from_source(
    source: &BTreeV2Source<'_, impl StoredRead + ?Sized>,
    header: &BTreeV2Header,
    offset_width: OffsetWidth,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }
    let layout = BTreeV2Layout::for_collection(header, offset_width)?;
    let root = layout.root().ok_or(FormatError::InvalidBTreeV2Signature)?;
    let mut visited = BTreeSet::new();
    let mut records = Vec::new();
    collect_node_from_source(source, root, header.node_size, &mut visited, &mut records)?;
    Ok(records)
}

fn collect_node_from_source(
    source: &BTreeV2Source<'_, impl StoredRead + ?Sized>,
    node: BTreeV2Node<'_>,
    node_size: u32,
    visited: &mut BTreeSet<StoredAddress>,
    records: &mut Vec<BTreeV2Record>,
) -> Result<(), FormatError> {
    if !visited.insert(node.address()) {
        return Err(FormatError::InvalidBTreeV2Signature);
    }
    let address = node.address();
    let node_len = u64::from(node_size)
        .min(
            StoredAddress::new(source.len())
                .checked_distance_from(address)
                .unwrap_or(0),
        )
        .to_usize()?;
    let bytes = source.read_metadata(address.get(), node_len)?;
    let view = node.parse(&bytes)?;
    for index in 0..=view.record_count() {
        if let Some(child) = view.child(index)? {
            collect_node_from_source(source, child, node_size, visited, records)?;
        }
        if index < view.record_count() {
            records.push(BTreeV2Record {
                data: view.record(index)?.to_vec(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    #[cfg(feature = "std")]
    use std::io::Cursor;

    use hdf5_pure_format::__private::BTreeV2Plan;
    use hdf5_pure_format::__private::{BTREE_V2_LINK_NAME, BTREE_V2_NODE_SIZE};
    use test_util::btree_v2;
    use test_util::image::Image;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::OpenedImageContext;
    use crate::address::StoredAddressExt;
    use crate::address::{BaseAddress, BaseAddressExt};
    use crate::source::BytesSource;
    use crate::source::PhysicalBytes;
    use crate::source::PhysicalSource;
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
            search_btree_v2(
                &PhysicalSource::for_test(&source, addresses).stored(),
                &layout,
                |record| Ok(record[0].cmp(&wanted))
            )
            .unwrap()
            .map(|record| record[0]),
            expected
        );
        assert_eq!(source.reads.get(), reads);
        assert_eq!(
            search_btree_v2(
                &PhysicalBytes::for_test(tree.file.as_slice(), addresses)
                    .stored()
                    .unwrap(),
                &layout,
                |record| Ok(record[0].cmp(&wanted))
            )
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
                &PhysicalSource::for_test(
                    &source,
                    AddressContext::from_effective_base(BaseAddress::ZERO)
                )
                .stored(),
                &layout,
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
        let image = plan
            .serialize(
                &numbered_records(count, 11),
                StoredAddress::new(64),
                OffsetWidth::Eight,
                LengthWidth::Eight,
            )
            .unwrap();
        let mut bytes = image.header;
        bytes.resize(64, 0);
        bytes.extend_from_slice(&image.nodes);
        let header = BTreeV2Header::parse(&bytes, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let mut comparisons = 0;
        let result = search_btree_v2(
            &PhysicalBytes::for_test(
                bytes.as_slice(),
                AddressContext::from_effective_base(BaseAddress::ZERO),
            )
            .stored()
            .unwrap(),
            &layout,
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

    #[rstest::rstest]
    #[case::zero_base_endpoint(0, u64::MAX - 4, FormatError::OffsetOverflow { offset: u64::MAX - 4, length: 21 }, 0)]
    #[case::relocated_endpoint(512, u64::MAX - 516, FormatError::OffsetOverflow { offset: u64::MAX - 4, length: 21 }, 0)]
    #[case::relocated_start(512, u64::MAX - 4, FormatError::OffsetOverflow { offset: u64::MAX - 4, length: 512 }, 0)]
    #[case::backend_error(512, 0, FormatError::InvalidLinkType(99), 1)]
    fn indexed_search_checks_resolved_bounds_before_reading_metadata(
        #[case] base: u64,
        #[case] root: u64,
        #[case] expected: FormatError,
        #[case] metadata_reads: usize,
    ) {
        let bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, root, 1).build(WIDTHS);
        let header = BTreeV2Header::parse(&bytes, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OFFSET_WIDTH).unwrap();
        let source = FailingMetadataSource {
            reads: Cell::new(0),
        };
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
        assert_eq!(
            search_btree_v2(
                &PhysicalSource::for_test(&source, addresses).stored(),
                &layout,
                |_| panic!("the metadata read fails before comparing records"),
            )
            .unwrap_err(),
            expected,
        );
        assert_eq!(source.reads.get(), metadata_reads);
    }

    struct FailingMetadataSource {
        reads: Cell<usize>,
    }

    impl Source for FailingMetadataSource {
        fn len(&self) -> u64 {
            u64::MAX
        }
        fn read_at(&self, _: u64, _: &mut [u8]) -> Result<(), FormatError> {
            Err(FormatError::InvalidLinkType(98))
        }
        fn read_metadata(&self, _: u64, _: usize) -> Result<MetadataBytes<'_>, FormatError> {
            self.reads.set(self.reads.get() + 1);
            Err(FormatError::InvalidLinkType(99))
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
            search_btree_v2(
                &PhysicalSource::for_test(&source, addresses).stored(),
                &layout,
                |record| {
                    assert_eq!(
                        source.read_metadata_at(0, btree_v2::SIGNATURE.len())?,
                        btree_v2::SIGNATURE
                    );
                    Ok(record[0].cmp(&0))
                }
            )
            .unwrap(),
            Some(rec(0))
        );
        assert_eq!(
            search_btree_v2(
                &PhysicalSource::for_test(&source, addresses).stored(),
                &layout,
                |_| { Err(FormatError::InvalidLinkType(99)) }
            )
            .unwrap_err(),
            FormatError::InvalidLinkType(99)
        );
    }

    #[rstest::rstest]
    #[case::zero_base(0, 0)]
    #[case::userblock(512, 512)]
    #[case::forward_relocated(0, 512)]
    #[case::backward_relocated(512, 0)]
    fn stored_tree_operations_use_the_reconciled_context(
        #[case] encoded_base: u64,
        #[case] signature: u64,
        #[values(false, true)] resident: bool,
    ) {
        let tree = depth_3_tree();
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let header_len = btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 0, 0)
            .build(WIDTHS)
            .len();
        let addresses = OpenedImageContext::reconcile(
            BaseAddress::new(encoded_base),
            encoded_base + tree.file.len().to_u64(),
            signature,
        )
        .unwrap()
        .addresses();
        let mut bytes = vec![0; signature.to_usize().unwrap()];
        bytes.extend_from_slice(&tree.file);
        let source = OriginSource {
            bytes: &bytes,
            origin: 37,
        };
        let source_view = PhysicalSource::for_test(&source, addresses).stored();
        let resident_view = PhysicalBytes::for_test(&bytes, addresses).stored().unwrap();
        let reader: &dyn StoredRead = if resident {
            &resident_view
        } else {
            &source_view
        };
        assert_eq!(
            search_btree_v2(reader, &layout, |record| Ok(record[0].cmp(&14))).unwrap(),
            Some(rec(14)),
        );
        assert_eq!(
            collect_btree_v2_records_from_source(
                &BTreeV2Source::new(reader),
                &header,
                OFFSET_WIDTH
            )
            .unwrap(),
            (0..15)
                .map(|id| BTreeV2Record { data: rec(id) })
                .collect::<Vec<_>>(),
        );
        let mut expected = tree
            .nodes
            .iter()
            .map(|address| {
                Extent::new(StoredAddress::new(*address), u64::from(BTREE_V2_NODE_SIZE)).unwrap()
            })
            .collect::<Vec<_>>();
        expected.push(Extent::new(StoredAddress::ZERO, header_len.to_u64()).unwrap());
        expected.sort_unstable();
        assert_eq!(
            collect_btree_v2_storage_extents(reader, StoredAddress::ZERO, FORMAT_WIDTHS).unwrap(),
            expected,
        );
        let origin = signature + if resident { 0 } else { 37 };
        assert_eq!(reader.metadata_origin().unwrap().get(), origin);
        let adapter = BTreeV2Source::new(reader);
        let metadata = adapter.read_metadata(0, header_len).unwrap();
        assert_eq!(metadata.offset(), origin);
        assert_eq!(metadata.as_ref(), &tree.file[..header_len]);
        let resident = BTreeV2Source::resident(resident_view);
        let metadata = resident.read_metadata(0, header_len).unwrap();
        assert_eq!(metadata.offset(), signature);
        assert_eq!(metadata.as_ref(), &tree.file[..header_len]);
        assert_eq!(
            metadata.as_ptr(),
            bytes[signature.to_usize().unwrap()..].as_ptr()
        );
    }

    struct OriginSource<'a> {
        bytes: &'a [u8],
        origin: u64,
    }

    impl Source for OriginSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len().to_u64()
        }
        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
            BytesSource::new(self.bytes).read_at(offset, buffer)
        }
        fn metadata_origin(&self) -> Result<u64, FormatError> {
            Ok(self.origin)
        }
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
            search_btree_v2(
                &PhysicalBytes::for_test(tree.file.as_slice(), addresses)
                    .stored()
                    .unwrap(),
                &layout,
                |record| Ok(record[0].cmp(&14))
            )
            .unwrap(),
            Some(rec(14))
        );
        assert_eq!(
            search_btree_v2(
                &PhysicalBytes::for_test(tree.file.as_slice(), addresses)
                    .stored()
                    .unwrap(),
                &layout,
                |record| Ok(record[0].cmp(&0))
            )
            .unwrap_err(),
            FormatError::InvalidBTreeV2Signature
        );
    }

    #[test]
    fn indexed_search_reports_a_truncated_visited_node() {
        let tree = depth_3_tree();
        let header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let expected = tree.nodes[14] + layout.root().unwrap().encoded_len().unwrap() as u64;
        let bytes = &tree.file[..usize::try_from(expected - 1).unwrap()];
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        assert_eq!(
            search_btree_v2(
                &PhysicalBytes::for_test(bytes, addresses).stored().unwrap(),
                &layout,
                |_| Ok(Ordering::Equal)
            )
            .unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected,
                available: expected - 1
            }
        );
        assert_eq!(
            search_btree_v2(
                &PhysicalSource::for_test(&BytesSource::new(bytes), addresses).stored(),
                &layout,
                |_| Ok(Ordering::Equal)
            )
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
            search_btree_v2(
                &PhysicalBytes::for_test(tree.file.as_slice(), addresses)
                    .stored()
                    .unwrap(),
                &layout,
                |record| Ok(record[0].cmp(&0))
            )
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
        let expected: Vec<_> = (0..15).map(|id| BTreeV2Record { data: rec(id) }).collect();
        assert_eq!(
            collect_btree_v2_records(
                crate::source::zero_base_bytes(&tree.file),
                &header,
                OffsetWidth::Eight
            )
            .unwrap(),
            expected
        );

        #[cfg(feature = "std")]
        {
            let source = BytesSource::new(&tree.file);
            let header = BTreeV2Header::parse_from_source(
                &StoredMetadata(&crate::source::zero_base_source(&source)),
                0,
                OFFSET_SIZE,
                LENGTH_SIZE,
            )
            .unwrap();
            assert_eq!(
                collect_btree_v2_records_from_source(
                    &BTreeV2Source::new(&crate::source::zero_base_source(&source)),
                    &header,
                    OffsetWidth::Eight
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn record_walks_preserve_legacy_subtree_count_compatibility() {
        let tree = depth_3_tree();
        let mut header = BTreeV2Header::parse(&tree.file, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        header.total_records = 1;
        let expected: Vec<_> = (0..15).map(|id| BTreeV2Record { data: rec(id) }).collect();
        assert_eq!(
            collect_btree_v2_records(
                crate::source::zero_base_bytes(&tree.file),
                &header,
                OFFSET_WIDTH
            )
            .unwrap(),
            expected
        );
        assert_eq!(
            collect_btree_v2_records_from_source(
                &BTreeV2Source::new(&crate::source::zero_base_source(&BytesSource::new(
                    &tree.file
                ))),
                &header,
                OFFSET_WIDTH
            )
            .unwrap(),
            expected
        );
    }

    #[test]
    fn storage_walk_of_a_leaf_returns_header_and_full_node() {
        let (file, root) = single_leaf_tree(WIDTHS);
        let source = BytesSource::new(&file);
        let extents = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
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
                Extent::new(StoredAddress::new(0), header_len).unwrap(),
                Extent::new(StoredAddress::new(root), u64::from(BTREE_V2_NODE_SIZE)).unwrap(),
            ]
        );
    }

    #[test]
    fn storage_walk_enumerates_every_depth_3_node_once() {
        let tree = depth_3_tree();
        let source = BytesSource::new(&tree.file);
        let extents = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
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
                .into_iter()
                .map(StoredAddress::new)
                .collect::<Vec<_>>()
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
        let extents = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
        .unwrap();
        assert_eq!(
            extents,
            [Extent::new(StoredAddress::new(0), header.len().to_u64()).unwrap()]
        );
    }

    #[test]
    fn storage_walk_rejects_an_empty_tree_with_a_defined_root() {
        let header = btree_v2::Header::new(5, 11, 0x100, 0).build(WIDTHS);
        let source = BytesSource::new(header);

        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
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

        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
        .unwrap_err();

        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_rejects_a_nonempty_tree_without_a_root() {
        let root = StoredAddress::undefined(OFFSET_WIDTH).get();
        let header = btree_v2::Header::new(5, 11, root, 1).build(WIDTHS);
        let source = BytesSource::new(header);

        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
        .unwrap_err();

        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_reports_a_header_extent_above_four_gibibytes() {
        let source = BytesSource::new([]);
        let address = 4_294_967_536;
        assert_eq!(
            collect_btree_v2_storage_extents(
                &crate::source::zero_base_source(&source),
                StoredAddress::new(address),
                FORMAT_WIDTHS
            )
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
        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
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
        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
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

        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
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
        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
        .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Version(1));
    }

    #[test]
    fn storage_walk_rejects_a_node_with_a_different_tree_type() {
        let (mut file, root) = single_leaf_tree(WIDTHS);
        file[root.to_usize().unwrap() + 5] = 6;
        let source = BytesSource::new(file);
        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
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

        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
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

        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
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
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
        .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn storage_walk_rejects_a_child_past_eof() {
        let (file, bad_address) = tree_with_one_invalid_child(4096);
        let source = BytesSource::new(&file);
        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
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
        let err = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
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
                &crate::source::zero_base_source(&source),
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
                        StoredAddress::new(0),
                        hdf5_pure_format::__private::btree_v2_header_size(
                            offset_width,
                            length_width,
                        )
                        .to_u64(),
                    )
                    .unwrap(),
                    Extent::new(StoredAddress::new(root), u64::from(BTREE_V2_NODE_SIZE)).unwrap(),
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
            &crate::source::zero_base_source(&source),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
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
            &crate::source::zero_base_source(&memory),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
        )
        .unwrap();
        let seek_extents = collect_btree_v2_storage_extents(
            &crate::source::zero_base_source(&seek),
            StoredAddress::new(0),
            FORMAT_WIDTHS,
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
        let records = collect_btree_v2_records(
            crate::source::zero_base_bytes(&file_data),
            &hdr,
            OffsetWidth::Eight,
        )
        .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data, rec1.to_vec());
        assert_eq!(records[1].data, rec2.to_vec());
    }

    #[test]
    fn empty_tree() {
        let header = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        let hdr = BTreeV2Header::parse(&header, 0, 8, 8).unwrap();
        let records = collect_btree_v2_records(
            crate::source::zero_base_bytes(&header),
            &hdr,
            OffsetWidth::Eight,
        )
        .unwrap();
        assert_eq!(records, []);
        assert_eq!(
            collect_btree_v2_records_from_source(
                &BTreeV2Source::new(&crate::source::zero_base_source(&BytesSource::new(&header))),
                &hdr,
                OffsetWidth::Eight
            )
            .unwrap(),
            []
        );
    }

    #[rstest::rstest]
    #[case::signature(0, b'X', FormatError::InvalidBTreeV2Signature)]
    #[case::version(4, 1, FormatError::InvalidBTreeV2Version(1))]
    #[case::owner_type(5, 11, FormatError::InvalidBTreeNodeType(11))]
    fn record_walks_reject_invalid_node_prefixes(
        #[values(
            NodePosition::RootLeaf,
            NodePosition::RootInternal,
            NodePosition::ChildInternal,
            NodePosition::ChildLeaf
        )]
        position: NodePosition,
        #[values(false, true)] restamp: bool,
        #[case] at: usize,
        #[case] value: u8,
        #[case] expected: FormatError,
    ) {
        let (mut image, start, encoded_len) = position.image();
        image[start + at] = value;
        if restamp {
            test_util::checksum::restamp(&mut image, start, encoded_len);
        }
        let header = BTreeV2Header::parse(&image, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        assert_eq!(
            collect_btree_v2_records(
                crate::source::zero_base_bytes(&image),
                &header,
                OFFSET_WIDTH
            )
            .unwrap_err(),
            expected
        );
        let source = BytesSource::new(&image);
        assert_eq!(
            collect_btree_v2_records_from_source(
                &BTreeV2Source::new(&crate::source::zero_base_source(&source)),
                &header,
                OFFSET_WIDTH
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(
            collect_btree_v2_storage_extents(
                &crate::source::zero_base_source(&source),
                StoredAddress::new(0),
                FORMAT_WIDTHS
            )
            .unwrap_err(),
            expected
        );
    }

    #[derive(Clone, Copy)]
    enum NodePosition {
        ChildInternal,
        ChildLeaf,
        RootInternal,
        RootLeaf,
    }

    impl NodePosition {
        // `H5B2__hdr_init` in `H5B2hdr.c` (HDF5 2.2.0) computes one byte for direct counts and
        // two bytes for subtree counts in these internal nodes.
        fn image(self) -> (Vec<u8>, usize, usize) {
            let leaf_extent = btree_v2::leaf(BTREE_V2_LINK_NAME, &[rec(0)]).len();
            let (index, extent) = match self {
                Self::ChildInternal => (
                    12,
                    btree_v2::NODE_PREFIX
                        + rec(0).len()
                        + 2 * (WIDTHS.offset + 1 + 2)
                        + core::mem::size_of::<u32>(),
                ),
                Self::ChildLeaf => (0, leaf_extent),
                Self::RootInternal => (
                    14,
                    btree_v2::NODE_PREFIX
                        + rec(0).len()
                        + 2 * (WIDTHS.offset + 1 + 2)
                        + core::mem::size_of::<u32>(),
                ),
                Self::RootLeaf => {
                    let (image, root) = single_leaf_tree(WIDTHS);
                    return (image, root.to_usize().unwrap(), leaf_extent);
                }
            };
            let tree = depth_3_tree();
            (tree.file, tree.nodes[index].to_usize().unwrap(), extent)
        }
    }

    #[rstest::rstest]
    #[case::one_byte(1)]
    #[case::complete_field(4)]
    fn record_walks_require_the_complete_checksum_field(#[case] missing: usize) {
        let (mut image, start, encoded_len) = NodePosition::RootLeaf.image();
        image.truncate(start + encoded_len - missing);
        let header = BTreeV2Header::parse(&image, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();
        let expected = FormatError::UnexpectedEof {
            expected: encoded_len,
            available: encoded_len - missing,
        };
        assert_eq!(
            collect_btree_v2_records(
                crate::source::zero_base_bytes(&image),
                &header,
                OFFSET_WIDTH
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(
            collect_btree_v2_records_from_source(
                &BTreeV2Source::new(&crate::source::zero_base_source(&BytesSource::new(&image))),
                &header,
                OFFSET_WIDTH
            )
            .unwrap_err(),
            expected
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_btree_matches_buffered() {
        let rec1 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let rec2 = [11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let leaf = build_leaf_node(5, &[&rec1, &rec2]);
        let leaf_offset = 256usize;
        let header = btree_v2::Header::new(5, 11, leaf_offset as u64, 2).build(WIDTHS);
        let mut file_data = vec![0u8; 512];
        file_data[..header.len()].copy_from_slice(&header);
        file_data[leaf_offset..leaf_offset + leaf.len()].copy_from_slice(&leaf);

        let hdr = BTreeV2Header::parse(&file_data, 0, 8, 8).unwrap();
        let buffered: Vec<_> = collect_btree_v2_records(
            crate::source::zero_base_bytes(&file_data),
            &hdr,
            OffsetWidth::Eight,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.data)
        .collect();

        let mem = BytesSource::new(&file_data);
        let hdr_mem = BTreeV2Header::parse_from_source(
            &StoredMetadata(&crate::source::zero_base_source(&mem)),
            0,
            8,
            8,
        )
        .unwrap();
        assert_eq!(hdr_mem.root_node_address, hdr.root_node_address);
        let from_mem: Vec<_> = collect_btree_v2_records_from_source(
            &BTreeV2Source::new(&crate::source::zero_base_source(&mem)),
            &hdr_mem,
            OffsetWidth::Eight,
        )
        .unwrap()
        .into_iter()
        .map(|r| r.data)
        .collect();

        let seek = ReadSeekSource::new(Cursor::new(file_data)).unwrap();
        let hdr_seek = BTreeV2Header::parse_from_source(
            &StoredMetadata(&crate::source::zero_base_source(&seek)),
            0,
            8,
            8,
        )
        .unwrap();
        let from_seek: Vec<_> = collect_btree_v2_records_from_source(
            &BTreeV2Source::new(&crate::source::zero_base_source(&seek)),
            &hdr_seek,
            OffsetWidth::Eight,
        )
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
        let image = plan
            .serialize(
                &records,
                nodes_address,
                OffsetWidth::Eight,
                LengthWidth::Eight,
            )
            .unwrap();
        let mut file = image.header.clone();
        file.extend_from_slice(&image.nodes);

        let header = BTreeV2Header::parse(&file, 0, OFFSET_SIZE, LENGTH_SIZE).expect("header");
        assert_eq!(header.node_size, BTREE_V2_NODE_SIZE);
        assert_eq!(header.total_records, count as u64);
        let read = collect_btree_v2_records(
            crate::source::zero_base_bytes(&file),
            &header,
            OffsetWidth::Eight,
        )
        .expect("read");
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
