#![cfg(feature = "__hdf5-1.10")]
//! Allocation ranges decoded from controlled fractal-heap fixtures.

use std::num::NonZeroU16;
use std::ops::Range;

use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    BTREE_V2_HUGE_OBJECT, BTREE_V2_HUGE_OBJECT_DIRECT, BTreeV2Header, BTreeV2NodeInfo,
    BTreeV2Record, FormatWidths, FractalHeapFiltering, FractalHeapHeader,
    FreeSpaceManagerHeaderFrame, LengthWidth, OffsetWidth,
};
use test_util::btree_v2;
use test_util::checksum;
use test_util::fractal_heap;

/// A fixture's heap metadata and disjoint allocated ranges.
///
/// Addresses are relative to the file base, and the input bytes begin at that base. The fixtures
/// use eight-byte addresses and lengths. Filtered direct blocks retain their stored sizes, and
/// manager section lists include their full allocated capacity.
pub(super) struct Heap<'a> {
    /// The caller's parsed heap header.
    pub(super) header: &'a FractalHeapHeader,
    /// The internal free-space manager header, when present.
    pub(super) manager: Option<FreeSpaceManagerHeaderFrame>,
    /// Nonempty, in-file allocations in address order.
    pub(super) allocations: Vec<Allocation>,
    /// The number of managed indirect blocks, including nested blocks.
    pub(super) indirect_blocks: usize,
    /// The huge-object index depth, when the index is present.
    pub(super) huge_index_depth: Option<u16>,
}

impl<'a> Heap<'a> {
    /// Decodes the allocations of the heap whose parsed header is `header`.
    ///
    /// # Panics
    ///
    /// Panics if a fixture allocation is empty, out of bounds, or overlaps another allocation,
    /// or if the metadata fails the fixture's structural assertions. Filtered huge-object indexes
    /// are unsupported and cause a panic.
    pub(super) fn new(bytes: &[u8], address: u64, header: &'a FractalHeapHeader) -> Self {
        let at = usize::try_from(address).unwrap();
        let mut heap = Self {
            header,
            manager: None,
            allocations: vec![Allocation::new(
                bytes,
                address,
                fractal_heap::encoded_header_len(bytes, at),
                AllocationClass::Metadata,
            )],
            indirect_blocks: 0,
            huge_index_depth: None,
        };
        heap.managed(bytes, address);
        heap.huge(bytes);
        let manager = heap.header.managed_block_free_space_manager_address;
        if !manager.is_undefined(OffsetWidth::Eight) {
            let at = usize::try_from(manager.get()).unwrap();
            let widths = FormatWidths::from_sizes(8, 8).unwrap();
            let frame =
                FreeSpaceManagerHeaderFrame::parse(widths, bytes.get(at..).unwrap()).unwrap();
            heap.push(
                bytes,
                manager.get(),
                format::free_space_manager_header_len(widths),
            );
            if let DecodedStoredAddress::Defined(list) = frame.section_list_addr() {
                heap.push(bytes, list.get(), frame.section_list_allocated());
            } else {
                assert_eq!(frame.section_list_allocated(), 0);
            }
            heap.manager = Some(frame);
        }
        heap.allocations
            .sort_unstable_by_key(|allocation| allocation.range.start);
        for pair in heap.allocations.windows(2) {
            assert!(
                pair[0].range.end <= pair[1].range.start,
                "overlapping heap allocations"
            );
        }
        heap
    }

    /// Returns each allocation as its base-relative address and allocated length.
    pub(super) fn extents(&self) -> impl ExactSizeIterator<Item = (u64, u64)> + '_ {
        self.allocations.iter().map(Allocation::extent)
    }

    fn push(&mut self, bytes: &[u8], address: u64, length: u64) {
        self.allocations.push(Allocation::new(
            bytes,
            address,
            length,
            AllocationClass::Metadata,
        ));
    }

    fn managed(&mut self, bytes: &[u8], address: u64) {
        let root = self.header.root_block_address;
        if root.is_undefined(OffsetWidth::Eight) {
            assert_eq!(self.header.allocated_managed_space, 0);
            assert_eq!(self.header.managed_objects_count, 0);
            return;
        }
        assert!(self.header.starting_block_size.is_power_of_two());
        assert!(self.header.max_direct_block_size.is_power_of_two());
        assert!(self.header.max_direct_block_size >= self.header.starting_block_size);
        assert!(self.header.table_width.is_power_of_two());
        assert!((1..=64).contains(&self.header.max_heap_size));
        if self.header.current_rows_in_root_indirect_block == 0 {
            let size = match self.header.filtering {
                FractalHeapFiltering::Unfiltered => self.header.starting_block_size,
                FractalHeapFiltering::Filtered {
                    root_direct_block_size,
                    ..
                } => root_direct_block_size,
            };
            self.push(bytes, root.get(), size);
            if matches!(self.header.filtering, FractalHeapFiltering::Unfiltered) {
                self.block_prefix(
                    bytes,
                    root.get(),
                    fractal_heap::DIRECT_BLOCK_SIGNATURE,
                    address,
                    0,
                );
            }
            return;
        }
        self.indirect(
            bytes,
            address,
            root.get(),
            self.header.current_rows_in_root_indirect_block,
            0,
        );
    }

    // The doubling-table rows and filtered entry widths are defined in section
    // `subsec_fmt4_infra_fractalheap`, version 4.0.
    fn indirect(&mut self, bytes: &[u8], owner: u64, address: u64, rows: u16, heap_offset: u64) {
        assert!(rows > 0);
        assert!(
            !self
                .allocations
                .iter()
                .any(|allocation| allocation.extent().0 == address),
            "repeated heap block at {address}"
        );
        let width = u64::from(self.header.table_width);
        let direct_rows = self
            .header
            .max_direct_block_size
            .trailing_zeros()
            .checked_sub(self.header.starting_block_size.trailing_zeros())
            .unwrap()
            .checked_add(2)
            .unwrap();
        let filtered = matches!(self.header.filtering, FractalHeapFiltering::Filtered { .. });
        let direct_entries = u64::from(rows)
            .min(u64::from(direct_rows))
            .checked_mul(width)
            .unwrap();
        let entries = u64::from(rows).checked_mul(width).unwrap();
        let offset_width = u64::from(self.header.max_heap_size).div_ceil(8);
        let prefix_len = u64::try_from(fractal_heap::DIRECT_BLOCK_OWNER_AT)
            .unwrap()
            .checked_add(u64::from(OffsetWidth::Eight.get()))
            .unwrap()
            .checked_add(offset_width)
            .unwrap();
        let len = prefix_len
            .checked_add(entries.checked_mul(8).unwrap())
            .unwrap()
            .checked_add(if filtered {
                direct_entries
                    .checked_mul(u64::from(LengthWidth::Eight.get()) + FILTER_MASK_LEN)
                    .unwrap()
            } else {
                0
            })
            .unwrap()
            .checked_add(u64::try_from(checksum::CHECKSUM).unwrap())
            .unwrap();
        self.push(bytes, address, len);
        self.indirect_blocks = self.indirect_blocks.checked_add(1).unwrap();
        self.block_prefix(
            bytes,
            address,
            fractal_heap::INDIRECT_BLOCK_SIGNATURE,
            owner,
            heap_offset,
        );
        let mut cursor = usize::try_from(address.checked_add(prefix_len).unwrap()).unwrap();
        let mut row_offset = 0u64;
        for row in 0..u32::from(rows) {
            let block_size = self
                .header
                .starting_block_size
                .checked_mul(1u64.checked_shl(row.saturating_sub(1)).unwrap())
                .unwrap();
            for column in 0..width {
                let child = read_integer(bytes, &mut cursor, 8);
                let size = if filtered && row < direct_rows {
                    let size = read_integer(bytes, &mut cursor, 8);
                    read_integer(
                        bytes,
                        &mut cursor,
                        usize::try_from(FILTER_MASK_LEN).unwrap(),
                    );
                    size
                } else {
                    block_size
                };
                if child == u64::MAX {
                    continue;
                }
                let child_offset = heap_offset
                    .checked_add(row_offset)
                    .unwrap()
                    .checked_add(column.checked_mul(block_size).unwrap())
                    .unwrap();
                if row < direct_rows {
                    self.push(bytes, child, size);
                    if !filtered {
                        self.block_prefix(
                            bytes,
                            child,
                            fractal_heap::DIRECT_BLOCK_SIGNATURE,
                            owner,
                            child_offset,
                        );
                    }
                } else {
                    let first_row_bits = self
                        .header
                        .starting_block_size
                        .trailing_zeros()
                        .checked_add(width.trailing_zeros())
                        .unwrap();
                    let child_rows = block_size
                        .trailing_zeros()
                        .checked_sub(first_row_bits)
                        .unwrap()
                        .checked_add(1)
                        .unwrap();
                    assert!(child_rows < u32::from(rows));
                    self.indirect(
                        bytes,
                        owner,
                        child,
                        u16::try_from(child_rows).unwrap(),
                        child_offset,
                    );
                }
            }
            row_offset = row_offset
                .checked_add(width.checked_mul(block_size).unwrap())
                .unwrap();
        }
        assert_eq!(
            u64::try_from(cursor)
                .unwrap()
                .checked_add(u64::try_from(checksum::CHECKSUM).unwrap())
                .unwrap(),
            address.checked_add(len).unwrap()
        );
    }

    fn block_prefix(
        &self,
        bytes: &[u8],
        address: u64,
        signature: &[u8; 4],
        owner: u64,
        offset: u64,
    ) {
        let allocation = self
            .allocations
            .iter()
            .find(|allocation| allocation.extent().0 == address)
            .unwrap();
        let bytes = bytes.get(allocation.range.clone()).unwrap();
        let mut cursor = 0usize;
        let end = cursor.checked_add(signature.len()).unwrap();
        assert_eq!(bytes.get(cursor..end).unwrap(), signature);
        cursor = end;
        assert_eq!(
            read_integer(bytes, &mut cursor, 1),
            u64::from(fractal_heap::BLOCK_VERSION)
        );
        assert_eq!(read_integer(bytes, &mut cursor, 8), owner);
        assert_eq!(
            read_integer(
                bytes,
                &mut cursor,
                usize::from(self.header.max_heap_size).div_ceil(8)
            ),
            offset
        );
    }

    fn huge(&mut self, bytes: &[u8]) {
        let address = self.header.btree_huge_objects_address;
        if address.is_undefined(OffsetWidth::Eight) {
            assert_eq!(self.header.huge_objects_count, 0);
            return;
        }
        assert_eq!(
            self.header.filtering,
            FractalHeapFiltering::Unfiltered,
            "filtered huge-object indexes are unsupported fixtures"
        );
        let tree =
            BTreeV2Header::parse(bytes, usize::try_from(address.get()).unwrap(), 8, 8).unwrap();
        assert!(
            tree.tree_type == BTREE_V2_HUGE_OBJECT || tree.tree_type == BTREE_V2_HUGE_OBJECT_DIRECT
        );
        self.push(
            bytes,
            address.get(),
            u64::try_from(format::btree_v2_header_size(
                OffsetWidth::Eight,
                LengthWidth::Eight,
            ))
            .unwrap(),
        );
        self.huge_index_depth = Some(tree.depth);
        let node_info = BTreeV2NodeInfo::compute(
            tree.node_size,
            tree.record_size,
            OffsetWidth::Eight,
            tree.depth,
        );
        let mut records = Vec::new();
        self.huge_node(
            bytes,
            &tree,
            &node_info,
            tree.root_node_address.get(),
            tree.num_records_in_root,
            tree.depth,
            &mut records,
        );
        assert_eq!(
            u64::try_from(records.len()).unwrap(),
            self.header.huge_objects_count
        );
        assert_eq!(tree.total_records, self.header.huge_objects_count);
        let mut total = 0u64;
        for record in records {
            let (address, length) = if tree.tree_type == BTREE_V2_HUGE_OBJECT {
                let record = record
                    .huge_object(OffsetWidth::Eight, LengthWidth::Eight)
                    .unwrap();
                (record.address.get(), record.length)
            } else {
                let record = record
                    .huge_object_direct(OffsetWidth::Eight, LengthWidth::Eight)
                    .unwrap();
                (record.address.get(), record.length)
            };
            self.allocations.push(Allocation::new(
                bytes,
                address,
                length,
                AllocationClass::RawData,
            ));
            total = total.checked_add(length).unwrap();
        }
        assert_eq!(total, self.header.huge_objects_size);
    }

    // Version 2 B-tree nodes store records after a six-byte prefix, as defined in section
    // `subsubsec_fmt4_infra_btrees_v2`, version 4.0.
    #[expect(
        clippy::too_many_arguments,
        reason = "The recursive walk shares one record accumulator and immutable tree geometry."
    )]
    fn huge_node(
        &mut self,
        bytes: &[u8],
        tree: &BTreeV2Header,
        node_info: &BTreeV2NodeInfo,
        address: u64,
        count: u16,
        depth: u16,
        records: &mut Vec<BTreeV2Record>,
    ) {
        assert!(
            !self
                .allocations
                .iter()
                .any(|allocation| allocation.extent().0 == address),
            "repeated huge-index node at {address}"
        );
        let allocation = Allocation::new(
            bytes,
            address,
            u64::from(tree.node_size),
            AllocationClass::Metadata,
        );
        let node = bytes.get(allocation.range.clone()).unwrap();
        self.allocations.push(allocation);
        assert_eq!(
            *node.get(btree_v2::LEAF_SIGNATURE.len()).unwrap(),
            BTREE_NODE_VERSION
        );
        assert_eq!(*node.get(5).unwrap(), tree.tree_type);
        let Some(depth) = NonZeroU16::new(depth) else {
            records.extend(
                format::parse_btree_v2_leaf_records(node, 0, count, tree.record_size).unwrap(),
            );
            return;
        };
        let children = format::parse_btree_v2_internal_child_pointers(
            node,
            count,
            depth,
            tree.record_size,
            OffsetWidth::Eight,
            node_info,
        )
        .unwrap();
        for (index, (child, child_count)) in children.into_iter().enumerate() {
            self.huge_node(
                bytes,
                tree,
                node_info,
                child.get(),
                child_count,
                depth.get() - 1,
                records,
            );
            if index < usize::from(count) {
                let start = btree_v2::NODE_PREFIX
                    .checked_add(index.checked_mul(usize::from(tree.record_size)).unwrap())
                    .unwrap();
                let end = start.checked_add(usize::from(tree.record_size)).unwrap();
                records.push(BTreeV2Record {
                    data: node.get(start..end).unwrap().to_vec(),
                });
            }
        }
    }
}

/// A nonempty allocated range within the fixture bytes and its page class.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Allocation {
    range: Range<usize>,
    class: AllocationClass,
}

impl Allocation {
    fn new(bytes: &[u8], address: u64, length: u64, class: AllocationClass) -> Self {
        assert_ne!(length, 0, "empty allocation at {address}");
        let end = address.checked_add(length).unwrap();
        let range = usize::try_from(address).unwrap()..usize::try_from(end).unwrap();
        bytes.get(range.clone()).unwrap();
        Self { range, class }
    }

    /// Returns the base-relative address and allocated length.
    pub(super) fn extent(&self) -> (u64, u64) {
        (
            u64::try_from(self.range.start).unwrap(),
            u64::try_from(self.range.len()).unwrap(),
        )
    }

    /// Returns the page class for the allocated structure or payload.
    pub(super) fn class(&self) -> AllocationClass {
        self.class
    }
}

/// The page class for a heap allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AllocationClass {
    /// Heap blocks, index metadata, and internal free-space manager allocations.
    Metadata,
    /// Separately allocated huge-object payloads.
    RawData,
}

fn read_integer(bytes: &[u8], cursor: &mut usize, width: usize) -> u64 {
    assert!((1..=8).contains(&width));
    let end = cursor.checked_add(width).unwrap();
    let mut encoded = [0; 8];
    encoded
        .get_mut(..width)
        .unwrap()
        .copy_from_slice(bytes.get(*cursor..end).unwrap());
    *cursor = end;
    u64::from_le_bytes(encoded)
}

/// The size in bytes of a filtered direct block's filter mask.
///
/// Defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const FILTER_MASK_LEN: u64 = 4;

/// The on-disk node version in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
const BTREE_NODE_VERSION: u8 = 0;
