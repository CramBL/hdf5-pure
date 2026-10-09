//! Reads fractal-heap objects and proves the file allocations owned by supported heaps.
//!
//! [`HeapObjectReader`] resolves managed, huge, and tiny objects from a file image or a [`Source`].
//! [`collect_fractal_heap_storage_from_source`] walks the structural allocation graph without
//! reading object payloads and returns [`FractalHeapStorage`] only when that graph is complete for
//! the supported heap shape.

use alloc::collections::BTreeSet;
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT_DIRECT;
use hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use hdf5_pure_format::__private::FractalHeapChild;
pub use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdView;
use hdf5_pure_format::__private::FractalHeapStorageError;
use hdf5_pure_format::__private::HugeObjectDirectRecord;
use hdf5_pure_format::__private::HugeObjectRecord;
use hdf5_pure_format::__private::HugeObjectReference;

use hdf5_pure_space::__private::Extent;

use crate::address::StoredAddress;
use crate::btree_v2::{
    BTreeV2Header, BTreeV2Record, collect_btree_v2_records, collect_btree_v2_records_from_source,
};
use crate::bytes;
use crate::convert::{Narrow, is_undefined_addr};
use crate::error::FormatError;
#[cfg(test)]
use crate::source::BytesSource;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Copy `len` bytes at `addr` out of an in-memory file image, bounds-checked.
fn slice_object(file_data: &[u8], addr: u64, len: usize) -> Result<Vec<u8>, FormatError> {
    let start = addr.to_usize()?;
    let end = start.checked_add(len).ok_or(FormatError::OffsetOverflow {
        offset: addr,
        length: len as u64,
    })?;
    if end > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: end,
            available: file_data.len(),
        });
    }
    Ok(file_data[start..end].to_vec())
}

/// Read `len` bytes at `addr` from a streaming source.
fn read_object_at_source<S: Source + ?Sized>(
    source: &S,
    addr: u64,
    len: usize,
) -> Result<Vec<u8>, FormatError> {
    source.read_metadata_at(addr, len)
}

/// Checks the B-tree header against the indexed, unfiltered huge-object record layout.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedHugeObjectBTree`] if the record type differs or the
/// records are too short, and [`FormatError::InvalidOffsetSize`] or
/// [`FormatError::InvalidLengthSize`] if a field width is not 2, 4, or 8.
fn check_huge_object_btree(
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
) -> Result<(), FormatError> {
    let required = HugeObjectRecord::size(
        OffsetWidth::try_from(offset_size)?,
        LengthWidth::try_from(length_size)?,
    );
    if header.tree_type != BTREE_V2_HUGE_OBJECT || header.record_size < required {
        return Err(FormatError::UnexpectedHugeObjectBTree {
            tree_type: header.tree_type,
            record_size: usize::from(header.record_size),
            required: usize::from(required),
        });
    }
    Ok(())
}

/// Every huge object a heap holds, decoded from its huge-objects v2 B-tree.
///
/// Resolving even one huge object costs the whole index: that B-tree is read by
/// collecting its records rather than by descending to a key. So the records are
/// decoded here once, and kept by the [`HeapObjectReader`] that asked for them.
///
/// Only the type-1 record layout appears here, which
/// [`check_huge_object_btree`] establishes before any record reaches this.
struct HugeObjectIndex {
    /// `(id, address, length)`, ordered by id so a lookup is a binary search.
    /// The B-tree already stores them in that order; sorting again costs one
    /// pass and means correctness here does not rest on that.
    entries: Vec<(u64, StoredAddress, u64)>,
}

#[cfg(test)]
std::thread_local! {
    /// How many huge-object indexes have been decoded on this thread.
    ///
    /// Parsing one index per object rather than one per walk returns exactly the
    /// same objects while making the walk quadratic in their number, so the
    /// count of decodes is the only thing a test can hold the invariant to. The
    /// walks build their reader internally, so the count has to be reachable
    /// from outside the reader that did the decoding; it is per-thread because
    /// the test harness runs tests concurrently, and each test's walk runs on
    /// its own thread.
    static HUGE_INDEX_DECODES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Start counting huge-object index decodes on this thread from zero.
#[cfg(test)]
pub(crate) fn reset_huge_index_decodes() {
    HUGE_INDEX_DECODES.with(|count| count.set(0));
}

/// How many huge-object indexes this thread has decoded since the last reset.
#[cfg(test)]
pub(crate) fn huge_index_decodes() -> usize {
    HUGE_INDEX_DECODES.with(|count| count.get())
}

impl HugeObjectIndex {
    fn decode(
        records: &[BTreeV2Record],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        #[cfg(test)]
        HUGE_INDEX_DECODES.with(|count| count.set(count.get() + 1));

        let offset_width = OffsetWidth::try_from(offset_size)?;
        let length_width = LengthWidth::try_from(length_size)?;
        let mut entries = Vec::with_capacity(records.len());
        for record in records {
            let HugeObjectRecord {
                address,
                length,
                id,
            } = record.huge_object(offset_width, length_width)?;
            entries.push((id, address, length));
        }
        entries.sort_unstable();
        Ok(Self { entries })
    }

    /// The `(address, length)` of the object with `huge_id`.
    fn locate(&self, huge_id: u64) -> Result<(StoredAddress, u64), FormatError> {
        match self
            .entries
            .binary_search_by_key(&huge_id, |&(id, _, _)| id)
        {
            Ok(i) => {
                let (_, addr, len) = self.entries[i];
                Ok((addr, len))
            }
            Err(_) => Err(FormatError::HugeObjectNotFound(huge_id)),
        }
    }
}

/// Represents the complete set of file allocations owned by one supported fractal heap.
///
/// The set contains the `FRHP` header, every allocated managed direct and indirect block, the
/// huge-object version 2 B-tree when present, and every separately allocated huge-object payload.
/// Tiny objects are stored entirely in heap IDs, so the extent set omits them.
///
/// The walker returns this type only after it has accounted for every supported allocation and
/// proven the returned extents pairwise disjoint. Unsupported heap features prevent construction.
/// The allocation classes are defined in "Fractal Heap" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FractalHeapStorage {
    extents: Vec<Extent>,
}

#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
impl FractalHeapStorage {
    /// Returns the proven owned allocations in file-address order.
    pub(crate) fn extents(&self) -> &[Extent] {
        &self.extents
    }
}

/// Builds the complete ownership proof for the fractal heap at `heap_header_address`.
///
/// The read-only walk operates directly on [`Source`]. It bounds each complete allocation before
/// reading its structural bytes and does not read managed-object or huge-object payload contents to
/// establish ownership. Success accounts for the heap header, every allocated managed block, and
/// the complete huge-object index and payload set.
///
/// Filtered heaps are outside the supported ownership set because their managed-space
/// indirect-block entries use the filtered layout. A heap with a defined managed-space free-space
/// manager is also unsupported because this walker does not enumerate that manager's
/// client-specific metadata. Tiny objects require no allocation walk because their bytes are stored
/// in heap IDs.
///
/// The owned structures are defined in "Fractal Heap" of the [format specification, version
/// 4.0][spec]. This ownership boundary follows `H5HF_delete` and `H5HF__hdr_delete`, which treat
/// the managed blocks, managed-space free-space manager, huge-object tracker and payloads, and heap
/// header as one heap's storage (`H5HFhdr.c`, HDF5 2.2.0).
///
/// # Errors
///
/// Returns [`FractalHeapStorageError::UnsupportedOwnership`] for a filtered heap or a defined
/// managed-space free-space manager. Malformed geometry, duplicate managed-block addresses,
/// inconsistent block prefixes, contradictory empty-heap bookkeeping, an unexpected huge-object
/// index layout, invalid huge-object payload addresses, huge-object accounting mismatches, and
/// overlapping owned allocations return
/// [`FractalHeapStorageError::InvalidStorage`]. Truncated allocations and lower-level parser errors
/// are wrapped in [`FractalHeapStorageError::Format`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
pub(crate) fn collect_fractal_heap_storage_from_source<S: Source + ?Sized>(
    source: &S,
    heap_header_address: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<FractalHeapStorage, FractalHeapStorageError> {
    let offset_width = OffsetWidth::try_from(offset_size)?;
    let length_width = LengthWidth::try_from(length_size)?;
    reject_filtered_ownership(source, heap_header_address)?;
    let header = FractalHeapHeader::parse_from_source(
        &SourceMetadata(source),
        heap_header_address.get(),
        offset_size,
        length_size,
    )?;
    if !is_undefined_addr(
        header.managed_block_free_space_manager_address.get(),
        offset_size,
    ) {
        return Err(FractalHeapStorageError::UnsupportedOwnership(
            "managed-space free-space manager metadata",
        ));
    }

    // The parser reads through a bounded prefix window. The allocation itself has the exact
    // encoded header size.
    let header_len = u64::try_from(FractalHeapHeader::serialized_size(
        offset_width,
        length_width,
    ))
    .map_err(|_conversion_error| FractalHeapStorageError::InvalidStorage)?;
    let mut extents = vec![extent_in_source(
        source,
        heap_header_address.get(),
        header_len,
    )?];

    let mut managed = ManagedStorageWalk::new(
        source,
        &header,
        heap_header_address,
        offset_size,
        &mut extents,
    );
    managed.collect()?;

    collect_huge_storage(
        source,
        &header,
        offset_size,
        length_size,
        offset_width,
        length_width,
        &mut extents,
    )?;

    extents.sort_unstable();
    if extents
        .windows(2)
        .any(|pair| pair[0].end() > pair[1].start())
    {
        return Err(FractalHeapStorageError::InvalidStorage);
    }
    Ok(FractalHeapStorage { extents })
}

#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
fn reject_filtered_ownership<S: Source + ?Sized>(
    source: &S,
    heap_header_address: StoredAddress,
) -> Result<(), FractalHeapStorageError> {
    let prefix = source.read_metadata_at(heap_header_address.get(), 9)?;
    if &prefix[..4] != b"FRHP" {
        return Err(FormatError::InvalidFractalHeapSignature.into());
    }
    if prefix[4] != 0 {
        return Err(FormatError::InvalidFractalHeapVersion(prefix[4]).into());
    }
    if u16::from_le_bytes([prefix[7], prefix[8]]) != 0 {
        return Err(FractalHeapStorageError::UnsupportedOwnership(
            "filtered managed blocks",
        ));
    }
    Ok(())
}

/// Buffered convenience counterpart to [`collect_fractal_heap_storage_from_source`].
#[cfg(test)]
fn collect_fractal_heap_storage(
    file_data: &[u8],
    heap_header_address: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<FractalHeapStorage, FractalHeapStorageError> {
    collect_fractal_heap_storage_from_source(
        &BytesSource::new(file_data),
        heap_header_address,
        offset_size,
        length_size,
    )
}

#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
#[derive(Clone, Copy)]
enum ManagedBlock {
    Direct {
        address: StoredAddress,
        block_size: u64,
        heap_offset: u64,
    },
    Indirect {
        address: StoredAddress,
        nrows: u16,
        heap_offset: u64,
    },
}

#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
struct ManagedStorageWalk<'a, 'e, S: Source + ?Sized> {
    source: &'a S,
    header: &'a FractalHeapHeader,
    heap_header_address: StoredAddress,
    offset_size: u8,
    visited: BTreeSet<StoredAddress>,
    extents: &'e mut Vec<Extent>,
    allocated_direct_space: u64,
}

#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
impl<'a, 'e, S: Source + ?Sized> ManagedStorageWalk<'a, 'e, S> {
    fn new(
        source: &'a S,
        header: &'a FractalHeapHeader,
        heap_header_address: StoredAddress,
        offset_size: u8,
        extents: &'e mut Vec<Extent>,
    ) -> Self {
        Self {
            source,
            header,
            heap_header_address,
            offset_size,
            visited: BTreeSet::new(),
            extents,
            allocated_direct_space: 0,
        }
    }

    fn collect(&mut self) -> Result<(), FractalHeapStorageError> {
        // Validate the declared geometry even for an empty heap so malformed fields cannot become
        // accepted merely because no root happens to be allocated.
        self.header.storage_max_root_rows()?;
        let root = self.header.root_block_address;
        if is_undefined_addr(root.get(), self.offset_size) {
            if self.header.current_rows_in_root_indirect_block != 0
                || self.header.free_space_in_managed_blocks != 0
                || self.header.managed_space != 0
                || self.header.allocated_managed_space != 0
                || self.header.direct_block_allocation_iterator_offset != 0
                || self.header.managed_objects_count != 0
            {
                return Err(FractalHeapStorageError::InvalidStorage);
            }
            return Ok(());
        }

        let first = if self.header.current_rows_in_root_indirect_block == 0 {
            let block_size = self.header.storage_block_size_for_row(0)?;
            if self.header.managed_space != block_size {
                return Err(FractalHeapStorageError::InvalidStorage);
            }
            ManagedBlock::Direct {
                address: root,
                block_size,
                heap_offset: 0,
            }
        } else {
            if self.header.current_rows_in_root_indirect_block
                > self.header.storage_max_root_rows()?
                || self.header.managed_space
                    != self.header.storage_indirect_heap_size(
                        self.header.current_rows_in_root_indirect_block,
                    )?
            {
                return Err(FractalHeapStorageError::InvalidStorage);
            }
            ManagedBlock::Indirect {
                address: root,
                nrows: self.header.current_rows_in_root_indirect_block,
                heap_offset: 0,
            }
        };

        let mut pending = vec![first];
        while let Some(block) = pending.pop() {
            let address = match block {
                ManagedBlock::Direct { address, .. } | ManagedBlock::Indirect { address, .. } => {
                    address
                }
            };
            if !self.visited.insert(address) {
                return Err(FractalHeapStorageError::InvalidStorage);
            }

            match block {
                ManagedBlock::Direct {
                    address,
                    block_size,
                    heap_offset,
                } => self.collect_direct(address, block_size, heap_offset)?,
                ManagedBlock::Indirect {
                    address,
                    nrows,
                    heap_offset,
                } => self.collect_indirect(address, nrows, heap_offset, &mut pending)?,
            }
        }

        if self.allocated_direct_space != self.header.allocated_managed_space {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        Ok(())
    }

    fn collect_direct(
        &mut self,
        address: StoredAddress,
        block_size: u64,
        heap_offset: u64,
    ) -> Result<(), FractalHeapStorageError> {
        let block_offset_size = self.header.storage_block_offset_size()?;
        let prefix_len = 5usize
            .checked_add(usize::from(self.offset_size))
            .and_then(|n| n.checked_add(block_offset_size))
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        let encoded_prefix_len = prefix_len
            + if self.header.flags & FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED != 0 {
                4
            } else {
                0
            };
        if block_size
            < u64::try_from(encoded_prefix_len)
                .map_err(|_conversion_error| FractalHeapStorageError::InvalidStorage)?
        {
            return Err(FractalHeapStorageError::InvalidStorage);
        }

        let extent = extent_in_source(self.source, address.get(), block_size)?;
        let prefix = self.source.read_metadata_at(address.get(), prefix_len)?;
        validate_managed_block_prefix(
            &prefix,
            b"FHDB",
            self.heap_header_address,
            heap_offset,
            self.offset_size,
            block_offset_size,
        )?;
        self.extents.push(extent);
        self.allocated_direct_space = self
            .allocated_direct_space
            .checked_add(block_size)
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        Ok(())
    }

    fn collect_indirect(
        &mut self,
        address: StoredAddress,
        nrows: u16,
        heap_offset: u64,
        pending: &mut Vec<ManagedBlock>,
    ) -> Result<(), FractalHeapStorageError> {
        let block_size = self
            .header
            .storage_indirect_block_size(nrows, self.offset_size)?;
        let extent = extent_in_source(self.source, address.get(), block_size)?;
        let block = self
            .source
            .read_metadata_at(address.get(), block_size.to_usize()?)?;
        let block_offset_size = self.header.storage_block_offset_size()?;
        validate_managed_block_prefix(
            &block,
            b"FHIB",
            self.heap_header_address,
            heap_offset,
            self.offset_size,
            block_offset_size,
        )?;
        hdf5_pure_format::__private::verify_trailing(&block)?;
        self.extents.push(extent);

        let width = usize::from(self.header.table_width);
        let direct_rows = usize::from(nrows).min(self.header.storage_direct_rows()?);
        let entries_at = 5usize
            .checked_add(usize::from(self.offset_size))
            .and_then(|n| n.checked_add(block_offset_size))
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        let heap_limit =
            (self.header.max_heap_size < 64).then(|| 1u64 << u32::from(self.header.max_heap_size));

        for row in 0..usize::from(nrows) {
            let slot_size = self.header.storage_block_size_for_row(row)?;
            let child_rows = if row < direct_rows {
                None
            } else {
                let child_rows = self.header.storage_child_indirect_rows(row)?;
                if child_rows >= nrows
                    || self.header.storage_indirect_heap_size(child_rows)? != slot_size
                {
                    return Err(FractalHeapStorageError::InvalidStorage);
                }
                Some(child_rows)
            };
            let row_offset = self.header.storage_row_offset(row)?;
            for col in 0..width {
                let entry = row
                    .checked_mul(width)
                    .and_then(|n| n.checked_add(col))
                    .ok_or(FractalHeapStorageError::InvalidStorage)?;
                let pos = entries_at
                    .checked_add(
                        entry
                            .checked_mul(usize::from(self.offset_size))
                            .ok_or(FractalHeapStorageError::InvalidStorage)?,
                    )
                    .ok_or(FractalHeapStorageError::InvalidStorage)?;
                let child_address = bytes::read_offset(&block, pos, self.offset_size)?;
                if is_undefined_addr(child_address, self.offset_size) {
                    continue;
                }

                let child_offset = heap_offset
                    .checked_add(row_offset)
                    .and_then(|n| {
                        u64::try_from(col)
                            .ok()
                            .and_then(|col| col.checked_mul(slot_size))
                            .and_then(|delta| n.checked_add(delta))
                    })
                    .ok_or(FractalHeapStorageError::InvalidStorage)?;
                let child_end = child_offset
                    .checked_add(slot_size)
                    .ok_or(FractalHeapStorageError::InvalidStorage)?;
                if heap_limit.is_some_and(|limit| child_end > limit) {
                    return Err(FractalHeapStorageError::InvalidStorage);
                }

                pending.push(match child_rows {
                    Some(child_rows) => ManagedBlock::Indirect {
                        address: StoredAddress::new(child_address),
                        nrows: child_rows,
                        heap_offset: child_offset,
                    },
                    None => ManagedBlock::Direct {
                        address: StoredAddress::new(child_address),
                        block_size: slot_size,
                        heap_offset: child_offset,
                    },
                });
            }
        }
        Ok(())
    }
}

#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
fn validate_managed_block_prefix(
    block: &[u8],
    signature: &[u8; 4],
    heap_header_address: StoredAddress,
    heap_offset: u64,
    offset_size: u8,
    block_offset_size: usize,
) -> Result<(), FractalHeapStorageError> {
    bytes::ensure_len(block, 0, 5)?;
    if &block[..4] != signature {
        return Err(FormatError::InvalidFractalHeapSignature.into());
    }
    if block[4] != 0 {
        return Err(FormatError::InvalidFractalHeapVersion(block[4]).into());
    }
    let stored_header = bytes::read_offset(block, 5, offset_size)?;
    if stored_header != heap_header_address.get() {
        return Err(FractalHeapStorageError::InvalidStorage);
    }
    let offset_at = 5 + usize::from(offset_size);
    bytes::ensure_len(block, offset_at, block_offset_size)?;
    let mut stored_offset = 0u64;
    for (shift, byte) in block[offset_at..offset_at + block_offset_size]
        .iter()
        .copied()
        .enumerate()
    {
        stored_offset |= u64::from(byte) << (shift * 8);
    }
    if stored_offset != heap_offset {
        return Err(FractalHeapStorageError::InvalidStorage);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
fn collect_huge_storage<S: Source + ?Sized>(
    source: &S,
    heap: &FractalHeapHeader,
    offset_size: u8,
    length_size: u8,
    offset_width: OffsetWidth,
    length_width: LengthWidth,
    extents: &mut Vec<Extent>,
) -> Result<(), FractalHeapStorageError> {
    let tree_address = heap.btree_huge_objects_address;
    if is_undefined_addr(tree_address.get(), offset_size) {
        if heap.huge_objects_count != 0 || heap.huge_objects_size != 0 {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        return Ok(());
    }

    let header = BTreeV2Header::parse_from_source(
        &SourceMetadata(source),
        tree_address.get(),
        offset_size,
        length_size,
    )?;
    let direct = heap.huge_ids_direct(offset_size, length_size);
    let (expected_type, required_record_size) = if direct {
        (
            BTREE_V2_HUGE_OBJECT_DIRECT,
            HugeObjectDirectRecord::size(offset_width, length_width),
        )
    } else {
        (
            BTREE_V2_HUGE_OBJECT,
            HugeObjectRecord::size(offset_width, length_width),
        )
    };
    if header.tree_type != expected_type || header.record_size != required_record_size {
        return Err(FractalHeapStorageError::InvalidStorage);
    }

    let tree_extents = crate::btree_v2::collect_btree_v2_storage_extents(
        source,
        tree_address,
        offset_size,
        length_size,
    )?;
    let records = collect_btree_v2_records_from_source(source, &header, offset_size, length_size)?;
    if u64::try_from(records.len()).ok() != Some(heap.huge_objects_count)
        || header.total_records != heap.huge_objects_count
    {
        return Err(FractalHeapStorageError::InvalidStorage);
    }

    let mut payload_bytes = 0u64;
    let mut payload_extents = Vec::with_capacity(records.len());
    for record in records {
        let (address, length) = if direct {
            let record = record.huge_object_direct(offset_width, length_width)?;
            (record.address, record.length)
        } else {
            let record = record.huge_object(offset_width, length_width)?;
            (record.address, record.length)
        };
        if address.is_undefined(offset_size) {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        payload_bytes = payload_bytes
            .checked_add(length)
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        payload_extents.push(extent_in_source(source, address.get(), length)?);
    }
    if payload_bytes != heap.huge_objects_size {
        return Err(FractalHeapStorageError::InvalidStorage);
    }

    extents.extend(tree_extents);
    extents.extend(payload_extents);
    Ok(())
}

/// Returns a checked non-empty extent after proving its whole range lies in `source`.
#[allow(
    dead_code,
    reason = "crate-private ownership infrastructure is exercised by tests and crosschecks"
)]
fn extent_in_source<S: Source + ?Sized>(
    source: &S,
    address: u64,
    length: u64,
) -> Result<Extent, FractalHeapStorageError> {
    let extent = Extent::new(address, length).ok_or(FormatError::OffsetOverflow {
        offset: address,
        length,
    })?;
    if extent.end() > source.len() {
        return Err(FormatError::UnexpectedEof {
            expected: extent.end().to_usize().unwrap_or(usize::MAX),
            available: source.len().to_usize().unwrap_or(usize::MAX),
        }
        .into());
    }
    Ok(extent)
}

/// Reads objects from one fractal heap using parsed heap IDs.
///
/// The caller parses each [`FractalHeapIdView`] using the decoding parameters of the heap passed
/// to [`new`](Self::new). The reader caches the huge-object B-tree records after the first indexed
/// huge-object lookup. Inline huge objects and tiny objects are read directly from their parsed
/// fields.
pub struct HeapObjectReader<'h> {
    header: &'h FractalHeapHeader,
    offset_size: u8,
    length_size: u8,
    /// Parsed on demand, since most heaps hold no huge object at all.
    huge: Option<HugeObjectIndex>,
    /// Which backend the cached index was decoded from, once one has been.
    huge_backend: Option<Backend>,
}

/// Which of the two ways of reading a file an object came through. Recorded
/// only to hold a reader to one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// An in-memory image of the whole file.
    Buffered,
    /// A [`Source`], read from on demand.
    Streaming,
}

impl<'h> HeapObjectReader<'h> {
    /// Returns a reader of the objects of the heap `header` describes.
    ///
    /// `offset_size` and `length_size` are the superblock's "Size of Offsets" and "Size of
    /// Lengths" bytes.
    pub fn new(header: &'h FractalHeapHeader, offset_size: u8, length_size: u8) -> Self {
        HeapObjectReader {
            header,
            offset_size,
            length_size,
            huge: None,
            huge_backend: None,
        }
    }

    /// Reads the object identified by a parsed ID from a file image.
    ///
    /// Managed objects are read through the heap's doubling table. Indexed huge objects are
    /// looked up in the cached huge-object index. Tiny-object bytes are copied from `id`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] for a managed object in a filtered
    /// heap, [`FormatError::HugeObjectNotFound`] if an indexed huge object is absent, and
    /// [`FormatError::UnexpectedEof`] if a managed object has no allocated block or a required
    /// file region is truncated. Returns [`FormatError::ChunkedReadError`] if indirect-block
    /// traversal exceeds 64 levels, [`FormatError::ValueTooLargeForPlatform`] if a value does not
    /// fit the platform's address width, and [`FormatError::OffsetOverflow`] if a file offset
    /// overflows. Returns errors from parsing heap blocks and huge-object index metadata.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if indexed huge objects are read through both this method and
    /// [`read_from_source`](Self::read_from_source) on the same reader.
    pub fn read(
        &mut self,
        file_data: &[u8],
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        match id.kind() {
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } => self.read_managed_object(file_data, heap_offset, object_length),
            FractalHeapIdKind::Huge(reference) => self.read_huge(file_data, reference),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }

    /// Reads the object identified by a parsed ID through a [`Source`].
    ///
    /// # Errors
    ///
    /// Returns the resolution errors of [`read`](Self::read), and the error `source` returns if
    /// a read fails.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if indexed huge objects are read through both this method and
    /// [`read`](Self::read) on the same reader.
    pub fn read_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        match id.kind() {
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } => self.read_managed_object_from_source(source, heap_offset, object_length),
            FractalHeapIdKind::Huge(reference) => self.read_huge_from_source(source, reference),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }

    /// The `(address, length)` of huge object `huge_id`, parsing the heap's
    /// huge-object index out of `records` the first time one is asked for.
    ///
    /// What the index caches are file addresses, resolved against the image
    /// `backend` names. Every caller reads a whole heap through one backend, so
    /// a reader that saw both would be a caller that had changed which file it
    /// meant mid-walk — addresses applied to the wrong image read the wrong
    /// bytes and report no error, which is worth an assertion where the two meet.
    fn locate_huge<F>(
        &mut self,
        huge_id: u64,
        backend: Backend,
        records: F,
    ) -> Result<(StoredAddress, u64), FormatError>
    where
        F: FnOnce() -> Result<Vec<BTreeV2Record>, FormatError>,
    {
        if self.huge.is_none() {
            self.huge = Some(HugeObjectIndex::decode(
                &records()?,
                self.offset_size,
                self.length_size,
            )?);
            self.huge_backend = Some(backend);
        }
        debug_assert_eq!(
            self.huge_backend,
            Some(backend),
            "a heap's huge-object index holds addresses into the image it was decoded from",
        );
        self.huge
            .as_ref()
            .expect("filled immediately above")
            .locate(huge_id)
    }

    /// Reads a huge object at its inline location or through the cached index.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read`](Self::read) for a huge object.
    fn read_huge(
        &mut self,
        file_data: &[u8],
        reference: HugeObjectReference,
    ) -> Result<Vec<u8>, FormatError> {
        let (addr, len) = match reference {
            HugeObjectReference::Inline { addr, len } => (addr, len),
            HugeObjectReference::Indexed(huge_id) => {
                let (offset_size, length_size) = (self.offset_size, self.length_size);
                let btree_addr = self.header.btree_huge_objects_address.get().to_usize()?;
                self.locate_huge(huge_id, Backend::Buffered, || {
                    let header =
                        BTreeV2Header::parse(file_data, btree_addr, offset_size, length_size)?;
                    check_huge_object_btree(&header, offset_size, length_size)?;
                    collect_btree_v2_records(file_data, &header, offset_size, length_size)
                })?
            }
        };
        slice_object(file_data, addr.get(), len.to_usize()?)
    }

    /// Reads a huge object through a [`Source`] at its inline location or through the cached index.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read_from_source`](Self::read_from_source) for a huge object.
    fn read_huge_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        reference: HugeObjectReference,
    ) -> Result<Vec<u8>, FormatError> {
        let (addr, len) = match reference {
            HugeObjectReference::Inline { addr, len } => (addr, len),
            HugeObjectReference::Indexed(huge_id) => {
                let (offset_size, length_size) = (self.offset_size, self.length_size);
                let btree_addr = self.header.btree_huge_objects_address.get();
                self.locate_huge(huge_id, Backend::Streaming, || {
                    let header = BTreeV2Header::parse_from_source(
                        &SourceMetadata(source),
                        btree_addr,
                        offset_size,
                        length_size,
                    )?;
                    check_huge_object_btree(&header, offset_size, length_size)?;
                    collect_btree_v2_records_from_source(source, &header, offset_size, length_size)
                })?
            }
        };
        read_object_at_source(source, addr.get(), len.to_usize()?)
    }

    /// Reads a managed object through the heap's doubling table in `file_data`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read`](Self::read) for a managed object.
    fn read_managed_object(
        &self,
        file_data: &[u8],
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        // A filtered heap stores its direct blocks filter-encoded, and the reader has no decoder
        // for them, so it returns an error.
        if self.header.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }

        if is_undefined_addr(self.header.root_block_address.get(), self.offset_size) {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }

        if self.header.current_rows_in_root_indirect_block == 0 {
            // Root is a direct block
            self.read_from_direct_block(
                file_data,
                self.header.root_block_address.get().to_usize()?,
                self.header.starting_block_size,
                0, // block offset in heap = 0 for root
                heap_offset,
                object_length.to_usize()?,
            )
        } else {
            // The root is an indirect block, and the walk descends 64 levels at most.
            self.read_from_indirect_block(
                file_data,
                self.header.root_block_address.get().to_usize()?,
                self.header.current_rows_in_root_indirect_block,
                0, // block offset
                heap_offset,
                object_length.to_usize()?,
                64, // max recursion depth
            )
        }
    }

    /// Reads `length` bytes at heap offset `target_offset` from the direct block at `block_addr`,
    /// whose space begins at heap offset `block_heap_offset`.
    ///
    /// The prefix of the block is inside its heap space, so the object is at
    /// `block_addr + target_offset - block_heap_offset`.
    #[allow(clippy::too_many_arguments)]
    fn read_from_direct_block(
        &self,
        file_data: &[u8],
        block_addr: usize,
        _block_size: u64,
        block_heap_offset: u64,
        target_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let local_offset = (target_offset - block_heap_offset).to_usize()?;
        let pos = block_addr
            .checked_add(local_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: block_addr as u64,
                length: target_offset - block_heap_offset,
            })?;
        bytes::ensure_len(file_data, pos, length)?;
        Ok(file_data[pos..pos + length].to_vec())
    }

    /// Reads `length` bytes at heap offset `target_offset` from a direct block below the indirect
    /// block at `iblock_addr`, descending `depth_remaining` levels at most.
    #[allow(clippy::too_many_arguments)]
    fn read_from_indirect_block(
        &self,
        file_data: &[u8],
        iblock_addr: usize,
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
        length: usize,
        depth_remaining: u16,
    ) -> Result<Vec<u8>, FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::ChunkedReadError(
                "fractal heap: maximum recursion depth exceeded".into(),
            ));
        }
        bytes::ensure_len(file_data, iblock_addr, 4)?;
        let block = &file_data[iblock_addr..];
        match self.header.find_child_for_offset(
            block,
            nrows,
            iblock_heap_offset,
            target_offset,
            self.offset_size,
        )? {
            Some(FractalHeapChild::Direct {
                addr,
                block_size,
                heap_offset,
            }) => self.read_from_direct_block(
                file_data,
                addr.get().to_usize()?,
                block_size,
                heap_offset,
                target_offset,
                length,
            ),
            Some(FractalHeapChild::Indirect {
                addr,
                nrows: child_nrows,
                heap_offset,
            }) => self.read_from_indirect_block(
                file_data,
                addr.get().to_usize()?,
                child_nrows,
                heap_offset,
                target_offset,
                length,
                depth_remaining - 1,
            ),
            None => Err(FormatError::UnexpectedEof {
                expected: target_offset.to_usize()?.saturating_add(length),
                available: file_data.len(),
            }),
        }
    }

    /// Reads a managed object through the heap's doubling table in `source`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read_from_source`](Self::read_from_source) for a managed object.
    fn read_managed_object_from_source<S: Source + ?Sized>(
        &self,
        source: &S,
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        if self.header.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if is_undefined_addr(self.header.root_block_address.get(), self.offset_size) {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }
        if self.header.current_rows_in_root_indirect_block == 0 {
            self.read_from_direct_block_from_source(
                source,
                self.header.root_block_address.get(),
                0, // root direct block starts at heap offset 0
                heap_offset,
                object_length.to_usize()?,
            )
        } else {
            self.read_from_indirect_block_from_source(
                source,
                self.header.root_block_address.get(),
                self.header.current_rows_in_root_indirect_block,
                0,
                heap_offset,
                object_length.to_usize()?,
                64,
            )
        }
    }

    /// Reads `length` bytes at heap offset `target_offset` from the direct block at `block_addr`
    /// in `source`, whose space begins at heap offset `block_heap_offset`.
    fn read_from_direct_block_from_source<S: Source + ?Sized>(
        &self,
        source: &S,
        block_addr: u64,
        block_heap_offset: u64,
        target_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let local_offset = target_offset - block_heap_offset;
        let pos = block_addr
            .checked_add(local_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: block_addr,
                length: local_offset,
            })?;
        source.read_metadata_at(pos, length)
    }

    /// Reads `length` bytes at heap offset `target_offset` from a direct block below the indirect
    /// block at `iblock_addr` in `source`, descending `depth_remaining` levels at most.
    #[allow(clippy::too_many_arguments)]
    fn read_from_indirect_block_from_source<S: Source + ?Sized>(
        &self,
        source: &S,
        iblock_addr: u64,
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
        length: usize,
        depth_remaining: u16,
    ) -> Result<Vec<u8>, FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::ChunkedReadError(
                "fractal heap: maximum recursion depth exceeded".into(),
            ));
        }
        let region_len = self
            .header
            .indirect_block_entries_len(nrows, self.offset_size)?
            .min(source.len().saturating_sub(iblock_addr))
            .to_usize()?;
        let block = source.read_metadata_at(iblock_addr, region_len)?;
        match self.header.find_child_for_offset(
            &block,
            nrows,
            iblock_heap_offset,
            target_offset,
            self.offset_size,
        )? {
            Some(FractalHeapChild::Direct {
                addr, heap_offset, ..
            }) => self.read_from_direct_block_from_source(
                source,
                addr.get(),
                heap_offset,
                target_offset,
                length,
            ),
            Some(FractalHeapChild::Indirect {
                addr,
                nrows: child_nrows,
                heap_offset,
            }) => self.read_from_indirect_block_from_source(
                source,
                addr.get(),
                child_nrows,
                heap_offset,
                target_offset,
                length,
                depth_remaining - 1,
            ),
            None => Err(FormatError::UnexpectedEof {
                expected: target_offset.to_usize()?.saturating_add(length),
                available: source.len().to_usize().unwrap_or(usize::MAX),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::FractalHeapIdLayout;
    use hdf5_pure_format::__private::btree_v2_header_size;
    use test_util::btree_v2;

    use rstest::rstest;

    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use crate::source::BytesSource;

    use super::*;

    #[rstest]
    #[case::tiny(vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c', 0, 0, 0, 0])]
    #[case::inline_huge(vec![HUGE_ID_FIRST_BYTE, 0x20, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0])]
    fn a_parsed_inline_id_resolves_its_object(
        #[case] bytes: Vec<u8>,
        #[values(false, true)] streaming: bool,
    ) {
        let mut header = FractalHeapHeader::parse(
            &fractal_heap::Header::new(256).build(Widths::EIGHT),
            0,
            8,
            8,
        )
        .unwrap();
        header.heap_id_length = u16::try_from(bytes.len()).unwrap();
        let id = parse_id(&header, &bytes);
        let mut image = vec![0; 64];
        image[0x20..0x23].copy_from_slice(b"abc");
        let mut reader = HeapObjectReader::new(&header, 8, 8);
        let object = if streaming {
            reader.read_from_source(&BytesSource::new(image), id)
        } else {
            reader.read(&image, id)
        }
        .unwrap();
        assert_eq!(object, b"abc");
    }

    const STORAGE_HEAP_AT: u64 = 0x100;
    const STORAGE_ROOT_AT: u64 = 0x400;
    const STORAGE_WIDTH: u8 = 8;

    fn storage_header() -> FractalHeapHeader {
        FractalHeapHeader {
            heap_id_length: 7,
            io_filter_encoded_length: 0,
            flags: 0,
            max_managed_object_size: 64,
            next_huge_object_id: 0,
            btree_huge_objects_address: StoredAddress::new(u64::MAX),
            free_space_in_managed_blocks: 0,
            managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
            managed_space: 0,
            allocated_managed_space: 0,
            direct_block_allocation_iterator_offset: 0,
            managed_objects_count: 0,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: 4,
            starting_block_size: 128,
            max_direct_block_size: 1024,
            max_heap_size: 16,
            start_root_rows: 2,
            root_block_address: StoredAddress::new(u64::MAX),
            current_rows_in_root_indirect_block: 0,
        }
    }

    fn place(file: &mut [u8], at: u64, bytes: &[u8]) {
        let at = usize::try_from(at).unwrap();
        file[at..at + bytes.len()].copy_from_slice(bytes);
    }

    fn place_storage_header(file: &mut [u8], header: &FractalHeapHeader) {
        let bytes = header
            .serialize(OffsetWidth::Eight, LengthWidth::Eight)
            .unwrap();
        place(file, STORAGE_HEAP_AT, &bytes);
    }

    fn managed_prefix(signature: &[u8; 4], heap_offset: u64) -> Vec<u8> {
        let mut block = signature.to_vec();
        block.push(0);
        block.extend_from_slice(&STORAGE_HEAP_AT.to_le_bytes());
        block.extend_from_slice(&heap_offset.to_le_bytes()[..2]);
        block
    }

    fn direct_block(heap_offset: u64, size: usize) -> Vec<u8> {
        let mut block = managed_prefix(b"FHDB", heap_offset);
        block.resize(size, 0);
        block
    }

    fn indirect_block(
        header: &FractalHeapHeader,
        heap_offset: u64,
        nrows: u16,
        children: &[(usize, u64)],
    ) -> Vec<u8> {
        let size = header
            .storage_indirect_block_size(nrows, STORAGE_WIDTH)
            .unwrap();
        let mut block = managed_prefix(b"FHIB", heap_offset);
        let entries = usize::from(nrows) * usize::from(header.table_width);
        for entry in 0..entries {
            let address = children
                .iter()
                .find_map(|&(at, address)| (at == entry).then_some(address))
                .unwrap_or(u64::MAX);
            block.extend_from_slice(&address.to_le_bytes());
        }
        block.extend_from_slice(&[0; 4]);
        assert_eq!(u64::try_from(block.len()).unwrap(), size);
        let len = block.len();
        crate::checksum::stamp_trailing(&mut block, 0, len);
        block
    }

    fn extent(start: u64, len: u64) -> Extent {
        Extent::new(start, len).unwrap()
    }

    fn collect_storage(file: &[u8]) -> Result<FractalHeapStorage, FractalHeapStorageError> {
        collect_fractal_heap_storage(
            file,
            StoredAddress::new(STORAGE_HEAP_AT),
            STORAGE_WIDTH,
            STORAGE_WIDTH,
        )
    }

    #[test]
    fn an_empty_heap_owns_only_its_exact_header() {
        let header = storage_header();
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        let header_len = u64::try_from(FractalHeapHeader::serialized_size(
            OffsetWidth::Eight,
            LengthWidth::Eight,
        ))
        .unwrap();

        let storage = collect_storage(&file).unwrap();

        assert_eq!(storage.extents(), &[extent(STORAGE_HEAP_AT, header_len)]);
    }

    #[test]
    fn tiny_objects_add_no_file_allocation() {
        let header = FractalHeapHeader {
            tiny_objects_count: 3,
            tiny_objects_size: 11,
            ..storage_header()
        };
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        let header_len = u64::try_from(FractalHeapHeader::serialized_size(
            OffsetWidth::Eight,
            LengthWidth::Eight,
        ))
        .unwrap();

        let storage = collect_storage(&file).unwrap();

        assert_eq!(storage.extents(), &[extent(STORAGE_HEAP_AT, header_len)]);
    }

    #[test]
    fn a_direct_root_is_one_full_block_for_all_managed_and_tiny_objects() {
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            managed_space: 128,
            allocated_managed_space: 128,
            managed_objects_count: 3,
            tiny_objects_count: 2,
            tiny_objects_size: 9,
            ..storage_header()
        };
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        place(
            &mut file,
            STORAGE_ROOT_AT,
            &direct_block(0, usize::try_from(header.starting_block_size).unwrap()),
        );

        let buffered = collect_storage(&file).unwrap();
        let streamed = collect_fractal_heap_storage_from_source(
            &BytesSource::new(&file),
            StoredAddress::new(STORAGE_HEAP_AT),
            STORAGE_WIDTH,
            STORAGE_WIDTH,
        )
        .unwrap();
        let header_len = u64::try_from(FractalHeapHeader::serialized_size(
            OffsetWidth::Eight,
            LengthWidth::Eight,
        ))
        .unwrap();

        assert_eq!(buffered, streamed);
        assert_eq!(
            buffered.extents(),
            &[
                extent(STORAGE_HEAP_AT, header_len),
                extent(STORAGE_ROOT_AT, 128),
            ]
        );
    }

    #[test]
    fn an_indirect_root_reports_every_direct_and_nested_indirect_block_once() {
        const DIRECT_A: u64 = 0x800;
        const CHILD_INDIRECT: u64 = 0x1000;
        const DIRECT_B: u64 = 0x1400;
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            current_rows_in_root_indirect_block: 6,
            managed_space: 16 * 1024,
            allocated_managed_space: 256,
            managed_objects_count: 2,
            ..storage_header()
        };
        let child_heap_offset = header.storage_row_offset(5).unwrap();
        assert_eq!(child_heap_offset, 8192);
        let child_rows = header.storage_child_indirect_rows(5).unwrap();
        assert_eq!(child_rows, 3);

        let mut file = vec![0; 0x1800];
        place_storage_header(&mut file, &header);
        place(
            &mut file,
            STORAGE_ROOT_AT,
            &indirect_block(&header, 0, 6, &[(0, DIRECT_A), (20, CHILD_INDIRECT)]),
        );
        place(&mut file, DIRECT_A, &direct_block(0, 128));
        place(
            &mut file,
            CHILD_INDIRECT,
            &indirect_block(&header, child_heap_offset, child_rows, &[(0, DIRECT_B)]),
        );
        place(&mut file, DIRECT_B, &direct_block(child_heap_offset, 128));

        let storage = collect_storage(&file).unwrap();
        let mut expected = vec![
            extent(
                STORAGE_HEAP_AT,
                u64::try_from(FractalHeapHeader::serialized_size(
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                ))
                .unwrap(),
            ),
            extent(
                STORAGE_ROOT_AT,
                header
                    .storage_indirect_block_size(6, STORAGE_WIDTH)
                    .unwrap(),
            ),
            extent(DIRECT_A, 128),
            extent(
                CHILD_INDIRECT,
                header
                    .storage_indirect_block_size(child_rows, STORAGE_WIDTH)
                    .unwrap(),
            ),
            extent(DIRECT_B, 128),
        ];
        expected.sort_unstable();

        assert_eq!(storage.extents(), expected);
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn an_indirect_block_with_a_bad_checksum_is_rejected() {
        const DIRECT: u64 = 0x800;
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            current_rows_in_root_indirect_block: 2,
            managed_space: 1024,
            allocated_managed_space: 128,
            managed_objects_count: 1,
            ..storage_header()
        };
        let mut block = indirect_block(&header, 0, 2, &[(0, DIRECT)]);
        let checksum = block.len() - 4;
        let stored = u32::from_le_bytes(block[checksum..].try_into().unwrap());
        block[checksum] ^= 1;
        let mut file = vec![0; 0xc00];
        place_storage_header(&mut file, &header);
        place(&mut file, STORAGE_ROOT_AT, &block);
        place(&mut file, DIRECT, &direct_block(0, 128));

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::ChecksumMismatch {
                    expected: stored ^ 1,
                    computed: stored,
                }
            ))
        );
    }

    #[test]
    fn duplicate_managed_child_addresses_are_rejected() {
        const CHILD: u64 = 0x800;
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            current_rows_in_root_indirect_block: 2,
            managed_space: 1024,
            allocated_managed_space: 256,
            ..storage_header()
        };
        let mut file = vec![0; 0xc00];
        place_storage_header(&mut file, &header);
        place(
            &mut file,
            STORAGE_ROOT_AT,
            &indirect_block(&header, 0, 2, &[(0, CHILD), (1, CHILD)]),
        );
        place(&mut file, CHILD, &direct_block(128, 128));

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::InvalidStorage)
        );
    }

    #[test]
    fn a_truncated_direct_block_is_rejected_as_a_truncated_allocation() {
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(0x780),
            managed_space: 128,
            allocated_managed_space: 128,
            ..storage_header()
        };
        let mut file = vec![0; 0x7c0];
        place_storage_header(&mut file, &header);
        place(&mut file, 0x780, &direct_block(0, 64));

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::UnexpectedEof {
                    expected: 0x800,
                    available: 0x7c0,
                }
            ))
        );
    }

    #[test]
    fn a_truncated_indirect_block_is_rejected_as_a_truncated_allocation() {
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(0x780),
            current_rows_in_root_indirect_block: 2,
            managed_space: 1024,
            ..storage_header()
        };
        let block_len = header
            .storage_indirect_block_size(2, STORAGE_WIDTH)
            .unwrap();
        let mut file = vec![0; 0x7a0];
        place_storage_header(&mut file, &header);
        place(&mut file, 0x780, &managed_prefix(b"FHIB", 0));

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::UnexpectedEof {
                    expected: usize::try_from(0x780 + block_len).unwrap(),
                    available: 0x7a0,
                }
            ))
        );
    }

    #[test]
    fn a_filtered_heap_is_an_explicitly_unsupported_owner() {
        let header = storage_header();
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        let at = usize::try_from(STORAGE_HEAP_AT).unwrap();
        file[at + 7..at + 9].copy_from_slice(&1u16.to_le_bytes());

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::UnsupportedOwnership(
                "filtered managed blocks"
            ))
        );
    }

    #[test]
    fn a_managed_space_free_space_manager_is_an_explicitly_unsupported_owner() {
        let header = FractalHeapHeader {
            managed_block_free_space_manager_address: StoredAddress::new(0x600),
            ..storage_header()
        };
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::UnsupportedOwnership(
                "managed-space free-space manager metadata"
            ))
        );
    }

    fn encoded_huge_record(address: u64, length: u64, id: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        HugeObjectRecord {
            address: StoredAddress::new(address),
            length,
            id,
        }
        .encode(&mut bytes, OffsetWidth::Eight, LengthWidth::Eight);
        bytes
    }

    fn place_btree_node(file: &mut [u8], at: u64, node: &[u8], node_size: usize) {
        place(file, at, node);
        let at = usize::try_from(at).unwrap();
        file[at + node.len()..at + node_size].fill(0);
    }

    #[test]
    fn a_huge_object_index_contributes_all_nodes_and_payloads() {
        const TREE: u64 = 0x400;
        const ROOT: u64 = 0x500;
        const LEFT: u64 = 0x600;
        const RIGHT: u64 = 0x700;
        const NODE_SIZE: usize = 128;
        const PAYLOAD_A: u64 = 0x900;
        const PAYLOAD_B: u64 = 0xa00;
        const PAYLOAD_C: u64 = 0xb00;
        let records = [
            encoded_huge_record(PAYLOAD_A, 31, 1),
            encoded_huge_record(PAYLOAD_B, 37, 2),
            encoded_huge_record(PAYLOAD_C, 41, 3),
        ];
        let header = FractalHeapHeader {
            btree_huge_objects_address: StoredAddress::new(TREE),
            huge_objects_size: 31 + 37 + 41,
            huge_objects_count: 3,
            ..storage_header()
        };
        let tree_header = btree_v2::Header::new(BTREE_V2_HUGE_OBJECT, 24, ROOT, 1)
            .node_size(u32::try_from(NODE_SIZE).unwrap())
            .depth(1)
            .total_records(3)
            .build(Widths::EIGHT);
        let root = btree_v2::internal(
            BTREE_V2_HUGE_OBJECT,
            &[records[1].clone()],
            &[
                btree_v2::Child {
                    address: LEFT,
                    records: 1,
                    records_width: 1,
                    subtree: None,
                },
                btree_v2::Child {
                    address: RIGHT,
                    records: 1,
                    records_width: 1,
                    subtree: None,
                },
            ],
            Widths::EIGHT,
        );
        let left = btree_v2::leaf(BTREE_V2_HUGE_OBJECT, &[records[0].clone()]);
        let right = btree_v2::leaf(BTREE_V2_HUGE_OBJECT, &[records[2].clone()]);
        let mut file = vec![0; 0xc00];
        place_storage_header(&mut file, &header);
        place(&mut file, TREE, &tree_header);
        place_btree_node(&mut file, ROOT, &root, NODE_SIZE);
        place_btree_node(&mut file, LEFT, &left, NODE_SIZE);
        place_btree_node(&mut file, RIGHT, &right, NODE_SIZE);
        file[usize::try_from(PAYLOAD_A).unwrap()..usize::try_from(PAYLOAD_A + 31).unwrap()].fill(1);
        file[usize::try_from(PAYLOAD_B).unwrap()..usize::try_from(PAYLOAD_B + 37).unwrap()].fill(2);
        file[usize::try_from(PAYLOAD_C).unwrap()..usize::try_from(PAYLOAD_C + 41).unwrap()].fill(3);

        let storage = collect_storage(&file).unwrap();
        let tree_header_len =
            u64::try_from(btree_v2_header_size(OffsetWidth::Eight, LengthWidth::Eight)).unwrap();
        let mut expected = vec![
            extent(
                STORAGE_HEAP_AT,
                u64::try_from(FractalHeapHeader::serialized_size(
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                ))
                .unwrap(),
            ),
            extent(TREE, tree_header_len),
            extent(ROOT, u64::try_from(NODE_SIZE).unwrap()),
            extent(LEFT, u64::try_from(NODE_SIZE).unwrap()),
            extent(RIGHT, u64::try_from(NODE_SIZE).unwrap()),
            extent(PAYLOAD_A, 31),
            extent(PAYLOAD_B, 37),
            extent(PAYLOAD_C, 41),
        ];
        expected.sort_unstable();

        assert_eq!(storage.extents(), expected);
    }

    #[test]
    fn a_direct_huge_object_tree_still_contributes_its_payload() {
        const TREE: u64 = 0x400;
        const LEAF: u64 = 0x500;
        const PAYLOAD: u64 = 0x700;
        const NODE_SIZE: usize = 128;
        let mut record = Vec::new();
        record.extend_from_slice(&PAYLOAD.to_le_bytes());
        record.extend_from_slice(&29u64.to_le_bytes());
        let header = FractalHeapHeader {
            heap_id_length: 17,
            btree_huge_objects_address: StoredAddress::new(TREE),
            huge_objects_size: 29,
            huge_objects_count: 1,
            ..storage_header()
        };
        let tree_header = btree_v2::Header::new(BTREE_V2_HUGE_OBJECT_DIRECT, 16, LEAF, 1)
            .node_size(u32::try_from(NODE_SIZE).unwrap())
            .build(Widths::EIGHT);
        let leaf = btree_v2::leaf(BTREE_V2_HUGE_OBJECT_DIRECT, &[record]);
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        place(&mut file, TREE, &tree_header);
        place_btree_node(&mut file, LEAF, &leaf, NODE_SIZE);
        file[usize::try_from(PAYLOAD).unwrap()..usize::try_from(PAYLOAD + 29).unwrap()].fill(7);

        let storage = collect_storage(&file).unwrap();

        assert!(storage.extents().contains(&extent(PAYLOAD, 29)));
        assert!(
            storage
                .extents()
                .contains(&extent(LEAF, u64::try_from(NODE_SIZE).unwrap()))
        );
    }

    #[test]
    fn overlapping_owned_allocations_are_rejected() {
        const TREE: u64 = 0x400;
        const LEAF: u64 = 0x500;
        const NODE_SIZE: usize = 128;
        let mut record = Vec::new();
        record.extend_from_slice(&LEAF.to_le_bytes());
        record.extend_from_slice(&32u64.to_le_bytes());
        let header = FractalHeapHeader {
            heap_id_length: 17,
            btree_huge_objects_address: StoredAddress::new(TREE),
            huge_objects_size: 32,
            huge_objects_count: 1,
            ..storage_header()
        };
        let tree_header = btree_v2::Header::new(BTREE_V2_HUGE_OBJECT_DIRECT, 16, LEAF, 1)
            .node_size(u32::try_from(NODE_SIZE).unwrap())
            .build(Widths::EIGHT);
        let leaf = btree_v2::leaf(BTREE_V2_HUGE_OBJECT_DIRECT, &[record]);
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        place(&mut file, TREE, &tree_header);
        place_btree_node(&mut file, LEAF, &leaf, NODE_SIZE);

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::InvalidStorage)
        );
    }

    #[test]
    fn read_managed_object_from_direct_block() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

        // Build heap ID for the test data written in build_simple_heap.
        // The test data "Hello, World!" is at the data area of the direct block.
        // The direct block header is 5 + 8 + 2 = 15 bytes (for max_heap_size=16, ceil(16/8)=2).
        // Wait, max_heap_size=16, ceil(16/8)=2. Header = sig(4)+ver(1)+addr(8)+bo(2) = 15.
        // The data was placed at data_start = block_addr + 15.
        // Since offset is from block start, the object is at offset 15 within the block.
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8); // 15
        let offset: u64 = dblock_header_size as u64;
        let length: u64 = 13;
        let payload = offset | (length << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        id[0] = 0x00;
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let obj = HeapObjectReader::new(&hdr, 8, 8)
            .read(&file_data, parse_id(&hdr, &id))
            .unwrap();
        assert_eq!(&obj, b"Hello, World!");
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_managed_object_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);

        // Same heap ID as `read_managed_object_from_direct_block`.
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8);
        let payload = (dblock_header_size as u64) | (13u64 << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let buffered = HeapObjectReader::new(&hdr, 8, 8)
            .read(&file_data, parse_id(&hdr, &id))
            .unwrap();

        // Header parsed from a source, then the object fetched from a source.
        let mem = BytesSource::new(&file_data);
        let hdr_mem = FractalHeapHeader::parse_from_source(&SourceMetadata(&mem), 0, 8, 8).unwrap();
        let from_mem = HeapObjectReader::new(&hdr_mem, 8, 8)
            .read_from_source(&mem, parse_id(&hdr_mem, &id))
            .unwrap();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
        let hdr_seek =
            FractalHeapHeader::parse_from_source(&SourceMetadata(&seek), 0, 8, 8).unwrap();
        let from_seek = HeapObjectReader::new(&hdr_seek, 8, 8)
            .read_from_source(&seek, parse_id(&hdr_seek, &id))
            .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(&from_seek, b"Hello, World!");
    }

    #[test]
    fn filtered_managed_heap_is_refused_not_misparsed() {
        // A filtered managed heap stores its direct-block contents filter-encoded
        // and inserts per-child `filtered_size`/`filter_mask` fields in indirect
        // blocks. We do not decode either, so every managed read path must refuse
        // cleanly (UnsupportedFilteredHeapObject) rather than return raw bytes or
        // mis-stride the child-pointer walk.
        let managed_id = [0x00u8, 0, 0, 0, 0, 0, 0]; // type bits 0 -> Managed

        // Root is a direct block.
        let mut h = FractalHeapHeader {
            heap_id_length: 7,
            io_filter_encoded_length: 8,
            flags: 0,
            max_managed_object_size: 0,
            next_huge_object_id: 0,
            btree_huge_objects_address: StoredAddress::new(u64::MAX),
            free_space_in_managed_blocks: 0,
            managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
            managed_space: 0,
            allocated_managed_space: 0,
            direct_block_allocation_iterator_offset: 0,
            managed_objects_count: 0,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: 4,
            starting_block_size: 512,
            max_direct_block_size: 65536,
            max_heap_size: 64,
            start_root_rows: 1,
            root_block_address: StoredAddress::new(0x100),
            current_rows_in_root_indirect_block: 0,
        };
        let file = vec![0u8; 0x400];
        assert_eq!(
            HeapObjectReader::new(&h, 8, 8).read(&file, parse_id(&h, &managed_id)),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );

        // Root is an indirect block: the refusal happens before any child walk.
        h.current_rows_in_root_indirect_block = 2;
        assert_eq!(
            HeapObjectReader::new(&h, 8, 8).read(&file, parse_id(&h, &managed_id)),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );
    }

    /// Type-1 record: address(8) + length(8) + id(8), little-endian.
    fn huge_record(addr: u64, len: u64, id: u64) -> BTreeV2Record {
        let mut d = Vec::new();
        d.extend_from_slice(&addr.to_le_bytes());
        d.extend_from_slice(&len.to_le_bytes());
        d.extend_from_slice(&id.to_le_bytes());
        BTreeV2Record { data: d }
    }

    #[test]
    fn huge_object_index_locates_by_id() {
        let records = vec![
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
            huge_record(0x3000, 7000, 5),
        ];
        let index = HugeObjectIndex::decode(&records, 8, 8).unwrap();
        assert_eq!(index.locate(2).unwrap(), (StoredAddress::new(0x2000), 6000));
        assert_eq!(index.locate(5).unwrap(), (StoredAddress::new(0x3000), 7000));
        assert_eq!(
            index.locate(9),
            Err(FormatError::HugeObjectNotFound(9)),
            "an id no record carries is not found rather than mismatched"
        );
    }

    /// The lookup is a binary search, so the index has to impose the ordering
    /// itself rather than inherit whatever order the records arrived in.
    #[test]
    fn huge_object_index_orders_records_it_receives_out_of_order() {
        let records = vec![
            huge_record(0x3000, 7000, 5),
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
        ];
        let index = HugeObjectIndex::decode(&records, 8, 8).unwrap();
        for (id, want) in [
            (1, (StoredAddress::new(0x1000), 5000)),
            (2, (StoredAddress::new(0x2000), 6000)),
            (5, (StoredAddress::new(0x3000), 7000)),
        ] {
            assert_eq!(index.locate(id).unwrap(), want);
        }
    }

    /// A huge-objects B-tree that is not the type-1 layout is refused before its
    /// records are read as that layout.
    ///
    /// Type 2 is the filtered counterpart, whose records are longer and put the
    /// object ID somewhere else: decoding one as type 1 reads an ID out of the
    /// filter mask and returns some other object, or none, without complaint.
    #[test]
    fn a_huge_object_btree_of_another_type_is_refused() {
        let btree = |tree_type: u8, record_size: u16| BTreeV2Header {
            tree_type,
            node_size: 512,
            record_size,
            depth: 0,
            root_node_address: StoredAddress::new(0x100),
            num_records_in_root: 1,
            total_records: 1,
        };

        assert_eq!(
            check_huge_object_btree(&btree(2, 36), 8, 8),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 2,
                record_size: 36,
                required: 24,
            }),
            "a filtered huge-object tree is long enough to pass a length check alone"
        );
        assert_eq!(
            check_huge_object_btree(&btree(1, 16), 8, 8),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 1,
                record_size: 16,
                required: 24,
            }),
            "records too short for the layout the type claims"
        );
        assert_eq!(check_huge_object_btree(&btree(1, 24), 8, 8), Ok(()));
    }

    /// A record too short for the type-1 layout is a heap this reader has
    /// misread, not one object to skip past. The backstop behind
    /// [`check_huge_object_btree`], which refuses such a tree by its declared
    /// record size before any record reaches the decode.
    #[test]
    fn huge_object_index_refuses_a_truncated_record() {
        let records = vec![BTreeV2Record {
            data: vec![0u8; 20],
        }];
        assert!(matches!(
            HugeObjectIndex::decode(&records, 8, 8),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    /// One reader parses the heap's huge-object index once, however many huge
    /// objects are read through it, on both backends.
    ///
    /// Parsing it per object returns exactly the same bytes while making a walk
    /// over the heap quadratic in the number of huge objects, so only the parse
    /// count separates the two.
    #[test]
    fn a_reader_parses_the_huge_object_index_once() {
        use crate::source::BytesSource;
        const COUNT: u64 = 5;

        // Attributes too large for a managed heap object, so the heap holds
        // nothing but huge ones.
        let mut builder = crate::FileBuilder::new();
        for i in 0..COUNT {
            builder.set_attr(
                &format!("a{i}"),
                crate::AttrValue::StringArray(vec![format!("{i:0700}"); 100]),
            );
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = builder.finish().unwrap();

        let frhp = bytes
            .windows(4)
            .position(|w| w == b"FRHP")
            .expect("dense storage writes a fractal heap");
        let heap = FractalHeapHeader::parse(&bytes, frhp, 8, 8).unwrap();

        // A huge heap ID: type 1 in bits 4-5 of byte 0, then the object's id,
        // little-endian, across the rest of the ID. The writer's own encoding,
        // which lets this reach the objects without walking the name index.
        let heap_id = |id: u64| {
            let mut bytes = vec![0u8; heap.heap_id_length as usize];
            bytes[0] = 0x10;
            for (i, slot) in bytes[1..].iter_mut().take(8).enumerate() {
                *slot = ((id >> (i * 8)) & 0xFF) as u8;
            }
            bytes
        };
        let ids: Vec<Vec<u8>> = (1..=COUNT).map(heap_id).collect();

        reset_huge_index_decodes();
        let mut reader = HeapObjectReader::new(&heap, 8, 8);
        let buffered: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| reader.read(&bytes, parse_id(&heap, id)).unwrap())
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "buffered reads re-parsed the index"
        );

        reset_huge_index_decodes();
        let source = BytesSource::new(bytes.clone());
        let mut reader = HeapObjectReader::new(&heap, 8, 8);
        let streamed: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| {
                reader
                    .read_from_source(&source, parse_id(&heap, id))
                    .unwrap()
            })
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "streaming reads re-parsed the index"
        );

        assert_eq!(buffered, streamed);
        assert_eq!(buffered.len(), COUNT as usize);
        for (i, object) in buffered.iter().enumerate() {
            assert!(
                object.windows(2).any(|w| w == format!("a{i}").as_bytes()),
                "object {i} is not the attribute message it should be"
            );
        }
    }

    fn parse_id<'a>(header: &FractalHeapHeader, bytes: &'a [u8]) -> FractalHeapIdView<'a> {
        FractalHeapIdLayout::new(header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(bytes)
            .unwrap()
    }

    // A version 0 huge ID has type 1 in bits 4 and 5 ("Fractal Heap", specification 4.0).
    const HUGE_ID_FIRST_BYTE: u8 = 0x10;
    // A version 0 tiny ID has type 2 in bits 4 and 5 ("Fractal Heap", specification 4.0).
    const TINY_ID_FIRST_BYTE: u8 = 0x20;
}
