//! Proves the complete set of file allocations owned by supported fractal heaps.
//!
//! The ownership walk is read-only and operates through [`Source`]. It uses the validated
//! [`FractalHeapLayout`] from `hdf5-pure-format` for managed-block geometry while keeping I/O and
//! ownership policy in this crate.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT_DIRECT;
use hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapLayout;
use hdf5_pure_format::__private::FractalHeapLayoutError;
use hdf5_pure_format::__private::HugeObjectDirectRecord;
use hdf5_pure_format::__private::HugeObjectRecord;

use hdf5_pure_space::__private::Extent;

use crate::address::StoredAddress;
use crate::btree_v2::{BTreeV2Header, collect_btree_v2_records_from_source};
use crate::bytes;
use crate::convert::{Narrow, is_undefined_addr};
use crate::error::FormatError;
#[cfg(test)]
use crate::source::BytesSource;
use crate::source::{Source, SourceMetadata};
use crate::width::{LengthWidth, OffsetWidth};

/// Describes a failure to prove complete ownership of a fractal heap's file storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FractalHeapStorageError {
    /// A lower-level HDF5 parser or source read failed.
    Format(FormatError),
    /// The heap header does not define valid doubling-table geometry.
    Layout(FractalHeapLayoutError),
    /// The heap's ownership graph or structural metadata is inconsistent.
    InvalidStorage,
    /// The named heap feature prevents a complete ownership proof.
    UnsupportedOwnership(&'static str),
}

impl From<FormatError> for FractalHeapStorageError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<FractalHeapLayoutError> for FractalHeapStorageError {
    fn from(error: FractalHeapLayoutError) -> Self {
        Self::Layout(error)
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
/// managed-space free-space manager. Malformed doubling-table geometry returns
/// [`FractalHeapStorageError::Layout`]. Duplicate managed-block addresses, inconsistent block
/// prefixes, contradictory empty-heap bookkeeping, an unexpected huge-object index layout,
/// invalid huge-object payload addresses, huge-object accounting mismatches, and overlapping owned
/// allocations return [`FractalHeapStorageError::InvalidStorage`]. Truncated allocations and
/// lower-level parser errors are wrapped in [`FractalHeapStorageError::Format`].
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
    let layout = FractalHeapLayout::new(&header, offset_width, length_width)?;
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
        &layout,
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
    layout: &'a FractalHeapLayout,
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
        layout: &'a FractalHeapLayout,
        heap_header_address: StoredAddress,
        offset_size: u8,
        extents: &'e mut Vec<Extent>,
    ) -> Self {
        Self {
            source,
            header,
            layout,
            heap_header_address,
            offset_size,
            visited: BTreeSet::new(),
            extents,
            allocated_direct_space: 0,
        }
    }

    fn collect(&mut self) -> Result<(), FractalHeapStorageError> {
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
            let block_size = self.layout.block_size_for_row(0)?;
            if self.header.managed_space != block_size {
                return Err(FractalHeapStorageError::InvalidStorage);
            }
            ManagedBlock::Direct {
                address: root,
                block_size,
                heap_offset: 0,
            }
        } else {
            if self.header.current_rows_in_root_indirect_block > self.layout.max_root_rows()
                || self.header.managed_space
                    != self
                        .layout
                        .indirect_heap_size(self.header.current_rows_in_root_indirect_block)?
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
        let block_offset_size = self.layout.block_offset_size();
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
        let block_size = self.layout.indirect_block_size(nrows)?;
        let extent = extent_in_source(self.source, address.get(), block_size)?;
        let block = self
            .source
            .read_metadata_at(address.get(), block_size.to_usize()?)?;
        let block_offset_size = self.layout.block_offset_size();
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

        let width = self.layout.table_width();
        let direct_rows = usize::from(nrows).min(self.layout.direct_rows());
        let entries_at = 5usize
            .checked_add(usize::from(self.offset_size))
            .and_then(|n| n.checked_add(block_offset_size))
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        let heap_limit =
            (self.header.max_heap_size < 64).then(|| 1u64 << u32::from(self.header.max_heap_size));

        for row in 0..usize::from(nrows) {
            let slot_size = self.layout.block_size_for_row(row)?;
            let child_rows = if row < direct_rows {
                None
            } else {
                let child_rows = self.layout.child_indirect_rows(row)?;
                if child_rows >= nrows || self.layout.indirect_heap_size(child_rows)? != slot_size {
                    return Err(FractalHeapStorageError::InvalidStorage);
                }
                Some(child_rows)
            };
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
                    .checked_add(self.layout.slot_offset(row, col)?)
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

/// Reads objects out of one fractal heap, holding the indexes that resolving
/// them needs.
///
/// A managed or tiny object is resolved against the heap header alone. Huge objects use the heap's
/// huge-objects B-tree. Every caller here walks a whole heap, so this type parses that index on the
/// first huge object, reuses it for the remaining huge objects, and leaves it absent when the heap
/// has none.
#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::btree_v2_header_size;
    use test_util::btree_v2;
    use test_util::widths::Widths;

    use super::*;

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

    fn storage_layout(header: &FractalHeapHeader) -> FractalHeapLayout {
        FractalHeapLayout::new(header, OffsetWidth::Eight, LengthWidth::Eight).unwrap()
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
        let size = storage_layout(header).indirect_block_size(nrows).unwrap();
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
        let child_heap_offset = storage_layout(&header).row_offset(5).unwrap();
        assert_eq!(child_heap_offset, 8192);
        let child_rows = storage_layout(&header).child_indirect_rows(5).unwrap();
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
                storage_layout(&header).indirect_block_size(6).unwrap(),
            ),
            extent(DIRECT_A, 128),
            extent(
                CHILD_INDIRECT,
                storage_layout(&header)
                    .indirect_block_size(child_rows)
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
        let block_len = storage_layout(&header).indirect_block_size(2).unwrap();
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
}
