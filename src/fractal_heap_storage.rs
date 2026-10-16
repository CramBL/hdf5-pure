//! Proves the complete set of file allocations owned by supported fractal heaps.
//!
//! The ownership walk is read-only and operates through [`Source`]. It uses the validated
//! [`FractalHeapBlockLayout`] for managed-block references. Checksummed direct blocks require
//! complete checksum validation before the walker records their ownership.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT_DIRECT;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapBlock;
use hdf5_pure_format::__private::FractalHeapBlockFilters;
use hdf5_pure_format::__private::FractalHeapBlockLayout;
use hdf5_pure_format::__private::FractalHeapDirectBlock;
use hdf5_pure_format::__private::FractalHeapDirectBlockRef;
use hdf5_pure_format::__private::FractalHeapHeader;
#[cfg(any(test, doc))]
use hdf5_pure_format::__private::FractalHeapLayout;
use hdf5_pure_format::__private::FractalHeapLayoutError;
use hdf5_pure_format::__private::FreeSpaceManagerHeaderRef;
use hdf5_pure_format::__private::HugeObjectDirectRecord;
use hdf5_pure_format::__private::HugeObjectRecord;

use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::Extent;

use crate::address::AddressContext;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
#[cfg(test)]
use crate::address::{BaseAddress, BaseAddressExt};
use crate::btree_v2::BTreeV2Header;
use crate::convert::Narrow;
use crate::error::FormatError;
#[cfg(test)]
use crate::source::BytesSource;
use crate::source::{Source, SourceMetadata};
#[cfg(test)]
use crate::width::{LengthWidth, OffsetWidth};

#[cfg(test)]
const FORMAT_WIDTHS: FormatWidths = FormatWidths {
    offsets: OffsetWidth::Eight,
    lengths: LengthWidth::Eight,
};

/// Describes a failure to prove complete ownership of a fractal heap's file storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FractalHeapStorageError {
    /// A filtered heap has a huge-object index, whose ownership the walker does not prove.
    FilteredHugeObjects,
    /// A lower-level HDF5 parser or source read failed.
    Format(FormatError),
    /// The heap's ownership graph or structural metadata is inconsistent.
    InvalidStorage,
    /// The heap header does not define valid doubling-table geometry.
    Layout(FractalHeapLayoutError),
    /// A direct block's filter pipeline has an active filter, identified by its filter ID, that
    /// this crate does not decode.
    UnsupportedFilter(u16),
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
/// When a managed-space manager is present, the set contains its header and any allocated section
/// list, including reserved capacity. Logical heap sections are excluded from the allocation set.
/// Tiny objects are stored entirely in heap IDs, so the set omits them.
///
/// The walker returns this type only after it has accounted for every supported allocation and
/// proven the allocations pairwise disjoint. Unsupported heap features prevent construction.
/// The heap structures are defined in "Fractal Heap" of the [format specification, version
/// 4.0][heap]. The manager structures are defined in "Free-space Index" of the same
/// [specification][manager].
///
/// [heap]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
/// [manager]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FractalHeapStorage {
    allocations: Vec<(Extent, FractalHeapAllocation)>,
}

impl FractalHeapStorage {
    /// Returns the proven owned allocations, each with its kind, in file-address order.
    pub(crate) fn allocations(&self) -> &[(Extent, FractalHeapAllocation)] {
        &self.allocations
    }
}

/// The structure a fractal heap allocation holds.
///
/// The heap structures are defined in "Fractal Heap" of the [format specification, version
/// 4.0][heap]. The manager structures are defined in "Free-space Index" of the same
/// [specification][manager].
///
/// [heap]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
/// [manager]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FractalHeapAllocation {
    /// A managed direct block, `FHDB`.
    DirectBlock,
    /// The `FRHP` heap header.
    Header,
    /// The separately allocated payload of a huge object.
    HugeObject,
    /// The header or a node of the version 2 B-tree that indexes the huge objects.
    HugeObjectIndex,
    /// A managed indirect block, `FHIB`.
    IndirectBlock,
    /// The `FSHD` header of the managed-space free-space manager.
    ManagedFsmHeader,
    /// The complete `FSSE` allocation, including reserved capacity.
    ManagedFsmSectionList,
}

impl FractalHeapAllocation {
    /// Returns the allocation class for a separately allocated heap structure.
    ///
    /// PAGE reclamation also considers the surrounding allocations when a huge payload occupies
    /// a metadata page.
    // libhdf5 2.2.0 allocates the header and indirect blocks as `H5FD_MEM_OHDR`, direct blocks
    // as `H5FD_MEM_LHEAP` and huge objects as `H5FD_MEM_DRAW` (the `H5FD_MEM_FHEAP_*` aliases in
    // `H5FDdevelop.h`), and the huge-object B-tree as `H5FD_MEM_BTREE` (`H5B2hdr.c`,
    // `H5B2internal.c`, `H5B2leaf.c`). The sec2 driver's `H5FD_FLMAP_DICHOTOMY` map
    // (`H5FDsec2.c`) puts an `H5FD_MEM_DRAW` allocation smaller than a page in raw-data pages and
    // the other types in metadata pages. `H5MF__alloc_to_fs_type` (`H5MF.c`) gives a larger
    // allocation the generic large-page type.
    // `H5FS_create` and `H5FS_alloc_sect` in H5FS.c, HDF5 2.2.0, allocate FSHD and FSSE
    // with the metadata aliases `H5FD_MEM_FSPACE_HDR` and `H5FD_MEM_FSPACE_SINFO`.
    pub(crate) fn allocation_class(self) -> AllocationClass {
        match self {
            Self::Header
            | Self::DirectBlock
            | Self::IndirectBlock
            | Self::HugeObjectIndex
            | Self::ManagedFsmHeader
            | Self::ManagedFsmSectionList => AllocationClass::Metadata,
            Self::HugeObject => AllocationClass::Raw,
        }
    }
}

/// Returns the complete allocation set for the heap at `heap_header_address`.
///
/// `source` is framed at the effective base in `addresses`. Owned extents use base-relative
/// coordinates. The walker bounds every allocation and parses managed blocks through
/// [`FractalHeapBlockLayout`]. Checksummed direct blocks are read and validated in full,
/// regardless of the `checksum` feature. Direct blocks without checksums require only their prefix.
/// A filtered direct block is read at its filtered size and decoded, and its decoded bytes are
/// validated in full.
/// Managed-space manager headers and section lists also require checksum validation, regardless
/// of the `checksum` feature. The walker reads only the used section-list bytes and accounts for
/// the complete allocation. Huge-object payload contents are omitted from the reads.
///
/// # Errors
///
/// Returns [`FractalHeapStorageError::UnsupportedFilter`] if a filtered direct block has an active
/// filter this crate does not decode, and [`FractalHeapStorageError::FilteredHugeObjects`] if a
/// filtered heap has a huge-object index. Returns [`FractalHeapStorageError::InvalidStorage`]
/// for duplicate blocks, contradictory allocation accounting, an unexpected huge-object index
/// layout, or overlapping allocations. Returns [`FractalHeapStorageError::Layout`] for invalid
/// layout calculations and [`FractalHeapStorageError::Format`] for parser or source errors,
/// including malformed block geometry, incorrect heap-header owner addresses, truncated
/// allocations, invalid checksums, filtered blocks that do not decode, and unsupported serialized
/// manager section classes.
pub(crate) fn collect_fractal_heap_storage_from_source<S: Source + ?Sized>(
    source: &S,
    addresses: AddressContext,
    heap_header_address: StoredAddress,
    widths: FormatWidths,
) -> Result<FractalHeapStorage, FractalHeapStorageError> {
    let (header, header_range) = crate::fractal_heap::read_header_and_range_from_source(
        source,
        addresses,
        heap_header_address,
        widths,
    )?;
    let blocks =
        FractalHeapBlockLayout::new(&header, widths.offsets, widths.lengths, heap_header_address)?;

    let mut allocations = vec![(
        extent_in_source(source, heap_header_address.get(), header_range.length())?,
        FractalHeapAllocation::Header,
    )];

    let mut managed =
        ManagedStorageWalk::new(source, addresses, &header, &blocks, &mut allocations);
    managed.collect()?;

    collect_huge_storage(source, &header, widths, &mut allocations)?;

    collect_managed_fsm_storage(source, addresses, &header, widths, &mut allocations)?;

    allocations.sort_unstable_by_key(|&(extent, _)| extent);
    if allocations
        .windows(2)
        .any(|pair| pair[0].0.end() > pair[1].0.start())
    {
        return Err(FractalHeapStorageError::InvalidStorage);
    }
    Ok(FractalHeapStorage { allocations })
}

/// Returns fractal-heap storage from a buffered test image.
#[cfg(test)]
fn collect_fractal_heap_storage(
    file_data: &[u8],
    heap_header_address: StoredAddress,
) -> Result<FractalHeapStorage, FractalHeapStorageError> {
    collect_fractal_heap_storage_from_source(
        &BytesSource::new(file_data),
        AddressContext::from_effective_base(BaseAddress::ZERO),
        heap_header_address,
        FORMAT_WIDTHS,
    )
}

struct ManagedStorageWalk<'a, 'e, S: Source + ?Sized> {
    source: &'a S,
    addresses: AddressContext,
    header: &'a FractalHeapHeader,
    blocks: &'a FractalHeapBlockLayout,
    visited: BTreeSet<StoredAddress>,
    allocations: &'e mut Vec<(Extent, FractalHeapAllocation)>,
    allocated_direct_space: u64,
}

impl<'a, 'e, S: Source + ?Sized> ManagedStorageWalk<'a, 'e, S> {
    fn new(
        source: &'a S,
        addresses: AddressContext,
        header: &'a FractalHeapHeader,
        blocks: &'a FractalHeapBlockLayout,
        allocations: &'e mut Vec<(Extent, FractalHeapAllocation)>,
    ) -> Self {
        Self {
            source,
            addresses,
            header,
            blocks,
            allocations,
            visited: BTreeSet::new(),
            allocated_direct_space: 0,
        }
    }

    fn collect(&mut self) -> Result<(), FractalHeapStorageError> {
        let Some(first) = self.blocks.root()? else {
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
        };
        let span = match first {
            FractalHeapBlock::Direct(block) => block.decoded_size(),
            FractalHeapBlock::Indirect(block) => {
                self.blocks.layout().indirect_heap_size(block.rows())?
            }
        };
        if self.header.managed_space != span {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        let mut pending = vec![first];
        while let Some(block) = pending.pop() {
            let address = match block {
                FractalHeapBlock::Direct(block) => block.address(),
                FractalHeapBlock::Indirect(block) => block.address(),
            };
            if !self.visited.insert(address) {
                return Err(FractalHeapStorageError::InvalidStorage);
            }
            match block {
                FractalHeapBlock::Direct(block) => {
                    let reference = block.at(self.addresses.resolve(address)?)?;
                    let extent =
                        extent_in_source(self.source, address.get(), reference.range().length())?;
                    match block.filters() {
                        Some(filters) => self.check_filtered_direct(block, reference, filters)?,
                        None => self.check_direct(reference)?,
                    }
                    self.allocations
                        .push((extent, FractalHeapAllocation::DirectBlock));
                    self.allocated_direct_space = self
                        .allocated_direct_space
                        .checked_add(block.decoded_size())
                        .ok_or(FractalHeapStorageError::InvalidStorage)?;
                }
                FractalHeapBlock::Indirect(block) => {
                    let reference = block.at(self.addresses.resolve(address)?)?;
                    let range = reference.parse_range();
                    let extent = extent_in_source(self.source, address.get(), range.length())?;
                    let bytes =
                        crate::fractal_heap::read_header_range(self.source, self.addresses, range)?;
                    let view = reference.parse(FileRegion::new(range, &bytes)?)?;
                    self.allocations
                        .push((extent, FractalHeapAllocation::IndirectBlock));
                    for child in view.children() {
                        pending.push(child?);
                    }
                }
            }
        }
        if self.allocated_direct_space != self.header.allocated_managed_space {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        Ok(())
    }

    /// Validates an unfiltered direct block: in full if the heap header requires checksums, and
    /// its prefix otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::Format`] if the read or the validation fails.
    fn check_direct(
        &self,
        reference: FractalHeapDirectBlockRef<'_>,
    ) -> Result<(), FractalHeapStorageError> {
        let checksummed = self.header.flags
            & hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED
            != 0;
        let range = if checksummed {
            reference.parse_range()
        } else {
            reference.prefix_range()?
        };
        let bytes = crate::fractal_heap::read_header_range(self.source, self.addresses, range)?;
        let region = FileRegion::new(range, &bytes)?;
        if checksummed {
            reference.parse_checked(region)?;
        } else {
            reference.parse_prefix(region)?;
        }
        Ok(())
    }

    /// Decodes a filtered direct block and validates its decoded bytes in full.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::UnsupportedFilter`] if an active filter has no decoder
    /// or an active shuffle filter has an element width other than one byte, and
    /// [`FractalHeapStorageError::Format`] if the read, the decoding, or the validation of the
    /// decoded bytes fails otherwise.
    fn check_filtered_direct(
        &self,
        block: FractalHeapDirectBlock<'_>,
        reference: FractalHeapDirectBlockRef<'_>,
        filters: FractalHeapBlockFilters<'_>,
    ) -> Result<(), FractalHeapStorageError> {
        let encoded =
            crate::fractal_heap::read_header_range(self.source, self.addresses, reference.range())?;
        let decoded =
            crate::fractal_heap::decode_direct_block(&encoded, filters, block.decoded_size())
                .map_err(|error| match error {
                    FormatError::UnsupportedFilter(filter_id) => {
                        FractalHeapStorageError::UnsupportedFilter(filter_id)
                    }
                    error => FractalHeapStorageError::Format(error),
                })?;
        reference.parse_decoded_checked(&decoded)?;
        Ok(())
    }
}

/// Records the manager header and complete section-list allocation after client-0 validation.
///
/// Section records describe logical heap space and are excluded from the owned file extents.
///
/// # Errors
///
/// Returns [`FractalHeapStorageError::InvalidStorage`] if the section-list address is undefined
/// with nonzero size or space totals. Returns [`FractalHeapStorageError::Format`] for parser,
/// source, or range failures, including unsupported serialized section classes.
fn collect_managed_fsm_storage(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    heap: &FractalHeapHeader,
    widths: FormatWidths,
    allocations: &mut Vec<(Extent, FractalHeapAllocation)>,
) -> Result<(), FractalHeapStorageError> {
    let address = heap.managed_block_free_space_manager_address;
    if address.is_undefined(widths.offsets) {
        return Ok(());
    }
    let reference = FreeSpaceManagerHeaderRef::new(widths, addresses.resolve(address)?)?;
    let extent = extent_in_source(source, address.get(), reference.range().length())?;
    let range = reference.parse_range();
    let bytes = crate::fractal_heap::read_header_range(source, addresses, range)?;
    let frame = reference.parse_checked(FileRegion::new(range, &bytes)?)?;
    let manager = frame.fractal_heap_header(heap)?;
    allocations.push((extent, FractalHeapAllocation::ManagedFsmHeader));
    let DecodedStoredAddress::Defined(list_address) = frame.section_list_addr() else {
        if frame.section_list_used() != 0
            || frame.section_list_allocated() != 0
            || frame.total_space() != 0
        {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        return Ok(());
    };
    let reference = manager.section_list_ref(address, addresses.resolve(list_address)?)?;
    let extent = extent_in_source(source, list_address.get(), reference.range().length())?;
    let range = reference.parse_range();
    let bytes = crate::fractal_heap::read_header_range(source, addresses, range)?;
    reference.parse_checked(FileRegion::new(range, &bytes)?)?;
    allocations.push((extent, FractalHeapAllocation::ManagedFsmSectionList));
    Ok(())
}

fn collect_huge_storage<S: Source + ?Sized>(
    source: &S,
    heap: &FractalHeapHeader,
    widths: FormatWidths,
    allocations: &mut Vec<(Extent, FractalHeapAllocation)>,
) -> Result<(), FractalHeapStorageError> {
    let tree_address = heap.btree_huge_objects_address;
    if tree_address.is_undefined(widths.offsets) {
        if heap.huge_objects_count != 0 || heap.huge_objects_size != 0 {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        return Ok(());
    }
    if heap.filtering.is_filtered() {
        return Err(FractalHeapStorageError::FilteredHugeObjects);
    }

    let header = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(source),
        tree_address.get(),
        widths,
    )?;
    let direct = heap.huge_ids_direct_with_widths(widths);
    let (expected_type, required_record_size) = if direct {
        (
            BTREE_V2_HUGE_OBJECT_DIRECT,
            HugeObjectDirectRecord::size(widths.offsets, widths.lengths),
        )
    } else {
        (
            BTREE_V2_HUGE_OBJECT,
            HugeObjectRecord::size(widths.offsets, widths.lengths),
        )
    };
    if header.tree_type != expected_type || header.record_size != required_record_size {
        return Err(FractalHeapStorageError::InvalidStorage);
    }

    let tree_extents =
        crate::btree_v2::collect_btree_v2_storage_extents(source, tree_address, widths)?;
    let records =
        crate::btree_v2::collect_btree_v2_records_from_source(source, &header, widths.offsets)?;
    if u64::try_from(records.len()).ok() != Some(heap.huge_objects_count)
        || header.total_records != heap.huge_objects_count
    {
        return Err(FractalHeapStorageError::InvalidStorage);
    }

    let mut payload_bytes = 0u64;
    let mut payloads = Vec::with_capacity(records.len());
    for record in records {
        let (address, length) = if direct {
            let record = record.huge_object_direct(widths.offsets, widths.lengths)?;
            (record.address, record.length)
        } else {
            let record = record.huge_object(widths.offsets, widths.lengths)?;
            (record.address, record.length)
        };
        if address.is_undefined(widths.offsets) {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        payload_bytes = payload_bytes
            .checked_add(length)
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        payloads.push((
            extent_in_source(source, address.get(), length)?,
            FractalHeapAllocation::HugeObject,
        ));
    }
    if payload_bytes != heap.huge_objects_size {
        return Err(FractalHeapStorageError::InvalidStorage);
    }

    allocations.extend(
        tree_extents
            .into_iter()
            .map(|extent| (extent, FractalHeapAllocation::HugeObjectIndex)),
    );
    allocations.extend(payloads);
    Ok(())
}

/// Returns a checked non-empty extent after proving its whole range lies in `source`.
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

#[cfg(test)]
mod tests {
    use core::cell::RefCell;

    use hdf5_pure_format::__private::FilterDescription;
    use hdf5_pure_format::__private::FilterPipeline;
    use hdf5_pure_format::__private::FractalHeapFiltering;
    use hdf5_pure_format::__private::FractalHeapSectionClass;
    use hdf5_pure_format::__private::btree_v2_header_size;
    use test_util::btree_v2;
    use test_util::fractal_heap;
    use test_util::free_space;
    use test_util::widths::Widths;

    use crate::access_mode::AccessMode;
    use crate::address::OpenedImageContext;
    use crate::fractal_heap::HeapObjectReader;
    use crate::source::BaseOffsetSource;
    use crate::test_data;

    use super::*;

    const STORAGE_HEAP_AT: u64 = 0x100;
    const STORAGE_ROOT_AT: u64 = 0x400;

    fn storage_header() -> FractalHeapHeader {
        FractalHeapHeader {
            heap_id_length: 7,
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
            filtering: FractalHeapFiltering::Unfiltered,
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
        block.push(test_data::BLOCK_VERSION);
        block.extend_from_slice(&STORAGE_HEAP_AT.to_le_bytes());
        block.extend_from_slice(&heap_offset.to_le_bytes()[..2]);
        block
    }

    fn direct_block(heap_offset: u64, size: usize) -> Vec<u8> {
        let mut block = managed_prefix(fractal_heap::DIRECT_BLOCK_SIGNATURE, heap_offset);
        block.resize(size, 0);
        block
    }

    fn indirect_block(
        header: &FractalHeapHeader,
        heap_offset: u64,
        nrows: u16,
        children: &[(usize, u64, Option<u64>)],
    ) -> Vec<u8> {
        let layout = storage_layout(header);
        let filtered_entries = if header.filtering.is_filtered() {
            usize::from(nrows).min(layout.direct_rows()) * layout.table_width()
        } else {
            0
        };
        let mut block = managed_prefix(fractal_heap::INDIRECT_BLOCK_SIGNATURE, heap_offset);
        for entry in 0..usize::from(nrows) * layout.table_width() {
            let (address, filtered_size) = children
                .iter()
                .find_map(|&(at, address, size)| (at == entry).then_some((address, size)))
                .unwrap_or((u64::MAX, None));
            block.extend_from_slice(&address.to_le_bytes());
            if entry < filtered_entries {
                block.extend_from_slice(&filtered_size.unwrap_or(0).to_le_bytes());
                block.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        block.extend_from_slice(&[0; 4]);
        assert_eq!(
            len_u64(block.len()),
            layout.indirect_block_size(nrows).unwrap()
        );
        let len = block.len();
        crate::checksum::stamp_trailing(&mut block, 0, len);
        block
    }

    fn allocation(
        start: u64,
        len: u64,
        kind: FractalHeapAllocation,
    ) -> (Extent, FractalHeapAllocation) {
        (Extent::new(start, len).unwrap(), kind)
    }

    fn collect_storage(file: &[u8]) -> Result<FractalHeapStorage, FractalHeapStorageError> {
        collect_fractal_heap_storage(file, StoredAddress::new(STORAGE_HEAP_AT))
    }

    #[rstest::rstest]
    #[case::header(FractalHeapAllocation::Header, AllocationClass::Metadata)]
    #[case::direct(FractalHeapAllocation::DirectBlock, AllocationClass::Metadata)]
    #[case::indirect(FractalHeapAllocation::IndirectBlock, AllocationClass::Metadata)]
    #[case::huge_index(FractalHeapAllocation::HugeObjectIndex, AllocationClass::Metadata)]
    #[case::manager_header(FractalHeapAllocation::ManagedFsmHeader, AllocationClass::Metadata)]
    #[case::manager_sections(
        FractalHeapAllocation::ManagedFsmSectionList,
        AllocationClass::Metadata
    )]
    #[case::huge_payload(FractalHeapAllocation::HugeObject, AllocationClass::Raw)]
    fn heap_metadata_and_huge_payloads_have_distinct_allocation_classes(
        #[case] kind: FractalHeapAllocation,
        #[case] class: AllocationClass,
    ) {
        assert_eq!(kind.allocation_class(), class);
    }

    #[rstest::rstest]
    #[case::valid(false)]
    #[case::corrupt(true)]
    fn checksummed_direct_ownership_requires_complete_checksum_validation(#[case] corrupt: bool) {
        let header = FractalHeapHeader {
            flags: hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            managed_space: 128,
            allocated_managed_space: 128,
            ..storage_header()
        };
        let mut block = direct_block(0, 128);
        block[19..].fill(0x5A);
        let stored = crate::checksum::jenkins_lookup3(&block);
        block[15..19].copy_from_slice(&stored.to_le_bytes());
        if corrupt {
            block[127] ^= 1;
        }
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        place(&mut file, STORAGE_ROOT_AT, &block);
        let source = PrefixSource {
            bytes: &file,
            reads: RefCell::new(Vec::new()),
        };
        let result = collect_fractal_heap_storage_from_source(
            &source,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(STORAGE_HEAP_AT),
            FORMAT_WIDTHS,
        );
        if corrupt {
            block[15..19].fill(0);
            assert_eq!(
                result.unwrap_err(),
                FractalHeapStorageError::Format(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed: crate::checksum::jenkins_lookup3(&block),
                })
            );
        } else {
            assert_eq!(
                result.unwrap().allocations()[1],
                allocation(STORAGE_ROOT_AT, 128, FractalHeapAllocation::DirectBlock)
            );
        }
        assert_eq!(
            source.reads.borrow().last().copied(),
            Some((STORAGE_ROOT_AT, 128))
        );
    }

    #[test]
    fn ownership_reads_only_prefixes_of_unchecksummed_direct_blocks() {
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            starting_block_size: 1 << 20,
            max_direct_block_size: 4 << 20,
            max_heap_size: 32,
            managed_space: 1 << 20,
            allocated_managed_space: 1 << 20,
            ..storage_header()
        };
        let mut file = vec![0; STORAGE_ROOT_AT as usize + (1 << 20)];
        place_storage_header(&mut file, &header);
        let mut prefix = b"FHDB".to_vec();
        prefix.push(0);
        prefix.extend_from_slice(&STORAGE_HEAP_AT.to_le_bytes());
        prefix.extend_from_slice(&0u32.to_le_bytes());
        place(&mut file, STORAGE_ROOT_AT, &prefix);
        let source = PrefixSource {
            bytes: &file,
            reads: RefCell::new(Vec::new()),
        };
        let storage = collect_fractal_heap_storage_from_source(
            &source,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(STORAGE_HEAP_AT),
            FORMAT_WIDTHS,
        )
        .unwrap();
        assert_eq!(
            storage.allocations()[1],
            allocation(STORAGE_ROOT_AT, 1 << 20, FractalHeapAllocation::DirectBlock,)
        );
        assert_eq!(
            *source.reads.borrow(),
            vec![
                (
                    STORAGE_HEAP_AT,
                    hdf5_pure_format::__private::FractalHeapHeaderFrame::PREFIX_LEN
                ),
                (
                    STORAGE_HEAP_AT,
                    FractalHeapHeader::serialized_size(OffsetWidth::Eight, LengthWidth::Eight)
                ),
                (STORAGE_ROOT_AT, prefix.len()),
            ]
        );
    }

    struct PrefixSource<'a> {
        bytes: &'a [u8],
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl Source for PrefixSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }
        fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
            BytesSource::new(self.bytes).read_at(offset, bytes)
        }
        fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
            self.reads.borrow_mut().push((offset, length));
            BytesSource::new(self.bytes).read_metadata_at(offset, length)
        }
    }

    #[rstest::rstest]
    #[case::direct(0)]
    #[case::indirect(2)]
    #[case::nested(6)]
    fn region_traversal_reads_managed_objects_at_the_effective_base(
        #[case] rows: u16,
        #[values(0, 512, 1024)] effective_base: usize,
    ) {
        const DIRECT: u64 = 0x1400;
        const NESTED: u64 = 0x1000;
        let mut header = storage_header();
        header.root_block_address = StoredAddress::new(STORAGE_ROOT_AT);
        header.current_rows_in_root_indirect_block = rows;
        let offset = if rows == 6 {
            storage_layout(&header).row_offset(5).unwrap()
        } else {
            0
        };
        let mut image = vec![0; 0x1800];
        if rows != 0 {
            place(
                &mut image,
                STORAGE_ROOT_AT,
                &indirect_block(
                    &header,
                    0,
                    rows,
                    &[(
                        if rows == 6 { 20 } else { 0 },
                        if rows == 6 { NESTED } else { DIRECT },
                        None,
                    )],
                ),
            );
        }
        if rows == 6 {
            place(
                &mut image,
                NESTED,
                &indirect_block(&header, offset, 3, &[(0, DIRECT, None)]),
            );
        }
        let direct_at = if rows == 0 { STORAGE_ROOT_AT } else { DIRECT };
        let mut direct = direct_block(offset, 128);
        direct[15..18].copy_from_slice(b"abc");
        place(&mut image, direct_at, &direct);
        let mut id = vec![0];
        id.extend_from_slice(&u16::try_from(offset + 15).unwrap().to_le_bytes());
        id.extend_from_slice(&3u32.to_le_bytes());
        let ids = hdf5_pure_format::__private::FractalHeapIdLayout::new(
            &header,
            OffsetWidth::Eight,
            LengthWidth::Eight,
        );
        let parsed = ids.parse(&id).unwrap();
        let addresses = OpenedImageContext::reconcile(
            BaseAddress::new(512),
            image.len() as u64 + 512,
            effective_base as u64,
        )
        .unwrap()
        .addresses();
        let buffered = HeapObjectReader::new(
            &header,
            addresses,
            StoredAddress::new(STORAGE_HEAP_AT),
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        )
        .read(&image, parsed)
        .unwrap();
        let mut physical = vec![0xFF; effective_base];
        physical.extend_from_slice(&image);
        let source = BytesSource::new(&physical);
        let framed = BaseOffsetSource {
            inner: &source,
            base: addresses.effective_base(),
        };
        let streamed = HeapObjectReader::new(
            &header,
            addresses,
            StoredAddress::new(STORAGE_HEAP_AT),
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        )
        .read_from_source(&framed, parsed)
        .unwrap();
        assert_eq!(buffered, b"abc");
        assert_eq!(streamed, buffered);
    }

    #[rstest::rstest]
    #[case::direct(false)]
    #[case::indirect(true)]
    fn ownership_regions_reject_a_wrong_heap_backlink(#[case] indirect: bool) {
        let header = FractalHeapHeader {
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            current_rows_in_root_indirect_block: if indirect { 2 } else { 0 },
            managed_space: if indirect { 1024 } else { 128 },
            allocated_managed_space: 128,
            ..storage_header()
        };
        let mut block = if indirect {
            indirect_block(&header, 0, 2, &[])
        } else {
            direct_block(0, 128)
        };
        block[5..13].copy_from_slice(&0x200u64.to_le_bytes());
        let mut file = vec![0; 0x800];
        place_storage_header(&mut file, &header);
        place(&mut file, STORAGE_ROOT_AT, &block);
        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::InvalidFractalHeapBlockOwner {
                    expected: STORAGE_HEAP_AT,
                    actual: 0x200
                },
            ))
        );
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

        assert_eq!(
            storage.allocations(),
            &[allocation(
                STORAGE_HEAP_AT,
                header_len,
                FractalHeapAllocation::Header
            )]
        );
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

        assert_eq!(
            storage.allocations(),
            &[allocation(
                STORAGE_HEAP_AT,
                header_len,
                FractalHeapAllocation::Header
            )]
        );
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
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(STORAGE_HEAP_AT),
            FORMAT_WIDTHS,
        )
        .unwrap();
        let header_len = u64::try_from(FractalHeapHeader::serialized_size(
            OffsetWidth::Eight,
            LengthWidth::Eight,
        ))
        .unwrap();

        assert_eq!(buffered, streamed);
        assert_eq!(
            buffered.allocations(),
            &[
                allocation(STORAGE_HEAP_AT, header_len, FractalHeapAllocation::Header),
                allocation(STORAGE_ROOT_AT, 128, FractalHeapAllocation::DirectBlock),
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
            &indirect_block(
                &header,
                0,
                6,
                &[(0, DIRECT_A, None), (20, CHILD_INDIRECT, None)],
            ),
        );
        place(&mut file, DIRECT_A, &direct_block(0, 128));
        place(
            &mut file,
            CHILD_INDIRECT,
            &indirect_block(
                &header,
                child_heap_offset,
                child_rows,
                &[(0, DIRECT_B, None)],
            ),
        );
        place(&mut file, DIRECT_B, &direct_block(child_heap_offset, 128));

        let storage = collect_storage(&file).unwrap();
        let mut expected = vec![
            allocation(
                STORAGE_HEAP_AT,
                u64::try_from(FractalHeapHeader::serialized_size(
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                ))
                .unwrap(),
                FractalHeapAllocation::Header,
            ),
            allocation(
                STORAGE_ROOT_AT,
                storage_layout(&header).indirect_block_size(6).unwrap(),
                FractalHeapAllocation::IndirectBlock,
            ),
            allocation(DIRECT_A, 128, FractalHeapAllocation::DirectBlock),
            allocation(
                CHILD_INDIRECT,
                storage_layout(&header)
                    .indirect_block_size(child_rows)
                    .unwrap(),
                FractalHeapAllocation::IndirectBlock,
            ),
            allocation(DIRECT_B, 128, FractalHeapAllocation::DirectBlock),
        ];
        expected.sort_unstable_by_key(|&(extent, _)| extent);

        assert_eq!(storage.allocations(), expected);
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
        let mut block = indirect_block(&header, 0, 2, &[(0, DIRECT, None)]);
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
            &indirect_block(&header, 0, 2, &[(0, CHILD, None), (1, CHILD, None)]),
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

    #[cfg(feature = "deflate")]
    #[test]
    fn a_filtered_direct_root_is_owned_at_its_stored_sizes() {
        let block = checksummed_direct_block(0, 128);
        let encoded = test_data::encode_block(&block, &test_data::deflate_pipeline());
        let header = filtered_header(
            FractalHeapHeader {
                flags: hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
                root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
                managed_space: 128,
                allocated_managed_space: 128,
                managed_objects_count: 1,
                ..storage_header()
            },
            len_u64(encoded.len()),
            test_data::deflate_pipeline(),
        );
        let mut file = vec![0; 0x800];
        let header_len = place_filtered_storage_header(&mut file, &header);
        place(&mut file, STORAGE_ROOT_AT, &encoded);
        let source = PrefixSource {
            bytes: &file,
            reads: RefCell::new(Vec::new()),
        };

        let storage = collect_fractal_heap_storage_from_source(
            &source,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(STORAGE_HEAP_AT),
            FORMAT_WIDTHS,
        )
        .unwrap();

        assert_eq!(
            storage.allocations(),
            &[
                allocation(STORAGE_HEAP_AT, header_len, FractalHeapAllocation::Header),
                allocation(
                    STORAGE_ROOT_AT,
                    len_u64(encoded.len()),
                    FractalHeapAllocation::DirectBlock
                ),
            ]
        );
        assert_eq!(
            *source.reads.borrow(),
            vec![
                (
                    STORAGE_HEAP_AT,
                    hdf5_pure_format::__private::FractalHeapHeaderFrame::PREFIX_LEN
                ),
                (STORAGE_HEAP_AT, usize::try_from(header_len).unwrap()),
                (STORAGE_ROOT_AT, encoded.len()),
            ]
        );
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn a_filtered_indirect_root_owns_its_children_at_their_stored_sizes() {
        const DIRECT_A: u64 = 0x800;
        const CHILD_INDIRECT: u64 = 0x1000;
        const DIRECT_B: u64 = 0x1400;
        let header = filtered_header(
            FractalHeapHeader {
                root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
                current_rows_in_root_indirect_block: 6,
                managed_space: 16 * 1024,
                allocated_managed_space: 256,
                managed_objects_count: 2,
                ..storage_header()
            },
            0,
            test_data::deflate_pipeline(),
        );
        let layout = storage_layout(&header);
        let child_heap_offset = layout.row_offset(5).unwrap();
        let child_rows = layout.child_indirect_rows(5).unwrap();
        let encoded_a =
            test_data::encode_block(&direct_block(0, 128), &test_data::deflate_pipeline());
        let encoded_b = test_data::encode_block(
            &direct_block(child_heap_offset, 128),
            &test_data::deflate_pipeline(),
        );
        let mut file = vec![0; 0x1800];
        let header_len = place_filtered_storage_header(&mut file, &header);
        place(
            &mut file,
            STORAGE_ROOT_AT,
            &indirect_block(
                &header,
                0,
                6,
                &[
                    (0, DIRECT_A, Some(len_u64(encoded_a.len()))),
                    (20, CHILD_INDIRECT, None),
                ],
            ),
        );
        place(&mut file, DIRECT_A, &encoded_a);
        place(
            &mut file,
            CHILD_INDIRECT,
            &indirect_block(
                &header,
                child_heap_offset,
                child_rows,
                &[(0, DIRECT_B, Some(len_u64(encoded_b.len())))],
            ),
        );
        place(&mut file, DIRECT_B, &encoded_b);

        let storage = collect_storage(&file).unwrap();

        assert_eq!(
            storage.allocations(),
            &[
                allocation(STORAGE_HEAP_AT, header_len, FractalHeapAllocation::Header),
                allocation(
                    STORAGE_ROOT_AT,
                    layout.indirect_block_size(6).unwrap(),
                    FractalHeapAllocation::IndirectBlock,
                ),
                allocation(
                    DIRECT_A,
                    len_u64(encoded_a.len()),
                    FractalHeapAllocation::DirectBlock
                ),
                allocation(
                    CHILD_INDIRECT,
                    layout.indirect_block_size(child_rows).unwrap(),
                    FractalHeapAllocation::IndirectBlock,
                ),
                allocation(
                    DIRECT_B,
                    len_u64(encoded_b.len()),
                    FractalHeapAllocation::DirectBlock
                ),
            ]
        );
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn a_filtered_heap_owns_its_managed_free_space_manager() {
        let (mut file, used) = managed_fsm_image(64);
        let unfiltered = FractalHeapHeader::parse_bytes(
            &file[usize::try_from(STORAGE_HEAP_AT).unwrap()..],
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .unwrap();
        let encoded =
            test_data::encode_block(&direct_block(0, 128), &test_data::deflate_pipeline());
        let header = filtered_header(
            unfiltered,
            len_u64(encoded.len()),
            test_data::deflate_pipeline(),
        );
        let header_len = place_filtered_storage_header(&mut file, &header);
        place(&mut file, STORAGE_ROOT_AT, &encoded);

        assert_eq!(
            collect_storage(&file).unwrap().allocations(),
            &[
                allocation(STORAGE_HEAP_AT, header_len, FractalHeapAllocation::Header),
                allocation(
                    STORAGE_ROOT_AT,
                    len_u64(encoded.len()),
                    FractalHeapAllocation::DirectBlock
                ),
                allocation(MANAGED_FSM_AT, 82, FractalHeapAllocation::ManagedFsmHeader),
                allocation(
                    MANAGED_FSSE_AT,
                    used + 64,
                    FractalHeapAllocation::ManagedFsmSectionList
                ),
            ]
        );
    }

    #[test]
    fn huge_objects_in_a_filtered_heap_are_an_unsupported_owner() {
        let header = filtered_header(
            FractalHeapHeader {
                btree_huge_objects_address: StoredAddress::new(0x600),
                huge_objects_count: 1,
                huge_objects_size: 64,
                ..storage_header()
            },
            0,
            test_data::filter_pipeline(vec![]),
        );
        let mut file = vec![0; 0x800];
        place_filtered_storage_header(&mut file, &header);

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::FilteredHugeObjects)
        );
    }

    #[test]
    fn a_filter_without_a_decoder_makes_the_heap_an_unsupported_owner() {
        let block = direct_block(0, 128);
        let header = filtered_header(
            FractalHeapHeader {
                root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
                managed_space: 128,
                allocated_managed_space: 128,
                ..storage_header()
            },
            len_u64(block.len()),
            test_data::filter_pipeline(vec![FilterDescription {
                filter_id: test_data::FILTER_BZIP2,
                name: None,
                flags: 0,
                client_data: vec![],
            }]),
        );
        let mut file = vec![0; 0x800];
        place_filtered_storage_header(&mut file, &header);
        place(&mut file, STORAGE_ROOT_AT, &block);

        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::UnsupportedFilter(
                test_data::FILTER_BZIP2
            ))
        );
    }

    #[cfg(feature = "deflate")]
    #[rstest::rstest]
    #[case::zero_root_size(FilteredDamage::ZeroRootSize)]
    #[case::zero_child_size(FilteredDamage::ZeroChildSize)]
    #[case::past_the_end_of_the_file(FilteredDamage::PastTheEndOfTheFile)]
    #[case::overlapping_sibling(FilteredDamage::OverlappingSibling)]
    #[case::overlapping_header(FilteredDamage::OverlappingHeader)]
    #[case::undecodable(FilteredDamage::Undecodable)]
    #[case::short_decoded_block(FilteredDamage::ShortDecodedBlock)]
    #[case::wrong_owner(FilteredDamage::WrongOwner)]
    #[case::bad_checksum(FilteredDamage::BadChecksum)]
    fn malformed_filtered_direct_blocks_are_rejected(#[case] damage: FilteredDamage) {
        const DIRECT_A: u64 = 0x800;
        const DIRECT_B: u64 = 0x900;
        let indirect = matches!(
            damage,
            FilteredDamage::ZeroChildSize | FilteredDamage::OverlappingSibling
        );
        let mut block_a = checksummed_direct_block(0, 128);
        let mut computed = 0;
        let checksum = test_data::DIRECT_BLOCK_CHECKSUM_AT..test_data::DIRECT_BLOCK_DATA_AT;
        let stored = u32::from_le_bytes(block_a[checksum.clone()].try_into().unwrap());
        match damage {
            FilteredDamage::WrongOwner => {
                let owner = test_data::DIRECT_BLOCK_OWNER_AT;
                block_a[owner..owner + 8].copy_from_slice(&0x200u64.to_le_bytes());
            }
            FilteredDamage::BadChecksum => {
                *block_a.last_mut().unwrap() ^= 1;
                let mut zeroed = block_a.clone();
                zeroed[checksum].fill(0);
                computed = crate::checksum::jenkins_lookup3(&zeroed);
            }
            FilteredDamage::ShortDecodedBlock => {
                block_a.pop();
            }
            _ => {}
        }
        let mut encoded_a = test_data::encode_block(&block_a, &test_data::deflate_pipeline());
        if damage == FilteredDamage::Undecodable {
            encoded_a.fill(0xFF);
        }
        let encoded_b = test_data::encode_block(
            &checksummed_direct_block(128, 128),
            &test_data::deflate_pipeline(),
        );
        let mut size_a = len_u64(encoded_a.len());
        let mut file = vec![0; 0xc00];
        let file_len = len_u64(file.len());
        let root_at = match damage {
            FilteredDamage::PastTheEndOfTheFile => {
                size_a = file_len - 0xb00 + 1;
                0xb00
            }
            FilteredDamage::OverlappingHeader => STORAGE_HEAP_AT - size_a,
            _ => STORAGE_ROOT_AT,
        };
        match damage {
            FilteredDamage::ZeroRootSize | FilteredDamage::ZeroChildSize => size_a = 0,
            FilteredDamage::OverlappingSibling => size_a = DIRECT_B - DIRECT_A + 1,
            FilteredDamage::OverlappingHeader => size_a += 1,
            _ => {}
        }
        let header = filtered_header(
            FractalHeapHeader {
                flags: hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
                root_block_address: StoredAddress::new(root_at),
                current_rows_in_root_indirect_block: if indirect { 2 } else { 0 },
                managed_space: if indirect { 1024 } else { 128 },
                allocated_managed_space: if indirect { 256 } else { 128 },
                ..storage_header()
            },
            if indirect { 0 } else { size_a },
            test_data::deflate_pipeline(),
        );
        if indirect {
            place(
                &mut file,
                root_at,
                &indirect_block(
                    &header,
                    0,
                    2,
                    &[
                        (0, DIRECT_A, Some(size_a)),
                        (1, DIRECT_B, Some(len_u64(encoded_b.len()))),
                    ],
                ),
            );
            place(&mut file, DIRECT_A, &encoded_a);
            place(&mut file, DIRECT_B, &encoded_b);
        } else {
            let length = encoded_a
                .len()
                .min(file.len() - usize::try_from(root_at).unwrap());
            place(&mut file, root_at, &encoded_a[..length]);
        }
        place_filtered_storage_header(&mut file, &header);

        let error = collect_storage(&file).unwrap_err();
        let expected = match damage {
            FilteredDamage::ZeroRootSize | FilteredDamage::ZeroChildSize => {
                FractalHeapStorageError::Format(FormatError::InvalidFractalHeapGeometry)
            }
            FilteredDamage::PastTheEndOfTheFile => {
                FractalHeapStorageError::Format(FormatError::UnexpectedEof {
                    expected: file.len() + 1,
                    available: file.len(),
                })
            }
            FilteredDamage::OverlappingSibling | FilteredDamage::OverlappingHeader => {
                FractalHeapStorageError::InvalidStorage
            }
            FilteredDamage::Undecodable => {
                let FractalHeapStorageError::Format(FormatError::FilterError(reason)) = &error
                else {
                    panic!("expected FilterError, got {error:?}");
                };
                assert!(
                    test_data::DEFLATE_HEADER_ERRORS.contains(&reason.as_str()),
                    "{reason:?}"
                );
                return;
            }
            FilteredDamage::ShortDecodedBlock => {
                FractalHeapStorageError::Format(FormatError::DataSizeMismatch {
                    expected: 128,
                    actual: 127,
                })
            }
            FilteredDamage::WrongOwner => {
                FractalHeapStorageError::Format(FormatError::InvalidFractalHeapBlockOwner {
                    expected: STORAGE_HEAP_AT,
                    actual: 0x200,
                })
            }
            FilteredDamage::BadChecksum => {
                FractalHeapStorageError::Format(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                })
            }
        };
        assert_eq!(error, expected);
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum FilteredDamage {
        BadChecksum,
        OverlappingHeader,
        OverlappingSibling,
        PastTheEndOfTheFile,
        ShortDecodedBlock,
        Undecodable,
        WrongOwner,
        ZeroChildSize,
        ZeroRootSize,
    }

    fn filtered_header(
        header: FractalHeapHeader,
        root_direct_block_size: u64,
        pipeline: FilterPipeline,
    ) -> FractalHeapHeader {
        FractalHeapHeader {
            filtering: FractalHeapFiltering::Filtered {
                root_direct_block_size,
                root_filter_mask: 0,
                pipeline,
            },
            ..header
        }
    }

    /// Writes the filtered `header` at [`STORAGE_HEAP_AT`] and returns its encoded length.
    fn place_filtered_storage_header(file: &mut [u8], header: &FractalHeapHeader) -> u64 {
        let FractalHeapFiltering::Filtered {
            root_direct_block_size,
            root_filter_mask,
            pipeline,
        } = &header.filtering
        else {
            panic!("the header is filtered");
        };
        let unfiltered = FractalHeapHeader {
            filtering: FractalHeapFiltering::Unfiltered,
            ..header.clone()
        }
        .serialize(OffsetWidth::Eight, LengthWidth::Eight)
        .unwrap();
        let bytes = test_data::with_filtering(
            &unfiltered,
            Widths::EIGHT,
            *root_direct_block_size,
            *root_filter_mask,
            &pipeline.serialize().unwrap(),
        );
        place(file, STORAGE_HEAP_AT, &bytes);
        len_u64(bytes.len())
    }

    fn checksummed_direct_block(heap_offset: u64, size: usize) -> Vec<u8> {
        let mut block = direct_block(heap_offset, size);
        block[test_data::DIRECT_BLOCK_DATA_AT..].fill(0x5A);
        let checksum = crate::checksum::jenkins_lookup3(&block);
        block[test_data::DIRECT_BLOCK_CHECKSUM_AT..test_data::DIRECT_BLOCK_DATA_AT]
            .copy_from_slice(&checksum.to_le_bytes());
        block
    }

    fn len_u64(len: usize) -> u64 {
        u64::try_from(len).unwrap()
    }

    #[rstest::rstest]
    #[case::exact(0, 0)]
    #[case::reserved(64, 0)]
    #[case::userblock(64, 512)]
    fn managed_fsm_ownership_includes_reserved_capacity(#[case] spare: u64, #[case] base: usize) {
        let (file, used) = managed_fsm_image(spare);
        let mut physical = vec![0xFF; base];
        physical.extend_from_slice(&file);
        let inner = BytesSource::new(&physical);
        let addresses =
            AddressContext::from_effective_base(BaseAddress::new(u64::try_from(base).unwrap()));
        let source = BaseOffsetSource {
            inner: &inner,
            base: addresses.effective_base(),
        };
        let result = collect_fractal_heap_storage_from_source(
            &source,
            addresses,
            StoredAddress::new(STORAGE_HEAP_AT),
            FORMAT_WIDTHS,
        )
        .unwrap();
        assert_eq!(
            result.allocations(),
            &[
                allocation(
                    STORAGE_HEAP_AT,
                    u64::try_from(FractalHeapHeader::serialized_size(
                        OffsetWidth::Eight,
                        LengthWidth::Eight
                    ))
                    .unwrap(),
                    FractalHeapAllocation::Header
                ),
                allocation(STORAGE_ROOT_AT, 128, FractalHeapAllocation::DirectBlock),
                allocation(MANAGED_FSM_AT, 82, FractalHeapAllocation::ManagedFsmHeader),
                allocation(
                    MANAGED_FSSE_AT,
                    used + spare,
                    FractalHeapAllocation::ManagedFsmSectionList
                ),
            ]
        );
        assert_eq!(
            FractalHeapAllocation::ManagedFsmHeader.allocation_class(),
            AllocationClass::Metadata
        );
        assert_eq!(
            FractalHeapAllocation::ManagedFsmSectionList.allocation_class(),
            AllocationClass::Metadata
        );
    }

    #[test]
    fn managed_fsm_ownership_reads_only_used_section_list_bytes() {
        let (file, used) = managed_fsm_image(512);
        let source = PrefixSource {
            bytes: &file,
            reads: RefCell::new(Vec::new()),
        };
        let storage = collect_fractal_heap_storage_from_source(
            &source,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(STORAGE_HEAP_AT),
            FORMAT_WIDTHS,
        )
        .unwrap();
        assert_eq!(
            storage.allocations().last().unwrap(),
            &allocation(
                MANAGED_FSSE_AT,
                used + 512,
                FractalHeapAllocation::ManagedFsmSectionList
            )
        );
        assert_eq!(
            *source.reads.borrow(),
            vec![
                (
                    STORAGE_HEAP_AT,
                    hdf5_pure_format::__private::FractalHeapHeaderFrame::PREFIX_LEN
                ),
                (
                    STORAGE_HEAP_AT,
                    FractalHeapHeader::serialized_size(OffsetWidth::Eight, LengthWidth::Eight)
                ),
                (STORAGE_ROOT_AT, managed_prefix(b"FHDB", 0).len()),
                (MANAGED_FSM_AT, 82),
                (MANAGED_FSSE_AT, usize::try_from(used).unwrap()),
            ]
        );
    }

    #[rstest::rstest]
    #[case::header(MANAGED_FSM_AT, 82)]
    #[case::section_list(MANAGED_FSSE_AT, 23)]
    fn managed_fsm_ownership_requires_valid_checksums(#[case] at: u64, #[case] len: usize) {
        let (mut file, _) = managed_fsm_image(64);
        let at = usize::try_from(at).unwrap();
        let checksum_at = at + len - 4;
        let computed = crate::checksum::jenkins_lookup3(&file[at..checksum_at]);
        file[checksum_at] ^= 1;
        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::ChecksumMismatch {
                    expected: computed ^ 1,
                    computed
                }
            ))
        );
    }

    #[test]
    fn managed_fsm_ownership_rejects_an_incorrect_backlink() {
        let field = 5;
        let value = 0x123u64;
        let reason = "the section list belongs to the header at 291, not to the header at 1536";
        let (mut file, used) = managed_fsm_image(64);
        let at = usize::try_from(MANAGED_FSSE_AT).unwrap();
        file[at + field..at + field + 8].copy_from_slice(&value.to_le_bytes());
        crate::checksum::stamp_trailing(&mut file, at, usize::try_from(used).unwrap());
        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::InvalidFreeSpaceManager(reason.into())
            ))
        );
    }

    #[rstest::rstest]
    #[case::used_bytes(20)]
    #[case::reserved_bytes(30)]
    fn managed_fsm_ownership_requires_the_complete_allocated_section_list(#[case] present: usize) {
        let (mut file, used) = managed_fsm_image(64);
        file.truncate(usize::try_from(MANAGED_FSSE_AT).unwrap() + present);
        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::UnexpectedEof {
                    expected: usize::try_from(MANAGED_FSSE_AT + used + 64).unwrap(),
                    available: file.len(),
                }
            ))
        );
    }

    #[rstest::rstest]
    #[case::direct_block(false)]
    #[case::manager_header(true)]
    fn managed_fsm_ownership_rejects_overlapping_reserved_capacity(#[case] manager_header: bool) {
        let (mut file, used) = managed_fsm_image(64);
        if manager_header {
            let list = file[usize::try_from(MANAGED_FSSE_AT).unwrap()
                ..usize::try_from(MANAGED_FSSE_AT + used).unwrap()]
                .to_vec();
            place(&mut file, 0x500, &list);
            let at = usize::try_from(MANAGED_FSM_AT).unwrap();
            file[at + 54..at + 62].copy_from_slice(&0x500u64.to_le_bytes());
            file[at + 70..at + 78].copy_from_slice(&512u64.to_le_bytes());
            crate::checksum::stamp_trailing(&mut file, at, 82);
            assert_eq!(
                collect_storage(&file),
                Err(FractalHeapStorageError::InvalidStorage)
            );
            return;
        }
        let mut header = FractalHeapHeader::parse_bytes(
            &file[usize::try_from(STORAGE_HEAP_AT).unwrap()..],
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .unwrap();
        header.root_block_address = StoredAddress::new(MANAGED_FSSE_AT + used);
        place_storage_header(&mut file, &header);
        place(
            &mut file,
            header.root_block_address.get(),
            &direct_block(0, 128),
        );
        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::InvalidStorage)
        );
    }

    #[rstest::rstest]
    #[case::file_client(5, 1, "the client ID is 1, not 0, the ID of the fractal heap client")]
    #[case::wrong_heap_width(44, 15, "the heap geometry and manager header fields disagree")]
    #[case::section_count(14, 2, "the heap geometry and manager header fields disagree")]
    fn managed_fsm_ownership_rejects_inconsistent_header_fields(
        #[case] field: usize,
        #[case] value: u8,
        #[case] reason: &str,
    ) {
        let (mut file, _) = managed_fsm_image(0);
        let at = usize::try_from(MANAGED_FSM_AT).unwrap();
        file[at + field] = value;
        crate::checksum::stamp_trailing(&mut file, at, 82);
        assert_eq!(
            collect_storage(&file),
            Err(FractalHeapStorageError::Format(
                FormatError::InvalidFreeSpaceManager(reason.into())
            ))
        );
    }

    fn managed_fsm_image(spare: u64) -> (Vec<u8>, u64) {
        let header = FractalHeapHeader {
            managed_block_free_space_manager_address: StoredAddress::new(MANAGED_FSM_AT),
            root_block_address: StoredAddress::new(STORAGE_ROOT_AT),
            managed_space: 128,
            allocated_managed_space: 128,
            free_space_in_managed_blocks: 16,
            managed_objects_count: 1,
            ..storage_header()
        };
        let mut list = free_space::SECTIONS_SIGNATURE.to_vec();
        list.push(FSM_VERSION);
        list.extend_from_slice(&MANAGED_FSM_AT.to_le_bytes());
        list.push(1);
        list.extend_from_slice(&16u16.to_le_bytes());
        list.extend_from_slice(&18u16.to_le_bytes());
        list.push(FractalHeapSectionClass::Single as u8);
        list.extend_from_slice(&crate::checksum::jenkins_lookup3(&list).to_le_bytes());
        let used = u64::try_from(list.len()).unwrap();
        let mut manager = free_space::SIGNATURE.to_vec();
        manager.extend_from_slice(&[FSM_VERSION, FRACTAL_HEAP_FSM_CLIENT_ID]);
        for value in [16u64, 1, 1, 0] {
            manager.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            FRACTAL_HEAP_SECTION_CLASSES,
            FSM_SHRINK_PERCENT,
            FSM_EXPAND_PERCENT,
            16,
        ] {
            manager.extend_from_slice(&value.to_le_bytes());
        }
        manager.extend_from_slice(&1024u64.to_le_bytes());
        manager.extend_from_slice(&MANAGED_FSSE_AT.to_le_bytes());
        manager.extend_from_slice(&used.to_le_bytes());
        manager.extend_from_slice(&(used + spare).to_le_bytes());
        manager.extend_from_slice(&crate::checksum::jenkins_lookup3(&manager).to_le_bytes());
        let mut file = vec![0; 0x1000];
        place_storage_header(&mut file, &header);
        place(&mut file, STORAGE_ROOT_AT, &direct_block(0, 128));
        place(&mut file, MANAGED_FSM_AT, &manager);
        place(&mut file, MANAGED_FSSE_AT, &list);
        (file, used)
    }

    const MANAGED_FSM_AT: u64 = 0x600;
    const MANAGED_FSSE_AT: u64 = 0x700;
    // "Free-space Index" of the format specification, version 4.0, defines version 0 and client 0.
    const FSM_VERSION: u8 = 0;
    const FRACTAL_HEAP_FSM_CLIENT_ID: u8 = 0;
    // `H5HF__space_start` in H5HFspace.c, HDF5 2.2.0, registers four semantic section classes.
    const FRACTAL_HEAP_SECTION_CLASSES: u16 = 4;
    // `H5HF_FSPACE_SHRINK` and `H5HF_FSPACE_EXPAND` in H5HFspace.c, HDF5 2.2.0, define the defaults.
    const FSM_SHRINK_PERCENT: u16 = 80;
    const FSM_EXPAND_PERCENT: u16 = 120;

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
        let node_len = u64::try_from(NODE_SIZE).unwrap();
        let mut expected = vec![
            allocation(
                STORAGE_HEAP_AT,
                u64::try_from(FractalHeapHeader::serialized_size(
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                ))
                .unwrap(),
                FractalHeapAllocation::Header,
            ),
            allocation(
                TREE,
                tree_header_len,
                FractalHeapAllocation::HugeObjectIndex,
            ),
            allocation(ROOT, node_len, FractalHeapAllocation::HugeObjectIndex),
            allocation(LEFT, node_len, FractalHeapAllocation::HugeObjectIndex),
            allocation(RIGHT, node_len, FractalHeapAllocation::HugeObjectIndex),
            allocation(PAYLOAD_A, 31, FractalHeapAllocation::HugeObject),
            allocation(PAYLOAD_B, 37, FractalHeapAllocation::HugeObject),
            allocation(PAYLOAD_C, 41, FractalHeapAllocation::HugeObject),
        ];
        expected.sort_unstable_by_key(|&(extent, _)| extent);

        assert_eq!(storage.allocations(), expected);
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

        assert_eq!(
            storage.allocations(),
            &[
                allocation(
                    STORAGE_HEAP_AT,
                    u64::try_from(FractalHeapHeader::serialized_size(
                        OffsetWidth::Eight,
                        LengthWidth::Eight,
                    ))
                    .unwrap(),
                    FractalHeapAllocation::Header,
                ),
                allocation(
                    TREE,
                    u64::try_from(btree_v2_header_size(OffsetWidth::Eight, LengthWidth::Eight))
                        .unwrap(),
                    FractalHeapAllocation::HugeObjectIndex,
                ),
                allocation(
                    LEAF,
                    u64::try_from(NODE_SIZE).unwrap(),
                    FractalHeapAllocation::HugeObjectIndex,
                ),
                allocation(PAYLOAD, 29, FractalHeapAllocation::HugeObject),
            ]
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
