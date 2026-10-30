//! Proves the complete set of file allocations owned by supported fractal heaps.
//!
//! The ownership walk is read-only and operates through [`Source`]. It uses the validated
//! [`FractalHeapBlockLayout`] for managed-block references. Checksummed direct blocks require
//! complete checksum validation before the walker records their ownership.
//! For huge objects, the walker reads the tracker metadata and accounts for encoded payload
//! extents. It compares the record count and summed decoded sizes with the heap header.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_core::__private::FileRegion;
#[cfg(test)]
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
#[cfg(test)]
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT_DIRECT;
use hdf5_pure_format::__private::BTreeV2Layout;
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
#[cfg(test)]
use hdf5_pure_format::__private::HugeObjectRecord;
use hdf5_pure_format::__private::HugeObjectRecordContext;
use hdf5_pure_format::__private::HugeObjectRecordDecoder;
use hdf5_pure_format::__private::HugeObjectTrackerRecord;

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

#[cfg(all(test, feature = "deflate"))]
test_util_hdf5::hdf5_bundled_tests! {
    #[cfg(test)]
    mod native_crosschecks;
}

#[cfg(test)]
const FORMAT_WIDTHS: FormatWidths = FormatWidths {
    offsets: OffsetWidth::Eight,
    lengths: LengthWidth::Eight,
};

/// Describes a failure to prove complete ownership of a fractal heap's file storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FractalHeapStorageError {
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
/// The walker decodes huge-object tracker types 1 through 4 and records payload extents at their
/// encoded lengths. It checks the header's huge-object size total against the sum of decoded
/// lengths.
///
/// # Errors
///
/// Returns [`FractalHeapStorageError::UnsupportedFilter`] if a filtered direct block has an active
/// filter this crate does not decode. Returns [`FractalHeapStorageError::InvalidStorage`]
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

/// Records the huge-object tracker allocations and encoded payload extents.
///
/// # Errors
///
/// Returns [`FractalHeapStorageError::InvalidStorage`] if record counts or decoded-size totals
/// disagree, numeric keys repeat, or the tracker encoding differs from the heap's parameters.
/// Returns [`FractalHeapStorageError::Format`] for parser, source, or extent errors.
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
    let header = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(source),
        tree_address.get(),
        widths,
    )?;
    let decoder = HugeObjectRecordContext::new(heap, widths)?
        .resolve(&header)
        .map_err(|_error| FractalHeapStorageError::InvalidStorage)?;

    if header.total_records != heap.huge_objects_count {
        return Err(FractalHeapStorageError::InvalidStorage);
    }
    let nodes = BTreeV2Layout::new(&header, widths.offsets)?;
    allocations.push((
        extent_in_source(
            source,
            tree_address.get(),
            hdf5_pure_format::__private::btree_v2_header_size(widths.offsets, widths.lengths)
                .to_u64(),
        )?,
        FractalHeapAllocation::HugeObjectIndex,
    ));
    HugeStorageWalk {
        source,
        decoder,
        allocations,
        keys: BTreeSet::new(),
        payload_bytes: 0,
        count: 0,
    }
    .collect(&header, &nodes, heap.huge_objects_size)
}

/// Accumulates huge-object allocations, numeric keys, and decoded-size totals.
struct HugeStorageWalk<'a, 'e, S: Source + ?Sized> {
    source: &'a S,
    decoder: HugeObjectRecordDecoder,
    allocations: &'e mut Vec<(Extent, FractalHeapAllocation)>,
    keys: BTreeSet<u64>,
    /// The sum of decoded object sizes from the tracker records.
    payload_bytes: u64,
    count: u64,
}

impl<S: Source + ?Sized> HugeStorageWalk<'_, '_, S> {
    /// Collects complete index allocations and checks record and decoded-size totals.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] if subtree counts or final totals
    /// disagree, and the errors of [`record`](Self::record) or parsing a tracker node otherwise.
    fn collect(
        mut self,
        header: &BTreeV2Header,
        nodes: &BTreeV2Layout,
        decoded_size: u64,
    ) -> Result<(), FractalHeapStorageError> {
        let mut pending: Vec<_> = nodes.root().into_iter().collect();
        let mut visited = BTreeSet::new();
        while let Some(node) = pending.pop() {
            if !visited.insert(node.address()) {
                return Err(FormatError::InvalidBTreeV2Signature.into());
            }
            let Self {
                source,
                decoder: _,
                allocations,
                keys: _,
                payload_bytes: _,
                count,
            } = &mut self;
            let allocation =
                extent_in_source(*source, node.address().get(), u64::from(header.node_size))?;
            let bytes = source.read_metadata(allocation.start(), allocation.len().to_usize()?)?;
            let view = nodes.parse(&bytes, node)?;
            allocations.push((allocation, FractalHeapAllocation::HugeObjectIndex));
            *count = count
                .checked_add(view.record_count().to_u64())
                .ok_or(FractalHeapStorageError::InvalidStorage)?;
            for index in 0..view.record_count() {
                self.record(view.record(index)?)?;
            }
            let mut total = view.record_count().to_u64();
            for index in 0..=view.record_count() {
                if let Some(child) = view.child(index)? {
                    total = total
                        .checked_add(child.total_records())
                        .ok_or(FractalHeapStorageError::InvalidStorage)?;
                    pending.push(child);
                }
            }
            if total != node.total_records() {
                return Err(FractalHeapStorageError::InvalidStorage);
            }
        }
        if self.count != header.total_records || self.payload_bytes != decoded_size {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        Ok(())
    }

    /// Records one encoded payload extent and adds its decoded size to the total.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] if a numeric key repeats or the
    /// decoded-size sum overflows. Returns [`FractalHeapStorageError::Format`] if parsing the
    /// record or bounding its complete encoded allocation fails.
    fn record(&mut self, bytes: &[u8]) -> Result<(), FractalHeapStorageError> {
        let Self {
            source,
            decoder,
            allocations,
            keys,
            payload_bytes,
            count: _,
        } = self;
        let (key, address, encoded_size, decoded_size) = match decoder.parse(bytes)? {
            HugeObjectTrackerRecord::Indirect(record) => (
                Some(record.id),
                record.address,
                record.length,
                record.length,
            ),
            HugeObjectTrackerRecord::FilteredIndirect(record) => {
                let location = record.location();
                (
                    Some(record.id()),
                    location.address(),
                    location.encoded_size(),
                    location.decoded_size(),
                )
            }
            HugeObjectTrackerRecord::Direct(record) => {
                (None, record.address, record.length, record.length)
            }
            HugeObjectTrackerRecord::FilteredDirect(record) => {
                let location = record.location();
                (
                    None,
                    location.address(),
                    location.encoded_size(),
                    location.decoded_size(),
                )
            }
        };
        if key.is_some_and(|key| !keys.insert(key)) {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        *payload_bytes = payload_bytes
            .checked_add(decoded_size)
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        allocations.push((
            extent_in_source(*source, address.get(), encoded_size)?,
            FractalHeapAllocation::HugeObject,
        ));
        Ok(())
    }
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
        return Err(FormatError::UnexpectedSourceEof {
            expected: extent.end(),
            available: source.len(),
        }
        .into());
    }
    Ok(extent)
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::ToString;
    use core::cell::RefCell;

    use hdf5_pure_format::__private;
    use hdf5_pure_format::__private::FilterDescription;
    use hdf5_pure_format::__private::FilterPipeline;
    use hdf5_pure_format::__private::FractalHeapFiltering;
    use hdf5_pure_format::__private::FractalHeapSectionClass;

    use test_util::btree_v2;
    use test_util::fractal_heap;
    use test_util::free_space;
    use test_util::widths::Widths;
    use test_util_hdf5::heap_allocations::{AllocationKind, Heap};

    use crate::access_mode::AccessMode;
    use crate::address::OpenedImageContext;
    use crate::fractal_heap::HeapObjectReader;
    use crate::source::BaseOffsetSource;
    use crate::test_data;
    use crate::{AttrValue, FileBuilder};

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
        block.push(fractal_heap::BLOCK_VERSION);
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
            self.reads.borrow_mut().push((offset, bytes.len()));
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

    #[rstest::rstest]
    #[case::direct_root(false)]
    #[case::nested_indirect_root(true)]
    fn pure_written_managed_heaps_have_exact_allocations(#[case] nested: bool) {
        let mut builder = FileBuilder::new();
        for index in 0..if nested { 12 } else { 9 } {
            let value = if nested {
                AttrValue::AsciiString(
                    char::from(b'a' + u8::try_from(index).unwrap())
                        .to_string()
                        .repeat(64_000),
                )
            } else {
                AttrValue::I64(index)
            };
            let attr_name = if nested {
                format!("a{index:04}")
            } else {
                format!("a{index}")
            };
            builder.set_attr(&attr_name, value);
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        let file = builder.finish().unwrap();
        let heaps = fractal_heap::header_offsets(&file);
        assert_eq!(heaps.len(), 1);
        let address = u64::try_from(heaps[0]).unwrap();
        let header = FractalHeapHeader::parse(&file, heaps[0], 8, 8).unwrap();
        assert!(header.managed_objects_count > 0);
        assert_eq!(header.huge_objects_count, 0);
        assert_eq!(
            header.managed_block_free_space_manager_address.get(),
            u64::MAX
        );
        assert_eq!(header.current_rows_in_root_indirect_block > 0, nested);
        let inspected = Heap::new(&file, address, &header);
        if nested {
            assert!(
                inspected
                    .allocations
                    .iter()
                    .filter(|allocation| allocation.kind() == AllocationKind::IndirectBlock)
                    .count()
                    >= 2,
                "the fixture contains a nested indirect block"
            );
        }
        let expected = inspected.allocations.iter().map(|item| {
            let (address, length) = item.extent();
            allocation(
                address,
                length,
                match item.kind() {
                    AllocationKind::Header => FractalHeapAllocation::Header,
                    AllocationKind::DirectBlock => FractalHeapAllocation::DirectBlock,
                    AllocationKind::IndirectBlock => FractalHeapAllocation::IndirectBlock,
                    kind => panic!("unexpected managed fixture allocation: {kind:?}"),
                },
            )
        });
        let actual = collect_fractal_heap_storage(&file, StoredAddress::new(address)).unwrap();
        assert!(
            actual.allocations().iter().copied().eq(expected),
            "{:?} != {:?}",
            actual.allocations(),
            inspected.allocations
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
                FormatError::UnexpectedSourceEof {
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
                FormatError::UnexpectedSourceEof {
                    expected: 0x780 + block_len,
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

    #[rstest::rstest]
    fn filtered_huge_ownership_uses_encoded_extents_without_reading_payloads(
        #[values(false, true)] direct: bool,
        #[values(false, true)] internal: bool,
        #[values(false, true)] managed: bool,
        #[values(0, 512)] base: usize,
    ) {
        let fixture = FilteredHugeStorage::new(direct, internal, managed);
        let mut physical = vec![0; base];
        physical.extend_from_slice(&fixture.bytes);
        let source = PrefixSource {
            bytes: &physical,
            reads: RefCell::new(Vec::new()),
        };
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base.to_u64()));
        let storage = collect_fractal_heap_storage_from_source(
            &BaseOffsetSource {
                inner: &source,
                base: addresses.effective_base(),
            },
            addresses,
            StoredAddress::new(STORAGE_HEAP_AT),
            FORMAT_WIDTHS,
        )
        .unwrap();
        let header_len = fixture.header_bytes().len().to_u64();
        let mut expected = vec![
            allocation(STORAGE_HEAP_AT, header_len, FractalHeapAllocation::Header),
            allocation(
                FILTERED_HUGE_TREE,
                __private::btree_v2_header_size(OffsetWidth::Eight, LengthWidth::Eight).to_u64(),
                FractalHeapAllocation::HugeObjectIndex,
            ),
            allocation(
                FILTERED_HUGE_ROOT,
                FILTERED_HUGE_NODE_SIZE.to_u64(),
                FractalHeapAllocation::HugeObjectIndex,
            ),
            allocation(0x1000, 17, FractalHeapAllocation::HugeObject),
            allocation(0x1100, 19, FractalHeapAllocation::HugeObject),
            allocation(0x1200, 23, FractalHeapAllocation::HugeObject),
        ];
        if internal {
            expected.extend([
                allocation(
                    FILTERED_HUGE_LEFT,
                    FILTERED_HUGE_NODE_SIZE.to_u64(),
                    FractalHeapAllocation::HugeObjectIndex,
                ),
                allocation(
                    FILTERED_HUGE_RIGHT,
                    FILTERED_HUGE_NODE_SIZE.to_u64(),
                    FractalHeapAllocation::HugeObjectIndex,
                ),
            ]);
        }
        if managed {
            expected.push(allocation(
                STORAGE_ROOT_AT,
                128,
                FractalHeapAllocation::DirectBlock,
            ));
        }
        expected.sort_unstable_by_key(|&(extent, _)| extent);
        assert_eq!(storage.allocations(), expected);
        assert!(
            source
                .reads
                .borrow()
                .iter()
                .all(|&(at, len)| at + len.to_u64() <= base.to_u64() + 0x1000)
        );
    }

    #[rstest::rstest]
    #[case::count(|fixture: &mut FilteredHugeStorage| fixture.header.huge_objects_count = 2)]
    #[case::missing_tracker(|fixture: &mut FilteredHugeStorage| fixture.header.btree_huge_objects_address = StoredAddress::new(u64::MAX))]
    #[case::encoded_accounting(|fixture: &mut FilteredHugeStorage| fixture.header.huge_objects_size = 17 + 19 + 23)]
    #[case::decoded_accounting(|fixture: &mut FilteredHugeStorage| fixture.header.huge_objects_size += 1)]
    #[case::duplicate_payload(|fixture: &mut FilteredHugeStorage| fixture.records[1][..FILTERED_HUGE_ENCODED_AT].copy_from_slice(&0x1000u64.to_le_bytes()))]
    #[case::overlapping_payload(|fixture: &mut FilteredHugeStorage| fixture.records[1][..FILTERED_HUGE_ENCODED_AT].copy_from_slice(&0x1001u64.to_le_bytes()))]
    #[case::metadata_overlap(|fixture: &mut FilteredHugeStorage| fixture.records[0][..FILTERED_HUGE_ENCODED_AT].copy_from_slice(&FILTERED_HUGE_ROOT.to_le_bytes()))]
    #[case::decoded_total_overflow(|fixture: &mut FilteredHugeStorage| fixture.records[0][FILTERED_HUGE_DECODED_AT..FILTERED_HUGE_KEY_AT].copy_from_slice(&u64::MAX.to_le_bytes()))]
    fn filtered_huge_ownership_rejects_inconsistent_accounting(
        #[case] damage: fn(&mut FilteredHugeStorage),
        #[values(false, true)] direct: bool,
    ) {
        let mut fixture = FilteredHugeStorage::new(direct, false, false);
        damage(&mut fixture);
        fixture.write();
        assert_eq!(
            collect_storage(&fixture.bytes),
            Err(FractalHeapStorageError::InvalidStorage)
        );
    }

    #[rstest::rstest]
    #[case::undefined(0, u64::MAX, FormatError::InvalidFractalHeapGeometry)]
    #[case::encoded_zero(FILTERED_HUGE_ENCODED_AT, 0, FormatError::InvalidFractalHeapGeometry)]
    #[case::decoded_zero(FILTERED_HUGE_DECODED_AT, 0, FormatError::InvalidFractalHeapGeometry)]
    #[case::mask(FILTERED_HUGE_MASK_AT, 2, FormatError::InvalidFractalHeapGeometry)]
    #[case::skipped_size(FILTERED_HUGE_MASK_AT, 1, FormatError::InvalidFractalHeapGeometry)]
    #[case::overflow(0, u64::MAX - 1, FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 17 })]
    fn filtered_huge_ownership_rejects_invalid_record_fields(
        #[case] at: usize,
        #[case] value: u64,
        #[case] expected: FormatError,
        #[values(false, true)] direct: bool,
    ) {
        let mut fixture = FilteredHugeStorage::new(direct, false, false);
        let width = if at == FILTERED_HUGE_MASK_AT {
            core::mem::size_of::<u32>()
        } else {
            core::mem::size_of::<u64>()
        };
        fixture.records[0][at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
        fixture.write();
        assert_eq!(
            collect_storage(&fixture.bytes),
            Err(FractalHeapStorageError::Format(expected))
        );
    }

    #[test]
    fn filtered_huge_ownership_rejects_duplicate_numeric_keys() {
        let mut fixture = FilteredHugeStorage::new(false, false, false);
        fixture.records[1][FILTERED_HUGE_KEY_AT..].copy_from_slice(&1u64.to_le_bytes());
        fixture.write();
        assert_eq!(
            collect_storage(&fixture.bytes),
            Err(FractalHeapStorageError::InvalidStorage)
        );
    }

    #[rstest::rstest]
    #[case::record_type(|fixture: &mut FilteredHugeStorage| fixture.tree_type = BTREE_V2_HUGE_OBJECT)]
    #[case::size(|fixture: &mut FilteredHugeStorage| fixture.record_size += 1)]
    #[case::tree_count(|fixture: &mut FilteredHugeStorage| fixture.tree_count += 1)]
    fn filtered_huge_ownership_rejects_inconsistent_tracker_metadata(
        #[case] damage: fn(&mut FilteredHugeStorage),
        #[values(false, true)] direct: bool,
    ) {
        let mut fixture = FilteredHugeStorage::new(direct, false, false);
        damage(&mut fixture);
        fixture.write();
        assert_eq!(
            collect_storage(&fixture.bytes),
            Err(FractalHeapStorageError::InvalidStorage)
        );
    }

    #[rstest::rstest]
    fn filtered_huge_ownership_rejects_inconsistent_leaf_counts(
        #[values(false, true)] direct: bool,
    ) {
        let mut fixture = FilteredHugeStorage::new(direct, false, false);
        fixture.root_count = 2;
        fixture.write();
        assert_eq!(
            collect_storage(&fixture.bytes),
            Err(FractalHeapStorageError::Format(
                FormatError::InvalidBTreeV2Signature
            ))
        );
    }

    #[rstest::rstest]
    #[case::consistent(3, None)]
    #[case::incorrect_sum(4, Some(FractalHeapStorageError::InvalidStorage))]
    #[case::impossible_child(
        2,
        Some(FractalHeapStorageError::Format(FormatError::InvalidBTreeV2Signature))
    )]
    fn filtered_huge_ownership_validates_subtree_record_counts(
        #[case] child_total: u64,
        #[case] expected: Option<FractalHeapStorageError>,
        #[values(false, true)] direct: bool,
    ) {
        let fixture = FilteredHugeStorage::two_levels(direct, child_total);
        let result = collect_storage(&fixture.bytes);
        if let Some(expected) = expected {
            assert_eq!(result.unwrap_err(), expected);
        } else {
            let storage = result.unwrap();
            assert_eq!(storage.allocations().len(), 16);
            assert_eq!(
                storage
                    .allocations()
                    .iter()
                    .filter(|(_, kind)| *kind == FractalHeapAllocation::HugeObject)
                    .count(),
                7
            );
        }
    }

    #[rstest::rstest]
    #[case::payload(0x1200 + 22, 0x1200 + 23)]
    #[case::node_padding(FILTERED_HUGE_ROOT + FILTERED_HUGE_NODE_SIZE as u64 - 1, FILTERED_HUGE_ROOT + FILTERED_HUGE_NODE_SIZE as u64)]
    fn filtered_huge_ownership_requires_complete_allocations(
        #[case] present: u64,
        #[case] expected: u64,
        #[values(false, true)] direct: bool,
    ) {
        let mut fixture = FilteredHugeStorage::new(direct, false, false);
        fixture.bytes.truncate(present.to_usize().unwrap());
        assert_eq!(
            collect_storage(&fixture.bytes),
            Err(FractalHeapStorageError::Format(
                FormatError::UnexpectedSourceEof {
                    expected,
                    available: present
                }
            ))
        );
    }

    struct FilteredHugeStorage {
        header: FractalHeapHeader,
        records: Vec<Vec<u8>>,
        tree_type: u8,
        record_size: u16,
        tree_count: u64,
        root_count: u16,
        internal: bool,
        bytes: Vec<u8>,
    }

    impl FilteredHugeStorage {
        fn two_levels(direct: bool, child_total: u64) -> Self {
            let mut fixture = Self::new(direct, true, false);
            fixture.bytes.resize(0x3000, 0);
            fixture.records = (0..7u64)
                .map(|index| {
                    let mut record = fixture.records[0].clone();
                    record[..FILTERED_HUGE_ENCODED_AT]
                        .copy_from_slice(&(0x1000 + index * 0x100).to_le_bytes());
                    if !direct {
                        record[FILTERED_HUGE_KEY_AT..].copy_from_slice(&(index + 1).to_le_bytes());
                    }
                    record
                })
                .collect();
            fixture.header.huge_objects_count = 7;
            fixture.header.huge_objects_size = 7000;
            place_filtered_storage_header(&mut fixture.bytes, &fixture.header);
            let header = btree_v2::Header::new(
                fixture.tree_type,
                fixture.record_size,
                FILTERED_HUGE_ROOT,
                1,
            )
            .node_size(u32::try_from(FILTERED_HUGE_NODE_SIZE).unwrap())
            .depth(2)
            .total_records(7)
            .build(Widths::EIGHT);
            place(&mut fixture.bytes, FILTERED_HUGE_TREE, &header);
            let root = btree_v2::internal(
                fixture.tree_type,
                &[fixture.records[3].clone()],
                &[
                    btree_v2::Child {
                        address: FILTERED_HUGE_LEFT,
                        records: 1,
                        records_width: 1,
                        subtree: Some(btree_v2::Subtree {
                            records: child_total,
                            width: 1,
                        }),
                    },
                    btree_v2::Child {
                        address: FILTERED_HUGE_RIGHT,
                        records: 1,
                        records_width: 1,
                        subtree: Some(btree_v2::Subtree {
                            records: 3,
                            width: 1,
                        }),
                    },
                ],
                Widths::EIGHT,
            );
            place_btree_node(
                &mut fixture.bytes,
                FILTERED_HUGE_ROOT,
                &root,
                FILTERED_HUGE_NODE_SIZE,
            );
            for (address, middle, left, right) in [
                (FILTERED_HUGE_LEFT, 1, 0x2000, 0x2200),
                (FILTERED_HUGE_RIGHT, 5, 0x2400, 0x2600),
            ] {
                let node = btree_v2::internal(
                    fixture.tree_type,
                    &[fixture.records[middle].clone()],
                    &[
                        btree_v2::Child {
                            address: left,
                            records: 1,
                            records_width: 1,
                            subtree: None,
                        },
                        btree_v2::Child {
                            address: right,
                            records: 1,
                            records_width: 1,
                            subtree: None,
                        },
                    ],
                    Widths::EIGHT,
                );
                place_btree_node(&mut fixture.bytes, address, &node, FILTERED_HUGE_NODE_SIZE);
                place_btree_node(
                    &mut fixture.bytes,
                    left,
                    &btree_v2::leaf(fixture.tree_type, &[fixture.records[middle - 1].clone()]),
                    FILTERED_HUGE_NODE_SIZE,
                );
                place_btree_node(
                    &mut fixture.bytes,
                    right,
                    &btree_v2::leaf(fixture.tree_type, &[fixture.records[middle + 1].clone()]),
                    FILTERED_HUGE_NODE_SIZE,
                );
            }
            fixture
        }

        fn new(direct: bool, internal: bool, managed: bool) -> Self {
            let mut header = filtered_header(
                FractalHeapHeader {
                    heap_id_length: if direct {
                        FILTERED_HUGE_DIRECT_ID_LEN
                    } else {
                        7
                    },
                    btree_huge_objects_address: StoredAddress::new(FILTERED_HUGE_TREE),
                    huge_objects_count: 3,
                    huge_objects_size: 1000 + 2000 + 3000,
                    ..storage_header()
                },
                if managed { 128 } else { 0 },
                test_data::filter_pipeline(vec![FilterDescription {
                    filter_id: test_data::FILTER_BZIP2,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                }]),
            );
            if managed {
                header.root_block_address = StoredAddress::new(STORAGE_ROOT_AT);
                header.managed_space = 128;
                header.allocated_managed_space = 128;
                header.managed_objects_count = 1;
                let FractalHeapFiltering::Filtered {
                    root_filter_mask, ..
                } = &mut header.filtering
                else {
                    panic!("expected filtered heap")
                };
                *root_filter_mask = 1;
            }
            let records = [
                (0x1000u64, 17u64, 1000u64, 1u64),
                (0x1100, 19, 2000, 2),
                (0x1200, 23, 3000, 3),
            ]
            .map(|(address, encoded, decoded, key)| {
                let mut record = address.to_le_bytes().to_vec();
                record.extend_from_slice(&encoded.to_le_bytes());
                record.extend_from_slice(&0u32.to_le_bytes());
                record.extend_from_slice(&decoded.to_le_bytes());
                if !direct {
                    record.extend_from_slice(&key.to_le_bytes());
                }
                record
            })
            .to_vec();
            let mut fixture = Self {
                tree_type: if direct {
                    hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT
                } else {
                    hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT
                },
                record_size: u16::try_from(records[0].len()).unwrap(),
                tree_count: 3,
                root_count: if internal { 1 } else { 3 },
                header,
                records,
                internal,
                bytes: vec![0; 0x1300],
            };
            fixture.write();
            fixture
        }

        fn write(&mut self) {
            place_filtered_storage_header(&mut self.bytes, &self.header);
            place(
                &mut self.bytes,
                FILTERED_HUGE_TREE,
                &btree_v2::Header::new(
                    self.tree_type,
                    self.record_size,
                    FILTERED_HUGE_ROOT,
                    self.root_count,
                )
                .node_size(u32::try_from(FILTERED_HUGE_NODE_SIZE).unwrap())
                .depth(u16::from(self.internal))
                .total_records(self.tree_count)
                .build(Widths::EIGHT),
            );
            if self.internal {
                let root = btree_v2::internal(
                    self.tree_type,
                    &[self.records[1].clone()],
                    &[
                        btree_v2::Child {
                            address: FILTERED_HUGE_LEFT,
                            records: 1,
                            records_width: 1,
                            subtree: None,
                        },
                        btree_v2::Child {
                            address: FILTERED_HUGE_RIGHT,
                            records: 1,
                            records_width: 1,
                            subtree: None,
                        },
                    ],
                    Widths::EIGHT,
                );
                place_btree_node(
                    &mut self.bytes,
                    FILTERED_HUGE_ROOT,
                    &root,
                    FILTERED_HUGE_NODE_SIZE,
                );
                place_btree_node(
                    &mut self.bytes,
                    FILTERED_HUGE_LEFT,
                    &btree_v2::leaf(self.tree_type, &[self.records[0].clone()]),
                    FILTERED_HUGE_NODE_SIZE,
                );
                place_btree_node(
                    &mut self.bytes,
                    FILTERED_HUGE_RIGHT,
                    &btree_v2::leaf(self.tree_type, &[self.records[2].clone()]),
                    FILTERED_HUGE_NODE_SIZE,
                );
            } else {
                place_btree_node(
                    &mut self.bytes,
                    FILTERED_HUGE_ROOT,
                    &btree_v2::leaf(
                        self.tree_type,
                        &self.records[..usize::from(self.root_count)],
                    ),
                    FILTERED_HUGE_NODE_SIZE,
                );
            }
            if !self
                .header
                .root_block_address
                .is_undefined(OffsetWidth::Eight)
            {
                place(&mut self.bytes, STORAGE_ROOT_AT, &direct_block(0, 128));
            }
        }

        fn header_bytes(&self) -> Vec<u8> {
            let mut bytes = vec![0; 0x400];
            let len = place_filtered_storage_header(&mut bytes, &self.header);
            bytes[STORAGE_HEAP_AT.to_usize().unwrap()..(STORAGE_HEAP_AT + len).to_usize().unwrap()]
                .to_vec()
        }
    }

    /// The encoded-length offset in a filtered huge record with eight-byte addresses.
    ///
    /// The fields are defined in "Version 2 B-trees" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
    const FILTERED_HUGE_ENCODED_AT: usize = OffsetWidth::Eight.get() as usize;
    /// The filter-mask offset at the widths used by [`FILTERED_HUGE_ENCODED_AT`].
    const FILTERED_HUGE_MASK_AT: usize =
        FILTERED_HUGE_ENCODED_AT + LengthWidth::Eight.get() as usize;
    /// The decoded-size offset at the widths used by [`FILTERED_HUGE_ENCODED_AT`].
    const FILTERED_HUGE_DECODED_AT: usize = FILTERED_HUGE_MASK_AT + core::mem::size_of::<u32>();
    /// The numeric-key offset in an indirect record at these widths.
    const FILTERED_HUGE_KEY_AT: usize =
        FILTERED_HUGE_DECODED_AT + LengthWidth::Eight.get() as usize;
    /// The direct filtered huge-ID length at these widths, including its initial ID byte.
    ///
    /// The ID is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    const FILTERED_HUGE_DIRECT_ID_LEN: u16 = 1 + FILTERED_HUGE_KEY_AT as u16;

    const FILTERED_HUGE_TREE: u64 = 0x600;
    const FILTERED_HUGE_ROOT: u64 = 0x800;
    const FILTERED_HUGE_LEFT: u64 = 0xa00;
    const FILTERED_HUGE_RIGHT: u64 = 0xc00;
    const FILTERED_HUGE_NODE_SIZE: usize = 256;

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
        let checksum = fractal_heap::DIRECT_BLOCK_CHECKSUM_AT..fractal_heap::DIRECT_BLOCK_DATA_AT;
        let stored = u32::from_le_bytes(block_a[checksum.clone()].try_into().unwrap());
        match damage {
            FilteredDamage::WrongOwner => {
                let owner = fractal_heap::DIRECT_BLOCK_OWNER_AT;
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
                FractalHeapStorageError::Format(FormatError::UnexpectedSourceEof {
                    expected: (file.len() + 1).to_u64(),
                    available: file.len().to_u64(),
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

    #[cfg(feature = "deflate")]
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
        let bytes = fractal_heap::with_filtering(
            &unfiltered,
            Widths::EIGHT,
            *root_direct_block_size,
            *root_filter_mask,
            &pipeline.serialize().unwrap(),
        );
        place(file, STORAGE_HEAP_AT, &bytes);
        len_u64(bytes.len())
    }

    #[cfg(feature = "deflate")]
    fn checksummed_direct_block(heap_offset: u64, size: usize) -> Vec<u8> {
        let mut block = direct_block(heap_offset, size);
        block[fractal_heap::DIRECT_BLOCK_DATA_AT..].fill(0x5A);
        let checksum = crate::checksum::jenkins_lookup3(&block);
        block[fractal_heap::DIRECT_BLOCK_CHECKSUM_AT..fractal_heap::DIRECT_BLOCK_DATA_AT]
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
                FormatError::UnexpectedSourceEof {
                    expected: MANAGED_FSSE_AT + used + 64,
                    available: file.len().to_u64(),
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
        let tree_header_len = u64::try_from(__private::btree_v2_header_size(
            OffsetWidth::Eight,
            LengthWidth::Eight,
        ))
        .unwrap();
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
                    u64::try_from(__private::btree_v2_header_size(
                        OffsetWidth::Eight,
                        LengthWidth::Eight,
                    ))
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

    #[rstest::rstest]
    #[case::indexed_exact(false, 0)]
    #[case::indexed_reserved(false, 64)]
    #[case::direct_exact(true, 0)]
    #[case::direct_reserved(true, 64)]
    fn mixed_managed_huge_and_manager_allocations_are_exact(
        #[case] direct: bool,
        #[case] spare: u64,
    ) {
        const TREE: u64 = 0x800;
        const LEAF: u64 = 0x900;
        const PAYLOAD: u64 = 0xa00;
        const PAYLOAD_LEN: u64 = 37;
        const NODE_SIZE: usize = 128;
        let (mut file, used) = managed_fsm_image(spare);
        let mut header =
            FractalHeapHeader::parse(&file, usize::try_from(STORAGE_HEAP_AT).unwrap(), 8, 8)
                .unwrap();
        header.btree_huge_objects_address = StoredAddress::new(TREE);
        header.huge_objects_count = 1;
        header.huge_objects_size = PAYLOAD_LEN;
        let (tree_type, record_size, record) = if direct {
            header.heap_id_length = 17;
            let mut record = PAYLOAD.to_le_bytes().to_vec();
            record.extend_from_slice(&PAYLOAD_LEN.to_le_bytes());
            (BTREE_V2_HUGE_OBJECT_DIRECT, 16, record)
        } else {
            (
                BTREE_V2_HUGE_OBJECT,
                24,
                encoded_huge_record(PAYLOAD, PAYLOAD_LEN, 1),
            )
        };
        place_storage_header(&mut file, &header);
        place(
            &mut file,
            TREE,
            &btree_v2::Header::new(tree_type, record_size, LEAF, 1)
                .node_size(u32::try_from(NODE_SIZE).unwrap())
                .build(Widths::EIGHT),
        );
        place_btree_node(
            &mut file,
            LEAF,
            &btree_v2::leaf(tree_type, &[record]),
            NODE_SIZE,
        );
        let expected = vec![
            allocation(
                STORAGE_HEAP_AT,
                len_u64(FractalHeapHeader::serialized_size(
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                )),
                FractalHeapAllocation::Header,
            ),
            allocation(STORAGE_ROOT_AT, 128, FractalHeapAllocation::DirectBlock),
            allocation(MANAGED_FSM_AT, 82, FractalHeapAllocation::ManagedFsmHeader),
            allocation(
                MANAGED_FSSE_AT,
                used + spare,
                FractalHeapAllocation::ManagedFsmSectionList,
            ),
            allocation(
                TREE,
                len_u64(__private::btree_v2_header_size(
                    OffsetWidth::Eight,
                    LengthWidth::Eight,
                )),
                FractalHeapAllocation::HugeObjectIndex,
            ),
            allocation(
                LEAF,
                len_u64(NODE_SIZE),
                FractalHeapAllocation::HugeObjectIndex,
            ),
            allocation(PAYLOAD, PAYLOAD_LEN, FractalHeapAllocation::HugeObject),
        ];
        assert_eq!(collect_storage(&file).unwrap().allocations(), expected);
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
