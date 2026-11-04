//! Stored-address layout planning for file creation.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_format::__private::SectionInfoCounts;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::ManagerKind;

use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
pub(super) use crate::chunked_write::ArrayMeasure;
use crate::chunked_write::ChunkArrayGeometry;
use crate::chunked_write::EmptyChunkIndex;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::file_allocation::FileAllocation;
use crate::file_allocation::FileManager;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::free_space_manager;

use super::WIDTHS;

/// A stored file address and the byte length reserved there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Allocation {
    pub(super) address: StoredAddress,
    pub(super) length: u64,
}

impl Allocation {
    /// Reserves a region for the supplied allocation class.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FileAllocation::reserve`].
    fn reserve(
        allocation: &mut FileAllocation,
        length: u64,
        class: AllocationClass,
    ) -> Result<Self, FormatError> {
        let address = allocation.reserve(length, class)?;
        Ok(Self { address, length })
    }
}

/// The byte lengths of an object's header and optional dense storage.
///
/// Dense attribute and link storage include their heaps and indexes.
pub(super) struct ObjectMeasure {
    pub(super) header: u64,
    pub(super) attributes: Option<u64>,
    pub(super) links: Option<u64>,
}

impl ObjectMeasure {
    /// Reserves the header, dense attributes, and dense links in that order.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FileAllocation::reserve`].
    pub(super) fn reserve(
        self,
        allocation: &mut FileAllocation,
    ) -> Result<ObjectAllocation, FormatError> {
        let Self {
            header,
            attributes,
            links,
        } = self;
        let header = Allocation::reserve(allocation, header, AllocationClass::Metadata)?;
        let attributes = attributes
            .map(|length| Allocation::reserve(allocation, length, AllocationClass::Metadata))
            .transpose()?;
        let links = links
            .map(|length| Allocation::reserve(allocation, length, AllocationClass::Metadata))
            .transpose()?;
        Ok(ObjectAllocation {
            header,
            attributes,
            links,
        })
    }
}

/// Stored reservations for one object's header and optional dense storage.
pub(super) struct ObjectAllocation {
    pub(super) header: Allocation,
    pub(super) attributes: Option<Allocation>,
    pub(super) links: Option<Allocation>,
}

/// A dataset's payload length, storage kind, and optional array geometry.
///
/// The writer retains array block lengths separately from the chunk payload length.
pub(super) struct DataMeasure {
    /// The payload length, excluding the index when `array` is present.
    raw_length: u64,
    kind: DataKind,
    /// Address-independent geometry for an allocated chunk array.
    pub(super) array: Option<ArrayMeasure>,
}

impl DataMeasure {
    /// Records a complete data-region length.
    pub(super) fn new(raw_length: u64, kind: DataKind) -> Self {
        Self {
            raw_length,
            kind,
            array: None,
        }
    }

    /// Measures chunked storage from its payload length and optional array geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the combined payload and index length exceeds
    /// [`u64::MAX`]. Propagates array planning and length errors.
    pub(super) fn chunked(
        raw_length: u64,
        geometry: Option<ChunkArrayGeometry<'_>>,
    ) -> Result<Self, FormatError> {
        let measure = Self {
            raw_length,
            kind: DataKind::Chunked,
            array: geometry.map(ArrayMeasure::new).transpose()?,
        };
        measure.length()?;
        Ok(measure)
    }

    /// Returns whether the region uses a chunked layout.
    pub(super) fn is_chunked(&self) -> bool {
        matches!(self.kind, DataKind::Chunked)
    }

    /// Returns scalar geometry for an allocated array with no stored chunks.
    pub(super) fn empty_index(&self) -> Option<EmptyChunkIndex> {
        match &self.array {
            Some(ArrayMeasure::Empty(index)) => Some(*index),
            _ => None,
        }
    }

    /// Returns the combined payload and index length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the combined length exceeds [`u64::MAX`].
    /// Propagates errors from [`EmptyChunkIndex::length`].
    pub(super) fn length(&self) -> Result<u64, FormatError> {
        let Self {
            raw_length,
            kind: _,
            array,
        } = self;
        checked_end(
            *raw_length,
            array.as_ref().map_or(Ok(0), ArrayMeasure::length)?,
        )
    }

    /// Reserves raw storage, returning [`None`] for an empty contiguous region.
    ///
    /// A zero-length chunked region has a reservation at the current end of allocation.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::length`] and [`FileAllocation::reserve`].
    pub(super) fn reserve(
        &self,
        allocation: &mut FileAllocation,
    ) -> Result<Option<Allocation>, FormatError> {
        if self.empty_contiguous() {
            Ok(None)
        } else {
            Allocation::reserve(allocation, self.length()?, AllocationClass::Raw).map(Some)
        }
    }

    fn empty_contiguous(&self) -> bool {
        let Self {
            raw_length,
            kind,
            array: _,
        } = self;
        *raw_length == 0 && matches!(kind, DataKind::Contiguous)
    }
}

/// The storage layout used to interpret a dataset region's length.
#[derive(Clone, Copy)]
pub(super) enum DataKind {
    Chunked,
    Contiguous,
}

/// The superblock extension, final allocation end, and optional manager for a classic strategy.
pub(super) struct ClassicLayout {
    pub(super) extension: Option<Allocation>,
    pub(super) end: StoredAddress,
    pub(super) manager: Option<FileManager>,
}

impl ClassicLayout {
    /// Places the optional extension and releases unused aggregate tails.
    ///
    /// With `persist`, serializes remaining reusable space into a manager at the file's tail.
    ///
    /// # Errors
    ///
    /// Propagates reservation and cleanup errors from [`FileAllocation::reserve`],
    /// [`FileAllocation::finish`], and [`FileAllocation::finish_persisted`].
    pub(super) fn finish(
        mut allocation: FileAllocation,
        extension_len: Option<u64>,
        persist: bool,
    ) -> Result<Self, FormatError> {
        let extension = extension_len
            .map(|length| Allocation::reserve(&mut allocation, length, AllocationClass::Metadata))
            .transpose()?;
        let (end, manager) = if persist {
            allocation.finish_persisted(WIDTHS)?
        } else {
            (allocation.finish()?, None)
        };
        Ok(Self {
            extension,
            end,
            manager,
        })
    }
}

/// Free page tails around global-heap collections, grouped by manager kind.
#[derive(Default)]
pub(super) struct CollectionSections {
    pub(super) metadata: Vec<FreeSection>,
    pub(super) raw: Vec<FreeSection>,
    pub(super) large: Vec<FreeSection>,
}

/// Page-aligned global-heap reservations and the free space around them.
pub(super) struct PagedCollections {
    page: FileSpacePageSize,
    /// The first stored address available for following object metadata.
    pub(super) end: StoredAddress,
    pub(super) sections: CollectionSections,
}

impl PagedCollections {
    /// Creates an empty collection layout with its cursor at `start`.
    ///
    /// The first reservation is aligned to `page`.
    pub(super) fn new(start: StoredAddress, page: FileSpacePageSize) -> Self {
        Self {
            page,
            end: start,
            sections: CollectionSections::default(),
        }
    }

    /// Reserves a collection on a fresh page run and records the surrounding free space.
    ///
    /// Classifies the final page tail as small raw or generic large according to `length`.
    /// Free space before the collection belongs to the metadata manager.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the region or page padding exceeds [`u64::MAX`].
    pub(super) fn reserve(&mut self, length: u64) -> Result<Allocation, FormatError> {
        let Self {
            page,
            end: cursor,
            sections,
        } = self;
        let CollectionSections {
            metadata,
            raw,
            large,
        } = sections;
        let start = align(*cursor, *page)?;
        if start > *cursor {
            // The comparison orders the endpoints of this padding.
            metadata.push(FreeSection {
                addr: *cursor,
                size: start.checked_distance_from(*cursor).unwrap(),
            });
        }
        let end = start.checked_add(length)?;
        *cursor = align(end, *page)?;
        if *cursor > end {
            // Alignment advances the end, so the padding length is positive.
            let section = FreeSection {
                addr: end,
                size: cursor.checked_distance_from(end).unwrap(),
            };
            if length >= page.get() {
                large.push(section);
            } else {
                raw.push(section);
            }
        }
        Ok(Allocation {
            address: start,
            length,
        })
    }
}

/// Dataset placements, serialized managers, and region boundaries for PAGE output.
///
/// Addresses and boundaries are stored file coordinates. The caller supplies each dataset's
/// retained value through the placement callback.
pub(super) struct PagedLayout<T> {
    pub(super) extension: Allocation,
    /// Callback results in dataset measurement order.
    pub(super) data: Vec<T>,
    /// Dataset indices in the shared small-raw page run.
    pub(super) small: Vec<usize>,
    /// Dataset indices with separate page-aligned runs.
    pub(super) large: Vec<usize>,
    /// Manager header addresses in File Space Info slot order, undefined for inactive slots.
    pub(super) slots: [DecodedStoredAddress; NUM_FILE_FSM_MANAGERS],
    /// Metadata, small-raw, and generic-large managers in emission order.
    pub(super) managers: [Option<FileManager>; 3],
    /// The stored address after the extension and manager blocks, before page padding.
    pub(super) metadata_end: StoredAddress,
    /// The first stored address of the raw-data region.
    pub(super) raw_start: StoredAddress,
    /// The stored address after the small raw regions, before page padding.
    pub(super) small_end: StoredAddress,
    /// The page-aligned end of allocation.
    pub(super) end: StoredAddress,
}

impl<T> PagedLayout<T> {
    /// Plans the extension, managers, and dataset regions after the object metadata.
    ///
    /// `metadata_end` is the stored address before the extension. `sections` contains free
    /// page tails from collection placement. Small extension and manager blocks stay within
    /// one page, and blocks at least one page long start at page boundaries. With `persist`,
    /// serializes the active managers with capacity for their placement gaps.
    ///
    /// Calls `bind` once per dataset in measurement order with its index and allocation,
    /// passing [`None`] for an empty contiguous region. Small regions share a page run, and
    /// each large region starts on a page boundary. Stores callback results in [`Self::data`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a region or its padding exceeds [`u64::MAX`],
    /// or [`FormatError::Internal`] if manager capacity exceeds its section bound or serialized
    /// lengths differ from their reservations. Propagates errors from `bind` and manager
    /// serialization.
    ///
    /// # Performance
    ///
    /// Measures the supplied section sizes before retrying tail placement. Retries use retained
    /// counts and size groups, without calling `bind` or serializing managers. Serializes each
    /// active manager after placement.
    pub(super) fn new<'a>(
        page: FileSpacePageSize,
        persist: bool,
        metadata_end: StoredAddress,
        extension_length: u64,
        measurements: impl ExactSizeIterator<Item = &'a DataMeasure> + Clone,
        sections: CollectionSections,
        mut bind: impl FnMut(usize, Option<Allocation>) -> Result<T, FormatError>,
    ) -> Result<Self, FormatError> {
        let CollectionSections {
            metadata: mut meta_sections,
            raw: mut raw_sections,
            large: mut large_sections,
        } = sections;
        let mut small = Vec::new();
        let mut large = Vec::new();
        let mut small_total = 0;
        let mut large_sizes = Vec::new();
        for (index, measure) in measurements.clone().enumerate() {
            if measure.empty_contiguous() {
                continue;
            }
            let length = measure.length()?;
            if length < page.get() {
                small.push(index);
                small_total = checked_end(small_total, length)?;
            } else {
                large.push(index);
                let padding = (page.get() - length % page.get()) % page.get();
                length
                    .checked_add(padding)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: length,
                        length: padding,
                    })?;
                if padding > 0 {
                    large_sizes.push(padding);
                }
            }
        }
        large_sizes.extend(large_sections.iter().map(|section| section.size));
        let mut raw_sizes: Vec<_> = raw_sections.iter().map(|section| section.size).collect();
        let raw_tail = (page.get() - small_total % page.get()) % page.get();
        if raw_tail > 0 {
            raw_sizes.push(raw_tail);
        }
        let TailLayout {
            extension,
            locations,
            arena,
        } = TailLayout::new(
            page,
            persist,
            metadata_end,
            extension_length,
            &meta_sections,
            &raw_sizes,
            &large_sizes,
        )?;
        let MetadataArena {
            page: _,
            cursor: raw_start,
            occupied_end: metadata_end,
            metadata,
            large: padding,
        } = arena;
        meta_sections.extend(metadata);
        large_sections.extend(padding);
        let mut slots = [DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS];
        for location in locations.iter().flatten() {
            let (slot, _) = free_space_manager::manager_wire(location.kind);
            slots[slot] = DecodedStoredAddress::Defined(location.header);
        }
        let small_end = raw_start.checked_add(small_total)?;
        let mut small_cursor = raw_start;
        let mut cursor = align(small_end, page)?;
        if small_total > 0 && cursor > small_end {
            // Alignment advances the nonempty raw region past its end.
            raw_sections.push(FreeSection {
                addr: small_end,
                size: cursor.checked_distance_from(small_end).unwrap(),
            });
        }
        let mut data = Vec::with_capacity(measurements.len());
        for (index, measure) in measurements.enumerate() {
            let length = measure.length()?;
            let placed = if measure.empty_contiguous() {
                None
            } else if length < page.get() {
                let address = small_cursor;
                small_cursor = small_cursor.checked_add(length)?;
                Some(Allocation { address, length })
            } else {
                let start = align(cursor, page)?;
                let end = start.checked_add(length)?;
                cursor = align(end, page)?;
                if cursor > end {
                    // The aligned cursor follows this large allocation.
                    large_sections.push(FreeSection {
                        addr: end,
                        size: cursor.checked_distance_from(end).unwrap(),
                    });
                }
                Some(Allocation {
                    address: start,
                    length,
                })
            };
            data.push(bind(index, placed)?);
        }
        let mut managers = [None, None, None];
        for ((target, location), sections) in
            locations
                .into_iter()
                .enumerate()
                .zip([meta_sections, raw_sections, large_sections])
        {
            if let Some(location) = location {
                managers[target] = Some(location.serialize(&sections)?);
            }
        }
        Ok(Self {
            extension,
            data,
            small,
            large,
            slots,
            managers,
            metadata_end,
            raw_start,
            small_end,
            end: cursor,
        })
    }
}

/// Complete extension and manager reservations with their classified placement gaps.
struct TailLayout {
    extension: Allocation,
    locations: [Option<ManagerLocation>; 3],
    arena: MetadataArena,
}

impl TailLayout {
    /// Places complete blocks and reserves sufficient manager capacity for their free gaps.
    ///
    /// With `persist`, activates a metadata manager immediately for supplied metadata sections
    /// or gaps before the extension. Resolves required generic-manager capacity before activation
    /// for gaps introduced by manager placement. Capacity grows within bounds on counts and
    /// distinct sizes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if an address, length, or count overflows.
    /// Returns [`FormatError::Internal`] if a manager exceeds its section bound or a generic
    /// section has no active manager. Propagates section measurement errors.
    fn new(
        page: FileSpacePageSize,
        persist: bool,
        start: StoredAddress,
        extension_length: u64,
        metadata: &[FreeSection],
        raw_sizes: &[u64],
        large_sizes: &[u64],
    ) -> Result<Self, FormatError> {
        let mut arena = MetadataArena::new(page, start);
        let extension = arena.reserve(extension_length)?;
        if !persist {
            arena.finish()?;
            return Ok(Self {
                extension,
                locations: [None, None, None],
                arena,
            });
        }
        let metadata_measure = SectionMeasure::new(metadata.iter().map(|section| section.size))?;
        let large_measure = SectionMeasure::new(large_sizes.iter().copied())?;
        let mut metadata_active = !metadata.is_empty() || !arena.metadata.is_empty();
        let raw_length = (!raw_sizes.is_empty())
            .then(|| hdf5_pure_format::__private::section_info_len(WIDTHS, raw_sizes));
        let mut metadata_capacity = metadata_measure.encoded_len(core::iter::once(0))?;
        let mut large_capacity = large_measure.encoded_len(core::iter::empty())?;
        let metadata_limit = checked_end(metadata_measure.capacity(TAIL_GAP_LIMIT)?, 1)?;
        let large_limit = large_measure.capacity(TAIL_BLOCK_LIMIT)?;
        let header_length = hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS);
        let after_extension = arena;
        loop {
            let large_active = !large_sizes.is_empty()
                || !after_extension.large.is_empty()
                || [metadata_active.then_some(metadata_capacity), raw_length]
                    .into_iter()
                    .flatten()
                    .chain(metadata_active.then_some(header_length))
                    .any(|length| length >= page.get() && length % page.get() != 0);
            let mut arena = after_extension.clone();
            let locations = arena.managers([
                metadata_active.then_some(metadata_capacity),
                raw_length,
                large_active.then_some(large_capacity),
            ])?;
            arena.finish()?;
            let large_length =
                large_measure.encoded_len(arena.large.iter().map(|section| section.size))?;
            if large_length > large_limit {
                return Err(FormatError::Internal(
                    "PAGE large manager length exceeds its section bound".into(),
                ));
            }
            if !large_active && !arena.large.is_empty() {
                return Err(FormatError::Internal(
                    "PAGE placement introduces an unplanned large section".into(),
                ));
            }
            if large_active && large_length > large_capacity {
                large_capacity = large_length;
                continue;
            }
            if !metadata_active && !arena.metadata.is_empty() {
                metadata_active = true;
                continue;
            }
            if metadata_active && metadata.is_empty() && arena.metadata.is_empty() {
                metadata_capacity = checked_end(metadata_capacity, 1)?;
                if metadata_capacity > metadata_limit {
                    return Err(FormatError::Internal(
                        "PAGE metadata capacity exceeds its section bound".into(),
                    ));
                }
                continue;
            }
            let metadata_length =
                metadata_measure.encoded_len(arena.metadata.iter().map(|section| section.size))?;
            if metadata_length > metadata_limit {
                return Err(FormatError::Internal(
                    "PAGE metadata manager length exceeds its section bound".into(),
                ));
            }
            if metadata_active && metadata_length > metadata_capacity {
                metadata_capacity = metadata_length;
                continue;
            }
            return Ok(Self {
                extension,
                locations,
                arena,
            });
        }
    }
}

/// A stored-address cursor with free gaps classified by complete allocation length.
///
/// The occupied end excludes page padding after the last reserved block.
#[derive(Clone)]
struct MetadataArena {
    page: FileSpacePageSize,
    cursor: StoredAddress,
    occupied_end: StoredAddress,
    metadata: Vec<FreeSection>,
    large: Vec<FreeSection>,
}

impl MetadataArena {
    /// Starts placement at `start` with no allocations or gaps.
    fn new(page: FileSpacePageSize, start: StoredAddress) -> Self {
        Self {
            page,
            cursor: start,
            occupied_end: start,
            metadata: Vec::new(),
            large: Vec::new(),
        }
    }

    /// Places a small block within one page or a large block on a page boundary.
    ///
    /// Records skipped small-page space as metadata and the final page suffix of a large
    /// allocation as generic large space. The returned length excludes both kinds of gap.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the allocation or its page padding overflows.
    /// Returns [`FormatError::Internal`] if a padding endpoint precedes its cursor.
    fn reserve(&mut self, length: u64) -> Result<Allocation, FormatError> {
        let remaining = self.page.get() - self.cursor.get() % self.page.get();
        if length >= self.page.get() || length > remaining {
            self.gap_to(align(self.cursor, self.page)?)?;
        }
        let address = self.cursor;
        let end = address.checked_add(length)?;
        self.occupied_end = end;
        self.cursor = end;
        if length >= self.page.get() {
            self.cursor = align(end, self.page)?;
            if self.cursor > end {
                self.large.push(FreeSection {
                    addr: end,
                    size: self.cursor.checked_distance_from(end).ok_or_else(|| {
                        FormatError::Internal("PAGE padding ends before its allocation".into())
                    })?,
                });
            }
        }
        Ok(Allocation { address, length })
    }

    /// Reserves each active manager's header and allocated section list separately.
    ///
    /// Uses metadata, small raw, and generic large order. Each optional length is the complete
    /// section-list allocation length.
    ///
    /// # Errors
    ///
    /// Propagates placement errors from [`Self::reserve`].
    fn managers(
        &mut self,
        lengths: [Option<u64>; 3],
    ) -> Result<[Option<ManagerLocation>; 3], FormatError> {
        let mut locations = [None, None, None];
        for ((target, length), kind) in lengths.into_iter().enumerate().zip([
            ManagerKind::PagedSmallMetadata,
            ManagerKind::PagedSmallRaw,
            ManagerKind::PagedLargeGeneric,
        ]) {
            if let Some(length) = length {
                let header = self.reserve(
                    hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS),
                )?;
                let list = self.reserve(length)?;
                locations[target] = Some(ManagerLocation {
                    header: header.address,
                    section_list: list.address,
                    length,
                    kind,
                });
            }
        }
        Ok(locations)
    }

    /// Records the remaining small metadata tail and advances to a page boundary.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if page alignment overflows.
    /// Propagates padding errors from [`Self::gap_to`].
    fn finish(&mut self) -> Result<(), FormatError> {
        self.gap_to(align(self.cursor, self.page)?)
    }

    /// Records a nonempty metadata gap from the cursor to `end` and advances the cursor.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `end` precedes the cursor.
    fn gap_to(&mut self, end: StoredAddress) -> Result<(), FormatError> {
        let length = end.checked_distance_from(self.cursor).ok_or_else(|| {
            FormatError::Internal("PAGE metadata padding ends before its cursor".into())
        })?;
        if length == 0 {
            return Ok(());
        }
        let section = FreeSection {
            addr: self.cursor,
            size: length,
        };
        self.metadata.push(section);
        self.cursor = end;
        Ok(())
    }
}

/// A retained section count and the distinct sizes needed to measure a section list.
struct SectionMeasure {
    count: u64,
    sizes: BTreeSet<u64>,
}

impl SectionMeasure {
    /// Counts the supplied sections and retains their distinct sizes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the section count overflows.
    fn new(sizes: impl Iterator<Item = u64>) -> Result<Self, FormatError> {
        let mut result = Self {
            count: 0,
            sizes: BTreeSet::new(),
        };
        for size in sizes {
            result.count = checked_end(result.count, 1)?;
            result.sizes.insert(size);
        }
        Ok(result)
    }

    /// Measures the section list with the supplied additional sizes.
    ///
    /// Includes every additional section in the record count, grouping equal sizes together.
    /// Keeps the original measurement unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a count or encoded length overflows.
    /// Propagates inconsistent-count errors from [`SectionInfoCounts::new`].
    fn encoded_len(&self, additional: impl Iterator<Item = u64>) -> Result<u64, FormatError> {
        let Self { count, sizes } = self;
        let mut count = *count;
        let mut extra = BTreeSet::new();
        for size in additional {
            count = checked_end(count, 1)?;
            if !sizes.contains(&size) {
                extra.insert(size);
            }
        }
        SectionInfoCounts::new(
            count,
            checked_end(sizes.len().to_u64(), extra.len().to_u64())?,
        )?
        .encoded_len(WIDTHS)
    }

    /// Bounds encoded capacity when every additional section has a distinct size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a count or encoded length overflows.
    /// Propagates inconsistent-count errors from [`SectionInfoCounts::new`].
    fn capacity(&self, additional: u64) -> Result<u64, FormatError> {
        let Self { count, sizes } = self;
        SectionInfoCounts::new(
            checked_end(*count, additional)?,
            checked_end(sizes.len().to_u64(), additional)?,
        )?
        .encoded_len(WIDTHS)
    }
}

/// The extension and the header and section list of each of the three managers.
const TAIL_BLOCK_LIMIT: u64 = 7;
/// The maximum number of small metadata gaps, including the final page tail.
const TAIL_GAP_LIMIT: u64 = TAIL_BLOCK_LIMIT + 1;

/// Stored reservations for a manager's header and section list.
struct ManagerLocation {
    header: StoredAddress,
    section_list: StoredAddress,
    length: u64,
    kind: ManagerKind,
}

impl ManagerLocation {
    /// Serializes the manager at its reserved addresses and checks its lengths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if serialized lengths differ from their reservations.
    /// Propagates manager serialization errors.
    fn serialize(self, sections: &[FreeSection]) -> Result<FileManager, FormatError> {
        let Self {
            header,
            section_list,
            length,
            kind,
        } = self;
        let (_, class) = free_space_manager::manager_wire(kind);
        let (header_bytes, section_bytes) =
            hdf5_pure_format::__private::serialize_free_space_manager_with_allocation(
                WIDTHS,
                sections,
                header,
                section_list,
                class,
                Some(length),
            )?;
        if header_bytes.len() as u64
            != hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS)
            || section_bytes.len() as u64 != length
        {
            return Err(FormatError::Internal(
                "free-space manager bytes differ from their allocation".into(),
            ));
        }
        Ok(FileManager {
            header_address: header,
            section_list_address: section_list,
            header: header_bytes,
            section_list: section_bytes,
        })
    }
}

fn checked_end(offset: u64, length: u64) -> Result<u64, FormatError> {
    offset
        .checked_add(length)
        .ok_or(FormatError::OffsetOverflow { offset, length })
}

fn align(offset: StoredAddress, page: FileSpacePageSize) -> Result<StoredAddress, FormatError> {
    offset.checked_add((page.get() - offset.get() % page.get()) % page.get())
}

#[cfg(test)]
mod tests {
    use alloc::borrow::Cow;

    use hdf5_pure_format::__private as format;
    use hdf5_pure_format::__private::FreeSection;
    use hdf5_pure_format::__private::FreeSpaceManagerHeader;
    use rstest::rstest;

    use super::Allocation;
    use super::ArrayMeasure;
    use super::CollectionSections;
    use super::DataKind;
    use super::DataMeasure;
    use super::ObjectMeasure;
    use super::PagedCollections;
    use super::PagedLayout;
    use crate::address::DecodedStoredAddress;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::chunked_write::ChunkArrayGeometry;
    use crate::chunked_write::ChunkArrayKind;
    use crate::error::FormatError;
    use crate::file_allocation::FileAllocation;
    use crate::file_space_info::FileSpacePageSize;
    use crate::file_writer::WIDTHS;

    #[rstest]
    #[case::fixed(ChunkArrayKind::FixedArray, &[0, 1024], 1025, &[28, 8227])]
    #[case::extensible(ChunkArrayKind::ExtensibleArray, &[4], 5, &[72, 298, 150])]
    fn chunked_measurements_retain_complete_array_blocks(
        #[case] kind: ChunkArrayKind,
        #[case] occupied: &[u64],
        #[case] slots: u64,
        #[case] expected: &[u64],
    ) {
        let measurement = DataMeasure::chunked(
            16,
            Some(ChunkArrayGeometry {
                kind,
                occupied: Cow::Borrowed(occupied),
                slots,
                chunk_bytes: 8,
                filtered: false,
            }),
        )
        .unwrap();
        let lengths = match measurement.array.as_ref().unwrap() {
            ArrayMeasure::Fixed(plan) => plan.block_lengths().collect::<Vec<_>>(),
            ArrayMeasure::Extensible(plan) => plan.block_lengths().collect::<Vec<_>>(),
            ArrayMeasure::Empty(_) => panic!("expected occupied array blocks"),
        };
        assert_eq!(lengths, expected);
        assert_eq!(measurement.raw_length, 16);
        assert_eq!(
            measurement.length().unwrap(),
            16 + expected.iter().sum::<u64>()
        );
        assert!(measurement.is_chunked());
        assert!(measurement.empty_index().is_none());
        let mut allocation = FileAllocation::new(StoredAddress::new(48), false, None);
        assert_eq!(
            measurement.reserve(&mut allocation).unwrap(),
            Some(span(48, 16 + expected.iter().sum::<u64>()))
        );
    }

    #[rstest]
    #[case::fixed(u64::MAX, ChunkArrayKind::FixedArray, 1, FormatError::OffsetOverflow { offset: u64::MAX, length: 54 })]
    #[case::extensible(u64::MAX, ChunkArrayKind::ExtensibleArray, 1, FormatError::OffsetOverflow { offset: u64::MAX, length: 370 })]
    #[case::fixed_slots(
        0,
        ChunkArrayKind::FixedArray,
        u64::MAX,
        FormatError::InvalidFixedArrayGeometry
    )]
    fn chunked_measurements_reject_unrepresentable_regions(
        #[case] raw_length: u64,
        #[case] kind: ChunkArrayKind,
        #[case] slots: u64,
        #[case] expected: FormatError,
    ) {
        let error = DataMeasure::chunked(
            raw_length,
            Some(ChunkArrayGeometry {
                kind,
                occupied: Cow::Borrowed(&[]),
                slots,
                chunk_bytes: 8,
                filtered: false,
            }),
        )
        .err()
        .unwrap();
        assert_eq!(error, expected);
    }

    #[rstest]
    #[case::direct(false)]
    #[case::aggregated(true)]
    fn object_metadata_reservations_keep_header_attribute_and_link_order(#[case] aggregated: bool) {
        let mut allocation = FileAllocation::new(StoredAddress::new(48), aggregated, None);
        let placed = ObjectMeasure {
            header: 80,
            attributes: Some(24),
            links: Some(64),
        }
        .reserve(&mut allocation)
        .unwrap();
        let super::ObjectAllocation {
            header,
            attributes,
            links,
        } = placed;
        assert_eq!(header, span(48, 80));
        assert_eq!(attributes, Some(span(128, 24)));
        assert_eq!(links, Some(span(152, 64)));
    }

    #[rstest]
    #[case::empty_contiguous(DataKind::Contiguous, 0, None, 48)]
    #[case::empty_chunked(DataKind::Chunked, 0, Some((48, 0)), 48)]
    #[case::contiguous(DataKind::Contiguous, 7, Some((48, 7)), 55)]
    #[case::chunked(DataKind::Chunked, 7, Some((48, 7)), 55)]
    fn data_reservations_distinguish_empty_contiguous_storage(
        #[case] kind: DataKind,
        #[case] length: u64,
        #[case] expected: Option<(u64, u64)>,
        #[case] end: u64,
    ) {
        let mut allocation = FileAllocation::new(StoredAddress::new(48), false, None);
        let placed = DataMeasure::new(length, kind)
            .reserve(&mut allocation)
            .unwrap();
        assert_eq!(
            placed,
            expected.map(|(address, length)| span(address, length))
        );
        assert_eq!(allocation.end(), StoredAddress::new(end));
    }

    #[rstest]
    #[case::minimum_page(512, 0)]
    #[case::non_power_of_two(3000, 0)]
    #[case::above_four_gibibytes(512, 5_120_000_000)]
    fn paged_layout_records_small_large_and_empty_data(#[case] page: u64, #[case] offset: u64) {
        let measurements = [
            DataMeasure::new(0, DataKind::Contiguous),
            DataMeasure::new(7, DataKind::Contiguous),
            DataMeasure::new(0, DataKind::Chunked),
            DataMeasure::new(page + 188, DataKind::Contiguous),
        ];
        let plan = PagedLayout::new(
            FileSpacePageSize::try_from(page).unwrap(),
            false,
            StoredAddress::new(offset + 200),
            20,
            measurements.iter(),
            CollectionSections::default(),
            |_, allocation| Ok(allocation),
        )
        .unwrap();
        let PagedLayout {
            extension,
            data,
            small,
            large,
            slots,
            managers,
            metadata_end,
            raw_start,
            small_end,
            end,
        } = plan;
        assert_eq!(extension, span(offset + 200, 20));
        assert_eq!(metadata_end, StoredAddress::new(offset + 220));
        assert_eq!(raw_start, StoredAddress::new(offset + page));
        assert_eq!(small_end, StoredAddress::new(offset + page + 7));
        assert_eq!(end, StoredAddress::new(offset + 4 * page));
        assert_eq!(small, [1, 2]);
        assert_eq!(large, [3]);
        assert_eq!(
            data,
            [
                None,
                Some(span(offset + page, 7)),
                Some(span(offset + page + 7, 0)),
                Some(span(offset + 2 * page, page + 188))
            ]
        );
        assert_eq!(
            slots,
            [DecodedStoredAddress::Undefined; super::NUM_FILE_FSM_MANAGERS]
        );
        assert_eq!(managers.map(|manager| manager.is_some()), [false; 3]);
    }

    #[rstest]
    #[case::metadata(PageTail::Metadata, 0, 0)]
    #[case::small_raw(PageTail::Raw, 1, 2)]
    #[case::large(PageTail::Large, 2, 6)]
    fn paged_managers_describe_the_planned_page_tails(
        #[case] kind: PageTail,
        #[case] manager_index: usize,
        #[case] slot: usize,
    ) {
        let page = FileSpacePageSize::try_from(512).unwrap();
        let plan = PagedLayout::new(
            page,
            true,
            StoredAddress::new(200),
            20,
            [
                DataMeasure::new(7, DataKind::Contiguous),
                DataMeasure::new(700, DataKind::Chunked),
            ]
            .iter(),
            CollectionSections::default(),
            |_, allocation| Ok(allocation),
        )
        .unwrap();
        let PagedLayout {
            data,
            slots,
            managers,
            metadata_end,
            raw_start,
            small_end,
            end,
            extension: _,
            small: _,
            large: _,
        } = plan;
        let manager = managers[manager_index].as_ref().unwrap();
        assert_eq!(
            slots[slot],
            DecodedStoredAddress::Defined(manager.header_address)
        );
        let header = FreeSpaceManagerHeader::parse(WIDTHS, &manager.header).unwrap();
        let sections = format::parse_section_info(
            WIDTHS,
            &manager.section_list,
            manager.header_address,
            &header,
        )
        .unwrap();
        let expected = match kind {
            PageTail::Metadata => FreeSection {
                addr: metadata_end,
                size: raw_start.checked_distance_from(metadata_end).unwrap(),
            },
            PageTail::Raw => FreeSection {
                addr: small_end,
                size: 505,
            },
            PageTail::Large => FreeSection {
                addr: data[1].unwrap().address.checked_add(700).unwrap(),
                size: 324,
            },
        };
        if matches!(kind, PageTail::Metadata) {
            assert_eq!(
                sections,
                [
                    FreeSection {
                        addr: StoredAddress::new(472),
                        size: 40
                    },
                    expected
                ]
            );
        } else {
            assert_eq!(sections, [expected]);
        }
        assert_eq!(end.get() % page.get(), 0);
    }

    #[test]
    fn paged_collections_keep_metadata_and_raw_page_tails_separate() {
        let mut collections = PagedCollections::new(
            StoredAddress::new(48),
            FileSpacePageSize::try_from(512).unwrap(),
        );
        assert_eq!(collections.end, StoredAddress::new(48));
        assert_eq!(collections.reserve(3).unwrap(), span(512, 3));
        assert_eq!(collections.reserve(700).unwrap(), span(1024, 700));
        assert_eq!(collections.reserve(512).unwrap(), span(2048, 512));
        let PagedCollections {
            end,
            sections:
                CollectionSections {
                    metadata,
                    raw,
                    large,
                },
            page: _,
        } = collections;
        assert_eq!(end, StoredAddress::new(2560));
        assert_eq!(
            metadata,
            [FreeSection {
                addr: StoredAddress::new(48),
                size: 464
            }]
        );
        assert_eq!(
            raw,
            [FreeSection {
                addr: StoredAddress::new(515),
                size: 509
            }]
        );
        assert_eq!(
            large,
            [FreeSection {
                addr: StoredAddress::new(1724),
                size: 324
            }]
        );
    }

    #[rstest]
    #[case::extension_crossing(500, 20, 0, 0)]
    #[case::extension_one_page(500, 512, 0, 0)]
    #[case::extension_large(500, 600, 0, 0)]
    #[case::metadata_large(1_048_576, 20, 64, 0)]
    #[case::count_width_before(1_048_576, 20, 246, 0)]
    #[case::count_width_after(1_048_576, 20, 247, 0)]
    #[case::raw_list_one_page(1_048_576, 20, 0, 54)]
    #[case::raw_list_large(1_048_576, 20, 0, 100)]
    fn tail_blocks_and_free_sections_partition_page_storage(
        #[case] start: u64,
        #[case] extension: u64,
        #[case] metadata_count: usize,
        #[case] raw_count: usize,
    ) {
        let metadata: Vec<_> = (0..metadata_count)
            .map(|index| FreeSection {
                addr: StoredAddress::new(u64::try_from(index).unwrap() * 512 + 48),
                size: u64::try_from(index % 127 + 1).unwrap(),
            })
            .collect();
        let raw: Vec<_> = (0..raw_count)
            .map(|index| FreeSection {
                addr: StoredAddress::new(u64::try_from(index).unwrap() * 512 + 256),
                size: 1,
            })
            .collect();
        let plan = PagedLayout::new(
            FileSpacePageSize::try_from(512).unwrap(),
            true,
            StoredAddress::new(start),
            extension,
            core::iter::empty(),
            CollectionSections {
                metadata: metadata.clone(),
                raw: raw.clone(),
                large: Vec::new(),
            },
            |_, placed| Ok(placed),
        )
        .unwrap();
        let mut occupied = vec![plan.extension];
        let mut actual = [Vec::new(), Vec::new(), Vec::new()];
        for (target, manager) in plan.managers.iter().enumerate() {
            let Some(manager) = manager else {
                continue;
            };
            let header = FreeSpaceManagerHeader::parse(WIDTHS, &manager.header).unwrap();
            assert_eq!(
                header.section_list_used(),
                u64::try_from(manager.section_list.len()).unwrap()
            );
            assert_eq!(header.section_list_allocated(), header.section_list_used());
            assert_eq!(header.section_list_addr(), manager.section_list_address);
            occupied.extend([
                span(
                    manager.header_address.get(),
                    u64::try_from(manager.header.len()).unwrap(),
                ),
                span(
                    manager.section_list_address.get(),
                    header.section_list_allocated(),
                ),
            ]);
            actual[target] = format::parse_section_info(
                WIDTHS,
                &manager.section_list,
                manager.header_address,
                &header,
            )
            .unwrap();
        }
        occupied.sort_by_key(|block| block.address);
        let mut expected = [metadata, raw, Vec::new()];
        let mut cursor = StoredAddress::new(start);
        let mut previous_large = false;
        for block in occupied
            .iter()
            .chain(core::iter::once(&span(plan.raw_start.get(), 0)))
        {
            let boundary = if previous_large {
                super::align(cursor, FileSpacePageSize::try_from(512).unwrap())
                    .unwrap()
                    .min(block.address)
            } else {
                cursor
            };
            if boundary > cursor {
                expected[2].push(FreeSection {
                    addr: cursor,
                    size: boundary.checked_distance_from(cursor).unwrap(),
                });
                cursor = boundary;
            }
            if block.address > cursor {
                expected[0].push(FreeSection {
                    addr: cursor,
                    size: block.address.checked_distance_from(cursor).unwrap(),
                });
            }
            if block.length >= 512 {
                assert_eq!(block.address.get() % 512, 0);
            } else if block.length > 0 {
                assert_eq!(
                    block.address.get() / 512,
                    (block.address.get() + block.length - 1) / 512
                );
            }
            cursor = block.address.checked_add(block.length).unwrap();
            previous_large = block.length >= 512;
        }
        for (actual, expected) in actual.iter_mut().zip(&mut expected) {
            actual.sort_by_key(|section| section.addr);
            expected.sort_by_key(|section| section.addr);
            assert_eq!(actual, expected);
        }
        assert_eq!(plan.end, plan.raw_start);
        if metadata_count > 0 || raw_count > 54 || extension > 512 {
            assert!(plan.managers[2].is_some());
        }
        if raw_count == 54 {
            assert_eq!(plan.managers[1].as_ref().unwrap().section_list.len(), 512);
            assert!(plan.managers[2].is_none());
        }
    }

    #[rstest]
    #[case::empty(512, 492, 20, &[], false, (512, 512), None)]
    #[case::alignment_feedback(512, 375, 20, &[], true, (1_024, 1_024), None)]
    #[case::raw_manager(512, 375, 20, &[511], false, (512, 1_024), None)]
    #[case::large_manager(512, 375, 20, &[513], false, (512, 1_536), Some(35))]
    #[case::raw_and_large_managers(512, 258, 20, &[511, 513], false, (512, 2_048), Some(35))]
    #[case::generic_capacity_feedback(
        540,
        5_400,
        1_079,
        &[1_079; 47],
        false,
        (7_020, 57_780),
        Some(458)
    )]
    fn metadata_manager_terminates_when_its_blocks_fill_the_page(
        #[case] page: u64,
        #[case] start: u64,
        #[case] length: u64,
        #[case] data_lengths: &[u64],
        #[case] active: bool,
        #[case] boundaries: (u64, u64),
        #[case] large_list_length: Option<usize>,
    ) {
        let page = FileSpacePageSize::try_from(page).unwrap();
        let (raw_start, end) = boundaries;
        let measurements: Vec<_> = data_lengths
            .iter()
            .map(|&length| DataMeasure::new(length, DataKind::Contiguous))
            .collect();
        let plan = PagedLayout::new(
            page,
            true,
            StoredAddress::new(start),
            length,
            measurements.iter(),
            CollectionSections::default(),
            |_, placed| Ok(placed),
        )
        .unwrap();
        assert_eq!(plan.managers[0].is_some(), active);
        assert_eq!(
            plan.managers[1].is_some(),
            data_lengths.iter().any(|&length| length < page.get())
        );
        assert_eq!(
            plan.managers[2].is_some(),
            data_lengths.iter().any(|&length| length >= page.get())
        );
        assert_eq!(
            plan.managers[2]
                .as_ref()
                .map(|manager| manager.section_list.len()),
            large_list_length
        );
        assert_eq!(plan.raw_start.get(), raw_start);
        assert_eq!(plan.end.get(), end);
        if let Some(manager) = &plan.managers[0] {
            let header = FreeSpaceManagerHeader::parse(WIDTHS, &manager.header).unwrap();
            let mut sections = format::parse_section_info(
                WIDTHS,
                &manager.section_list,
                manager.header_address,
                &header,
            )
            .unwrap();
            sections.sort_by_key(|section| section.addr);
            assert_eq!(
                sections,
                [
                    FreeSection {
                        addr: StoredAddress::new(477),
                        size: 35
                    },
                    FreeSection {
                        addr: StoredAddress::new(565),
                        size: 459
                    }
                ]
            );
            assert_eq!(header.section_list_allocated(), 53);
        }
    }

    #[rstest]
    #[case::extension(u64::MAX, 1, u64::MAX, 1)]
    #[case::header_alignment(u64::MAX - 10, 0, u64::MAX - 10, 11)]
    #[case::list_alignment(u64::MAX - 82, 0, u64::MAX, 1)]
    fn tail_placement_rejects_address_overflow(
        #[case] start: u64,
        #[case] extension: u64,
        #[case] offset: u64,
        #[case] length: u64,
    ) {
        let error = PagedLayout::new(
            FileSpacePageSize::try_from(512).unwrap(),
            true,
            StoredAddress::new(start),
            extension,
            core::iter::once(&DataMeasure::new(7, DataKind::Contiguous)),
            CollectionSections::default(),
            |_, placed| Ok(placed),
        )
        .err()
        .unwrap();
        assert_eq!(error, FormatError::OffsetOverflow { offset, length });
    }

    #[rstest]
    #[case::extension(u64::MAX, 1, None, u64::MAX, 1)]
    #[case::metadata_padding(u64::MAX - 1, 0, None, u64::MAX - 1, 2)]
    #[case::large_extent_padding(200, 20, Some(u64::MAX), u64::MAX, 1)]
    fn paged_layout_rejects_overflow_before_emission(
        #[case] metadata_end: u64,
        #[case] extension: u64,
        #[case] data_length: Option<u64>,
        #[case] offset: u64,
        #[case] length: u64,
    ) {
        let measurements: Vec<_> = data_length
            .into_iter()
            .map(|length| DataMeasure::new(length, DataKind::Chunked))
            .collect();
        let error = PagedLayout::new(
            FileSpacePageSize::try_from(512).unwrap(),
            false,
            StoredAddress::new(metadata_end),
            extension,
            measurements.iter(),
            CollectionSections::default(),
            |_, allocation| Ok(allocation),
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(error, FormatError::OffsetOverflow { offset, length });
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[rstest]
    #[case::small(128, 512, false, "page_bound_records", "page_bound_records_control")]
    #[case::large(
        512,
        512,
        false,
        "page_many_bound_records",
        "page_many_bound_records_control"
    )]
    #[case::persistent(
        512,
        512,
        true,
        "persistent_page_records",
        "persistent_page_records_control"
    )]
    #[case::persistent_large_page(
        512,
        1_073_741_824,
        true,
        "persistent_large_page_records",
        "persistent_large_page_records_control"
    )]
    fn paged_planning_retains_one_record_table(
        #[case] count: usize,
        #[case] page: u64,
        #[case] persistent: bool,
        #[case] region: &'static str,
        #[case] control_region: &'static str,
    ) {
        use core::mem;

        use test_util::allocation;

        let (control, positive) = allocation::measure(control_region, || {
            std::hint::black_box(alloc::vec![0_u8; 1024])
        });
        assert!(positive.blocks > 0, "{positive}");
        assert!(positive.peak_bytes >= 1024, "{positive}");
        drop(control);
        let measurements: Vec<_> = (0..count)
            .map(|_| DataMeasure::new(16, DataKind::Contiguous))
            .collect();
        let mut calls = 0;
        let (plan, measured) = allocation::measure(region, || {
            PagedLayout::new(
                FileSpacePageSize::try_from(page).unwrap(),
                persistent,
                StoredAddress::new(200),
                20,
                measurements.iter(),
                CollectionSections::default(),
                |index, allocation| {
                    calls += 1;
                    Ok((index, allocation, [0xa5_u8; 256]))
                },
            )
            .unwrap()
        });
        let PagedLayout {
            data,
            small,
            large,
            raw_start,
            extension: _,
            slots: _,
            managers,
            metadata_end: _,
            small_end: _,
            end: _,
        } = plan;
        let retained = data.capacity() * mem::size_of_val(data.first().unwrap())
            + (small.capacity() + large.capacity()) * mem::size_of::<usize>()
            + managers
                .iter()
                .flatten()
                .map(|manager| manager.header.capacity() + manager.section_list.capacity())
                .sum::<usize>();
        assert_eq!(calls, count);
        assert_eq!(data.len(), count);
        assert_eq!(small, (0..count).collect::<alloc::vec::Vec<_>>());
        assert!(large.is_empty());
        for (index, (actual, allocation, bytes)) in data.iter().enumerate() {
            assert_eq!(*actual, index);
            assert_eq!(
                *allocation,
                Some(Allocation {
                    address: raw_start.checked_add(index as u64 * 16).unwrap(),
                    length: 16
                })
            );
            assert_eq!(*bytes, [0xa5; 256]);
        }
        assert_eq!(measured.live_bytes, retained as u64, "{measured}");
        assert!(measured.peak_bytes >= retained as u64, "{measured}");
        assert!(measured.peak_bytes <= retained as u64 + 1024, "{measured}");
    }

    enum PageTail {
        Large,
        Metadata,
        Raw,
    }

    fn span(address: u64, length: u64) -> Allocation {
        Allocation {
            address: StoredAddress::new(address),
            length,
        }
    }
}
