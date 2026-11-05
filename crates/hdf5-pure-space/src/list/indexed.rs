//! Address and size indexes borrowed from a canonical free list.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::mem;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FileSpacePageSize, StoredAddress};

use super::FreeList;
use crate::admission::TrackedSpace;
use crate::extent::Extent;
use crate::reservation::{ReservationRequest, ReservationWork};

/// A borrowed free list with address, size, and PAGE placement indexes.
///
/// Lookup ties use the containing extent's stored start address. The caller restores unused extents
/// before releasing the borrow. Successful allocations remain excluded from the owning list.
/// The PAGE block indexes are built on the first block request.
pub(crate) struct IndexedFreeList<'a> {
    owner: &'a mut FreeList,
    regions: BTreeMap<StoredAddress, Extent>,
    sizes: BTreeSet<(ByteLength, StoredAddress)>,
    interiors: BTreeSet<(ByteLength, StoredAddress)>,
    alignment: Option<NonZeroByteLength>,
    blocks: Option<Box<BlockIndex>>,
    buffer: Vec<Extent>,
    total_bytes: ByteLength,
    work: ReservationWork,
    restored: bool,
}

impl<'a> IndexedFreeList<'a> {
    /// Moves the owner's extents into lookup indexes, retaining its vector for restoration.
    pub(crate) fn new(owner: &'a mut FreeList, alignment: Option<NonZeroByteLength>) -> Self {
        let mut buffer = mem::take(&mut owner.regions);
        let total_bytes = mem::replace(&mut owner.total_bytes, ByteLength::new(0));
        let mut indexed = Self {
            owner,
            regions: BTreeMap::new(),
            sizes: BTreeSet::new(),
            interiors: BTreeSet::new(),
            alignment,
            blocks: None,
            buffer: Vec::new(),
            total_bytes,
            work: ReservationWork::default(),
            restored: false,
        };
        for extent in buffer.drain(..) {
            indexed.insert(extent);
            indexed.work.indexed_extents = indexed.work.indexed_extents.saturating_add(1);
        }
        indexed.buffer = buffer;
        indexed
    }

    /// Reserves a prefix of the smallest sufficient extent, retaining any remainder.
    pub(crate) fn allocate(&mut self, size: NonZeroByteLength) -> Option<Extent> {
        self.work.lookups = self.work.lookups.saturating_add(1);
        let &(_, start) = self
            .sizes
            .range((size.length(), StoredAddress::new(0))..)
            .next()?;
        let region = self.remove(start);
        // The size index selects a nonempty extent at least as long as the request.
        let allocation = Extent::new(region.start(), size.length())
            .expect("the allocation fits inside the indexed extent");
        if let Some(remainder) = Extent::between(allocation.end(), region.end()) {
            self.insert(remainder);
        }
        // The selected allocation is included in the indexed free-byte total.
        self.total_bytes = self
            .total_bytes
            .checked_sub(allocation.len())
            .expect("the indexed total includes the selected allocation");
        Some(allocation)
    }

    /// Reserves a PAGE block from the smallest sufficient legal capacity, retaining both edges.
    ///
    /// The caller passes a block request and the same page size for every call on this index.
    pub(crate) fn allocate_block(
        &mut self,
        request: ReservationRequest,
        page_size: FileSpacePageSize,
    ) -> Option<Extent> {
        self.work.lookups = self.work.lookups.saturating_add(1);
        let blocks = self.blocks.get_or_insert_with(|| {
            let mut blocks = BlockIndex::new(page_size);
            for &extent in self.regions.values() {
                blocks.insert(extent);
            }
            Box::new(blocks)
        });
        let start = blocks.find(request.size())?;
        let region = self.regions.get(&start).copied()?;
        let allocation = request
            .place_after(region.start(), Some(page_size))
            .ok()?
            .extent();
        self.remove(start);
        if let Some(head) = Extent::between(region.start(), allocation.start()) {
            self.insert(head);
        }
        if let Some(tail) = Extent::between(allocation.end(), region.end()) {
            self.insert(tail);
        }
        // The selected allocation is included in the indexed free-byte total.
        self.total_bytes = self
            .total_bytes
            .checked_sub(allocation.len())
            .expect("the indexed total includes the selected allocation");
        Some(allocation)
    }

    /// Reserves whole aligned units from the smallest sufficient aligned interior.
    ///
    /// Returns [`None`] if alignment was not configured or no interior fits.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `size` is not a multiple of the configured alignment.
    pub(crate) fn allocate_whole_units(&mut self, size: NonZeroByteLength) -> Option<Extent> {
        self.work.lookups = self.work.lookups.saturating_add(1);
        let alignment = self.alignment?;
        debug_assert_eq!(size.get() % alignment.get(), 0);
        let &(_, start) = self
            .interiors
            .range((size.length(), StoredAddress::new(0))..)
            .next()?;
        let region = self.remove(start);
        // Entries in the interior index have a checked aligned interior of sufficient length.
        let interior = region
            .aligned_interior(alignment.length())
            .expect("the indexed region contains whole units");
        let allocation = Extent::new(interior.start(), size.length())
            .expect("the allocation fits inside the indexed interior");
        if let Some(head) = Extent::between(region.start(), allocation.start()) {
            self.insert(head);
        }
        if let Some(tail) = Extent::between(allocation.end(), region.end()) {
            self.insert(tail);
        }
        // The selected allocation is included in the indexed free-byte total.
        self.total_bytes = self
            .total_bytes
            .checked_sub(allocation.len())
            .expect("the indexed total includes the selected allocation");
        Some(allocation)
    }

    /// Inserts an allocation remainder, merging adjacent free extents.
    ///
    /// # Panics
    ///
    /// Panics if `extent` overlaps recorded free space.
    pub(crate) fn free(&mut self, extent: Extent) {
        let mut start = extent.start();
        let mut end = extent.end();
        if let Some((&previous, region)) = self.regions.range(..start).next_back() {
            assert!(
                region.end() <= start,
                "returned allocation overlaps free space"
            );
            if region.end() == start {
                start = self.remove(previous).start();
            }
        }
        if let Some((&next, region)) = self.regions.range(start..).next() {
            assert!(
                end <= region.start(),
                "returned allocation overlaps free space"
            );
            if end == region.start() {
                end = self.remove(next).end();
            }
        }
        // The disjoint allocation and its adjacent free regions have a checked enclosing range.
        self.insert(Extent::between(start, end).expect("the merged extent is nonempty"));
        self.total_bytes = self
            .total_bytes
            .checked_add(extent.len())
            .expect("disjoint free extents fit in the address space");
    }

    /// Returns the unused extents and cached total to the owner once.
    pub(crate) fn restore(&mut self) {
        if self.restored {
            return;
        }
        let Self {
            owner,
            regions,
            sizes,
            interiors,
            alignment: _,
            blocks,
            buffer,
            total_bytes,
            work,
            restored,
        } = self;
        for extent in mem::take(regions).into_values() {
            buffer.push(extent);
            work.restored_extents = work.restored_extents.saturating_add(1);
        }
        *restored = true;
        owner.regions = mem::take(buffer);
        owner.total_bytes = mem::replace(total_bytes, ByteLength::new(0));
        sizes.clear();
        interiors.clear();
        *blocks = None;
    }

    /// Returns operation counts accumulated by this index.
    pub(crate) fn work(&self) -> ReservationWork {
        self.work
    }

    fn insert(&mut self, extent: Extent) {
        let Self {
            owner: _,
            regions,
            sizes,
            interiors,
            alignment,
            blocks,
            buffer: _,
            total_bytes: _,
            work,
            restored: _,
        } = self;
        if let Some(blocks) = blocks {
            blocks.insert(extent);
        }
        regions.insert(extent.start(), extent);
        sizes.insert((extent.len(), extent.start()));
        if let Some(interior) =
            alignment.and_then(|alignment| extent.aligned_interior(alignment.length()))
        {
            interiors.insert((interior.len(), extent.start()));
        }
        work.inserted_extents = work.inserted_extents.saturating_add(1);
    }

    fn remove(&mut self, start: StoredAddress) -> Extent {
        let Self {
            owner: _,
            regions,
            sizes,
            interiors,
            alignment,
            blocks,
            buffer: _,
            total_bytes: _,
            work,
            restored: _,
        } = self;
        // Every start in either lookup index has an entry in the address map.
        let extent = regions.remove(&start).expect("the indexed extent exists");
        if let Some(blocks) = blocks {
            blocks.remove(extent);
        }
        sizes.remove(&(extent.len(), start));
        if let Some(interior) =
            alignment.and_then(|alignment| extent.aligned_interior(alignment.length()))
        {
            interiors.remove(&(interior.len(), start));
        }
        work.removed_extents = work.removed_extents.saturating_add(1);
        extent
    }
}

impl TrackedSpace for IndexedFreeList<'_> {
    type Class = ();

    fn merges(&self, extent: Extent) -> bool {
        self.regions
            .range(..extent.start())
            .next_back()
            .is_some_and(|(_, previous)| previous.end() == extent.start())
            || self.regions.contains_key(&extent.end())
    }

    fn track(&mut self, extent: Extent, (): ()) {
        self.free(extent);
    }
}

/// Indexes single-page capacities and page-aligned suffix lengths by containing extent start.
struct BlockIndex {
    page_size: FileSpacePageSize,
    small: BTreeSet<(ByteLength, StoredAddress)>,
    large: BTreeSet<(ByteLength, StoredAddress)>,
}

impl BlockIndex {
    fn new(page_size: FileSpacePageSize) -> Self {
        Self {
            page_size,
            small: BTreeSet::new(),
            large: BTreeSet::new(),
        }
    }

    fn find(&self, size: NonZeroByteLength) -> Option<StoredAddress> {
        let index = if size.get() < self.page_size.get() {
            &self.small
        } else {
            &self.large
        };
        index
            .range((size.length(), StoredAddress::new(0))..)
            .next()
            .map(|&(_, start)| start)
    }

    fn insert(&mut self, extent: Extent) {
        self.small.insert((
            extent.block_capacity(self.page_size, ByteLength::new(1)),
            extent.start(),
        ));
        self.large.insert((
            extent.block_capacity(self.page_size, ByteLength::new(self.page_size.get())),
            extent.start(),
        ));
    }

    fn remove(&mut self, extent: Extent) {
        self.small.remove(&(
            extent.block_capacity(self.page_size, ByteLength::new(1)),
            extent.start(),
        ));
        self.large.remove(&(
            extent.block_capacity(self.page_size, ByteLength::new(self.page_size.get())),
            extent.start(),
        ));
    }
}
