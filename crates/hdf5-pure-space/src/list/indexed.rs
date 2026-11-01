//! Address and size indexes borrowed from a canonical free list.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::mem;
use core::num::NonZeroU64;

use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;

use super::FreeList;
use crate::admission::TrackedSpace;
use crate::extent::Extent;
use crate::reservation::ReservationWork;

/// A borrowed free list indexed by address, length, and optional aligned-interior length.
///
/// Lookup ties use the containing extent's stored start address. The caller restores unused extents
/// before releasing the borrow. Successful allocations remain excluded from the owning list.
pub(crate) struct IndexedFreeList<'a> {
    owner: &'a mut FreeList,
    regions: BTreeMap<StoredAddress, Extent>,
    sizes: BTreeSet<(u64, StoredAddress)>,
    interiors: BTreeSet<(u64, StoredAddress)>,
    alignment: Option<NonZeroU64>,
    buffer: Vec<Extent>,
    total_bytes: u64,
    work: ReservationWork,
    restored: bool,
}

impl<'a> IndexedFreeList<'a> {
    /// Moves the owner's extents into lookup indexes, retaining its vector for restoration.
    pub(crate) fn new(owner: &'a mut FreeList, alignment: Option<NonZeroU64>) -> Self {
        let mut buffer = mem::take(&mut owner.regions);
        let total_bytes = mem::replace(&mut owner.total_bytes, 0);
        let mut indexed = Self {
            owner,
            regions: BTreeMap::new(),
            sizes: BTreeSet::new(),
            interiors: BTreeSet::new(),
            alignment,
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
    pub(crate) fn allocate(&mut self, size: NonZeroU64) -> Option<Extent> {
        self.work.lookups = self.work.lookups.saturating_add(1);
        let &(_, start) = self
            .sizes
            .range((size.get(), StoredAddress::new(0))..)
            .next()?;
        let region = self.remove(start);
        // The size index selects a nonempty extent at least as long as the request.
        let allocation = Extent::new(region.start(), size.get())
            .expect("the allocation fits inside the indexed extent");
        if let Some(remainder) = Extent::between(allocation.end(), region.end()) {
            self.insert(remainder);
        }
        self.total_bytes -= allocation.len();
        Some(allocation)
    }

    /// Reserves whole aligned units from the smallest sufficient aligned interior.
    ///
    /// Returns [`None`] if alignment was not configured or no interior fits.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `size` is not a multiple of the configured alignment.
    pub(crate) fn allocate_whole_units(&mut self, size: NonZeroU64) -> Option<Extent> {
        self.work.lookups = self.work.lookups.saturating_add(1);
        let alignment = self.alignment?;
        debug_assert_eq!(size.get() % alignment.get(), 0);
        let &(_, start) = self
            .interiors
            .range((size.get(), StoredAddress::new(0))..)
            .next()?;
        let region = self.remove(start);
        // Entries in the interior index have a checked aligned interior of sufficient length.
        let interior = region
            .aligned_interior(alignment.get())
            .expect("the indexed region contains whole units");
        let allocation = Extent::new(interior.start(), size.get())
            .expect("the allocation fits inside the indexed interior");
        if let Some(head) = Extent::between(region.start(), allocation.start()) {
            self.insert(head);
        }
        if let Some(tail) = Extent::between(allocation.end(), region.end()) {
            self.insert(tail);
        }
        self.total_bytes -= allocation.len();
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
        owner.total_bytes = mem::replace(total_bytes, 0);
        sizes.clear();
        interiors.clear();
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
            buffer: _,
            total_bytes: _,
            work,
            restored: _,
        } = self;
        regions.insert(extent.start(), extent);
        sizes.insert((extent.len(), extent.start()));
        if let Some(interior) =
            alignment.and_then(|alignment| extent.aligned_interior(alignment.get()))
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
            buffer: _,
            total_bytes: _,
            work,
            restored: _,
        } = self;
        // Every start in either lookup index has an entry in the address map.
        let extent = regions.remove(&start).expect("the indexed extent exists");
        sizes.remove(&(extent.len(), start));
        if let Some(interior) =
            alignment.and_then(|alignment| extent.aligned_interior(alignment.get()))
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
