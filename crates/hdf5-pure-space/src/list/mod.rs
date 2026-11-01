use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt;

use hdf5_pure_core::StoredAddress;

use super::admission::TrackedSpace;
use super::extent::Extent;
use super::persistence::TRAILING_RESERVE_TAILS;

pub(crate) mod indexed;

/// A sorted, coalesced set of free regions in stored-address coordinates.
///
/// Each region has [`StoredAddress`] endpoints relative to the file's address base.
///
/// Invariants, upheld by every method: regions are sorted by start address, are non-empty, and
/// never touch or overlap (any two that would are merged on insertion). Allocation is best-fit to
/// fragmentation.
///
/// The cached byte total equals the sum of the region lengths.
#[derive(Clone, Debug, Default)]
pub struct FreeList {
    /// Disjoint regions, sorted ascending by address and never adjacent.
    regions: Vec<Extent>,
    /// The sum of the region lengths, maintained by every insertion and removal.
    total_bytes: u64,
}

impl FreeList {
    /// An empty free list.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            total_bytes: 0,
        }
    }

    /// Constructs a free list from disjoint extents in ascending address order, coalescing
    /// adjacent extents.
    ///
    /// Processes the input in one pass and caches the exact sum of its lengths. Empty input
    /// produces an empty list.
    ///
    /// # Errors
    ///
    /// Returns [`FreeListError::Unsorted`] if an extent starts before the preceding input extent,
    /// or [`FreeListError::Overlap`] if their starts are ordered but their ranges overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_space::__private::{Extent, FreeList};
    ///
    /// let first = Extent::new(StoredAddress::new(100), 20).unwrap();
    /// let second = Extent::new(first.end(), 30).unwrap();
    /// let free = FreeList::from_sorted([first, second]).unwrap();
    /// assert_eq!(free.extents(), [Extent::new(first.start(), 50).unwrap()]);
    /// assert_eq!(free.total_bytes(), 50);
    /// ```
    pub fn from_sorted(extents: impl IntoIterator<Item = Extent>) -> Result<Self, FreeListError> {
        let mut list = Self::new();
        let mut previous: Option<Extent> = None;
        for extent in extents {
            if let Some(previous) = previous {
                if extent.start() < previous.start() {
                    return Err(FreeListError::Unsorted {
                        previous,
                        next: extent,
                    });
                }
                if extent.start() < previous.end() {
                    return Err(FreeListError::Overlap {
                        previous,
                        next: extent,
                    });
                }
            }
            // Sorted, disjoint extents have exclusive ends at most `u64::MAX`, so their total fits.
            list.total_bytes = list
                .total_bytes
                .checked_add(extent.len())
                .expect("disjoint extents fit in the address space");
            match list.regions.last_mut() {
                Some(last) if last.end() == extent.start() => {
                    // The ordered endpoints enclose both nonempty, adjacent extents.
                    *last = Extent::between(last.start(), extent.end())
                        .expect("adjacent extents have a checked union");
                }
                _ => list.regions.push(extent),
            }
            previous = Some(extent);
        }
        Ok(list)
    }

    /// Merges disjoint free lists into one canonical list, coalescing adjacent regions.
    ///
    /// For a fixed number of lists, runs in linear time in the total number of extents.
    ///
    /// # Errors
    ///
    /// Returns [`FreeListError::Overlap`] if extents from different lists overlap.
    pub(crate) fn from_lists<const N: usize>(lists: [&Self; N]) -> Result<Self, FreeListError> {
        let mut regions = lists.map(|list| list.regions.iter());
        Self::from_sorted(core::iter::from_fn(|| {
            let (index, _) = regions
                .iter()
                .enumerate()
                .filter_map(|(index, regions)| {
                    regions
                        .as_slice()
                        .first()
                        .map(|extent| (index, extent.start()))
                })
                .min_by_key(|&(_, start)| start)?;
            regions[index].next().copied()
        }))
    }

    /// Records `extent` as free, merging it with any adjacent or overlapping regions so the list
    /// stays canonical.
    ///
    /// Overlapping an already-free region is a caller bug (a double-free). Debug builds panic.
    /// Release builds absorb the overlap into the merge and keep the list canonical.
    pub fn free(&mut self, extent: Extent) {
        // Find the first region that ends at or after `extent.start()`. This is the leftmost one
        // that could touch or overlap the freed range. Everything before it is strictly to the
        // left with a gap and stays untouched.
        let mut lo = 0;
        while lo < self.regions.len() && self.regions[lo].end() < extent.start() {
            lo += 1;
        }

        // Find the end of the run of regions that touch or overlap `extent`: any region whose start
        // is <= the merged end is adjacent/overlapping and folds into the merged region.
        let mut hi = lo;
        let mut merged_start = extent.start();
        let mut merged_end = extent.end();
        let mut total_bytes = self.total_bytes;
        while hi < self.regions.len() && self.regions[hi].start() <= merged_end {
            debug_assert!(
                self.regions[hi].start() >= extent.end()
                    || self.regions[hi].end() <= extent.start(),
                "double-free: [{}, {}) overlaps free region [{}, {})",
                extent.start().get(),
                extent.end().get(),
                self.regions[hi].start().get(),
                self.regions[hi].end().get()
            );
            // Every merged region is already included in the cached total.
            total_bytes = total_bytes
                .checked_sub(self.regions[hi].len())
                .expect("merged regions are included in the cached total");
            merged_start = merged_start.min(self.regions[hi].start());
            merged_end = merged_end.max(self.regions[hi].end());
            hi += 1;
        }

        let merged = Extent::between(merged_start, merged_end)
            .expect("a merged extent contains the newly freed extent");
        // Canonical extents are disjoint and have exclusive ends at most `u64::MAX`, so the
        // replacement union and retained extents have a representable total.
        let total_bytes = total_bytes
            .checked_add(merged.len())
            .expect("canonical free regions fit in the address space");
        self.regions.splice(lo..hi, [merged]);
        self.total_bytes = total_bytes;
    }

    /// Records each of `extents` that is at least `threshold` bytes long or adjoins a recorded
    /// region, and returns `eoa` less the dropped extents that end it.
    ///
    /// A shorter extent merges into the region it adjoins, which may be one this call recorded,
    /// and is dropped where no region adjoins it. Two shorter extents that adjoin only each other
    /// are both dropped.
    ///
    /// A dropped extent that ends at `eoa` lowers the end of allocation to its start, and so does
    /// each further dropped extent that then ends it. A recorded region that ends at `eoa` stays
    /// in the list.
    pub fn release_all(
        &mut self,
        threshold: u64,
        eoa: StoredAddress,
        extents: impl IntoIterator<Item = Extent>,
    ) -> StoredAddress {
        let mut dropped =
            self.release_each(threshold, extents.into_iter().map(|extent| (extent, ())));
        // A dropped extent that ends the allocation shrinks it, as libhdf5 shrinks the file by a
        // freed block that ends it (`H5MF_try_shrink` in `H5MF.c` and
        // `H5MF__sect_simple_can_shrink` in `H5MFsection.c`, HDF5 1.14.6). libhdf5 shrinks it by a
        // tracked section that ends it as well, which this leaves to the caller.
        dropped.sort_unstable_by_key(|&(extent, ())| Reverse(extent.start()));
        dropped.into_iter().fold(eoa, |eoa, (extent, ())| {
            if extent.end() == eoa {
                extent.start()
            } else {
                eoa
            }
        })
    }

    /// Reserves `len` bytes from a free region, returning the extent handed out, or `None` if no
    /// single region is large enough.
    ///
    /// Best-fit uses the smallest region that fits to keep large runs intact. The allocation is
    /// taken from the low end of the chosen region. Any remainder stays free. `len` of 0 returns
    /// `None` (nothing to allocate).
    pub fn alloc(&mut self, len: u64) -> Option<Extent> {
        if len == 0 {
            return None;
        }
        let mut best: Option<usize> = None;
        for (i, extent) in self.regions.iter().enumerate() {
            if extent.len() >= len
                && best.is_none_or(|best| extent.len() < self.regions[best].len())
            {
                best = Some(i);
            }
        }
        let i = best?;
        let region = self.regions[i];
        // `len <= region.len()`, so construction cannot pass the already-checked end of `region`.
        let allocation = Extent::new(region.start(), len)
            .expect("a non-empty allocation fits inside the selected extent");
        if allocation.end() == region.end() {
            self.regions.remove(i);
        } else {
            self.regions[i] = Extent::between(allocation.end(), region.end())
                .expect("the remainder is a non-empty suffix of the selected extent");
        }
        self.subtract_bytes(len);
        Some(allocation)
    }

    /// Reserves `len` bytes from a run of whole `align`-sized units inside a free region, returning
    /// that extent, or `None` when no region contains one long enough. The caller rounds `len` up
    /// to a whole number of units. Whatever lies either side of the taken run stays free.
    ///
    /// This is how one page type claims space from the other's list. A paged file keeps free space
    /// per page type because a page may hold only one of them. A page holding *nothing* belongs to
    /// neither type, so it may be reopened as either. Only the whole, aligned interior of a
    /// region is provably in that state: the partial edges sit in pages whose other bytes are live,
    /// and those keep their type.
    ///
    /// Keeping one list per type lets a freed chunk-data run and the freed index abutting it
    /// coalesce into the single hole the dataset that vacated both needs (issue #261).
    ///
    /// `len` or `align` of 0 returns `None`.
    pub fn alloc_whole_units(&mut self, len: u64, align: u64) -> Option<Extent> {
        if len == 0 || align == 0 {
            return None;
        }
        // Whole units in, whole units out. This keeps both leftovers aligned and inside pages of
        // the type that already held them. An unrounded `len` would hand the caller's type the
        // front of a page and leave the back of that same page in the other type's list. That would
        // mix page types, which paging prevents. The caller supplies a rounded length and the unit.
        // Every caller is in this crate, so construction enforces the invariant.
        debug_assert_eq!(
            len % align,
            0,
            "alloc_whole_units takes a whole number of units"
        );
        // Best-fit uses the *aligned interior*, which is the part that can serve the request.
        let mut best: Option<(usize, Extent)> = None;
        for (i, region) in self.regions.iter().enumerate() {
            if let Some(interior) = region.aligned_interior(align)
                && interior.len() >= len
                && best.is_none_or(|(_, best)| interior.len() < best.len())
            {
                best = Some((i, interior));
            }
        }
        let (i, interior) = best?;
        let region = self.regions[i];
        // `len <= interior.len()`, so construction cannot pass `interior.end()`.
        let allocation = Extent::new(interior.start(), len)
            .expect("a non-empty allocation fits inside the aligned interior");
        let mut replacement = Vec::with_capacity(2);
        if allocation.start() > region.start() {
            replacement.push(
                Extent::between(region.start(), allocation.start())
                    .expect("the lower remainder is non-empty and inside the selected extent"),
            );
        }
        if region.end() > allocation.end() {
            replacement.push(
                Extent::between(allocation.end(), region.end())
                    .expect("the upper remainder is non-empty and inside the selected extent"),
            );
        }
        self.regions.splice(i..=i, replacement);
        self.subtract_bytes(len);
        Some(allocation)
    }

    /// Removes a suffix of the free run reaching `eoa` and returns the new end of allocation.
    ///
    /// Retains [`TRAILING_RESERVE_TAILS`] times `tail_len` bytes for future manager tails.
    /// Releases space only when at least that many bytes can be removed. Returns `eoa` and
    /// preserves the list if the run is too short or the reserve arithmetic overflows.
    pub fn release_trailing_run(&mut self, eoa: StoredAddress, tail_len: u64) -> StoredAddress {
        let run_start = trailing_run_start([&*self], eoa);
        let Some(keep) = TRAILING_RESERVE_TAILS.checked_mul(tail_len) else {
            return eoa;
        };
        let Some(release_min) = keep.checked_mul(2) else {
            return eoa;
        };
        if eoa
            .checked_distance_from(run_start)
            .is_none_or(|length| length < release_min)
        {
            return eoa;
        }
        let Ok(cut) = run_start.checked_add(keep) else {
            return eoa;
        };
        let Some(released) = Extent::between(cut, eoa) else {
            return eoa;
        };
        self.take_range(released);
        cut
    }

    /// Reserves the first `len` bytes of the smallest free region whose allocation ends at or
    /// below `limit`.
    pub(crate) fn alloc_below(&mut self, len: u64, limit: StoredAddress) -> Option<Extent> {
        let extent = self
            .regions
            .iter()
            .filter(|extent| {
                extent.len() >= len
                    && extent
                        .start()
                        .checked_add(len)
                        .is_ok_and(|end| end <= limit)
            })
            .min_by_key(|extent| extent.len())?;
        let allocation = Extent::new(extent.start(), len)?;
        self.take_range(allocation);
        Some(allocation)
    }

    /// Removes whatever part of `extent` this list holds, leaving the parts of each overlapped
    /// region that fall outside it free.
    ///
    /// [`alloc`](Self::alloc) chooses a free range. This method applies a range chosen elsewhere.
    /// A partially free or fully allocated range is valid because the caller has already decided
    /// its fate and uses this method to update the list. The paged editor uses it to lift a whole
    /// free page out of the per-page-type lists before
    /// re-filing it as one free page without a page-type classification.
    pub fn take_range(&mut self, extent: Extent) {
        let mut out = Vec::with_capacity(self.regions.len() + 1);
        let mut total_bytes = self.total_bytes;
        for region in self.regions.drain(..) {
            // Disjoint from the range: keep the region whole.
            let Some(overlap) = region.intersection(extent) else {
                out.push(region);
                continue;
            };
            let covered = overlap.len();
            // The overlap is contained in this recorded region and included in the total.
            total_bytes = total_bytes
                .checked_sub(covered)
                .expect("removed bytes are included in the cached total");
            // Overlapping: keep whatever lies below and above the range. Either side may be empty,
            // and both are when the range covers the region.
            if region.start() < extent.start() {
                out.push(
                    Extent::between(region.start(), extent.start())
                        .expect("the lower remainder is non-empty and inside the free extent"),
                );
            }
            if region.end() > extent.end() {
                out.push(
                    Extent::between(extent.end(), region.end())
                        .expect("the upper remainder is non-empty and inside the free extent"),
                );
            }
        }
        self.regions = out;
        self.total_bytes = total_bytes;
    }

    /// Removes the bytes covered by `ranges`, retaining the portions of each region outside them.
    ///
    /// Ranges may cover allocated bytes as well as free bytes. Runs in linear time in the combined
    /// numbers of input and output extents.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_space::__private::{Extent, FreeList};
    ///
    /// let extent = |start, len| Extent::new(StoredAddress::new(start), len).unwrap();
    /// let mut free = FreeList::from_sorted([extent(100, 100)]).unwrap();
    /// let removed = FreeList::from_sorted([extent(120, 10), extent(150, 20)]).unwrap();
    /// free.take_ranges(&removed);
    /// assert_eq!(free.extents(), [extent(100, 20), extent(130, 20), extent(170, 30)]);
    /// assert_eq!(free.total_bytes(), 70);
    /// ```
    pub fn take_ranges(&mut self, ranges: &Self) {
        if self.is_empty() || ranges.is_empty() {
            return;
        }
        let mut removed = ranges.regions.iter().peekable();
        let mut out = Vec::with_capacity(self.regions.len());
        let mut total_bytes = self.total_bytes;
        for region in self.regions.drain(..) {
            let mut start = region.start();
            while removed.peek().is_some_and(|range| range.end() <= start) {
                removed.next();
            }
            while let Some(&&range) = removed.peek() {
                if range.start() >= region.end() {
                    break;
                }
                if let Some(head) = Extent::between(start, range.start()) {
                    out.push(head);
                }
                let end = range.end().min(region.end());
                // Each overlap is a nonempty part of this region, counted once in the total.
                total_bytes = total_bytes
                    .checked_sub(
                        Extent::between(start.max(range.start()), end)
                            .expect("the current removal intersects the current region")
                            .len(),
                    )
                    .expect("removed bytes are included in the cached total");
                start = end;
                if range.end() >= region.end() {
                    break;
                }
                removed.next();
            }
            if let Some(tail) = Extent::between(start, region.end()) {
                out.push(tail);
            }
        }
        self.regions = out;
        self.total_bytes = total_bytes;
    }

    /// Returns the free bytes common to both lists, sorted by address and coalesced.
    ///
    /// Runs in linear time in the combined numbers of input and output extents.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_space::__private::{Extent, FreeList};
    ///
    /// let extent = |start, len| Extent::new(StoredAddress::new(start), len).unwrap();
    /// let left = FreeList::from_sorted([extent(100, 30), extent(150, 20)]).unwrap();
    /// let right = FreeList::from_sorted([extent(120, 40)]).unwrap();
    /// let common = left.intersection(&right);
    /// assert_eq!(common.extents(), [extent(120, 10), extent(150, 10)]);
    /// assert_eq!(common.total_bytes(), 20);
    /// ```
    pub fn intersection(&self, other: &Self) -> Self {
        let mut right = other.regions.iter().peekable();
        let mut intersections = Vec::new();
        for &left in &self.regions {
            while right
                .peek()
                .is_some_and(|range| range.end() <= left.start())
            {
                right.next();
            }
            while let Some(&&range) = right.peek() {
                if range.start() >= left.end() {
                    break;
                }
                if let Some(overlap) = left.intersection(range) {
                    intersections.push(overlap);
                }
                if range.end() >= left.end() {
                    break;
                }
                right.next();
            }
        }
        // Intersections of two canonical lists are nonempty, sorted and disjoint.
        Self::from_sorted(intersections).expect("canonical intersections are sorted and disjoint")
    }

    /// Returns the free extents, sorted ascending by address and fully coalesced.
    pub fn extents(&self) -> &[Extent] {
        &self.regions
    }

    /// Returns the free regions as `(addr, len)` pairs for test assertions.
    #[cfg(test)]
    pub(crate) fn sections(&self) -> Vec<(u64, u64)> {
        self.regions
            .iter()
            .map(|extent| (extent.start().get(), extent.len()))
            .collect()
    }

    /// Consumes the list and returns its checked extents.
    pub(crate) fn into_extents(self) -> Vec<Extent> {
        self.regions
    }

    /// Returns the cached sum of the free region lengths in bytes.
    ///
    /// Takes constant time and does not allocate.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Returns whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The largest single run this list could satisfy an allocation from, or `0` when it is empty.
    ///
    /// Allocation is best-fit over *contiguous* regions, so a list holding plenty of bytes in small
    /// pieces may not satisfy a large request. An in-place append's reserve uses this value to
    /// determine whether it must draw more (issue #387).
    pub fn largest(&self) -> u64 {
        self.regions
            .iter()
            .map(|extent| extent.len())
            .max()
            .unwrap_or(0)
    }

    /// Returns the largest run of whole `align`-sized units inside any one region, or `0` when no
    /// region holds a whole unit.
    ///
    /// [`alloc_whole_units`](Self::alloc_whole_units) can return at most this many bytes in one
    /// call. `align` of 0 reports `0` because no aligned interior exists.
    ///
    /// The whole-page counterpart of [`largest`](Self::largest), for the same caller: on a paged
    /// file an append's reserve may claim whole free pages out of the other page type's list, so
    /// how much it can draw is the larger of this and its own list's `largest`.
    pub fn largest_whole_units(&self, align: u64) -> u64 {
        self.regions
            .iter()
            .filter_map(|extent| extent.aligned_interior(align).map(Extent::len))
            .max()
            .unwrap_or(0)
    }

    /// Removes the free region ending at `eof` and returns its stored start address.
    ///
    /// The caller passes the stored end-of-allocation position and resolves the returned address
    /// before truncating the physical file. Returns `None` if the highest free region does not
    /// reach `eof`.
    ///
    /// Because the list is coalesced, at most one region can end at `eof`, and it is the last one.
    pub fn take_trailing(&mut self, eof: StoredAddress) -> Option<StoredAddress> {
        match self.regions.last() {
            Some(last) if last.end() == eof => {
                let start = last.start();
                self.subtract_bytes(last.len());
                self.regions.pop();
                Some(start)
            }
            _ => None,
        }
    }

    /// Returns `true` if a region adjoins `extent`.
    pub fn adjoins(&self, extent: Extent) -> bool {
        self.regions.iter().any(|region| region.adjoins(extent))
    }

    /// Returns `true` if a region adjoins `extent` inside one `unit`, at an address that is not a
    /// multiple of `unit`.
    pub fn adjoins_within(&self, extent: Extent, unit: u64) -> bool {
        self.regions.iter().any(|region| {
            (region.end() == extent.start() && !extent.start().get().is_multiple_of(unit))
                || (region.start() == extent.end() && !extent.end().get().is_multiple_of(unit))
        })
    }

    /// Returns how many bytes of `extent` the regions cover.
    pub fn covered_len(&self, extent: Extent) -> u64 {
        self.regions
            .iter()
            .filter_map(|region| region.intersection(extent))
            .map(Extent::len)
            .sum()
    }

    /// Returns `true` if a region of at least `unit` bytes adjoins `extent` at a multiple of
    /// `unit`.
    pub fn adjoins_whole_unit(&self, extent: Extent, unit: u64) -> bool {
        self.regions.iter().any(|region| {
            (region.end() == extent.start()
                && extent.start().get().is_multiple_of(unit)
                && region.len() >= unit)
                || (region.start() == extent.end()
                    && extent.end().get().is_multiple_of(unit)
                    && region.len() >= unit)
        })
    }
    fn subtract_bytes(&mut self, length: u64) {
        // Each caller removes an allocation or trailing extent included in the cached total.
        self.total_bytes = self
            .total_bytes
            .checked_sub(length)
            .expect("removed bytes are included in the cached total");
    }
}

/// An error constructing a [`FreeList`] from sorted extents.
///
/// Each variant identifies the consecutive input extents that failed construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreeListError {
    /// The starts are ordered, but the extents overlap.
    Overlap {
        /// The preceding input extent.
        previous: Extent,
        /// The overlapping input extent.
        next: Extent,
    },
    /// An extent starts before the preceding input extent.
    Unsorted {
        /// The preceding input extent.
        previous: Extent,
        /// The input extent with a lower start address.
        next: Extent,
    },
}

impl core::error::Error for FreeListError {}

impl fmt::Display for FreeListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, previous, next) = match self {
            Self::Overlap { previous, next } => ("overlapping", previous, next),
            Self::Unsorted { previous, next } => ("out-of-order", previous, next),
        };
        write!(
            f,
            "{kind} free extents [{}, {}) and [{}, {})",
            previous.start().get(),
            previous.end().get(),
            next.start().get(),
            next.end().get()
        )
    }
}

impl TrackedSpace for FreeList {
    type Class = ();

    fn merges(&self, extent: Extent) -> bool {
        self.adjoins(extent)
    }

    fn track(&mut self, extent: Extent, (): ()) {
        self.free(extent);
    }
}

/// Returns the lowest address of the run of free space that reaches `eof`, across several lists
/// that are individually coalesced and mutually disjoint. Returns `eof` itself when nothing there
/// is free.
///
/// The paged counterpart of [`FreeList::take_trailing`]. A paged file keeps free space in one list
/// per page type, plus unclassified space and dead space it may record but never hand out. The run
/// at the end of the file can therefore be split across those lists. For
/// example, free metadata may be followed by free raw data and then a dead fragment, with no single
/// list containing the whole run. The union reports whether the file's last bytes are all
/// unreferenced, which determines whether the file can shrink.
///
/// The regions are walked from the top, joining those that touch: because each list is coalesced
/// and the lists do not overlap, a region can only extend the run when its end meets the run's
/// start, so one descending pass is exact.
pub fn trailing_run_start<'a, I>(lists: I, eof: StoredAddress) -> StoredAddress
where
    I: IntoIterator<Item = &'a FreeList>,
{
    let mut all: Vec<Extent> = lists
        .into_iter()
        .flat_map(|list| list.regions.iter().copied())
        .collect();
    all.sort_unstable_by_key(|extent| extent.start());
    let mut start = eof;
    for extent in all.iter().rev() {
        if extent.end() < start {
            break;
        }
        start = start.min(extent.start());
    }
    start
}

#[cfg(test)]
mod tests {
    #[cfg(debug_assertions)]
    extern crate std;

    #[cfg(debug_assertions)]
    use alloc::string::String;
    use alloc::vec;
    #[cfg(debug_assertions)]
    use std::panic::{self, AssertUnwindSafe};

    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::StoredAddress;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty([vec![], vec![], vec![]], &[])]
    #[case::interleaved(
        [vec![(100, 20), (300, 20)], vec![(200, 20)], vec![(400, 20)]],
        &[(100, 20), (200, 20), (300, 20), (400, 20)]
    )]
    #[case::adjacent([vec![(100, 20)], vec![(120, 20)], vec![(140, 20)]], &[(100, 60)])]
    #[case::address_limit(
        [vec![(0, u64::MAX - 20)], vec![], vec![(u64::MAX - 20, 20)]], &[(0, u64::MAX)]
    )]
    fn merging_lists_preserves_sorted_extents_and_totals(
        #[case] inputs: [Vec<(u64, u64)>; 3],
        #[case] expected: &[(u64, u64)],
    ) {
        let lists = inputs.map(|extents| {
            FreeList::from_sorted(
                extents
                    .iter()
                    .map(|&(start, len)| Extent::new(StoredAddress::new(start), len).unwrap()),
            )
            .unwrap()
        });
        let merged = FreeList::from_lists(lists.each_ref()).unwrap();
        assert_eq!(regions(&merged), expected);
    }

    #[rstest]
    #[case::overlap(110)]
    #[case::duplicate(100)]
    fn merging_lists_rejects_overlapping_classes(#[case] start: u64) {
        let previous = Extent::new(StoredAddress::new(100), 20).unwrap();
        let next = Extent::new(StoredAddress::new(start), 20).unwrap();
        let first = FreeList::from_sorted([previous]).unwrap();
        let second = FreeList::from_sorted([next]).unwrap();
        assert_eq!(
            FreeList::from_lists([&first, &second]).unwrap_err(),
            FreeListError::Overlap { previous, next }
        );
    }

    #[test]
    fn free_into_empty_list() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(100), 50).unwrap());
        assert_eq!(regions(&fl), [(100, 50)]);
    }

    #[test]
    fn zero_length_free_is_noop_at_the_boundary() {
        let mut fl = FreeList::new();
        if let Some(extent) = Extent::new(StoredAddress::new(100), 0) {
            fl.free(extent);
        }
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn disjoint_frees_stay_sorted() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(300), 10).unwrap());
        fl.free(Extent::new(StoredAddress::new(100), 10).unwrap());
        fl.free(Extent::new(StoredAddress::new(200), 10).unwrap());
        assert_eq!(regions(&fl), [(100, 10), (200, 10), (300, 10)]);
    }

    #[test]
    fn coalesce_with_right_neighbor() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(200), 50).unwrap()); // [200, 250)
        fl.free(Extent::new(StoredAddress::new(150), 50).unwrap());
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_with_left_neighbor() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(150), 50).unwrap()); // [150, 200)
        fl.free(Extent::new(StoredAddress::new(200), 50).unwrap());
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_bridges_gap_between_two() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(100), 50).unwrap()); // [100, 150)
        fl.free(Extent::new(StoredAddress::new(250), 50).unwrap()); // [250, 300)
        fl.free(Extent::new(StoredAddress::new(150), 100).unwrap()); // [150, 250) bridges the two
        assert_eq!(regions(&fl), [(100, 200)]);
    }

    #[test]
    fn no_coalesce_when_gap_remains() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(100), 50).unwrap()); // [100, 150)
        fl.free(Extent::new(StoredAddress::new(151), 50).unwrap()); // [151, 201) one byte gap
        assert_eq!(regions(&fl), [(100, 50), (151, 50)]);
    }

    #[test]
    fn alloc_best_fit_chooses_smallest_sufficient() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(0), 100).unwrap()); // big
        fl.free(Extent::new(StoredAddress::new(200), 30).unwrap());
        fl.free(Extent::new(StoredAddress::new(400), 60).unwrap()); // medium
        let allocated = fl.alloc(30).unwrap();
        assert_eq!(allocated, Extent::new(StoredAddress::new(200), 30).unwrap());
        // The 30-byte region is consumed exactly. The others remain.
        assert_eq!(regions(&fl), [(0, 100), (400, 60)]);
    }

    #[test]
    fn alloc_splits_remainder() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(1000), 100).unwrap());
        let allocated = fl.alloc(40).unwrap();
        assert_eq!(
            allocated,
            Extent::new(StoredAddress::new(1000), 40).unwrap()
        );
        assert_eq!(regions(&fl), [(1040, 60)]);
    }

    #[test]
    fn alloc_none_when_nothing_fits() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(0), 10).unwrap());
        fl.free(Extent::new(StoredAddress::new(100), 20).unwrap());
        assert!(fl.alloc(50).is_none());
        // List is unchanged on a failed allocation.
        assert_eq!(regions(&fl), [(0, 10), (100, 20)]);
    }

    #[test]
    fn alloc_zero_returns_none() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(0), 100).unwrap());
        assert!(fl.alloc(0).is_none());
        assert_eq!(fl.total_bytes(), 100);
    }

    #[rstest]
    #[case::empty(&[], 10, 100, None, &[])]
    #[case::zero(&[(100, 100)], 0, 200, None, &[(100, 100)])]
    #[case::exact(&[(100, 20)], 20, 120, Some((100, 20)), &[])]
    #[case::smallest(&[(100, 100), (300, 30), (400, 60)], 20, 500,
        Some((300, 20)), &[(100, 100), (320, 10), (400, 60)])]
    #[case::equal_lengths(&[(100, 30), (200, 30)], 20, 300,
        Some((100, 20)), &[(120, 10), (200, 30)])]
    #[case::allocation_end(&[(100, 100)], 20, 120, Some((100, 20)), &[(120, 80)])]
    #[case::limit_excludes_smallest(&[(100, 100), (300, 30)], 20, 250,
        Some((100, 20)), &[(120, 80), (300, 30)])]
    #[case::past_limit(&[(100, 100)], 20, 119, None, &[(100, 100)])]
    #[case::too_short(&[(100, 10)], 20, 200, None, &[(100, 10)])]
    #[case::overflow(&[(u64::MAX - 10, 10)], 20, u64::MAX,
        None, &[(u64::MAX - 10, 10)])]
    #[case::maximum_end(&[(u64::MAX - 10, 10)], 5, u64::MAX,
        Some((u64::MAX - 10, 5)), &[(u64::MAX - 5, 5)])]
    fn allocation_below_limit_preserves_unallocated_space(
        #[case] initial: &[(u64, u64)],
        #[case] len: u64,
        #[case] limit: u64,
        #[case] expected: Option<(u64, u64)>,
        #[case] remaining: &[(u64, u64)],
    ) {
        let mut free = FreeList::new();
        for &(start, len) in initial {
            free.free(Extent::new(StoredAddress::new(start), len).unwrap());
        }
        assert_eq!(
            free.alloc_below(len, StoredAddress::new(limit)),
            expected.map(|(start, len)| Extent::new(StoredAddress::new(start), len).unwrap())
        );
        assert_eq!(free.sections(), remaining);
        assert_eq!(
            free.total_bytes(),
            remaining.iter().map(|&(_, len)| len).sum::<u64>()
        );
    }

    #[rstest]
    #[case::empty(&[], 1_000, 10, 1_000, &[])]
    #[case::live_end(&[(100, 100)], 1_000, 10, 1_000, &[(100, 100)])]
    #[case::below_threshold(&[(100, 79)], 179, 10, 179, &[(100, 79)])]
    #[case::threshold(&[(100, 80)], 180, 10, 140, &[(100, 40)])]
    #[case::above_threshold(&[(100, 200)], 300, 10, 140, &[(100, 40)])]
    #[case::interior_space(&[(10, 20), (100, 200)], 300, 10, 140,
        &[(10, 20), (100, 40)])]
    #[case::zero_tail(&[(100, 200)], 300, 0, 100, &[])]
    #[case::zero_tail_empty(&[], 0, 0, 0, &[])]
    #[case::reserve_overflow(&[(100, 200)], 300, u64::MAX, 300, &[(100, 200)])]
    #[case::threshold_overflow(&[(100, 200)], 300, u64::MAX / 4, 300, &[(100, 200)])]
    #[case::maximum_end(&[(u64::MAX - 80, 80)], u64::MAX, 10,
        u64::MAX - 40, &[(u64::MAX - 80, 40)])]
    fn trailing_release_retains_the_tail_reserve(
        #[case] initial: &[(u64, u64)],
        #[case] eoa: u64,
        #[case] tail_len: u64,
        #[case] expected: u64,
        #[case] remaining: &[(u64, u64)],
    ) {
        let mut free = FreeList::new();
        for &(start, len) in initial {
            free.free(Extent::new(StoredAddress::new(start), len).unwrap());
        }
        assert_eq!(
            free.release_trailing_run(StoredAddress::new(eoa), tail_len),
            StoredAddress::new(expected)
        );
        assert_eq!(free.sections(), remaining);
        assert_eq!(
            free.total_bytes(),
            remaining.iter().map(|&(_, len)| len).sum::<u64>()
        );
    }

    #[test]
    fn alloc_then_free_roundtrips() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(0), 100).unwrap());
        let allocated = fl.alloc(40).unwrap();
        fl.free(allocated); // give it back
        assert_eq!(regions(&fl), [(0, 100)]); // coalesced back to whole
    }

    #[test]
    fn is_empty_and_largest_report_the_list() {
        let mut fl = FreeList::new();
        assert!(fl.is_empty());
        assert_eq!(fl.largest(), 0);
        fl.free(Extent::new(StoredAddress::new(100), 10).unwrap());
        fl.free(Extent::new(StoredAddress::new(200), 40).unwrap());
        fl.free(Extent::new(StoredAddress::new(400), 25).unwrap());
        assert!(!fl.is_empty());
        assert_eq!(fl.largest(), 40);
        // Coalescing is what `largest` reports over, not the frees as issued.
        fl.free(Extent::new(StoredAddress::new(240), 40).unwrap());
        assert_eq!(fl.largest(), 80);
    }

    /// `largest_whole_units` reports exactly what `alloc_whole_units` could take
    /// in one call: the aligned interior, not the region.
    #[test]
    fn largest_whole_units_reports_the_aligned_interior() {
        let mut fl = FreeList::new();
        assert_eq!(fl.largest_whole_units(16), 0);
        // [10, 40): thirty bytes, and the only whole unit in it is [16, 32).
        fl.free(Extent::new(StoredAddress::new(10), 30).unwrap());
        assert_eq!(fl.largest(), 30);
        assert_eq!(fl.largest_whole_units(16), 16);
        // [100, 110): ten bytes with no whole unit at all.
        fl.free(Extent::new(StoredAddress::new(100), 10).unwrap());
        assert_eq!(fl.largest_whole_units(16), 16);
        // [200, 260): sixty bytes whose whole units are [208, 256), three of them.
        fl.free(Extent::new(StoredAddress::new(200), 60).unwrap());
        assert_eq!(fl.largest_whole_units(16), 48);
        assert_eq!(fl.largest_whole_units(0), 0);
        // The allocator agrees: that run is claimable in one call, and one unit
        // more is not.
        let mut probe = fl.clone();
        assert_eq!(probe.alloc_whole_units(64, 16), None);
        assert_eq!(
            probe.alloc_whole_units(48, 16),
            Some(Extent::new(StoredAddress::new(208), 48).unwrap())
        );
    }

    #[test]
    fn take_trailing_at_eof() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(500), 100).unwrap()); // [500, 600)
        let cut = fl.take_trailing(StoredAddress::new(600));
        assert_eq!(cut, Some(StoredAddress::new(500)));
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn take_trailing_none_when_not_at_eof() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(500), 100).unwrap()); // [500, 600)
        assert_eq!(fl.take_trailing(StoredAddress::new(700)), None);
        assert_eq!(regions(&fl), [(500, 100)]); // unchanged
    }

    #[test]
    fn take_trailing_only_cuts_the_tail_region() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(100), 50).unwrap()); // interior hole [100, 150)
        fl.free(Extent::new(StoredAddress::new(500), 100).unwrap()); // trailing [500, 600)
        let cut = fl.take_trailing(StoredAddress::new(600));
        assert_eq!(cut, Some(StoredAddress::new(500)));
        assert_eq!(regions(&fl), [(100, 50)]); // interior hole preserved
    }

    #[test]
    fn take_trailing_empty_list() {
        let mut fl = FreeList::new();
        assert_eq!(fl.take_trailing(StoredAddress::new(0)), None);
    }

    #[test]
    fn trailing_run_start_joins_across_lists() {
        // Free metadata, then free raw data, then a dead fragment, all abutting
        // and reaching end-of-file: the run starts where the metadata does, and
        // no single list can say so.
        let (mut meta, mut raw, mut dead) = (FreeList::new(), FreeList::new(), FreeList::new());
        meta.free(Extent::new(StoredAddress::new(400), 100).unwrap()); // [400, 500)
        raw.free(Extent::new(StoredAddress::new(500), 50).unwrap()); // [500, 550)
        dead.free(Extent::new(StoredAddress::new(550), 50).unwrap()); // [550, 600)
        assert_eq!(
            trailing_run_start([&meta, &raw, &dead], StoredAddress::new(600)),
            StoredAddress::new(400)
        );
    }

    #[test]
    fn trailing_run_start_stops_at_the_first_live_gap() {
        let (mut meta, mut raw) = (FreeList::new(), FreeList::new());
        meta.free(Extent::new(StoredAddress::new(100), 100).unwrap());
        raw.free(Extent::new(StoredAddress::new(400), 200).unwrap()); // [400, 600)
        assert_eq!(
            trailing_run_start([&meta, &raw], StoredAddress::new(600)),
            StoredAddress::new(400)
        );
    }

    #[test]
    fn trailing_run_start_is_eof_when_the_last_byte_is_live() {
        let mut meta = FreeList::new();
        meta.free(Extent::new(StoredAddress::new(400), 100).unwrap());
        assert_eq!(
            trailing_run_start([&meta], StoredAddress::new(600)),
            StoredAddress::new(600)
        );
        assert_eq!(
            trailing_run_start([&FreeList::new()], StoredAddress::new(600)),
            StoredAddress::new(600)
        );
    }

    #[test]
    fn take_range_splits_the_region_around_it() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(100), 100).unwrap()); // [100, 200)
        fl.take_range(Extent::new(StoredAddress::new(120), 30).unwrap()); // [120, 150)
        assert_eq!(regions(&fl), [(100, 20), (150, 50)]);
    }

    #[test]
    fn take_range_spanning_several_regions_keeps_only_the_edges() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(100), 50).unwrap()); // [100, 150)
        fl.free(Extent::new(StoredAddress::new(200), 50).unwrap()); // [200, 250)
        fl.free(Extent::new(StoredAddress::new(300), 50).unwrap()); // [300, 350)
        fl.take_range(Extent::new(StoredAddress::new(120), 200).unwrap()); // [120, 320)
        assert_eq!(regions(&fl), [(100, 20), (320, 30)]);
    }

    #[test]
    fn take_range_tolerates_a_range_that_is_not_free() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(StoredAddress::new(100), 50).unwrap());
        fl.take_range(Extent::new(StoredAddress::new(300), 50).unwrap()); // wholly outside the list
        fl.take_range(Extent::new(StoredAddress::new(140), 20).unwrap()); // half inside it
        assert_eq!(regions(&fl), [(100, 40)]);
    }

    #[rstest]
    #[case::a_dropped_extent_ending_it(&[], &[(900, 50)], 900, &[])]
    #[case::dropped_extents_ending_it_in_turn(&[], &[(900, 50), (850, 50)], 850, &[])]
    #[case::a_dropped_extent_ending_it_only_later(&[], &[(850, 50), (900, 50)], 850, &[])]
    #[case::a_dropped_extent_below_a_tracked_one(&[], &[(800, 50), (850, 100)], 950, &[(800, 150)])]
    #[case::a_tracked_extent_ending_it(&[], &[(850, 100)], 950, &[(850, 100)])]
    #[case::a_merged_extent_ending_it(&[(800, 100)], &[(900, 50)], 950, &[(800, 150)])]
    #[case::a_merged_extent_ending_it_only_later(
        &[(800, 50)],
        &[(900, 50), (850, 50)],
        950,
        &[(800, 150)]
    )]
    #[case::a_dropped_extent_below_it(&[], &[(800, 50)], 950, &[])]
    fn releasing_lowers_the_end_of_allocation_by_the_dropped_extents_that_end_it(
        #[case] tracked: &[(u64, u64)],
        #[case] extents: &[(u64, u64)],
        #[case] expected_eoa: u64,
        #[case] expected_regions: &[(u64, u64)],
    ) {
        let mut fl = FreeList::new();
        for &(addr, len) in tracked {
            fl.free(Extent::new(StoredAddress::new(addr), len).unwrap());
        }

        assert_eq!(
            fl.release_all(
                THRESHOLD,
                StoredAddress::new(950),
                extents
                    .iter()
                    .map(|&(addr, len)| Extent::new(StoredAddress::new(addr), len).unwrap()),
            ),
            StoredAddress::new(expected_eoa)
        );
        assert_eq!(regions(&fl), expected_regions);
    }

    #[rstest]
    #[case::split_edges(10, 30, 16, 16, Some((16, 16)), &[(10, 6), (32, 8)], 14)]
    #[case::lower_edge(10, 22, 16, 16, Some((16, 16)), &[(10, 6)], 6)]
    #[case::upper_edge(16, 24, 16, 16, Some((16, 16)), &[(32, 8)], 8)]
    #[case::exact(16, 16, 16, 16, Some((16, 16)), &[], 0)]
    #[case::too_large(10, 30, 32, 16, None, &[(10, 30)], 30)]
    #[case::zero_length(10, 30, 0, 16, None, &[(10, 30)], 30)]
    #[case::zero_alignment(10, 30, 16, 0, None, &[(10, 30)], 30)]
    #[case::maximum_end(u64::MAX - 31, 31, 16, 16,
        Some((u64::MAX - 31, 16)), &[(u64::MAX - 15, 15)], 15)]
    fn cached_total_tracks_aligned_allocation(
        #[case] start: u64,
        #[case] length: u64,
        #[case] requested: u64,
        #[case] alignment: u64,
        #[case] allocated: Option<(u64, u64)>,
        #[case] remaining: &[(u64, u64)],
        #[case] total: u64,
    ) {
        let mut free = FreeList::new();
        free.free(Extent::new(StoredAddress::new(start), length).unwrap());
        assert_eq!(
            free.alloc_whole_units(requested, alignment),
            allocated
                .map(|(start, length)| Extent::new(StoredAddress::new(start), length).unwrap())
        );
        assert_eq!(regions(&free), remaining);
        assert_eq!(free.total_bytes(), total);
    }

    #[test]
    fn cached_total_covers_the_address_space_and_clones_independently() {
        let mut free = FreeList::default();
        assert_eq!(free.total_bytes(), 0);
        free.free(Extent::new(StoredAddress::new(0), u64::MAX - 10).unwrap());
        free.free(Extent::new(StoredAddress::new(u64::MAX - 10), 10).unwrap());
        assert_eq!(free.total_bytes(), u64::MAX);
        let snapshot = free.clone();
        assert_eq!(free.alloc(10), Extent::new(StoredAddress::new(0), 10));
        assert_eq!(free.total_bytes(), u64::MAX - 10);
        free.take_range(Extent::new(StoredAddress::new(20), 10).unwrap());
        assert_eq!(free.total_bytes(), u64::MAX - 20);
        assert_eq!(
            free.take_trailing(StoredAddress::new(u64::MAX)),
            Some(StoredAddress::new(30))
        );
        assert_eq!(free.total_bytes(), 10);
        assert_eq!(snapshot.total_bytes(), u64::MAX);
        free = snapshot;
        assert_eq!(
            free.alloc(u64::MAX),
            Extent::new(StoredAddress::new(0), u64::MAX)
        );
        assert_eq!(free.total_bytes(), 0);
        assert_eq!(regions(&free), []);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn rejected_overlap_preserves_the_cached_total() {
        let mut free = FreeList::new();
        free.free(Extent::new(StoredAddress::new(0), 50).unwrap());
        free.free(Extent::new(StoredAddress::new(100), 100).unwrap());
        let error = panic::catch_unwind(AssertUnwindSafe(|| {
            free.free(Extent::new(StoredAddress::new(50), 100).unwrap());
        }))
        .unwrap_err();
        assert_eq!(
            error.downcast_ref::<String>().map(String::as_str),
            Some("double-free: [50, 150) overlaps free region [100, 200)")
        );
        assert_eq!(regions(&free), [(0, 50), (100, 100)]);
        assert_eq!(free.total_bytes(), 150);
    }

    #[cfg(not(debug_assertions))]
    #[rstest]
    #[case::partial(&[(100, 100)], (150, 100), &[(100, 150)], 150)]
    #[case::contained(&[(100, 100)], (120, 10), &[(100, 100)], 100)]
    #[case::containing(&[(100, 100)], (50, 200), &[(50, 200)], 200)]
    #[case::several(&[(100, 50), (200, 50)], (125, 100), &[(100, 150)], 150)]
    #[case::repeated(&[(100, 100)], (100, 100), &[(100, 100)], 100)]
    #[case::maximum_end(&[(0, u64::MAX - 10)], (u64::MAX - 20, 20), &[(0, u64::MAX)], u64::MAX)]
    fn cached_total_counts_overlapping_frees_once(
        #[case] initial: &[(u64, u64)],
        #[case] (start, length): (u64, u64),
        #[case] remaining: &[(u64, u64)],
        #[case] total: u64,
    ) {
        let mut free = FreeList::new();
        for &(start, length) in initial {
            free.free(Extent::new(StoredAddress::new(start), length).unwrap());
        }
        free.free(Extent::new(StoredAddress::new(start), length).unwrap());
        assert_eq!(regions(&free), remaining);
        assert_eq!(free.total_bytes(), total);
    }

    /// Checks the cached total against the region lengths and returns `(addr, len)` pairs.
    fn regions(fl: &FreeList) -> Vec<(u64, u64)> {
        assert_eq!(
            u128::from(fl.total_bytes()),
            fl.extents()
                .iter()
                .map(|extent| u128::from(extent.len()))
                .sum::<u128>()
        );
        fl.sections()
    }

    const THRESHOLD: u64 = 64;
}
