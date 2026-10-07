use core::cmp::Reverse;

use super::admission::TrackedSpace;
use super::extent::Extent;

/// A sorted, coalesced set of free regions in a single file being edited.
///
/// Invariants, upheld by every method: regions are sorted by start address, are non-empty, and
/// never touch or overlap (any two that would are merged on insertion). Allocation is best-fit to
/// fragmentation.
#[derive(Clone, Debug, Default)]
pub(crate) struct FreeList {
    /// Disjoint regions, sorted ascending by address and never adjacent.
    regions: Vec<Extent>,
}

impl FreeList {
    /// An empty free list.
    pub(crate) fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Records `extent` as free, merging it with any adjacent or overlapping regions so the list
    /// stays canonical.
    ///
    /// Overlapping an already-free region is a caller bug (a double-free). Debug builds panic.
    /// Release builds absorb the overlap into the merge and keep the list canonical.
    pub(crate) fn free(&mut self, extent: Extent) {
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
        while hi < self.regions.len() && self.regions[hi].start() <= merged_end {
            debug_assert!(
                self.regions[hi].start() >= extent.end()
                    || self.regions[hi].end() <= extent.start(),
                "double-free: [{}, {}) overlaps free region [{}, {})",
                extent.start(),
                extent.end(),
                self.regions[hi].start(),
                self.regions[hi].end()
            );
            merged_start = merged_start.min(self.regions[hi].start());
            merged_end = merged_end.max(self.regions[hi].end());
            hi += 1;
        }

        let merged = Extent::new(merged_start, merged_end - merged_start)
            .expect("a merged extent contains the newly freed extent");
        self.regions.splice(lo..hi, [merged]);
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
    pub(crate) fn release_all(
        &mut self,
        threshold: u64,
        eoa: u64,
        extents: impl IntoIterator<Item = Extent>,
    ) -> u64 {
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
    pub(crate) fn alloc(&mut self, len: u64) -> Option<Extent> {
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
            self.regions[i] = Extent::new(allocation.end(), region.end() - allocation.end())
                .expect("the remainder is a non-empty suffix of the selected extent");
        }
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
    pub(crate) fn alloc_whole_units(&mut self, len: u64, align: u64) -> Option<Extent> {
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
                Extent::new(region.start(), allocation.start() - region.start())
                    .expect("the lower remainder is non-empty and inside the selected extent"),
            );
        }
        if region.end() > allocation.end() {
            replacement.push(
                Extent::new(allocation.end(), region.end() - allocation.end())
                    .expect("the upper remainder is non-empty and inside the selected extent"),
            );
        }
        self.regions.splice(i..=i, replacement);
        Some(allocation)
    }

    /// Removes whatever part of `extent` this list holds, leaving the parts of each overlapped
    /// region that fall outside it free.
    ///
    /// [`alloc`](Self::alloc) chooses a free range. This method applies a range chosen elsewhere.
    /// A partially free or fully allocated range is valid because the caller has already decided
    /// its fate and uses this method to update the list. The paged editor uses it to lift a whole
    /// free page out of the per-page-type lists before
    /// re-filing it as one free page without a page-type classification
    /// (`PagedEdit::promote_whole_free_pages`).
    pub(crate) fn take_range(&mut self, extent: Extent) {
        let mut out = Vec::with_capacity(self.regions.len() + 1);
        for region in self.regions.drain(..) {
            // Disjoint from the range: keep the region whole.
            if region.end() <= extent.start() || region.start() >= extent.end() {
                out.push(region);
                continue;
            }
            // Overlapping: keep whatever lies below and above the range. Either side may be empty,
            // and both are when the range covers the region.
            if region.start() < extent.start() {
                out.push(
                    Extent::new(region.start(), extent.start() - region.start())
                        .expect("the lower remainder is non-empty and inside the free extent"),
                );
            }
            if region.end() > extent.end() {
                out.push(
                    Extent::new(extent.end(), region.end() - extent.end())
                        .expect("the upper remainder is non-empty and inside the free extent"),
                );
            }
        }
        self.regions = out;
    }

    /// Returns the free extents, sorted ascending by address and fully coalesced.
    pub(crate) fn extents(&self) -> Vec<Extent> {
        self.regions.clone()
    }

    /// Returns the free regions as `(addr, len)` pairs, sorted ascending by address and fully
    /// coalesced.
    ///
    /// Used to persist the free list to disk (issue #21) and to report the session's live reusable
    /// free space (issue #150).
    pub(crate) fn sections(&self) -> Vec<(u64, u64)> {
        self.regions
            .iter()
            .map(|extent| (extent.start(), extent.len()))
            .collect()
    }

    /// Returns whether the list is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The largest single run this list could satisfy an allocation from, or `0` when it is empty.
    ///
    /// Allocation is best-fit over *contiguous* regions, so a list holding plenty of bytes in small
    /// pieces may not satisfy a large request. An in-place append's reserve uses this value to
    /// determine whether it must draw more (issue #387).
    pub(crate) fn largest(&self) -> u64 {
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
    pub(crate) fn largest_whole_units(&self, align: u64) -> u64 {
        self.regions
            .iter()
            .filter_map(|extent| extent.aligned_interior(align).map(Extent::len))
            .max()
            .unwrap_or(0)
    }

    /// If a free region ends exactly at `eof` (the current end-of-file), removes it from the list
    /// and returns its start address. The file can be truncated to that address. Returns `None` if
    /// the highest free region does not reach end-of-file.
    ///
    /// Because the list is coalesced, at most one region can end at `eof`, and it is the last one.
    pub(crate) fn take_trailing(&mut self, eof: u64) -> Option<u64> {
        match self.regions.last() {
            Some(last) if last.end() == eof => {
                let start = last.start();
                self.regions.pop();
                Some(start)
            }
            _ => None,
        }
    }

    /// Returns `true` if a region adjoins `extent`.
    pub(crate) fn adjoins(&self, extent: Extent) -> bool {
        self.regions.iter().any(|region| region.adjoins(extent))
    }

    /// Returns `true` if a region adjoins `extent` inside one `unit`, at an address that is not a
    /// multiple of `unit`.
    pub(crate) fn adjoins_within(&self, extent: Extent, unit: u64) -> bool {
        self.regions.iter().any(|region| {
            (region.end() == extent.start() && !extent.start().is_multiple_of(unit))
                || (region.start() == extent.end() && !extent.end().is_multiple_of(unit))
        })
    }

    /// Returns how many bytes of `extent` the regions cover.
    pub(crate) fn covered_len(&self, extent: Extent) -> u64 {
        self.regions
            .iter()
            .filter_map(|region| region.intersection(extent))
            .map(Extent::len)
            .sum()
    }

    /// Returns `true` if a region of at least `unit` bytes adjoins `extent` at a multiple of
    /// `unit`.
    pub(crate) fn adjoins_whole_unit(&self, extent: Extent, unit: u64) -> bool {
        self.regions.iter().any(|region| {
            (region.end() == extent.start()
                && extent.start().is_multiple_of(unit)
                && region.len() >= unit)
                || (region.start() == extent.end()
                    && extent.end().is_multiple_of(unit)
                    && region.len() >= unit)
        })
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
/// per page type, plus space whose page type is unproven and space it may record but never hand out
/// ([`crate::edit`]). The run at the end of the file can therefore be split across those lists. For
/// example, free metadata may be followed by free raw data and then a dead fragment, with no single
/// list containing the whole run. The union reports whether the file's last bytes are all
/// unreferenced, which determines whether the file can shrink.
///
/// The regions are walked from the top, joining those that touch: because each list is coalesced
/// and the lists do not overlap, a region can only extend the run when its end meets the run's
/// start, so one descending pass is exact.
pub(crate) fn trailing_run_start<'a, I>(lists: I, eof: u64) -> u64
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
    use rstest::rstest;

    use super::*;

    /// Exposes the canonical region list as `(addr, len)` pairs for assertions.
    fn regions(fl: &FreeList) -> Vec<(u64, u64)> {
        fl.sections()
    }

    #[test]
    fn free_into_empty_list() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(100, 50).unwrap());
        assert_eq!(regions(&fl), [(100, 50)]);
    }

    #[test]
    fn zero_length_free_is_noop_at_the_boundary() {
        let mut fl = FreeList::new();
        if let Some(extent) = Extent::new(100, 0) {
            fl.free(extent);
        }
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn disjoint_frees_stay_sorted() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(300, 10).unwrap());
        fl.free(Extent::new(100, 10).unwrap());
        fl.free(Extent::new(200, 10).unwrap());
        assert_eq!(regions(&fl), [(100, 10), (200, 10), (300, 10)]);
    }

    #[test]
    fn coalesce_with_right_neighbor() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(200, 50).unwrap()); // [200, 250)
        fl.free(Extent::new(150, 50).unwrap()); // [150, 200) touches left edge of the above
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_with_left_neighbor() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(150, 50).unwrap()); // [150, 200)
        fl.free(Extent::new(200, 50).unwrap()); // [200, 250) touches right edge of the above
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_bridges_gap_between_two() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(100, 50).unwrap()); // [100, 150)
        fl.free(Extent::new(250, 50).unwrap()); // [250, 300)
        fl.free(Extent::new(150, 100).unwrap()); // [150, 250) bridges the two
        assert_eq!(regions(&fl), [(100, 200)]);
    }

    #[test]
    fn no_coalesce_when_gap_remains() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(100, 50).unwrap()); // [100, 150)
        fl.free(Extent::new(151, 50).unwrap()); // [151, 201) one byte gap
        assert_eq!(regions(&fl), [(100, 50), (151, 50)]);
    }

    #[test]
    fn alloc_best_fit_chooses_smallest_sufficient() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(0, 100).unwrap()); // big
        fl.free(Extent::new(200, 30).unwrap()); // exact-ish, smallest that fits 30
        fl.free(Extent::new(400, 60).unwrap()); // medium
        let allocated = fl.alloc(30).unwrap();
        assert_eq!(allocated, Extent::new(200, 30).unwrap());
        // The 30-byte region is consumed exactly. The others remain.
        assert_eq!(regions(&fl), [(0, 100), (400, 60)]);
    }

    #[test]
    fn alloc_splits_remainder() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(1000, 100).unwrap());
        let allocated = fl.alloc(40).unwrap();
        assert_eq!(allocated, Extent::new(1000, 40).unwrap());
        assert_eq!(regions(&fl), [(1040, 60)]);
    }

    #[test]
    fn alloc_none_when_nothing_fits() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(0, 10).unwrap());
        fl.free(Extent::new(100, 20).unwrap());
        assert!(fl.alloc(50).is_none());
        // List is unchanged on a failed allocation.
        assert_eq!(regions(&fl), [(0, 10), (100, 20)]);
    }

    #[test]
    fn alloc_zero_returns_none() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(0, 100).unwrap());
        assert!(fl.alloc(0).is_none());
    }

    #[test]
    fn alloc_then_free_roundtrips() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(0, 100).unwrap());
        let allocated = fl.alloc(40).unwrap();
        fl.free(allocated); // give it back
        assert_eq!(regions(&fl), [(0, 100)]); // coalesced back to whole
    }

    #[test]
    fn is_empty_and_largest_report_the_list() {
        let mut fl = FreeList::new();
        assert!(fl.is_empty());
        assert_eq!(fl.largest(), 0);
        fl.free(Extent::new(100, 10).unwrap());
        fl.free(Extent::new(200, 40).unwrap());
        fl.free(Extent::new(400, 25).unwrap());
        assert!(!fl.is_empty());
        assert_eq!(fl.largest(), 40);
        // Coalescing is what `largest` reports over, not the frees as issued.
        fl.free(Extent::new(240, 40).unwrap());
        assert_eq!(fl.largest(), 80);
    }

    /// `largest_whole_units` reports exactly what `alloc_whole_units` could take
    /// in one call: the aligned interior, not the region.
    #[test]
    fn largest_whole_units_reports_the_aligned_interior() {
        let mut fl = FreeList::new();
        assert_eq!(fl.largest_whole_units(16), 0);
        // [10, 40): thirty bytes, and the only whole unit in it is [16, 32).
        fl.free(Extent::new(10, 30).unwrap());
        assert_eq!(fl.largest(), 30);
        assert_eq!(fl.largest_whole_units(16), 16);
        // [100, 110): ten bytes with no whole unit at all.
        fl.free(Extent::new(100, 10).unwrap());
        assert_eq!(fl.largest_whole_units(16), 16);
        // [200, 260): sixty bytes whose whole units are [208, 256), three of them.
        fl.free(Extent::new(200, 60).unwrap());
        assert_eq!(fl.largest_whole_units(16), 48);
        assert_eq!(fl.largest_whole_units(0), 0);
        // The allocator agrees: that run is claimable in one call, and one unit
        // more is not.
        let mut probe = fl.clone();
        assert_eq!(probe.alloc_whole_units(64, 16), None);
        assert_eq!(
            probe.alloc_whole_units(48, 16),
            Some(Extent::new(208, 48).unwrap())
        );
    }

    #[test]
    fn take_trailing_at_eof() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(500, 100).unwrap()); // [500, 600)
        let cut = fl.take_trailing(600);
        assert_eq!(cut, Some(500));
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn take_trailing_none_when_not_at_eof() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(500, 100).unwrap()); // [500, 600)
        assert_eq!(fl.take_trailing(700), None); // live bytes between 600 and 700
        assert_eq!(regions(&fl), [(500, 100)]); // unchanged
    }

    #[test]
    fn take_trailing_only_cuts_the_tail_region() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(100, 50).unwrap()); // interior hole [100, 150)
        fl.free(Extent::new(500, 100).unwrap()); // trailing [500, 600)
        let cut = fl.take_trailing(600);
        assert_eq!(cut, Some(500));
        assert_eq!(regions(&fl), [(100, 50)]); // interior hole preserved
    }

    #[test]
    fn take_trailing_empty_list() {
        let mut fl = FreeList::new();
        assert_eq!(fl.take_trailing(0), None);
    }

    #[test]
    fn trailing_run_start_joins_across_lists() {
        // Free metadata, then free raw data, then a dead fragment, all abutting
        // and reaching end-of-file: the run starts where the metadata does, and
        // no single list can say so.
        let (mut meta, mut raw, mut dead) = (FreeList::new(), FreeList::new(), FreeList::new());
        meta.free(Extent::new(400, 100).unwrap()); // [400, 500)
        raw.free(Extent::new(500, 50).unwrap()); // [500, 550)
        dead.free(Extent::new(550, 50).unwrap()); // [550, 600)
        assert_eq!(trailing_run_start([&meta, &raw, &dead], 600), 400);
    }

    #[test]
    fn trailing_run_start_stops_at_the_first_live_gap() {
        let (mut meta, mut raw) = (FreeList::new(), FreeList::new());
        meta.free(Extent::new(100, 100).unwrap()); // [100, 200), below a live gap
        raw.free(Extent::new(400, 200).unwrap()); // [400, 600)
        assert_eq!(trailing_run_start([&meta, &raw], 600), 400);
    }

    #[test]
    fn trailing_run_start_is_eof_when_the_last_byte_is_live() {
        let mut meta = FreeList::new();
        meta.free(Extent::new(400, 100).unwrap()); // [400, 500), then live bytes to 600
        assert_eq!(trailing_run_start([&meta], 600), 600);
        assert_eq!(trailing_run_start([&FreeList::new()], 600), 600);
    }

    #[test]
    fn take_range_splits_the_region_around_it() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(100, 100).unwrap()); // [100, 200)
        fl.take_range(Extent::new(120, 30).unwrap()); // [120, 150)
        assert_eq!(regions(&fl), [(100, 20), (150, 50)]);
    }

    #[test]
    fn take_range_spanning_several_regions_keeps_only_the_edges() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(100, 50).unwrap()); // [100, 150)
        fl.free(Extent::new(200, 50).unwrap()); // [200, 250)
        fl.free(Extent::new(300, 50).unwrap()); // [300, 350)
        fl.take_range(Extent::new(120, 200).unwrap()); // [120, 320)
        assert_eq!(regions(&fl), [(100, 20), (320, 30)]);
    }

    #[test]
    fn take_range_tolerates_a_range_that_is_not_free() {
        let mut fl = FreeList::new();
        fl.free(Extent::new(100, 50).unwrap());
        fl.take_range(Extent::new(300, 50).unwrap()); // wholly outside the list
        fl.take_range(Extent::new(140, 20).unwrap()); // half inside it
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
            fl.free(Extent::new(addr, len).unwrap());
        }

        assert_eq!(
            fl.release_all(
                THRESHOLD,
                950,
                extents
                    .iter()
                    .map(|&(addr, len)| Extent::new(addr, len).unwrap()),
            ),
            expected_eoa
        );
        assert_eq!(regions(&fl), expected_regions);
    }

    const THRESHOLD: u64 = 64;
}
