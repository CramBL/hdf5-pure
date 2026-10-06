//! The free list of a read-write session, which records the space its commits vacate.
//!
//! A commit vacates the object headers it supersedes and the blocks of the objects it deletes.
//! Under a strategy with free-space managers, `H5F_FSPACE_STRATEGY_FSM_AGGR` or
//! `H5F_FSPACE_STRATEGY_PAGE`, the session records the vacated extents in a [`FreeList`] and
//! writes a later object into a free region that fits it. [`FreeList::release_all`] records an
//! extent at least as long as the file's threshold, the smallest free-space section the managers
//! track, or one that adjoins a recorded region.
//!
//! A session on a file without persistence starts with an empty list. On a file created with
//! `persist = true`, [`File::open_rw`](crate::File::open_rw) seeds the list from the file's
//! free-space managers, and each commit writes the list back to them.

use core::cmp::Reverse;

/// A contiguous run of free bytes in the file, `[addr, addr + len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRegion {
    addr: u64,
    len: u64,
}

impl FreeRegion {
    /// One past the last byte of the region.
    fn end(&self) -> u64 {
        self.addr + self.len
    }

    /// The whole `align`-sized units inside this region, as `(start, span)`, or
    /// `None` when the region contains no whole unit.
    ///
    /// This is the part of a region that is provably in no unit shared with
    /// anything live: the partial edges sit in units whose other bytes may be
    /// occupied, so they are left out. Both the allocation over such interiors
    /// ([`FreeList::alloc_whole_units`]) and the question of how large one is
    /// ([`FreeList::largest_whole_units`]) are defined by this one function, so
    /// the two cannot disagree about what counts.
    fn aligned_interior(&self, align: u64) -> Option<(u64, u64)> {
        let start = self.addr.next_multiple_of(align);
        let end = (self.end() / align) * align;
        // `then`, not `then_some`: a region with no aligned interior at all has
        // `end < start`, and `then_some`'s argument is evaluated whatever the
        // condition says.
        (end > start).then(|| (start, end - start))
    }
}

/// A sorted, coalesced set of free regions in a single file being edited.
///
/// Invariants, upheld by every method: regions are sorted by `addr`, are
/// non-empty, and never touch or overlap (any two that would are merged on
/// insertion). Allocation is best-fit to limit fragmentation.
#[derive(Debug, Default, Clone)]
pub(crate) struct FreeList {
    /// Disjoint regions, sorted ascending by address and never adjacent.
    regions: Vec<FreeRegion>,
}

impl FreeList {
    /// An empty free list.
    pub(crate) fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Record `[addr, addr + len)` as free, merging it with any adjacent or
    /// overlapping regions so the list stays canonical.
    ///
    /// A zero-length free is a no-op. Overlapping an already-free region is a
    /// caller bug (a double-free): in debug builds it panics; in release builds
    /// the overlap is absorbed by the merge rather than corrupting the list.
    pub(crate) fn free(&mut self, addr: u64, len: u64) {
        if len == 0 {
            return;
        }
        let new_end = addr + len;

        // Find the first region that ends at or after `addr` — the leftmost one
        // that could touch or overlap the freed range. Everything before it is
        // strictly to the left with a gap and stays untouched.
        let mut lo = 0;
        while lo < self.regions.len() && self.regions[lo].end() < addr {
            lo += 1;
        }

        // Find the end of the run of regions that touch or overlap `[addr,
        // new_end)`: any region whose start is <= new_end is adjacent/overlapping
        // and folds into the merged region.
        let mut hi = lo;
        let mut merged_addr = addr;
        let mut merged_end = new_end;
        while hi < self.regions.len() && self.regions[hi].addr <= merged_end {
            debug_assert!(
                self.regions[hi].addr >= new_end || self.regions[hi].end() <= addr,
                "double-free: [{addr}, {new_end}) overlaps free region [{}, {})",
                self.regions[hi].addr,
                self.regions[hi].end()
            );
            merged_addr = merged_addr.min(self.regions[hi].addr);
            merged_end = merged_end.max(self.regions[hi].end());
            hi += 1;
        }

        let merged = FreeRegion {
            addr: merged_addr,
            len: merged_end - merged_addr,
        };
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
        extents: impl IntoIterator<Item = (u64, u64)>,
    ) -> u64 {
        // The extents of at least the threshold are tracked first, and the smaller ones are offered
        // again until none merges, so what is tracked does not depend on the order of `extents`.
        let (mut pending, eligible): (Vec<_>, Vec<_>) =
            extents.into_iter().partition(|&(_, len)| len < threshold);
        for (addr, len) in eligible {
            self.free(addr, len);
        }
        loop {
            let offered = pending.len();
            pending.retain(|&(addr, len)| self.release(threshold, addr, len) == Release::Drop);
            if pending.len() == offered {
                break;
            }
        }
        // A dropped extent that ends the allocation shrinks it, as libhdf5 shrinks the file by a
        // freed block that ends it (`H5MF_try_shrink` in `H5MF.c` and
        // `H5MF__sect_simple_can_shrink` in `H5MFsection.c`, HDF5 1.14.6). libhdf5 shrinks it by a
        // tracked section that ends it as well, which this leaves to the caller.
        pending.sort_unstable_by_key(|&(addr, _)| Reverse(addr));
        pending.into_iter().fold(
            eoa,
            |eoa, (addr, len)| if addr + len == eoa { addr } else { eoa },
        )
    }

    /// Records `[addr, addr + len)` if it is at least `threshold` bytes long or adjoins a region,
    /// and returns which [`Release`] applies.
    ///
    /// An empty extent is dropped whatever `threshold` is.
    fn release(&mut self, threshold: u64, addr: u64, len: u64) -> Release {
        // `H5MF_xfree` (`H5MF.c`, HDF5 1.14.6) adds a section of at least the threshold and merges a
        // smaller one into an adjoining tracked section through `H5FS_sect_try_merge`
        // (`H5FSsection.c`), dropping it where none adjoins.
        let release = if len == 0 {
            Release::Drop
        } else if len >= threshold {
            Release::Track
        } else if self.adjoins(addr, len) {
            Release::Merge
        } else {
            Release::Drop
        };
        if release != Release::Drop {
            self.free(addr, len);
        }
        release
    }

    /// Reserve `len` bytes from a free region, returning the address handed out,
    /// or `None` if no single region is large enough.
    ///
    /// Best-fit: the smallest region that fits, to keep large runs intact. The
    /// allocation is taken from the low end of the chosen region; any remainder
    /// stays free. `len` of 0 returns `None` (nothing to allocate).
    pub(crate) fn alloc(&mut self, len: u64) -> Option<u64> {
        if len == 0 {
            return None;
        }
        let mut best: Option<usize> = None;
        for (i, r) in self.regions.iter().enumerate() {
            if r.len >= len && best.is_none_or(|b| r.len < self.regions[b].len) {
                best = Some(i);
            }
        }
        let i = best?;
        let addr = self.regions[i].addr;
        if self.regions[i].len == len {
            self.regions.remove(i);
        } else {
            self.regions[i].addr += len;
            self.regions[i].len -= len;
        }
        Some(addr)
    }

    /// Reserve `len` bytes from a run of whole `align`-sized units inside a free
    /// region, returning that address, or `None` when no region contains one long
    /// enough. `len` is rounded up to a whole number of units by the caller;
    /// whatever lies either side of the taken run stays free.
    ///
    /// This is how one page type claims space from the other's list. A paged file
    /// keeps free space per page type because a page may hold only one of them —
    /// but a page holding *nothing* belongs to neither, so it may be reopened as
    /// either. Only the whole, aligned interior of a free region is provably in
    /// that state: the partial edges sit in pages whose other bytes are live, and
    /// those keep their type.
    ///
    /// Kept as one list per type rather than promoting empty pages into a third
    /// list, so a freed chunk-data run and the freed index abutting it still
    /// coalesce into the single hole the dataset that vacated both needs
    /// (issue #261).
    ///
    /// `len` or `align` of 0 returns `None`.
    pub(crate) fn alloc_whole_units(&mut self, len: u64, align: u64) -> Option<u64> {
        if len == 0 || align == 0 {
            return None;
        }
        // Whole units in, whole units out: this is what makes *both* leftovers
        // aligned, and so keeps each of them inside pages of the type that already
        // held them. An unrounded `len` would hand the caller's type the front of a
        // page and leave the back of that same page in the other type's list — page
        // mixing, silently, which is the one thing paging exists to prevent. The
        // caller does the rounding because only it knows the unit, and every caller
        // is in this crate, so this is a construction-enforced invariant rather
        // than a refusal.
        debug_assert_eq!(
            len % align,
            0,
            "alloc_whole_units takes a whole number of units"
        );
        // Best-fit over the *aligned interior*, which is the part that can serve
        // the request, rather than over the region as a whole.
        let mut best: Option<(usize, u64, u64)> = None;
        for (i, r) in self.regions.iter().enumerate() {
            if let Some((start, span)) = r.aligned_interior(align)
                && span >= len
                && best.is_none_or(|(_, _, b)| span < b)
            {
                best = Some((i, start, span));
            }
        }
        let (i, addr, _) = best?;
        let r = self.regions[i];
        let mut replacement = Vec::with_capacity(2);
        if addr > r.addr {
            replacement.push(FreeRegion {
                addr: r.addr,
                len: addr - r.addr,
            });
        }
        if r.end() > addr + len {
            replacement.push(FreeRegion {
                addr: addr + len,
                len: r.end() - (addr + len),
            });
        }
        self.regions.splice(i..=i, replacement);
        Some(addr)
    }

    /// Remove whatever part of `[addr, addr + len)` this list holds, leaving the
    /// parts of each overlapped region that fall outside it free.
    ///
    /// Unlike [`alloc`](Self::alloc) this reserves a *stated* range rather than
    /// asking for one, and unlike a failed `alloc` it is not an error for the
    /// range to be free only in part (or not at all): it is how a caller that has
    /// decided a range's fate elsewhere makes the list agree. The paged editor
    /// uses it to lift a whole free page out of the per-page-type lists before
    /// re-filing it as one typeless free page (`PagedEdit::promote_whole_free_pages`).
    ///
    /// A zero-length range is a no-op.
    pub(crate) fn take_range(&mut self, addr: u64, len: u64) {
        if len == 0 {
            return;
        }
        let end = addr + len;
        let mut out = Vec::with_capacity(self.regions.len() + 1);
        for r in self.regions.drain(..) {
            // Disjoint from the range: keep the region whole.
            if r.end() <= addr || r.addr >= end {
                out.push(r);
                continue;
            }
            // Overlapping: keep whatever lies below and above the range. Either
            // side may be empty, and both are when the range covers the region.
            if r.addr < addr {
                out.push(FreeRegion {
                    addr: r.addr,
                    len: addr - r.addr,
                });
            }
            if r.end() > end {
                out.push(FreeRegion {
                    addr: end,
                    len: r.end() - end,
                });
            }
        }
        self.regions = out;
    }

    /// The free regions as `(addr, len)` pairs, sorted ascending by address and
    /// fully coalesced. Used to persist the free list to disk (issue #21) and to
    /// report the session's live reusable free space (issue #150).
    pub(crate) fn sections(&self) -> Vec<(u64, u64)> {
        self.regions.iter().map(|r| (r.addr, r.len)).collect()
    }

    /// Whether this list holds no free space at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The largest single run this list could satisfy an allocation from, or `0`
    /// when it is empty. Allocation is best-fit over *contiguous* regions, so a
    /// list holding plenty of bytes in small pieces can still refuse a large
    /// request — which is the question an in-place append's reserve has to ask
    /// before it decides whether to draw more (issue #387).
    pub(crate) fn largest(&self) -> u64 {
        self.regions.iter().map(|r| r.len).max().unwrap_or(0)
    }

    /// The largest run of whole `align`-sized units inside any one region — the
    /// most [`alloc_whole_units`](Self::alloc_whole_units) could hand out in a
    /// single call — or `0` when no region holds a whole unit. `align` of 0
    /// reports `0`, as that allocator refuses it.
    ///
    /// The whole-page counterpart of [`largest`](Self::largest), for the same
    /// caller: on a paged file an append's reserve may claim whole free pages
    /// out of the other page type's list, so how much it can draw is the larger
    /// of this and its own list's `largest`.
    pub(crate) fn largest_whole_units(&self, align: u64) -> u64 {
        if align == 0 {
            return 0;
        }
        self.regions
            .iter()
            .filter_map(|r| r.aligned_interior(align).map(|(_, span)| span))
            .max()
            .unwrap_or(0)
    }

    /// If a free region ends exactly at `eof` (the current end-of-file), remove
    /// it from the list and return its start address — the file can be truncated
    /// to that address. Returns `None` if the highest free region does not reach
    /// end-of-file.
    ///
    /// Because the list is coalesced, at most one region can end at `eof`, and it
    /// is the last one.
    pub(crate) fn take_trailing(&mut self, eof: u64) -> Option<u64> {
        match self.regions.last() {
            Some(last) if last.end() == eof => {
                let addr = last.addr;
                self.regions.pop();
                Some(addr)
            }
            _ => None,
        }
    }

    /// Returns `true` if a region ends at `addr` or starts at `addr + len`.
    fn adjoins(&self, addr: u64, len: u64) -> bool {
        let end = addr + len;
        self.regions
            .iter()
            .any(|r| r.end() == addr || r.addr == end)
    }
}

/// What [`FreeList::release`] does with a freed extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Release {
    /// The extent is not recorded.
    Drop,
    /// The extent is shorter than the threshold and is recorded with a region it adjoins.
    Merge,
    /// The extent is at least the threshold and is recorded.
    Track,
}

/// The lowest address of the run of free space that reaches `eof`, across
/// several lists that are individually coalesced and mutually disjoint. `eof`
/// itself when nothing there is free.
///
/// The paged counterpart of [`FreeList::take_trailing`]. A paged file keeps free
/// space in one list per page type, plus space whose page type is unproven and
/// space it may record but never hand out ([`crate::edit`]), so the run at the
/// end of the file can be split across them — free metadata, then free raw data,
/// then a dead fragment — and no single list sees it whole. Only the union
/// answers whether the file's last bytes are all unreferenced, which is what a
/// shrink turns on.
///
/// The regions are walked from the top, joining those that touch: because each
/// list is coalesced and the lists do not overlap, a region can only extend the
/// run when its end meets the run's current start, so one descending pass is
/// exact.
pub(crate) fn trailing_run_start<'a, I>(lists: I, eof: u64) -> u64
where
    I: IntoIterator<Item = &'a FreeList>,
{
    let mut all: Vec<FreeRegion> = lists
        .into_iter()
        .flat_map(|l| l.regions.iter().copied())
        .collect();
    all.sort_unstable_by_key(|r| r.addr);
    let mut start = eof;
    for r in all.iter().rev() {
        if r.end() < start {
            break;
        }
        start = start.min(r.addr);
    }
    start
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Expose the canonical region list as `(addr, len)` pairs for assertions.
    fn regions(fl: &FreeList) -> Vec<(u64, u64)> {
        fl.regions.iter().map(|r| (r.addr, r.len)).collect()
    }

    #[test]
    fn free_into_empty_list() {
        let mut fl = FreeList::new();
        fl.free(100, 50);
        assert_eq!(regions(&fl), [(100, 50)]);
    }

    #[test]
    fn zero_length_free_is_noop() {
        let mut fl = FreeList::new();
        fl.free(100, 0);
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn disjoint_frees_stay_sorted() {
        let mut fl = FreeList::new();
        fl.free(300, 10);
        fl.free(100, 10);
        fl.free(200, 10);
        assert_eq!(regions(&fl), [(100, 10), (200, 10), (300, 10)]);
    }

    #[test]
    fn coalesce_with_right_neighbor() {
        let mut fl = FreeList::new();
        fl.free(200, 50); // [200, 250)
        fl.free(150, 50); // [150, 200) touches left edge of the above
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_with_left_neighbor() {
        let mut fl = FreeList::new();
        fl.free(150, 50); // [150, 200)
        fl.free(200, 50); // [200, 250) touches right edge of the above
        assert_eq!(regions(&fl), [(150, 100)]);
    }

    #[test]
    fn coalesce_bridges_gap_between_two() {
        let mut fl = FreeList::new();
        fl.free(100, 50); // [100, 150)
        fl.free(250, 50); // [250, 300)
        fl.free(150, 100); // [150, 250) bridges the two
        assert_eq!(regions(&fl), [(100, 200)]);
    }

    #[test]
    fn no_coalesce_when_gap_remains() {
        let mut fl = FreeList::new();
        fl.free(100, 50); // [100, 150)
        fl.free(151, 50); // [151, 201) one byte gap
        assert_eq!(regions(&fl), [(100, 50), (151, 50)]);
    }

    #[test]
    fn alloc_best_fit_chooses_smallest_sufficient() {
        let mut fl = FreeList::new();
        fl.free(0, 100); // big
        fl.free(200, 30); // exact-ish, smallest that fits 30
        fl.free(400, 60); // medium
        let addr = fl.alloc(30).unwrap();
        assert_eq!(addr, 200);
        // The 30-region is consumed exactly; the others remain.
        assert_eq!(regions(&fl), [(0, 100), (400, 60)]);
    }

    #[test]
    fn alloc_splits_remainder() {
        let mut fl = FreeList::new();
        fl.free(1000, 100);
        let addr = fl.alloc(40).unwrap();
        assert_eq!(addr, 1000);
        assert_eq!(regions(&fl), [(1040, 60)]);
    }

    #[test]
    fn alloc_none_when_nothing_fits() {
        let mut fl = FreeList::new();
        fl.free(0, 10);
        fl.free(100, 20);
        assert!(fl.alloc(50).is_none());
        // List is unchanged on a failed allocation.
        assert_eq!(regions(&fl), [(0, 10), (100, 20)]);
    }

    #[test]
    fn alloc_zero_returns_none() {
        let mut fl = FreeList::new();
        fl.free(0, 100);
        assert!(fl.alloc(0).is_none());
    }

    #[test]
    fn alloc_then_free_roundtrips() {
        let mut fl = FreeList::new();
        fl.free(0, 100);
        let a = fl.alloc(40).unwrap();
        fl.free(a, 40); // give it back
        assert_eq!(regions(&fl), [(0, 100)]); // coalesced back to whole
    }

    #[test]
    fn is_empty_and_largest_report_the_list() {
        let mut fl = FreeList::new();
        assert!(fl.is_empty());
        assert_eq!(fl.largest(), 0);
        fl.free(100, 10);
        fl.free(200, 40);
        fl.free(400, 25);
        assert!(!fl.is_empty());
        assert_eq!(fl.largest(), 40);
        // Coalescing is what `largest` reports over, not the frees as issued.
        fl.free(240, 40);
        assert_eq!(fl.largest(), 80);
    }

    /// `largest_whole_units` reports exactly what `alloc_whole_units` could take
    /// in one call: the aligned interior, not the region.
    #[test]
    fn largest_whole_units_reports_the_aligned_interior() {
        let mut fl = FreeList::new();
        assert_eq!(fl.largest_whole_units(16), 0);
        // [10, 40): thirty bytes, and the only whole unit in it is [16, 32).
        fl.free(10, 30);
        assert_eq!(fl.largest(), 30);
        assert_eq!(fl.largest_whole_units(16), 16);
        // [100, 110): ten bytes with no whole unit at all.
        fl.free(100, 10);
        assert_eq!(fl.largest_whole_units(16), 16);
        // [200, 260): sixty bytes whose whole units are [208, 256), three of them.
        fl.free(200, 60);
        assert_eq!(fl.largest_whole_units(16), 48);
        assert_eq!(fl.largest_whole_units(0), 0);
        // The allocator agrees: that run is claimable in one call, and one unit
        // more is not.
        let mut probe = fl.clone();
        assert_eq!(probe.alloc_whole_units(64, 16), None);
        assert_eq!(probe.alloc_whole_units(48, 16), Some(208));
    }

    #[test]
    fn take_trailing_at_eof() {
        let mut fl = FreeList::new();
        fl.free(500, 100); // [500, 600)
        let cut = fl.take_trailing(600);
        assert_eq!(cut, Some(500));
        assert!(regions(&fl).is_empty());
    }

    #[test]
    fn take_trailing_none_when_not_at_eof() {
        let mut fl = FreeList::new();
        fl.free(500, 100); // [500, 600)
        assert_eq!(fl.take_trailing(700), None); // live bytes between 600 and 700
        assert_eq!(regions(&fl), [(500, 100)]); // unchanged
    }

    #[test]
    fn take_trailing_only_cuts_the_tail_region() {
        let mut fl = FreeList::new();
        fl.free(100, 50); // interior hole [100, 150)
        fl.free(500, 100); // trailing [500, 600)
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
        meta.free(400, 100); // [400, 500)
        raw.free(500, 50); // [500, 550)
        dead.free(550, 50); // [550, 600)
        assert_eq!(trailing_run_start([&meta, &raw, &dead], 600), 400);
    }

    #[test]
    fn trailing_run_start_stops_at_the_first_live_gap() {
        let (mut meta, mut raw) = (FreeList::new(), FreeList::new());
        meta.free(100, 100); // [100, 200), below a live gap
        raw.free(400, 200); // [400, 600)
        assert_eq!(trailing_run_start([&meta, &raw], 600), 400);
    }

    #[test]
    fn trailing_run_start_is_eof_when_the_last_byte_is_live() {
        let mut meta = FreeList::new();
        meta.free(400, 100); // [400, 500), then live bytes to 600
        assert_eq!(trailing_run_start([&meta], 600), 600);
        assert_eq!(trailing_run_start([&FreeList::new()], 600), 600);
    }

    #[test]
    fn take_range_splits_the_region_around_it() {
        let mut fl = FreeList::new();
        fl.free(100, 100); // [100, 200)
        fl.take_range(120, 30); // [120, 150)
        assert_eq!(regions(&fl), [(100, 20), (150, 50)]);
    }

    #[test]
    fn take_range_spanning_several_regions_keeps_only_the_edges() {
        let mut fl = FreeList::new();
        fl.free(100, 50); // [100, 150)
        fl.free(200, 50); // [200, 250)
        fl.free(300, 50); // [300, 350)
        fl.take_range(120, 200); // [120, 320)
        assert_eq!(regions(&fl), [(100, 20), (320, 30)]);
    }

    #[test]
    fn take_range_tolerates_a_range_that_is_not_free() {
        let mut fl = FreeList::new();
        fl.free(100, 50);
        fl.take_range(300, 50); // wholly outside the list
        fl.take_range(140, 20); // half inside it
        fl.take_range(0, 0); // empty
        assert_eq!(regions(&fl), [(100, 40)]);
    }

    #[rstest]
    #[case::below_the_threshold(THRESHOLD - 1, Release::Drop, &[(0, 100)])]
    #[case::at_the_threshold(THRESHOLD, Release::Track, &[(0, 100), (500, THRESHOLD)])]
    #[case::above_the_threshold(
        THRESHOLD + 1,
        Release::Track,
        &[(0, 100), (500, THRESHOLD + 1)]
    )]
    fn an_isolated_extent_is_tracked_from_the_threshold_up(
        #[case] len: u64,
        #[case] release: Release,
        #[case] expected: &[(u64, u64)],
    ) {
        let mut fl = FreeList::new();
        fl.free(0, 100);

        assert_eq!(fl.release(THRESHOLD, 500, len), release);
        assert_eq!(regions(&fl), expected);
    }

    #[rstest]
    #[case::below_it(&[(100, 100)], (50, 50), &[(50, 150)])]
    #[case::above_it(&[(100, 100)], (200, 50), &[(100, 150)])]
    #[case::between_two(&[(100, 100), (250, 100)], (200, 50), &[(100, 250)])]
    fn a_sub_threshold_extent_merges_into_an_adjoining_tracked_section(
        #[case] tracked: &[(u64, u64)],
        #[case] (addr, len): (u64, u64),
        #[case] expected: &[(u64, u64)],
    ) {
        let mut fl = FreeList::new();
        for &(addr, len) in tracked {
            fl.free(addr, len);
        }

        assert_eq!(fl.release(THRESHOLD, addr, len), Release::Merge);
        assert_eq!(regions(&fl), expected);
    }

    #[test]
    fn adjoining_sub_threshold_extents_are_both_dropped() {
        let mut fl = FreeList::new();

        assert_eq!(
            [
                fl.release(THRESHOLD, 0, THRESHOLD - 1),
                fl.release(THRESHOLD, THRESHOLD - 1, THRESHOLD - 1),
            ],
            [Release::Drop, Release::Drop]
        );
        assert!(fl.is_empty());
    }

    #[test]
    fn an_empty_extent_is_dropped() {
        let mut fl = FreeList::new();

        assert_eq!(fl.release(0, 100, 0), Release::Drop);
        assert!(fl.is_empty());
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
            fl.free(addr, len);
        }

        assert_eq!(
            fl.release_all(THRESHOLD, 950, extents.iter().copied()),
            expected_eoa
        );
        assert_eq!(regions(&fl), expected_regions);
    }

    const THRESHOLD: u64 = 64;
}
