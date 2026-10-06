//! Session-local free-space tracking for in-place editing (issue #21).
//!
//! [`File::open_rw`](crate::File::open_rw) writes by appending at end-of-file and,
//! on each commit, leaves the superseded object headers and any deleted-object
//! blocks behind as dead bytes. This module records those freed regions so a
//! later allocation can reuse them instead of growing the file, and so a run of
//! free space that reaches end-of-file can be truncated away. A session records
//! them only under a strategy with free-space managers,
//! `H5F_FSPACE_STRATEGY_FSM_AGGR` or `H5F_FSPACE_STRATEGY_PAGE`.
//!
//! It is the in-memory half of HDF5's "free-space management". For a file opened
//! without persistence (the default) it is purely session-local: freed-but-
//! unreused space is invisible to other tools, exactly as the reference C
//! library's default `FSM_AGGR` strategy with persistence off leaves it. When the
//! file was created with `persist = true`, [`File::open_rw`](crate::File::open_rw)
//! seeds this list from the on-disk free-space managers (the `FSHD`/`FSSE` blocks
//! the File Space Info superblock-extension message points at) on open and writes
//! it back on each commit, so reuse spans sessions (see
//! [`free_space_manager`](crate::free_space_manager)).
//!
//! The structure is a sorted, fully coalesced list of disjoint `[addr, addr+len)`
//! regions. Every public operation preserves both invariants (sorted by address,
//! no two regions touching or overlapping), so the list is always in a canonical
//! form and `trailing_free` is a single comparison against the highest region.

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
}
