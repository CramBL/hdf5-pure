use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;

use super::TRAILING_RESERVE_TAILS;
use super::admission::TrackedSpace;
use super::extent::Extent;
use super::list::{self, FreeList};

/// Identifies whether an allocation belongs to metadata or raw-data pages.
///
/// A paged file keeps the two kinds of allocation in separate pages and tracks their reusable
/// space independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PageType {
    /// File metadata.
    Meta,
    /// Raw dataset data.
    Raw,
}

/// Classifies a vacated extent by the PAGE reuse policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FreeClass {
    /// The extent is dead but its page type is not known.
    Dead,
    /// The extent belongs to pages of this type.
    Page(PageType),
}

impl From<PageType> for FreeClass {
    fn from(ty: PageType) -> Self {
        Self::Page(ty)
    }
}

/// Describes the padding required before an append changes the tail page type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PageTransition {
    /// Padding is required before the append.
    Pad {
        extent: Extent,
        track_as: Option<PageType>,
    },
    /// The append may use the current tail page.
    Unchanged,
}

/// Snapshots the free sections that map to PAGE managers.
pub(crate) struct PagedSections {
    metadata: Vec<(u64, u64)>,
    raw: Vec<(u64, u64)>,
    unclassified: Vec<(u64, u64)>,
}

impl PagedSections {
    pub(crate) fn metadata(&self) -> &[(u64, u64)] {
        &self.metadata
    }

    pub(crate) fn raw(&self) -> &[(u64, u64)] {
        &self.raw
    }

    pub(crate) fn unclassified(&self) -> &[(u64, u64)] {
        &self.unclassified
    }
}

/// Stores reusable allocation state for rollback after a failed commit attempt.
pub(crate) struct PagedAllocationSnapshot {
    meta: FreeList,
    raw: FreeList,
}

/// Tracks reusable-space state for a paged editing session.
pub(crate) struct PagedEdit {
    page_size: FileSpacePageSize,
    meta: FreeList,
    raw: FreeList,
    unclassified: FreeList,
    dead: FreeList,
    last: Option<PageType>,
    meta_pad: Vec<Extent>,
    raw_pad: Vec<Extent>,
}

impl PagedEdit {
    pub(crate) fn new(page_size: FileSpacePageSize) -> Self {
        Self {
            page_size,
            meta: FreeList::new(),
            raw: FreeList::new(),
            unclassified: FreeList::new(),
            dead: FreeList::new(),
            last: None,
            meta_pad: Vec::new(),
            raw_pad: Vec::new(),
        }
    }

    pub(crate) fn page_size(&self) -> FileSpacePageSize {
        self.page_size
    }

    /// Returns read-only PAGE sections for assertions.
    #[cfg(test)]
    pub(crate) fn sections(&self) -> PagedSections {
        PagedSections {
            metadata: self.meta.sections(),
            raw: self.raw.sections(),
            unclassified: self.unclassified.sections(),
        }
    }

    /// Returns the type of the current tail page when this session established it.
    #[cfg(test)]
    pub(crate) fn tail_type(&self) -> Option<PageType> {
        self.last
    }

    /// Returns pending page-tail padding for `ty`.
    #[cfg(test)]
    pub(crate) fn pending_padding(&self, ty: PageType) -> &[Extent] {
        match ty {
            PageType::Meta => &self.meta_pad,
            PageType::Raw => &self.raw_pad,
        }
    }

    /// Returns the transition required before an append of `ty` at `eoa`.
    pub(crate) fn plan_transition(
        &self,
        eoa: u64,
        ty: PageType,
    ) -> Result<PageTransition, FormatError> {
        let page_size = self.page_size.get();
        if eoa % page_size == 0 || self.last == Some(ty) {
            return Ok(PageTransition::Unchanged);
        }
        let pad_len = page_size - eoa % page_size;
        let extent = Extent::new(eoa, pad_len).ok_or(FormatError::OffsetOverflow {
            offset: eoa,
            length: pad_len,
        })?;
        Ok(PageTransition::Pad {
            extent,
            track_as: self.last,
        })
    }

    /// Records a transition after its padding has been written successfully.
    pub(crate) fn record_transition(&mut self, ty: PageType, transition: PageTransition) {
        if let PageTransition::Pad { extent, track_as } = transition {
            self.record_padding(extent, track_as);
        }
        self.last = Some(ty);
    }

    /// Returns the padding needed to end the file on a page boundary.
    pub(crate) fn page_padding(&self, eoa: u64) -> Result<Option<PageTransition>, FormatError> {
        let page_size = self.page_size.get();
        if eoa % page_size == 0 {
            return Ok(None);
        }
        let pad_len = page_size - eoa % page_size;
        let extent = Extent::new(eoa, pad_len).ok_or(FormatError::OffsetOverflow {
            offset: eoa,
            length: pad_len,
        })?;
        Ok(Some(PageTransition::Pad {
            extent,
            track_as: self.last,
        }))
    }

    /// Records page-boundary padding without changing the tail page type.
    pub(crate) fn record_page_padding(&mut self, transition: PageTransition) {
        if let PageTransition::Pad { extent, track_as } = transition {
            self.record_padding(extent, track_as);
        }
    }

    fn record_padding(&mut self, extent: Extent, track_as: Option<PageType>) {
        match track_as {
            Some(PageType::Meta) => self.meta_pad.push(extent),
            Some(PageType::Raw) => self.raw_pad.push(extent),
            None => {}
        }
    }

    /// Seeds one extent from persistent PAGE manager slot `slot`.
    ///
    /// Slots 0 and 2 establish metadata and raw-data pages. Slot 6 establishes raw reuse only for
    /// whole aligned pages because the generic-large manager does not record a page type. Other
    /// sections remain unclassified and are never allocated independently.
    pub(crate) fn seed_persisted(&mut self, slot: usize, extent: Extent) {
        let page_size = self.page_size.get();
        let ty = match slot {
            0 => Some(PageType::Meta),
            2 => Some(PageType::Raw),
            6 if extent.start() % page_size == 0 && extent.len() % page_size == 0 => {
                Some(PageType::Raw)
            }
            _ => None,
        };
        self.seed(extent, ty);
    }

    /// Seeds one extent with its established page type.
    pub(crate) fn seed(&mut self, extent: Extent, ty: Option<PageType>) {
        match ty {
            Some(ty) => self.track(extent, ty.into()),
            None => self.unclassified.free(extent),
        }
    }

    /// Promotes every page that the seeded lists show to be wholly empty.
    pub(crate) fn finish_seed(&mut self) {
        Self::promote_whole_free_pages(
            &mut self.meta,
            &mut self.raw,
            &mut self.dead,
            self.page_size,
        );
    }

    /// Allocates `len` bytes that may be used for `ty`.
    pub(crate) fn allocate(&mut self, len: u64, ty: PageType) -> Option<Extent> {
        let (own, other) = match ty {
            PageType::Meta => (&mut self.meta, &mut self.raw),
            PageType::Raw => (&mut self.raw, &mut self.meta),
        };
        if let Some(extent) = own.alloc(len) {
            return Some(extent);
        }
        let span = len.checked_next_multiple_of(self.page_size.get())?;
        let claimed = other.alloc_whole_units(span, self.page_size.get())?;
        if span == len {
            return Some(claimed);
        }
        let allocation = Extent::new(claimed.start(), len)
            .expect("the allocation is a non-empty prefix of the claimed extent");
        let remainder = Extent::new(allocation.end(), span - len)
            .expect("the remainder is a non-empty suffix of the claimed extent");
        own.free(remainder);
        Some(allocation)
    }

    /// Returns the longest run that [`Self::allocate`] can serve for `ty`.
    pub(crate) fn largest(&self, ty: PageType) -> u64 {
        let (own, other) = match ty {
            PageType::Meta => (&self.meta, &self.raw),
            PageType::Raw => (&self.raw, &self.meta),
        };
        own.largest()
            .max(other.largest_whole_units(self.page_size.get()))
    }

    /// Returns an allocation to the free space of `ty`.
    pub(crate) fn return_allocation(&mut self, extent: Extent, ty: PageType) {
        self.track(extent, ty.into());
    }

    /// Returns reusable sections in ascending address order.
    pub(crate) fn reusable_sections(&self) -> Vec<(u64, u64)> {
        let mut out = self.meta.sections();
        out.extend(self.raw.sections());
        out.sort_unstable_by_key(|&(addr, _)| addr);
        debug_assert!(
            out.windows(2)
                .all(|w| w[0].0.saturating_add(w[0].1) <= w[1].0),
            "a region is free in both the metadata and the raw list, so the two \
             page-type lists have stopped being disjoint"
        );
        out
    }

    /// Returns reusable sections with additional raw extents folded into the snapshot.
    pub(crate) fn reusable_sections_with_raw(
        &self,
        raw_extents: impl IntoIterator<Item = Extent>,
    ) -> Vec<(u64, u64)> {
        let mut raw = self.raw.clone();
        for extent in raw_extents {
            raw.free(extent);
        }
        let mut out = self.meta.sections();
        out.extend(raw.sections());
        out.sort_unstable_by_key(|&(addr, _)| addr);
        debug_assert!(
            out.windows(2).all(|w| w[0].0 < w[1].0),
            "the same address is free in both the metadata and the raw list, so the two \
             page-type lists have stopped being disjoint"
        );
        out
    }

    /// Returns the start of the free run that reaches `eoa` across all PAGE lists.
    pub(crate) fn trailing_run_start(&self, eoa: u64) -> u64 {
        list::trailing_run_start([&self.meta, &self.raw, &self.dead, &self.unclassified], eoa)
    }

    /// Returns the allocation state that a failed commit may restore.
    pub(crate) fn allocation_snapshot(&self) -> PagedAllocationSnapshot {
        PagedAllocationSnapshot {
            meta: self.meta.clone(),
            raw: self.raw.clone(),
        }
    }

    /// Restores reusable allocations after a commit fails before publication.
    pub(crate) fn restore_allocation(&mut self, snapshot: PagedAllocationSnapshot) {
        self.meta = snapshot.meta;
        self.raw = snapshot.raw;
    }

    /// Builds the PAGE free-space state that a commit would publish.
    pub(crate) fn post_free(
        &self,
        threshold: u64,
        eoa: u64,
        freed: impl IntoIterator<Item = (Extent, FreeClass)>,
        old_blocks: impl IntoIterator<Item = Extent>,
    ) -> (PagedPostFree, u64) {
        let mut post = PagedPostFree {
            page_size: self.page_size,
            meta: self.meta.clone(),
            raw: self.raw.clone(),
            dead: self.dead.clone(),
            unclassified: self.unclassified.clone(),
        };
        for &extent in &self.meta_pad {
            post.track(extent, PageType::Meta.into());
        }
        for &extent in &self.raw_pad {
            post.track(extent, PageType::Raw.into());
        }
        let old_blocks = old_blocks
            .into_iter()
            .map(|extent| (extent, FreeClass::Page(PageType::Meta)));
        let eoa = post.release_all(threshold, eoa, freed.into_iter().chain(old_blocks));
        Self::promote_whole_free_pages(
            &mut post.meta,
            &mut post.raw,
            &mut post.dead,
            post.page_size,
        );
        (post, eoa)
    }

    /// Adopts a published post-free state and records an appended metadata tail.
    pub(crate) fn adopt_post_free(
        &mut self,
        post: PagedPostFree,
        appended_metadata: bool,
        appended_tail: Option<Extent>,
    ) {
        self.meta = post.meta;
        self.raw = post.raw;
        self.dead = post.dead;
        self.unclassified = post.unclassified;
        self.meta_pad.clear();
        self.raw_pad.clear();
        if appended_metadata {
            self.last = Some(PageType::Meta);
        }
        if let Some(tail) = appended_tail {
            self.meta.free(tail);
        }
    }

    fn track(&mut self, extent: Extent, class: FreeClass) {
        Self::route_free(&mut self.meta, &mut self.raw, &mut self.dead, extent, class);
    }

    fn route_free(
        meta: &mut FreeList,
        raw: &mut FreeList,
        dead: &mut FreeList,
        extent: Extent,
        class: FreeClass,
    ) {
        match class {
            FreeClass::Dead => dead.free(extent),
            FreeClass::Page(PageType::Meta) => meta.free(extent),
            FreeClass::Page(PageType::Raw) => raw.free(extent),
        }
    }

    fn promote_whole_free_pages(
        meta: &mut FreeList,
        raw: &mut FreeList,
        dead: &mut FreeList,
        page_size: FileSpacePageSize,
    ) {
        let page_size = page_size.get();
        let mut all = meta.extents();
        all.extend(raw.extents());
        all.extend(dead.extents());
        all.sort_unstable_by_key(|extent| extent.start());
        let mut runs: Vec<Extent> = Vec::with_capacity(all.len());
        for extent in all {
            match runs.last_mut() {
                Some(run) if run.end() >= extent.start() => {
                    *run = Extent::new(run.start(), run.end().max(extent.end()) - run.start())
                        .expect("merged free extents stay non-empty and in bounds");
                }
                _ => runs.push(extent),
            }
        }
        for extent in runs {
            let Some(first) = extent.start().checked_next_multiple_of(page_size) else {
                continue;
            };
            let last = (extent.end() / page_size) * page_size;
            let Some(whole_pages) = Extent::new(first, last.saturating_sub(first)) else {
                continue;
            };
            meta.take_range(whole_pages);
            raw.take_range(whole_pages);
            dead.take_range(whole_pages);
            raw.free(whole_pages);
        }
    }
}

/// Holds the free-space state a paged commit is preparing to publish.
pub(crate) struct PagedPostFree {
    page_size: FileSpacePageSize,
    meta: FreeList,
    raw: FreeList,
    dead: FreeList,
    unclassified: FreeList,
}

impl PagedPostFree {
    /// Returns read-only sections for persistence planning.
    pub(crate) fn sections(&self) -> PagedSections {
        PagedSections {
            metadata: self.meta.sections(),
            raw: self.raw.sections(),
            unclassified: self.unclassified.sections(),
        }
    }

    /// Releases the trailing whole-page run while retaining manager-tail reserve space.
    pub(crate) fn release_trailing(&mut self, eoa: u64, tail_len: u64) -> u64 {
        let start =
            list::trailing_run_start([&self.meta, &self.raw, &self.dead, &self.unclassified], eoa);
        let page_size = self.page_size.get();
        let cut = start.div_ceil(page_size) * page_size;
        if cut >= eoa {
            return eoa;
        }
        let reserve = TRAILING_RESERVE_TAILS * tail_len;
        let keep = reserve.div_ceil(page_size) * page_size;
        if eoa - cut < 2 * keep {
            return eoa;
        }
        let published_eoa = cut + keep;
        let released = Extent::new(published_eoa, eoa - published_eoa)
            .expect("the released range ends at the current end of allocation");
        self.meta.take_range(released);
        self.raw.take_range(released);
        self.dead.take_range(released);
        self.unclassified.take_range(released);
        published_eoa
    }

    fn release_all(
        &mut self,
        threshold: u64,
        eoa: u64,
        extents: impl IntoIterator<Item = (Extent, FreeClass)>,
    ) -> u64 {
        let extents: Vec<_> = extents.into_iter().collect();
        let mut dropped = self.release_each(threshold, extents.iter().copied());
        if threshold > self.page_size.get() {
            let completed = self.pages_completed_by_merges(&extents, &dropped);
            dropped.extend(self.release_pages(threshold, completed));
        }
        self.give_back(eoa, dropped)
    }

    fn pages_completed_by_merges(
        &self,
        extents: &[(Extent, FreeClass)],
        dropped: &[(Extent, FreeClass)],
    ) -> BTreeSet<Extent> {
        let page_size = self.page_size.get();
        let dropped: BTreeSet<_> = dropped.iter().map(|&(extent, _)| extent).collect();
        extents
            .iter()
            .map(|&(extent, _)| extent)
            .filter(|extent| (1..page_size).contains(&extent.len()) && !dropped.contains(extent))
            .flat_map(|extent| [extent.start(), extent.end() - 1])
            .map(|addr| addr / page_size * page_size)
            .filter_map(|page| Extent::new(page, page_size))
            .filter(|&page| {
                [&self.meta, &self.raw, &self.dead]
                    .into_iter()
                    .map(|list| list.covered_len(page))
                    .sum::<u64>()
                    == page_size
            })
            .collect()
    }

    fn release_pages(
        &mut self,
        threshold: u64,
        pages: BTreeSet<Extent>,
    ) -> Vec<(Extent, FreeClass)> {
        let pages: Vec<_> = pages.into_iter().collect();
        for &page in &pages {
            for list in [&mut self.meta, &mut self.raw, &mut self.dead] {
                list.take_range(page);
            }
        }
        self.release_each(
            threshold,
            pages
                .into_iter()
                .map(|page| (page, FreeClass::Page(PageType::Raw))),
        )
    }

    fn give_back(&mut self, eoa: u64, mut dropped: Vec<(Extent, FreeClass)>) -> u64 {
        let page_size = self.page_size.get();
        dropped.retain(|&(extent, _)| extent.len() >= page_size);
        dropped.sort_unstable_by_key(|&(extent, _)| Reverse(extent.start()));
        let mut start = eoa;
        let mut head = None;
        for (extent, class) in dropped {
            if extent.end() != start {
                break;
            }
            start = extent.start();
            head = Some(class);
        }
        if let Some(class) = head.filter(|_| !start.is_multiple_of(page_size)) {
            let aligned = start.div_ceil(page_size) * page_size;
            let head = Extent::new(start, aligned - start)
                .expect("page alignment leaves a non-empty prefix");
            self.track(head, class);
        }
        start.div_ceil(page_size) * page_size
    }
}

impl TrackedSpace for PagedPostFree {
    type Class = FreeClass;

    fn merges(&self, extent: Extent) -> bool {
        let page_size = self.page_size.get();
        let typed = [&self.meta, &self.raw, &self.dead];
        if extent.len() < page_size {
            return typed
                .into_iter()
                .any(|list| list.adjoins_within(extent, page_size));
        }
        typed
            .into_iter()
            .any(|list| list.adjoins_whole_unit(extent, page_size))
            || self.unclassified.adjoins(extent)
    }

    fn track(&mut self, extent: Extent, class: FreeClass) {
        PagedEdit::route_free(&mut self.meta, &mut self.raw, &mut self.dead, extent, class);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::free_space::Release;

    const PAGE: u64 = FileSpacePageSize::DEFAULT.get();
    const META: FreeClass = FreeClass::Page(PageType::Meta);
    const RAW: FreeClass = FreeClass::Page(PageType::Raw);
    const THRESHOLD_IN_A_PAGE: u64 = 64;
    const SHORT_OF_THRESHOLD: u64 = THRESHOLD_IN_A_PAGE - 1;
    const EXTENT_AT: u64 = PAGE + 1024;
    const THRESHOLD_OF_THREE_PAGES: u64 = 3 * PAGE;

    #[test]
    fn an_aligned_page_transition_needs_no_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Meta);
        assert_eq!(
            paged.plan_transition(PAGE, PageType::Raw).unwrap(),
            PageTransition::Unchanged
        );
    }

    #[test]
    fn a_same_type_partial_page_transition_needs_no_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Raw);
        assert_eq!(
            paged.plan_transition(5000, PageType::Raw).unwrap(),
            PageTransition::Unchanged
        );
    }

    #[test]
    fn a_metadata_to_raw_transition_tracks_metadata_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Meta);
        assert_eq!(
            paged.plan_transition(5000, PageType::Raw).unwrap(),
            PageTransition::Pad {
                extent: Extent::new(5000, 3192).unwrap(),
                track_as: Some(PageType::Meta),
            }
        );
    }

    #[test]
    fn a_raw_to_metadata_transition_tracks_raw_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Raw);
        assert_eq!(
            paged.plan_transition(5000, PageType::Meta).unwrap(),
            PageTransition::Pad {
                extent: Extent::new(5000, 3192).unwrap(),
                track_as: Some(PageType::Raw),
            }
        );
    }

    #[test]
    fn an_unknown_tail_transition_leaves_padding_unclassified() {
        let paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        assert_eq!(
            paged.plan_transition(5000, PageType::Meta).unwrap(),
            PageTransition::Pad {
                extent: Extent::new(5000, 3192).unwrap(),
                track_as: None,
            }
        );
    }

    #[test]
    fn a_non_power_of_two_page_size_plans_exact_padding() {
        let page_size = FileSpacePageSize::try_from(1000).unwrap();
        let mut paged = PagedEdit::new(page_size);
        paged.last = Some(PageType::Meta);
        assert_eq!(
            paged.plan_transition(1500, PageType::Raw).unwrap(),
            PageTransition::Pad {
                extent: Extent::new(1500, 500).unwrap(),
                track_as: Some(PageType::Meta),
            }
        );
    }

    #[test]
    fn a_page_that_is_wholly_free_or_dead_is_promoted_whole() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.meta.free(Extent::new(PAGE, 2048).unwrap());
        paged.dead.free(Extent::new(PAGE + 2048, 2048).unwrap());
        paged.dead.free(Extent::new(2 * PAGE, PAGE).unwrap());
        paged.raw.free(Extent::new(3 * PAGE + 1024, 1024).unwrap());
        paged.finish_seed();
        assert_eq!(
            paged.raw.sections(),
            [(PAGE, 2 * PAGE), (3 * PAGE + 1024, 1024)]
        );
        assert!(paged.meta.sections().is_empty());
        assert!(paged.dead.sections().is_empty());
    }

    #[test]
    fn dead_space_short_of_a_whole_page_is_not_promoted() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.dead.free(Extent::new(PAGE, 512).unwrap());
        paged.raw.free(Extent::new(PAGE + 512, 1024).unwrap());
        paged.finish_seed();
        assert_eq!(paged.dead.sections(), [(PAGE, 512)]);
        assert_eq!(paged.raw.sections(), [(PAGE + 512, 1024)]);
    }

    #[rstest]
    #[case::metadata_short_of_it(META, THRESHOLD_IN_A_PAGE - 1, Release::Drop, [vec![], vec![], vec![]])]
    #[case::metadata_at_it(
        META,
        THRESHOLD_IN_A_PAGE,
        Release::Track,
        [vec![(EXTENT_AT, THRESHOLD_IN_A_PAGE)], vec![], vec![]]
    )]
    #[case::metadata_past_it(
        META,
        THRESHOLD_IN_A_PAGE + 1,
        Release::Track,
        [vec![(EXTENT_AT, THRESHOLD_IN_A_PAGE + 1)], vec![], vec![]]
    )]
    #[case::raw_short_of_it(RAW, THRESHOLD_IN_A_PAGE - 1, Release::Drop, [vec![], vec![], vec![]])]
    #[case::raw_at_it(
        RAW,
        THRESHOLD_IN_A_PAGE,
        Release::Track,
        [vec![], vec![(EXTENT_AT, THRESHOLD_IN_A_PAGE)], vec![]]
    )]
    #[case::dead_short_of_it(FreeClass::Dead, THRESHOLD_IN_A_PAGE - 1, Release::Drop, [vec![], vec![], vec![]])]
    #[case::dead_at_it(
        FreeClass::Dead,
        THRESHOLD_IN_A_PAGE,
        Release::Track,
        [vec![], vec![], vec![(EXTENT_AT, THRESHOLD_IN_A_PAGE)]]
    )]
    fn an_isolated_paged_extent_is_tracked_from_the_threshold_up(
        #[case] class: FreeClass,
        #[case] len: u64,
        #[case] release: Release,
        #[case] expected: [Vec<(u64, u64)>; 3],
    ) {
        let mut post = paged_free_space(&[vec![], vec![], vec![]]);
        assert_eq!(
            post.release(
                THRESHOLD_IN_A_PAGE,
                Extent::new(EXTENT_AT, len).unwrap(),
                class
            ),
            release
        );
        assert_eq!(paged_sections(&post), expected);
    }

    #[rstest]
    #[case::metadata_above_metadata(
        META,
        [vec![(EXTENT_AT - 100, 100)], vec![], vec![]],
        [vec![(EXTENT_AT - 100, 100 + SHORT_OF_THRESHOLD)], vec![], vec![]]
    )]
    #[case::raw_below_raw(
        RAW,
        [vec![], vec![(EXTENT_AT + SHORT_OF_THRESHOLD, 100)], vec![]],
        [vec![], vec![(EXTENT_AT, SHORT_OF_THRESHOLD + 100)], vec![]]
    )]
    #[case::dead_above_raw(
        FreeClass::Dead,
        [vec![], vec![(EXTENT_AT - 100, 100)], vec![]],
        [vec![], vec![(EXTENT_AT - 100, 100)], vec![(EXTENT_AT, SHORT_OF_THRESHOLD)]]
    )]
    #[case::raw_below_dead(
        RAW,
        [vec![], vec![], vec![(EXTENT_AT + SHORT_OF_THRESHOLD, 100)]],
        [vec![], vec![(EXTENT_AT, SHORT_OF_THRESHOLD)], vec![(EXTENT_AT + SHORT_OF_THRESHOLD, 100)]]
    )]
    fn a_sub_threshold_paged_extent_merges_beside_free_space_in_its_page(
        #[case] class: FreeClass,
        #[case] tracked: [Vec<(u64, u64)>; 3],
        #[case] expected: [Vec<(u64, u64)>; 3],
    ) {
        let mut post = paged_free_space(&tracked);
        assert_eq!(
            post.release(
                THRESHOLD_IN_A_PAGE,
                Extent::new(EXTENT_AT, SHORT_OF_THRESHOLD).unwrap(),
                class
            ),
            Release::Merge
        );
        assert_eq!(paged_sections(&post), expected);
    }

    #[rstest]
    #[case::metadata_above_metadata(META, PAGE, [vec![(PAGE - 100, 100)], vec![], vec![]])]
    #[case::raw_below_metadata(RAW, 2 * PAGE - SHORT_OF_THRESHOLD, [vec![(2 * PAGE, 100)], vec![], vec![]])]
    #[case::raw_below_a_free_page(
        RAW,
        2 * PAGE - SHORT_OF_THRESHOLD,
        [vec![], vec![(2 * PAGE, PAGE)], vec![]]
    )]
    #[case::dead_above_dead(FreeClass::Dead, PAGE, [vec![], vec![], vec![(PAGE - 100, 100)]])]
    fn a_sub_threshold_paged_extent_beside_free_space_across_a_page_boundary_is_dropped(
        #[case] class: FreeClass,
        #[case] at: u64,
        #[case] tracked: [Vec<(u64, u64)>; 3],
    ) {
        let mut post = paged_free_space(&tracked);
        assert_eq!(
            post.release(
                THRESHOLD_IN_A_PAGE,
                Extent::new(at, SHORT_OF_THRESHOLD).unwrap(),
                class
            ),
            Release::Drop
        );
        assert_eq!(paged_sections(&post), tracked);
    }

    #[rstest]
    #[case::metadata_above_a_metadata_page(
        META,
        [vec![(PAGE, PAGE)], vec![], vec![]],
        [vec![(PAGE, 3 * PAGE)], vec![], vec![]]
    )]
    #[case::raw_below_raw_pages(
        RAW,
        [vec![], vec![(4 * PAGE, 2 * PAGE)], vec![]],
        [vec![], vec![(2 * PAGE, 4 * PAGE)], vec![]]
    )]
    #[case::metadata_below_a_raw_page(
        META,
        [vec![], vec![(4 * PAGE, PAGE)], vec![]],
        [vec![(2 * PAGE, 2 * PAGE)], vec![(4 * PAGE, PAGE)], vec![]]
    )]
    fn a_paged_extent_of_pages_below_the_threshold_merges_beside_a_whole_free_page(
        #[case] class: FreeClass,
        #[case] tracked: [Vec<(u64, u64)>; 3],
        #[case] expected: [Vec<(u64, u64)>; 3],
    ) {
        let mut post = paged_free_space(&tracked);
        assert_eq!(
            post.release(
                THRESHOLD_OF_THREE_PAGES,
                Extent::new(2 * PAGE, 2 * PAGE).unwrap(),
                class
            ),
            Release::Merge
        );
        assert_eq!(paged_sections(&post), expected);
    }

    #[rstest]
    #[case::above_part_of_a_page(2 * PAGE, [vec![], vec![(2 * PAGE - 100, 100)], vec![]])]
    #[case::inside_a_page(2 * PAGE + 100, [vec![], vec![(2 * PAGE, 100)], vec![]])]
    fn a_paged_extent_of_pages_below_the_threshold_beside_part_of_a_page_is_dropped(
        #[case] at: u64,
        #[case] tracked: [Vec<(u64, u64)>; 3],
    ) {
        let mut post = paged_free_space(&tracked);
        assert_eq!(
            post.release(
                THRESHOLD_OF_THREE_PAGES,
                Extent::new(at, 2 * PAGE).unwrap(),
                RAW
            ),
            Release::Drop
        );
        assert_eq!(paged_sections(&post), tracked);
    }

    #[test]
    fn a_paged_extent_of_pages_below_the_threshold_merges_beside_an_unclassified_section() {
        let mut post = paged_free_space(&[vec![], vec![], vec![]]);
        post.unclassified.free(Extent::new(4 * PAGE, 100).unwrap());
        assert_eq!(
            post.release(
                THRESHOLD_OF_THREE_PAGES,
                Extent::new(2 * PAGE, 2 * PAGE).unwrap(),
                RAW
            ),
            Release::Merge
        );
        assert_eq!(
            (paged_sections(&post), post.unclassified.sections()),
            (
                [vec![], vec![(2 * PAGE, 2 * PAGE)], vec![]],
                vec![(4 * PAGE, 100)]
            )
        );
    }

    #[rstest]
    #[case::pages_ending_it(&[(8 * PAGE, 2 * PAGE)], 8 * PAGE, vec![])]
    #[case::pages_ending_it_in_turn(
        &[(6 * PAGE, 2 * PAGE), (8 * PAGE, 2 * PAGE)],
        6 * PAGE,
        vec![]
    )]
    #[case::pages_from_inside_a_page(
        &[(7 * PAGE + 100, 3 * PAGE - 100)],
        8 * PAGE,
        vec![(7 * PAGE + 100, PAGE - 100)]
    )]
    #[case::part_of_a_page_ending_it(&[(10 * PAGE - 100, 100)], 10 * PAGE, vec![])]
    #[case::pages_below_it(&[(6 * PAGE, 2 * PAGE)], 10 * PAGE, vec![])]
    fn releasing_paged_extents_lowers_the_end_of_allocation_by_dropped_pages_that_end_it(
        #[case] extents: &[(u64, u64)],
        #[case] expected: u64,
        #[case] raw: Vec<(u64, u64)>,
    ) {
        let mut post = paged_free_space(&[vec![], vec![], vec![]]);
        assert_eq!(
            post.release_all(
                THRESHOLD_OF_THREE_PAGES,
                10 * PAGE,
                extents
                    .iter()
                    .map(|&(addr, len)| (Extent::new(addr, len).unwrap(), RAW))
            ),
            expected
        );
        assert_eq!(paged_sections(&post), [vec![], raw, vec![]]);
    }

    #[rstest]
    #[case::alone(
        [vec![], vec![(PAGE, PAGE - 100)], vec![]],
        10 * PAGE,
        10 * PAGE,
        [vec![], vec![], vec![]]
    )]
    #[case::beside_a_whole_free_page(
        [vec![], vec![(PAGE, PAGE - 100), (2 * PAGE, PAGE)], vec![]],
        10 * PAGE,
        10 * PAGE,
        [vec![], vec![(PAGE, 2 * PAGE)], vec![]]
    )]
    #[case::beside_part_of_a_page(
        [vec![(PAGE - 100, 100)], vec![(PAGE, PAGE - 100)], vec![]],
        10 * PAGE,
        10 * PAGE,
        [vec![(PAGE - 100, 100)], vec![], vec![]]
    )]
    #[case::ending_the_allocation(
        [vec![], vec![(PAGE, PAGE - 100)], vec![]],
        2 * PAGE,
        PAGE,
        [vec![], vec![], vec![]]
    )]
    fn a_page_a_merge_completes_below_the_threshold_is_offered_as_a_whole_free_page(
        #[case] tracked: [Vec<(u64, u64)>; 3],
        #[case] eoa: u64,
        #[case] released: u64,
        #[case] expected: [Vec<(u64, u64)>; 3],
    ) {
        let mut post = paged_free_space(&tracked);
        assert_eq!(
            (
                post.release_all(
                    THRESHOLD_OF_THREE_PAGES,
                    eoa,
                    [(Extent::new(2 * PAGE - 100, 100).unwrap(), RAW)],
                ),
                paged_sections(&post)
            ),
            (released, expected)
        );
    }

    fn paged_free_space([meta, raw, dead]: &[Vec<(u64, u64)>; 3]) -> PagedPostFree {
        let list = |regions: &[(u64, u64)]| {
            let mut list = FreeList::new();
            for &(addr, len) in regions {
                list.free(Extent::new(addr, len).unwrap());
            }
            list
        };
        PagedPostFree {
            page_size: FileSpacePageSize::DEFAULT,
            meta: list(meta),
            raw: list(raw),
            dead: list(dead),
            unclassified: FreeList::new(),
        }
    }

    fn paged_sections(post: &PagedPostFree) -> [Vec<(u64, u64)>; 3] {
        [&post.meta, &post.raw, &post.dead].map(FreeList::sections)
    }
}
