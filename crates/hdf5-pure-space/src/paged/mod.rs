use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::cmp::Reverse;

use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::StoredAddress;

use super::admission::TrackedSpace;
use super::extent::Extent;
use super::list::{self, FreeList, FreeListError};
use super::persistence::{ManagerKind, TRAILING_RESERVE_TAILS};
use super::snapshot::AdmittedFreeSnapshot;

/// Identifies whether an allocation belongs to metadata or raw-data pages.
///
/// A paged file keeps the two kinds of allocation in separate pages and tracks their reusable
/// space independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageType {
    /// File metadata.
    Meta,
    /// Raw dataset data.
    Raw,
}

/// Classifies a vacated extent by the PAGE reuse policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreeClass {
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
pub enum PageTransition {
    /// Padding is required before the append.
    Pad {
        extent: Extent,
        track_as: Option<PageType>,
    },
    /// The append may use the current tail page.
    Unchanged,
}

/// Snapshots the checked free extents that map to PAGE managers.
pub struct PagedSections {
    metadata: Vec<Extent>,
    raw: Vec<Extent>,
    unclassified: Vec<Extent>,
}

impl PagedSections {
    /// Returns free sections whose pages are metadata pages.
    pub fn metadata(&self) -> &[Extent] {
        &self.metadata
    }

    /// Returns free sections whose pages are raw-data pages.
    pub fn raw(&self) -> &[Extent] {
        &self.raw
    }

    /// Returns free sections whose page type is not established.
    pub fn unclassified(&self) -> &[Extent] {
        &self.unclassified
    }
}

/// Stores reusable allocation state for rollback after a failed commit attempt.
pub struct PagedAllocationSnapshot {
    meta: FreeList,
    raw: FreeList,
}

/// Stores the complete PAGE state for an append rollback after truncation.
///
/// Includes all free lists, their cached totals, the tail page type, and pending padding.
pub struct PagedAppendSnapshot {
    paged: PagedEdit,
}

/// Tracks reusable-space state for a paged editing session.
#[derive(Clone)]
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

    /// Constructs canonical PAGE lists from admitted sections in ascending address order.
    ///
    /// Retains partial-page classifications and promotes complete pages covered by metadata and
    /// raw sections. Runs in linear time in the number of sections.
    ///
    /// # Errors
    ///
    /// Returns [`FreeListError::Unsorted`] if a section starts before the preceding section, or
    /// [`FreeListError::Overlap`] if their starts are ordered but their ranges overlap.
    pub(crate) fn from_persisted(
        page_size: FileSpacePageSize,
        sections: impl IntoIterator<Item = (Option<ManagerKind>, Extent)>,
    ) -> Result<Self, FreeListError> {
        let mut metadata = Vec::new();
        let mut raw = Vec::new();
        let mut unclassified = Vec::new();
        let mut previous: Option<Extent> = None;
        for (kind, extent) in sections {
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
            for (part, ty) in Self::persisted_parts(page_size, kind, extent)
                .into_iter()
                .flatten()
            {
                match ty {
                    Some(PageType::Meta) => metadata.push(part),
                    Some(PageType::Raw) => raw.push(part),
                    None => unclassified.push(part),
                }
            }
            previous = Some(extent);
        }
        let mut paged = Self {
            meta: FreeList::from_sorted(metadata)?,
            raw: FreeList::from_sorted(raw)?,
            unclassified: FreeList::from_sorted(unclassified)?,
            ..Self::new(page_size)
        };
        paged.finish_seed();
        Ok(paged)
    }

    pub(crate) fn page_size(&self) -> FileSpacePageSize {
        self.page_size
    }

    /// Returns a read-only snapshot of PAGE sections.
    pub(crate) fn sections(&self) -> PagedSections {
        PagedSections {
            metadata: self.meta.extents().to_vec(),
            raw: self.raw.extents().to_vec(),
            unclassified: self.unclassified.extents().to_vec(),
        }
    }

    /// Returns the type of the current tail page when this session established it.
    pub(crate) fn tail_type(&self) -> Option<PageType> {
        self.last
    }

    /// Returns pending page-tail padding for `ty`.
    pub(crate) fn pending_padding(&self, ty: PageType) -> &[Extent] {
        match ty {
            PageType::Meta => &self.meta_pad,
            PageType::Raw => &self.raw_pad,
        }
    }

    /// Returns the transition required before an append of `ty` at `eoa`.
    pub(crate) fn plan_transition(
        &self,
        eoa: StoredAddress,
        ty: PageType,
    ) -> Result<PageTransition, FormatError> {
        let page_size = self.page_size.get();
        if eoa.get() % page_size == 0 || self.last == Some(ty) {
            return Ok(PageTransition::Unchanged);
        }
        let pad_len = page_size - eoa.get() % page_size;
        let extent = Extent::new(eoa, pad_len).ok_or(FormatError::OffsetOverflow {
            offset: eoa.get(),
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
    pub(crate) fn page_padding(
        &self,
        eoa: StoredAddress,
    ) -> Result<Option<PageTransition>, FormatError> {
        let page_size = self.page_size.get();
        if eoa.get() % page_size == 0 {
            return Ok(None);
        }
        let pad_len = page_size - eoa.get() % page_size;
        let extent = Extent::new(eoa, pad_len).ok_or(FormatError::OffsetOverflow {
            offset: eoa.get(),
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

    /// Seeds reusable space by manager kind and retains unclassified fragments.
    ///
    /// Records sections from small metadata and raw managers with their page type. From a
    /// generic-large section, records complete interior pages as raw free space.
    /// The caller can allocate these pages for either page type through
    /// [`allocate`](Self::allocate). Partial edges and sections from other manager kinds remain
    /// unclassified.
    pub(crate) fn seed_persisted(&mut self, kind: Option<ManagerKind>, extent: Extent) {
        for (part, ty) in Self::persisted_parts(self.page_size, kind, extent)
            .into_iter()
            .flatten()
        {
            self.seed(part, ty);
        }
    }

    /// Splits a persisted extent into address-ordered parts with their established page type.
    ///
    /// Generic-large aligned interiors are raw space. Partial edges and sections from other
    /// manager kinds are unclassified.
    fn persisted_parts(
        page_size: FileSpacePageSize,
        kind: Option<ManagerKind>,
        extent: Extent,
    ) -> [Option<(Extent, Option<PageType>)>; 3] {
        match kind {
            Some(ManagerKind::PagedSmallMetadata) => {
                [Some((extent, Some(PageType::Meta))), None, None]
            }
            Some(ManagerKind::PagedSmallRaw) => [Some((extent, Some(PageType::Raw))), None, None],
            Some(ManagerKind::PagedLargeGeneric) => {
                let Some(pages) = extent.aligned_interior(page_size.get()) else {
                    return [Some((extent, None)), None, None];
                };
                [
                    Extent::between(extent.start(), pages.start()).map(|head| (head, None)),
                    Some((pages, Some(PageType::Raw))),
                    Extent::between(pages.end(), extent.end()).map(|tail| (tail, None)),
                ]
            }
            Some(ManagerKind::Flat) | None => [Some((extent, None)), None, None],
        }
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

    /// Borrows classified and unclassified admitted extents and their cached byte totals.
    pub(crate) fn admitted_snapshot(&self) -> AdmittedFreeSnapshot<'_> {
        AdmittedFreeSnapshot::paged(self.page_size, &self.meta, &self.raw, &self.unclassified)
    }

    /// Returns the start of the free run that reaches `eoa` across all PAGE lists.
    pub(crate) fn trailing_run_start(&self, eoa: StoredAddress) -> StoredAddress {
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

    /// Copies all PAGE state, including the tail page type and pending padding.
    pub(crate) fn append_snapshot(&self) -> PagedAppendSnapshot {
        PagedAppendSnapshot {
            paged: self.clone(),
        }
    }

    /// Restores all PAGE state after truncation to the pre-append length.
    pub(crate) fn restore_append(&mut self, snapshot: PagedAppendSnapshot) {
        *self = snapshot.paged;
    }

    /// Bounds the section count after page splitting and tail reservation.
    ///
    /// Includes pending padding, dead extents, and `extents` in the count.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the count exceeds [`usize::MAX`].
    pub(crate) fn tail_section_limit(
        &self,
        extents: impl IntoIterator<Item = Extent>,
    ) -> Result<usize, FormatError> {
        let page = self.page_size.get();
        self.meta
            .extents()
            .iter()
            .chain(self.raw.extents())
            .chain(self.dead.extents())
            .chain(self.unclassified.extents())
            .chain(&self.meta_pad)
            .chain(&self.raw_pad)
            .copied()
            .chain(extents)
            .try_fold(3usize, |count, extent| {
                let pieces = (extent.end().get() - 1) / page - extent.start().get() / page + 1;
                usize::try_from(pieces)
                    .ok()
                    .and_then(|pieces| count.checked_add(pieces))
                    .ok_or(FormatError::ValueTooLargeForPlatform {
                        value: pieces,
                        target: "PAGE persistence section count",
                    })
            })
    }

    /// Reserves metadata capacity in a copy of the allocation state.
    ///
    /// Applies pending frees only after reserving the allocation. Includes metadata-page padding
    /// in the final free sections and excludes any whole pages released above the final EOA.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the allocation, page boundary, or trailing
    /// reserve exceeds [`u64::MAX`].
    pub(crate) fn plan_tail(
        &self,
        threshold: u64,
        eoa: StoredAddress,
        freed: &[(Extent, FreeClass)],
        old_blocks: &[Extent],
        capacity: u64,
        appended: bool,
    ) -> Result<Option<PagedTail>, FormatError> {
        eoa.get()
            .checked_next_multiple_of(self.page_size.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: eoa.get(),
                length: self.page_size.get(),
            })?;
        capacity
            .checked_mul(TRAILING_RESERVE_TAILS * 2)
            .and_then(|reserve| reserve.checked_next_multiple_of(self.page_size.get()))
            .ok_or(FormatError::OffsetOverflow {
                offset: capacity,
                length: TRAILING_RESERVE_TAILS * 2,
            })?;
        let mut available = self.clone();
        let allocation = if appended {
            Extent::new(eoa, capacity).ok_or(FormatError::OffsetOverflow {
                offset: eoa.get(),
                length: capacity,
            })?
        } else {
            let Some(allocation) = available.allocate(capacity, PageType::Meta) else {
                return Ok(None);
            };
            allocation
        };
        let (mut post, released_eoa) = available.post_free(
            threshold,
            eoa,
            freed.iter().copied(),
            old_blocks.iter().copied(),
        );
        let final_eoa = if appended {
            if let Some(extent) = Extent::between(released_eoa, eoa) {
                post.track(extent, PageType::Raw.into());
            }
            let end = allocation.end();
            let aligned = end
                .get()
                .checked_next_multiple_of(self.page_size.get())
                .map(StoredAddress::new)
                .ok_or(FormatError::OffsetOverflow {
                    offset: end.get(),
                    length: self.page_size.get(),
                })?;
            if let Some(tail) = Extent::between(end, aligned) {
                post.track(tail, PageType::Meta.into());
            }
            aligned
        } else {
            post.release_trailing(released_eoa, capacity)
        };
        Ok(Some(PagedTail {
            allocation,
            post,
            final_eoa,
            appended,
        }))
    }

    /// Builds the PAGE free-space state that a commit would publish.
    pub(crate) fn post_free(
        &self,
        threshold: u64,
        eoa: StoredAddress,
        freed: impl IntoIterator<Item = (Extent, FreeClass)>,
        old_blocks: impl IntoIterator<Item = Extent>,
    ) -> (PagedPostFree, StoredAddress) {
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

    /// Promotes complete aligned interiors of adjacent metadata, raw, and dead regions to raw space.
    ///
    /// Retains each partial edge's original class. Runs in linear time in the number of extents.
    fn promote_whole_free_pages(
        meta: &mut FreeList,
        raw: &mut FreeList,
        dead: &mut FreeList,
        page_size: FileSpacePageSize,
    ) {
        // The PAGE classes track disjoint bytes, and each list is canonical.
        let runs = FreeList::from_lists([&*meta, &*raw, &*dead])
            .expect("PAGE classes have disjoint free extents");
        // Aligned interiors stay inside their sorted, disjoint containing runs.
        let whole_pages = FreeList::from_sorted(
            runs.extents()
                .iter()
                .filter_map(|extent| extent.aligned_interior(page_size.get())),
        )
        .expect("whole-page interiors are sorted and disjoint");
        meta.take_ranges(&whole_pages);
        raw.take_ranges(&whole_pages);
        dead.take_ranges(&whole_pages);
        // The subtraction removes every whole-page interior from the raw list before this merge.
        *raw = FreeList::from_lists([&*raw, &whole_pages])
            .expect("retained raw fragments are disjoint from promoted pages");
    }
}

/// Describes a metadata reservation and the PAGE state to publish with it.
///
/// The allocation's endpoints and EOA are [`StoredAddress`] values relative to the file's address
/// base. Planning leaves the session unchanged, and the caller adopts [`Self::post`] after
/// successful publication.
pub struct PagedTail {
    /// The complete metadata allocation, including any owned padding.
    pub allocation: Extent,
    /// The final free-space state, excluding the reservation and including its page tail.
    pub post: PagedPostFree,
    /// The page-aligned end of allocation to publish.
    pub final_eoa: StoredAddress,
    /// Whether the reservation begins at the previous end of allocation.
    pub appended: bool,
}

/// Holds the free-space state a paged commit is preparing to publish.
pub struct PagedPostFree {
    page_size: FileSpacePageSize,
    meta: FreeList,
    raw: FreeList,
    dead: FreeList,
    unclassified: FreeList,
}

impl PagedPostFree {
    /// Returns read-only sections for persistence planning.
    pub fn sections(&self) -> PagedSections {
        PagedSections {
            metadata: self.meta.extents().to_vec(),
            raw: self.raw.extents().to_vec(),
            unclassified: self.unclassified.extents().to_vec(),
        }
    }

    /// Releases the trailing whole-page run while retaining manager-tail reserve space.
    pub fn release_trailing(&mut self, eoa: StoredAddress, tail_len: u64) -> StoredAddress {
        let start =
            list::trailing_run_start([&self.meta, &self.raw, &self.dead, &self.unclassified], eoa);
        let page_size = self.page_size.get();
        let Some(cut) = start
            .get()
            .checked_next_multiple_of(page_size)
            .map(StoredAddress::new)
        else {
            return eoa;
        };
        if cut >= eoa {
            return eoa;
        }
        let Some(keep) = TRAILING_RESERVE_TAILS
            .checked_mul(tail_len)
            .and_then(|reserve| reserve.checked_next_multiple_of(page_size))
        else {
            return eoa;
        };
        let Some(release_min) = keep.checked_mul(2) else {
            return eoa;
        };
        if eoa
            .checked_distance_from(cut)
            .is_none_or(|length| length < release_min)
        {
            return eoa;
        }
        let Ok(published_eoa) = cut.checked_add(keep) else {
            return eoa;
        };
        let released = Extent::between(published_eoa, eoa)
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
        eoa: StoredAddress,
        extents: impl IntoIterator<Item = (Extent, FreeClass)>,
    ) -> StoredAddress {
        let extents: Vec<_> = extents.into_iter().collect();
        let mut dropped = self.release_each(threshold, extents.iter().copied());
        if threshold > self.page_size.get() {
            let completed = self.pages_completed_by_merges(&extents, &dropped);
            dropped.extend(self.release_pages(threshold, completed));
        }
        self.give_back(eoa, dropped)
    }

    /// Returns candidate pages wholly covered by metadata, raw, and dead regions.
    ///
    /// Candidates come from the boundary pages of admitted fragments shorter than one page.
    fn pages_completed_by_merges(
        &self,
        extents: &[(Extent, FreeClass)],
        dropped: &[(Extent, FreeClass)],
    ) -> BTreeSet<Extent> {
        let page_size = self.page_size.get();
        let dropped: BTreeSet<_> = dropped.iter().map(|&(extent, _)| extent).collect();
        let candidates: BTreeSet<_> = extents
            .iter()
            .map(|&(extent, _)| extent)
            .filter(|extent| (1..page_size).contains(&extent.len()) && !dropped.contains(extent))
            .flat_map(|extent| [extent.start().get(), extent.end().get() - 1])
            .map(|addr| StoredAddress::new(addr / page_size * page_size))
            .filter_map(|page| Extent::new(page, page_size))
            .collect();
        // The PAGE classes track disjoint bytes, and adjacent bytes form one canonical run.
        let runs = FreeList::from_lists([&self.meta, &self.raw, &self.dead])
            .expect("PAGE classes have disjoint free extents");
        let mut runs = runs.extents().iter().peekable();
        candidates
            .into_iter()
            .filter(|page| {
                while runs.peek().is_some_and(|run| run.end() <= page.start()) {
                    runs.next();
                }
                runs.peek()
                    .is_some_and(|run| run.start() <= page.start() && run.end() >= page.end())
            })
            .collect()
    }

    /// Applies whole-page threshold admission to each completed page after batch subtraction.
    fn release_pages(
        &mut self,
        threshold: u64,
        pages: BTreeSet<Extent>,
    ) -> Vec<(Extent, FreeClass)> {
        let pages: Vec<_> = pages.into_iter().collect();
        // The set contains distinct whole pages in ascending address order.
        let removed = FreeList::from_sorted(pages.iter().copied())
            .expect("completed pages are sorted and disjoint");
        for list in [&mut self.meta, &mut self.raw, &mut self.dead] {
            list.take_ranges(&removed);
        }
        self.release_each(
            threshold,
            pages
                .into_iter()
                .map(|page| (page, FreeClass::Page(PageType::Raw))),
        )
    }

    fn give_back(
        &mut self,
        eoa: StoredAddress,
        mut dropped: Vec<(Extent, FreeClass)>,
    ) -> StoredAddress {
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
        let Some(aligned) = start
            .get()
            .checked_next_multiple_of(page_size)
            .map(StoredAddress::new)
        else {
            return eoa;
        };
        if let Some(class) = head.filter(|_| start != aligned) {
            let head =
                Extent::between(start, aligned).expect("page alignment leaves a non-empty prefix");
            self.track(head, class);
        }
        aligned
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
    use alloc::vec;

    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::StoredAddress;
    use rstest::rstest;

    use super::*;
    use crate::admission::Release;
    use crate::persistence::ManagerSections;

    mod normalization;

    const PAGE: u64 = FileSpacePageSize::DEFAULT.get();
    const META: FreeClass = FreeClass::Page(PageType::Meta);
    const RAW: FreeClass = FreeClass::Page(PageType::Raw);
    const THRESHOLD_IN_A_PAGE: u64 = 64;
    const SHORT_OF_THRESHOLD: u64 = THRESHOLD_IN_A_PAGE - 1;
    const EXTENT_AT: u64 = PAGE + 1024;
    const THRESHOLD_OF_THREE_PAGES: u64 = 3 * PAGE;

    #[rstest]
    #[case::aligned(0, 0, 1, 3)]
    #[case::unaligned_head(1, 0, 2, 2)]
    #[case::unaligned_tail(0, 1, 1, 2)]
    #[case::unaligned_edges(1, 1, 2, 1)]
    fn generic_sections_reuse_complete_pages_and_preserve_unclassified_edges(
        #[case] head_offset: u64,
        #[case] tail_offset: u64,
        #[case] first_page: u64,
        #[case] page_count: u64,
        #[values(512, 1_536, 4_096)] page_size: u64,
        #[values(PageType::Meta, PageType::Raw)] ty: PageType,
    ) {
        let page = FileSpacePageSize::try_from(page_size).unwrap();
        let start = page_size + head_offset;
        let end = 4 * page_size - tail_offset;
        let section = extent(start, end - start);
        let pages = extent(first_page * page_size, page_count * page_size);
        let edges: Vec<_> = [
            Extent::between(StoredAddress::new(start), pages.start()),
            Extent::between(pages.end(), StoredAddress::new(end)),
        ]
        .into_iter()
        .flatten()
        .collect();
        let mut paged = PagedEdit::new(page);
        paged.seed_persisted(Some(ManagerKind::PagedLargeGeneric), section);
        paged.finish_seed();
        assert_eq!(paged.sections().metadata(), []);
        assert_eq!(paged.sections().raw(), [pages]);
        assert_eq!(paged.sections().unclassified(), edges);
        assert_eq!(
            paged.admitted_snapshot().reusable_bytes().unwrap(),
            pages.len()
        );
        assert_eq!(paged.largest(ty), pages.len());
        assert_eq!(paged.allocate(pages.len() + 1, ty), None);

        let mut reopened = PagedEdit::new(page);
        for manager in ManagerSections::paged(&paged.sections(), page) {
            for &extent in manager.sections() {
                reopened.seed_persisted(Some(manager.kind()), extent);
            }
        }
        reopened.finish_seed();
        assert_eq!(reopened.sections().metadata(), []);
        assert_eq!(reopened.sections().raw(), [pages]);
        assert_eq!(reopened.sections().unclassified(), edges);

        let snapshot = paged.allocation_snapshot();
        let length = pages.len() - page_size / 2;
        let allocation = Extent::new(pages.start(), length).unwrap();
        assert_eq!(paged.allocate(length, ty), Some(allocation));
        assert_eq!(paged.sections().unclassified(), edges);
        assert_eq!(
            paged.admitted_snapshot().reusable_bytes().unwrap(),
            page_size / 2
        );
        assert_eq!(paged.allocate(page_size / 2 + 1, ty), None);
        paged.restore_allocation(snapshot);
        assert_eq!(paged.sections().metadata(), []);
        assert_eq!(paged.sections().raw(), [pages]);
        assert_eq!(paged.sections().unclassified(), edges);
        assert_eq!(paged.allocate(pages.len(), ty), Some(pages));
        assert_eq!(paged.allocate(1, ty), None);
        assert_eq!(paged.sections().unclassified(), edges);
    }

    #[rstest]
    #[case::inside_page(PAGE + 1, PAGE - 2)]
    #[case::crosses_boundary(2 * PAGE - 1, 2)]
    #[case::unaligned_page_length(PAGE + 1, PAGE)]
    #[case::last_partial_page(u64::MAX - 3, 3)]
    fn generic_sections_composed_of_partial_pages_remain_unclassified(
        #[case] start: u64,
        #[case] length: u64,
    ) {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        let section = extent(start, length);
        paged.seed_persisted(Some(ManagerKind::PagedLargeGeneric), section);
        paged.finish_seed();
        assert_eq!(paged.sections().metadata(), []);
        assert_eq!(paged.sections().raw(), []);
        assert_eq!(paged.sections().unclassified(), [section]);
        assert_eq!(paged.admitted_snapshot().reusable_bytes().unwrap(), 0);
        assert_eq!(paged.allocate(1, PageType::Meta), None);
        assert_eq!(paged.allocate(1, PageType::Raw), None);
    }

    #[rstest]
    #[case::flat(Some(ManagerKind::Flat))]
    #[case::unknown(None)]
    fn flat_and_unknown_manager_sections_remain_unclassified(#[case] kind: Option<ManagerKind>) {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        let section = extent(PAGE + 1, 3 * PAGE);
        paged.seed_persisted(kind, section);
        paged.finish_seed();
        assert_eq!(paged.sections().metadata(), []);
        assert_eq!(paged.sections().raw(), []);
        assert_eq!(paged.sections().unclassified(), [section]);
        assert_eq!(paged.allocate(1, PageType::Meta), None);
        assert_eq!(paged.allocate(1, PageType::Raw), None);
    }

    #[test]
    fn generic_section_at_the_address_limit_reuses_only_its_complete_page() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        let start = u64::MAX - 2 * PAGE + 1;
        let section = extent(start, 2 * PAGE - 1);
        let tail = extent(start + PAGE, PAGE - 1);
        paged.seed_persisted(Some(ManagerKind::PagedLargeGeneric), section);
        paged.finish_seed();
        assert_eq!(paged.sections().raw(), [extent(start, PAGE)]);
        assert_eq!(paged.sections().unclassified(), [tail]);
        assert_eq!(
            paged.allocate(PAGE, PageType::Raw),
            Some(extent(start, PAGE))
        );
        assert_eq!(paged.allocate(1, PageType::Raw), None);
        assert_eq!(paged.sections().unclassified(), [tail]);
    }

    #[test]
    fn reusable_bytes_check_the_page_list_sum() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.meta.free(extent(0, u64::MAX));
        paged.raw.free(extent(0, 1));
        assert_eq!(
            paged.admitted_snapshot().reusable_bytes(),
            Err(FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            })
        );
    }

    #[rstest]
    #[case::unknown(None)]
    #[case::metadata(Some(PageType::Meta))]
    #[case::raw(Some(PageType::Raw))]
    fn append_snapshot_restores_every_page_state_field(#[case] tail: Option<PageType>) {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.seed(extent(PAGE + 16, 100), Some(PageType::Meta));
        paged.seed(extent(2 * PAGE + 16, 100), Some(PageType::Raw));
        paged.track(extent(3 * PAGE + 16, 100), FreeClass::Dead);
        paged.seed(extent(4 * PAGE + 16, 100), None);
        paged.record_page_padding(PageTransition::Pad {
            extent: extent(5 * PAGE + 16, 100),
            track_as: Some(PageType::Meta),
        });
        paged.record_page_padding(PageTransition::Pad {
            extent: extent(6 * PAGE + 16, 100),
            track_as: Some(PageType::Raw),
        });
        paged.last = tail;
        let snapshot = paged.append_snapshot();
        assert_eq!(
            paged.allocate(50, PageType::Meta),
            Some(extent(PAGE + 16, 50))
        );
        assert_eq!(
            paged.allocate(50, PageType::Raw),
            Some(extent(2 * PAGE + 16, 50))
        );
        let (post, _) = paged.post_free(
            1,
            StoredAddress::new(12 * PAGE),
            [(extent(7 * PAGE + 16, 100), FreeClass::Dead)],
            [],
        );
        paged.adopt_post_free(post, true, None);
        paged.seed(extent(8 * PAGE + 16, 100), None);
        let next = if tail == Some(PageType::Meta) {
            PageType::Raw
        } else {
            PageType::Meta
        };
        paged.record_transition(next, PageTransition::Unchanged);
        paged.record_page_padding(
            paged
                .page_padding(StoredAddress::new(9 * PAGE + 16))
                .unwrap()
                .unwrap(),
        );
        paged.restore_append(snapshot);
        let PagedEdit {
            page_size,
            meta,
            raw,
            unclassified,
            dead,
            last,
            meta_pad,
            raw_pad,
        } = paged;
        assert_eq!(page_size, FileSpacePageSize::DEFAULT);
        assert_eq!(meta.extents(), [extent(PAGE + 16, 100)]);
        assert_eq!(raw.extents(), [extent(2 * PAGE + 16, 100)]);
        assert_eq!(dead.extents(), [extent(3 * PAGE + 16, 100)]);
        assert_eq!(unclassified.extents(), [extent(4 * PAGE + 16, 100)]);
        assert_eq!(
            [
                meta.total_bytes(),
                raw.total_bytes(),
                dead.total_bytes(),
                unclassified.total_bytes()
            ],
            [100; 4]
        );
        assert_eq!(last, tail);
        assert_eq!(meta_pad, [extent(5 * PAGE + 16, 100)]);
        assert_eq!(raw_pad, [extent(6 * PAGE + 16, 100)]);
    }

    #[test]
    fn allocation_snapshot_retains_page_state_for_the_grown_image() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.seed(extent(PAGE + 16, 100), Some(PageType::Raw));
        let snapshot = paged.allocation_snapshot();
        assert_eq!(
            paged.allocate(50, PageType::Raw),
            Some(extent(PAGE + 16, 50))
        );
        paged.record_transition(PageType::Raw, PageTransition::Unchanged);
        let padding = paged
            .page_padding(StoredAddress::new(2 * PAGE + 16))
            .unwrap()
            .unwrap();
        paged.record_page_padding(padding);
        paged.record_transition(PageType::Meta, PageTransition::Unchanged);
        paged.restore_allocation(snapshot);
        assert_eq!(paged.sections().raw(), [extent(PAGE + 16, 100)]);
        assert_eq!(paged.tail_type(), Some(PageType::Meta));
        assert_eq!(
            paged.pending_padding(PageType::Raw),
            [extent(2 * PAGE + 16, PAGE - 16)]
        );
        assert_eq!(paged.pending_padding(PageType::Meta), []);
    }

    #[test]
    fn page_mutations_update_and_restore_cached_totals() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        let totals = |paged: &PagedEdit| {
            [
                paged.meta.total_bytes(),
                paged.raw.total_bytes(),
                paged.dead.total_bytes(),
                paged.unclassified.total_bytes(),
            ]
        };
        paged.seed(extent(PAGE, PAGE / 2), Some(PageType::Meta));
        paged.track(extent(PAGE + PAGE / 2, PAGE / 2), FreeClass::Dead);
        paged.seed(extent(3 * PAGE + 100, 50), None);
        assert_eq!(totals(&paged), [PAGE / 2, 0, PAGE / 2, 50]);
        paged.finish_seed();
        assert_eq!(totals(&paged), [0, PAGE, 0, 50]);
        let snapshot = paged.allocation_snapshot();
        assert_eq!(paged.allocate(100, PageType::Meta), Some(extent(PAGE, 100)));
        assert_eq!(totals(&paged), [PAGE - 100, 0, 0, 50]);
        paged.restore_allocation(snapshot);
        assert_eq!(totals(&paged), [0, PAGE, 0, 50]);
        let (post, eoa) = paged.post_free(
            1,
            StoredAddress::new(6 * PAGE),
            [(extent(2 * PAGE, 100), PageType::Raw.into())],
            [extent(4 * PAGE + 100, 50)],
        );
        assert_eq!(eoa, StoredAddress::new(6 * PAGE));
        assert_eq!(
            [
                post.meta.total_bytes(),
                post.raw.total_bytes(),
                post.dead.total_bytes(),
                post.unclassified.total_bytes(),
            ],
            [50, PAGE + 100, 0, 50]
        );
        assert_eq!(totals(&paged), [0, PAGE, 0, 50]);
        paged.adopt_post_free(post, false, None);
        assert_eq!(totals(&paged), [50, PAGE + 100, 0, 50]);
    }

    #[rstest::rstest]
    #[case::below_threshold(8_192, 200)]
    #[case::page_tail(1, 512)]
    #[case::multiple_pages(1, 4_096)]
    fn appended_tail_is_part_of_the_planned_state(#[case] threshold: u64, #[case] capacity: u64) {
        let paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        let plan = paged
            .plan_tail(
                threshold,
                StoredAddress::new(8_192),
                &[],
                &[],
                capacity,
                true,
            )
            .unwrap()
            .unwrap();
        assert_eq!(plan.allocation, extent(8_192, capacity));
        assert_eq!(
            plan.final_eoa,
            StoredAddress::new((8_192 + capacity).next_multiple_of(4_096))
        );
        let expected = Extent::between(StoredAddress::new(8_192 + capacity), plan.final_eoa);
        assert_eq!(plan.post.sections().metadata(), expected.as_slice());
        assert!(paged.sections().metadata().is_empty());
    }

    #[rstest::rstest]
    #[case::pending_object(false)]
    #[case::old_manager(true)]
    fn pending_frees_cannot_fund_a_tail_reservation(#[case] old_manager: bool) {
        let paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        let freed = if old_manager {
            Vec::new()
        } else {
            alloc::vec![(extent(4_096, 500), PageType::Meta.into())]
        };
        let old_blocks = if old_manager {
            alloc::vec![extent(4_096, 500)]
        } else {
            Vec::new()
        };
        assert!(
            paged
                .plan_tail(
                    1,
                    StoredAddress::new(8_192),
                    &freed,
                    &old_blocks,
                    300,
                    false
                )
                .unwrap()
                .is_none()
        );
        let appended = paged
            .plan_tail(1, StoredAddress::new(8_192), &freed, &old_blocks, 300, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            appended.post.sections().metadata(),
            [extent(4_096, 500), extent(8_492, 3_796)]
        );
    }

    #[test]
    fn a_pure_tail_plan_reserves_only_durable_metadata() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.seed(extent(100, 500), Some(PageType::Meta));
        paged.seed(extent(4_096, 4_096), Some(PageType::Raw));
        let plan = paged
            .plan_tail(1, StoredAddress::new(8_192), &[], &[], 300, false)
            .unwrap()
            .unwrap();
        assert_eq!(plan.allocation, extent(100, 300));
        assert!(!plan.appended);
        assert_eq!(plan.post.sections().metadata(), [extent(400, 200)]);
        assert_eq!(paged.sections().metadata(), [extent(100, 500)]);
    }

    #[test]
    fn tail_section_limit_includes_hidden_dead_fragments_and_page_splits() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.track(extent(4_000, 5_000), FreeClass::Dead);
        paged.meta_pad.push(extent(100, 50));
        assert_eq!(paged.tail_section_limit([extent(12_000, 500)]).unwrap(), 9);
    }

    #[test]
    fn a_tail_plan_rejects_an_overflowing_page_boundary() {
        let paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        assert!(matches!(
            paged.plan_tail(1, StoredAddress::new(u64::MAX - 1), &[], &[], 1, true),
            Err(FormatError::OffsetOverflow {
                offset: value,
                length: 4_096
            }) if value == u64::MAX - 1
        ));
    }

    #[test]
    fn an_aligned_page_transition_needs_no_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Meta);
        assert_eq!(
            paged
                .plan_transition(StoredAddress::new(PAGE), PageType::Raw)
                .unwrap(),
            PageTransition::Unchanged
        );
    }

    #[test]
    fn a_same_type_partial_page_transition_needs_no_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Raw);
        assert_eq!(
            paged
                .plan_transition(StoredAddress::new(5000), PageType::Raw)
                .unwrap(),
            PageTransition::Unchanged
        );
    }

    #[test]
    fn a_metadata_to_raw_transition_tracks_metadata_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Meta);
        assert_eq!(
            paged
                .plan_transition(StoredAddress::new(5000), PageType::Raw)
                .unwrap(),
            PageTransition::Pad {
                extent: Extent::new(StoredAddress::new(5000), 3192).unwrap(),
                track_as: Some(PageType::Meta),
            }
        );
    }

    #[test]
    fn a_raw_to_metadata_transition_tracks_raw_padding() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged.last = Some(PageType::Raw);
        assert_eq!(
            paged
                .plan_transition(StoredAddress::new(5000), PageType::Meta)
                .unwrap(),
            PageTransition::Pad {
                extent: Extent::new(StoredAddress::new(5000), 3192).unwrap(),
                track_as: Some(PageType::Raw),
            }
        );
    }

    #[test]
    fn an_unknown_tail_transition_leaves_padding_unclassified() {
        let paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        assert_eq!(
            paged
                .plan_transition(StoredAddress::new(5000), PageType::Meta)
                .unwrap(),
            PageTransition::Pad {
                extent: Extent::new(StoredAddress::new(5000), 3192).unwrap(),
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
            paged
                .plan_transition(StoredAddress::new(1500), PageType::Raw)
                .unwrap(),
            PageTransition::Pad {
                extent: Extent::new(StoredAddress::new(1500), 500).unwrap(),
                track_as: Some(PageType::Meta),
            }
        );
    }

    #[test]
    fn a_page_that_is_wholly_free_or_dead_is_promoted_whole() {
        let mut paged = PagedEdit::new(FileSpacePageSize::DEFAULT);
        paged
            .meta
            .free(Extent::new(StoredAddress::new(PAGE), 2048).unwrap());
        paged
            .dead
            .free(Extent::new(StoredAddress::new(PAGE + 2048), 2048).unwrap());
        paged
            .dead
            .free(Extent::new(StoredAddress::new(2 * PAGE), PAGE).unwrap());
        paged
            .raw
            .free(Extent::new(StoredAddress::new(3 * PAGE + 1024), 1024).unwrap());
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
        paged
            .dead
            .free(Extent::new(StoredAddress::new(PAGE), 512).unwrap());
        paged
            .raw
            .free(Extent::new(StoredAddress::new(PAGE + 512), 1024).unwrap());
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
                Extent::new(StoredAddress::new(EXTENT_AT), len).unwrap(),
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
                Extent::new(StoredAddress::new(EXTENT_AT), SHORT_OF_THRESHOLD).unwrap(),
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
                Extent::new(StoredAddress::new(at), SHORT_OF_THRESHOLD).unwrap(),
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
                Extent::new(StoredAddress::new(2 * PAGE), 2 * PAGE).unwrap(),
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
                Extent::new(StoredAddress::new(at), 2 * PAGE).unwrap(),
                RAW
            ),
            Release::Drop
        );
        assert_eq!(paged_sections(&post), tracked);
    }

    #[test]
    fn a_paged_extent_of_pages_below_the_threshold_merges_beside_an_unclassified_section() {
        let mut post = paged_free_space(&[vec![], vec![], vec![]]);
        post.unclassified
            .free(Extent::new(StoredAddress::new(4 * PAGE), 100).unwrap());
        assert_eq!(
            post.release(
                THRESHOLD_OF_THREE_PAGES,
                Extent::new(StoredAddress::new(2 * PAGE), 2 * PAGE).unwrap(),
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
                StoredAddress::new(10 * PAGE),
                extents
                    .iter()
                    .map(|&(addr, len)| (Extent::new(StoredAddress::new(addr), len).unwrap(), RAW))
            ),
            StoredAddress::new(expected)
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
                    StoredAddress::new(eoa),
                    [(
                        Extent::new(StoredAddress::new(2 * PAGE - 100), 100).unwrap(),
                        RAW
                    )],
                ),
                paged_sections(&post)
            ),
            (StoredAddress::new(released), expected)
        );
    }

    #[rstest]
    #[case::reserve_overflow(16 * PAGE, u64::MAX)]
    #[case::rounded_reserve_overflow(16 * PAGE, u64::MAX / TRAILING_RESERVE_TAILS)]
    #[case::release_minimum_overflow(16 * PAGE, u64::MAX / (2 * TRAILING_RESERVE_TAILS))]
    #[case::unrepresentable_boundary(u64::MAX, 1)]
    fn unrepresentable_trailing_reserves_preserve_the_free_sections(
        #[case] end: u64,
        #[case] tail_length: u64,
    ) {
        let start = if end == u64::MAX { end - 1 } else { PAGE };
        let mut post = paged_free_space(&[vec![(start, end - start)], vec![], vec![]]);
        let before = paged_sections(&post);
        assert_eq!(
            post.release_trailing(StoredAddress::new(end), tail_length),
            StoredAddress::new(end)
        );
        assert_eq!(paged_sections(&post), before);
    }

    fn paged_free_space([meta, raw, dead]: &[Vec<(u64, u64)>; 3]) -> PagedPostFree {
        let list = |regions: &[(u64, u64)]| {
            let mut list = FreeList::new();
            for &(addr, len) in regions {
                list.free(Extent::new(StoredAddress::new(addr), len).unwrap());
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
        for list in [&post.meta, &post.raw, &post.dead, &post.unclassified] {
            assert_eq!(
                u128::from(list.total_bytes()),
                list.extents()
                    .iter()
                    .map(|extent| u128::from(extent.len()))
                    .sum::<u128>()
            );
        }
        [&post.meta, &post.raw, &post.dead].map(FreeList::sections)
    }
    fn extent(start: u64, len: u64) -> Extent {
        Extent::new(StoredAddress::new(start), len).unwrap()
    }
}
