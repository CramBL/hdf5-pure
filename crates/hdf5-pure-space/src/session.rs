use alloc::vec::Vec;
use core::iter::Peekable;

use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_core::FormatError;

use super::allocation::AllocationClass;
use super::extent::Extent;
use super::list::{self, FreeList};
use super::paged::{
    FreeClass, PageTransition, PageType, PagedAllocationSnapshot, PagedEdit, PagedPostFree,
    PagedSections, PagedTail,
};
use super::persistence::ManagerKind;

const EMPTY_EXTENTS: &[Extent] = &[];

/// The reusable-space strategy active for one editing session.
pub enum SessionSpace {
    /// Reusable allocation is disabled.
    Disabled,
    /// One generic free list, as used by the implemented part of `FSM_AGGR`.
    Flat(FlatSpace),
    /// PAGE allocation with separate metadata and raw-data space.
    Paged(PagedSpace),
}

/// Reusable-space state for an unpaged manager strategy.
pub struct FlatSpace {
    free: FreeList,
    threshold: u64,
}

/// Reusable-space state for the PAGE strategy.
pub struct PagedSpace {
    paged: PagedEdit,
    threshold: u64,
}

/// The allocation state a failed commit may restore without undoing written PAGE padding.
pub enum SessionSpaceSnapshot {
    Disabled,
    Flat(FreeList),
    Paged(PagedAllocationSnapshot),
}

/// Merges two address-sorted extent iterators without allocating another snapshot.
struct MergeExtents<I, J>
where
    I: Iterator<Item = Extent>,
    J: Iterator<Item = Extent>,
{
    first: Peekable<I>,
    second: Peekable<J>,
}

impl<I, J> MergeExtents<I, J>
where
    I: Iterator<Item = Extent>,
    J: Iterator<Item = Extent>,
{
    fn new(first: I, second: J) -> Self {
        Self {
            first: first.peekable(),
            second: second.peekable(),
        }
    }
}

impl<I, J> Iterator for MergeExtents<I, J>
where
    I: Iterator<Item = Extent>,
    J: Iterator<Item = Extent>,
{
    type Item = Extent;

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.first.peek().copied();
        let second = self.second.peek().copied();
        match (first, second) {
            (Some(first), Some(second)) if first.start() <= second.start() => {
                debug_assert!(
                    first.end() <= second.start(),
                    "reusable extent lists must be mutually disjoint"
                );
                self.first.next()
            }
            (Some(first), Some(second)) => {
                debug_assert!(
                    second.end() <= first.start(),
                    "reusable extent lists must be mutually disjoint"
                );
                self.second.next()
            }
            (Some(_), None) => self.first.next(),
            (None, Some(_)) => self.second.next(),
            (None, None) => None,
        }
    }
}

impl SessionSpace {
    /// Builds the runtime allocation strategy from normalized file-space settings.
    ///
    /// `None` represents a strategy that could not be read safely. A missing File Space Info
    /// message is normalized by the caller to the default `FSM_AGGR` strategy before this call.
    pub fn from_strategy(
        strategy: Option<FileSpaceStrategy>,
        threshold: u64,
        page_size: FileSpacePageSize,
    ) -> Self {
        match strategy {
            Some(FileSpaceStrategy::FsmAggr) => Self::Flat(FlatSpace {
                free: FreeList::new(),
                threshold,
            }),
            Some(FileSpaceStrategy::Page) => Self::Paged(PagedSpace {
                paged: PagedEdit::new(page_size),
                threshold,
            }),
            Some(FileSpaceStrategy::Aggr) | Some(FileSpaceStrategy::None) | None => Self::Disabled,
        }
    }

    /// Returns whether this strategy maintains reusable free space.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Returns whether this session uses the PAGE strategy.
    pub fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }

    /// Returns the PAGE size when this session is paged.
    pub fn page_size(&self) -> Option<FileSpacePageSize> {
        match self {
            Self::Paged(space) => Some(space.paged.page_size()),
            Self::Disabled | Self::Flat(_) => None,
        }
    }

    /// Classifies one complete object-header chunk for reclamation.
    ///
    /// `address` is a stored, base-relative address, and `length` includes the chunk's framing.
    /// For PAGE, returns [`FreeClass::Dead`] if the chunk starts on a page boundary and occupies
    /// at least one page. Whole pages remain recoverable, while a partial tail stays outside the
    /// small metadata and raw-data free lists. Other chunks are classified as metadata.
    ///
    /// `H5MF__alloc_to_fs_type` in `H5MF.c`, HDF5 2.2.0, assigns PAGE allocations of at least one
    /// page to the generic-large manager for contiguous drivers.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy};
    /// use hdf5_pure_space::__private::{FreeClass, PageType, SessionSpace};
    ///
    /// let page_size = FileSpacePageSize::DEFAULT;
    /// let space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 1, page_size,
    /// );
    /// let stored_address = page_size.get();
    /// let complete_length = page_size.get() + 1;
    /// assert_eq!(
    ///     space.object_header_free_class(stored_address, complete_length),
    ///     FreeClass::Dead,
    /// );
    /// assert_eq!(
    ///     space.object_header_free_class(stored_address, page_size.get() - 1),
    ///     FreeClass::Page(PageType::Meta),
    /// );
    /// assert_eq!(
    ///     space.object_header_free_class(stored_address + 1, complete_length),
    ///     FreeClass::Page(PageType::Meta),
    /// );
    /// ```
    pub fn object_header_free_class(&self, address: u64, length: u64) -> FreeClass {
        if self
            .page_size()
            .is_some_and(|page| address.is_multiple_of(page.get()) && length >= page.get())
        {
            FreeClass::Dead
        } else {
            FreeClass::Page(PageType::Meta)
        }
    }

    /// Seeds one persisted free-space section under the active strategy.
    ///
    /// `kind` carries PAGE manager semantics. A flat strategy ignores it because all manager
    /// sections share one allocation class.
    pub fn seed_persisted(&mut self, kind: Option<ManagerKind>, extent: Extent) {
        match self {
            Self::Disabled => {}
            Self::Flat(space) => space.free.free(extent),
            Self::Paged(space) => space.paged.seed_persisted(kind, extent),
        }
    }

    /// Finishes strategy-specific normalization after persisted sections are seeded.
    pub fn finish_seed(&mut self) {
        if let Self::Paged(space) = self {
            space.paged.finish_seed();
        }
    }

    /// Reserves `len` reusable bytes for `class`, or returns `None` if no matching space is available.
    ///
    /// PAGE requests use the corresponding [`PageType`]. Flat allocation uses one free list for
    /// both classes.
    pub fn allocate(&mut self, len: u64, class: AllocationClass) -> Option<Extent> {
        self.allocate_with_page_type(len, class, class.into())
    }

    /// Reserves reusable bytes with an explicit PAGE type.
    ///
    /// `page_type` determines PAGE placement independently of the request's `class`. Array chunk
    /// indexes use this operation to retain metadata intent while sharing raw-data pages. Flat
    /// allocation uses one free list for both classes. Returns `None` if no matching space is
    /// available.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy};
    /// use hdf5_pure_space::__private::{AllocationClass, Extent, ManagerKind, PageType, SessionSpace};
    ///
    /// let mut space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT,
    /// );
    /// space.seed_persisted(
    ///     Some(ManagerKind::PagedSmallRaw), Extent::new(8_208, 256).unwrap(),
    /// );
    /// space.finish_seed();
    /// let raw = space.allocate(64, AllocationClass::Raw).unwrap();
    /// space.return_allocation(raw, AllocationClass::Raw);
    /// let index = space.allocate_with_page_type(
    ///     64, AllocationClass::Metadata, PageType::Raw,
    /// ).unwrap();
    /// assert_eq!(index, Extent::new(8_208, 64).unwrap());
    /// space.return_allocation_with_page_type(index, AllocationClass::Metadata, PageType::Raw);
    /// assert_eq!(space.allocate(64, AllocationClass::Raw), Some(index));
    /// ```
    pub fn allocate_with_page_type(
        &mut self,
        len: u64,
        _class: AllocationClass,
        page_type: PageType,
    ) -> Option<Extent> {
        match self {
            Self::Disabled => None,
            Self::Flat(space) => space.free.alloc(len),
            Self::Paged(space) => space.paged.allocate(len, page_type),
        }
    }

    /// Returns a reservation to reusable space for `class`.
    ///
    /// The caller passes the same class used to reserve `extent`. For an explicit PAGE type, use
    /// [`Self::return_allocation_with_page_type`].
    pub fn return_allocation(&mut self, extent: Extent, class: AllocationClass) {
        self.return_allocation_with_page_type(extent, class, class.into());
    }

    /// Returns a reservation using its explicit PAGE type.
    ///
    /// The caller passes the same `class` and `page_type` used with
    /// [`Self::allocate_with_page_type`].
    pub fn return_allocation_with_page_type(
        &mut self,
        extent: Extent,
        _class: AllocationClass,
        page_type: PageType,
    ) {
        match self {
            Self::Disabled => {}
            Self::Flat(space) => space.free.free(extent),
            Self::Paged(space) => space.paged.return_allocation(extent, page_type),
        }
    }

    /// Returns the largest reusable run available for `class`.
    ///
    /// Returns zero when reusable allocation is disabled.
    pub fn largest(&self, class: AllocationClass) -> u64 {
        match self {
            Self::Disabled => 0,
            Self::Flat(space) => space.free.largest(),
            Self::Paged(space) => space.paged.largest(class.into()),
        }
    }

    /// Returns reusable extents in ascending address order.
    pub fn reusable_sections(&self) -> impl Iterator<Item = Extent> + '_ {
        let (first, second) = match self {
            Self::Disabled => (EMPTY_EXTENTS, EMPTY_EXTENTS),
            Self::Flat(space) => (space.free.extents(), EMPTY_EXTENTS),
            Self::Paged(space) => space.paged.reusable_extents(),
        };
        MergeExtents::new(first.iter().copied(), second.iter().copied())
    }

    /// Returns reusable extents with extra raw allocations folded back into the snapshot.
    pub fn reusable_sections_with_raw(
        &self,
        raw_extents: impl IntoIterator<Item = Extent>,
    ) -> impl Iterator<Item = Extent> + '_ {
        let (first, second) = match self {
            Self::Disabled => (EMPTY_EXTENTS, Vec::new()),
            Self::Flat(space) => {
                let mut free = space.free.clone();
                for extent in raw_extents {
                    free.free(extent);
                }
                (EMPTY_EXTENTS, free.into_extents())
            }
            Self::Paged(space) => {
                let (metadata, _) = space.paged.reusable_extents();
                (metadata, space.paged.reusable_raw_with(raw_extents))
            }
        };
        MergeExtents::new(first.iter().copied(), second.into_iter())
    }

    /// Returns the start of the free run reaching `eoa` under this strategy.
    pub fn trailing_run_start(&self, eoa: u64) -> u64 {
        match self {
            Self::Disabled => eoa,
            Self::Flat(space) => list::trailing_run_start([&space.free], eoa),
            Self::Paged(space) => space.paged.trailing_run_start(eoa),
        }
    }

    /// Records extents returned by a non-persisting commit and returns a smaller EOA when one is
    /// released from the file tail.
    pub fn release_freed(
        &mut self,
        eof: u64,
        extents: impl IntoIterator<Item = Extent>,
    ) -> Option<u64> {
        let Self::Flat(space) = self else {
            return None;
        };
        let eoa = space.free.release_all(space.threshold, eof, extents);
        let eoa = space.free.take_trailing(eoa).unwrap_or(eoa);
        (eoa < eof).then_some(eoa)
    }

    /// Builds the flat free-space state that a persisting commit would publish.
    pub fn flat_post_free(
        &self,
        eof: u64,
        extents: impl IntoIterator<Item = Extent>,
    ) -> Option<(FreeList, u64)> {
        let Self::Flat(space) = self else {
            return None;
        };
        let mut post = space.free.clone();
        let eoa = post.release_all(space.threshold, eof, extents);
        Some((post, eoa))
    }

    /// Adopts the flat free list after its publication succeeds.
    pub fn adopt_flat_post_free(&mut self, post: FreeList) {
        if let Self::Flat(space) = self {
            space.free = post;
        } else {
            debug_assert!(
                false,
                "a flat post-free state requires a flat session strategy"
            );
        }
    }

    /// Builds the PAGE free-space state that a persisting commit would publish.
    pub fn paged_post_free(
        &self,
        eoa: u64,
        freed: impl IntoIterator<Item = (Extent, FreeClass)>,
        old_blocks: impl IntoIterator<Item = Extent>,
    ) -> Option<(PagedPostFree, u64)> {
        let Self::Paged(space) = self else {
            return None;
        };
        Some(
            space
                .paged
                .post_free(space.threshold, eoa, freed, old_blocks),
        )
    }

    /// Bounds the PAGE section count after adding `extents` and reserving a metadata tail.
    ///
    /// Includes tracked free space, pending page padding, and dead extents that may form whole
    /// free pages. Counts every page crossed by an extent separately.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the strategy is not PAGE, or
    /// [`FormatError::ValueTooLargeForPlatform`] if the count exceeds [`usize::MAX`].
    pub fn paged_tail_section_limit(
        &self,
        extents: impl IntoIterator<Item = Extent>,
    ) -> Result<usize, FormatError> {
        let Self::Paged(space) = self else {
            return Err(FormatError::Internal(
                "PAGE persistence requires PAGE session space".into(),
            ));
        };
        space.paged.tail_section_limit(extents)
    }

    /// Plans a metadata reservation and its final PAGE free sections without changing the session.
    ///
    /// With `appended`, reserves `capacity` bytes at `eoa`. Otherwise, reserves only reusable
    /// metadata space and returns `None` if no extent fits. Pending `freed` extents and live
    /// `old_blocks` become free in the planned state but cannot fund the reservation.
    ///
    /// All addresses are stored addresses. The caller passes a page-aligned `eoa` and a nonzero
    /// `capacity`. The result includes page padding, threshold admission, and trailing-page release.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::FileSpacePageSize;
    /// use hdf5_pure_core::FileSpaceStrategy;
    /// use hdf5_pure_space::__private::Extent;
    /// use hdf5_pure_space::__private::FreeClass;
    /// use hdf5_pure_space::__private::ManagerKind;
    /// use hdf5_pure_space::__private::PageType;
    /// use hdf5_pure_space::__private::SessionSpace;
    ///
    /// let mut space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT,
    /// );
    /// let durable = Extent::new(4_096, 512).unwrap();
    /// space.seed_persisted(Some(ManagerKind::PagedSmallMetadata), durable);
    /// let freed = [(Extent::new(8_192, 128).unwrap(), FreeClass::Page(PageType::Meta))];
    /// let old_blocks = [Extent::new(8_448, 128).unwrap()];
    /// let limit = space.paged_tail_section_limit(
    ///     freed.iter().map(|&(extent, _)| extent).chain(old_blocks),
    /// )?;
    ///
    /// let reused = space.plan_paged_tail(12_288, &freed, &old_blocks, 256, false)?.unwrap();
    /// assert_eq!(reused.allocation, Extent::new(4_096, 256).unwrap());
    /// assert_eq!(reused.post.sections().metadata(), [
    ///     Extent::new(4_352, 256).unwrap(), freed[0].0, old_blocks[0],
    /// ]);
    /// assert!(reused.post.sections().metadata().len() <= limit);
    /// assert_eq!(space.paged_sections().unwrap().metadata(), [durable]);
    ///
    /// let appended = space.plan_paged_tail(12_288, &freed, &old_blocks, 256, true)?.unwrap();
    /// assert_eq!(appended.allocation, Extent::new(12_288, 256).unwrap());
    /// assert_eq!(appended.final_eoa, 16_384);
    /// assert_eq!(appended.post.sections().metadata().last(),
    ///     Some(&Extent::new(12_544, 3_840).unwrap()));
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the strategy is not PAGE, or
    /// [`FormatError::OffsetOverflow`] if the allocation, page boundary, or trailing reserve
    /// exceeds [`u64::MAX`].
    pub fn plan_paged_tail(
        &self,
        eoa: u64,
        freed: &[(Extent, FreeClass)],
        old_blocks: &[Extent],
        capacity: u64,
        appended: bool,
    ) -> Result<Option<PagedTail>, FormatError> {
        let Self::Paged(space) = self else {
            return Err(FormatError::Internal(
                "PAGE persistence requires PAGE session space".into(),
            ));
        };
        space
            .paged
            .plan_tail(space.threshold, eoa, freed, old_blocks, capacity, appended)
    }

    /// Adopts PAGE post-free state after its publication succeeds.
    pub fn adopt_paged_post_free(
        &mut self,
        post: PagedPostFree,
        appended_metadata: bool,
        appended_tail: Option<Extent>,
    ) {
        if let Self::Paged(space) = self {
            space
                .paged
                .adopt_post_free(post, appended_metadata, appended_tail);
        } else {
            debug_assert!(
                false,
                "a PAGE post-free state requires a paged session strategy"
            );
        }
    }

    /// Returns the transition required before appending `ty`, or `None` for an unpaged strategy.
    pub fn plan_transition(
        &self,
        eoa: u64,
        ty: PageType,
    ) -> Result<Option<PageTransition>, FormatError> {
        match self {
            Self::Paged(space) => space.paged.plan_transition(eoa, ty).map(Some),
            Self::Disabled | Self::Flat(_) => Ok(None),
        }
    }

    /// Records a PAGE transition after its padding has been written.
    pub fn record_transition(&mut self, ty: PageType, transition: PageTransition) {
        if let Self::Paged(space) = self {
            space.paged.record_transition(ty, transition);
        }
    }

    /// Returns PAGE-boundary padding, or `None` for an unpaged or aligned file.
    pub fn page_padding(&self, eoa: u64) -> Result<Option<PageTransition>, FormatError> {
        match self {
            Self::Paged(space) => space.paged.page_padding(eoa),
            Self::Disabled | Self::Flat(_) => Ok(None),
        }
    }

    /// Records PAGE-boundary padding after it has been written.
    pub fn record_page_padding(&mut self, transition: PageTransition) {
        if let Self::Paged(space) = self {
            space.paged.record_page_padding(transition);
        }
    }

    /// Snapshots the allocation state that rollback may restore safely.
    pub fn allocation_snapshot(&self) -> SessionSpaceSnapshot {
        match self {
            Self::Disabled => SessionSpaceSnapshot::Disabled,
            Self::Flat(space) => SessionSpaceSnapshot::Flat(space.free.clone()),
            Self::Paged(space) => SessionSpaceSnapshot::Paged(space.paged.allocation_snapshot()),
        }
    }

    /// Restores a prior allocation snapshot without rewinding PAGE tail/padding state.
    pub fn restore_allocation(&mut self, snapshot: SessionSpaceSnapshot) {
        match (self, snapshot) {
            (Self::Disabled, SessionSpaceSnapshot::Disabled) => {}
            (Self::Flat(space), SessionSpaceSnapshot::Flat(free)) => space.free = free,
            (Self::Paged(space), SessionSpaceSnapshot::Paged(paged)) => {
                space.paged.restore_allocation(paged);
            }
            _ => debug_assert!(false, "a session allocation snapshot changed strategy"),
        }
    }

    /// Returns PAGE section state when this session is paged.
    pub fn paged_sections(&self) -> Option<PagedSections> {
        match self {
            Self::Paged(space) => Some(space.paged.sections()),
            Self::Disabled | Self::Flat(_) => None,
        }
    }

    /// Returns the established PAGE tail type when this session is paged.
    ///
    /// The outer `Option` distinguishes a non-PAGE strategy. The inner value is `None` until the
    /// session establishes a tail page type.
    pub fn paged_tail_type(&self) -> Option<Option<PageType>> {
        match self {
            Self::Paged(space) => Some(space.paged.tail_type()),
            Self::Disabled | Self::Flat(_) => None,
        }
    }

    /// Returns pending PAGE padding for `ty` when this session is paged.
    pub fn pending_padding(&self, ty: PageType) -> Option<&[Extent]> {
        match self {
            Self::Paged(space) => Some(space.paged.pending_padding(ty)),
            Self::Disabled | Self::Flat(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::small_aligned(0, -1, FreeClass::Page(PageType::Meta))]
    #[case::exact_page(0, 0, FreeClass::Dead)]
    #[case::large_aligned(0, 1, FreeClass::Dead)]
    #[case::large_unaligned(1, 1, FreeClass::Page(PageType::Meta))]
    fn object_header_chunks_retain_their_allocation_class(
        #[values(512, 4_096, 8_192)] page_size: u64,
        #[case] offset: u64,
        #[case] length_delta: i64,
        #[case] expected: FreeClass,
    ) {
        let space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            1,
            FileSpacePageSize::try_from(page_size).unwrap(),
        );
        assert_eq!(
            space.object_header_free_class(
                2 * page_size + offset,
                page_size.checked_add_signed(length_delta).unwrap()
            ),
            expected,
        );
    }

    #[rstest]
    fn unpaged_object_header_chunks_remain_metadata(
        #[values(
            None,
            Some(FileSpaceStrategy::FsmAggr),
            Some(FileSpaceStrategy::Aggr),
            Some(FileSpaceStrategy::None)
        )]
        strategy: Option<FileSpaceStrategy>,
    ) {
        let space = SessionSpace::from_strategy(strategy, 1, FileSpacePageSize::DEFAULT);
        assert_eq!(
            space.object_header_free_class(4_096, 8_193),
            FreeClass::Page(PageType::Meta)
        );
    }

    #[rstest]
    #[case::metadata(AllocationClass::Metadata, ManagerKind::PagedSmallMetadata, 4_112)]
    #[case::raw(AllocationClass::Raw, ManagerKind::PagedSmallRaw, 8_208)]
    fn allocation_intent_selects_the_matching_persisted_page(
        #[case] class: AllocationClass,
        #[case] manager: ManagerKind,
        #[case] address: u64,
    ) {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            1,
            FileSpacePageSize::DEFAULT,
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(4_112, 128).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallRaw),
            Extent::new(8_208, 256).unwrap(),
        );
        space.finish_seed();
        let expected = Extent::new(address, 64).unwrap();
        assert_eq!(space.allocate(64, class), Some(expected));
        space.return_allocation(expected, class);
        assert_eq!(
            space.largest(class),
            if manager == ManagerKind::PagedSmallMetadata {
                128
            } else {
                256
            }
        );
        assert_eq!(space.allocate(64, class), Some(expected));
    }

    #[test]
    fn metadata_array_index_retains_metadata_intent_on_a_raw_page() {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            1,
            FileSpacePageSize::DEFAULT,
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(4_112, 128).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallRaw),
            Extent::new(8_208, 256).unwrap(),
        );
        space.finish_seed();
        let snapshot = space.allocation_snapshot();
        let index = space
            .allocate_with_page_type(64, AllocationClass::Metadata, PageType::Raw)
            .unwrap();
        assert_eq!(index, Extent::new(8_208, 64).unwrap());
        space.return_allocation_with_page_type(index, AllocationClass::Metadata, PageType::Raw);
        assert_eq!(space.allocate(64, AllocationClass::Raw), Some(index));
        space.restore_allocation(snapshot);
        assert_eq!(space.allocate(64, AllocationClass::Raw), Some(index));
        assert_eq!(
            space.allocate(64, AllocationClass::Metadata),
            Some(Extent::new(4_112, 64).unwrap())
        );
    }

    const THRESHOLD: u64 = 17;

    #[test]
    fn fsm_aggr_constructs_flat_space() {
        let space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::FsmAggr),
            THRESHOLD,
            FileSpacePageSize::DEFAULT,
        );
        let SessionSpace::Flat(flat) = space else {
            panic!("FSM_AGGR must construct flat reusable space");
        };
        assert_eq!(flat.threshold, THRESHOLD);
    }

    #[test]
    fn page_constructs_paged_space() {
        let page_size = FileSpacePageSize::try_from(8192).unwrap();
        let space =
            SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), THRESHOLD, page_size);
        let SessionSpace::Paged(paged) = space else {
            panic!("PAGE must construct paged reusable space");
        };
        assert_eq!(paged.threshold, THRESHOLD);
        assert_eq!(paged.paged.page_size(), page_size);
    }

    #[test]
    fn aggr_constructs_disabled_space() {
        assert!(matches!(
            SessionSpace::from_strategy(
                Some(FileSpaceStrategy::Aggr),
                THRESHOLD,
                FileSpacePageSize::DEFAULT,
            ),
            SessionSpace::Disabled
        ));
    }

    #[test]
    fn none_constructs_disabled_space() {
        assert!(matches!(
            SessionSpace::from_strategy(
                Some(FileSpaceStrategy::None),
                THRESHOLD,
                FileSpacePageSize::DEFAULT,
            ),
            SessionSpace::Disabled
        ));
    }

    #[test]
    fn unreadable_strategy_constructs_disabled_space() {
        assert!(matches!(
            SessionSpace::from_strategy(None, THRESHOLD, FileSpacePageSize::DEFAULT),
            SessionSpace::Disabled
        ));
    }
}
