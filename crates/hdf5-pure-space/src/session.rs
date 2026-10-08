use alloc::vec::Vec;

use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_core::FormatError;

use super::extent::Extent;
use super::list::{self, FreeList};
use super::paged::{
    FreeClass, PageTransition, PageType, PagedAllocationSnapshot, PagedEdit, PagedPostFree,
    PagedSections,
};
use super::persistence::ManagerKind;

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

    /// Allocates `len` reusable bytes for `ty`.
    pub fn allocate(&mut self, len: u64, ty: PageType) -> Option<Extent> {
        match self {
            Self::Disabled => None,
            Self::Flat(space) => space.free.alloc(len),
            Self::Paged(space) => space.paged.allocate(len, ty),
        }
    }

    /// Returns an allocation to the strategy it came from.
    pub fn return_allocation(&mut self, extent: Extent, ty: PageType) {
        match self {
            Self::Disabled => {}
            Self::Flat(space) => space.free.free(extent),
            Self::Paged(space) => space.paged.return_allocation(extent, ty),
        }
    }

    /// Returns the largest reusable run available for `ty`.
    pub fn largest(&self, ty: PageType) -> u64 {
        match self {
            Self::Disabled => 0,
            Self::Flat(space) => space.free.largest(),
            Self::Paged(space) => space.paged.largest(ty),
        }
    }

    /// Returns reusable sections in ascending address order.
    pub fn reusable_sections(&self) -> Vec<(u64, u64)> {
        match self {
            Self::Disabled => Vec::new(),
            Self::Flat(space) => space.free.sections(),
            Self::Paged(space) => space.paged.reusable_sections(),
        }
    }

    /// Returns reusable sections with extra raw allocations folded back into the snapshot.
    pub fn reusable_sections_with_raw(
        &self,
        raw_extents: impl IntoIterator<Item = Extent>,
    ) -> Vec<(u64, u64)> {
        match self {
            Self::Disabled => Vec::new(),
            Self::Flat(space) => {
                let mut free = space.free.clone();
                for extent in raw_extents {
                    free.free(extent);
                }
                free.sections()
            }
            Self::Paged(space) => space.paged.reusable_sections_with_raw(raw_extents),
        }
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
    use super::*;

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
