use core::iter::Peekable;

use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::StoredAddress;

use super::admission::TrackedSpace;
use super::allocation::AllocationClass;
use super::extent::Extent;
use super::list::{self, FreeList, FreeListError};
use super::paged::{
    FreeClass, PageTransition, PageType, PagedAllocationSnapshot, PagedAppendSnapshot, PagedEdit,
    PagedPostFree, PagedSections, PagedTail,
};
use super::persistence::ManagerKind;
use super::reservation::ReservationBatch;
use super::snapshot::AdmittedFreeSnapshot;

/// The reusable-space strategy active for one editing session.
///
/// Extent endpoints and end-of-allocation positions are [`StoredAddress`] values relative to
/// the file's address base.
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

/// Stores reusable-space state for an append rollback after truncation.
///
/// For PAGE, includes all free lists, their cached totals, the tail page type, and pending padding.
pub enum SessionAppendSnapshot {
    /// Reusable allocation is disabled.
    Disabled,
    /// The unpaged free list and its cached total.
    Flat(FreeList),
    /// The complete PAGE state.
    Paged(PagedAppendSnapshot),
}

/// Merges two address-sorted extent iterators without allocating another snapshot.
pub(super) struct MergeExtents<I, J>
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
    pub(super) fn new(first: I, second: J) -> Self {
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

    /// Constructs canonical reusable space from admitted sections in ascending address order.
    ///
    /// Coalesces adjacent sections within each allocation class. Under `FSM_AGGR`, all sections
    /// share one free list. Under PAGE, sections from small managers retain their page type, and
    /// complete pages covered by classified sections become reusable raw space. Generic-large
    /// aligned interiors are raw space. Their partial edges and sections from other manager kinds
    /// remain unclassified.
    ///
    /// Records all admitted sections regardless of `threshold`, which applies to subsequent
    /// frees. For disabled strategies, returns [`Self::Disabled`] without consuming `sections`.
    /// Runs in linear time in the number of sections.
    ///
    /// # Errors
    ///
    /// Returns [`FreeListError::Unsorted`] if a section starts before the preceding section, or
    /// [`FreeListError::Overlap`] if their starts are ordered but their ranges overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{AllocationClass, Extent, ManagerKind, SessionSpace};
    ///
    /// let metadata = Extent::new(StoredAddress::new(4_096), 1_024).unwrap();
    /// let raw = Extent::new(metadata.end(), 3_072).unwrap();
    /// let mut space = SessionSpace::from_persisted(
    ///     Some(FileSpaceStrategy::Page),
    ///     8_192,
    ///     FileSpacePageSize::DEFAULT,
    ///     [
    ///         (Some(ManagerKind::PagedSmallMetadata), metadata),
    ///         (Some(ManagerKind::PagedSmallRaw), raw),
    ///     ],
    /// )?;
    /// let page = Extent::new(metadata.start(), 4_096).unwrap();
    /// assert_eq!(space.paged_sections().unwrap().raw(), [page]);
    /// assert_eq!(space.reusable_bytes().unwrap(), 4_096);
    /// assert_eq!(space.allocate(4_096, AllocationClass::Metadata), Some(page));
    /// # Ok::<(), hdf5_pure_space::__private::FreeListError>(())
    /// ```
    pub fn from_persisted(
        strategy: Option<FileSpaceStrategy>,
        threshold: u64,
        page_size: FileSpacePageSize,
        sections: impl IntoIterator<Item = (Option<ManagerKind>, Extent)>,
    ) -> Result<Self, FreeListError> {
        Ok(match strategy {
            Some(FileSpaceStrategy::FsmAggr) => Self::Flat(FlatSpace {
                free: FreeList::from_sorted(sections.into_iter().map(|(_, extent)| extent))?,
                threshold,
            }),
            Some(FileSpaceStrategy::Page) => Self::Paged(PagedSpace {
                paged: PagedEdit::from_persisted(page_size, sections)?,
                threshold,
            }),
            Some(FileSpaceStrategy::Aggr) | Some(FileSpaceStrategy::None) | None => Self::Disabled,
        })
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
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{FreeClass, PageType, SessionSpace};
    ///
    /// let page_size = FileSpacePageSize::DEFAULT;
    /// let space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 1, page_size,
    /// );
    /// let stored_address = StoredAddress::new(page_size.get());
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
    ///     space.object_header_free_class(stored_address.checked_add(1).unwrap(), complete_length),
    ///     FreeClass::Page(PageType::Meta),
    /// );
    /// ```
    pub fn object_header_free_class(&self, address: StoredAddress, length: u64) -> FreeClass {
        if self
            .page_size()
            .is_some_and(|page| address.get().is_multiple_of(page.get()) && length >= page.get())
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

    /// Borrows reusable space for sequential reservations with shared lookup state.
    ///
    /// `requests` is the expected request count, used to select the lookup mechanism. A count of at
    /// most one, a disabled allocator, or a flat list with at most one extent uses scalar allocation.
    /// Other batches index each reusable list once. The count does not limit calls to
    /// [`ReservationBatch::reserve`].
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroU64;
    ///
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{AllocationClass, Extent, ReservationRequest, SessionSpace};
    ///
    /// let extent = |start, len| Extent::new(StoredAddress::new(start), len).unwrap();
    /// let mut space = SessionSpace::from_persisted(
    ///     Some(FileSpaceStrategy::FsmAggr),
    ///     1,
    ///     FileSpacePageSize::DEFAULT,
    ///     [(None, extent(100, 100)), (None, extent(300, 100))],
    /// )?;
    /// let request = |size| ReservationRequest::new(
    ///     NonZeroU64::new(size).unwrap(), AllocationClass::Raw,
    /// );
    /// let mut batch = space.reservation_batch(2);
    /// let reserved = batch.reserve(request(64)).unwrap();
    /// assert_eq!(reserved.extent(), extent(100, 64));
    /// assert_eq!(batch.reserve(request(128)), None);
    /// batch.finish();
    /// assert_eq!(space.reusable_sections().collect::<Vec<_>>(),
    ///     [extent(164, 36), extent(300, 100)]);
    /// let (extent, class, page_type) = reserved.into_parts();
    /// space.return_allocation_with_page_type(extent, class, page_type);
    /// assert_eq!(space.reusable_bytes()?, 200);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reservation_batch(&mut self, requests: usize) -> ReservationBatch<'_> {
        let scalar = requests <= 1
            || match self {
                Self::Disabled => true,
                Self::Flat(space) => space.free.extents().len() <= 1,
                Self::Paged(_) => false,
            };
        if scalar {
            return ReservationBatch::scalar(self);
        }
        match self {
            Self::Disabled => ReservationBatch::disabled(),
            Self::Flat(space) => ReservationBatch::flat(&mut space.free, space.threshold),
            Self::Paged(space) => space.paged.reservation_batch(),
        }
    }

    /// Borrows reusable space for reservations interleaved with aggregate returns.
    ///
    /// Flat batches with more than one expected request use an index even when initially empty or
    /// contiguous. Other cases use [`Self::reservation_batch`].
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroU64;
    ///
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{AllocationClass, Extent, ReservationRequest, SessionSpace};
    ///
    /// let mut space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::FsmAggr), 1, FileSpacePageSize::DEFAULT,
    /// );
    /// let spare = Extent::new(StoredAddress::new(100), 32).unwrap();
    /// let mut batch = space.reservation_batch_with_returns(2);
    /// assert!(batch.release_aggregate(StoredAddress::new(1_000), spare));
    /// let request = ReservationRequest::new(NonZeroU64::new(16).unwrap(), AllocationClass::Raw);
    /// assert_eq!(batch.reserve(request).unwrap().extent(),
    ///     Extent::new(StoredAddress::new(100), 16).unwrap());
    /// batch.finish();
    /// assert_eq!(space.reusable_sections().collect::<Vec<_>>(),
    ///     [Extent::new(StoredAddress::new(116), 16).unwrap()]);
    /// ```
    pub fn reservation_batch_with_returns(&mut self, requests: usize) -> ReservationBatch<'_> {
        if requests <= 1 {
            return self.reservation_batch(requests);
        }
        match self {
            Self::Flat(space) => ReservationBatch::flat(&mut space.free, space.threshold),
            _ => self.reservation_batch(requests),
        }
    }

    /// Reserves `len` reusable bytes for `class`, or returns `None` if no matching space is available.
    ///
    /// PAGE requests use the corresponding [`PageType`]. Flat allocation uses one free list for
    /// both classes.
    pub fn allocate(&mut self, len: u64, class: AllocationClass) -> Option<Extent> {
        self.allocate_with_page_type(len, class, class.into())
    }

    /// Reserves `len` bytes from the smallest flat free region whose allocation ends at or below
    /// `limit`.
    ///
    /// Allocates from the start of the region and retains any remainder, including one shorter
    /// than the tracking threshold. Equal-sized regions are selected by lowest address.
    /// Returns `None` and preserves the session if `len` is zero, no region fits, or the strategy
    /// is disabled or paged.
    pub fn allocate_flat_below(&mut self, len: u64, limit: StoredAddress) -> Option<Extent> {
        let Self::Flat(space) = self else {
            return None;
        };
        space.free.alloc_below(len, limit)
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
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{
    ///     AllocationClass, Extent, ManagerKind, PageType, SessionSpace,
    /// };
    ///
    /// let mut space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT,
    /// );
    /// space.seed_persisted(
    ///     Some(ManagerKind::PagedSmallRaw), Extent::new(StoredAddress::new(8_208), 256).unwrap(),
    /// );
    /// space.finish_seed();
    /// let raw = space.allocate(64, AllocationClass::Raw).unwrap();
    /// space.return_allocation(raw, AllocationClass::Raw);
    /// let index = space.allocate_with_page_type(
    ///     64, AllocationClass::Metadata, PageType::Raw,
    /// ).unwrap();
    /// assert_eq!(index, Extent::new(StoredAddress::new(8_208), 64).unwrap());
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

    /// Returns the sum of the cached reusable free-list byte totals.
    ///
    /// Returns zero when reusable allocation is disabled. The caller accounts for append
    /// reserves and aggregate reservations separately.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the paged free-list total exceeds [`u64::MAX`].
    pub fn reusable_bytes(&self) -> Result<u64, FormatError> {
        self.admitted_snapshot().reusable_bytes()
    }

    /// Returns reusable extents in ascending address order.
    pub fn reusable_sections(&self) -> impl Iterator<Item = Extent> + '_ {
        self.admitted_snapshot().reusable_sections()
    }

    /// Returns reusable extents with extra raw allocations folded back into the snapshot.
    pub fn reusable_sections_with_raw(
        &self,
        raw_extents: impl IntoIterator<Item = Extent>,
    ) -> impl Iterator<Item = Extent> + '_ {
        self.admitted_snapshot()
            .reusable_sections_with_raw(raw_extents)
    }

    /// Borrows admitted free extents, their classes, and their cached byte totals.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{AllocationClass, Extent, FreeClass, SessionSpace};
    ///
    /// let free = Extent::new(StoredAddress::new(4_096), 64).unwrap();
    /// let mut space = SessionSpace::from_persisted(
    ///     Some(FileSpaceStrategy::FsmAggr),
    ///     1,
    ///     FileSpacePageSize::DEFAULT,
    ///     [(None, free)],
    /// )?;
    /// let snapshot = space.admitted_snapshot();
    /// assert_eq!(snapshot.reusable_bytes()?, 64);
    /// assert_eq!(snapshot.sections().collect::<Vec<_>>(), [(free, FreeClass::Dead)]);
    /// assert_eq!(
    ///     space.allocate(32, AllocationClass::Raw),
    ///     Extent::new(free.start(), 32),
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn admitted_snapshot(&self) -> AdmittedFreeSnapshot<'_> {
        match self {
            Self::Disabled => AdmittedFreeSnapshot::disabled(),
            Self::Flat(space) => AdmittedFreeSnapshot::flat(&space.free),
            Self::Paged(space) => space.paged.admitted_snapshot(),
        }
    }

    /// Returns the start of the free run reaching `eoa` under this strategy.
    pub fn trailing_run_start(&self, eoa: StoredAddress) -> StoredAddress {
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
        eof: StoredAddress,
        extents: impl IntoIterator<Item = Extent>,
    ) -> Option<StoredAddress> {
        let Self::Flat(space) = self else {
            return None;
        };
        let eoa = space.free.release_all(space.threshold, eof, extents);
        let eoa = space.free.take_trailing(eoa).unwrap_or(eoa);
        (eoa < eof).then_some(eoa)
    }

    /// Records an already retired extent under the session's admission policy.
    ///
    /// The caller establishes retirement and persistence ordering. `eoa` and `extent` use stored
    /// addresses. Under disabled strategies, the allocator discards the extent. Under PAGE, the
    /// allocator applies page-class rules and includes pending page padding in the resulting
    /// free-space state.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{
    ///     AllocationClass, Extent, FreeClass, PageType, SessionSpace,
    /// };
    ///
    /// let mut space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 64, FileSpacePageSize::DEFAULT,
    /// );
    /// let retired = Extent::new(StoredAddress::new(8_192), 128).unwrap();
    /// space.record_freed(StoredAddress::new(12_288), retired, FreeClass::Page(PageType::Raw));
    /// assert_eq!(space.allocate(128, AllocationClass::Metadata), None);
    /// assert_eq!(space.allocate(128, AllocationClass::Raw), Some(retired));
    /// ```
    pub fn record_freed(&mut self, eoa: StoredAddress, extent: Extent, class: FreeClass) {
        match self {
            Self::Disabled => {}
            Self::Flat(space) => {
                space.free.release(space.threshold, extent, ());
            }
            Self::Paged(space) => {
                let (post, _) = space
                    .paged
                    .post_free(space.threshold, eoa, [(extent, class)], []);
                space.paged.adopt_post_free(post, false, None);
            }
        }
    }

    /// Builds the flat free-space state that a persisting commit would publish.
    pub fn flat_post_free(
        &self,
        eof: StoredAddress,
        extents: impl IntoIterator<Item = Extent>,
    ) -> Option<(FreeList, StoredAddress)> {
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
        eoa: StoredAddress,
        freed: impl IntoIterator<Item = (Extent, FreeClass)>,
        old_blocks: impl IntoIterator<Item = Extent>,
    ) -> Option<(PagedPostFree, StoredAddress)> {
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
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::Extent;
    /// use hdf5_pure_space::__private::FreeClass;
    /// use hdf5_pure_space::__private::ManagerKind;
    /// use hdf5_pure_space::__private::PageType;
    /// use hdf5_pure_space::__private::SessionSpace;
    ///
    /// let mut space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT,
    /// );
    /// let durable = Extent::new(StoredAddress::new(4_096), 512).unwrap();
    /// space.seed_persisted(Some(ManagerKind::PagedSmallMetadata), durable);
    /// let freed = [(
    ///     Extent::new(StoredAddress::new(8_192), 128).unwrap(),
    ///     FreeClass::Page(PageType::Meta),
    /// )];
    /// let old_blocks = [Extent::new(StoredAddress::new(8_448), 128).unwrap()];
    /// let limit = space.paged_tail_section_limit(
    ///     freed.iter().map(|&(extent, _)| extent).chain(old_blocks),
    /// )?;
    ///
    /// let reused = space.plan_paged_tail(
    ///     StoredAddress::new(12_288), &freed, &old_blocks, 256, false,
    /// )?.unwrap();
    /// assert_eq!(reused.allocation, Extent::new(StoredAddress::new(4_096), 256).unwrap());
    /// assert_eq!(reused.post.sections().metadata(), [
    ///     Extent::new(StoredAddress::new(4_352), 256).unwrap(), freed[0].0, old_blocks[0],
    /// ]);
    /// assert!(reused.post.sections().metadata().len() <= limit);
    /// assert_eq!(space.paged_sections().unwrap().metadata(), [durable]);
    ///
    /// let appended = space.plan_paged_tail(
    ///     StoredAddress::new(12_288), &freed, &old_blocks, 256, true,
    /// )?.unwrap();
    /// assert_eq!(appended.allocation, Extent::new(StoredAddress::new(12_288), 256).unwrap());
    /// assert_eq!(appended.final_eoa, StoredAddress::new(16_384));
    /// assert_eq!(appended.post.sections().metadata().last(),
    ///     Some(&Extent::new(StoredAddress::new(12_544), 3_840).unwrap()));
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
        eoa: StoredAddress,
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
        eoa: StoredAddress,
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
    pub fn page_padding(&self, eoa: StoredAddress) -> Result<Option<PageTransition>, FormatError> {
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

    /// Copies the reusable-space state preceding an append.
    ///
    /// For PAGE, includes the tail page type and pending padding. The caller restores this state
    /// through [`Self::restore_append`] after discarding unpublished writes and truncating any
    /// appended bytes.
    pub fn append_snapshot(&self) -> SessionAppendSnapshot {
        match self {
            Self::Disabled => SessionAppendSnapshot::Disabled,
            Self::Flat(space) => SessionAppendSnapshot::Flat(space.free.clone()),
            Self::Paged(space) => SessionAppendSnapshot::Paged(space.paged.append_snapshot()),
        }
    }

    /// Restores reusable-space state after truncating an unpublished append.
    ///
    /// The caller discards unpublished writes and truncates any appended bytes before restoration.
    /// For PAGE, restores all free lists, their cached totals, the tail page type, and pending
    /// padding. The tracking threshold is unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
    /// use hdf5_pure_space::__private::{
    ///     AllocationClass, Extent, ManagerKind, PageTransition, PageType, SessionSpace,
    /// };
    ///
    /// let mut space = SessionSpace::from_strategy(
    ///     Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT,
    /// );
    /// let free = Extent::new(StoredAddress::new(4_112), 100).unwrap();
    /// space.seed_persisted(Some(ManagerKind::PagedSmallRaw), free);
    /// space.record_transition(PageType::Raw, PageTransition::Unchanged);
    /// let snapshot = space.append_snapshot();
    /// assert_eq!(
    ///     space.allocate(50, AllocationClass::Raw),
    ///     Extent::new(StoredAddress::new(4_112), 50),
    /// );
    /// let padding = space.page_padding(StoredAddress::new(8_208))?.unwrap();
    /// space.record_page_padding(padding);
    /// space.record_transition(PageType::Meta, PageTransition::Unchanged);
    /// space.restore_append(snapshot);
    /// assert_eq!(space.reusable_sections().collect::<Vec<_>>(), [free]);
    /// assert_eq!(space.paged_tail_type(), Some(Some(PageType::Raw)));
    /// assert_eq!(space.pending_padding(PageType::Raw), Some([].as_slice()));
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `snapshot` belongs to a different strategy.
    pub fn restore_append(&mut self, snapshot: SessionAppendSnapshot) {
        match (self, snapshot) {
            (Self::Disabled, SessionAppendSnapshot::Disabled) => {}
            (Self::Flat(space), SessionAppendSnapshot::Flat(free)) => space.free = free,
            (Self::Paged(space), SessionAppendSnapshot::Paged(paged)) => {
                space.paged.restore_append(paged);
            }
            _ => debug_assert!(false, "a session append snapshot changed strategy"),
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
    use alloc::vec::Vec;

    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_core::StoredAddress;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::disabled(FileSpaceStrategy::None)]
    #[case::flat(FileSpaceStrategy::FsmAggr)]
    #[case::page(FileSpaceStrategy::Page)]
    fn append_snapshot_restores_the_session_strategy_state(#[case] strategy: FileSpaceStrategy) {
        let mut space = SessionSpace::from_strategy(Some(strategy), 1, FileSpacePageSize::DEFAULT);
        let original = Extent::new(StoredAddress::new(4_112), 100).unwrap();
        space.seed_persisted(Some(ManagerKind::PagedSmallRaw), original);
        space.record_transition(PageType::Raw, PageTransition::Unchanged);
        let snapshot = space.append_snapshot();
        let prior = space.reusable_sections().collect::<Vec<_>>();
        let tail = space.paged_tail_type();
        let allocated = space.allocate(50, AllocationClass::Raw);
        assert_eq!(
            allocated,
            if strategy == FileSpaceStrategy::None {
                None
            } else {
                Extent::new(StoredAddress::new(4_112), 50)
            }
        );
        if let Some(transition) = space.page_padding(StoredAddress::new(8_208)).unwrap() {
            space.record_page_padding(transition);
        }
        space.record_transition(PageType::Meta, PageTransition::Unchanged);
        space.restore_append(snapshot);
        assert_eq!(space.reusable_sections().collect::<Vec<_>>(), prior);
        assert_eq!(space.paged_tail_type(), tail);
        assert_eq!(
            space.pending_padding(PageType::Raw),
            if strategy == FileSpaceStrategy::Page {
                Some([].as_slice())
            } else {
                None
            }
        );
        assert_eq!(
            space.pending_padding(PageType::Meta),
            if strategy == FileSpaceStrategy::Page {
                Some([].as_slice())
            } else {
                None
            }
        );
    }

    #[test]
    fn flat_mutations_update_and_restore_cached_total() {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::FsmAggr),
            50,
            FileSpacePageSize::DEFAULT,
        );
        let total = |space: &SessionSpace| {
            let SessionSpace::Flat(flat) = space else {
                panic!("expected flat space");
            };
            flat.free.total_bytes()
        };
        space.seed_persisted(None, Extent::new(StoredAddress::new(100), 100).unwrap());
        space.seed_persisted(None, Extent::new(StoredAddress::new(300), 60).unwrap());
        assert_eq!(total(&space), 160);
        let snapshot = space.allocation_snapshot();
        let allocation = space.allocate(40, AllocationClass::Raw).unwrap();
        assert_eq!(
            allocation,
            Extent::new(StoredAddress::new(300), 40).unwrap()
        );
        assert_eq!(total(&space), 120);
        space.return_allocation(allocation, AllocationClass::Raw);
        assert_eq!(total(&space), 160);
        let (post, eoa) = space
            .flat_post_free(
                StoredAddress::new(1_000),
                [
                    Extent::new(StoredAddress::new(500), 30).unwrap(),
                    Extent::new(StoredAddress::new(550), 100).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(eoa, StoredAddress::new(1_000));
        assert_eq!(post.total_bytes(), 260);
        assert_eq!(total(&space), 160);
        space.adopt_flat_post_free(post);
        assert_eq!(total(&space), 260);
        space.restore_allocation(snapshot);
        assert_eq!(total(&space), 160);
        assert_eq!(
            space.reusable_sections().collect::<Vec<_>>(),
            [
                Extent::new(StoredAddress::new(100), 100).unwrap(),
                Extent::new(StoredAddress::new(300), 60).unwrap()
            ]
        );
        assert_eq!(
            space.release_freed(
                StoredAddress::new(1_000),
                [Extent::new(StoredAddress::new(900), 100).unwrap()]
            ),
            Some(StoredAddress::new(900))
        );
        assert_eq!(total(&space), 160);
    }

    #[test]
    fn flat_limit_allocation_retains_remainders_below_threshold() {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::FsmAggr),
            50,
            FileSpacePageSize::DEFAULT,
        );
        space.seed_persisted(None, Extent::new(StoredAddress::new(100), 100).unwrap());
        space.seed_persisted(None, Extent::new(StoredAddress::new(300), 60).unwrap());
        assert_eq!(
            space.allocate_flat_below(40, StoredAddress::new(340)),
            Extent::new(StoredAddress::new(300), 40)
        );
        assert_eq!(
            space.reusable_sections().collect::<Vec<_>>(),
            [
                Extent::new(StoredAddress::new(100), 100).unwrap(),
                Extent::new(StoredAddress::new(340), 20).unwrap()
            ]
        );
        space.return_allocation(
            Extent::new(StoredAddress::new(300), 40).unwrap(),
            AllocationClass::Metadata,
        );
        assert_eq!(
            space.reusable_sections().collect::<Vec<_>>(),
            [
                Extent::new(StoredAddress::new(100), 100).unwrap(),
                Extent::new(StoredAddress::new(300), 60).unwrap()
            ]
        );
    }

    #[rstest]
    #[case::below_threshold(32, false, 0)]
    #[case::at_threshold(64, false, 64)]
    #[case::above_threshold(128, false, 128)]
    #[case::merges_below_threshold(32, true, 48)]
    fn recorded_frees_obey_strategy_threshold_and_raw_page_class(
        #[case] len: u64,
        #[case] neighbor: bool,
        #[case] expected: u64,
        #[values(
            FileSpaceStrategy::FsmAggr,
            FileSpaceStrategy::Page,
            FileSpaceStrategy::Aggr,
            FileSpaceStrategy::None
        )]
        strategy: FileSpaceStrategy,
    ) {
        let mut space = SessionSpace::from_strategy(Some(strategy), 64, FileSpacePageSize::DEFAULT);
        let start = 8192;
        if neighbor {
            space.seed_persisted(
                Some(ManagerKind::PagedSmallRaw),
                Extent::new(StoredAddress::new(start), 16).unwrap(),
            );
        }
        space.finish_seed();
        let extent = Extent::new(
            StoredAddress::new(start + if neighbor { 16 } else { 0 }),
            len,
        )
        .unwrap();
        space.record_freed(extent.end(), extent, FreeClass::Page(PageType::Raw));
        let expected = if space.is_enabled() { expected } else { 0 };
        assert_eq!(space.reusable_bytes().unwrap(), expected);
        if space.is_paged() {
            assert_eq!(space.allocate(1, AllocationClass::Metadata), None);
        }
        if expected > 0 {
            assert_eq!(
                space.allocate(expected, AllocationClass::Raw),
                Extent::new(StoredAddress::new(start), expected)
            );
        } else {
            assert_eq!(space.allocate(1, AllocationClass::Raw), None);
        }
        assert_eq!(space.reusable_bytes().unwrap(), 0);
    }

    #[rstest]
    #[case::disabled(Some(FileSpaceStrategy::None))]
    #[case::paged(Some(FileSpaceStrategy::Page))]
    fn flat_limit_allocation_preserves_other_strategies(
        #[case] strategy: Option<FileSpaceStrategy>,
    ) {
        let mut space = SessionSpace::from_strategy(strategy, 1, FileSpacePageSize::DEFAULT);
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(StoredAddress::new(4_112), 128).unwrap(),
        );
        space.finish_seed();
        let before = space.reusable_sections().collect::<Vec<_>>();
        assert_eq!(
            space.allocate_flat_below(64, StoredAddress::new(4_176)),
            None
        );
        assert_eq!(space.reusable_sections().collect::<Vec<_>>(), before);
    }

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
                StoredAddress::new(2 * page_size + offset),
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
            space.object_header_free_class(StoredAddress::new(4_096), 8_193),
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
            Extent::new(StoredAddress::new(4_112), 128).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallRaw),
            Extent::new(StoredAddress::new(8_208), 256).unwrap(),
        );
        space.finish_seed();
        let expected = Extent::new(StoredAddress::new(address), 64).unwrap();
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
            Extent::new(StoredAddress::new(4_112), 128).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallRaw),
            Extent::new(StoredAddress::new(8_208), 256).unwrap(),
        );
        space.finish_seed();
        let snapshot = space.allocation_snapshot();
        let index = space
            .allocate_with_page_type(64, AllocationClass::Metadata, PageType::Raw)
            .unwrap();
        assert_eq!(index, Extent::new(StoredAddress::new(8_208), 64).unwrap());
        space.return_allocation_with_page_type(index, AllocationClass::Metadata, PageType::Raw);
        assert_eq!(space.allocate(64, AllocationClass::Raw), Some(index));
        space.restore_allocation(snapshot);
        assert_eq!(space.allocate(64, AllocationClass::Raw), Some(index));
        assert_eq!(
            space.allocate(64, AllocationClass::Metadata),
            Some(Extent::new(StoredAddress::new(4_112), 64).unwrap())
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
