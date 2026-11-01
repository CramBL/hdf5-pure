//! Reusable-space reservations and indexed batch accounting.

use core::num::NonZeroU64;

use hdf5_pure_core::FileSpacePageSize;

use crate::allocation::AllocationClass;
use crate::extent::Extent;
use crate::list::FreeList;
use crate::list::indexed::IndexedFreeList;
use crate::paged::PageType;
use crate::session::SessionSpace;

/// A nonempty allocation request with its purpose and effective PAGE type.
///
/// The caller selects PAGE placement independently of the allocation class. Reservation addresses use
/// stored coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReservationRequest {
    size: NonZeroU64,
    class: AllocationClass,
    page_type: PageType,
}

impl ReservationRequest {
    /// Creates a request using the PAGE type corresponding to `class`.
    pub fn new(size: NonZeroU64, class: AllocationClass) -> Self {
        Self::with_page_type(size, class, class.into())
    }

    /// Creates a request with a PAGE type independent of its allocation purpose.
    pub fn with_page_type(size: NonZeroU64, class: AllocationClass, page_type: PageType) -> Self {
        Self {
            size,
            class,
            page_type,
        }
    }

    /// Returns the allocation length in bytes.
    pub fn size(self) -> NonZeroU64 {
        self.size
    }

    /// Returns the allocation's purpose.
    pub fn class(self) -> AllocationClass {
        self.class
    }

    /// Returns the page type used for PAGE placement and return.
    pub fn page_type(self) -> PageType {
        self.page_type
    }
}

/// A complete stored-address allocation removed from reusable space.
///
/// Dropping this value leaves its bytes unavailable. The caller establishes whether the allocation
/// can be returned through [`SessionSpace::return_allocation_with_page_type`].
#[derive(Debug, Eq, PartialEq)]
pub struct ReusableReservation {
    extent: Extent,
    class: AllocationClass,
    page_type: PageType,
}

impl ReusableReservation {
    /// Returns the complete reserved range in stored coordinates.
    pub fn extent(&self) -> Extent {
        self.extent
    }

    /// Returns the allocation's purpose.
    pub fn class(&self) -> AllocationClass {
        self.class
    }

    /// Returns the page type used for PAGE placement and return.
    pub fn page_type(&self) -> PageType {
        self.page_type
    }

    /// Consumes the reservation into the arguments for an explicit return to reusable space.
    pub fn into_parts(self) -> (Extent, AllocationClass, PageType) {
        let Self {
            extent,
            class,
            page_type,
        } = self;
        (extent, class, page_type)
    }
}

/// Sequential reusable-space reservations under one mutable allocator borrow.
///
/// The caller creates a batch through [`SessionSpace::reservation_batch`]. Finishing or dropping
/// the batch restores unused free space and keeps every successful reservation unavailable.
/// Publication and abandonment of successful reservations remain the caller's responsibility.
pub struct ReservationBatch<'a> {
    state: ReservationState<'a>,
}

impl<'a> ReservationBatch<'a> {
    pub(crate) fn scalar(space: &'a mut SessionSpace) -> Self {
        Self {
            state: ReservationState::Scalar(space),
        }
    }

    pub(crate) fn disabled() -> Self {
        Self {
            state: ReservationState::Disabled,
        }
    }

    pub(crate) fn flat(free: &'a mut FreeList) -> Self {
        Self {
            state: ReservationState::Flat(IndexedFreeList::new(free, None)),
        }
    }

    pub(crate) fn paged(
        meta: &'a mut FreeList,
        raw: &'a mut FreeList,
        page_size: FileSpacePageSize,
    ) -> Self {
        // FileSpacePageSize construction requires at least 512 bytes.
        let alignment = NonZeroU64::new(page_size.get()).expect("the page size is nonzero");
        Self {
            state: ReservationState::Paged {
                meta: IndexedFreeList::new(meta, Some(alignment)),
                raw: IndexedFreeList::new(raw, Some(alignment)),
                page_size,
            },
        }
    }

    /// Reserves bytes under the allocator's best-fit and PAGE policies.
    ///
    /// Equal-sized eligible regions are selected by lowest stored address. Returns [`None`] if no
    /// eligible region fits, including when the required PAGE rounding exceeds [`u64::MAX`]. Prior
    /// successful reservations remain unavailable after a failed request.
    pub fn reserve(&mut self, request: ReservationRequest) -> Option<ReusableReservation> {
        let ReservationRequest {
            size,
            class,
            page_type,
        } = request;
        let extent = match &mut self.state {
            ReservationState::Disabled => None,
            ReservationState::Scalar(space) => {
                space.allocate_with_page_type(size.get(), class, page_type)
            }
            ReservationState::Flat(free) => free.allocate(size),
            ReservationState::Paged {
                meta,
                raw,
                page_size,
            } => {
                let (own, other) = match page_type {
                    PageType::Meta => (meta, raw),
                    PageType::Raw => (raw, meta),
                };
                own.allocate(size).or_else(|| {
                    let span =
                        NonZeroU64::new(size.get().checked_next_multiple_of(page_size.get())?)?;
                    let claimed = other.allocate_whole_units(span)?;
                    // The claimed whole-page extent is at least as long as the nonempty request.
                    let extent = Extent::new(claimed.start(), size.get())
                        .expect("the allocation fits inside the claimed pages");
                    if let Some(remainder) = Extent::between(extent.end(), claimed.end()) {
                        own.free(remainder);
                    }
                    Some(extent)
                })
            }
        }?;
        Some(ReusableReservation {
            extent,
            class,
            page_type,
        })
    }

    /// Restores unused free space and returns the final indexed-work counts.
    pub fn finish(mut self) -> ReservationWork {
        self.restore();
        self.work()
    }

    /// Returns indexed-work counts accumulated so far.
    ///
    /// Scalar and disabled paths report zero. Restoration is counted when the batch finishes.
    pub fn work(&self) -> ReservationWork {
        match &self.state {
            ReservationState::Disabled | ReservationState::Scalar(_) => ReservationWork::default(),
            ReservationState::Flat(free) => free.work(),
            ReservationState::Paged {
                meta,
                raw,
                page_size: _,
            } => meta.work().combine(raw.work()),
        }
    }

    fn restore(&mut self) {
        match &mut self.state {
            ReservationState::Disabled | ReservationState::Scalar(_) => {}
            ReservationState::Flat(free) => free.restore(),
            ReservationState::Paged {
                meta,
                raw,
                page_size: _,
            } => {
                meta.restore();
                raw.restore();
            }
        }
    }
}

impl Drop for ReservationBatch<'_> {
    fn drop(&mut self) {
        self.restore();
    }
}

enum ReservationState<'a> {
    Disabled,
    Flat(IndexedFreeList<'a>),
    Paged {
        meta: IndexedFreeList<'a>,
        raw: IndexedFreeList<'a>,
        page_size: FileSpacePageSize,
    },
    Scalar(&'a mut SessionSpace),
}

/// Saturating operation counts for the indexed paths of a reservation batch.
///
/// Counts describe extent operations, excluding ordered-map comparisons and scalar-path work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReservationWork {
    /// Original extents inserted during index construction.
    pub indexed_extents: u64,
    /// Best-fit queries, including unsuccessful queries and PAGE fallback queries.
    pub lookups: u64,
    /// Extents removed from the address map and lookup indexes.
    pub removed_extents: u64,
    /// Extents inserted during construction, splitting, and remainder merging.
    pub inserted_extents: u64,
    /// Remaining extents copied back into the owning free lists.
    pub restored_extents: u64,
}

impl ReservationWork {
    fn combine(self, other: Self) -> Self {
        let Self {
            indexed_extents,
            lookups,
            removed_extents,
            inserted_extents,
            restored_extents,
        } = self;
        Self {
            indexed_extents: indexed_extents.saturating_add(other.indexed_extents),
            lookups: lookups.saturating_add(other.lookups),
            removed_extents: removed_extents.saturating_add(other.removed_extents),
            inserted_extents: inserted_extents.saturating_add(other.inserted_extents),
            restored_extents: restored_extents.saturating_add(other.restored_extents),
        }
    }
}
