//! Reusable-space reservations and indexed batch accounting.

use core::num::NonZeroU64;

use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::{FileSpacePageSize, FormatError, StoredAddress};

use crate::admission::TrackedSpace;
use crate::allocation::AllocationClass;
use crate::extent::Extent;
use crate::list::FreeList;
use crate::list::indexed::IndexedFreeList;
use crate::paged::{FreeClass, PageType};
use crate::session::SessionSpace;

/// A nonempty allocation request with its purpose and effective PAGE type.
///
/// The caller selects PAGE placement independently of the allocation class. Reservation addresses use
/// stored coordinates. Constructors create compound requests. Use [`Self::for_block`] for one
/// independently read block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReservationRequest {
    size: NonZeroByteLength,
    class: AllocationClass,
    page_type: PageType,
    kind: ReservationKind,
}

impl ReservationRequest {
    /// Creates a request using the PAGE type corresponding to `class`.
    pub fn new(size: NonZeroByteLength, class: AllocationClass) -> Self {
        Self::with_page_type(size, class, class.into())
    }

    /// Creates a request with a PAGE type independent of its allocation purpose.
    pub fn with_page_type(
        size: NonZeroByteLength,
        class: AllocationClass,
        page_type: PageType,
    ) -> Self {
        Self {
            size,
            class,
            page_type,
            kind: ReservationKind::Compound,
        }
    }

    /// Returns a request for one independently read block.
    ///
    /// With PAGE allocation, a block smaller than a page fits within one page. A block of at least
    /// one page starts on a page boundary. These rules apply to the complete reserved length,
    /// including capacity beyond the encoded bytes.
    pub fn for_block(self) -> Self {
        Self {
            kind: ReservationKind::Block,
            ..self
        }
    }

    /// Places the complete allocation at the earliest permitted address at or after `start`.
    ///
    /// Applies the rules of [`Self::for_block`] when this is a block request and `page_size` is
    /// [`Some`]. Otherwise, the allocation starts at `start`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if alignment or the exclusive end exceeds
    /// [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::{ByteLength, NonZeroByteLength, StoredAddressExt};
    /// use hdf5_pure_core::{FileSpacePageSize, StoredAddress};
    /// use hdf5_pure_space::__private::{AllocationClass, Extent, ReservationRequest};
    ///
    /// let request = ReservationRequest::new(
    ///     NonZeroByteLength::new(100).unwrap(), AllocationClass::Metadata,
    /// );
    /// let start = StoredAddress::new(4_000);
    /// let page_size = Some(FileSpacePageSize::DEFAULT);
    /// let compound = request.place_after(start, page_size)?;
    /// assert_eq!(compound.extent(), Extent::new(start, ByteLength::new(100)).unwrap());
    ///
    /// let block = request.for_block().place_after(start, page_size)?;
    /// assert_eq!(block.request(), request.for_block());
    /// assert_eq!(
    ///     block.extent(),
    ///     Extent::new(StoredAddress::new(4_096), ByteLength::new(100)).unwrap(),
    /// );
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn place_after(
        self,
        start: StoredAddress,
        page_size: Option<FileSpacePageSize>,
    ) -> Result<ReservationPlacement, FormatError> {
        let start = if let Some(page_size) = page_size.filter(|_| self.is_block()) {
            let page_size = page_size.get();
            let remaining = page_size - start.get() % page_size;
            if start.get() % page_size != 0
                && (self.size.get() >= page_size || self.size.get() > remaining)
            {
                start.checked_add(remaining)?
            } else {
                start
            }
        } else {
            start
        };
        let extent = Extent::new(start, self.size.length()).ok_or(FormatError::OffsetOverflow {
            offset: start.get(),
            length: self.size.get(),
        })?;
        Ok(ReservationPlacement {
            request: self,
            extent,
        })
    }

    /// Returns whether this request requires placement as one complete block.
    pub fn is_block(self) -> bool {
        self.kind == ReservationKind::Block
    }

    /// Returns the allocation length in bytes.
    pub fn size(self) -> NonZeroByteLength {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReservationKind {
    Block,
    Compound,
}

/// A checked stored-address range for an allocation request.
///
/// The range has the request's complete length. A block placed with a page size satisfies the
/// geometry of [`ReservationRequest::for_block`]. The caller establishes ownership and resolves
/// physical addresses separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReservationPlacement {
    request: ReservationRequest,
    extent: Extent,
}

impl ReservationPlacement {
    /// Returns the request used to establish this placement.
    pub fn request(self) -> ReservationRequest {
        self.request
    }

    /// Returns the complete allocation in stored coordinates.
    pub fn extent(self) -> Extent {
        self.extent
    }
}

/// A complete stored-address allocation removed from reusable space.
///
/// Dropping this value leaves its bytes unavailable. The caller establishes whether the allocation
/// can be returned through [`SessionSpace::return_allocation_with_page_type`].
#[derive(Debug, Eq, PartialEq)]
pub struct ReusableReservation {
    placement: ReservationPlacement,
}

impl ReusableReservation {
    /// Returns the complete reserved range in stored coordinates.
    pub fn extent(&self) -> Extent {
        self.placement.extent()
    }

    /// Returns the checked geometry and request for this reusable allocation.
    pub fn placement(&self) -> ReservationPlacement {
        self.placement
    }

    /// Returns the allocation's purpose.
    pub fn class(&self) -> AllocationClass {
        self.placement.request().class()
    }

    /// Returns the page type used for PAGE placement and return.
    pub fn page_type(&self) -> PageType {
        self.placement.request().page_type()
    }

    /// Consumes the reservation into the arguments for an explicit return to reusable space.
    pub fn into_parts(self) -> (Extent, AllocationClass, PageType) {
        let request = self.placement.request();
        (self.extent(), request.class(), request.page_type())
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

    pub(crate) fn flat(free: &'a mut FreeList, threshold: u64) -> Self {
        Self {
            state: ReservationState::Flat {
                free: IndexedFreeList::new(free, None),
                threshold,
            },
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
    /// Within the request's PAGE type, the allocator compares the largest single-page capacity
    /// for blocks smaller than a page and the page-aligned suffix length for blocks of at least
    /// one page. For compound requests, it compares region lengths. For whole-page claims from
    /// the other type, it compares aligned interiors. Ties use the containing region's lowest
    /// stored address.
    ///
    /// Returns [`None`] if no eligible region fits, including when the required PAGE rounding
    /// exceeds [`u64::MAX`]. Prior successful reservations remain unavailable after a failed request.
    pub fn reserve(&mut self, request: ReservationRequest) -> Option<ReusableReservation> {
        let ReservationRequest {
            size,
            class: _,
            page_type,
            kind,
        } = request;
        let page_size = match &self.state {
            ReservationState::Scalar(space) => space.page_size(),
            ReservationState::Paged { page_size, .. } => Some(*page_size),
            ReservationState::Disabled | ReservationState::Flat { .. } => None,
        };
        let extent = match &mut self.state {
            ReservationState::Disabled => None,
            ReservationState::Scalar(space) => space.allocate_request(request),
            ReservationState::Flat { free, threshold: _ } => free.allocate(size),
            ReservationState::Paged {
                meta,
                raw,
                page_size,
            } => {
                let (own, other) = match page_type {
                    PageType::Meta => (meta, raw),
                    PageType::Raw => (raw, meta),
                };
                let extent = match kind {
                    ReservationKind::Block => own.allocate_block(request, *page_size),
                    ReservationKind::Compound => own.allocate(size),
                };
                extent.or_else(|| {
                    let span = NonZeroByteLength::new(
                        size.get().checked_next_multiple_of(page_size.get())?,
                    )?;
                    let claimed = other.allocate_whole_units(span)?;
                    // The claimed whole-page extent is at least as long as the nonempty request.
                    let extent = Extent::new(claimed.start(), size.length())
                        .expect("the allocation fits inside the claimed pages");
                    if let Some(remainder) = Extent::between(extent.end(), claimed.end()) {
                        own.free(remainder);
                    }
                    Some(extent)
                })
            }
        }?;
        // Each allocation path selects a range that contains the request at its legal start.
        let placement = request
            .place_after(extent.start(), page_size)
            .expect("the selected allocation has representable block geometry");
        debug_assert_eq!(placement.extent(), extent);
        Some(ReusableReservation { placement })
    }

    /// Returns abandoned aggregate spare through ordinary threshold admission.
    ///
    /// Returns `true` for a flat reusable allocator, including when `extent` is below the threshold
    /// and cannot be merged. Returns `false` for PAGE or disabled allocation. The caller supplies a
    /// stored EOA and an allocation disjoint from recorded free space.
    ///
    /// # Panics
    ///
    /// Panics if an admitted extent overlaps an indexed free region. Scalar batches panic in debug
    /// builds for admitted overlaps.
    pub fn release_aggregate(&mut self, eoa: StoredAddress, extent: Extent) -> bool {
        match &mut self.state {
            ReservationState::Flat { free, threshold } => {
                free.release(*threshold, extent, ());
                true
            }
            ReservationState::Scalar(space) if !space.is_paged() && space.is_enabled() => {
                space.record_freed(eoa, extent, FreeClass::Dead);
                true
            }
            ReservationState::Disabled
            | ReservationState::Paged { .. }
            | ReservationState::Scalar(_) => false,
        }
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
            ReservationState::Flat { free, threshold: _ } => free.work(),
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
            ReservationState::Flat { free, threshold: _ } => free.restore(),
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
    Flat {
        free: IndexedFreeList<'a>,
        threshold: u64,
    },
    Paged {
        meta: IndexedFreeList<'a>,
        raw: IndexedFreeList<'a>,
        page_size: FileSpacePageSize,
    },
    Scalar(&'a mut SessionSpace),
}

/// Saturating operation counts for the indexed paths of a reservation batch.
///
/// Counts describe address-map operations and best-fit queries, excluding lookup-index
/// maintenance, ordered-map comparisons, and scalar-path work.
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
