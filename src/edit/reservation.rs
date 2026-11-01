//! Checked allocation plans and ordered content writes for an editing session.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::{FileSpacePageSize, PhysicalAddress};
use hdf5_pure_space::__private::{
    Extent, PageTransition, PageType, ReservationBatch, ReservationRequest, ReservationWork,
    ReusableReservation,
};

use crate::address::{AddressContext, StoredAddress};
use crate::convert::Narrow;
use crate::error::{Error, FormatError};

use super::aggregation::AggregateSpace;
use super::block::{BlockRequest, EncodingDependency, PlannedBlock};
use super::commit::PublicationState;
use super::{Placement, WriteEngine};

/// Allocations reserved before emission, with a private end-of-allocation cursor.
///
/// The engine remains borrowed while the plan is used. Complete ranges, PAGE transitions, and
/// aggregate capacity are retained for writes in request order through [`Self::write_next`].
/// Dropping the plan keeps reused allocations unavailable. Publication and rollback remain the
/// caller's responsibility.
pub(super) struct ReservationPlan<'a> {
    engine: &'a mut WriteEngine,
    reservations: Reservations,
    next: usize,
    failed: bool,
    end: PhysicalAddress,
    unresolved: Option<ReusableReservation>,
    work: ReservationWork,
}

impl<'a> ReservationPlan<'a> {
    fn new(
        engine: &'a mut WriteEngine,
        requests: impl ExactSizeIterator<Item = BlockRequest>,
        mode: ReservationMode,
    ) -> Result<Self, Box<ReservationFailure<'a>>> {
        let mut plan = Self {
            end: PhysicalAddress::new(engine.image.len()),
            engine,
            reservations: if requests.len() <= 1 {
                Reservations::Scalar(None)
            } else {
                Reservations::Batch(Vec::with_capacity(requests.len()))
            },
            next: 0,
            failed: false,
            unresolved: None,
            work: ReservationWork::default(),
        };
        match plan.build(requests, mode) {
            Ok(()) => Ok(plan),
            Err(error) => Err(Box::new(ReservationFailure { plan, error })),
        }
    }

    /// Writes the next allocation at its planned physical range.
    ///
    /// The byte length must match the planned encoding length. Allocation capacity beyond that
    /// encoding remains owned. A length mismatch permits retrying the same allocation. A failed
    /// padding or content write prevents further writes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain,
    /// [`FormatError::SerializationError`] if the lengths differ, or [`FormatError::Internal`] if
    /// no allocation remains or a prior write failed. Returns address-conversion, padding, and I/O
    /// errors encountered during placement.
    pub(super) fn write_next(&mut self, bytes: &[u8]) -> Result<PhysicalAddress, Error> {
        let Self {
            engine,
            reservations,
            next,
            failed,
            end: _,
            unresolved: _,
            work: _,
        } = self;
        engine.context.allocations.available()?;
        if *failed {
            return Err(FormatError::Internal(
                "reservation plan has an earlier failed write".into(),
            )
            .into());
        }
        let reservation = reservations.as_slice().get(*next).ok_or_else(|| {
            FormatError::Internal("reservation plan has no pending allocation".into())
        })?;
        let region = reservation.block.region(bytes)?;
        let addr = region.range().start().get();
        let len = region.range().length();
        let placement = match reservation.placement {
            Placement::Aggregated { end, .. } => Placement::Aggregated { addr, len, end },
            Placement::Appended { .. } => Placement::Appended { addr, len },
            Placement::Reused { .. } => Placement::Reused { addr, len },
        };
        let result = Self::prepare_transition(
            engine,
            reservation.request().page_type(),
            reservation.transition,
        )
        .and_then(|()| engine.place(placement, region.bytes()))
        .and_then(|address| {
            if engine.image.len() < reservation.range().end().get() {
                engine.pad_zeros_to(reservation.range().end().get())?;
            }
            Ok(address)
        });
        match result {
            Ok(address) => {
                *next += 1;
                Ok(PhysicalAddress::new(address))
            }
            Err(error) => {
                *failed = true;
                Err(error)
            }
        }
    }

    /// Completes an emitted plan and returns its indexed-work counts.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a write failed or any allocation remains unresolved or
    /// unwritten. Returns [`FormatError::TruncatedFile`] if the image excludes planned capacity.
    pub(super) fn finish(self) -> Result<ReservationWork, Error> {
        let Self {
            engine,
            reservations,
            next,
            failed,
            end,
            unresolved,
            work,
        } = self;
        if failed || next != reservations.as_slice().len() || unresolved.is_some() {
            return Err(
                FormatError::Internal("reservation plan has unwritten allocations".into()).into(),
            );
        }
        if end.get() > engine.image.len() {
            return Err(FormatError::TruncatedFile {
                expected_eof: end.get(),
                actual_len: engine.image.len(),
            }
            .into());
        }
        Ok(work)
    }

    /// Returns the successfully planned allocations in request order.
    pub(super) fn reservations(&self) -> &[PlannedReservation] {
        self.reservations.as_slice()
    }

    #[cfg(test)]
    pub(super) fn end(&self) -> PhysicalAddress {
        self.end
    }

    #[cfg(test)]
    pub(super) fn work(&self) -> ReservationWork {
        self.work
    }

    #[cfg(test)]
    pub(super) fn unresolved(&self) -> Option<&ReusableReservation> {
        self.unresolved.as_ref()
    }

    fn into_scalar(self) -> Result<Placement, Error> {
        let reservation = self
            .reservations()
            .first()
            .ok_or_else(|| FormatError::Internal("scalar reservation is absent".into()))?;
        let placement = reservation.placement;
        let transition = reservation.transition;
        let page_type = reservation.request().page_type();
        Self::prepare_transition(self.engine, page_type, transition)?;
        Ok(placement)
    }

    fn prepare_transition(
        engine: &mut WriteEngine,
        page_type: PageType,
        transition: Option<PageTransition>,
    ) -> Result<(), Error> {
        if let Some(transition) = transition {
            if let PageTransition::Pad { extent, .. } = transition {
                let start = engine.context.addresses().resolve(extent.start())?.get();
                if engine.image.len() != start {
                    return Err(FormatError::Internal(
                        "planned PAGE padding does not begin at the image end".into(),
                    )
                    .into());
                }
                let length = extent.len().to_usize()?;
                #[cfg(test)]
                super::observations::record_zeros(length, length);
                engine.append(&vec![0; length])?;
            }
            engine.space.record_transition(page_type, transition);
        }
        Ok(())
    }

    fn build(
        &mut self,
        mut requests: impl ExactSizeIterator<Item = BlockRequest>,
        mode: ReservationMode,
    ) -> Result<(), Error> {
        let Self {
            engine,
            reservations,
            next: _,
            failed: _,
            end,
            unresolved,
            work,
        } = self;
        engine.context.allocations.available()?;
        if engine.publication == PublicationState::Uncertain {
            return Err(Error::AllocationPublicationUncertain);
        }
        let addresses = engine.context.addresses();
        let mut cursor = addresses.relative(*end)?;
        let reuse = engine.fresh_page_boundary.is_none()
            && match mode {
                ReservationMode::Immediate => engine.immediate_reuse_allowed(),
                ReservationMode::Staged => true,
            };
        let aggregate = match mode {
            ReservationMode::Immediate => engine.persist.is_none(),
            ReservationMode::Staged => true,
        };
        let mut aggregation = engine.aggregation.filter(|_| aggregate);
        if let Some(aggregation) = aggregation {
            cursor = cursor.max(aggregation.eoa());
            *end = addresses.resolve(cursor)?;
        }
        let mut page = AppendCursor {
            addresses,
            cursor,
            page_size: engine.space.page_size(),
            tail_type: engine.space.paged_tail_type().flatten(),
            fresh_boundary: engine.fresh_page_boundary,
        };
        let mut batch = if reuse && aggregation.is_some() {
            engine.space.reservation_batch_with_returns(requests.len())
        } else {
            engine
                .space
                .reservation_batch(if reuse { requests.len() } else { 0 })
        };
        let result = requests.try_for_each(|request| {
            let planned = Self::reserve_one(
                &mut batch,
                &mut aggregation,
                &mut page,
                request,
                reuse,
                unresolved,
            )?;
            *end = addresses.resolve(page.cursor)?;
            reservations.push(planned);
            Ok(())
        });
        *work = batch.finish();
        if aggregate {
            engine.aggregation = aggregation;
        }
        result
    }

    fn reserve_one(
        batch: &mut ReservationBatch<'_>,
        aggregation: &mut Option<AggregateSpace>,
        page: &mut AppendCursor,
        block_request: BlockRequest,
        reuse: bool,
        unresolved: &mut Option<ReusableReservation>,
    ) -> Result<PlannedReservation, Error> {
        let request = block_request.reservation();
        if reuse && let Some(reservation) = batch.reserve(request) {
            let extent = reservation.extent();
            let range = match page.physical_range(extent.start(), extent.len()) {
                Ok(range) => range,
                Err(error) => {
                    *unresolved = Some(reservation);
                    return Err(error.into());
                }
            };
            return Ok(PlannedReservation {
                placement: Placement::Reused {
                    addr: range.start().get(),
                    len: range.length(),
                },
                block: PlannedBlock::new(range, block_request)?,
                transition: None,
            });
        }
        if let Some(aggregation) = aggregation {
            let placement = aggregation.reserve_with(
                page.addresses,
                request.size(),
                request.class(),
                |eoa, extent| batch.release_aggregate(eoa, extent),
            )?;
            let range = FileRange::new(PhysicalAddress::new(placement.address()), placement.len())?;
            page.cursor = aggregation.eoa();
            return Ok(PlannedReservation {
                placement,
                block: PlannedBlock::new(range, block_request)?,
                transition: None,
            });
        }
        let (range, transition) = page.reserve(request)?;
        Ok(PlannedReservation {
            placement: Placement::Appended {
                addr: range.start().get(),
                len: range.length(),
            },
            block: PlannedBlock::new(range, block_request)?,
            transition,
        })
    }
}

/// A complete physical allocation, its request, and any preceding PAGE transition.
pub(super) struct PlannedReservation {
    placement: Placement,
    block: PlannedBlock,
    transition: Option<PageTransition>,
}

impl PlannedReservation {
    pub(super) fn range(&self) -> FileRange {
        self.block.allocation()
    }

    pub(super) fn request(&self) -> ReservationRequest {
        self.block.request().reservation()
    }
    pub(super) fn encoding_dependency(&self) -> EncodingDependency {
        self.block.request().dependency()
    }
}

/// A reservation error with its successful prefix and any unresolved reusable allocation.
///
/// The caller retains failure ownership through the partial plan. Dropping it keeps reused
/// allocations unavailable until a safe rollback or explicit return.
pub(super) struct ReservationFailure<'a> {
    pub(super) plan: ReservationPlan<'a>,
    pub(super) error: Error,
}

impl ReservationFailure<'_> {
    /// Drops the partial plan and returns its error.
    pub(super) fn into_error(self: Box<Self>) -> Error {
        let Self { plan, error } = *self;
        drop(plan);
        error
    }
}

enum Reservations {
    Batch(Vec<PlannedReservation>),
    Scalar(Option<PlannedReservation>),
}

impl Reservations {
    fn push(&mut self, reservation: PlannedReservation) {
        match self {
            Self::Batch(reservations) => reservations.push(reservation),
            Self::Scalar(slot) => {
                assert!(slot.is_none(), "a scalar plan has one request");
                *slot = Some(reservation);
            }
        }
    }

    fn as_slice(&self) -> &[PlannedReservation] {
        match self {
            Self::Batch(reservations) => reservations,
            Self::Scalar(reservation) => reservation.as_slice(),
        }
    }
}

/// A provisional stored EOA with PAGE alignment and tail-type state.
struct AppendCursor {
    addresses: AddressContext,
    cursor: StoredAddress,
    page_size: Option<FileSpacePageSize>,
    tail_type: Option<PageType>,
    fresh_boundary: Option<StoredAddress>,
}

impl AppendCursor {
    /// Advances EOA past a complete allocation and any PAGE alignment gap.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if stored or physical range arithmetic overflows, or
    /// [`FormatError::Internal`] if placement precedes the fresh-page boundary.
    fn reserve(
        &mut self,
        request: ReservationRequest,
    ) -> Result<(FileRange, Option<PageTransition>), FormatError> {
        let Self {
            addresses,
            cursor,
            page_size,
            tail_type,
            fresh_boundary,
        } = self;
        let mut start = *cursor;
        let transition = if let Some(page_size) = *page_size {
            let page_size = page_size.get();
            let transition =
                if start.get() % page_size == 0 || *tail_type == Some(request.page_type()) {
                    PageTransition::Unchanged
                } else {
                    let length = page_size - start.get() % page_size;
                    let extent = Extent::new(start, length).ok_or(FormatError::OffsetOverflow {
                        offset: start.get(),
                        length,
                    })?;
                    start = extent.end();
                    addresses.resolve(start)?;
                    PageTransition::Pad {
                        extent,
                        track_as: *tail_type,
                    }
                };
            Some(transition)
        } else {
            None
        };
        if fresh_boundary.is_some_and(|boundary| start < boundary) {
            return Err(FormatError::Internal(
                "PAGE allocation precedes its fresh-page boundary".into(),
            ));
        }
        let end = start.checked_add(request.size().get())?;
        let range = FileRange::new(addresses.resolve(start)?, request.size().get())?;
        addresses.resolve(end)?;
        *cursor = end;
        if page_size.is_some() {
            *tail_type = Some(request.page_type());
        }
        Ok((range, transition))
    }

    fn physical_range(
        &self,
        address: StoredAddress,
        length: u64,
    ) -> Result<FileRange, FormatError> {
        FileRange::new(self.addresses.resolve(address)?, length)
    }
}

#[derive(Clone, Copy)]
enum ReservationMode {
    Immediate,
    Staged,
}

impl WriteEngine {
    /// Reserves staged allocations with checked physical ranges before writing content.
    ///
    /// # Errors
    ///
    /// Returns a [`ReservationFailure`] with the successful prefix if a request fails. Its error is
    /// [`Error::AllocationPublicationUncertain`] after uncertain publication, or [`Error::Format`]
    /// for invalid aggregate, PAGE, or address geometry.
    pub(super) fn plan_reservations(
        &mut self,
        requests: &[ReservationRequest],
    ) -> Result<ReservationPlan<'_>, Box<ReservationFailure<'_>>> {
        ReservationPlan::new(
            self,
            requests
                .iter()
                .copied()
                .map(|request| BlockRequest::complete(request, EncodingDependency::Allocation)),
            ReservationMode::Staged,
        )
    }

    /// Reserves complete block capacity before writing each block's encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns the reservation failure described by [`Self::plan_reservations`].
    pub(super) fn plan_blocks(
        &mut self,
        requests: &[BlockRequest],
    ) -> Result<ReservationPlan<'_>, Box<ReservationFailure<'_>>> {
        ReservationPlan::new(self, requests.iter().copied(), ReservationMode::Staged)
    }

    /// Reserves one allocation under the session's immediate-write reuse policy.
    ///
    /// # Errors
    ///
    /// Returns the reservation failure described by [`Self::plan_reservations`].
    pub(super) fn plan_immediate_reservation(
        &mut self,
        request: ReservationRequest,
    ) -> Result<ReservationPlan<'_>, Box<ReservationFailure<'_>>> {
        ReservationPlan::new(
            self,
            core::iter::once(BlockRequest::complete(
                request,
                EncodingDependency::Allocation,
            )),
            ReservationMode::Immediate,
        )
    }

    /// Reserves one staged allocation and writes any preceding PAGE padding.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] after uncertain publication,
    /// [`Error::Format`] for invalid placement geometry, or the padding write error.
    pub(super) fn reserve_request(
        &mut self,
        request: ReservationRequest,
    ) -> Result<Placement, Error> {
        self.plan_reservations(core::slice::from_ref(&request))
            .map_err(ReservationFailure::into_error)?
            .into_scalar()
    }
}

#[cfg(test)]
mod tests;
