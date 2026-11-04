//! Checked allocation plans and ordered content writes for an editing session.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::{FileSpacePageSize, PhysicalAddress};
use hdf5_pure_space::__private::{
    Extent, PageTransition, PageType, ReservationBatch, ReservationPlacement, ReservationRequest,
    ReservationWork, ReusableReservation,
};

use crate::address::{AddressContext, StoredAddress};
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
    /// Plans requests under the selected staged or immediate allocation policy.
    ///
    /// # Errors
    ///
    /// Returns [`ReservationFailure`] with the partial plan if publication or placement fails.
    fn new(
        engine: &'a mut WriteEngine,
        requests: impl ExactSizeIterator<Item = BlockRequest>,
        mode: ReservationMode,
    ) -> Result<Self, Box<ReservationFailure<'a>>> {
        let mut plan = Self::empty(engine);
        plan.reservations.reserve(requests.len());
        match plan.build(requests, mode) {
            Ok(()) => Ok(plan),
            Err(error) => Err(Box::new(ReservationFailure { plan, error })),
        }
    }

    /// Plans a preferred reusable allocation or fallback requests before a common suffix.
    ///
    /// # Errors
    ///
    /// Returns the failure described by [`WriteEngine::plan_blocks_with_fallback`].
    fn with_preferred(
        engine: &'a mut WriteEngine,
        preferred: BlockRequest,
        suffix: &[BlockRequest],
        fallback: impl FnOnce() -> Result<Vec<BlockRequest>, FormatError>,
    ) -> Result<(ReservationChoice, Self), Box<ReservationFailure<'a>>> {
        let mut plan = Self::empty(engine);
        let result = (|| {
            plan.engine.context.allocations.available()?;
            if plan.engine.publication == PublicationState::Uncertain {
                return Err(Error::AllocationPublicationUncertain);
            }
            if plan.engine.fresh_page_boundary.is_none() {
                let addresses = plan.engine.context.addresses();
                let mut batch = plan.engine.space.reservation_batch(1);
                let reservation = batch.reserve(preferred.reservation());
                batch.finish();
                if let Some(reservation) = reservation {
                    plan.reservations.push(Self::reused(
                        addresses,
                        reservation,
                        preferred,
                        &mut plan.unresolved,
                    )?);
                    plan.reservations.reserve(suffix.len());
                    if !suffix.is_empty() {
                        plan.build(suffix.iter().copied(), ReservationMode::Staged)?;
                    }
                    return Ok(ReservationChoice::Preferred);
                }
            }
            let mut requests = fallback()?;
            requests.extend_from_slice(suffix);
            plan.reservations.reserve(requests.len());
            plan.build(requests.into_iter(), ReservationMode::Staged)?;
            Ok(ReservationChoice::Fallback)
        })();
        match result {
            Ok(choice) => Ok((choice, plan)),
            Err(error) => Err(Box::new(ReservationFailure { plan, error })),
        }
    }

    /// Starts an empty inline plan at the image's current physical end.
    fn empty(engine: &'a mut WriteEngine) -> Self {
        Self {
            end: PhysicalAddress::new(engine.image.len()),
            engine,
            reservations: Reservations::Scalar(None),
            next: 0,
            failed: false,
            unresolved: None,
            work: ReservationWork::default(),
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
        self.next_reservation()?.block.region(bytes)?;
        self.write_next_with(|writer| writer.write(bytes))
    }

    /// Writes the next encoding from slices supplied by a producer.
    ///
    /// The producer must write exactly the planned encoding length through
    /// [`ReservedEncoding::write`]. The caller retains complete allocation capacity on success
    /// and failure. After a producer, padding, or content error, further writes through the plan
    /// return an error. Catching a writer error in the producer also prevents completion.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain,
    /// [`FormatError::SerializationError`] if the emitted length differs from the planned length,
    /// or [`FormatError::Internal`] if no allocation remains or a prior write failed. Returns
    /// errors from the producer, padding, or content writes.
    pub(super) fn write_next_with(
        &mut self,
        emit: impl FnOnce(&mut ReservedEncoding<'_>) -> Result<(), Error>,
    ) -> Result<PhysicalAddress, Error> {
        self.next_reservation()?;
        let Self {
            engine,
            reservations,
            next,
            failed,
            end: _,
            unresolved: _,
            work: _,
        } = self;
        let reservation = &reservations.as_slice()[*next];
        let PlannedReservation {
            placement: _,
            block,
            transition,
        } = reservation;
        let result = Self::prepare_transition(
            engine,
            block.request().reservation().page_type(),
            *transition,
        )
        .and_then(|()| {
            let mut writer = ReservedEncoding {
                cursor: block.encoding().start().get(),
                engine,
                reservation,
                failed: false,
            };
            emit(&mut writer)?;
            writer.finish()
        });
        match result {
            Ok(address) => {
                *next += 1;
                Ok(address)
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

    /// Returns the reservations after the successfully emitted prefix, in request order.
    pub(super) fn pending(&self) -> &[PlannedReservation] {
        &self.reservations.as_slice()[self.next..]
    }

    /// Returns the address context of the borrowed editing session.
    pub(super) fn addresses(&self) -> AddressContext {
        self.engine.context.addresses()
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

    /// Borrows the next reservation after checking publication and write state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain,
    /// or [`FormatError::Internal`] if no allocation remains or a prior write failed.
    fn next_reservation(&self) -> Result<&PlannedReservation, Error> {
        self.engine.context.allocations.available()?;
        if self.failed {
            return Err(FormatError::Internal(
                "reservation plan has an earlier failed write".into(),
            )
            .into());
        }
        self.reservations.as_slice().get(self.next).ok_or_else(|| {
            FormatError::Internal("reservation plan has no pending allocation".into()).into()
        })
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
                engine.pad_page_extent(extent)?;
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
            return Self::reused(page.addresses, reservation, block_request, unresolved);
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
                block: PlannedBlock::placed(
                    page.addresses,
                    request.place_after(page.addresses.relative(range.start())?, None)?,
                    block_request,
                )?,
                transition: None,
            });
        }
        let (range, placement, transition) = page.reserve(request)?;
        Ok(PlannedReservation {
            placement: Placement::Appended {
                addr: range.start().get(),
                len: range.length(),
            },
            block: PlannedBlock::placed(page.addresses, placement, block_request)?,
            transition,
        })
    }

    /// Associates a reusable allocation with a checked physical block.
    ///
    /// Moves the reservation into `unresolved` if block construction fails.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with the error from [`PlannedBlock::placed`].
    fn reused(
        addresses: AddressContext,
        reservation: ReusableReservation,
        request: BlockRequest,
        unresolved: &mut Option<ReusableReservation>,
    ) -> Result<PlannedReservation, Error> {
        let block = match PlannedBlock::placed(addresses, reservation.placement(), request) {
            Ok(block) => block,
            Err(error) => {
                *unresolved = Some(reservation);
                return Err(error.into());
            }
        };
        Ok(PlannedReservation {
            placement: Placement::Reused {
                addr: block.allocation().start().get(),
                len: block.allocation().length(),
            },
            block,
            transition: None,
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
    /// Returns the allocation request, encoded length, and address dependency.
    pub(super) fn block_request(&self) -> BlockRequest {
        self.block.request()
    }
    pub(super) fn encoding_dependency(&self) -> EncodingDependency {
        self.block.request().dependency()
    }
}

/// A borrowed writer limited to one reservation's encoding range.
///
/// The producer writes consecutive slices through [`Self::write`]. After a failed write, further
/// writes and completion return an error, including when the producer catches the write error.
pub(super) struct ReservedEncoding<'a> {
    engine: &'a mut WriteEngine,
    reservation: &'a PlannedReservation,
    cursor: u64,
    failed: bool,
}

impl ReservedEncoding<'_> {
    /// Writes an entire slice at the current encoding position.
    ///
    /// After an error, further writes and completion return an error. A partial I/O write can
    /// change source bytes before failing.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SerializationError`] if the slice exceeds the remaining encoding
    /// length, or [`FormatError::Internal`] if a prior write failed. Returns errors encountered
    /// during the content write.
    pub(super) fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let Self {
            engine,
            reservation,
            cursor,
            failed,
        } = self;
        if *failed {
            return Err(FormatError::Internal(
                "reserved encoding has an earlier failed write".into(),
            )
            .into());
        }
        let PlannedReservation {
            placement,
            block,
            transition: _,
        } = *reservation;
        let encoding = block.encoding();
        let length = bytes.len() as u64;
        if length > encoding.end().get() - *cursor {
            *failed = true;
            return Err(FormatError::SerializationError(format!(
                "a placement reserved {} bytes but built {}",
                encoding.length(),
                u128::from(*cursor - encoding.start().get()) + u128::from(length),
            ))
            .into());
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let end = *cursor + length;
        let placement = match *placement {
            Placement::Aggregated { .. } => Placement::Aggregated {
                addr: *cursor,
                len: length,
                end,
            },
            Placement::Appended { .. } => Placement::Appended {
                addr: *cursor,
                len: length,
            },
            Placement::Reused { .. } => Placement::Reused {
                addr: *cursor,
                len: length,
            },
        };
        match engine.place(placement, bytes) {
            Ok(_) => {
                *cursor = end;
                Ok(())
            }
            Err(error) => {
                *failed = true;
                Err(error)
            }
        }
    }

    fn finish(self) -> Result<PhysicalAddress, Error> {
        let Self {
            engine,
            reservation,
            cursor,
            failed,
        } = self;
        if failed {
            return Err(FormatError::Internal(
                "reserved encoding has an earlier failed write".into(),
            )
            .into());
        }
        let PlannedReservation {
            placement,
            block,
            transition: _,
        } = reservation;
        let encoding = block.encoding();
        if cursor != encoding.end().get() {
            return Err(FormatError::SerializationError(format!(
                "a placement reserved {} bytes but built {}",
                encoding.length(),
                cursor - encoding.start().get(),
            ))
            .into());
        }
        if let Placement::Aggregated { end, .. } = *placement {
            AggregateSpace::place(engine.image.as_mut(), cursor, &[], end)?;
        }
        if engine.image.len() < block.allocation().end().get() {
            engine.pad_zeros_to(block.allocation().end().get())?;
        }
        Ok(encoding.start())
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
    /// Reserves additional slots, keeping at most one reservation inline.
    fn reserve(&mut self, additional: usize) {
        match self {
            Self::Batch(reservations) => reservations.reserve(additional),
            Self::Scalar(slot) if usize::from(slot.is_some()) + additional > 1 => {
                let mut reservations = Vec::with_capacity(usize::from(slot.is_some()) + additional);
                reservations.extend(slot.take());
                *self = Self::Batch(reservations);
            }
            Self::Scalar(_) => {}
        }
    }

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
    /// Applies [`ReservationRequest::for_block`] geometry to block requests before resolving the
    /// physical range. Preserves the cursor and tail page type on error.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if stored or physical range arithmetic overflows, or
    /// [`FormatError::Internal`] if placement precedes the fresh-page boundary.
    fn reserve(
        &mut self,
        request: ReservationRequest,
    ) -> Result<(FileRange, ReservationPlacement, Option<PageTransition>), FormatError> {
        let Self {
            addresses,
            cursor,
            page_size,
            tail_type,
            fresh_boundary,
        } = self;
        let start = if let Some(page_size) = *page_size
            && cursor.get() % page_size.get() != 0
            && *tail_type != Some(request.page_type())
        {
            cursor.checked_add(page_size.get() - cursor.get() % page_size.get())?
        } else {
            *cursor
        };
        let placement = request.place_after(start, *page_size)?;
        let start = placement.extent().start();
        let transition = page_size.map(|_| match Extent::between(*cursor, start) {
            Some(extent) => PageTransition::Pad {
                extent,
                track_as: *tail_type,
            },
            None => PageTransition::Unchanged,
        });
        if fresh_boundary.is_some_and(|boundary| start < boundary) {
            return Err(FormatError::Internal(
                "PAGE allocation precedes its fresh-page boundary".into(),
            ));
        }
        let end = placement.extent().end();
        let range = FileRange::new(addresses.resolve(start)?, request.size().get())?;
        addresses.resolve(end)?;
        *cursor = end;
        if page_size.is_some() {
            *tail_type = Some(request.page_type());
        }
        Ok((range, placement, transition))
    }
}

#[derive(Clone, Copy)]
enum ReservationMode {
    Immediate,
    Staged,
}

/// The request prefix selected before a common suffix.
#[derive(Debug, PartialEq)]
pub(super) enum ReservationChoice {
    /// The requests returned by the fallback constructor.
    Fallback,
    /// One allocation reserved from reusable space for the preferred request.
    Preferred,
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

    /// Plans a reusable preferred request or fallback requests before a common suffix.
    ///
    /// Checks publication availability, then tries `preferred` in reusable space when the session
    /// permits reuse. Evaluates `fallback` once if the preferred request remains unreserved. The
    /// selected requests precede `suffix`, with complete capacity reserved before content emission.
    ///
    /// The caller retains allocation ownership on success and failure, including unresolved
    /// reusable allocations, and handles rollback and publication within the enclosing commit.
    ///
    /// # Performance
    ///
    /// The preferred lookup uses scalar allocation. Batch lookup state is constructed at most
    /// once for the selected remaining requests. A preferred allocation with an empty suffix stays
    /// inline. Indexed-work counts exclude scalar searches.
    ///
    /// # Errors
    ///
    /// Returns [`ReservationFailure`] with the partial plan on publication, fallback preparation,
    /// or placement errors. Its error is [`Error::AllocationPublicationUncertain`] after uncertain
    /// publication or [`Error::Format`] for invalid requests or placement geometry.
    pub(super) fn plan_blocks_with_fallback(
        &mut self,
        preferred: BlockRequest,
        suffix: &[BlockRequest],
        fallback: impl FnOnce() -> Result<Vec<BlockRequest>, FormatError>,
    ) -> Result<(ReservationChoice, ReservationPlan<'_>), Box<ReservationFailure<'_>>> {
        ReservationPlan::with_preferred(self, preferred, suffix, fallback)
    }

    /// Reserves complete blocks under the session's immediate-write reuse policy.
    ///
    /// # Errors
    ///
    /// Returns the reservation failure described by [`Self::plan_reservations`].
    pub(super) fn plan_immediate_blocks(
        &mut self,
        requests: &[BlockRequest],
    ) -> Result<ReservationPlan<'_>, Box<ReservationFailure<'_>>> {
        ReservationPlan::new(self, requests.iter().copied(), ReservationMode::Immediate)
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
