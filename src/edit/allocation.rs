//! Allocation publication, allocator rollback, and reference permissions for an editing session.

use std::sync::Arc;

use hdf5_pure_core::__private::{FileRange, PhysicalAddress};
use hdf5_pure_space::__private::{FreeClass, SessionSpaceSnapshot};

use super::aggregation::AggregateSpace;
use super::commit::{PublicationPurpose, PublicationState};
use super::object_header;
use super::{SyncPolicy, TailPlacement, WriteEngine};
use crate::address::{StoredAddress, StoredAddressExt};
use crate::free_sections::PersistedManagerSet;
use crate::source::Source;

use crate::error::{Error, FormatError};

/// The allocation publication state of one editing session.
///
/// Allocations in a batch share a generation. The storage layer confirms that generation after
/// its writes and publication barriers succeed. An uncertain outcome prevents further publication.
pub(crate) struct AllocationPublication {
    session: Arc<SessionIdentity>,
    state: State,
    sequence: u64,
}

impl AllocationPublication {
    /// Creates a session with no pending allocations.
    pub(crate) fn new() -> Self {
        Self {
            session: Arc::new(SessionIdentity),
            state: State::Confirmed(0),
            sequence: 0,
        }
    }

    /// Records `range` in the pending batch without granting reference permission.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if the session has an uncertain outcome,
    /// or [`FormatError::Internal`] if the publication generation exceeds [`u64::MAX`].
    pub(crate) fn provision(&mut self, range: FileRange) -> Result<ProvisionalAllocation, Error> {
        let generation = match self.state {
            State::Confirmed(confirmed) => {
                let generation = self.sequence.checked_add(1).ok_or_else(|| {
                    FormatError::Internal("allocation publication generation exceeds u64".into())
                })?;
                self.sequence = generation;
                self.state = State::Provisional {
                    confirmed,
                    generation,
                };
                generation
            }
            State::Provisional { generation, .. } => generation,
            State::Uncertain { .. } => return Err(Error::AllocationPublicationUncertain),
        };
        Ok(ProvisionalAllocation {
            session: Arc::clone(&self.session),
            generation,
            range,
        })
    }

    /// Starts publication of pending allocations, or returns the current confirmation.
    ///
    /// The pending generation remains uncertain until [`complete`](Self::complete) succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    pub(crate) fn begin(&mut self) -> Result<PublicationAction, Error> {
        match self.state {
            State::Confirmed(generation) => Ok(PublicationAction::Current(ConfirmedPublication {
                session: Arc::clone(&self.session),
                generation,
            })),
            State::Provisional {
                confirmed,
                generation,
            } => {
                self.state = State::Uncertain {
                    confirmed,
                    generation,
                };
                Ok(PublicationAction::Publish(PublicationAttempt {
                    session: Arc::clone(&self.session),
                    generation,
                }))
            }
            State::Uncertain { .. } => Err(Error::AllocationPublicationUncertain),
        }
    }

    /// Confirms an attempt after the storage layer completes its publication barriers.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `attempt` belongs to another session or generation,
    /// or this session has no corresponding uncertain attempt.
    pub(crate) fn complete(
        &mut self,
        attempt: PublicationAttempt,
    ) -> Result<ConfirmedPublication, Error> {
        if !Arc::ptr_eq(&self.session, &attempt.session)
            || !matches!(self.state, State::Uncertain { generation, .. } if generation == attempt.generation)
        {
            return Err(mismatched_publication());
        }
        self.state = State::Confirmed(attempt.generation);
        Ok(ConfirmedPublication {
            session: attempt.session,
            generation: attempt.generation,
        })
    }

    /// Restores the preceding confirmation after a publication that was not attempted on disk.
    ///
    /// The caller establishes that no superblock publication was attempted. The canceled
    /// generation is excluded from subsequent confirmations.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the attempt belongs to another session or generation,
    /// or the session has no corresponding uncertain attempt.
    fn cancel(&mut self, attempt: PublicationAttempt) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.session, &attempt.session) {
            return Err(mismatched_publication());
        }
        let State::Uncertain {
            confirmed,
            generation,
        } = self.state
        else {
            return Err(mismatched_publication());
        };
        if generation != attempt.generation {
            return Err(mismatched_publication());
        }
        self.state = State::Confirmed(confirmed);
        Ok(())
    }

    /// Discards an unpublished batch while retaining the preceding confirmation.
    ///
    /// The caller restores the allocator snapshot before discarding the batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if publication has an uncertain outcome.
    fn discard_provisional(&mut self) -> Result<(), Error> {
        match self.state {
            State::Confirmed(_) => Ok(()),
            State::Provisional { confirmed, .. } => {
                self.state = State::Confirmed(confirmed);
                Ok(())
            }
            State::Uncertain { .. } => Err(Error::AllocationPublicationUncertain),
        }
    }

    /// Prevents further publication after an unconfirmed write or synchronization outcome.
    pub(crate) fn make_uncertain(&mut self) {
        let (confirmed, generation) = match self.state {
            State::Confirmed(generation) => (generation, generation),
            State::Provisional {
                confirmed,
                generation,
            }
            | State::Uncertain {
                confirmed,
                generation,
            } => (confirmed, generation),
        };
        self.state = State::Uncertain {
            confirmed,
            generation,
        };
    }

    /// Requires a confirmed or provisional publication state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    pub(crate) fn available(&self) -> Result<(), Error> {
        match self.state {
            State::Uncertain { .. } => Err(Error::AllocationPublicationUncertain),
            State::Confirmed(_) | State::Provisional { .. } => Ok(()),
        }
    }

    /// Requires confirmation of every pending allocation before a reference write.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if allocations remain provisional, or
    /// [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    pub(crate) fn before_reference(&self) -> Result<(), Error> {
        match self.state {
            State::Confirmed(_) => Ok(()),
            State::Provisional { .. } => Err(FormatError::Internal(
                "reference publication precedes allocation ownership publication".into(),
            )
            .into()),
            State::Uncertain { .. } => Err(Error::AllocationPublicationUncertain),
        }
    }

    /// Returns the complete range of an allocation confirmed by this session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain, or
    /// [`FormatError::Internal`] if `allocation` belongs to another session or an unconfirmed
    /// generation.
    pub(crate) fn require(&self, allocation: &ConfirmedAllocation) -> Result<FileRange, Error> {
        self.available()?;
        let confirmed = match self.state {
            State::Confirmed(confirmed) | State::Provisional { confirmed, .. } => confirmed,
            State::Uncertain { .. } => return Err(Error::AllocationPublicationUncertain),
        };
        if !Arc::ptr_eq(&self.session, &allocation.session) || allocation.generation > confirmed {
            return Err(mismatched_publication());
        }
        Ok(allocation.range)
    }
}

/// The shared identity retained by a session and its allocation permissions.
struct SessionIdentity;

/// The confirmed generation and any pending publication outcome.
enum State {
    Confirmed(u64),
    Provisional { confirmed: u64, generation: u64 },
    Uncertain { confirmed: u64, generation: u64 },
}

/// A complete allocation range awaiting publication in its session and generation.
pub(crate) struct ProvisionalAllocation {
    session: Arc<SessionIdentity>,
    generation: u64,
    range: FileRange,
}

impl ProvisionalAllocation {
    /// Returns the allocated range without granting reference permission.
    pub(crate) fn range(&self) -> FileRange {
        self.range
    }
}

/// The storage action required before confirming a batch's allocations.
pub(crate) enum PublicationAction {
    /// The session has no provisional allocations to publish.
    Current(ConfirmedPublication),
    /// The storage layer must publish the pending batch and complete its barriers.
    Publish(PublicationAttempt),
}

/// An in-progress publication bound to its session and generation.
pub(crate) struct PublicationAttempt {
    session: Arc<SessionIdentity>,
    generation: u64,
}

/// Permission to confirm allocations from one successfully published generation.
pub(crate) struct ConfirmedPublication {
    session: Arc<SessionIdentity>,
    generation: u64,
}

impl ConfirmedPublication {
    /// Converts a provisional allocation from this batch into reference permission.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `allocation` belongs to another session or generation.
    pub(crate) fn confirm(
        &self,
        allocation: ProvisionalAllocation,
    ) -> Result<ConfirmedAllocation, Error> {
        if !Arc::ptr_eq(&self.session, &allocation.session)
            || allocation.generation != self.generation
        {
            return Err(mismatched_publication());
        }
        Ok(ConfirmedAllocation {
            session: allocation.session,
            generation: allocation.generation,
            range: allocation.range,
        })
    }
}

/// A complete allocation range confirmed by its editing session.
///
/// The caller uses [`AllocationPublication::require`] before publishing a reference to it.
pub(crate) struct ConfirmedAllocation {
    session: Arc<SessionIdentity>,
    generation: u64,
    range: FileRange,
}

/// Tracks allocator rollback and checked manager authority for immediate allocations.
pub(super) struct AppendAllocator {
    /// Allocator state preceding the current persistent allocation batch.
    pending: Option<AppendAllocationSnapshot>,
    /// Whether persisted manager ownership has been checked in this session.
    authority: ManagerAuthority,
}

impl AppendAllocator {
    /// Creates allocation state without a pending batch or checked manager authority.
    pub(super) fn new() -> Self {
        Self {
            pending: None,
            authority: ManagerAuthority::Unchecked,
        }
    }
}

/// Whether persisted managers can authorize session allocations.
enum ManagerAuthority {
    Checked,
    Unchecked,
}

/// Allocator and persistence state preceding a provisional persistent batch.
struct AppendAllocationSnapshot {
    space: SessionSpaceSnapshot,
    aggregation: Option<AggregateSpace>,
    fsm_dirty: bool,
}

impl WriteEngine {
    /// Checks persisted authority and saves allocator state before a provisional batch.
    ///
    /// Drains earlier pending writes before reusing retired storage. In persistent batches,
    /// releases unused aggregate spare through the session free-space policy and allocates without
    /// aggregate over-reservation.
    ///
    /// # Errors
    ///
    /// Returns metadata read, parse, buffered-write, or address-conversion errors. A failed drain
    /// makes allocation publication uncertain.
    pub(super) fn prepare_append_allocation(&mut self) -> Result<(), Error> {
        if self.append_allocator.pending.is_some() {
            return Ok(());
        }
        if self.persist.is_some()
            && matches!(self.append_allocator.authority, ManagerAuthority::Unchecked)
        {
            self.append_allocator.authority = self.read_append_manager_authority()?;
        }
        if matches!(self.context.allocations.state, State::Confirmed(_))
            && let Err(error) = self.image.flush()
        {
            self.context.allocations.make_uncertain();
            if let Some(aggregation) = &mut self.aggregation {
                aggregation.publication_confirmed = false;
            }
            return Err(error);
        }
        if self.persist.is_none() {
            return Ok(());
        }
        self.append_allocator.pending = Some(AppendAllocationSnapshot {
            space: self.space.allocation_snapshot(),
            aggregation: self.aggregation,
            fsm_dirty: self.fsm_dirty,
        });
        if let Some(aggregation) = self.aggregation {
            let eoa = self.end_of_allocation()?.get();
            for extent in aggregation.unused() {
                self.space.record_freed(eoa, extent, FreeClass::Dead);
            }
            self.aggregation = Some(AggregateSpace::new(eoa));
        }
        Ok(())
    }

    /// Restores a persistent batch's allocator state after a known unpublished failure.
    ///
    /// The caller establishes that superblock publication was not attempted. Without a saved
    /// persistent snapshot, marks the allocation state uncertain.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if the allocation state cannot discard
    /// its provisional batch because publication is uncertain.
    pub(super) fn restore_append_allocation(&mut self) -> Result<(), Error> {
        if let Some(snapshot) = self.append_allocator.pending.take() {
            self.space.restore_allocation(snapshot.space);
            self.aggregation = snapshot.aggregation;
            self.fsm_dirty = snapshot.fsm_dirty;
            self.context.allocations.discard_provisional()
        } else {
            self.context.allocations.make_uncertain();
            Ok(())
        }
    }

    /// Publishes pending allocation ownership before granting reference permission.
    ///
    /// In persistent sessions, replaces standard free-space managers while preserving the current
    /// root and staged edits. Excludes complete owned allocations and the managers' storage from
    /// free sections. Unissued free space remains recorded under the configured threshold policy.
    /// Without persistence, publishes any required EOF change.
    ///
    /// Completes the content and publication barriers before confirming the generation. A failure
    /// before attempted manager publication restores allocator state. After an uncertain outcome,
    /// rejects dependent reference and reuse requests.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    /// Propagates metadata-planning, address-conversion, write, synchronization, and truncation
    /// errors. Returns [`FormatError::Internal`] if the publication attempt has a different
    /// session or generation.
    pub(super) fn publish_append_allocations(&mut self) -> Result<ConfirmedPublication, Error> {
        let attempt = match self.context.allocations.begin()? {
            PublicationAction::Current(publication) => return Ok(publication),
            PublicationAction::Publish(attempt) => attempt,
        };
        if self.persist.is_some() {
            self.publication = PublicationState::Ready;
            let root = self
                .context
                .addresses()
                .resolve(self.context.superblock().root_group_address)?;
            let result = self.publish_persisting(
                root,
                Vec::new(),
                TailPlacement::Anywhere,
                PublicationPurpose::Allocation,
            );
            if let Err(error) = result {
                if !self.publication.attempted() {
                    self.context.allocations.cancel(attempt)?;
                    self.restore_append_allocation()?;
                }
                return Err(error);
            }
        } else {
            match self.sync_policy {
                SyncPolicy::Always => self.image.sync_data()?,
                SyncPolicy::OnClose => self.image.flush()?,
            }
            if self.image.len() != self.context.superblock().eof_address {
                let mut superblock = self.context.superblock().clone();
                superblock.eof_address = self.image.len();
                let publication = self.context.propose(superblock)?.encode()?;
                self.image
                    .write_at(self.context.signature().get(), publication.bytes())?;
                match self.sync_policy {
                    SyncPolicy::Always => self.image.sync_all()?,
                    SyncPolicy::OnClose => self.image.flush()?,
                }
                self.context.adopt(publication.into_proposed())?;
            }
        }
        self.append_allocator.pending = None;
        self.context.allocations.complete(attempt)
    }

    /// Reads persisted managers and rejects free sections overlapping their published storage.
    ///
    /// Checks complete header and section-list allocations, including unused section-list
    /// capacity, and the current extension chunks. Manager checksum verification applies
    /// independently of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if persistent allocation has no File Space Info message.
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if a free section overlaps published
    /// manager or extension storage. Propagates metadata read, parse, and address-range errors.
    fn read_append_manager_authority(&self) -> Result<ManagerAuthority, Error> {
        let (extension, info) = Self::extension_fsinfo(&self.image(), &self.context)?.ok_or(
            Error::EditUnsupported("persistent allocation requires File Space Info"),
        )?;
        let addresses = self.context.addresses();
        let eof = addresses.relative(self.context.superblock().eof_address)?;
        let (report, mut allocations) = PersistedManagerSet::read(
            &self.image(),
            self.context.widths(),
            addresses.effective_base(),
            eof.get(),
            &info,
        )?
        .into_parts();
        for chunk in object_header::read_oh_chunks(&self.image(), extension, addresses)? {
            allocations.push(FileRange::new(
                PhysicalAddress::new(chunk.span.0),
                chunk.span.1,
            )?);
        }
        for section in report.free_sections().sections() {
            let range = FileRange::new(
                addresses.resolve(StoredAddress::new(section.address()))?,
                section.length().get(),
            )?;
            if let Some(allocation) = allocations.iter().find(|allocation| {
                range.start() < allocation.end() && allocation.start() < range.end()
            }) {
                return Err(FormatError::InvalidFreeSpaceManager(format!(
                    "the free section at {}, {} bytes long, overlaps published manager or extension storage at {}, {} bytes long",
                    range.start().get(),
                    range.length(),
                    allocation.start().get(),
                    allocation.length()
                ))
                .into());
            }
        }
        Ok(ManagerAuthority::Checked)
    }
}

fn mismatched_publication() -> Error {
    FormatError::Internal("allocation belongs to another session or publication".into()).into()
}

#[cfg(test)]
mod tests;
