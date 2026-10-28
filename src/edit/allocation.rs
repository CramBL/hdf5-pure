//! Allocation publication state and reference permissions for an editing session.

use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;

use crate::error::{Error, FormatError};

/// The allocation publication state of one editing session.
///
/// Allocations in a batch share a generation. The storage layer confirms that generation after
/// its writes and publication barriers succeed. An uncertain outcome prevents further publication.
pub(crate) struct AllocationPublication {
    session: Arc<SessionIdentity>,
    state: State,
}

impl AllocationPublication {
    /// Creates a session with no pending allocations.
    pub(crate) fn new() -> Self {
        Self {
            session: Arc::new(SessionIdentity),
            state: State::Confirmed(0),
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
                let generation = confirmed.checked_add(1).ok_or_else(|| {
                    FormatError::Internal("allocation publication generation exceeds u64".into())
                })?;
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

fn mismatched_publication() -> Error {
    FormatError::Internal("allocation belongs to another session or publication".into()).into()
}

#[cfg(test)]
mod tests;
