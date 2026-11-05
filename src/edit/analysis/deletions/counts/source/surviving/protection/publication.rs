//! Retirement exclusions and EOA bounds for original count allocations.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use crate::ObjectReferenceCountUpdateIssue;
use crate::edit::WriteEngine;
use crate::edit::retirement::PendingRootFrees;
use crate::error::Error;
use crate::object_graph::ObjectId;

use super::CountProtectionError;
use super::ProtectedCount;
use super::ProtectedCounts;

mod completion;
mod current;
mod prepare;
#[cfg(test)]
mod test_support;

impl ProtectedCounts {
    /// Checks pending retirements against complete original count allocations.
    ///
    /// Rechecks the recorded session, source root, publication availability, and allocator claims.
    /// Preserves the original source association and each retirement range and class. For an empty
    /// count batch, skips engine and retirement inspection.
    ///
    /// Source freshness, confirmed publication, and permission to write require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`CountPublicationError::Protection`] for an inapplicable session or root,
    /// unavailable publication, conflicting allocator claims, or address-conversion errors.
    /// Returns [`CountPublicationError::Retirement`] if a pending retirement intersects a
    /// protected allocation.
    ///
    /// # Performance
    ///
    /// Borrows allocator lists and uses the existing count-allocation index. Uses retained metadata
    /// without source reads or topology discovery.
    pub(in crate::edit) fn before_publication<'a>(
        self,
        engine: &'a mut WriteEngine,
        retirement: PendingRootFrees,
    ) -> Result<CountPublicationPhase<'a>, CountPublicationError> {
        if let Some(source) = &self.source {
            source
                .reconstructions(engine, [])
                .map_err(CountPublicationError::Protection)?;
            self.exclude_space(engine)
                .map_err(CountPublicationError::Protection)?;
            for &(range, class) in retirement.batch.entries() {
                if let Some(update) = self.overlapping(range) {
                    return Err(CountPublicationError::Retirement {
                        object: update.object(),
                        allocation: update.allocation().range(),
                        retired: range,
                        class,
                    });
                }
            }
        }
        Ok(CountPublicationPhase {
            engine,
            counts: self,
            retirement,
        })
    }
}

/// Protected count allocations checked against the engine and pending retirements.
///
/// The engine is exclusively borrowed, and the original retirement batch is retained. The caller
/// can propose another EOA while retaining those associations. Dropping the phase leaves
/// allocator state unchanged. Source freshness and write permission require separate evidence.
pub(in crate::edit) struct CountPublicationPhase<'a> {
    engine: &'a mut WriteEngine,
    counts: ProtectedCounts,
    retirement: PendingRootFrees,
}

impl<'a> CountPublicationPhase<'a> {
    /// Checks a physical EOA against every complete protected allocation.
    ///
    /// # Errors
    ///
    /// Returns [`CountEndFailure`] with [`CountPublicationError::Truncation`] if `end` precedes
    /// a protected allocation's end. The caller can recover the phase and propose another EOA.
    ///
    /// # Performance
    ///
    /// Checks the maximal allocation end in constant time. On success, retains the existing phase
    /// without allocating or repeating allocator and retirement checks. On failure, boxes the
    /// phase and error.
    pub(in crate::edit) fn at_end(
        self,
        end: PhysicalAddress,
    ) -> Result<CountPublicationBounds<'a>, Box<CountEndFailure<'a>>> {
        if let Err(error) = CountAllocationEnd::new(&self.counts, end) {
            return Err(Box::new(CountEndFailure { phase: self, error }));
        }
        Ok(CountPublicationBounds { phase: self, end })
    }
}

/// A physical EOA at or beyond every complete protected count allocation.
///
/// The engine borrow and original retirement batch are retained through [`CountPublicationPhase`].
/// The predecessor source root and prospective publication root remain distinct.
/// [`Self::with_publication`] consumes the original retirement batch and calls the publisher.
pub(in crate::edit) struct CountPublicationBounds<'a> {
    phase: CountPublicationPhase<'a>,
    end: PhysicalAddress,
}

impl<'a> CountPublicationBounds<'a> {
    /// Recovers the checked phase for another EOA proposal.
    pub(in crate::edit) fn into_phase(self) -> CountPublicationPhase<'a> {
        self.phase
    }

    /// Returns the predecessor topology root, or `None` for an empty count batch.
    pub(in crate::edit) fn original_root(&self) -> Option<ObjectId> {
        self.phase.counts.root()
    }

    /// Returns the prospective root associated with the pending retirements.
    pub(in crate::edit) fn root(&self) -> PhysicalAddress {
        self.phase.retirement.root
    }

    /// Returns the checked physical EOA.
    pub(in crate::edit) fn end(&self) -> PhysicalAddress {
        self.end
    }

    /// Borrows original count fields and positive remainders in allocation-address order.
    pub(in crate::edit) fn updates(&self) -> &[ProtectedCount] {
        self.phase.counts.updates()
    }
}

/// A physical EOA at or beyond every complete protected count allocation.
struct CountAllocationEnd(PhysicalAddress);

impl CountAllocationEnd {
    /// Checks the maximal end of the disjoint, address-ordered protected allocations.
    ///
    /// # Errors
    ///
    /// Returns [`CountPublicationError::Truncation`] if `end` precedes that allocation end.
    fn new(counts: &ProtectedCounts, end: PhysicalAddress) -> Result<Self, CountPublicationError> {
        if let Some(update) = counts.updates().last()
            && end < update.allocation().range().end()
        {
            return Err(CountPublicationError::Truncation {
                object: update.object(),
                allocation: update.allocation().range(),
                end,
            });
        }
        Ok(Self(end))
    }

    fn address(self) -> PhysicalAddress {
        self.0
    }
}

/// A rejected EOA with its unchanged publication phase.
pub(in crate::edit) struct CountEndFailure<'a> {
    /// The original phase, available for another EOA proposal.
    pub(in crate::edit) phase: CountPublicationPhase<'a>,
    /// The reason the proposed EOA was rejected.
    pub(in crate::edit) error: CountPublicationError,
}

/// An inapplicable source, intersecting retirement, or truncated count allocation.
#[derive(Debug)]
pub(in crate::edit) enum CountPublicationError {
    /// An unsuccessful source, publication, or allocator-claim check.
    Protection(CountProtectionError),
    /// A pending retirement intersecting a complete original count allocation.
    Retirement {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The complete original count allocation.
        allocation: FileRange,
        /// The complete retirement range.
        retired: FileRange,
        /// The retirement allocation's original free-space class.
        class: FreeClass,
    },
    /// A proposed EOA below the end of a complete original count allocation.
    Truncation {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The complete original count allocation.
        allocation: FileRange,
        /// The proposed physical EOA.
        end: PhysicalAddress,
    },
}

impl From<CountPublicationError> for Error {
    /// Reports overlapping retirements or truncation as unsupported count updates.
    fn from(error: CountPublicationError) -> Self {
        match error {
            CountPublicationError::Protection(error) => error.into(),
            CountPublicationError::Retirement { object, .. }
            | CountPublicationError::Truncation { object, .. } => {
                Self::ObjectReferenceCountUpdateUnsupported {
                    address: object.header_address().get(),
                    issue: ObjectReferenceCountUpdateIssue::AllocationUnavailable,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
