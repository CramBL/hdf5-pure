//! Count updates with an undo journal and retry after unchanged or restored writes.

use core::fmt;

use hdf5_pure_format::__private::ObjectReferenceCountUpdate;

use super::super::prepare::CountDirection;
use super::super::prepare::CountPreparationError;
use super::CountWriteScope;
use crate::edit::commit::JournalScope;
use crate::error::Error;

impl<'engine, 'counts> CountWriteScope<'engine, 'counts> {
    /// Prepares and journals count updates after the associated tree publication.
    ///
    /// Reads current headers and writes each count before its checksum. Uses the existing
    /// journal's completion and restoration barriers. Earlier journal entries and the
    /// published root, EOA, and adopted free-space state remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CountWriteFailure::Preparation`] with this scope if current-header preparation
    /// fails. Returns [`CountWriteFailure::Restored`] with prepared updates if dependent bytes
    /// are unchanged or successfully restored and ordered after a journal or barrier error.
    /// Returns [`CountWriteFailure::Uncertain`] and marks allocation publication uncertain if
    /// publication permission is unavailable or restoration fails.
    ///
    /// # Performance
    ///
    /// An empty batch requires no allocation or source reads. Retains compact encoded updates
    /// and releases acquired header buffers during preparation. Journal storage and image
    /// buffering are separate costs.
    pub(in crate::edit) fn write_counts(
        self,
    ) -> Result<(), Box<CountWriteFailure<'engine, 'counts>>> {
        self.write_changes(CountDirection::All)
    }

    /// Prepares and journals only counts below their original observed values.
    ///
    /// Uses the current-header preparation, journal barriers, and failure ownership of
    /// [`Self::write_counts`]. On success, a batch without decreases requires no allocation or
    /// source reads.
    ///
    /// # Errors
    ///
    /// Returns the preparation, restored-write, or uncertain-publication errors described by
    /// [`Self::write_counts`].
    pub(in crate::edit) fn write_decreases(
        self,
    ) -> Result<(), Box<CountWriteFailure<'engine, 'counts>>> {
        self.write_changes(CountDirection::Decrease)
    }

    /// Prepares matching updates before writing through the undo journal.
    ///
    /// # Errors
    ///
    /// Returns the preparation, restored-write, or uncertain-publication errors described by
    /// [`Self::write_counts`].
    fn write_changes(
        self,
        direction: CountDirection,
    ) -> Result<(), Box<CountWriteFailure<'engine, 'counts>>> {
        if self.counts.updates().is_empty() {
            return Ok(());
        }
        if let Err(error) = self.engine.context.allocations.before_reference() {
            self.engine.make_allocation_uncertain();
            return Err(Box::new(CountWriteFailure::Uncertain { error }));
        }
        let updates = match self.counts.prepare_updates(self.engine, direction) {
            Ok(updates) => updates,
            Err(error) => {
                return Err(Box::new(CountWriteFailure::Preparation {
                    scope: self,
                    error,
                }));
            }
        };
        PreparedCountWrites {
            scope: self,
            updates,
        }
        .apply()
    }
}

/// Encoded count updates with an exclusive engine loan for application or retry.
///
/// The caller retains the loan through preparation, application, and any retry. After a failed
/// attempt, retry requires unchanged dependent bytes or successful, ordered restoration.
/// Acquired header buffers are not retained.
pub(in crate::edit) struct PreparedCountWrites<'engine, 'counts> {
    scope: CountWriteScope<'engine, 'counts>,
    updates: Vec<ObjectReferenceCountUpdate>,
}

impl<'engine, 'counts> PreparedCountWrites<'engine, 'counts> {
    /// Writes encoded updates through the undo journal.
    ///
    /// Checks publication permission before writing. Journal reads acquire the prior field
    /// and checksum bytes, and image buffering may require additional reads.
    ///
    /// # Errors
    ///
    /// Returns [`CountWriteFailure::Restored`] with these updates if dependent bytes are unchanged
    /// or successfully restored and ordered after an operation error. Returns
    /// [`CountWriteFailure::Uncertain`] and marks allocation publication uncertain if permission
    /// is unavailable or restoration fails. Failed restoration retains both errors in
    /// [`Error::CommitPartiallyApplied`].
    pub(in crate::edit) fn apply(self) -> Result<(), Box<CountWriteFailure<'engine, 'counts>>> {
        let Self { scope, updates } = self;
        if let Err(error) = scope.engine.context.allocations.before_reference() {
            scope.engine.make_allocation_uncertain();
            return Err(Box::new(CountWriteFailure::Uncertain { error }));
        }
        let result = JournalScope::run(scope.engine, |journal| {
            for update in &updates {
                for (range, bytes) in update.writes() {
                    journal.write(range.start(), bytes)?;
                }
            }
            Ok(())
        });
        match result {
            Ok(()) => Ok(()),
            Err(error @ Error::CommitPartiallyApplied { .. }) => {
                scope.engine.make_allocation_uncertain();
                Err(Box::new(CountWriteFailure::Uncertain { error }))
            }
            Err(error) => Err(Box::new(CountWriteFailure::Restored {
                writes: Self { scope, updates },
                error,
            })),
        }
    }
}

/// A failed count update with its remaining application or retry permission.
pub(in crate::edit) enum CountWriteFailure<'engine, 'counts> {
    /// A preparation failure before count writes.
    Preparation {
        /// The original scope, with file bytes unchanged by count preparation.
        scope: CountWriteScope<'engine, 'counts>,
        /// The current-header acquisition, framing, or encoding error.
        error: CountPreparationError,
    },
    /// An operation failure with dependent bytes unchanged or restored and ordered.
    Restored {
        /// The encoded updates and exclusive scope available for retry.
        writes: PreparedCountWrites<'engine, 'counts>,
        /// The original journal read, write, or completion-barrier error.
        error: Error,
    },
    /// Unavailable publication permission or failed restoration, with retry prohibited.
    Uncertain {
        /// The permission error, or both the operation and restoration errors.
        error: Error,
    },
}

impl fmt::Debug for CountWriteFailure<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (variant, error): (&str, &dyn fmt::Debug) = match self {
            Self::Preparation { scope: _, error } => ("Preparation", error),
            Self::Restored { writes: _, error } => ("Restored", error),
            Self::Uncertain { error } => ("Uncertain", error),
        };
        formatter
            .debug_struct(variant)
            .field("error", error)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
