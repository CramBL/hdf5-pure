//! Count updates ordered around tree publication and checked against protected allocations.

use std::collections::BTreeMap;

use hdf5_pure_core::PhysicalAddress;

use self::write::CountWriteFailure;
use super::CountAllocationEnd;
use super::CountProtectionError;
use super::CountPublicationBounds;
use super::CountPublicationError;
use super::CountPublicationPhase;
use super::ProtectedCounts;
use super::prepare::CountDirection;
use super::prepare::CountPreparationError;
use crate::edit::WriteEngine;
use crate::edit::commit::JournalScope;
use crate::edit::commit::PublicationState;
use crate::edit::commit::PublishedTree;
use crate::error::Error;
use crate::error::FormatError;

mod write;

impl CountPublicationBounds<'_> {
    /// Publishes the associated tree with its original retirements before calling `complete`.
    ///
    /// Calls `complete` after stored-reference updates, root reclamation, and requested truncation
    /// succeed. Checks the root passed to the publisher and the confirmed EOA. For a nonempty
    /// count batch, also checks the allocation session, publication availability, and allocator
    /// claims. The callback borrows the engine and protected counts with the final adopted root
    /// and EOA.
    /// Source freshness requires separate evidence.
    ///
    /// # Errors
    ///
    /// Returns the publication, reference-update, root-reclamation, or address-resolution error
    /// before calling `complete`. Passes a [`CountCompletionError`] to `complete` if the
    /// post-publication checks fail. Returns the error from `complete` if completion fails.
    pub(in crate::edit) fn with_publication<T>(
        self,
        relocations: BTreeMap<u64, u64>,
        complete: impl for<'engine, 'counts> FnOnce(
            Result<CountWriteScope<'engine, 'counts>, CountCompletionError>,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let Self { phase, end } = self;
        let CountPublicationPhase {
            engine,
            counts,
            retirement,
        } = phase;
        let request = CountCompletion {
            counts: &counts,
            root: retirement.root,
            proposed_end: end,
        };
        engine.publish_pending_tree_with_completion(retirement, relocations, |published| {
            complete(request.into_write_scope(published))
        })
    }

    /// Writes count increases before publishing references and decreases after publication.
    ///
    /// Prepares each direction from current complete headers. Journals increases through
    /// reference publication, restoring them on failure before a publication attempt.
    /// Preserves increases after confirmed or uncertain publication. After confirmed publication,
    /// checks the protected allocations before preparing decreases and writing through the journal.
    /// Uses the associated tree's original retirement batch and the existing publication barriers.
    ///
    /// # Errors
    ///
    /// Returns [`CountPublicationUpdateError::Preparation`] if current-header preparation fails,
    /// [`CountPublicationUpdateError::Completion`] if post-publication allocation checks fail, or
    /// [`CountPublicationUpdateError::Operation`] for allocation permission, publication, journal,
    /// or barrier errors.
    /// After a publication attempt, marks allocation publication uncertain on error, except for
    /// the existing nonpersistent PAGE recovery state. Failed restoration also marks publication
    /// uncertain and retains both errors in [`Error::CommitPartiallyApplied`].
    pub(in crate::edit) fn publish_counts(
        self,
        relocations: BTreeMap<u64, u64>,
    ) -> Result<(), CountPublicationUpdateError> {
        let Self { phase, end } = self;
        let CountPublicationPhase {
            engine,
            counts,
            retirement,
        } = phase;
        engine
            .context
            .allocations
            .before_reference()
            .map_err(CountPublicationUpdateError::Operation)?;
        let increases = counts
            .prepare_updates(engine, CountDirection::Increase)
            .map_err(CountPublicationUpdateError::Preparation)?;
        let request = CountCompletion {
            counts: &counts,
            root: retirement.root,
            proposed_end: end,
        };
        let result = JournalScope::with_publication(
            engine,
            |journal| {
                for update in increases {
                    for (range, bytes) in update.writes() {
                        journal.write(range.start(), bytes)?;
                    }
                }
                Ok(())
            },
            |engine| {
                engine.publish_pending_tree_with_completion(retirement, relocations, |published| {
                    Ok(match request.into_write_scope(published) {
                        Ok(scope) => scope.write_decreases().map_err(|failure| (*failure).into()),
                        Err(error) => Err(CountPublicationUpdateError::Completion(error)),
                    })
                })
            },
        )
        .map_err(CountPublicationUpdateError::Operation)
        .and_then(core::convert::identity);
        if result.is_err()
            && engine.publication.attempted()
            && !(engine.fresh_page_boundary.is_some()
                && engine.publication == PublicationState::Uncertain)
            || matches!(
                &result,
                Err(CountPublicationUpdateError::Operation(
                    Error::CommitPartiallyApplied { .. }
                ))
            )
        {
            engine.make_allocation_uncertain();
        }
        result
    }
}

/// Borrowed count facts and the root and EOA proposed before publication.
struct CountCompletion<'counts> {
    counts: &'counts ProtectedCounts,
    root: PhysicalAddress,
    proposed_end: PhysicalAddress,
}

impl<'counts> CountCompletion<'counts> {
    /// Checks the publication's requested root and the protected allocations in the adopted state.
    ///
    /// For a nonempty count batch, checks the allocation session, publication availability, and
    /// reusable, retained, and aggregate-spare ranges. Checks the confirmed EOA for every batch.
    /// Source freshness requires separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`CountCompletionError::Root`] if the publisher was called with another root.
    /// Returns [`CountCompletionError::Protection`] for an inapplicable session, unavailable
    /// publication, conflicting allocator claims, address-conversion errors, or an EOA below a
    /// protected allocation's end.
    ///
    /// # Performance
    ///
    /// On success, borrows the protected facts and allocator lists without allocating.
    /// Uses retained metadata without source reads or topology discovery.
    fn into_write_scope<'engine>(
        self,
        published: PublishedTree<'engine>,
    ) -> Result<CountWriteScope<'engine, 'counts>, CountCompletionError> {
        let Self {
            counts,
            root: expected,
            proposed_end,
        } = self;
        let actual = published.requested_root();
        if actual != expected {
            return Err(CountCompletionError::Root { expected, actual });
        }
        let root = published.root();
        let end = published.end();
        let engine = published.into_engine();
        if let Some(source) = &counts.source {
            if !source.session.same(&engine.context.allocations.session()) {
                return Err(CountCompletionError::Protection(
                    CountPublicationError::Protection(CountProtectionError::Session),
                ));
            }
            engine
                .context
                .allocations
                .before_reference()
                .map_err(CountProtectionError::Publication)
                .map_err(CountPublicationError::Protection)
                .map_err(CountCompletionError::Protection)?;
            counts
                .exclude_space(engine)
                .map_err(CountPublicationError::Protection)
                .map_err(CountCompletionError::Protection)?;
        }
        let end = CountAllocationEnd::new(counts, end)
            .map_err(CountCompletionError::Protection)?
            .address();
        Ok(CountWriteScope {
            engine,
            counts,
            root,
            end,
            proposed_end,
        })
    }
}

/// An exclusive engine loan with protected count facts checked after tree publication.
///
/// The publisher consumed the original retirement batch before creating this scope.
/// The producer checks the root passed to the publisher and the confirmed EOA. For a nonempty
/// count batch, it also checks the allocation session, publication availability, and allocator
/// claims. Source freshness requires separate evidence.
pub(in crate::edit) struct CountWriteScope<'engine, 'counts> {
    engine: &'engine mut WriteEngine,
    counts: &'counts ProtectedCounts,
    /// The final adopted physical root, including any root reclamation.
    root: PhysicalAddress,
    /// The confirmed physical EOA after publication and requested truncation.
    end: PhysicalAddress,
    /// The physical EOA checked before publication.
    proposed_end: PhysicalAddress,
}

/// An incompatible publication root or failed post-publication allocation check.
#[derive(Debug)]
pub(in crate::edit) enum CountCompletionError {
    /// An inapplicable session, unavailable publication, conflicting claim, or truncated allocation.
    Protection(CountPublicationError),
    /// A requested root different from the original retirement batch's root.
    Root {
        /// The root associated with the original retirement batch.
        expected: PhysicalAddress,
        /// The root requested in the completed publication call.
        actual: PhysicalAddress,
    },
}

/// A preparation, publication, or protected-allocation error during ordered count updates.
#[derive(Debug)]
pub(in crate::edit) enum CountPublicationUpdateError {
    /// A failed post-publication root or allocation check.
    Completion(CountCompletionError),
    /// An allocation permission, publication, journal, or barrier error, including failed
    /// restoration.
    Operation(Error),
    /// A current-header acquisition, framing, or encoding error.
    Preparation(CountPreparationError),
}

impl From<CountWriteFailure<'_, '_>> for CountPublicationUpdateError {
    fn from(failure: CountWriteFailure<'_, '_>) -> Self {
        match failure {
            CountWriteFailure::Preparation { scope: _, error } => Self::Preparation(error),
            CountWriteFailure::Restored { writes: _, error }
            | CountWriteFailure::Uncertain { error } => Self::Operation(error),
        }
    }
}

impl From<CountCompletionError> for Error {
    /// Converts allocation failures and inconsistent roots to public errors.
    ///
    /// Reports an inconsistent root through [`FormatError::Internal`].
    fn from(error: CountCompletionError) -> Self {
        match error {
            CountCompletionError::Protection(error) => error.into(),
            CountCompletionError::Root { .. } => FormatError::Internal(
                "count completion has an inconsistent publication root".into(),
            )
            .into(),
        }
    }
}

impl From<CountPublicationUpdateError> for Error {
    /// Returns operation errors unchanged and converts preparation and completion failures.
    fn from(error: CountPublicationUpdateError) -> Self {
        match error {
            CountPublicationUpdateError::Completion(error) => error.into(),
            CountPublicationUpdateError::Preparation(error) => error.into(),
            CountPublicationUpdateError::Operation(error) => error,
        }
    }
}

#[cfg(test)]
mod tests;
