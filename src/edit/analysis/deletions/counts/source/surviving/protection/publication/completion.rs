//! Completion scopes for protected count allocations after tree publication.

use std::collections::BTreeMap;

use hdf5_pure_core::PhysicalAddress;

use super::CountAllocationEnd;
use super::CountProtectionError;
use super::CountPublicationBounds;
use super::CountPublicationError;
use super::CountPublicationPhase;
use super::ProtectedCounts;
use crate::edit::WriteEngine;
use crate::edit::commit::PublishedTree;
use crate::error::Error;

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

#[cfg(test)]
mod tests;
