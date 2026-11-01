//! Staged transaction planning, application, publication, and recovery.
//!
//! Planning borrows the staged edits. Application consumes them after preparation succeeds.
//! Rollback restores values recorded in the undo journal and allocator state only before
//! publication is attempted.
//! Publication and recovery use [`PublicationState`] to preserve that boundary after I/O errors.

use hdf5_pure_space::__private::ReturnedExtents;

#[cfg(doc)]
use crate::edit::SyncPolicy;
use crate::error::Error;

use self::plan::{CommitPreparation, PreparedCommit};
use super::WriteEngine;
use super::staging::StagedEdits;

mod apply;
pub(super) mod plan;
mod publish;
mod rollback;

/// Selects manager sizing and buffering for allocation or staged publication.
#[derive(Clone, Copy)]
pub(in crate::edit) enum PublicationPurpose {
    /// Drains every buffering mode before confirming immediate allocation ownership.
    Allocation,
    /// Uses the staged commit's synchronization and buffering policy.
    Commit,
}

/// Records the outcome of the latest superblock publication attempt.
///
/// The publisher enters [`Uncertain`](Self::Uncertain) before writing and enters
/// [`Published`](Self::Published) after the write and its policy-dependent barrier succeed.
/// Both states prevent rollback because the file may reference the new allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PublicationState {
    /// The publication write and its barrier completed successfully.
    Published,
    /// Publication has not been attempted since the state was reset.
    Ready,
    /// Publication was attempted, with completion unconfirmed.
    Uncertain,
}

impl PublicationState {
    /// Returns whether rollback must preserve the allocations used by this attempt.
    pub(super) fn attempted(self) -> bool {
        !matches!(self, Self::Ready)
    }
}

/// Distinguishes a rejected plan's staged inputs from consumed application inputs.
#[allow(
    clippy::large_enum_variant,
    reason = "the rejected staged set returns to the engine without a heap allocation"
)]
enum CommitOutcome {
    /// Application consumed the staged inputs, or the staged set was empty.
    Applied(Result<(), Error>),
    /// Planning failed while the staged inputs remained available for retry.
    Rejected { staged: StagedEdits, error: Error },
}

impl WriteEngine {
    /// Issues pending writes, synchronizes the file, and completes pending PAGE publication.
    ///
    /// The synchronization barrier is durable under every [`SyncPolicy`]. Publication recovery
    /// verifies issued bytes before adopting the superblock and leaves later staged edits pending.
    /// Outside recoverable nonpersistent PAGE publication, the editor discards queued writes and
    /// marks the session uncertain after a failed synchronization.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if writing or synchronization fails. Returns the publication recovery
    /// error if the issued superblock cannot be read or verified.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    /// Returns [`crate::FormatError::Internal`] if an allocation batch remains provisional.
    pub(crate) fn force_sync(&mut self) -> Result<(), Error> {
        self.context.allocations.before_reference()?;
        let result = self
            .image
            .sync_all()
            .and_then(|()| self.complete_page_publication());
        if result.is_err() && self.fresh_page_boundary.is_none() {
            self.make_allocation_uncertain();
        }
        result
    }

    /// Publishes pending free-space state and releases unused aggregate tails.
    ///
    /// In a persistent session, the editor rewrites stale free-space managers and releases unused
    /// aggregate spare through those managers. With classic aggregation, it releases unused space
    /// at EOA after publishing the shorter EOA with the current root. Uncertain publication
    /// prevents aggregate cleanup. Non-persisting sessions without classic aggregation require no
    /// finalization. Pending immediate retirements require manager publication even when EOF is
    /// unchanged.
    /// After a failed publication attempt, the editor discards queued writes and marks the session
    /// uncertain, except for recoverable nonpersistent PAGE publication.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if metadata serialization, aggregate planning, or address
    /// resolution fails. Returns [`Error::Io`] if writing, syncing, or truncating fails.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    /// Returns [`crate::FormatError::Internal`] if an allocation batch remains provisional.
    pub(crate) fn finalize_persist(&mut self) -> Result<(), Error> {
        self.context.allocations.before_reference()?;
        if self.fresh_page_boundary.is_none() {
            self.publication = PublicationState::Ready;
        }
        let result = self.finalize_persist_inner();
        if result.is_err() && self.publication.attempted() && self.fresh_page_boundary.is_none() {
            self.make_allocation_uncertain();
        }
        result
    }

    /// Applies staged edits using the session's file-space strategy and synchronization policy.
    ///
    /// Tree edits rebuild affected group headers and publish the new root through the
    /// superblock. Dataset value writes may update existing storage directly. On success,
    /// the staged set is cleared.
    /// With no staged edits, finalizes pending manager changes from immediate tail retirement.
    ///
    /// Rejection during preflight preserves the staged set for retry. Once application
    /// consumes the staged set, a failed attempt discards that work. Writes into reusable
    /// or newly allocated space may remain unreferenced after a failure.
    ///
    /// On failure before root publication is attempted, the editor restores allocator state and
    /// attempts to restore prior values from the undo journal. Rollback is disabled once
    /// publication is attempted because a failed publication write may already have changed
    /// the file.
    ///
    /// After publication, reference patching or optional root-header reclamation can fail.
    /// The committed tree retains the batch's values, including its in-place overwrites.
    /// Outside recoverable nonpersistent PAGE publication, the editor discards queued writes and
    /// marks the session uncertain if an operation fails after attempted publication.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if an edit or an encountered file structure cannot
    /// be reproduced. Other preparation and I/O errors propagate from planning, writing,
    /// or synchronizing the file. Returns [`Error::CommitPartiallyApplied`] if restoring prior
    /// values from the undo journal fails, with both the original failure and the restore error.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    /// Returns [`crate::FormatError::Internal`] if an allocation batch remains provisional.
    pub fn commit(&mut self) -> Result<(), Error> {
        self.context.allocations.before_reference()?;
        if self.fresh_page_boundary.is_some() && self.publication.attempted() {
            self.barrier()?;
            self.complete_page_publication()?;
        }
        self.synchronize_aggregates()?;
        let snapshot = self.snapshot_free();
        self.publication = PublicationState::Ready;
        self.inplace_undo.clear();
        let staged = std::mem::take(&mut self.staged);
        let outcome = self.commit_inner(staged);
        let result = match outcome {
            CommitOutcome::Rejected { staged, error } => {
                Err(self.rollback_commit(snapshot, staged, error))
            }
            CommitOutcome::Applied(Err(error)) if !self.publication.attempted() => {
                Err(self.rollback_commit(snapshot, StagedEdits::default(), error))
            }
            CommitOutcome::Applied(result) => {
                self.staged_generation += 1;
                result
            }
        };
        if self.publication.attempted()
            && let Some(aggregation) = &mut self.aggregation
        {
            aggregation.publication_confirmed = result.is_ok();
        }
        self.inplace_undo.clear();
        if self.fresh_page_boundary.is_some() && self.publication.attempted() {
            if result.is_ok() {
                self.publication = PublicationState::Ready;
            } else {
                // A failed barrier can leave the new root on disk. Pending writes and uncertain
                // bytes remain owned until a later commit completes the barrier and recovery.
                self.recover_page_publication().ok();
            }
        }
        if result.is_err() && self.publication.attempted() && self.fresh_page_boundary.is_none() {
            self.make_allocation_uncertain();
        }
        result
    }

    /// Publishes pending manager state or releases unused aggregate tails.
    ///
    /// # Errors
    ///
    /// Propagates aggregate-finalization and persistent-publication errors.
    fn finalize_persist_inner(&mut self) -> Result<(), Error> {
        if self.aggregation.is_some() {
            if self.persist.is_some() {
                return self.finalize_persistent_aggregates();
            }
            return self.finalize_aggregates();
        }
        if self.persist.is_none() || (self.image.len() == self.fsm_len && !self.fsm_dirty) {
            return Ok(());
        }
        self.commit_persisting_releasing_reserve(
            self.context
                .addresses()
                .resolve(self.context.superblock().root_group_address)?,
            ReturnedExtents::default(),
        )
    }

    /// Prepares edits and transfers their ownership to application after planning succeeds.
    fn commit_inner(&mut self, staged: StagedEdits) -> CommitOutcome {
        if staged.is_empty() {
            return CommitOutcome::Applied(if self.fsm_dirty {
                self.finalize_persist()
            } else {
                Ok(())
            });
        }
        self.begin_commit(&staged);
        match PreparedCommit::prepare(self, staged) {
            CommitPreparation::Prepared(prepared) => {
                CommitOutcome::Applied(self.apply_commit(prepared))
            }
            CommitPreparation::Rejected { staged, error } => {
                CommitOutcome::Rejected { staged, error }
            }
        }
    }

    /// Clears attempt state and cached append geometry.
    fn begin_commit(&mut self, staged: &StagedEdits) {
        self.superseded_heaps.clear();
        self.invalidate_heap_provenance(staged);

        self.located.clear();
        self.resolved.clear();
        self.committed = true;
    }
}
