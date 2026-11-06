//! Global heap compaction through the undo journal.
//!
//! The caller establishes source freshness, permission to remove the selected members, and
//! protection of each collection's full allocation from reuse. Plans must describe distinct
//! collections in the engine's current image. [`after_publication`] requires association with
//! the completed publication. [`rootless`] preserves undo for the enclosing commit.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::GlobalHeapCompaction;

use crate::convert::Narrow;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;

use super::JournalScope;
use super::PublishedTree;

/// Applies each collection plan with a separate journal checkpoint after publication.
///
/// Checks each complete collection against the published EOA and the image length. After
/// recording undo entries, issues pending writes in every buffering mode, applies the engine's
/// synchronization policy, and releases that collection's undo entries on success. Earlier
/// journal entries remain available.
/// After a failure, completed collections remain compacted and subsequent plans are unattempted.
///
/// # Errors
///
/// Returns [`HeapMutationFailure`] with the completed prefix length, allocation state, and
/// bounds, read, write, completion, or restoration error. Failed restoration or uncertain
/// allocation permission prevents further allocation through the engine.
pub(in crate::edit) fn after_publication(
    published: PublishedTree<'_>,
    plans: &[GlobalHeapCompaction],
) -> Result<(), HeapMutationFailure> {
    let end = published.end().get();
    let engine = published.into_engine();
    published_batch(engine, plans, end)
}

fn published_batch(
    engine: &mut WriteEngine,
    plans: &[GlobalHeapCompaction],
    end: u64,
) -> Result<(), HeapMutationFailure> {
    for (completed, plan) in plans.iter().enumerate() {
        let result = match engine.context.allocations.before_reference() {
            Ok(()) => {
                if plan.range().end().get() > end {
                    Err(crate::source::unexpected_eof(plan.range(), end).into())
                } else {
                    JournalScope::run_issued(engine, |journal| apply(journal, plan))
                }
            }
            Err(error) => {
                engine.make_allocation_uncertain();
                Err(error)
            }
        };
        if let Err(error) = result {
            if matches!(
                error,
                Error::AllocationPublicationUncertain | Error::CommitPartiallyApplied { .. }
            ) {
                engine.make_allocation_uncertain();
            }
            let state = match engine.context.allocations.available() {
                Ok(()) => HeapMutationState::Restored,
                Err(_) => HeapMutationState::Uncertain,
            };
            return Err(HeapMutationFailure {
                completed,
                state,
                error,
            });
        }
    }
    Ok(())
}

/// Applies all collection plans while retaining undo for the enclosing unpublished commit.
///
/// With a nonempty journal, issues pending writes in every buffering mode and applies the
/// engine's synchronization policy on success. The enclosing commit retains all prior descriptor
/// and collection undo entries until its own completion. If applying the plans or completing
/// their writes fails, attempts to restore the whole journal in reverse write order, including
/// entries recorded before this call.
///
/// Copies records through a 4096-byte scratch buffer. Retained undo includes every overwritten
/// destination byte and free space header, in addition to the earlier journal. Repeated writes
/// retain separate prior-byte copies. Journal capacity, plans, image buffering, and any mirror
/// require additional memory.
///
/// # Errors
///
/// Returns [`Error::AllocationPublicationUncertain`] if allocation permission is uncertain,
/// or [`FormatError::Internal`] if allocations are provisional or publication was attempted.
/// Returns bounds, read, write, or completion errors. Returns [`Error::CommitPartiallyApplied`]
/// with both errors if restoration fails, and marks allocation state uncertain.
pub(in crate::edit) fn rootless(
    engine: &mut WriteEngine,
    plans: &[GlobalHeapCompaction],
) -> Result<(), Error> {
    engine.context.allocations.before_reference()?;
    JournalScope::run_rootless(engine, |journal| {
        for plan in plans {
            apply(journal, plan)?;
        }
        Ok(())
    })
}

/// Records the completed prefix and the failure of the next collection operation.
#[derive(Debug)]
pub(in crate::edit) struct HeapMutationFailure {
    /// The number of plans completed before the failure.
    pub(in crate::edit) completed: usize,
    /// The allocation state after the failed operation and any restoration attempt.
    pub(in crate::edit) state: HeapMutationState,
    /// The operation error, including a restoration error if restoration failed.
    pub(in crate::edit) error: Error,
}

/// Records allocation availability after a collection operation fails.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum HeapMutationState {
    /// The failed collection is unchanged or restored, and allocation state is available.
    Restored,
    /// Allocation permission or restoration could not be confirmed.
    Uncertain,
}

fn apply(journal: &mut JournalScope<'_>, plan: &GlobalHeapCompaction) -> Result<(), Error> {
    let parent = plan.range();
    if parent.end().get() > journal.length() {
        return Err(crate::source::unexpected_eof(parent, journal.length()).into());
    }
    let Some(free) = plan.free_space() else {
        return Ok(());
    };
    let mut scratch = [0; COPY_BYTES];
    for movement in plan.moves() {
        let source = movement.object().range();
        let destination = movement.destination();
        let mut copied = 0;
        while copied < source.length() {
            let length = (source.length() - copied).min(scratch.len().to_u64());
            let read = FileRange::new(source.start().checked_add(copied)?, length)?;
            let write = FileRange::new(destination.start().checked_add(copied)?, length)?;
            contained(parent, read)?;
            contained(parent, write)?;
            let bytes = &mut scratch[..length.to_usize()?];
            journal.read(read.start(), bytes)?;
            journal.write(write.start(), bytes)?;
            copied += length;
        }
    }
    contained(parent, free.header_range())?;
    journal.write(free.header_range().start(), free.bytes())
}

fn contained(parent: FileRange, range: FileRange) -> Result<(), FormatError> {
    if parent.contains(range) {
        Ok(())
    } else {
        Err(FormatError::InsufficientRegion {
            required_start: range.start().get(),
            required_length: range.length(),
            provided_start: parent.start().get(),
            provided_length: parent.length(),
        })
    }
}

const COPY_BYTES: usize = 4096;

#[cfg(test)]
mod tests;
