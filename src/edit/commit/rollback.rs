//! Allocator snapshots and restoration of value writes from the undo journal.

use std::collections::HashMap;

use hdf5_pure_space::__private::{FreeList, SessionSpaceSnapshot};

use crate::edit::WriteEngine;
use crate::edit::aggregation::AggregateSpace;
use crate::edit::staging::StagedEdits;
use crate::error::Error;
use crate::object_path::ObjectPathBuf;

/// A saved allocator state for rollback or retention after uncertain publication.
pub(in crate::edit) struct FreeSnapshot {
    space: SessionSpaceSnapshot,
    aggregation: Option<AggregateSpace>,
    /// Unspent allocations reserved for immediate appends.
    reserved: FreeList,
    /// Heap collections recorded for variable-length value replacements in this session.
    vl_overwrite_heaps: HashMap<ObjectPathBuf, Vec<(u64, u64)>>,
}

impl WriteEngine {
    /// Restores prior values, allocation state, and the unconsumed staged inputs.
    ///
    /// The caller must establish that publication has not been attempted. An application failure
    /// passes an empty staged set because application consumed those inputs.
    ///
    /// Returns `error` after successful restoration, or [`Error::CommitPartiallyApplied`] with
    /// both failures if restoring prior values fails.
    pub(super) fn rollback_commit(
        &mut self,
        snapshot: FreeSnapshot,
        staged: StagedEdits,
        error: Error,
    ) -> Error {
        let restored = self.undo_inplace_writes();
        self.restore_free(snapshot);
        self.staged = staged;
        match restored {
            Ok(()) => error,
            Err(restore) => Error::CommitPartiallyApplied {
                refusal: Box::new(error),
                restore: Box::new(restore),
            },
        }
    }

    /// Captures allocator state and heap allocation records at the current placement boundary.
    pub(in crate::edit) fn snapshot_free(&self) -> FreeSnapshot {
        FreeSnapshot {
            space: self.space.allocation_snapshot(),
            aggregation: self.aggregation,
            reserved: self.reserved.clone(),
            vl_overwrite_heaps: self.vl_overwrite_heaps.clone(),
        }
    }

    /// Restores allocator reservations and recorded heap ownership.
    ///
    /// For rollback, the caller restores values from the undo journal first and establishes that
    /// publication has not been attempted. After uncertain publication, the caller uses a snapshot
    /// taken after placement, in which every possibly referenced allocation remains allocated.
    pub(in crate::edit) fn restore_free(&mut self, snapshot: FreeSnapshot) {
        self.vl_overwrite_heaps = snapshot.vl_overwrite_heaps;
        self.space.restore_allocation(snapshot.space);
        self.aggregation = snapshot.aggregation;
        self.reserved = snapshot.reserved;
    }

    /// Records prior values before overwriting an existing dataset range.
    ///
    /// Planning must establish that `raw` fits the existing value range.
    ///
    /// # Errors
    ///
    /// Returns the read or write error if the prior values cannot be read or the replacement
    /// cannot be written. The journal entry is retained after a write error.
    pub(in crate::edit) fn write_inplace_journaled(
        &mut self,
        at: u64,
        raw: &[u8],
    ) -> Result<(), Error> {
        let prior = self.image().read_exact_at(at, raw.len())?;
        self.inplace_undo.push((at, prior));
        self.write_at(at, raw)
    }

    /// Restores values recorded in the undo journal in reverse write order and clears the journal.
    ///
    /// Every restoration is attempted, including after a write error. A successful restoration
    /// ends with a data barrier to order the restored bytes before subsequent writes.
    ///
    /// # Errors
    ///
    /// Returns the first restoration error, or the barrier error if every restoration succeeds.
    pub(in crate::edit) fn undo_inplace_writes(&mut self) -> Result<(), Error> {
        let journal = core::mem::take(&mut self.inplace_undo);
        if journal.is_empty() {
            return Ok(());
        }
        let mut failed = None;
        // Different hard-link paths can overwrite the same value range. Restore the most
        // recent entry first so the original bytes are written last.
        for (at, prior) in journal.into_iter().rev() {
            if let Err(e) = self.write_at(at, &prior) {
                failed.get_or_insert(e);
            }
        }
        match failed {
            Some(e) => Err(e),
            // Restore ordering requires a barrier even though the file length is unchanged.
            None => self.barrier_data(),
        }
    }
}
