//! Completion and restoration of writes recorded in the undo journal.

use hdf5_pure_core::PhysicalAddress;

use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;

/// Borrows the engine exclusively to journal writes and control their completion.
///
/// [`Self::run`] and [`Self::run_issued`] preserve entries before the operation's checkpoint.
/// [`Self::run_rootless`] retains the whole journal on success and attempts to restore it if
/// application or completion fails.
/// The caller establishes publication and allocation permissions.
pub(in crate::edit) struct JournalScope<'engine> {
    engine: &'engine mut WriteEngine,
    checkpoint: usize,
}

impl<'engine> JournalScope<'engine> {
    /// Runs `apply` with restoration limited to the operation's appended journal entries.
    ///
    /// A successful operation with new entries ends with the engine's completion barrier before
    /// those entries are released. On an operation or barrier error, attempts every restoration
    /// in reverse write order. Successful restoration ends with a data barrier. Returns the
    /// operation result directly when no entries were appended.
    ///
    /// # Errors
    ///
    /// Returns errors from `apply` or the completion barrier. If restoration also fails, returns
    /// [`Error::CommitPartiallyApplied`] with both the original and restoration errors.
    pub(in crate::edit) fn run(
        engine: &'engine mut WriteEngine,
        apply: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let checkpoint = engine.inplace_undo.len();
        let mut scope = Self { engine, checkpoint };
        let result = apply(&mut scope);
        scope.complete(result)
    }

    /// Issues writes in every buffering mode before releasing the appended undo entries.
    ///
    /// Applies the engine's synchronization policy after issuing pending writes, including
    /// writes queued before this call. On failure, attempts to restore only the appended entries
    /// in reverse order and issue the restored bytes. Earlier journal entries remain available.
    /// Returns the operation result directly when no entries were appended.
    ///
    /// # Errors
    ///
    /// Returns errors from `apply`, issuing writes, or the completion barrier. Returns
    /// [`Error::CommitPartiallyApplied`] with both errors if restoration fails, and marks
    /// allocation state uncertain.
    pub(in crate::edit) fn run_issued(
        engine: &'engine mut WriteEngine,
        apply: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let checkpoint = engine.inplace_undo.len();
        let mut scope = Self { engine, checkpoint };
        let result = apply(&mut scope);
        scope.complete_issued(result, false)
    }

    /// Retains the whole journal after issuing an unpublished operation's writes.
    ///
    /// Uses the completion and restoration rules of [`Self::run_issued`] with a checkpoint of
    /// zero. Successful completion retains all journal entries for the enclosing commit.
    /// If `apply` or completion fails, attempts to restore all entries, including those recorded
    /// before this call.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if publication was attempted. Otherwise returns the
    /// operation, completion, or restoration errors described by [`Self::run_issued`].
    pub(in crate::edit) fn run_rootless(
        engine: &'engine mut WriteEngine,
        apply: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if engine.publication.attempted() {
            return Err(FormatError::Internal(
                "rootless writes follow a publication attempt".into(),
            )
            .into());
        }
        let mut scope = Self {
            engine,
            checkpoint: 0,
        };
        let result = apply(&mut scope);
        scope.complete_issued(result, true)
    }

    /// Records prior bytes before writing at a physical file position.
    ///
    /// The caller establishes that the complete range may be overwritten.
    ///
    /// # Errors
    ///
    /// Returns the read or write errors of [`WriteEngine::write_inplace_journaled`]. The journal
    /// entry remains available for restoration after a write error.
    pub(in crate::edit) fn write(
        &mut self,
        at: PhysicalAddress,
        bytes: &[u8],
    ) -> Result<(), Error> {
        self.engine.write_inplace_journaled(at, bytes)
    }

    /// Reads from the current image, including pending writes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if reading into `bytes` at `at` fails.
    pub(in crate::edit) fn read(&self, at: PhysicalAddress, bytes: &mut [u8]) -> Result<(), Error> {
        self.engine.image().read_at(at.get(), bytes)?;
        Ok(())
    }

    /// Returns the image length, including pending extension writes.
    pub(in crate::edit) fn length(&self) -> u64 {
        self.engine.image().len()
    }

    fn complete_issued(self, result: Result<(), Error>, retain: bool) -> Result<(), Error> {
        let Self { engine, checkpoint } = self;
        if engine.inplace_undo.len() == checkpoint {
            return result;
        }
        let result = result
            .and_then(|()| engine.image.flush())
            .and_then(|()| engine.barrier());
        match result {
            Ok(()) => {
                if !retain {
                    engine.inplace_undo.truncate(checkpoint);
                }
                Ok(())
            }
            Err(refusal) => {
                let restore = engine
                    .undo_inplace_since(checkpoint)
                    .and_then(|()| engine.image.flush());
                match restore {
                    Ok(()) => Err(refusal),
                    Err(restore) => {
                        engine.make_allocation_uncertain();
                        Err(Error::CommitPartiallyApplied {
                            refusal: Box::new(refusal),
                            restore: Box::new(restore),
                        })
                    }
                }
            }
        }
    }

    /// Orders successful writes or restores failed writes after the saved checkpoint.
    ///
    /// Preserves earlier entries and skips barriers when no new entries were appended.
    ///
    /// # Errors
    ///
    /// Returns the operation, barrier, or restoration errors described by [`Self::run`].
    fn complete(self, result: Result<(), Error>) -> Result<(), Error> {
        let Self { engine, checkpoint } = self;
        if engine.inplace_undo.len() == checkpoint {
            return result;
        }
        let result = result.and_then(|()| engine.barrier());
        match result {
            Ok(()) => {
                engine.inplace_undo.truncate(checkpoint);
                Ok(())
            }
            Err(refusal) => match engine.undo_inplace_since(checkpoint) {
                Ok(()) => Err(refusal),
                Err(restore) => Err(Error::CommitPartiallyApplied {
                    refusal: Box::new(refusal),
                    restore: Box::new(restore),
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests;
