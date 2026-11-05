//! Completion and restoration of writes recorded in the undo journal.

use hdf5_pure_core::PhysicalAddress;

use crate::edit::WriteEngine;
use crate::error::Error;

/// An exclusive engine borrow with a journal checkpoint for one fallible write operation.
///
/// After the operation returns, completion releases or restores only its appended journal
/// entries. Earlier entries remain available for whole rollback. Publication and allocation
/// permissions remain the caller's responsibility.
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
