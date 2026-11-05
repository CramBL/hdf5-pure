//! Completion and restoration of writes recorded in the undo journal.

use hdf5_pure_core::PhysicalAddress;

use crate::edit::WriteEngine;
use crate::error::Error;

/// An exclusive engine borrow with a journal checkpoint for one fallible write operation.
///
/// Completion orders writes and releases or retains their appended undo entries according to
/// the chosen operation. Restoration uses only entries after the checkpoint. Earlier entries
/// remain available for whole rollback. Publication and allocation permissions remain the
/// caller's responsibility.
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
        Self::run_with_retention(engine, apply, JournalRetention::Release)
    }

    /// Orders writes recorded in the undo journal before calling `publish`.
    ///
    /// Retains undo entries through `publish`. Restores appended entries if `apply` or its
    /// completion barrier fails, or if `publish` fails before a publication attempt.
    /// Otherwise releases these entries without restoring
    /// the writes, including after uncertain publication. Preserves earlier journal entries.
    /// The caller establishes publication and allocation permissions and handles uncertainty.
    ///
    /// # Errors
    ///
    /// Returns errors from `apply`, the completion barrier, or `publish`. If restoration also
    /// fails, returns [`Error::CommitPartiallyApplied`] with both errors.
    pub(in crate::edit) fn with_publication<T>(
        engine: &'engine mut WriteEngine,
        apply: impl FnOnce(&mut Self) -> Result<(), Error>,
        publish: impl FnOnce(&mut WriteEngine) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let checkpoint = engine.inplace_undo.len();
        let mut scope = Self { engine, checkpoint };
        let result = apply(&mut scope);
        scope.complete(result, JournalRetention::Keep)?;
        match publish(scope.engine) {
            Err(error) if !scope.engine.publication.attempted() => Err(scope.restore(error)),
            result => {
                scope.engine.inplace_undo.truncate(checkpoint);
                result
            }
        }
    }

    fn run_with_retention(
        engine: &'engine mut WriteEngine,
        apply: impl FnOnce(&mut Self) -> Result<(), Error>,
        retention: JournalRetention,
    ) -> Result<(), Error> {
        let checkpoint = engine.inplace_undo.len();
        let mut scope = Self { engine, checkpoint };
        let result = apply(&mut scope);
        scope.complete(result, retention)
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
        self.engine.write_inplace_journaled(at.get(), bytes)
    }

    /// Orders successful writes or restores failed writes after the saved checkpoint.
    ///
    /// Releases or retains successful entries according to `retention`. Preserves earlier
    /// entries and skips barriers when no new entries were appended.
    ///
    /// # Errors
    ///
    /// Returns the operation, barrier, or restoration errors described by [`Self::run`].
    fn complete(
        &mut self,
        result: Result<(), Error>,
        retention: JournalRetention,
    ) -> Result<(), Error> {
        let Self { engine, checkpoint } = self;
        if engine.inplace_undo.len() == *checkpoint {
            return result;
        }
        let result = result.and_then(|()| engine.barrier());
        match result {
            Ok(()) => {
                if matches!(retention, JournalRetention::Release) {
                    engine.inplace_undo.truncate(*checkpoint);
                }
                Ok(())
            }
            Err(refusal) => Err(self.restore(refusal)),
        }
    }

    fn restore(&mut self, refusal: Error) -> Error {
        match self.engine.undo_inplace_since(self.checkpoint) {
            Ok(()) => refusal,
            Err(restore) => Error::CommitPartiallyApplied {
                refusal: Box::new(refusal),
                restore: Box::new(restore),
            },
        }
    }
}

/// Whether successful operation completion retains appended undo entries.
enum JournalRetention {
    Keep,
    Release,
}

#[cfg(test)]
mod tests;
