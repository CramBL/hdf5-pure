//! Storage-ownership inspection through a file's private editing backend.

use crate::edit::SpaceOwnership;
use crate::error::Error;
use crate::reader::Backend;
use crate::reader::File;

/// Analyzes committed storage through the file's editing session.
///
/// # Errors
///
/// Returns [`Error::ReadOnly`] if the file has no editing session, or the error from
/// [`WriteEngine::space_ownership`](crate::edit::WriteEngine::space_ownership).
///
/// # Panics
///
/// Panics if the editing-session mutex is poisoned.
pub(crate) fn ownership(file: &File) -> Result<SpaceOwnership, Error> {
    let Backend::Edit(engine) = &file.inner.backend else {
        return Err(Error::ReadOnly);
    };
    engine.lock().unwrap().space_ownership()
}
