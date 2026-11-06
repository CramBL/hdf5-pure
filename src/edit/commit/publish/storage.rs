//! Original storage exclusions for one root or persistent-manager publication.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::{Extent, ReturnedExtents};

use crate::edit::commit::plan::inventory::storage::{OriginalStorage, SpaceClaim, StorageConflict};
use crate::edit::{TailPlacement, WriteEngine};
use crate::error::{Error, FormatError};

/// A storage exclusion conflict or an error from the publication operation.
#[derive(Debug)]
pub(in crate::edit::commit) enum PublicationError {
    /// An allocator, retirement, metadata-placement, or end conflict.
    Storage(StorageConflict),
    /// A planning, serialization, I/O, or publication error.
    Write(Error),
}

impl PublicationError {
    /// Converts an unexpected storage conflict into an internal format error.
    ///
    /// The caller uses this conversion only for publication without storage exclusions.
    pub(super) fn without_storage(self) -> Error {
        match self {
            Self::Storage(conflict) => FormatError::Internal(format!(
                "an unconstrained publication reports a storage conflict: {conflict:?}"
            ))
            .into(),
            Self::Write(error) => error,
        }
    }
}

impl From<Error> for PublicationError {
    fn from(error: Error) -> Self {
        Self::Write(error)
    }
}

impl From<FormatError> for PublicationError {
    fn from(error: FormatError) -> Self {
        Self::Write(error.into())
    }
}

impl From<StorageConflict> for PublicationError {
    fn from(error: StorageConflict) -> Self {
        Self::Storage(error)
    }
}

impl OriginalStorage {
    /// Checks allocator space, pending frees, old managers, and any recycled tail target.
    ///
    /// Checks the complete target for [`TailPlacement::Recycle`], including any unused suffix.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::Overlap`] if a checked range overlaps protected storage.
    /// Returns [`StorageConflict::Source`] if an address or range cannot be resolved.
    pub(super) fn exclude_publication(
        &self,
        engine: &WriteEngine,
        returned: &ReturnedExtents,
        placement: TailPlacement,
    ) -> Result<(), StorageConflict> {
        self.exclude_space(engine)?;
        for &(extent, class) in returned.entries() {
            self.exclude_extent(engine, extent, SpaceClaim::Retirement(class))?;
        }
        if let Some(persist) = &engine.persist {
            for &extent in &persist.old_blocks {
                self.exclude_extent(engine, extent, SpaceClaim::OldManager)?;
            }
        }
        if let TailPlacement::Recycle(target) = placement {
            self.exclude_extent(engine, target, SpaceClaim::Recycle)?;
        }
        Ok(())
    }

    /// Resolves a stored extent before checking its complete physical range.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::Overlap`] with `claim` if the extent overlaps protected storage.
    /// Returns [`StorageConflict::Source`] if the address or range cannot be resolved.
    pub(super) fn exclude_extent(
        &self,
        engine: &WriteEngine,
        extent: Extent,
        claim: SpaceClaim,
    ) -> Result<(), StorageConflict> {
        let range = engine
            .addresses()
            .resolve(extent.start())
            .and_then(|start| FileRange::new(start, extent.len().get()))
            .map_err(|error| StorageConflict::Source(error.into()))?;
        self.exclude(range, claim)
    }
}
