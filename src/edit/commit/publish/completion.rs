//! Engine loans for completion work after tree publication.

use hdf5_pure_core::PhysicalAddress;

use crate::edit::WriteEngine;
use crate::edit::commit::{OriginalStorage, StorageConflict};
use crate::error::FormatError;

/// A mutable loan of the engine after a successful tree publication.
///
/// The publisher creates this loan after stored-reference updates, root reclamation, and
/// requested truncation succeed. Records the replacement root passed to the publisher separately
/// from the final adopted physical root and EOA. Root reclamation may relocate the root.
/// The caller must establish source freshness and allocation ownership separately.
pub(in crate::edit) struct PublishedTree<'engine> {
    engine: &'engine mut WriteEngine,
    requested_root: PhysicalAddress,
    root: PhysicalAddress,
    end: PhysicalAddress,
}

impl<'engine> PublishedTree<'engine> {
    /// Borrows the engine and records its adopted root and EOA alongside `requested_root`.
    ///
    /// The caller completes publication before constructing this loan and passes its requested
    /// physical root.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the stored root address plus the effective
    /// base exceeds [`u64::MAX`].
    pub(in crate::edit) fn new(
        engine: &'engine mut WriteEngine,
        requested_root: PhysicalAddress,
    ) -> Result<Self, FormatError> {
        let root = engine
            .context
            .addresses()
            .resolve(engine.context.superblock().root_group_address)?;
        let end = PhysicalAddress::new(engine.context.superblock().eof_address);
        Ok(Self {
            engine,
            requested_root,
            root,
            end,
        })
    }

    /// Returns the borrowed engine for completion work.
    pub(in crate::edit) fn into_engine(self) -> &'engine mut WriteEngine {
        let Self {
            engine,
            requested_root: _,
            root: _,
            end: _,
        } = self;
        engine
    }

    /// Returns the physical replacement root requested before any root reclamation.
    pub(in crate::edit) fn requested_root(&self) -> PhysicalAddress {
        self.requested_root
    }

    /// Returns the physical root adopted before the completion callback.
    pub(in crate::edit) fn root(&self) -> PhysicalAddress {
        self.root
    }

    /// Returns the physical EOA adopted before the completion callback.
    pub(in crate::edit) fn end(&self) -> PhysicalAddress {
        self.end
    }
    /// Checks final allocator ranges and both file ends against protected storage.
    ///
    /// Checks the adopted EOA and physical file length separately.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::Overlap`] or [`StorageConflict::Source`] for allocator conflicts
    /// or range-resolution errors. Returns [`StorageConflict::End`] if either end is below the
    /// greatest protected range end.
    pub(super) fn check_storage(&self, storage: &OriginalStorage) -> Result<(), StorageConflict> {
        storage.exclude_space(self.engine)?;
        storage.check_end(self.end)?;
        storage.check_end(PhysicalAddress::new(self.engine.image.len()))
    }
}
