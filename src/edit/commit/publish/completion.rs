//! Engine loans for completion work after tree publication.

use hdf5_pure_core::PhysicalAddress;

use crate::edit::WriteEngine;
use crate::error::FormatError;

/// A mutable loan of the engine after a successful tree publication.
///
/// The publisher creates this loan after stored-reference updates, root reclamation, and
/// requested truncation succeed. The root and EOA are the final adopted physical addresses.
/// Root reclamation may relocate the root from the initially requested replacement.
/// The caller must establish source freshness and allocation ownership separately.
pub(in crate::edit) struct PublishedTree<'engine> {
    engine: &'engine mut WriteEngine,
    root: PhysicalAddress,
    end: PhysicalAddress,
}

impl<'engine> PublishedTree<'engine> {
    /// Borrows the engine and records its adopted root and EOA.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the stored root address plus the effective
    /// base exceeds [`u64::MAX`].
    pub(super) fn new(engine: &'engine mut WriteEngine) -> Result<Self, FormatError> {
        let root = engine
            .context
            .addresses()
            .resolve(engine.context.superblock().root_group_address)?;
        let end = PhysicalAddress::new(engine.context.superblock().eof_address);
        Ok(Self { engine, root, end })
    }

    /// Returns the borrowed engine for completion work.
    pub(in crate::edit) fn into_engine(self) -> &'engine mut WriteEngine {
        let Self {
            engine,
            root: _,
            end: _,
        } = self;
        engine
    }

    /// Returns the physical root adopted before the completion callback.
    pub(in crate::edit) fn root(&self) -> PhysicalAddress {
        self.root
    }

    /// Returns the physical EOA adopted before the completion callback.
    pub(in crate::edit) fn end(&self) -> PhysicalAddress {
        self.end
    }
}
