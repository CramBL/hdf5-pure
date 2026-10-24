//! Reuse of a superseded root header after publication.

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::error::Error;
use crate::file_space_info::FileSpaceStrategy;

use super::FreeClass;
use super::FreeExtent;
use super::WriteEngine;

/// Holds equal-sized root-header extents for a post-publication copy.
pub(super) struct RootReclamation {
    /// The superseded extent withheld from ordinary free-section admission.
    target: FreeExtent,
    /// The replacement extent released after publishing the copied header.
    source: FreeExtent,
}

impl WriteEngine {
    /// Reserves a sub-threshold root header for reuse after publishing its replacement.
    ///
    /// Returns `None` unless both headers occupy one chunk of the same size, the replacement
    /// is at a higher address, and the old root has no incoming hard link. Reuse requires
    /// persistent [`FileSpaceStrategy::FsmAggr`] and a prospective tree proved free of object
    /// references. Removes the reserved extent from `to_free`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address resolution fails. Returns errors from
    /// [`Self::oh_chunk_spans`] when reading the replacement header and from
    /// [`prove_free_of_references`](crate::reference_patch::prove_free_of_references) when
    /// inspecting the prospective tree.
    pub(super) fn plan_root_reclamation(
        &self,
        new_root: u64,
        to_free: &mut Vec<(StoredAddress, u64, FreeClass)>,
    ) -> Result<Option<RootReclamation>, Error> {
        let Some(persist) = &self.persist else {
            return Ok(None);
        };
        if persist.strategy != FileSpaceStrategy::FsmAggr {
            return Ok(None);
        }
        if !to_free.iter().any(|&(addr, len, _)| {
            addr == self.superblock.root_group_address && len < persist.threshold
        }) {
            return Ok(None);
        }
        let old_root = self
            .addresses
            .absolute(self.superblock.root_group_address)?;
        if old_root >= new_root {
            return Ok(None);
        }
        let Ok(old_spans) = self.oh_chunk_spans(old_root) else {
            return Ok(None);
        };
        let [(target_addr, target_len)] = old_spans.as_slice() else {
            return Ok(None);
        };
        if *target_len >= persist.threshold {
            return Ok(None);
        }
        let new_spans = self.oh_chunk_spans(new_root)?;
        let [(source_addr, source_len)] = new_spans.as_slice() else {
            return Ok(None);
        };
        if source_len != target_len {
            return Ok(None);
        }
        let Some(incoming) = self.count_incoming_hard_links() else {
            return Ok(None);
        };
        if incoming.contains_key(&old_root) {
            return Ok(None);
        }
        let Some(target) = self.stored_extent(*target_addr, *target_len) else {
            return Ok(None);
        };
        let Some(source) = self.stored_extent(*source_addr, *source_len) else {
            return Ok(None);
        };
        let Some(index) = to_free
            .iter()
            .position(|&(addr, len, _)| addr.get() == target.start() && len == target.len())
        else {
            return Ok(None);
        };
        if !self.proved_free_of_references {
            let mut prospective = self.superblock.clone();
            prospective.root_group_address = self.addresses.relative(new_root)?;
            if !crate::reference_patch::prove_free_of_references(
                &self.image(),
                &prospective,
                self.widths,
                self.addresses,
                super::MAX_LINK_GRAPH_NODES,
            )? {
                return Ok(None);
            }
        }
        to_free.remove(index);
        Ok(Some(RootReclamation { target, source }))
    }

    /// Copies a committed root into its reserved extent and publishes that location.
    ///
    /// The caller must first publish the replacement root. The superseded extent is then
    /// unreferenced and available for the copy. The replacement extent is released through
    /// ordinary threshold admission after the second publication.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if reading, address resolution, or metadata serialization
    /// fails. Returns [`Error::Io`] if writing, synchronizing, or truncating fails.
    pub(super) fn reclaim_root_metadata(
        &mut self,
        reclamation: RootReclamation,
    ) -> Result<(), Error> {
        let RootReclamation { target, source } = reclamation;
        let source_addr = self
            .addresses
            .absolute(StoredAddress::new(source.start()))?;
        let target_addr = self
            .addresses
            .absolute(StoredAddress::new(target.start()))?;
        let bytes = self
            .image
            .read_exact_at(source_addr, source.len().to_usize()?)?;
        self.write_at(target_addr, &bytes)?;
        self.commit_persisting_releasing_reserve(
            target_addr,
            vec![(
                StoredAddress::new(source.start()),
                source.len(),
                FreeClass::Dead,
            )],
        )
    }
}

#[cfg(test)]
mod tests;
