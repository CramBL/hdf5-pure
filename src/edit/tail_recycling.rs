//! Recycling of persistent extension and manager storage after publication.

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::ReturnedExtents;

use crate::address::StoredAddress;
use crate::error::{Error, FormatError};
use crate::file_space_info::FileSpaceStrategy;

use super::FreeClass;
use super::FreeExtent;
use super::PersistState;
use super::TailPlacement;
use super::WriteEngine;
use super::free_snapshot::{FlatProjection, FlatTailLayout};

impl WriteEngine {
    /// Rewrites persistent metadata in a reserved allocation after publishing its replacement.
    ///
    /// The caller must first publish with [`TailPlacement::Preserve`], leaving `target`
    /// unreferenced and excluded from ordinary free space. Surrenders unused aggregate windows
    /// through ordinary threshold admission. Allocates elsewhere if the metadata outgrows
    /// `target`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`], [`Error::EditUnsupported`], or [`Error::Io`] from
    /// [`Self::commit_persisting`] if rewriting or publishing the replacement metadata fails.
    pub(super) fn recycle_flat_tail(
        &mut self,
        root: PhysicalAddress,
        mut to_free: ReturnedExtents,
        target: FreeExtent,
    ) -> Result<(), Error> {
        let spare = self
            .aggregation
            .map(|state| state.unused().collect::<Vec<_>>())
            .unwrap_or_default();
        if let Some(aggregation) = self.aggregation {
            self.aggregation = Some(aggregation.without_spare());
        }
        for extent in spare {
            to_free
                .insert(extent, FreeClass::Dead)
                .map_err(|error| FormatError::Internal(error.to_string()))?;
        }
        self.commit_persisting(root, to_free, TailPlacement::Recycle(target))
    }

    /// Returns a contiguous persistent metadata allocation below the free-section threshold.
    ///
    /// Requires [`FileSpaceStrategy::FsmAggr`] and contiguous extension and manager blocks.
    /// The caller reserves the returned extent during publication before recycling it.
    pub(super) fn recyclable_flat_tail(&self) -> Option<FreeExtent> {
        let PersistState {
            strategy,
            threshold,
            page_size: _,
            old_blocks,
        } = self.persist.as_ref()?;
        if *strategy != FileSpaceStrategy::FsmAggr {
            return None;
        }
        let len = old_blocks
            .iter()
            .try_fold(0u64, |len, extent| len.checked_add(extent.len()))?;
        if len >= *threshold {
            return None;
        }
        let mut blocks = old_blocks.clone();
        blocks.sort_unstable_by_key(|extent| extent.start());
        let start = blocks.first()?.start();
        let end = blocks.iter().try_fold(start, |end, extent| {
            (extent.start() == end).then_some(extent.end())
        })?;
        FreeExtent::between(start, end)
    }

    /// Computes the free sections, placement, length, and EOA for a reserved metadata allocation.
    ///
    /// Returns `None` if the replacement exceeds `target` or its size overflows. With tracked
    /// sections, retains the full allocation, with surplus capacity allocated to the section
    /// list. With no sections, retains only the extension's length and shortens EOA when the
    /// allocation ends there.
    pub(super) fn recycled_tail_layout(
        &self,
        eoa: StoredAddress,
        to_free: &ReturnedExtents,
        old_blocks: &[FreeExtent],
        extension_len: u64,
        target: FreeExtent,
    ) -> Option<FlatTailLayout> {
        let (post, mut eoa) = self.flat_post_free(eoa, to_free, old_blocks);
        let projection = FlatProjection::new(post, self.context.widths()).ok()?;
        let len = extension_len.checked_add(projection.managers().encoded_len())?;
        if len > target.len() {
            return None;
        }
        let len = if projection.is_empty() {
            len
        } else {
            target.len()
        };
        if eoa == target.end() {
            eoa = target.start().checked_add(len).ok()?;
        }
        Some(projection.place(Some(target.start()), len, eoa))
    }
}
