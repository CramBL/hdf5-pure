//! Borrowed reusable-space queries and owned proposed-state manager projections.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::{AdmittedFreeSnapshot, FreeList};

use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::free_sections::ReusableSpaceStats;
use crate::free_space_manager::ManagerProjection;
use crate::width::FormatWidths;

use super::WriteEngine;
use super::aggregation::AggregateSpace;

/// Reusable space and physical file geometry borrowed from one editing session.
///
/// The engine remains borrowed while the snapshot or its iterators are used. The caller limits
/// queries to a phase with unchanged source content.
pub(super) struct FreeSpaceSnapshot<'a> {
    engine: &'a WriteEngine,
    admitted: AdmittedFreeSnapshot<'a>,
}

impl<'a> FreeSpaceSnapshot<'a> {
    pub(super) fn new(engine: &'a WriteEngine) -> Self {
        Self {
            engine,
            admitted: engine.space.admitted_snapshot(),
        }
    }

    /// Returns file-image length and reusable bytes, including unused aggregate space.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the reusable byte sum exceeds [`u64::MAX`].
    pub(super) fn stats(&self) -> Result<ReusableSpaceStats, FormatError> {
        let Self { engine, admitted } = self;
        let mut reusable_free_bytes = admitted.reusable_bytes()?;
        if let Some(aggregation) = engine.aggregation {
            let unused = aggregation.unused_bytes()?;
            reusable_free_bytes =
                reusable_free_bytes
                    .checked_add(unused)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: reusable_free_bytes,
                        length: unused,
                    })?;
        }
        Ok(ReusableSpaceStats::new(
            engine.image.len(),
            reusable_free_bytes,
        ))
    }

    /// Returns reusable physical ranges, including unused aggregate space.
    ///
    /// Resolves stored addresses through the session's address context.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] in an iterator item if resolving a stored address
    /// or computing its physical end exceeds [`u64::MAX`].
    pub(super) fn ranges(self) -> impl Iterator<Item = Result<FileRange, FormatError>> + 'a {
        let Self { engine, admitted } = self;
        let aggregate_only = engine.aggregation.filter(|_| !admitted.is_enabled());
        let managers_only = engine.aggregation.is_none();
        let combined = aggregate_only.is_none() && !managers_only;
        aggregate_only
            .into_iter()
            .flat_map(AggregateSpace::unused)
            .chain(
                managers_only
                    .then(|| admitted.reusable_sections())
                    .into_iter()
                    .flatten(),
            )
            .chain(
                combined
                    .then(|| {
                        admitted.reusable_sections_with_raw(
                            engine
                                .aggregation
                                .into_iter()
                                .flat_map(AggregateSpace::unused),
                        )
                    })
                    .into_iter()
                    .flatten(),
            )
            .map(|extent| {
                FileRange::new(
                    engine.context.addresses().resolve(extent.start())?,
                    extent.len(),
                )
            })
    }
}

/// A proposed flat free list and its measured manager projection.
///
/// The proposed state remains private until publication permits adoption.
pub(super) struct FlatProjection {
    post: FreeList,
    managers: ManagerProjection,
}

impl FlatProjection {
    /// Measures manager geometry for a proposed flat state.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the combined manager length exceeds [`u64::MAX`].
    pub(super) fn new(post: FreeList, widths: FormatWidths) -> Result<Self, FormatError> {
        let managers = ManagerProjection::new(AdmittedFreeSnapshot::flat(&post), widths)?;
        Ok(Self { post, managers })
    }

    /// Returns the manager projection measured from this proposed state.
    pub(super) fn managers(&self) -> &ManagerProjection {
        &self.managers
    }

    pub(super) fn is_empty(&self) -> bool {
        self.post.is_empty()
    }

    /// Associates the proposed state and projection with a stored tail placement.
    ///
    /// The caller establishes allocation ownership and capacity. A `placed_at` of [`None`] denotes
    /// append placement.
    pub(super) fn place(
        self,
        placed_at: Option<StoredAddress>,
        length: u64,
        eoa: StoredAddress,
    ) -> FlatTailLayout {
        let Self { post, managers } = self;
        FlatTailLayout {
            post,
            managers,
            placed_at,
            length,
            eoa,
        }
    }
}

/// A proposed flat state and measured managers with their stored tail placement.
pub(super) struct FlatTailLayout {
    post: FreeList,
    managers: ManagerProjection,
    placed_at: Option<StoredAddress>,
    length: u64,
    eoa: StoredAddress,
}

impl FlatTailLayout {
    /// Returns the proposed state, projection, placement, owned length, and EOA.
    pub(super) fn into_parts(
        self,
    ) -> (
        FreeList,
        ManagerProjection,
        Option<StoredAddress>,
        u64,
        StoredAddress,
    ) {
        let Self {
            post,
            managers,
            placed_at,
            length,
            eoa,
        } = self;
        (post, managers, placed_at, length, eoa)
    }
}
