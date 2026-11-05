//! Owned allocation batches, collection candidate sources, and retained reference roots.

use std::collections::BTreeMap;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use super::NetCountPlan;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::CollectionCandidates;
use crate::edit::analysis::ReconstructionSource;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

impl NetCountPlan<'_, '_> {
    /// Collects supported allocations from eligible objects and retains the other reference roots.
    ///
    /// Enumerates each zero-count object's storage once through the existing ownership collector.
    /// Preserves original physical allocation boundaries and free classes, including independently
    /// proven storage when another part of an object's ownership is unavailable.
    ///
    /// Retains touched removal-view objects ineligible for collection, followed by referenced
    /// targets outside that view. Retains eligible object identities for collection-candidate
    /// extraction in the original source phase. Uses the existing plan without topology discovery.
    /// An empty plan requires no source reads or heap allocation.
    pub(in crate::edit) fn collect_storage(&self) -> CountStorage {
        let Self {
            engine,
            source,
            removals,
            referenced,
        } = self;
        let mut allocations = Vec::new();
        if let Some(removals) = removals {
            removals.collect_storage(&mut allocations);
        }
        let reference_roots = removals
            .iter()
            .flat_map(|removals| removals.retained_objects())
            .chain(referenced.keys().copied())
            .collect();
        let collection_objects = removals
            .iter()
            .flat_map(|removals| removals.counted_zero_objects())
            .map(|object| object.id())
            .collect();
        CountStorage {
            session: engine.context.allocations.session(),
            origin: source.as_ref().map(|source| ReconstructionSource {
                identity: Arc::clone(&source.identity),
                object: source.object(),
            }),
            allocations,
            reference_roots,
            collection_objects,
        }
    }
}

/// Original allocation facts and collection-reference sources from one count plan.
///
/// Nonempty plans retain their original phase identity, physical root, and allocation session.
/// Allocation boundaries and free classes are preserved. Source freshness, reference completeness,
/// admission, and publication permissions require separate evidence.
pub(in crate::edit) struct CountStorage {
    session: AllocationSession,
    origin: Option<ReconstructionSource>,
    allocations: Vec<(FileRange, FreeClass)>,
    reference_roots: Vec<ObjectId>,
    /// Eligible objects whose references have not been collected as candidates.
    collection_objects: Vec<ObjectId>,
}

impl CountStorage {
    /// Extracts collection candidates for eligible objects in the original source phase.
    ///
    /// Consumes the recorded object identities after checking associations. Reuses the phase's
    /// reference and allocation summaries, preserving the original allocation batch. Returns an
    /// empty union for failed or incomplete reference interpretation, and excludes collections
    /// whose allocation geometry cannot be parsed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the allocation session or recorded phase identity
    /// differs from `phase`.
    pub(in crate::edit) fn take_collections(
        &mut self,
        phase: &mut AnalysisPhase<'_>,
    ) -> Result<CollectionCandidates, FormatError> {
        if !self.same_session(phase.engine)
            || self.origin.as_ref().is_some_and(|origin| {
                !phase
                    .identity
                    .as_ref()
                    .is_some_and(|identity| Arc::ptr_eq(identity, &origin.identity))
            })
        {
            return Err(FormatError::Internal(
                "count collection candidates belong to another source phase".into(),
            ));
        }
        Ok(phase.object_collections(core::mem::take(&mut self.collection_objects)))
    }

    /// Creates fresh collection-reference analysis for the prospective root and retained objects.
    ///
    /// Uses `root` for the original root and `relocations` for other retained headers. The caller
    /// supplies physical addresses after relevant replacement and value writes. Reference
    /// completeness is established within the returned phase. Session equality is independent of
    /// source freshness.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `engine` belongs to another allocation session.
    pub(in crate::edit) fn reference_phase<'a>(
        &self,
        engine: &'a WriteEngine,
        root: PhysicalAddress,
        relocations: &BTreeMap<u64, u64>,
    ) -> Result<AnalysisPhase<'a>, FormatError> {
        if !self.same_session(engine) {
            return Err(FormatError::Internal(
                "retained collection roots belong to another allocation session".into(),
            ));
        }
        Ok(AnalysisPhase::prospective_retaining(
            engine,
            root,
            self.reference_roots.iter().map(|object| {
                if Some(*object) == self.root() {
                    root.get()
                } else {
                    let address = object.header_address().get();
                    relocations.get(&address).copied().unwrap_or(address)
                }
            }),
        ))
    }

    /// Returns whether the recorded allocation session matches `engine`, even for empty input.
    ///
    /// Session equality is independent of source freshness.
    pub(in crate::edit) fn same_session(&self, engine: &WriteEngine) -> bool {
        self.session.same(&engine.context.allocations.session())
    }

    /// Returns whether `source` has the same shared phase identity as this batch's origin.
    ///
    /// Returns `false` without a phase origin. Object identity and source freshness are separate
    /// conditions.
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        self.origin
            .as_ref()
            .is_some_and(|origin| Arc::ptr_eq(&origin.identity, &source.identity))
    }

    /// Returns the original physical root, or `None` for an empty plan.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.origin.as_ref().map(ReconstructionSource::object)
    }

    /// Borrows complete physical allocations in object propagation and collector order.
    ///
    /// The allocation set can be incomplete when ownership of other storage is unproven. Original
    /// boundaries and free classes are preserved without sorting or coalescing.
    pub(in crate::edit) fn allocations(&self) -> &[(FileRange, FreeClass)] {
        &self.allocations
    }

    /// Borrows each retained object once for reference analysis.
    ///
    /// Touched removal-view objects are in topology index order, followed by other reference
    /// targets in physical object-header address order. Includes targets outside the topology.
    pub(in crate::edit) fn reference_roots(&self) -> &[ObjectId] {
        &self.reference_roots
    }
}

#[cfg(test)]
mod tests;
