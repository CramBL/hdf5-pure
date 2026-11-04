//! Owned count inventories, source preflight, and positive explicit count batches.

use core::num::NonZeroU32;
use std::sync::Arc;

use hdf5_pure_format::__private::ObjectReferenceCountField;

use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::counts::HeaderCountAllocation;
use crate::edit::analysis::deletions::TopologyIdentity;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;

use super::SourceRemovalPlan;

pub(in crate::edit) use inventory::CountPreflight;
pub(in crate::edit) use inventory::CountProjectionIdentity;

mod inventory;
mod protection;

impl SourceRemovalPlan<'_, '_> {
    /// Copies known positive declared remainders into an owned batch.
    ///
    /// Checks the engine identity on every call. For a nonempty batch, records the allocation
    /// session, phase identity, and root resolved from the engine's superblock context. Omits zero
    /// and unknown remainders, retaining no source association for an empty batch. Uses existing
    /// plan metadata without reading the source or discovering topology.
    ///
    /// # Errors
    ///
    /// Returns [`CountHandoffError::Engine`] for a different engine, [`CountHandoffError::Root`]
    /// if a nonempty batch's topology root differs from the engine's root, or
    /// [`CountHandoffError::Source`] for root-address errors or a positive changed count without
    /// an explicit source field.
    pub(in crate::edit) fn surviving_counts(
        &self,
        engine: &WriteEngine,
    ) -> Result<SurvivingCounts, CountHandoffError> {
        let topology = self.plan.topology;
        if !std::ptr::eq(topology.engine, engine) {
            return Err(CountHandoffError::Engine);
        }
        let mut changes = self
            .changes()
            .filter_map(|change| NonZeroU32::new(change.remaining()).map(|count| (change, count)))
            .peekable();
        if changes.peek().is_none() {
            return Ok(SurvivingCounts {
                source: None,
                updates: Vec::new(),
            });
        }
        let root = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(CountHandoffError::Source)?;
        let expected = topology.graph.object(ObjectIndex::ROOT).value.id;
        if root != expected {
            return Err(CountHandoffError::Root {
                expected,
                actual: root,
            });
        }
        let updates = changes
            .map(|(change, count)| {
                let bound = change.bound();
                let field = bound.field().ok_or_else(|| {
                    CountHandoffError::Source(FormatError::Internal(
                        "a positive changed count has no explicit source field".into(),
                    ))
                })?;
                Ok(SurvivingCount {
                    object: bound.object(),
                    field,
                    count,
                    allocation: bound.header_allocation(),
                })
            })
            .collect::<Result<Vec<_>, CountHandoffError>>()?;
        Ok(SurvivingCounts {
            source: Some(CountSource {
                session: engine.context.allocations.session(),
                identity: Arc::clone(&topology.identity),
                root,
            }),
            updates,
        })
    }
}

/// Owned original fields and positive declared remainders from one source phase.
///
/// The caller may drop the borrowed plan and phase while retaining this batch. The original
/// observed count and any independent header allocation fact are preserved for each entry.
/// Source freshness, storage ownership, and permission to write require separate evidence.
pub(in crate::edit) struct SurvivingCounts {
    source: Option<CountSource>,
    updates: Vec<SurvivingCount>,
}

impl SurvivingCounts {
    /// Returns whether this batch is empty or its recorded allocation session matches `engine`.
    ///
    /// Session equality is independent of source freshness.
    pub(in crate::edit) fn same_session(&self, engine: &WriteEngine) -> bool {
        self.source
            .as_ref()
            .is_none_or(|source| source.session.same(&engine.context.allocations.session()))
    }

    /// Returns whether a nonempty batch and `source` share the same phase identity.
    ///
    /// Compares phase identity independently of the physical object.
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        self.source
            .as_ref()
            .is_some_and(|origin| Arc::ptr_eq(&origin.identity, &source.identity))
    }

    /// Returns the original physical topology root, or `None` for an empty batch.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.source.as_ref().map(|source| source.root)
    }

    /// Borrows positive count entries in the order established by the source plan.
    pub(in crate::edit) fn updates(&self) -> &[SurvivingCount] {
        &self.updates
    }
}

/// An original explicit field paired with its physical object and positive planned remainder.
///
/// Optional allocation geometry describes only the complete chunk containing that field.
pub(in crate::edit) struct SurvivingCount {
    object: ObjectId,
    field: ObjectReferenceCountField,
    count: NonZeroU32,
    allocation: Option<HeaderCountAllocation>,
}

impl SurvivingCount {
    /// Returns the physical identity of the original source object.
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }

    /// Returns the original observed count with its physical field and complete chunk associations.
    pub(in crate::edit) fn field(&self) -> ObjectReferenceCountField {
        self.field
    }

    /// Returns the positive declared count after the planned removals and any supplied additions.
    pub(in crate::edit) fn count(&self) -> NonZeroU32 {
        self.count
    }

    /// Returns the original checked complete header allocation, if available.
    pub(in crate::edit) fn header_allocation(&self) -> Option<HeaderCountAllocation> {
        self.allocation
    }
}

/// An engine, root, or field inconsistency while copying count evidence.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum CountHandoffError {
    /// A supplied engine different from the plan's borrowed engine.
    Engine,
    /// A nonempty handoff whose topology root differs from the engine's superblock root.
    Root {
        /// The physical root of the removal topology.
        expected: ObjectId,
        /// The physical root resolved from the engine's superblock context.
        actual: ObjectId,
    },
    /// A root-address error or a positive changed count without an explicit source field.
    Source(FormatError),
}

/// The allocation session, phase identity, and physical root of a nonempty batch.
struct CountSource {
    session: AllocationSession,
    identity: Arc<TopologyIdentity>,
    root: ObjectId,
}

#[cfg(test)]
mod tests;
