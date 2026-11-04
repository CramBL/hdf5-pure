//! Planned graph remainders and declared net counts with source-bound propagation criteria.
//!
//! Only ordinary non-root objects with matching evidence and both remainders at zero are queued.
//! Count changes remain planned values. Storage ownership, source freshness, references, and
//! publication require separate proofs.

use std::sync::Arc;

use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::counts::CountEvidence;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::edit::analysis::counts::ObjectReferenceBound;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;

use super::super::PlannedDeletion;
use super::CompleteTopology;
use super::LinkRemovalPlan;
use super::RemovedLinks;

mod added;
mod surviving;

impl<'source> AnalysisPhase<'source> {
    /// Plans removals using both visible graph counts and matching declared-count evidence.
    ///
    /// Returns `None` without complete topology, including for an empty batch. Target-only
    /// unresolved deletions do not decrement counts. Queries source evidence only for objects
    /// touched by actual decrements. Outgoing propagation requires an ordinary subject and zero
    /// graph and declared remainders, preserving positive excess and unknown subjects.
    ///
    /// # Errors
    ///
    /// Returns the contradiction and source errors of [`AnalysisPhase::count_evidence`], or
    /// [`CountEvidenceError::Topology`] for inconsistent deletion, topology, object, or count
    /// associations and checked-decrement failures.
    pub(in crate::edit) fn source_count_plan<'deletion>(
        &mut self,
        deletions: impl IntoIterator<Item = &'deletion PlannedDeletion>,
    ) -> Result<Option<SourceRemovalPlan<'_, 'source>>, CountEvidenceError> {
        let (removed, mut queries) = self.removal_queries(deletions)?;
        SourceRemovalPlan::new(removed, |object| queries.evidence(object))
    }
}

/// Planned graph remainders and declared net counts with checked propagation eligibility.
///
/// Original source counts, fields, and topology remain unchanged. Only recognized ordinary
/// non-root objects with zero graph and declared remainders are eligible. Allocation and
/// publication decisions remain the caller's responsibility.
/// Declared counts include prospective additions when supplied.
pub(in crate::edit) struct SourceRemovalPlan<'topology, 'source> {
    plan: LinkRemovalPlan<'topology, 'source, SourceCountState<'source>>,
}

impl<'topology, 'source> SourceRemovalPlan<'topology, 'source> {
    /// Applies source-bound criteria to the shared traversal of an already bound deletion batch.
    ///
    /// Calls `query` once per touched object and checks its evidence against the same source,
    /// topology, physical object, and original incoming count. Returns `None` without topology.
    ///
    /// # Errors
    ///
    /// Returns errors from `query`, or [`CountEvidenceError::Topology`] for inconsistent evidence,
    /// missing incoming counts, or checked-decrement failures.
    fn new(
        removed: RemovedLinks<'topology, 'source>,
        mut query: impl FnMut(ObjectId) -> Result<Option<CountEvidence<'source>>, CountEvidenceError>,
    ) -> Result<Option<Self>, CountEvidenceError> {
        let RemovedLinks {
            topology,
            edges,
            unresolved: _,
        } = removed;
        topology
            .map(|topology| {
                LinkRemovalPlan::build(
                    topology,
                    edges,
                    |index| SourceCountState::new(topology, index, &mut query),
                    SourceCountState::remove,
                )
                .map(|plan| Self { plan })
            })
            .transpose()
    }

    /// Returns whether the reconstruction origin has this plan's shared phase identity.
    ///
    /// Object identity, source freshness, and allocation ownership are separate conditions.
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        Arc::ptr_eq(&self.plan.topology.identity, &source.identity)
    }

    /// Returns the planned graph remainder, or the original graph count for an untouched object.
    ///
    /// Returns `None` outside the topology. An untouched object without an incoming-count entry
    /// has a count of zero.
    pub(in crate::edit) fn graph_remaining(&self, object: ObjectId) -> Option<u32> {
        let LinkRemovalPlan {
            topology,
            remaining,
            zero: _,
            #[cfg(test)]
                work: _,
        } = &self.plan;
        let &index = topology.indices.get(&object)?;
        Some(remaining.get(&index).map_or_else(
            || {
                topology
                    .incoming(object.header_address().get())
                    .unwrap_or(0)
            },
            |state| state.graph,
        ))
    }

    /// Returns the planned declared net count for a touched object with count evidence.
    ///
    /// Returns `None` for untouched or outside objects, or when no count evidence was supplied.
    pub(in crate::edit) fn declared_remaining(&self, object: ObjectId) -> Option<u32> {
        let LinkRemovalPlan {
            topology,
            remaining,
            zero: _,
            #[cfg(test)]
                work: _,
        } = &self.plan;
        remaining
            .get(topology.indices.get(&object)?)
            .and_then(|state| state.declared)
    }

    /// Returns eligible ordinary non-root objects in propagation order, once each.
    pub(in crate::edit) fn counted_zero_objects(
        &self,
    ) -> impl Iterator<Item = CountedZeroObject<'topology, 'source>> + '_ {
        let LinkRemovalPlan {
            topology,
            remaining: _,
            zero,
            #[cfg(test)]
                work: _,
        } = &self.plan;
        zero.iter()
            .map(|&index| CountedZeroObject { topology, index })
    }

    /// Returns touched objects lacking an ordinary subject or two zero remainders.
    ///
    /// Objects are in topology index order.
    pub(in crate::edit) fn retained_objects(&self) -> impl Iterator<Item = ObjectId> + '_ {
        let LinkRemovalPlan {
            topology,
            remaining,
            zero: _,
            #[cfg(test)]
                work: _,
        } = &self.plan;
        remaining
            .iter()
            .filter(|(_, state)| !state.reaches_zero())
            .map(|(&index, _)| topology.graph.object(index).value.id)
    }

    /// Returns original bounds and planned declared net counts for touched objects with evidence.
    ///
    /// Includes retained subjects with known counts, in topology index order. These numeric
    /// projections require separate source-write and publication permissions.
    pub(in crate::edit) fn changes(
        &self,
    ) -> impl Iterator<Item = DeclaredCountChange<'_, 'source>> {
        self.plan.remaining.values().filter_map(|state| {
            let SourceCountState {
                evidence,
                graph: _,
                declared,
            } = state;
            evidence
                .as_ref()
                .zip(*declared)
                .map(|(evidence, remaining)| DeclaredCountChange {
                    bound: evidence.bound(),
                    remaining,
                })
        })
    }
}

/// An ordinary non-root object with zero planned graph and declared count remainders.
///
/// Source, topology, object, and incoming-count associations match the original evidence.
/// The caller establishes source freshness, storage ownership, reference completeness, and
/// publication conditions before retirement.
pub(in crate::edit) struct CountedZeroObject<'topology, 'source> {
    topology: &'topology CompleteTopology<'source>,
    index: ObjectIndex,
}

impl CountedZeroObject<'_, '_> {
    /// Returns the physical object-header identity from the matching topology.
    pub(in crate::edit) fn id(&self) -> ObjectId {
        let Self { topology, index } = self;
        topology.graph.object(*index).value.id
    }
}

/// A planned declared net count paired with its unchanged original count bound.
///
/// The projection describes a numeric change independently of permission to write the field.
pub(in crate::edit) struct DeclaredCountChange<'plan, 'source> {
    bound: &'plan ObjectReferenceBound<'source>,
    remaining: u32,
}

impl<'source> DeclaredCountChange<'_, 'source> {
    /// Borrows the original source-bound count and any physical field association.
    pub(in crate::edit) fn bound(&self) -> &ObjectReferenceBound<'source> {
        self.bound
    }

    /// Returns the declared count after the planned removals and any supplied additions.
    pub(in crate::edit) fn remaining(&self) -> u32 {
        self.remaining
    }
}

/// Sparse graph and optional declared remainders with unchanged original evidence.
struct SourceCountState<'source> {
    evidence: Option<CountEvidence<'source>>,
    graph: u32,
    declared: Option<u32>,
}

impl<'source> SourceCountState<'source> {
    /// Initializes remainders from topology and optional matching source evidence.
    ///
    /// Without evidence, retains only the graph count and remains ineligible for propagation.
    ///
    /// # Errors
    ///
    /// Returns errors from `query`, or [`CountEvidenceError::Topology`] if an incoming count is
    /// missing or the source, topology, object, or incoming-count associations differ.
    fn new(
        topology: &CompleteTopology<'source>,
        index: ObjectIndex,
        query: &mut impl FnMut(ObjectId) -> Result<Option<CountEvidence<'source>>, CountEvidenceError>,
    ) -> Result<Self, CountEvidenceError> {
        let object = topology.graph.object(index).value.id;
        let graph = topology
            .incoming(object.header_address().get())
            .ok_or_else(|| {
                FormatError::Internal("a removed link target has no incoming count".into())
            })?;
        let evidence = query(object)?;
        if evidence.as_ref().is_some_and(|evidence| {
            let bound = evidence.bound();
            !bound.matches(topology) || bound.object() != object || bound.incoming() != graph
        }) {
            return Err(FormatError::Internal(
                "a declared count does not match its removal topology".into(),
            )
            .into());
        }
        let declared = evidence
            .as_ref()
            .map(|evidence| evidence.bound().original());
        Ok(Self {
            evidence,
            graph,
            declared,
        })
    }

    /// Decrements graph and known declared remainders together before testing eligibility.
    ///
    /// On failure, leaves both remainders unchanged. Original evidence remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CountEvidenceError::Topology`] if either known count would underflow.
    fn remove(&mut self) -> Result<bool, CountEvidenceError> {
        let Self {
            evidence: _,
            graph,
            declared,
        } = self;
        let next_graph = graph.checked_sub(1).ok_or_else(|| {
            FormatError::Internal("removed links exceed their target's incoming count".into())
        })?;
        let next_declared = declared
            .map(|count| {
                count.checked_sub(1).ok_or_else(|| {
                    FormatError::Internal(
                        "removed links exceed their target's declared count".into(),
                    )
                })
            })
            .transpose()?;
        *graph = next_graph;
        *declared = next_declared;
        Ok(self.reaches_zero())
    }

    /// Returns whether both remainders are zero and the subject is ordinary.
    fn reaches_zero(&self) -> bool {
        let Self {
            evidence,
            graph,
            declared,
        } = self;
        *graph == 0 && *declared == Some(0) && matches!(evidence, Some(CountEvidence::Ordinary(_)))
    }
}

#[cfg(test)]
mod tests;
