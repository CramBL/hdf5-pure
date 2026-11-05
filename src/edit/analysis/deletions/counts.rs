//! Count-removal traversal within one complete, root-reachable hard-link topology.
//!
//! The graph adapter uses visible incoming counts. The source-bound adapter also checks declared
//! count remainders and ordinary header subjects before outgoing propagation. Both exclude the
//! phase root. Allocation ownership, source freshness, reference completeness, and publication
//! remain separate obligations.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;

use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::counts::CountEvidence;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::edit::analysis::counts::ObjectReferenceBound;
use crate::error::FormatError;
use crate::object_graph::LinkIndex;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;

use super::CompleteTopology;
use super::PlannedDeletion;
use super::RemovedLinks;

mod ownership;

impl<'topology, 'source> RemovedLinks<'topology, 'source> {
    /// Plans graph-relative count changes for the identified removed edges.
    ///
    /// Returns `None` without complete topology. Target-only unresolved records do not reduce
    /// counts. Processes the remaining outgoing edges of each object that reaches zero, excluding
    /// the phase root. Leaves the original topology and incoming counts unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a target or its original incoming count is missing,
    /// or if the removals would exceed that count.
    pub(in crate::edit) fn count_plan(
        self,
    ) -> Result<Option<LinkRemovalPlan<'topology, 'source>>, FormatError> {
        let Self {
            topology,
            edges,
            unresolved: _,
        } = self;
        topology
            .map(|topology| LinkRemovalPlan::new(topology, edges))
            .transpose()
    }
}

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

/// Sparse count states and eligible non-root transitions within one borrowed topology.
///
/// With the default graph adapter, zero is the queueing criterion for [`u32`] incoming counts.
/// For other count states, the caller supplies the eligibility criterion to [`Self::build`].
/// Each queued object's outgoing adjacency is visited once, excluding explicitly removed edges
/// from those decrements. Positive cycles remain outside the default graph plan's zero-count set.
/// Allocation ownership, source freshness, references, and publication require separate proofs.
pub(in crate::edit) struct LinkRemovalPlan<'topology, 'source, Count = u32> {
    topology: &'topology CompleteTopology<'source>,
    remaining: BTreeMap<ObjectIndex, Count>,
    zero: Vec<ObjectIndex>,
    #[cfg(test)]
    work: RemovalWork,
}

impl<'topology, 'source> LinkRemovalPlan<'topology, 'source> {
    /// Uses original graph incoming counts and checked decrements as the propagation criterion.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if an explicit target is outside the topology, an
    /// incoming count is missing, or removals exceed the original count.
    fn new(
        topology: &'topology CompleteTopology<'source>,
        edges: Vec<(LinkIndex, ObjectId)>,
    ) -> Result<Self, FormatError> {
        Self::build(
            topology,
            edges,
            |index| {
                let object = topology.graph.object(index).value.id;
                topology.incoming(object).ok_or_else(|| {
                    FormatError::Internal("a removed link target has no incoming count".into())
                })
            },
            |count| {
                *count = count.checked_sub(1).ok_or_else(|| {
                    FormatError::Internal(
                        "removed links exceed their target's incoming count".into(),
                    )
                })?;
                Ok(*count == 0)
            },
        )
    }

    /// Returns an object's remaining graph-relative incoming count, including for the phase root.
    ///
    /// Returns `None` if `object` is outside the topology. For untouched objects, returns the
    /// original incoming count, or zero if there is no incoming-count entry.
    pub(in crate::edit) fn remaining(&self, object: ObjectId) -> Option<u32> {
        let Self {
            topology,
            remaining,
            zero: _,
            #[cfg(test)]
                work: _,
        } = self;
        let &index = topology.indices.get(&object)?;
        Some(
            remaining
                .get(&index)
                .copied()
                .unwrap_or_else(|| topology.incoming(object).unwrap_or(0)),
        )
    }

    /// Returns non-root objects that reached zero, once each, in transition order.
    pub(in crate::edit) fn zero_link_objects(
        &self,
    ) -> impl Iterator<Item = ZeroLinkObject<'topology, 'source>> + '_ {
        let Self {
            topology,
            remaining: _,
            zero,
            #[cfg(test)]
                work: _,
        } = self;
        zero.iter().map(|&index| ZeroLinkObject { topology, index })
    }
}

impl<'topology, 'source, Count> LinkRemovalPlan<'topology, 'source, Count> {
    /// Builds sparse count states with caller-supplied initialization and decrement criteria.
    ///
    /// The caller passes edges sorted by identity, with each physical edge once and its matching
    /// target. Calls `initial` once on first touching an object, then `decrement` for each removed
    /// incoming edge. The decrement callback returns `true` only when the object first becomes
    /// eligible, after establishing the required conditions.
    ///
    /// Eligible non-root objects are queued before their remaining outgoing edges are visited.
    /// Already explicit removals are skipped during propagation. The original topology remains
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns errors from `initial` or `decrement`. Converts [`FormatError::Internal`] into `E`
    /// if an explicit target is outside the topology.
    fn build<E: From<FormatError>>(
        topology: &'topology CompleteTopology<'source>,
        edges: Vec<(LinkIndex, ObjectId)>,
        mut initial: impl FnMut(ObjectIndex) -> Result<Count, E>,
        mut decrement: impl FnMut(&mut Count) -> Result<bool, E>,
    ) -> Result<Self, E> {
        let mut plan = Self {
            topology,
            remaining: BTreeMap::new(),
            zero: Vec::new(),
            #[cfg(test)]
            work: RemovalWork::default(),
        };
        for &(_, object) in &edges {
            let index = topology.indices.get(&object).copied().ok_or_else(|| {
                FormatError::Internal("a removed link target is absent from its topology".into())
            })?;
            #[cfg(test)]
            {
                plan.work.initial_edges += 1;
            }
            plan.remove_incoming(index, &mut initial, &mut decrement)?;
        }
        let mut next = 0;
        while let Some(&index) = plan.zero.get(next) {
            next += 1;
            #[cfg(test)]
            {
                plan.work.objects += 1;
            }
            for (position, link) in topology.graph.object(index).links.iter().enumerate() {
                #[cfg(test)]
                {
                    plan.work.outgoing_edges += 1;
                }
                if edges
                    .binary_search_by_key(&LinkIndex::new(index, position), |&(edge, _)| edge)
                    .is_err()
                {
                    plan.remove_incoming(link.target, &mut initial, &mut decrement)?;
                }
            }
        }
        Ok(plan)
    }

    /// Updates one sparse count state before queueing an eligible non-root transition.
    ///
    /// The callbacks follow the requirements of [`Self::build`]. Initialization runs only for a
    /// vacant count entry. Queue insertion follows a successful eligibility check.
    ///
    /// # Errors
    ///
    /// Returns errors from `initial` or `decrement`.
    fn remove_incoming<E: From<FormatError>>(
        &mut self,
        index: ObjectIndex,
        initial: &mut impl FnMut(ObjectIndex) -> Result<Count, E>,
        decrement: &mut impl FnMut(&mut Count) -> Result<bool, E>,
    ) -> Result<(), E> {
        let Self {
            topology: _,
            remaining,
            zero,
            #[cfg(test)]
            work,
        } = self;
        let count = match remaining.entry(index) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(initial(index)?),
        };
        let reaches_zero = decrement(count)?;
        #[cfg(test)]
        {
            work.decrements += 1;
        }
        if reaches_zero && index != ObjectIndex::ROOT {
            zero.push(index);
        }
        Ok(())
    }
}

/// An object with zero incoming links within the particular borrowed topology.
///
/// The phase root is excluded. The caller establishes allocation ownership, relevant reference
/// completeness, absence of incoming links from outside the graph, and publication conditions
/// before retiring storage.
pub(in crate::edit) struct ZeroLinkObject<'topology, 'source> {
    topology: &'topology CompleteTopology<'source>,
    index: ObjectIndex,
}

impl ZeroLinkObject<'_, '_> {
    /// Returns the physical object-header identity from the matching topology.
    pub(in crate::edit) fn id(&self) -> ObjectId {
        let Self { topology, index } = self;
        topology.graph.object(*index).value.id
    }
}

#[cfg(test)]
#[derive(Default)]
struct RemovalWork {
    initial_edges: usize,
    objects: usize,
    outgoing_edges: usize,
    decrements: usize,
}

/// Planned graph and declared remainders with eligibility checked before child propagation.
///
/// Original source counts, fields, and topology remain unchanged. Only recognized ordinary
/// non-root objects with zero graph and declared remainders are eligible. Allocation and
/// publication decisions remain the caller's responsibility.
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
            || topology.incoming(object).unwrap_or(0),
            |state| state.graph,
        ))
    }

    /// Returns the planned declared remainder for a touched object with count evidence.
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

    /// Returns original bounds and planned declared remainders for touched objects with evidence.
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

/// A planned declared remainder paired with its unchanged original count bound.
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

    /// Returns the declared count after the planned removals.
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
        let graph = topology.incoming(object).ok_or_else(|| {
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
