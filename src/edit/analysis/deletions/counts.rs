//! Count-removal traversal within one complete, root-reachable hard-link topology.
//!
//! The graph adapter uses visible incoming counts. The source-bound adapter also checks declared
//! count remainders and ordinary header subjects before outgoing propagation. Both exclude the
//! phase root. Allocation ownership, source freshness, reference completeness, and publication
//! remain separate obligations.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::error::FormatError;
use crate::object_graph::LinkIndex;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;

use super::CompleteTopology;
use super::RemovedLinks;

pub(in crate::edit) use source::DeclaredCountChange;
pub(in crate::edit) use source::SourceRemovalPlan;

mod source;

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
                topology
                    .incoming(object.header_address().get())
                    .ok_or_else(|| {
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
        Some(remaining.get(&index).copied().unwrap_or_else(|| {
            topology
                .incoming(object.header_address().get())
                .unwrap_or(0)
        }))
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

#[cfg(test)]
mod tests;
