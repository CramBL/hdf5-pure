//! Incoming-count transitions within one complete, root-reachable hard-link topology.
//!
//! Explicit removals and outgoing edges of non-root zero-count objects are counted separately from
//! allocation ownership and publication. Counts describe the discovered graph. The caller
//! establishes the absence of relevant incoming links from outside that graph before retirement.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::error::FormatError;
use crate::object_graph::LinkIndex;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;

use super::CompleteTopology;
use super::RemovedLinks;

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

/// Sparse remaining counts and non-root objects that reach zero within one borrowed topology.
///
/// Each zero-count object's outgoing adjacency is visited once. Explicitly removed edges are
/// excluded from those decrements. Positive-count cycles remain outside the zero-count set.
/// Allocation ownership, source freshness, references, and publication require separate proofs.
pub(in crate::edit) struct LinkRemovalPlan<'topology, 'source> {
    topology: &'topology CompleteTopology<'source>,
    remaining: BTreeMap<ObjectIndex, u32>,
    zero: Vec<ObjectIndex>,
    #[cfg(test)]
    work: RemovalWork,
}

impl<'topology, 'source> LinkRemovalPlan<'topology, 'source> {
    fn new(
        topology: &'topology CompleteTopology<'source>,
        edges: Vec<(LinkIndex, ObjectId)>,
    ) -> Result<Self, FormatError> {
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
            plan.remove_incoming(index)?;
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
                    plan.remove_incoming(link.target)?;
                }
            }
        }
        Ok(plan)
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

    fn remove_incoming(&mut self, index: ObjectIndex) -> Result<(), FormatError> {
        let Self {
            topology,
            remaining,
            zero,
            #[cfg(test)]
            work,
        } = self;
        let object = topology.graph.object(index).value.id;
        let count = match remaining.entry(index) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(
                topology
                    .incoming(object.header_address().get())
                    .ok_or_else(|| {
                        FormatError::Internal("a removed link target has no incoming count".into())
                    })?,
            ),
        };
        *count = count.checked_sub(1).ok_or_else(|| {
            FormatError::Internal("removed links exceed their target's incoming count".into())
        })?;
        #[cfg(test)]
        {
            work.decrements += 1;
        }
        if *count == 0 && index != ObjectIndex::ROOT {
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
