//! Count-removal traversal within one complete, root-reachable hard-link topology.
//!
//! Counts must reach zero in both the topology and the source count before propagation.
//! Only ordinary non-root subjects can propagate removals. Storage reuse and publication require
//! separate evidence.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;

use crate::edit::analysis::counts::CountEvidence;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::error::FormatError;
use crate::object_graph::LinkIndex;
use crate::object_graph::ObjectId;
use crate::object_graph::ObjectIndex;

use super::{CompleteTopology, PlannedDeletion, RemovedLinks};

pub(in crate::edit) use net::original::{OriginalOwnerFacts, OriginalOwnerInventory};
pub(in crate::edit) use net::{NetCountError, NetCountPlan};
pub(in crate::edit) use ownership::inventory::CountPreflight;
pub(in crate::edit) use ownership::inventory::CountPreflightError;
pub(in crate::edit) use ownership::inventory::CountProjectionIdentity;
pub(in crate::edit) use ownership::{CountHandoffError, CountProtectionError, ProtectedCounts};

mod added;
mod net;
mod ownership;

/// Sparse count states and eligible non-root transitions within one borrowed topology.
///
/// The caller supplies the eligibility criterion to [`Self::build`].
/// Visits each queued object's remaining outgoing edges once.
/// Allocation ownership, source freshness, references, and publication require separate proofs.
struct LinkRemovalPlan<'topology, 'source, Count> {
    topology: &'topology CompleteTopology<'source>,
    remaining: BTreeMap<ObjectIndex, Count>,
    zero: Vec<ObjectIndex>,
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
        };
        for &(_, object) in &edges {
            let index = topology.indices.get(&object).copied().ok_or_else(|| {
                FormatError::Internal("a removed link target is absent from its topology".into())
            })?;
            plan.remove_incoming(index, &mut initial, &mut decrement)?;
        }
        let mut next = 0;
        while let Some(&index) = plan.zero.get(next) {
            next += 1;
            for (position, link) in topology.graph.object(index).links.iter().enumerate() {
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
        } = self;
        let count = match remaining.entry(index) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(initial(index)?),
        };
        let reaches_zero = decrement(count)?;
        if reaches_zero && index != ObjectIndex::ROOT {
            zero.push(index);
        }
        Ok(())
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
    /// Returns the planned declared net count for a touched object with count evidence.
    ///
    /// Returns `None` for untouched or outside objects, or when no count evidence was supplied.
    pub(in crate::edit) fn declared_remaining(&self, object: ObjectId) -> Option<u32> {
        let LinkRemovalPlan {
            topology,
            remaining,
            zero: _,
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
        } = &self.plan;
        remaining
            .iter()
            .filter(|(_, state)| !state.reaches_zero())
            .map(|(&index, _)| topology.graph.object(index).value.id)
    }

    /// Appends supported complete allocations from each eligible object once.
    ///
    /// Uses propagation order and preserves the collector's allocation boundaries and free
    /// classes. Reference completeness, source freshness, and publication require separate proofs.
    fn collect_storage(&self, out: &mut Vec<(FileRange, FreeClass)>) {
        for object in self.counted_zero_objects() {
            self.plan
                .topology
                .engine
                .collect_object_storage(object.id(), out);
        }
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

/// Sparse graph remainders and optional declared net counts with unchanged original evidence.
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
