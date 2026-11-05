//! Deletion names and physical edges bound to one complete topology instance.
//!
//! Edge identity and target identity are retained separately. Allocation collection uses the
//! phase's reclamation policy, and collection-reference analysis uses the distinct target set.

use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::LinkIndex;
use crate::object_graph::ObjectId;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;

use super::AnalysisPhase;
use super::CompleteTopology;
use super::counts::CountQueries;

pub(in crate::edit) use counts::CountPreflight;
pub(in crate::edit) use counts::CountProjectionIdentity;
pub(in crate::edit) use counts::CountStorage;
pub(in crate::edit) use counts::NetCountError;

pub(in crate::edit) use counts::DeclaredCountChange;
pub(in crate::edit) use counts::NetCountPlan;
pub(in crate::edit) use counts::SourceRemovalPlan;

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "graph-relative count transitions do not authorize allocation retirement"
    )
)]
mod counts;

/// A leaf name paired with its physical target and any complete-topology edge identity.
///
/// An identified edge belongs to the particular topology instance that resolved it. A target
/// resolved without an edge identity remains separate from that topology evidence.
pub(in crate::edit) struct PlannedDeletion {
    name: LinkNameBuf,
    target: DeletionTarget,
}

impl PlannedDeletion {
    /// Records a resolved physical target without claiming an identified topology edge.
    pub(in crate::edit) fn unresolved(name: LinkNameBuf, target: PhysicalAddress) -> Self {
        Self {
            name,
            target: DeletionTarget::Unresolved(ObjectId::new(target)),
        }
    }

    /// Borrows the leaf name used to remove the Link record from its owner.
    pub(in crate::edit) fn name(&self) -> &LinkNameBuf {
        &self.name
    }
}

enum DeletionTarget {
    Complete {
        topology: Arc<TopologyIdentity>,
        edge: LinkIndex,
        object: ObjectId,
    },
    Unresolved(ObjectId),
}

/// A shared identity for a source phase's topology and reconstruction origins.
///
/// May be allocated before topology discovery succeeds. Only the identity allocation is shared.
/// The caller establishes source freshness and starts a new analysis phase after relevant writes.
#[derive(Debug)]
pub(super) struct TopologyIdentity;

/// Distinct edges bound to a borrowed topology, with separately resolved targets.
///
/// Without complete topology, only unresolved targets are retained. Distinct edges may target
/// one object, and different paths through owner aliases may identify the same edge.
pub(in crate::edit) struct RemovedLinks<'topology, 'source> {
    topology: Option<&'topology CompleteTopology<'source>>,
    edges: Vec<(LinkIndex, ObjectId)>,
    unresolved: Vec<ObjectId>,
}

impl<'topology, 'source> RemovedLinks<'topology, 'source> {
    /// Checks topology identities and retains each removed physical edge once.
    ///
    /// The caller supplies the applicable topology. Target-only unresolved records are retained
    /// separately from identified edges.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if an identified edge belongs to a different topology
    /// instance, including when no complete topology is supplied.
    fn bind<'deletion>(
        topology: Option<&'topology CompleteTopology<'source>>,
        deletions: impl IntoIterator<Item = &'deletion PlannedDeletion>,
    ) -> Result<Self, FormatError> {
        let mut removed = Self {
            topology,
            edges: Vec::new(),
            unresolved: Vec::new(),
        };
        for deletion in deletions {
            match &deletion.target {
                DeletionTarget::Complete {
                    topology: identity,
                    edge,
                    object,
                } => {
                    if !topology.is_some_and(|topology| Arc::ptr_eq(identity, &topology.identity)) {
                        return Err(FormatError::Internal(
                            "a removed link belongs to another analysis topology".into(),
                        ));
                    }
                    removed.edges.push((*edge, *object));
                }
                DeletionTarget::Unresolved(object) => removed.unresolved.push(*object),
            }
        }
        removed.edges.sort_unstable_by_key(|&(edge, _)| edge);
        removed.edges.dedup_by_key(|entry| entry.0);
        Ok(removed)
    }
}

impl RemovedLinks<'_, '_> {
    /// Appends allocations selected by reclamation policy and returns distinct physical targets.
    ///
    /// With complete topology, calls [`CompleteTopology::collect_free_spans`] once per target
    /// using the original incoming-link counts. Without it, appends no allocations. Returns
    /// targets in address order for separate collection-candidate discovery by the caller.
    pub(in crate::edit) fn collect_free_spans(
        self,
        out: &mut Vec<(FileRange, FreeClass)>,
    ) -> Vec<u64> {
        let Self {
            topology,
            edges,
            unresolved,
        } = self;
        let mut targets = edges
            .into_iter()
            .map(|(_, object)| object)
            .chain(unresolved)
            .map(|object| object.header_address().get())
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        if let Some(topology) = topology {
            for &target in &targets {
                topology.collect_free_spans(target, 0, out);
            }
        }
        targets
    }
}

impl<'source> AnalysisPhase<'source> {
    /// Binds deletion records to this phase's topology, retaining each physical edge once.
    ///
    /// Discovers topology only for nonempty input. Retains targets resolved without an edge
    /// identity for separate candidate discovery, even if topology is unavailable.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if an identified edge belongs to a different topology
    /// instance, including when this phase has no complete topology.
    pub(in crate::edit) fn removed_links<'deletion>(
        &mut self,
        deletions: impl IntoIterator<Item = &'deletion PlannedDeletion>,
    ) -> Result<RemovedLinks<'_, 'source>, Error> {
        self.removal_queries(deletions)
            .map(|(removed, _)| removed)
            .map_err(Error::from)
    }

    /// Returns bound deletions and lazy count queries over separate phase fields.
    ///
    /// Discovers topology only for nonempty input. An empty batch has no applicable topology in
    /// either returned view. Defers count reads until queried, using the existing phase cache.
    /// The caller can retain or consume the deletion batch while using the count-query view.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] for the topology-identity errors described by
    /// [`RemovedLinks::bind`].
    pub(in crate::edit) fn removal_queries<'deletion>(
        &mut self,
        deletions: impl IntoIterator<Item = &'deletion PlannedDeletion>,
    ) -> Result<(RemovedLinks<'_, 'source>, CountQueries<'_, 'source>), FormatError> {
        let mut deletions = deletions.into_iter().peekable();
        let has_deletions = deletions.peek().is_some();
        if has_deletions {
            self.topology();
        }
        let Self {
            engine,
            identity: _,
            topology,
            counts,
            budget: _,
            root: _,
            reference_roots: _,
            links: _,
            references: _,
            liveness: _,
            allocations: _,
            #[cfg(test)]
                candidate_lookups: _,
        } = self;
        let topology = if has_deletions {
            topology.as_ref().and_then(Option::as_ref)
        } else {
            None
        };
        Ok((
            RemovedLinks::bind(topology, deletions)?,
            CountQueries::new(engine, counts, topology),
        ))
    }
}

impl CompleteTopology<'_> {
    /// Resolves a deletion's leaf name, physical edge, and target within this topology instance.
    ///
    /// Returns `None` for an empty path or a missing component. Uses the per-owner name indexes
    /// described by [`Self::find_link`].
    pub(in crate::edit) fn deletion(&self, path: &ObjectPathBuf) -> Option<PlannedDeletion> {
        let (_, name) = path.split_leaf()?;
        let (edge, object) = self.find_link(path)?;
        Some(PlannedDeletion {
            name,
            target: DeletionTarget::Complete {
                topology: Arc::clone(&self.identity),
                edge,
                object,
            },
        })
    }
}

#[cfg(test)]
mod tests;
