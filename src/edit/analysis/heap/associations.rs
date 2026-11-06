//! Root-scoped associations between decoded heap references and allocated members.
//!
//! Reference interpretation and member framing are independent of logical ownership and
//! retirement permission.

use core::num::NonZeroU16;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::PhysicalAddress;
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

impl AnalysisPhase<'_> {
    /// Returns cached member associations for this phase's root topology.
    ///
    /// Discovers topology on the first query and reuses cached object references and collection
    /// framing. Retains partial reference identities and framing failures independently. Uses the
    /// phase root and requires a `source` from this phase. Source consistency,
    /// local allocation ownership, and physical coverage require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::heap_reference_facts`] and [`Self::reconstruction_source`].
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `source` belongs to another
    /// phase or the cached result cannot be retrieved.
    pub(in crate::edit) fn member_associations(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<&MemberAssociations, Error> {
        self.require_heap_phase(source)?;
        if self.associations.is_none() {
            let associations = MemberAssociations::collect(self, source)?;
            return Ok(self.associations.insert(associations));
        }
        self.associations.as_ref().ok_or_else(|| {
            FormatError::Internal("member associations were not retained".into()).into()
        })
    }
}

/// Observed global heap members and their referring objects in one phase's root topology.
///
/// Includes collections referenced by root-reachable objects. Topology availability, reference
/// interpretation, and collection framing are recorded separately. Exclusive ownership and
/// retirement permission require separate evidence.
pub(in crate::edit) struct MemberAssociations {
    source: ReconstructionSource,
    root: Option<ObjectId>,
    topology_available: bool,
    incomplete_references: Vec<ObjectId>,
    collections: BTreeMap<PhysicalAddress, CollectionAssociations>,
}

impl MemberAssociations {
    /// Collects reference identities from root-reachable objects and compares member framing.
    ///
    /// Reuses the phase's topology, reference, and collection caches. Retains distinct referring
    /// objects per member and sorts allocated indices absent from the observed references.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AnalysisPhase::reconstruction_source`] and
    /// [`AnalysisPhase::heap_reference_facts`].
    fn collect(
        phase: &mut AnalysisPhase<'_>,
        source: &ReconstructionSource,
    ) -> Result<Self, Error> {
        let engine = phase.engine;
        let root = phase.root.or_else(|| {
            ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).ok()
        });
        let mut associations = Self {
            source: ReconstructionSource {
                identity: Arc::clone(&source.identity),
                object: source.object(),
            },
            root,
            topology_available: false,
            incomplete_references: Vec::new(),
            collections: BTreeMap::new(),
        };
        let objects = match phase.topology() {
            Some(topology) => {
                associations.topology_available = true;
                topology.indices.keys().copied().collect::<Vec<_>>()
            }
            None => Vec::new(),
        };
        for object in objects {
            let origin = phase.reconstruction_source(engine, object)?;
            let references = phase.heap_reference_facts(&origin)?;
            if !references.is_complete() {
                associations.incomplete_references.push(object);
            }
            for &GlobalHeapId { collection, index } in references.objects() {
                associations
                    .collections
                    .entry(collection)
                    .or_insert_with(CollectionAssociations::new)
                    .observed
                    .entry(index)
                    .or_insert_with(|| MemberAssociation {
                        referrers: Vec::new(),
                        presence: MemberPresence::Unknown,
                    })
                    .referrers
                    .push(object);
            }
        }
        for (&address, collection) in &mut associations.collections {
            match phase.heap_members(source, address) {
                Ok(contents) => {
                    for member in collection.observed.values_mut() {
                        member.presence = MemberPresence::Missing;
                    }
                    for member in contents.members() {
                        if let Some(observed) = collection.observed.get_mut(&member.index()) {
                            observed.presence = MemberPresence::Allocated;
                        } else {
                            collection.unmatched.push(member.index());
                        }
                    }
                    collection.unmatched.sort_unstable();
                }
                Err(error) => collection.error = Some(error),
            }
        }
        Ok(associations)
    }

    /// Returns the physical root selected for topology discovery, if its address resolved.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.root
    }

    /// Returns whether `source` has the same phase identity.
    ///
    /// Source freshness and ownership require separate evidence.
    pub(in crate::edit) fn matches_phase(&self, source: &ReconstructionSource) -> bool {
        Arc::ptr_eq(&self.source.identity, &source.identity)
    }

    /// Returns whether root-reachable hard-link topology was completely discovered.
    pub(in crate::edit) fn topology_available(&self) -> bool {
        self.topology_available
    }

    /// Borrows sorted object identities with incomplete reference interpretation.
    ///
    /// Partial decoded identities are retained in [`Self::collections`].
    pub(in crate::edit) fn incomplete_references(&self) -> &[ObjectId] {
        &self.incomplete_references
    }

    /// Borrows observed collections keyed by physical address.
    pub(in crate::edit) fn collections(
        &self,
    ) -> &BTreeMap<PhysicalAddress, CollectionAssociations> {
        &self.collections
    }
}

/// Observed member indices and an independent framing outcome for one collection.
///
/// Allocated indices are compared with observed references only after successful framing.
/// Reference completeness and logical ownership require separate evidence.
pub(in crate::edit) struct CollectionAssociations {
    observed: BTreeMap<NonZeroU16, MemberAssociation>,
    unmatched: Vec<NonZeroU16>,
    error: Option<FormatError>,
}

impl CollectionAssociations {
    fn new() -> Self {
        Self {
            observed: BTreeMap::new(),
            unmatched: Vec::new(),
            error: None,
        }
    }

    /// Borrows descriptor member indices with their referring objects and allocation presence.
    pub(in crate::edit) fn observed(&self) -> &BTreeMap<NonZeroU16, MemberAssociation> {
        &self.observed
    }

    /// Borrows sorted allocated member indices absent from the observed reference set.
    ///
    /// Returns an empty slice after framing failure. Use [`Self::error`] to distinguish that
    /// outcome from a successful comparison.
    pub(in crate::edit) fn unmatched(&self) -> &[NonZeroU16] {
        &self.unmatched
    }

    /// Borrows the collection framing failure, if any.
    pub(in crate::edit) fn error(&self) -> Option<&FormatError> {
        self.error.as_ref()
    }
}

/// Distinct referring objects and parsed allocation presence for one observed member.
pub(in crate::edit) struct MemberAssociation {
    referrers: Vec<ObjectId>,
    presence: MemberPresence,
}

impl MemberAssociation {
    /// Borrows referring object identities in sorted order, with each object included once.
    pub(in crate::edit) fn referrers(&self) -> &[ObjectId] {
        &self.referrers
    }

    /// Returns allocation presence as established by collection framing.
    pub(in crate::edit) fn presence(&self) -> MemberPresence {
        self.presence
    }
}

/// The framing result for a member index decoded from an object's content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum MemberPresence {
    /// The index occurs among successfully parsed allocated members.
    Allocated,
    /// The index is absent from successfully parsed allocated members.
    Missing,
    /// Collection framing failed, so allocation presence is undetermined.
    Unknown,
}

#[cfg(test)]
mod tests;
