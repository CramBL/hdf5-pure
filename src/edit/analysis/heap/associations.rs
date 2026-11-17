//! Root-scoped associations between decoded heap references and allocated members.
//!
//! Reference interpretation and member framing are independent of logical ownership and
//! retirement permission.

use core::num::NonZeroU16;
use std::collections::BTreeMap;

use crate::PhysicalAddress;
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;

impl AnalysisPhase<'_> {
    /// Returns cached member associations for this phase's root topology.
    ///
    /// Discovers topology on the first query and reuses cached object references and collection
    /// framing. Retains decoded member identities after incomplete interpretation and records
    /// framing status independently. Uses the phase root and requires a `source` from this phase.
    /// Source consistency,
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

/// Observed global heap members and framing outcomes for one phase's root topology.
///
/// Includes collections referenced by root-reachable objects. Topology availability, reference
/// interpretation, and collection framing are recorded separately. Exclusive ownership and
/// retirement permission require separate evidence.
pub(in crate::edit) struct MemberAssociations {
    topology_available: bool,
    incomplete_references: bool,
    collections: BTreeMap<PhysicalAddress, CollectionAssociations>,
}

impl MemberAssociations {
    /// Collects reference identities from root-reachable objects and compares member framing.
    ///
    /// Reuses the phase's topology, reference, and collection caches. Stores member presence and
    /// flags for incomplete references, unmatched allocated indices, and framing failures.
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
        let mut associations = Self {
            topology_available: false,
            incomplete_references: false,
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
                associations.incomplete_references = true;
            }
            for &GlobalHeapId { collection, index } in references.objects() {
                let MemberAssociation { presence: _ } = associations
                    .collections
                    .entry(collection)
                    .or_insert_with(CollectionAssociations::new)
                    .observed
                    .entry(index)
                    .or_insert_with(MemberAssociation::new);
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
                            collection.unmatched = true;
                        }
                    }
                }
                Err(error) => collection.record_failure(error),
            }
        }
        Ok(associations)
    }

    /// Returns whether root-reachable hard-link topology was completely discovered.
    pub(in crate::edit) fn topology_available(&self) -> bool {
        self.topology_available
    }

    /// Returns whether reference interpretation was incomplete for any visited object.
    pub(in crate::edit) fn has_incomplete_references(&self) -> bool {
        self.incomplete_references
    }

    /// Borrows observed collections keyed by physical address.
    pub(in crate::edit) fn collections(
        &self,
    ) -> &BTreeMap<PhysicalAddress, CollectionAssociations> {
        &self.collections
    }
}

/// Observed member indices and collection framing status.
///
/// Allocated indices are compared with observed references only after successful framing.
/// Reference completeness and logical ownership require separate evidence.
pub(in crate::edit) struct CollectionAssociations {
    observed: BTreeMap<NonZeroU16, MemberAssociation>,
    unmatched: bool,

    framing_failed: bool,
}

impl CollectionAssociations {
    fn new() -> Self {
        Self {
            observed: BTreeMap::new(),
            unmatched: false,

            framing_failed: false,
        }
    }

    /// Borrows decoded member indices and their allocation presence.
    pub(in crate::edit) fn observed(&self) -> &BTreeMap<NonZeroU16, MemberAssociation> {
        &self.observed
    }

    /// Returns whether collection framing failed.
    pub(in crate::edit) fn framing_failed(&self) -> bool {
        self.framing_failed
    }

    /// Returns whether framing found an allocated index absent from observed references.
    pub(in crate::edit) fn has_unmatched(&self) -> bool {
        self.unmatched
    }

    /// Marks collection framing as failed.
    fn record_failure(&mut self, _error: FormatError) {
        self.framing_failed = true;
    }
}

/// Allocation presence for one observed member.
pub(in crate::edit) struct MemberAssociation {
    presence: MemberPresence,
}

impl MemberAssociation {
    fn new() -> Self {
        Self {
            presence: MemberPresence::Unknown,
        }
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
