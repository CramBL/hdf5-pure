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

/// Observed global heap member indices in one phase's root topology.
///
/// Includes collections referenced by root-reachable objects. Topology availability, reference
/// interpretation, and collection framing are recorded separately. Exclusive ownership and
/// retirement permission require separate evidence.
pub(in crate::edit) struct MemberAssociations {
    topology_available: bool,
    references_complete: bool,
    collections: BTreeMap<PhysicalAddress, CollectionAssociations>,
}

impl MemberAssociations {
    /// Collects reference identities from root-reachable objects and compares member framing.
    ///
    /// Reuses the phase's topology, reference, and collection caches. Records whether any
    /// allocated index is absent from the observed references.
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
            references_complete: true,
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
            associations.references_complete &= references.is_complete();
            for &GlobalHeapId { collection, index } in references.objects() {
                associations
                    .collections
                    .entry(collection)
                    .or_insert_with(CollectionAssociations::new)
                    .observed
                    .entry(index)
                    .or_insert(MemberPresence::Unknown);
            }
        }
        for (&address, collection) in &mut associations.collections {
            if let Ok(contents) = phase.heap_members(source, address) {
                collection.framed = true;
                for presence in collection.observed.values_mut() {
                    *presence = MemberPresence::Missing;
                }
                for member in contents.members() {
                    if let Some(observed) = collection.observed.get_mut(&member.index()) {
                        *observed = MemberPresence::Allocated;
                    } else {
                        collection.unmatched = true;
                    }
                }
            }
        }
        Ok(associations)
    }

    /// Returns whether root-reachable hard-link topology was completely discovered.
    pub(in crate::edit) fn topology_available(&self) -> bool {
        self.topology_available
    }

    /// Returns whether every examined object's references could be interpreted completely.
    pub(in crate::edit) fn references_complete(&self) -> bool {
        self.references_complete
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
    observed: BTreeMap<NonZeroU16, MemberPresence>,
    unmatched: bool,
    framed: bool,
}

impl CollectionAssociations {
    fn new() -> Self {
        Self {
            observed: BTreeMap::new(),
            unmatched: false,
            framed: false,
        }
    }

    /// Borrows descriptor member indices and their allocation presence.
    pub(in crate::edit) fn observed(&self) -> &BTreeMap<NonZeroU16, MemberPresence> {
        &self.observed
    }

    /// Returns whether an allocated member is absent from the observed references.
    ///
    /// Use [`Self::is_framed`] to establish that the comparison completed.
    pub(in crate::edit) fn has_unmatched(&self) -> bool {
        self.unmatched
    }

    /// Returns whether all allocated member records were parsed successfully.
    pub(in crate::edit) fn is_framed(&self) -> bool {
        self.framed
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
