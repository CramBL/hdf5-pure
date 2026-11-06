//! Original global heap member and parent collection qualification for one source phase.
//!
//! Predecessor observations and parent outcomes are retained independently of the common
//! source-domain outcome. Source freshness, prospective ownership, and retirement permission
//! require separate evidence.

use core::num::NonZeroU16;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use super::CollectionContents;
use super::associations::{MemberAssociations, MemberPresence};
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::analysis::{
    AnalysisPhase, CollectionCandidates, CollectionProvenance, ObjectReferences,
    ReconstructionSource,
};
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

impl AnalysisPhase<'_> {
    /// Collects owned parent framing and candidate-allocation outcomes for storage exclusion.
    ///
    /// Includes candidate parents and parents in predecessor observations. Reads framing through
    /// the phase cache independently of member associations and whole-parent qualification.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] for a phase or session mismatch.
    /// Retains framing failures separately for each parent.
    pub(in crate::edit) fn collection_storage(
        &mut self,
        source: &ReconstructionSource,
        candidates: &CollectionCandidates,
    ) -> Result<BTreeMap<PhysicalAddress, ParentStorage>, Error> {
        self.require_heap_phase(source)?;
        if let Some(observations) = &candidates.observations {
            if !observations
                .session
                .same(&self.engine.context.allocations.session())
                || observations
                    .identity
                    .as_ref()
                    .is_some_and(|identity| !Arc::ptr_eq(identity, &source.identity))
            {
                return Err(FormatError::Internal(
                    "collection storage belongs to another source phase or session".into(),
                )
                .into());
            }
        }
        let mut addresses: BTreeSet<_> = candidates.collections.keys().copied().collect();
        if let Some(observations) = &candidates.observations {
            for references in observations.predecessors.values() {
                addresses.extend(
                    references
                        .collections
                        .iter()
                        .copied()
                        .map(PhysicalAddress::new),
                );
                addresses.extend(references.objects.iter().map(|id| id.collection));
            }
        }
        Ok(addresses
            .into_iter()
            .map(|address| {
                let framing = self
                    .collection_contents(address)
                    .map(|contents| contents.range());
                let original = candidates
                    .collections
                    .get(&address)
                    .map(|allocation| (allocation.range, allocation.class));
                let allocation = match (&framing, original) {
                    (Err(_), _) => Err(ParentUnavailable::Framing),
                    (_, None) => Err(ParentUnavailable::Allocation),
                    (Ok(range), Some((original, _))) if *range != original => {
                        Err(ParentUnavailable::Range)
                    }
                    (_, Some((_, None))) => Err(ParentUnavailable::Class),
                    _ => Ok(()),
                };
                (
                    address,
                    ParentStorage {
                        framing,
                        original,
                        allocation,
                    },
                )
            })
            .collect())
    }

    /// Compares original candidate observations with cached framing and root references.
    ///
    /// Checks the source phase and allocation session before discovery. Reuses phase caches,
    /// retains incomplete observations, and skips domain discovery for empty default input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] for a phase or session mismatch.
    /// Returns the errors of [`Self::supported_owner_domain`] and [`Self::member_associations`].
    pub(in crate::edit) fn qualify_collections(
        &mut self,
        source: &ReconstructionSource,
        candidates: CollectionCandidates,
    ) -> Result<CollectionQualification, Error> {
        let checks = self.inspect_collections(source, &candidates)?;
        Ok(CollectionQualification {
            source: source.clone(),
            candidates,
            domain: checks.domain,
            parents: checks.parents,
        })
    }

    /// Collects owned original qualification and full framing for use outside this phase.
    ///
    /// Retains the candidates when the source is absent or qualification fails. Moves cached
    /// framing and framing errors for candidate and observed parents out of this phase when
    /// the source matches. Empty default input skips source and domain discovery.
    pub(in crate::edit) fn capture_collections(
        &mut self,
        source: Option<&ReconstructionSource>,
        candidates: CollectionCandidates,
    ) -> OriginalCollections {
        if candidates.collections.is_empty() && candidates.observations.is_none() {
            return OriginalCollections {
                qualification: OriginalQualification::Empty,
                contents: BTreeMap::new(),
            };
        }
        let mut addresses: BTreeSet<_> = candidates.collections.keys().copied().collect();
        if let Some(observations) = &candidates.observations {
            for references in observations.predecessors.values() {
                addresses.extend(
                    references
                        .collections
                        .iter()
                        .copied()
                        .map(PhysicalAddress::new),
                );
                addresses.extend(references.objects.iter().map(|id| id.collection));
            }
        }
        let checks = match source {
            Some(source) => self
                .inspect_collections(source, &candidates)
                .map(|checks| (source.clone(), checks))
                .map_err(OriginalCollectionError::Qualification),
            None => Err(OriginalCollectionError::Source),
        };
        let mut contents = BTreeMap::new();
        if let Some(source) = source
            && self.require_heap_phase(source).is_ok()
        {
            for address in addresses {
                self.collection_contents(address).ok();
                if let Some(framing) = self.allocations.remove(&address) {
                    contents.insert(address, framing);
                }
            }
        }
        let qualification = match checks {
            Ok((source, checks)) => OriginalQualification::Qualified(CollectionQualification {
                source,
                candidates,
                domain: checks.domain,
                parents: checks.parents,
            }),
            Err(error) => OriginalQualification::Unavailable { candidates, error },
        };
        OriginalCollections {
            qualification,
            contents,
        }
    }

    fn inspect_collections(
        &mut self,
        source: &ReconstructionSource,
        candidates: &CollectionCandidates,
    ) -> Result<CollectionChecks, Error> {
        self.require_heap_phase(source)?;
        if let Some(observations) = &candidates.observations {
            if !observations
                .session
                .same(&self.engine.context.allocations.session())
            {
                return Err(FormatError::Internal(
                    "collection qualification belongs to another allocation session".into(),
                )
                .into());
            }
            if observations
                .identity
                .as_ref()
                .is_some_and(|identity| !Arc::ptr_eq(identity, &source.identity))
            {
                return Err(FormatError::Internal(
                    "collection qualification belongs to another source phase".into(),
                )
                .into());
            }
        }
        let mut parents = BTreeMap::new();
        for &address in candidates.collections.keys() {
            parents.insert(address, BTreeMap::new());
        }
        if let Some(observations) = &candidates.observations {
            for references in observations.predecessors.values() {
                for &address in &references.collections {
                    parents.entry(PhysicalAddress::new(address)).or_default();
                }
                for &GlobalHeapId { collection, index } in &references.objects {
                    parents
                        .entry(collection)
                        .or_default()
                        .insert(index, MemberPresence::Unknown);
                }
            }
        }
        let domain = if parents.is_empty() && candidates.observations.is_none() {
            None
        } else {
            Some(self.supported_owner_domain(source)?)
        };
        let mut qualified = BTreeMap::new();
        for (address, mut members) in parents {
            let framing = match self.collection_contents(address) {
                Ok(contents) => {
                    for presence in members.values_mut() {
                        *presence = MemberPresence::Missing;
                    }
                    for member in contents.members() {
                        if let Some(presence) = members.get_mut(&member.index()) {
                            *presence = MemberPresence::Allocated;
                        }
                    }
                    Ok(contents.range())
                }
                Err(error) => Err(error),
            };
            let allocation = match (&framing, candidates.collections.get(&address)) {
                (Err(_), _) => Err(ParentUnavailable::Framing),
                (_, None) => Err(ParentUnavailable::Allocation),
                (Ok(range), Some(allocation)) if *range != allocation.range => {
                    Err(ParentUnavailable::Range)
                }
                (_, Some(allocation)) if allocation.class.is_none() => {
                    Err(ParentUnavailable::Class)
                }
                _ => Ok(()),
            };
            let associations = self.member_associations(source)?;
            let whole = allocation.and_then(|()| {
                if !associations.topology_available()
                    || !associations.incomplete_references().is_empty()
                {
                    return Err(ParentUnavailable::References);
                }
                let collection = associations
                    .collections()
                    .get(&address)
                    .ok_or(ParentUnavailable::Association)?;
                if collection.error().is_some() {
                    return Err(ParentUnavailable::Framing);
                }
                if collection
                    .observed()
                    .values()
                    .any(|member| member.presence() != MemberPresence::Allocated)
                {
                    return Err(ParentUnavailable::MissingMember);
                }
                if !collection.unmatched().is_empty() {
                    return Err(ParentUnavailable::Unmatched);
                }
                Ok(())
            });
            qualified.insert(
                address,
                ParentQualification {
                    framing,
                    members,
                    allocation,
                    whole,
                },
            );
        }
        Ok(CollectionChecks {
            domain,
            parents: qualified,
        })
    }
}

/// Original candidate qualification and full parent framing retained outside the source phase.
///
/// Qualification and framing failures are recorded independently. Source freshness,
/// prospective ownership, and permission to mutate or reuse storage require separate evidence.
pub(in crate::edit) struct OriginalCollections {
    qualification: OriginalQualification,
    contents: BTreeMap<PhysicalAddress, Result<CollectionContents, FormatError>>,
}

impl OriginalCollections {
    /// Borrows original qualification or the candidates retained after qualification failed.
    pub(in crate::edit) fn qualification(&self) -> &OriginalQualification {
        &self.qualification
    }

    /// Borrows captured full framing and framing failures by physical collection address.
    pub(in crate::edit) fn contents(
        &self,
    ) -> &BTreeMap<PhysicalAddress, Result<CollectionContents, FormatError>> {
        &self.contents
    }
}

/// Original collection observations or the inputs retained after qualification failed.
pub(in crate::edit) enum OriginalQualification {
    /// Empty default candidates without predecessor observations.
    Empty,
    /// Original observations with independent source-domain and parent outcomes.
    Qualified(CollectionQualification),
    /// The original candidates and the failure that prevented qualification.
    Unavailable {
        candidates: CollectionCandidates,
        error: OriginalCollectionError,
    },
}

/// An unavailable source association or an error from original collection qualification.
pub(in crate::edit) enum OriginalCollectionError {
    /// Source-phase, session, or discovery checks failed.
    Qualification(Error),
    /// The original source association was unavailable.
    Source,
}

struct CollectionChecks {
    domain: Option<OwnerDomain>,
    parents: BTreeMap<PhysicalAddress, ParentQualification>,
}

/// Owned framing and candidate-allocation observations for one global heap collection.
///
/// Qualification requires matching candidate and framing ranges and a known allocation class.
/// References to members and permission to reuse the collection require separate evidence.
pub(in crate::edit) struct ParentStorage {
    /// The parsed full collection range, or its framing error.
    pub(in crate::edit) framing: Result<FileRange, FormatError>,
    /// The candidate range and its allocation class, if recorded.
    pub(in crate::edit) original: Option<(FileRange, Option<FreeClass>)>,
    /// The local framing, candidate-range, and allocation-class outcome.
    pub(in crate::edit) allocation: Result<(), ParentUnavailable>,
}

/// Owned predecessor observations and member and parent outcomes for one source phase.
///
/// Full framing and root referrers are borrowed through the original [`AnalysisPhase`].
/// Member qualification requires a complete observation for its origin and an available
/// source domain. Whole-parent qualification also requires complete root references and
/// matching allocated members. Source freshness and storage reuse require separate evidence.
pub(in crate::edit) struct CollectionQualification {
    source: ReconstructionSource,
    candidates: CollectionCandidates,
    domain: Option<OwnerDomain>,
    parents: BTreeMap<PhysicalAddress, ParentQualification>,
}

impl CollectionQualification {
    /// Borrows the common source-domain outcome, or returns [`None`] for empty default input.
    pub(in crate::edit) fn domain(&self) -> Option<&OwnerDomain> {
        self.domain.as_ref()
    }

    /// Borrows independent outcomes keyed by physical collection address.
    pub(in crate::edit) fn parents(&self) -> &BTreeMap<PhysicalAddress, ParentQualification> {
        &self.parents
    }

    /// Borrows original reference observations keyed by object and content scope.
    pub(in crate::edit) fn predecessors(
        &self,
    ) -> Option<&BTreeMap<CollectionProvenance, ObjectReferences>> {
        self.candidates
            .observations
            .as_ref()
            .map(|facts| &facts.predecessors)
    }

    /// Borrows deletion roots whose candidate discovery was incomplete.
    pub(in crate::edit) fn incomplete_deletions(&self) -> Option<&BTreeSet<ObjectId>> {
        self.candidates
            .observations
            .as_ref()
            .map(|facts| &facts.incomplete_deletions)
    }

    /// Returns the original candidate range and allocation class, if recorded.
    pub(in crate::edit) fn original_allocation(
        &self,
        address: PhysicalAddress,
    ) -> Option<(FileRange, Option<FreeClass>)> {
        self.candidates
            .collections
            .get(&address)
            .map(|candidate| (candidate.range, candidate.class))
    }

    /// Borrows the object and content scopes associated with a candidate allocation.
    pub(in crate::edit) fn allocation_origins(
        &self,
        address: PhysicalAddress,
    ) -> Option<&BTreeSet<CollectionProvenance>> {
        self.candidates
            .collections
            .get(&address)
            .map(|candidate| &candidate.origins)
    }

    /// Returns whether all retained observations and deletion discovery are complete.
    ///
    /// Returns `true` when predecessor observations are absent.
    pub(in crate::edit) fn predecessors_complete(&self) -> bool {
        self.candidates.observations.as_ref().is_none_or(|facts| {
            facts.incomplete_deletions.is_empty()
                && facts
                    .predecessors
                    .values()
                    .all(|references| references.complete)
        })
    }

    /// Checks allocated predecessor evidence for `id` under `origin`.
    ///
    /// Whole-parent qualification requires matching references for every allocated member.
    /// Prospective ownership and permission to remove this member require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns the first unmet prerequisite as [`MemberUnavailable`].
    pub(in crate::edit) fn member(
        &self,
        origin: &CollectionProvenance,
        id: GlobalHeapId,
    ) -> Result<(), MemberUnavailable> {
        let references = self
            .predecessors()
            .and_then(|facts| facts.get(origin))
            .filter(|facts| facts.objects.contains(&id))
            .ok_or(MemberUnavailable::Observation)?;
        if !references.complete {
            return Err(MemberUnavailable::Interpretation);
        }
        let parent = self
            .parents
            .get(&id.collection)
            .ok_or(MemberUnavailable::Observation)?;
        match parent.members.get(&id.index) {
            Some(MemberPresence::Allocated) => {}
            Some(MemberPresence::Missing) => return Err(MemberUnavailable::Missing),
            _ => return Err(MemberUnavailable::Unknown),
        }
        parent.allocation.map_err(MemberUnavailable::Parent)?;
        if !matches!(self.domain, Some(OwnerDomain::Available(_))) {
            return Err(MemberUnavailable::Domain);
        }
        Ok(())
    }

    /// Checks whole-parent evidence with the common source-domain prerequisite.
    ///
    /// Prospective ownership and permission to reuse the collection require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`ParentUnavailable::Allocation`] for an absent parent, the parent's local
    /// restriction, or [`ParentUnavailable::Domain`] if the source domain is unavailable.
    pub(in crate::edit) fn whole_parent(
        &self,
        address: PhysicalAddress,
    ) -> Result<(), ParentUnavailable> {
        self.parents
            .get(&address)
            .ok_or(ParentUnavailable::Allocation)?
            .whole?;
        if !matches!(self.domain, Some(OwnerDomain::Available(_))) {
            return Err(ParentUnavailable::Domain);
        }
        Ok(())
    }

    /// Borrows full framing from the original phase for a recorded parent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] for a phase mismatch or an absent parent.
    /// Returns the errors of [`AnalysisPhase::heap_members`] if framing failed.
    pub(in crate::edit) fn contents<'a>(
        &self,
        phase: &'a mut AnalysisPhase<'_>,
        address: PhysicalAddress,
    ) -> Result<&'a CollectionContents, FormatError> {
        phase.require_heap_phase(&self.source)?;
        if !self.parents.contains_key(&address) {
            return Err(FormatError::Internal(
                "collection is absent from qualification".into(),
            ));
        }
        phase.heap_members(&self.source, address)
    }

    /// Borrows original root referrers and member associations from the associated phase.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AnalysisPhase::member_associations`].
    pub(in crate::edit) fn associations<'a>(
        &self,
        phase: &'a mut AnalysisPhase<'_>,
    ) -> Result<&'a MemberAssociations, Error> {
        phase.member_associations(&self.source)
    }
}

/// Independent framing, selected-member presence, and local outcomes for one parent.
///
/// The common source-domain outcome is checked by [`CollectionQualification::member`] and
/// [`CollectionQualification::whole_parent`].
pub(in crate::edit) struct ParentQualification {
    framing: Result<FileRange, FormatError>,
    members: BTreeMap<NonZeroU16, MemberPresence>,
    allocation: Result<(), ParentUnavailable>,
    whole: Result<(), ParentUnavailable>,
}

impl ParentQualification {
    /// Returns the parsed parent range or borrows its framing failure.
    ///
    /// # Errors
    ///
    /// Returns the recorded errors of [`AnalysisPhase::collection_contents`].
    pub(in crate::edit) fn framing(&self) -> Result<FileRange, &FormatError> {
        self.framing.as_ref().copied()
    }

    /// Borrows allocation presence for the member indices in predecessor observations.
    pub(in crate::edit) fn members(&self) -> &BTreeMap<NonZeroU16, MemberPresence> {
        &self.members
    }

    /// Returns the local framing, candidate-range, and allocation-class outcome.
    ///
    /// # Errors
    ///
    /// Returns the recorded local restriction as [`ParentUnavailable`].
    pub(in crate::edit) fn allocation(&self) -> Result<(), ParentUnavailable> {
        self.allocation
    }

    /// Returns the local whole-parent outcome independently of the common source domain.
    ///
    /// # Errors
    ///
    /// Returns the recorded allocation or root-reference restriction as [`ParentUnavailable`].
    pub(in crate::edit) fn whole(&self) -> Result<(), ParentUnavailable> {
        self.whole
    }
}

/// An unmet prerequisite for original allocated-member evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum MemberUnavailable {
    /// The common source domain is unavailable.
    Domain,
    /// The selected origin's reference interpretation is incomplete.
    Interpretation,
    /// The selected index is absent from successfully parsed allocated members.
    Missing,
    /// The selected origin or member observation is absent.
    Observation,
    /// A local parent allocation restriction.
    Parent(ParentUnavailable),
    /// Allocation presence is undetermined after framing failure.
    Unknown,
}

/// An unmet local or source-domain prerequisite for collection evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum ParentUnavailable {
    /// The candidate allocation or requested parent is absent.
    Allocation,
    /// Root-scoped member associations are absent for the collection.
    Association,
    /// The candidate allocation class is unknown.
    Class,
    /// The common source domain is unavailable.
    Domain,
    /// Collection framing failed.
    Framing,
    /// A root-observed index is absent from parsed allocated members.
    MissingMember,
    /// The candidate allocation range differs from the parsed parent range.
    Range,
    /// Root topology or reference interpretation is incomplete.
    References,
    /// An allocated member is absent from the root-observed reference set.
    Unmatched,
}

#[cfg(test)]
mod tests;
