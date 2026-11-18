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
use super::associations::MemberPresence;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::physical::claims::domain::{
    DomainUnavailable, residual::ResidualGroups,
};
use crate::edit::analysis::physical::claims::domain::{HeapEnvelopeDomain, OwnerDomain};
use crate::edit::analysis::{
    AnalysisPhase, CollectionCandidates, CollectionProvenance, ObjectReferences,
    ReconstructionSource,
};
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;

impl AnalysisPhase<'_> {
    /// Collects checked parent ranges for storage exclusion.
    ///
    /// Includes candidate parents and parents in predecessor observations. Reads framing through
    /// the phase cache independently of member associations and whole-parent qualification.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] for a phase or session mismatch.
    /// A parent whose framing fails has no parsed range.
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
                    .ok()
                    .map(CollectionContents::range);
                let original = candidates
                    .collections
                    .get(&address)
                    .map(|allocation| (allocation.range, allocation.class))
                    .or_else(|| {
                        self.collection_envelopes
                            .get(&address)
                            .map(|&range| (range, None))
                    });
                (
                    address,
                    ParentStorage {
                        framing,
                        original,
                        allocation_matches: matching_allocation(framing, original),
                    },
                )
            })
            .collect())
    }

    /// Collects owned original qualification and full framing for use outside this phase.
    ///
    /// Retains predecessor references when the source is absent or qualification fails. Moves
    /// successful framing for candidate and observed parents out of this phase when the source
    /// matches. Empty default input skips source and domain discovery.
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
        let checks = source.and_then(|source| {
            self.inspect_collections(source, &candidates)
                .ok()
                .map(|checks| (source.clone(), checks))
        });
        let mut contents = BTreeMap::new();
        if let Some(source) = source
            && self.require_heap_phase(source).is_ok()
        {
            for address in addresses {
                if self.collection_contents(address).is_ok()
                    && let Some(Ok(framing)) = self.allocations.remove(&address)
                {
                    contents.insert(address, framing);
                }
            }
        }
        let qualification = match checks {
            Some((source, checks)) => OriginalQualification::Qualified(CollectionQualification {
                source,
                candidates,
                domain: checks.domain,
                heap_envelopes: checks.heap_envelopes,
                residual_groups: checks.residual_groups,
                parents: checks.parents,
            }),
            None => OriginalQualification::Unavailable {
                predecessors: candidates.observations.map(|facts| facts.predecessors),
            },
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
        let heap_envelopes = if matches!(
            &domain,
            Some(OwnerDomain::Unavailable(DomainUnavailable::Collection))
        ) {
            self.heap_envelope_domain(source)?
        } else {
            None
        };
        let residual_groups = if matches!(
            &domain,
            Some(OwnerDomain::Unavailable(DomainUnavailable::Residual))
        ) {
            Some(self.residual_group_domain(source)?)
        } else {
            None
        };
        let mut qualified = BTreeMap::new();
        for (address, mut members) in parents {
            let framing = self.collection_contents(address).ok().map(|contents| {
                for member in contents.members() {
                    if let Some(presence) = members.get_mut(&member.index()) {
                        *presence = MemberPresence::Allocated;
                    }
                }
                contents.range()
            });
            let candidate = candidates
                .collections
                .get(&address)
                .map(|allocation| (allocation.range, allocation.class));
            let allocation = matching_allocation(framing, candidate);
            let associations = self.member_associations(source)?;
            let whole = allocation
                && associations.topology_available()
                && associations.references_complete()
                && associations
                    .collections()
                    .get(&address)
                    .is_some_and(|collection| {
                        collection.is_framed()
                            && collection
                                .observed()
                                .values()
                                .all(|&presence| presence == MemberPresence::Allocated)
                            && !collection.has_unmatched()
                    });
            qualified.insert(
                address,
                ParentQualification {
                    members,
                    allocation,
                    whole,
                },
            );
        }
        Ok(CollectionChecks {
            domain,
            heap_envelopes,
            residual_groups,
            parents: qualified,
        })
    }
}

fn matching_allocation(
    framing: Option<FileRange>,
    candidate: Option<(FileRange, Option<FreeClass>)>,
) -> bool {
    matches!(candidate, Some((range, Some(_))) if framing == Some(range))
}

/// Original candidate qualification and full parent framing retained outside the source phase.
///
/// Parent framing can be available even when qualification fails. Source freshness,
/// prospective ownership, and permission to mutate or reuse storage require separate evidence.
pub(in crate::edit) struct OriginalCollections {
    qualification: OriginalQualification,
    contents: BTreeMap<PhysicalAddress, CollectionContents>,
}

impl OriginalCollections {
    /// Borrows bounded residual roots captured with successful collection qualification.
    pub(in crate::edit) fn residual_groups(&self) -> Option<&ResidualGroups> {
        match &self.qualification {
            OriginalQualification::Qualified(qualification) => qualification.residual_groups(),
            _ => None,
        }
    }

    /// Borrows the original qualification outcome.
    pub(in crate::edit) fn qualification(&self) -> &OriginalQualification {
        &self.qualification
    }

    /// Borrows original predecessor references independently of qualification success.
    pub(in crate::edit) fn predecessors(
        &self,
    ) -> Option<&BTreeMap<CollectionProvenance, ObjectReferences>> {
        match &self.qualification {
            OriginalQualification::Empty => None,
            OriginalQualification::Qualified(qualification) => qualification.predecessors(),
            OriginalQualification::Unavailable { predecessors } => predecessors.as_ref(),
        }
    }

    /// Borrows successfully captured parent framing by physical collection address.
    pub(in crate::edit) fn contents(&self) -> &BTreeMap<PhysicalAddress, CollectionContents> {
        &self.contents
    }
}

/// Original collection qualification and retained predecessor references.
pub(in crate::edit) enum OriginalQualification {
    /// Empty default candidates without predecessor observations.
    Empty,
    /// Original observations with independent source-domain and parent outcomes.
    Qualified(CollectionQualification),
    /// Predecessor references retained without qualified allocation evidence.
    Unavailable {
        predecessors: Option<BTreeMap<CollectionProvenance, ObjectReferences>>,
    },
}

struct CollectionChecks {
    domain: Option<OwnerDomain>,
    heap_envelopes: Option<HeapEnvelopeDomain>,
    residual_groups: Option<Arc<Option<ResidualGroups>>>,
    parents: BTreeMap<PhysicalAddress, ParentQualification>,
}

/// Owned framing and candidate-allocation observations for one global heap collection.
///
/// Qualification requires matching candidate and framing ranges and a known allocation class.
/// References to members and permission to reuse the collection require separate evidence.
pub(in crate::edit) struct ParentStorage {
    /// The full collection range when framing succeeds.
    pub(in crate::edit) framing: Option<FileRange>,
    /// The candidate range and its allocation class, if recorded.
    pub(in crate::edit) original: Option<(FileRange, Option<FreeClass>)>,
    /// Whether the framing and original allocation agree and the allocation class is known.
    pub(in crate::edit) allocation_matches: bool,
}

/// Owned predecessor observations and member and parent outcomes for one source phase.
///
/// Member qualification requires a complete observation for its origin and an available
/// source domain. Whole-parent qualification also requires complete root references and
/// matching allocated members. Source freshness and storage reuse require separate evidence.
pub(in crate::edit) struct CollectionQualification {
    source: ReconstructionSource,
    candidates: CollectionCandidates,
    domain: Option<OwnerDomain>,
    heap_envelopes: Option<HeapEnvelopeDomain>,
    residual_groups: Option<Arc<Option<ResidualGroups>>>,
    parents: BTreeMap<PhysicalAddress, ParentQualification>,
}

impl CollectionQualification {
    /// Borrows the associated bounded residual interpretation, if all intervals were admitted.
    pub(in crate::edit) fn residual_groups(&self) -> Option<&ResidualGroups> {
        self.residual_groups.as_ref()?.as_ref().as_ref()
    }

    /// Borrows the original source association used for qualification.
    pub(in crate::edit) fn source(&self) -> &ReconstructionSource {
        &self.source
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

    /// Returns whether `id` has complete predecessor references and matched allocation evidence.
    ///
    /// Prospective ownership and permission to remove this member require separate evidence.
    pub(in crate::edit) fn member(&self, origin: &CollectionProvenance, id: GlobalHeapId) -> bool {
        self.predecessors()
            .and_then(|facts| facts.get(origin))
            .is_some_and(|references| references.complete && references.objects.contains(&id))
            && self.parents.get(&id.collection).is_some_and(|parent| {
                parent.allocation
                    && parent.members.get(&id.index) == Some(&MemberPresence::Allocated)
            })
            && self.domain_available()
    }

    /// Returns whether the entire parent has complete associated allocation evidence.
    ///
    /// Prospective absence and permission to reuse the collection require separate evidence.
    pub(in crate::edit) fn whole_parent(&self, address: PhysicalAddress) -> bool {
        self.parents
            .get(&address)
            .is_some_and(|parent| parent.whole)
            && self.domain_available()
    }

    /// Returns the complete allocation of an eligible parent with its source and session.
    ///
    /// Prospective absence, publication, and permission to release protected storage require
    /// separate evidence.
    pub(in crate::edit) fn whole_parent_allocation(
        &self,
        address: PhysicalAddress,
    ) -> Option<QualifiedParentAllocation> {
        if !self.whole_parent(address) {
            return None;
        }
        let allocation = self.candidates.collections.get(&address)?;
        let observations = self.candidates.observations.as_ref()?;
        Some(QualifiedParentAllocation {
            source: self.source.clone(),
            session: observations.session.clone(),
            range: allocation.range,
            class: allocation.class?,
        })
    }

    fn domain_available(&self) -> bool {
        matches!(self.domain, Some(OwnerDomain::Available(_)))
            || self.heap_envelopes.is_some()
            || self.residual_groups().is_some()
    }
}

/// A complete original parent allocation with qualified source, session, and free-space class.
///
/// [`CollectionQualification::whole_parent_allocation`] establishes whole-parent qualification.
/// The caller establishes prospective absence and publication or storage-release permission
/// separately.
pub(in crate::edit) struct QualifiedParentAllocation {
    source: ReconstructionSource,
    session: AllocationSession,
    range: FileRange,
    class: FreeClass,
}

impl QualifiedParentAllocation {
    /// Borrows the original source-phase and object association.
    pub(in crate::edit) fn source(&self) -> &ReconstructionSource {
        &self.source
    }

    /// Borrows the allocation session associated with the original range.
    pub(in crate::edit) fn session(&self) -> &AllocationSession {
        &self.session
    }

    /// Returns the complete original parent allocation range.
    pub(in crate::edit) fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the original allocation's free-space class.
    pub(in crate::edit) fn class(&self) -> FreeClass {
        self.class
    }
}

/// Selected-member presence and local allocation outcomes for one parent.
///
/// The common source-domain outcome is checked by [`CollectionQualification::member`] and
/// [`CollectionQualification::whole_parent`].
pub(in crate::edit) struct ParentQualification {
    members: BTreeMap<NonZeroU16, MemberPresence>,
    allocation: bool,
    whole: bool,
}
