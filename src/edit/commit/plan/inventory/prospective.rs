//! Heap references decoded after tree application and retained-root association outcomes.

use std::collections::BTreeSet;

use hdf5_pure_core::PhysicalAddress;

use super::OriginalEdits;
use super::dispositions::AppliedDispositions;
use super::dispositions::TransitionWithheld;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::ReachableHeapReferences;
use crate::edit::analysis::heap::qualification::OriginalCollections;
use crate::edit::analysis::heap::qualification::OriginalQualification;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

/// Heap-reference observations from the applied tree and translated count and residual roots.
///
/// Root applicability and reference interpretation are independent outcomes. The caller
/// establishes original member qualification, storage protection, and mutation permission
/// separately. The collector skips traversal for empty original candidate input.
#[derive(Default)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the owned state retains the checked evidence until its consuming boundary"
    )
)]
pub(in crate::edit::commit) struct ProspectiveCollections {
    root: Option<PhysicalAddress>,
    /// The allocation session used for the applied-tree observations.
    session: Option<AllocationSession>,
    /// The original source association retained with the prospective observations.
    source: Option<ReconstructionSource>,
    roots: BTreeSet<ObjectId>,
    unavailable: Option<ProspectiveUnavailable>,
    references: Option<ReachableHeapReferences>,
}

impl ProspectiveCollections {
    /// Collects references in a new phase after tree placement and in-place writes.
    ///
    /// Adds translated count-retained and residual-group roots to the applied root before one
    /// reference traversal.
    /// Records retained-root association failures independently of the decoded references.
    pub(super) fn collect(
        engine: &WriteEngine,
        original: &OriginalEdits,
        dispositions: &AppliedDispositions,
        collections: &OriginalCollections,
        root: PhysicalAddress,
    ) -> Self {
        if matches!(collections.qualification(), OriginalQualification::Empty) {
            return Self::default();
        }
        let mut roots = BTreeSet::from([ObjectId::new(root)]);
        let unavailable = Self::retained_roots(engine, original, dispositions, &mut roots)
            .and_then(|()| {
                Self::residual_roots(engine, original, dispositions, collections, &mut roots)
            })
            .err();
        let references = Some(
            AnalysisPhase::prospective(engine, root)
                .heap_references_reachable_from(roots.iter().copied()),
        );
        Self {
            root: Some(root),
            session: Some(engine.context.allocations.session()),
            source: original.source().cloned(),
            roots,
            unavailable,
            references,
        }
    }

    /// Returns the applied root used for traversal, or [`None`] for empty original input.
    pub(in crate::edit::commit) fn root(&self) -> Option<PhysicalAddress> {
        self.root
    }

    /// Borrows the original source association retained with prospective observations.
    ///
    /// Returns [`None`] if no source association was captured.
    pub(super) fn source(&self) -> Option<&ReconstructionSource> {
        self.source.as_ref()
    }

    /// Borrows the allocation session used for prospective observations.
    ///
    /// Returns [`None`] for empty original candidate input.
    pub(super) fn session(&self) -> Option<&AllocationSession> {
        self.session.as_ref()
    }

    /// Borrows the distinct physical roots supplied to the reference traversal.
    ///
    /// The set can be partial when retained-root applicability failed.
    #[cfg(test)]
    pub(in crate::edit::commit) fn roots(&self) -> &BTreeSet<ObjectId> {
        &self.roots
    }

    /// Borrows the failure that prevented complete retained-root association.
    #[cfg(test)]
    pub(in crate::edit::commit) fn unavailable(&self) -> Option<&ProspectiveUnavailable> {
        self.unavailable.as_ref()
    }

    /// Borrows decoded heap references and their traversal outcome.
    ///
    /// Returns [`None`] for empty original candidate input.
    #[cfg(test)]
    pub(in crate::edit::commit) fn references(&self) -> Option<&ReachableHeapReferences> {
        self.references.as_ref()
    }

    /// Returns whether root applicability, reference traversal, and interpretation completed.
    pub(in crate::edit::commit) fn is_complete(&self) -> bool {
        self.unavailable.is_none()
            && self
                .references
                .as_ref()
                .is_some_and(ReachableHeapReferences::is_complete)
    }

    /// Returns prospective reference presence for one physical member.
    pub(in crate::edit::commit) fn member_presence(&self, id: GlobalHeapId) -> ReferencePresence {
        self.presence(
            self.references
                .as_ref()
                .is_some_and(|references| references.objects().contains(&id)),
        )
    }

    /// Returns prospective reference presence for the collection at `address`.
    pub(in crate::edit::commit) fn parent_presence(
        &self,
        address: PhysicalAddress,
    ) -> ReferencePresence {
        self.presence(
            self.references
                .as_ref()
                .is_some_and(|references| references.collections().contains(&address.get())),
        )
    }

    /// Returns complete parent-absence evidence bound to the root, session, and original source.
    ///
    /// Returns [`None`] if a reference was observed, complete observations are unavailable, or a
    /// required association is absent. Allocation ownership and publication are checked separately.
    pub(in crate::edit::commit) fn parent_absence(
        &self,
        address: PhysicalAddress,
    ) -> Option<ParentAbsence> {
        if self.parent_presence(address) != ReferencePresence::Unreferenced {
            return None;
        }
        Some(ParentAbsence {
            address,
            root: self.root?,
            session: self.session.as_ref()?.clone(),
            source: self.source.as_ref()?.clone(),
        })
    }

    /// Classifies an observation before considering completeness.
    fn presence(&self, observed: bool) -> ReferencePresence {
        if observed {
            ReferencePresence::ObservedLive
        } else if self.is_complete() {
            ReferencePresence::Unreferenced
        } else {
            ReferencePresence::Unavailable
        }
    }

    /// Adds translated residual-group roots after checking source and session associations.
    ///
    /// Earlier roots remain if a later translation fails.
    ///
    /// # Errors
    ///
    /// Returns [`ProspectiveUnavailable::Source`] for incompatible original source associations,
    /// [`ProspectiveUnavailable::Session`] for a different allocation session, or
    /// [`ProspectiveUnavailable::Dispositions`] for failed placement reconciliation or translation.
    fn residual_roots(
        engine: &WriteEngine,
        original: &OriginalEdits,
        dispositions: &AppliedDispositions,
        collections: &OriginalCollections,
        roots: &mut BTreeSet<ObjectId>,
    ) -> Result<(), ProspectiveUnavailable> {
        let Some(groups) = collections.residual_groups() else {
            return Ok(());
        };
        if Some(groups.source()) != original.source()
            || dispositions.original().source().as_ref().ok() != Some(groups.source())
        {
            return Err(ProspectiveUnavailable::Source);
        }
        if !groups.session().same(&engine.context.allocations.session()) {
            return Err(ProspectiveUnavailable::Session);
        }
        for &object in groups.roots() {
            roots.insert(
                dispositions
                    .translate(object)
                    .map_err(|error| ProspectiveUnavailable::Dispositions(error.clone()))?,
            );
        }
        Ok(())
    }

    /// Adds translated original reference roots after checking count and storage facts.
    ///
    /// Checks source identity, allocation session, and placement reconciliation independently of
    /// owner-domain qualification. Earlier roots remain if a later translation fails. Removal
    /// permission requires separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`ProspectiveUnavailable::Original`] for unavailable count or storage facts,
    /// [`ProspectiveUnavailable::Source`] for a missing or incompatible source association,
    /// [`ProspectiveUnavailable::Session`] for a different allocation session, or
    /// [`ProspectiveUnavailable::Dispositions`] for failed reconciliation or translation.
    fn retained_roots(
        engine: &WriteEngine,
        original: &OriginalEdits,
        dispositions: &AppliedDispositions,
        roots: &mut BTreeSet<ObjectId>,
    ) -> Result<(), ProspectiveUnavailable> {
        let facts = original.facts().ok_or(ProspectiveUnavailable::Original)?;
        let source = original.source().ok_or(ProspectiveUnavailable::Source)?;
        if dispositions.original().source().as_ref().ok() != Some(source) {
            return Err(ProspectiveUnavailable::Source);
        }
        let storage = facts.storage();
        if !storage.same_session(engine) {
            return Err(ProspectiveUnavailable::Session);
        }
        dispositions
            .replacements()
            .map_err(|error| ProspectiveUnavailable::Dispositions(error.clone()))?;
        for &object in storage.reference_roots() {
            roots.insert(
                dispositions
                    .translate(object)
                    .map_err(|error| ProspectiveUnavailable::Dispositions(error.clone()))?,
            );
        }
        Ok(())
    }
}

/// Complete prospective absence for one physical collection address.
///
/// The collector retains the prospective root, allocation session, and original source
/// association. Original allocation qualification and permission to retire storage are separate.
pub(in crate::edit::commit) struct ParentAbsence {
    address: PhysicalAddress,
    root: PhysicalAddress,
    session: AllocationSession,
    source: ReconstructionSource,
}

impl ParentAbsence {
    /// Returns the collection address proved absent from the prospective references.
    pub(in crate::edit::commit) fn address(&self) -> PhysicalAddress {
        self.address
    }

    /// Returns the prospective root used to prove absence.
    pub(in crate::edit::commit) fn root(&self) -> PhysicalAddress {
        self.root
    }

    /// Borrows the allocation session used for the prospective observations.
    pub(in crate::edit::commit) fn session(&self) -> &AllocationSession {
        &self.session
    }

    /// Borrows the original source association retained with the prospective observations.
    pub(in crate::edit::commit) fn source(&self) -> &ReconstructionSource {
        &self.source
    }
}

/// The observed presence or proven absence of a prospective heap reference.
///
/// Observed references are retained even when other root or reference observations are incomplete.
/// Complete applicable observations are required to prove absence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum ReferencePresence {
    /// A reference was observed in the applied tree or a translated retained root.
    ObservedLive,
    /// No reference was observed, and complete root or reference evidence is unavailable.
    Unavailable,
    /// No reference was observed in complete, applicable prospective observations.
    Unreferenced,
}

/// The reason complete retained-root applicability is unavailable.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum ProspectiveUnavailable {
    /// Original predecessors or actual placements prevent complete identity translation.
    Dispositions(TransitionWithheld),
    /// Original count or storage facts are unavailable.
    Original,
    /// Retained original evidence belongs to another allocation session.
    Session,
    /// An original source association is absent or differs across retained evidence.
    Source,
}
