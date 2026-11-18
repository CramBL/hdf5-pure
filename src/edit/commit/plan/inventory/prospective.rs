//! Heap references decoded after tree application and retained-root association outcomes.

use std::collections::BTreeSet;

use hdf5_pure_core::PhysicalAddress;

use super::OriginalEdits;
use super::dispositions::AppliedDispositions;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::ReachableHeapReferences;
use crate::edit::analysis::heap::qualification::OriginalCollections;
use crate::edit::analysis::heap::qualification::OriginalQualification;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

/// Heap-reference observations from the applied tree and translated retained roots.
///
/// Root applicability and reference interpretation are independent outcomes. The caller
/// establishes original member qualification, storage protection, and mutation permission
/// separately. Empty original candidate input skips traversal.
#[derive(Default)]
pub(in crate::edit::commit) struct ProspectiveCollections {
    root: Option<PhysicalAddress>,
    session: Option<AllocationSession>,
    source: Option<ReconstructionSource>,
    applicable: bool,
    references: Option<ReachableHeapReferences>,
}

impl ProspectiveCollections {
    /// Collects references in a new phase after tree placement and in-place writes.
    ///
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
        let applicable = Self::retained_roots(engine, original, dispositions, &mut roots)
            .and_then(|()| {
                Self::residual_roots(engine, original, dispositions, collections, &mut roots)
            })
            .is_some();
        let references = Some(
            AnalysisPhase::prospective(engine, root)
                .heap_references_reachable_from(roots.iter().copied()),
        );
        Self {
            root: Some(root),
            session: Some(engine.context.allocations.session()),
            source: original.source().cloned(),
            applicable,
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

    /// Returns whether root applicability, reference traversal, and interpretation completed.
    pub(in crate::edit::commit) fn is_complete(&self) -> bool {
        self.applicable
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

    fn residual_roots(
        engine: &WriteEngine,
        original: &OriginalEdits,
        dispositions: &AppliedDispositions,
        collections: &OriginalCollections,
        roots: &mut BTreeSet<ObjectId>,
    ) -> Option<()> {
        let Some(groups) = collections.residual_groups() else {
            return Some(());
        };
        if Some(groups.source()) != original.source()
            || dispositions.source() != Some(groups.source())
        {
            return None;
        }
        if !groups.session().same(&engine.context.allocations.session()) {
            return None;
        }
        for &object in groups.roots() {
            roots.insert(dispositions.translate(object)?);
        }
        Some(())
    }

    fn retained_roots(
        engine: &WriteEngine,
        original: &OriginalEdits,
        dispositions: &AppliedDispositions,
        roots: &mut BTreeSet<ObjectId>,
    ) -> Option<()> {
        let facts = original.facts()?;
        let source = original.source()?;
        if dispositions.source() != Some(source) {
            return None;
        }
        let storage = facts.storage();
        if !storage.same_session(engine) {
            return None;
        }
        dispositions.replacements()?;
        roots.extend(dispositions.retained().iter().copied());
        for &object in storage.reference_roots() {
            roots.insert(dispositions.translate(object)?);
        }
        Some(())
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
