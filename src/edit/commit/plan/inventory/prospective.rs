//! Heap references decoded after tree application and retained-root association outcomes.

use std::collections::BTreeSet;

use hdf5_pure_core::PhysicalAddress;

use super::OriginalEdits;
use super::dispositions::AppliedDispositions;
use super::dispositions::TransitionWithheld;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::heap::ReachableHeapReferences;
use crate::edit::analysis::heap::qualification::OriginalCollections;
use crate::edit::analysis::heap::qualification::OriginalQualification;
use crate::object_graph::ObjectId;

/// Heap-reference observations from the applied tree and translated retained roots.
///
/// Root applicability and reference interpretation are independent outcomes. The caller
/// establishes original member qualification, storage protection, and mutation permission
/// separately. Empty original candidate input skips traversal.
#[derive(Default)]
pub(in crate::edit::commit) struct ProspectiveCollections {
    root: Option<PhysicalAddress>,
    roots: BTreeSet<ObjectId>,
    unavailable: Option<ProspectiveUnavailable>,
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
        let unavailable = Self::retained_roots(engine, original, dispositions, &mut roots).err();
        let references = Some(
            AnalysisPhase::prospective(engine, root)
                .heap_references_reachable_from(roots.iter().copied()),
        );
        Self {
            root: Some(root),
            roots,
            unavailable,
            references,
        }
    }

    /// Returns the applied root used for traversal, or [`None`] for empty original input.
    pub(in crate::edit::commit) fn root(&self) -> Option<PhysicalAddress> {
        self.root
    }

    /// Borrows the distinct physical roots supplied to the reference traversal.
    ///
    /// The set can be partial when retained-root applicability failed.
    pub(in crate::edit::commit) fn roots(&self) -> &BTreeSet<ObjectId> {
        &self.roots
    }

    /// Borrows the failure that prevented complete retained-root association.
    pub(in crate::edit::commit) fn unavailable(&self) -> Option<&ProspectiveUnavailable> {
        self.unavailable.as_ref()
    }

    /// Borrows decoded heap references and their traversal outcome.
    ///
    /// Returns [`None`] for empty original candidate input.
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

    fn retained_roots(
        engine: &WriteEngine,
        original: &OriginalEdits,
        dispositions: &AppliedDispositions,
        roots: &mut BTreeSet<ObjectId>,
    ) -> Result<(), ProspectiveUnavailable> {
        let owners = original.owners().ok_or(ProspectiveUnavailable::Original)?;
        if dispositions.original().source().as_ref().ok() != Some(owners.source()) {
            return Err(ProspectiveUnavailable::Source);
        }
        let storage = owners.facts().storage();
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

/// The reason complete retained-root applicability is unavailable.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum ProspectiveUnavailable {
    /// Original predecessors or actual placements prevent complete identity translation.
    Dispositions(TransitionWithheld),
    /// Original count and owner qualification is unavailable.
    Original,
    /// Original storage belongs to another allocation session.
    Session,
    /// Original owner facts and dispositions have different source associations.
    Source,
}
