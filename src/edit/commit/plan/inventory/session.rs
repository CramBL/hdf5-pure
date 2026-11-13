//! Exclusive session ownership retained through checked whole-parent publication.

use std::collections::{BTreeMap, BTreeSet};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use super::dispositions::{AppliedDispositions, Disposition, ReconstructionDispositions};
use super::prospective::{ProspectiveCollections, ReferencePresence};
use super::storage::{OriginalStorage, StorageOrigin};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::heap::qualification::OriginalCollections;
use crate::edit::analysis::{
    AnalysisPhase, CollectionContent, CollectionProvenance, ReconstructionSource,
};
use crate::edit::staging::StagedEdits;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

impl ReconstructionDispositions {
    /// Captures complete original parents belonging exclusively to deferred dataset overwrites.
    ///
    /// Requires matching source phases, a superseded owner, references to every allocated member,
    /// and an exact protected range with a known allocation class. Skips operations that may
    /// share references and parents recorded more than once. Reuses captured framing and references
    /// without another topology or payload traversal.
    pub(super) fn capture_session_parents(
        &mut self,
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        staged: &StagedEdits,
        collections: &OriginalCollections,
        storage: &OriginalStorage,
    ) {
        if engine.vl_overwrite_heaps.is_empty() || staged.shares_heap_references() {
            return;
        }
        let Ok(source) = self.source() else {
            return;
        };
        if phase
            .reconstruction_source(engine, source.object())
            .as_ref()
            != Ok(source)
        {
            return;
        }
        let mut parents = BTreeMap::new();
        let mut repeated = BTreeSet::new();
        for (path, values) in staged.writes() {
            if values.vl_string_staging().is_none() {
                continue;
            }
            let Some(predecessor) = self.predecessors().get(path) else {
                continue;
            };
            if predecessor.disposition != Disposition::Supersede
                || phase
                    .reconstruction_source(engine, predecessor.source.object())
                    .as_ref()
                    != Ok(&predecessor.source)
            {
                continue;
            }
            let Some(ranges) = engine.vl_overwrite_heaps.get(path) else {
                continue;
            };
            let origin = CollectionProvenance {
                object: predecessor.source.object(),
                content: CollectionContent::DatasetValues,
            };
            let Some(references) = collections
                .predecessors()
                .and_then(|facts| facts.get(&origin))
            else {
                continue;
            };
            if !references.is_complete() {
                continue;
            }
            for &range in ranges {
                let address = range.start();
                let Some(contents) = collections
                    .contents()
                    .get(&address)
                    .and_then(|contents| contents.as_ref().ok())
                else {
                    continue;
                };
                if contents.range() != range
                    || contents.members().iter().any(|member| {
                        !references.objects().contains(&GlobalHeapId {
                            collection: address,
                            index: member.index(),
                        })
                    })
                {
                    continue;
                }
                let Some(class) =
                    storage
                        .allocations()
                        .iter()
                        .find_map(|allocation| match allocation.origin {
                            StorageOrigin::Parent {
                                address: actual,
                                class,
                            } if actual == address && allocation.range == range => class,
                            _ => None,
                        })
                else {
                    continue;
                };
                if parents
                    .insert(
                        address,
                        SessionParent {
                            source: source.clone(),
                            session: engine.context.allocations.session(),
                            owner: predecessor.source.object(),
                            range,
                            class,
                        },
                    )
                    .is_some()
                {
                    repeated.insert(address);
                }
            }
        }
        parents.retain(|address, _| !repeated.contains(address));
        self.session_parents = parents;
    }
}

/// Exclusive session ownership with the original owner, protected capacity, and source association.
pub(super) struct SessionParent {
    source: ReconstructionSource,
    session: AllocationSession,
    owner: ObjectId,
    range: FileRange,
    class: FreeClass,
}

impl SessionParent {
    /// Returns retirement evidence after checked replacement of the captured owner.
    ///
    /// Requires matching original source and allocation session, a reconciled replacement with a
    /// different owner identity, and a resolved prospective root. Returns [`None`] for an observed
    /// prospective reference to the parent. Accepts incomplete prospective interpretation.
    pub(super) fn retirement(
        &self,
        dispositions: &AppliedDispositions,
        prospective: &ProspectiveCollections,
    ) -> Option<SessionParentRetirement> {
        let Self {
            source,
            session,
            owner,
            range,
            class,
        } = self;
        if dispositions.original().source().as_ref().ok() != Some(source)
            || prospective.source() != Some(source)
            || !prospective.session()?.same(session)
            || prospective.parent_presence(range.start()) == ReferencePresence::ObservedLive
            || dispositions
                .replacements()
                .ok()?
                .get(owner)
                .is_none_or(|&placed| placed == *owner)
        {
            return None;
        }
        Some(SessionParentRetirement {
            source: source.clone(),
            session: session.clone(),
            range: *range,
            class: *class,
            root: prospective.root()?,
        })
    }
}

/// Session-owned capacity bound to a reconciled owner replacement and prospective root.
///
/// The caller checks publication, storage protection, and allocation release separately.
pub(in crate::edit::commit) struct SessionParentRetirement {
    source: ReconstructionSource,
    session: AllocationSession,
    range: FileRange,
    class: FreeClass,
    root: PhysicalAddress,
}

impl SessionParentRetirement {
    /// Borrows the original root's source-phase association.
    pub(super) fn source(&self) -> &ReconstructionSource {
        &self.source
    }

    /// Borrows the allocation session matched with the prospective observations.
    pub(super) fn session(&self) -> &AllocationSession {
        &self.session
    }

    /// Returns the complete original parent allocation.
    pub(super) fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the original parent allocation's free-space class.
    pub(super) fn class(&self) -> FreeClass {
        self.class
    }

    /// Returns the prospective root associated with the reconciled owner replacement.
    pub(super) fn root(&self) -> PhysicalAddress {
        self.root
    }
}

#[cfg(test)]
mod tests;
