//! Original predecessor classifications and successful placement correspondence.
//!
//! The editor collects rebuilt groups and relocated datasets independently of count changes.
//! Complete translation requires supersession of every predecessor and agreement with the
//! applied tree's placements. The caller establishes source freshness, source-domain
//! applicability, storage protection, and publication permission separately.

use std::collections::{BTreeMap, BTreeSet};

use super::super::{ChildLink, PreparedTree};
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::{CountEvidence, CountEvidenceError, ROOT_REFERENCE_COUNT};
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::edit::commit::publish::AppliedTree;
use crate::error::Error;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

/// Original observations and expected placements for one prepared tree.
///
/// The editor records newly created groups with an absent predecessor and retains count
/// observations independently of each predecessor's classification.
pub(in crate::edit::commit) struct ReconstructionDispositions {
    source: Result<ReconstructionSource, Error>,
    predecessors: BTreeMap<ObjectPathBuf, Predecessor>,
    expected: BTreeMap<ObjectPathBuf, Option<ObjectId>>,
    placed: BTreeMap<ObjectPathBuf, ObjectId>,
    placement_error: Option<PlacementError>,
}

impl ReconstructionDispositions {
    /// Collects every rebuilt group's and relocated dataset's original association and counts.
    ///
    /// Retains source and count errors alongside withheld classifications. Requires a chain of
    /// superseded parents from the original root for each superseded non-root predecessor.
    pub(super) fn collect(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        tree: &PreparedTree,
    ) -> Self {
        let source = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(Error::from)
            .and_then(|root| {
                phase
                    .reconstruction_source(engine, root)
                    .map_err(Error::from)
            });
        let mut result = Self {
            source,
            predecessors: BTreeMap::new(),
            expected: BTreeMap::new(),
            placed: BTreeMap::new(),
            placement_error: None,
        };
        let mut seen = BTreeSet::new();
        for (path, node) in &tree.groups {
            result.expected.insert(
                path.clone(),
                node.prepared.replaces.as_ref().map(|old| old.object()),
            );
            if let Some(old) = &node.prepared.replaces {
                result.insert(engine, phase, tree, &mut seen, path.clone(), old.source());
            }
            for write in &node.prepared.writes {
                let path = path.join(&write.link);
                result
                    .expected
                    .insert(path.clone(), Some(write.source.object()));
                result.insert(engine, phase, tree, &mut seen, path, &write.source);
            }
        }
        result
    }

    /// Records one successful placement and retains the first event mismatch.
    ///
    /// Checks the path and optional predecessor against the prepared inputs. A newly created
    /// group has no predecessor. Repeated placements are mismatches even at the same address.
    pub(in crate::edit::commit) fn record(
        &mut self,
        path: &ObjectPathBuf,
        source: Option<ObjectId>,
        placed: ObjectId,
    ) {
        let Self {
            source: _,
            predecessors: _,
            expected,
            placed: placements,
            placement_error,
        } = self;
        let error = match expected.get(path) {
            None => Some(PlacementError::Unexpected(path.clone())),
            Some(expected) if expected != &source => Some(PlacementError::Source(path.clone())),
            Some(_) if placements.contains_key(path) => {
                Some(PlacementError::Repeated(path.clone()))
            }
            Some(_) => None,
        };
        if let Some(error) = error {
            placement_error.get_or_insert(error);
        } else {
            placements.insert(path.clone(), placed);
        }
    }

    /// Pairs the original observations with their placement reconciliation outcome.
    pub(super) fn finish(self, tree: &AppliedTree) -> AppliedDispositions {
        let result = self.reconcile(tree);
        AppliedDispositions {
            original: self,
            result,
        }
    }

    /// Borrows predecessor observations by their paths in the prepared tree.
    pub(in crate::edit::commit) fn predecessors(&self) -> &BTreeMap<ObjectPathBuf, Predecessor> {
        &self.predecessors
    }

    /// Borrows the original root's phase association or its resolution error.
    pub(in crate::edit::commit) fn source(&self) -> &Result<ReconstructionSource, Error> {
        &self.source
    }

    fn insert(
        &mut self,
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        tree: &PreparedTree,
        seen: &mut BTreeSet<ObjectId>,
        path: ObjectPathBuf,
        source: &ReconstructionSource,
    ) {
        let counts = phase.count_evidence(source.object()).map(|evidence| {
            evidence.map(|evidence| ObservedCounts {
                original: evidence.bound().original(),
                incoming: evidence.bound().incoming(),
                ordinary: matches!(evidence, CountEvidence::Ordinary(_)),
            })
        });
        let disposition = if !seen.insert(source.object()) {
            Disposition::Withhold(Withheld::RepeatedPredecessor(source.object()))
        } else if phase
            .reconstruction_source(engine, source.object())
            .as_ref()
            != Ok(source)
        {
            Disposition::Withhold(Withheld::Phase)
        } else {
            self.classify(phase, tree, &path, source, &counts)
        };
        if self.predecessors.contains_key(&path) {
            self.placement_error = Some(PlacementError::Repeated(path.clone()));
        }
        self.predecessors.insert(
            path,
            Predecessor {
                source: source.clone(),
                counts,
                disposition,
            },
        );
    }

    fn classify(
        &self,
        phase: &mut AnalysisPhase<'_>,
        tree: &PreparedTree,
        path: &ObjectPathBuf,
        source: &ReconstructionSource,
        counts: &Result<Option<ObservedCounts>, CountEvidenceError>,
    ) -> Disposition {
        let Some(topology) = phase.topology() else {
            return Disposition::Withhold(Withheld::Topology);
        };
        let Ok(root) = &self.source else {
            return Disposition::Withhold(Withheld::Source);
        };
        let Some((parent, leaf)) = path.split_leaf() else {
            if source != root {
                return Disposition::Withhold(Withheld::Source);
            }
            return match counts {
                Ok(Some(counts)) if counts.incoming != 0 => {
                    Disposition::Withhold(Withheld::RootLinks(counts.incoming))
                }
                Ok(Some(counts)) if counts.original == ROOT_REFERENCE_COUNT => {
                    Disposition::Supersede
                }
                Ok(Some(counts)) => Disposition::Withhold(Withheld::ExcessCount(counts.original)),
                Ok(None) => Disposition::Withhold(Withheld::CountUnavailable),
                Err(_) => Disposition::Withhold(Withheld::CountError),
            };
        };
        if topology.find_link(path).map(|(_, object)| object) != Some(source.object()) {
            return Disposition::Withhold(Withheld::Source);
        }
        let Some(owner) = self.predecessors.get(&parent) else {
            return Disposition::Withhold(Withheld::Parent(parent));
        };
        let parent_object = if parent.is_empty() {
            Some(root.object())
        } else {
            topology.find_link(&parent).map(|(_, object)| object)
        };
        if parent_object != Some(owner.source.object()) {
            return Disposition::Withhold(Withheld::Source);
        }
        let is_group = tree.groups.contains_key(path);
        if is_group
            && !tree.headers.children.get(&parent).is_some_and(|children| {
                children.iter().any(|child| {
                    matches!(child,
                ChildLink::Replace { path: child_path, leaf: child_leaf }
                    if child_path == path && child_leaf == &leaf)
                })
            })
        {
            return Disposition::Withhold(Withheld::Parent(parent));
        }
        match counts {
            Ok(Some(counts)) if counts.incoming != 1 || !counts.ordinary => {
                Disposition::Withhold(Withheld::CountUnavailable)
            }
            Ok(Some(counts)) if counts.original != 1 => {
                Disposition::Withhold(Withheld::ExcessCount(counts.original))
            }
            Ok(Some(_)) if owner.disposition == Disposition::Supersede => Disposition::Supersede,
            Ok(Some(_)) => Disposition::Withhold(Withheld::Parent(parent)),
            Ok(None) => Disposition::Withhold(Withheld::CountUnavailable),
            Err(_) => Disposition::Withhold(Withheld::CountError),
        }
    }

    /// Reconciles recorded placements before constructing the complete replacement map.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionWithheld::Placement`] for missing or mismatched events, relocations,
    /// or root placement, [`TransitionWithheld::Source`] for an unavailable original root, or
    /// [`TransitionWithheld::Predecessors`] if any predecessor is withheld.
    fn reconcile(
        &self,
        tree: &AppliedTree,
    ) -> Result<BTreeMap<ObjectId, ObjectId>, TransitionWithheld> {
        let Self {
            source,
            predecessors,
            expected,
            placed,
            placement_error,
        } = self;
        if let Some(error) = placement_error {
            return Err(TransitionWithheld::Placement(error.clone()));
        }
        for path in expected.keys() {
            if !placed.contains_key(path) {
                return Err(TransitionWithheld::Placement(PlacementError::Missing(
                    path.clone(),
                )));
            }
        }
        let replacements: BTreeMap<_, _> = predecessors
            .iter()
            .filter_map(|(path, old)| {
                placed
                    .get(path)
                    .map(|&placed| (old.source.object(), placed))
            })
            .collect();
        if replacements != tree.relocations {
            return Err(TransitionWithheld::Placement(PlacementError::Relocations));
        }
        if placed
            .get(&ObjectPathBuf::root())
            .map(|root| root.header_address())
            != Some(tree.retirement.root())
        {
            return Err(TransitionWithheld::Placement(PlacementError::Root));
        }
        if source.is_err() {
            return Err(TransitionWithheld::Source);
        }
        if predecessors
            .values()
            .any(|old| old.disposition != Disposition::Supersede)
        {
            return Err(TransitionWithheld::Predecessors);
        }
        Ok(replacements)
    }
}

/// Original observations paired with a successful tree's replacement-map outcome.
///
/// A withheld predecessor prevents complete translation even when every placement succeeded.
pub(in crate::edit::commit) struct AppliedDispositions {
    original: ReconstructionDispositions,
    result: Result<BTreeMap<ObjectId, ObjectId>, TransitionWithheld>,
}

impl AppliedDispositions {
    /// Borrows the original observations and recorded placement events.
    pub(in crate::edit::commit) fn original(&self) -> &ReconstructionDispositions {
        &self.original
    }

    /// Borrows the complete original-to-placed map after reconciliation.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionWithheld`] if placement reconciliation or predecessor supersession
    /// is incomplete.
    pub(in crate::edit::commit) fn replacements(
        &self,
    ) -> Result<&BTreeMap<ObjectId, ObjectId>, &TransitionWithheld> {
        self.result.as_ref()
    }

    /// Translates a replaced original identity, preserving identities absent from the map.
    ///
    /// # Errors
    ///
    /// Returns the error of [`Self::replacements`], including for an unchanged identity.
    pub(in crate::edit::commit) fn translate(
        &self,
        original: ObjectId,
    ) -> Result<ObjectId, &TransitionWithheld> {
        self.replacements()
            .map(|replacements| replacements.get(&original).copied().unwrap_or(original))
    }
}

/// One reconstructed object's original association, count outcome, and classification.
pub(in crate::edit::commit) struct Predecessor {
    pub(in crate::edit::commit) source: ReconstructionSource,
    /// Original count evidence, absent evidence, or its derivation error.
    pub(in crate::edit::commit) counts: Result<Option<ObservedCounts>, CountEvidenceError>,
    pub(in crate::edit::commit) disposition: Disposition,
}

/// Declared and incoming counts observed in the original source phase.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit::commit) struct ObservedCounts {
    /// The original declared reference count, including an implicit one.
    pub(in crate::edit::commit) original: u32,
    /// The number of hard-link edges targeting the object in the original topology.
    pub(in crate::edit::commit) incoming: u32,
    /// Whether the object was classified as [`CountEvidence::Ordinary`].
    ordinary: bool,
}

/// A predecessor's eligibility for replacement in the complete identity map.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum Disposition {
    /// A predecessor with compatible counts and a superseded path from the original root.
    Supersede,
    /// A predecessor excluded from complete translation, with its reason.
    Withhold(Withheld),
}

/// The reason an original predecessor is excluded from complete translation.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum Withheld {
    /// Derivation of the original count evidence failed.
    CountError,
    /// Complete count evidence for the required object classification is unavailable.
    CountUnavailable,
    /// The declared count differs from the required count of one.
    ExcessCount(u32),
    /// The parent is absent, retained, or missing its prepared child replacement.
    Parent(ObjectPathBuf),
    /// The predecessor belongs to a different source phase.
    Phase,
    /// The same original object occurs at another prepared predecessor path.
    RepeatedPredecessor(ObjectId),
    /// The original topology contains hard links to the root.
    RootLinks(u32),
    /// The original root association is unavailable, or a predecessor or parent association differs.
    Source,
    /// Complete original topology is unavailable.
    Topology,
}

/// The reason a complete original-to-placed identity map is unavailable.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum TransitionWithheld {
    /// Recorded placements disagree with the prepared inputs or applied tree.
    Placement(PlacementError),
    /// At least one original predecessor is withheld.
    Predecessors,
    /// Resolution of the original root association failed.
    Source,
}

/// A mismatch between expected placements, recorded events, and the applied tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum PlacementError {
    /// An expected group or relocated dataset has no placement event.
    Missing(ObjectPathBuf),
    /// The predecessor replacement map differs from the applied relocation map.
    Relocations,
    /// A prepared predecessor path or placement event occurs more than once.
    Repeated(ObjectPathBuf),
    /// The recorded root differs from the applied retirement root.
    Root,
    /// A placement event has a different optional predecessor.
    Source(ObjectPathBuf),
    /// A placement event has no corresponding prepared path.
    Unexpected(ObjectPathBuf),
}

#[cfg(test)]
mod tests;
