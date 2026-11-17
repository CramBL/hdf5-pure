//! Original predecessor classifications and successful placement correspondence.
//!
//! The editor collects rebuilt groups and relocated datasets independently of count changes.
//! Complete translation requires supersession of every predecessor and agreement with the
//! applied tree's placements. The caller establishes source freshness, source-domain
//! applicability, storage protection, and publication permission separately.

use std::collections::{BTreeMap, BTreeSet};

use super::super::ChildLink;
use super::super::Node;
use super::session::SessionParent;
use crate::PhysicalAddress;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::{CountEvidence, CountEvidenceError, ROOT_REFERENCE_COUNT};
use crate::edit::analysis::{AnalysisPhase, OriginalOwnerFacts, ReconstructionSource};
use crate::edit::commit::publish::AppliedTree;
use crate::error::Error;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

pub(in crate::edit::commit) use placements::PlacementCorrespondence;

mod placements;

/// Original observations and expected placements for one prepared tree.
///
/// The editor records newly created groups with an absent predecessor and retains count
/// observations independently of each predecessor's classification.
pub(in crate::edit::commit) struct ReconstructionDispositions {
    source: Result<ReconstructionSource, Error>,
    /// Complete protected parents captured from exclusive session overwrites.
    pub(super) session_parents: BTreeMap<PhysicalAddress, SessionParent>,
    predecessors: BTreeMap<ObjectPathBuf, Predecessor>,
    placements: PlacementCorrespondence,
}

impl ReconstructionDispositions {
    /// Collects every rebuilt group's and relocated dataset's original association and counts.
    ///
    /// Retains source and count errors alongside withheld classifications. Requires a chain of
    /// superseded parents from the original root for each superseded non-root predecessor. Uses
    /// completed local root-count projection when all original incoming root links are removed.
    pub(super) fn collect(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        groups: &BTreeMap<ObjectPathBuf, Node>,
        children: &BTreeMap<ObjectPathBuf, Vec<ChildLink>>,
        counts: Option<&OriginalOwnerFacts>,
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
            session_parents: BTreeMap::new(),
            predecessors: BTreeMap::new(),
            placements: PlacementCorrespondence::new(),
        };
        let inputs = GroupInputs {
            groups,
            children,
            counts,
        };
        let mut seen = BTreeSet::new();
        for (path, source) in PlacementCorrespondence::declarations(groups) {
            result.placements.declare(
                path.as_ref().clone(),
                source.map(ReconstructionSource::object),
            );
            if let Some(source) = source {
                result.insert(engine, phase, &inputs, &mut seen, path.into_owned(), source);
            }
        }
        result
    }

    /// Mutably borrows the physical declarations and placement records.
    pub(in crate::edit::commit) fn placements(&mut self) -> &mut PlacementCorrespondence {
        &mut self.placements
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

    /// Extracts the root source error or first predecessor-count error in path order.
    ///
    /// Consumes the observations and placement metadata. Converts a count contradiction to
    /// [`Error::InconsistentReferenceCount`] with its original address and count fields.
    pub(super) fn into_source_error(self) -> Option<Error> {
        let Self {
            source,
            session_parents: _,
            predecessors,
            placements: _,
        } = self;
        source.err().or_else(|| {
            predecessors.into_values().find_map(|predecessor| {
                predecessor.counts.err().map(|error| match error {
                    CountEvidenceError::Source(error) | CountEvidenceError::Topology(error) => {
                        Error::from(error)
                    }
                    CountEvidenceError::Contradiction(error) => Error::InconsistentReferenceCount {
                        address: error.object.header_address().get(),
                        observed: error.observed,
                        minimum: error.minimum,
                    },
                })
            })
        })
    }

    fn insert(
        &mut self,
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        inputs: &GroupInputs<'_>,
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
            self.classify(engine, phase, inputs, &path, source, &counts)
        };
        if self.predecessors.contains_key(&path) {
            self.placements.repeated(path.clone());
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
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        inputs: &GroupInputs<'_>,
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
                Ok(Some(counts))
                    if inputs.counts.is_some_and(|facts| {
                        facts.counts().permits_root_replacement(
                            engine,
                            source,
                            counts.original,
                            counts.incoming,
                        )
                    }) =>
                {
                    Disposition::Supersede
                }
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
        let GroupInputs {
            groups,
            children,
            counts: _,
        } = inputs;
        let is_group = groups.contains_key(path);
        if is_group
            && !children.get(&parent).is_some_and(|children| {
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

    /// Checks physical correspondence, original source availability, and predecessor permission.
    ///
    /// Returns the replacement map only after all three checks succeed, in that order.
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
            session_parents: _,
            predecessors,
            placements,
        } = self;
        let replacements = placements
            .check(
                tree,
                predecessors
                    .iter()
                    .map(|(path, old)| (path, old.source.object())),
            )
            .map_err(TransitionWithheld::Placement)?;
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

/// Borrowed group declarations, child-link edits, and count facts for predecessor classification.
struct GroupInputs<'a> {
    groups: &'a BTreeMap<ObjectPathBuf, Node>,
    children: &'a BTreeMap<ObjectPathBuf, Vec<ChildLink>>,
    counts: Option<&'a OriginalOwnerFacts>,
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

    /// Extracts an original source or count error, consuming the applied observations.
    ///
    /// Delegates to [`ReconstructionDispositions::into_source_error`].
    pub(super) fn into_source_error(self) -> Option<Error> {
        self.original.into_source_error()
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
