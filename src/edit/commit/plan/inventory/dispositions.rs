//! Original predecessor classifications and successful placement correspondence.
//!
//! The editor collects rebuilt groups and relocated datasets independently of count changes.
//! Complete correspondence requires checked supersession or retention of every predecessor and
//! agreement with the applied tree's placements. The caller establishes source freshness, source-domain
//! applicability, storage protection, and publication permission separately.

use std::collections::{BTreeMap, BTreeSet};

use hdf5_pure_core::PhysicalAddress;

use super::super::{ChildLink, PreparedTree};
use super::session::SessionParent;
use crate::edit::WriteEngine;
use crate::edit::analysis::counts::{CountEvidence, CountEvidenceError, ROOT_REFERENCE_COUNT};
use crate::edit::analysis::{AnalysisPhase, OriginalOwnerFacts, ReconstructionSource};
use crate::error::Error;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

/// Original observations and expected placements for one prepared tree.
///
/// The editor records newly created groups with an absent predecessor and keeps count
/// failures available for storage rejection.
pub(in crate::edit::commit) struct ReconstructionDispositions {
    source: Result<ReconstructionSource, Error>,
    pub(super) session_parents: BTreeMap<hdf5_pure_core::PhysicalAddress, SessionParent>,
    predecessors: BTreeMap<ObjectPathBuf, Predecessor>,
    expected: BTreeMap<ObjectPathBuf, Option<ObjectId>>,
    placed: BTreeMap<ObjectPathBuf, ObjectId>,
    placement_mismatch: bool,
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
        facts: Option<&OriginalOwnerFacts>,
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
            expected: BTreeMap::new(),
            placed: BTreeMap::new(),
            placement_mismatch: false,
        };
        let mut seen = BTreeSet::new();
        for (path, node) in &tree.groups {
            result.expected.insert(
                path.clone(),
                node.prepared.replaces.as_ref().map(|old| old.object()),
            );
            if let Some(old) = &node.prepared.replaces {
                result.insert(
                    engine,
                    phase,
                    tree,
                    facts,
                    &mut seen,
                    path.clone(),
                    old.source(),
                );
            }
            for write in &node.prepared.writes {
                let path = path.join(&write.link);
                result
                    .expected
                    .insert(path.clone(), Some(write.source.object()));
                result.insert(engine, phase, tree, facts, &mut seen, path, &write.source);
            }
        }
        result
    }

    /// Checks and records one placement. A mismatch prevents reconciliation.
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
            session_parents: _,
            predecessors: _,
            expected,
            placed: placements,
            placement_mismatch,
        } = self;
        if expected.get(path) != Some(&source) || placements.contains_key(path) {
            *placement_mismatch = true;
        } else {
            placements.insert(path.clone(), placed);
        }
    }

    /// Pairs the original observations with their placement reconciliation outcome.
    pub(super) fn finish(
        self,
        root: PhysicalAddress,
        relocations: BTreeMap<ObjectId, ObjectId>,
    ) -> AppliedDispositions {
        let replacements = self
            .matches_placements(root, &relocations)
            .then_some(relocations);
        let retained = self
            .predecessors
            .into_values()
            .filter(|old| old.disposition == Disposition::Retain)
            .map(|old| old.source.object())
            .collect();
        AppliedDispositions {
            source: self.source.ok(),
            session_parents: self.session_parents,
            retained,
            replacements,
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
            expected: _,
            placed: _,
            placement_mismatch: _,
        } = self;
        source.err().or_else(|| {
            predecessors.into_values().find_map(|predecessor| {
                predecessor.count_error.map(|error| match error {
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
        tree: &PreparedTree,
        facts: Option<&OriginalOwnerFacts>,
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
        let disposition = if !seen.insert(source.object())
            || phase
                .reconstruction_source(engine, source.object())
                .as_ref()
                != Ok(source)
        {
            Disposition::Withhold
        } else {
            self.classify(engine, phase, tree, facts, &path, source, &counts)
        };
        if self.predecessors.contains_key(&path) {
            self.placement_mismatch = true;
        }
        self.predecessors.insert(
            path,
            Predecessor {
                source: source.clone(),
                count_error: counts.err(),
                disposition,
            },
        );
    }

    fn classify(
        &self,
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        tree: &PreparedTree,
        facts: Option<&OriginalOwnerFacts>,
        path: &ObjectPathBuf,
        source: &ReconstructionSource,
        counts: &Result<Option<ObservedCounts>, CountEvidenceError>,
    ) -> Disposition {
        let Some(topology) = phase.topology() else {
            return Disposition::Withhold;
        };
        let Ok(root) = &self.source else {
            return Disposition::Withhold;
        };
        let Some((parent, leaf)) = path.split_leaf() else {
            if source != root {
                return Disposition::Withhold;
            }
            return match counts {
                Ok(Some(counts))
                    if facts.is_some_and(|facts| {
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
                Ok(Some(counts))
                    if counts.incoming == 1
                        && counts.original == 2
                        && tree.groups.len() == 1
                        && tree.groups.get(path).is_some_and(|node| {
                            node.prepared.deletes.is_empty()
                                && node.prepared.copies.is_empty()
                                && node.prepared.writes.is_empty()
                                && node.existing_links.iter().any(|name| {
                                    crate::object_path::LinkNameBuf::new(name).is_some_and(|name| {
                                        topology
                                            .find_link(&path.join(&name))
                                            .is_some_and(|(_, target)| target == source.object())
                                    })
                                })
                        }) =>
                {
                    Disposition::Retain
                }
                Ok(Some(counts))
                    if counts.incoming == 0 && counts.original == ROOT_REFERENCE_COUNT =>
                {
                    Disposition::Supersede
                }
                _ => Disposition::Withhold,
            };
        };
        if topology.find_link(path).map(|(_, object)| object) != Some(source.object()) {
            return Disposition::Withhold;
        }
        let Some(owner) = self.predecessors.get(&parent) else {
            return Disposition::Withhold;
        };
        let parent_object = if parent.is_empty() {
            Some(root.object())
        } else {
            topology.find_link(&parent).map(|(_, object)| object)
        };
        if parent_object != Some(owner.source.object()) {
            return Disposition::Withhold;
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
            return Disposition::Withhold;
        }
        match counts {
            Ok(Some(counts)) if counts.incoming == 1 && counts.ordinary && counts.original == 1 => {
                owner.disposition
            }
            _ => Disposition::Withhold,
        }
    }

    /// Checks that the application map matches every recorded predecessor placement.
    ///
    /// Repeated predecessor identities are withheld during collection. Matching every
    /// predecessor and the map length excludes additional relocations. Missing placements,
    /// an unavailable original root, and withheld predecessors prevent reconciliation.
    fn matches_placements(
        &self,
        root: PhysicalAddress,
        relocations: &BTreeMap<ObjectId, ObjectId>,
    ) -> bool {
        let Self {
            source,
            session_parents: _,
            predecessors,
            expected,
            placed,
            placement_mismatch,
        } = self;
        !*placement_mismatch
            && expected.keys().all(|path| placed.contains_key(path))
            && placed
                .get(&ObjectPathBuf::root())
                .map(|root| root.header_address())
                == Some(root)
            && source.is_ok()
            && predecessors.len() == relocations.len()
            && predecessors.iter().all(|(path, old)| {
                old.disposition != Disposition::Withhold
                    && placed
                        .get(path)
                        .is_some_and(|new| relocations.get(&old.source.object()) == Some(new))
            })
    }
}

/// Original associations and the checked replacement map after placement.
pub(in crate::edit::commit) struct AppliedDispositions {
    source: Option<ReconstructionSource>,
    pub(super) session_parents: BTreeMap<hdf5_pure_core::PhysicalAddress, SessionParent>,
    retained: BTreeSet<ObjectId>,
    replacements: Option<BTreeMap<ObjectId, ObjectId>>,
}

impl AppliedDispositions {
    pub(super) fn source(&self) -> Option<&ReconstructionSource> {
        self.source.as_ref()
    }

    pub(super) fn retained(&self) -> &BTreeSet<ObjectId> {
        &self.retained
    }

    /// Borrows the complete original-to-placed map after successful reconciliation.
    pub(super) fn replacements(&self) -> Option<&BTreeMap<ObjectId, ObjectId>> {
        self.replacements.as_ref()
    }

    /// Moves the checked application map into publication.
    pub(super) fn into_replacements(self) -> Option<BTreeMap<ObjectId, ObjectId>> {
        self.replacements
    }

    /// Translates superseded identities and preserves retained originals and absent identities.
    ///
    /// Returns `None` if reconciliation is incomplete, including for an unchanged identity.
    pub(super) fn translate(&self, original: ObjectId) -> Option<ObjectId> {
        self.replacements().map(|replacements| {
            if self.retained.contains(&original) {
                original
            } else {
                replacements.get(&original).copied().unwrap_or(original)
            }
        })
    }
}

/// One reconstructed object's original association and classification.
pub(in crate::edit::commit) struct Predecessor {
    pub(in crate::edit::commit) source: ReconstructionSource,
    count_error: Option<CountEvidenceError>,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum Disposition {
    /// A predecessor with compatible counts and a superseded path from the original root.
    Supersede,
    /// An original root or descendant preserved alongside its placed replacement.
    Retain,
    /// A predecessor excluded from complete translation.
    Withhold,
}
