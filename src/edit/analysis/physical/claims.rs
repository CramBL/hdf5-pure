//! Physical storage claims and their origins for one source phase.
//!
//! The collector reuses topology, local allocation, reference, and collection framing facts.
//! Source freshness, exclusive ownership, and permission to reuse storage require separate proofs.

use std::collections::BTreeSet;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

use super::{CapacityKind, FileStorageKind};
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use coverage::PhysicalCoverage;

pub(in crate::edit) mod coverage;
pub(in crate::edit) mod domain;

impl AnalysisPhase<'_> {
    /// Collects physical storage claims for this phase's hard-link root.
    ///
    /// On its first query, uses [`Self::topology`] and the per-object and collection caches.
    /// Reference interpretation can read payload bytes. Retains the batch for subsequent queries
    /// and records incomplete components in [`PhysicalClaims::problems`].
    ///
    /// Requires `source` from the same phase. The caller establishes source consistency and
    /// allocation ownership separately.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if `source` belongs to another
    /// phase or a fact-query invariant fails.
    pub(in crate::edit) fn physical_claims(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<&PhysicalClaims, Error> {
        if !self
            .identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, &source.identity))
        {
            return Err(FormatError::Internal(
                "physical claims belong to another source phase".into(),
            )
            .into());
        }
        if self.claims.is_none() {
            let claims = PhysicalClaims::collect(self, source)?;
            return Ok(self.claims.insert(claims));
        }
        self.claims
            .as_ref()
            .ok_or_else(|| FormatError::Internal("physical claims were not retained".into()).into())
    }
}

/// Physical range observations and independent support problems for one phase root.
///
/// The collector uses distinct topology identities for local allocation and reference queries,
/// and records each heap parent once by collection identity. Distinct file observations and
/// recorded capacities remain separate, including overlapping ranges. Complete source ownership
/// and reuse permission require separate evidence.
pub(in crate::edit) struct PhysicalClaims {
    /// The reconstruction source retained for phase comparisons.
    source: ReconstructionSource,
    /// The resolved phase root, independent of topology availability.
    root: Option<ObjectId>,
    /// The source image length recorded by the phase's file-storage query.
    physical_size: u64,
    /// The end address recorded in the session's superblock.
    logical_eoa: u64,
    claims: Vec<PhysicalClaim>,
    problems: Vec<ClaimProblem>,
    /// Coverage derived lazily from the original claims and recorded bounds.
    coverage: Option<PhysicalCoverage>,
}

impl PhysicalClaims {
    /// Collects file, capacity, object, and referenced heap claims through the phase caches.
    ///
    /// Retains original ranges and origins. Uses a temporary object-identity list and a distinct
    /// collection-address set. Member and reference summaries remain in their existing caches.
    ///
    /// # Errors
    ///
    /// Returns the invariant errors described by [`AnalysisPhase::physical_claims`].
    fn collect(
        phase: &mut AnalysisPhase<'_>,
        source: &ReconstructionSource,
    ) -> Result<Self, Error> {
        let engine = phase.engine;
        let root = phase.root.or_else(|| {
            ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).ok()
        });
        let physical = phase.physical_storage(source)?;
        let mut claims = Self {
            source: ReconstructionSource {
                identity: Arc::clone(&source.identity),
                object: source.object(),
            },
            root,
            physical_size: physical.physical_size(),
            logical_eoa: physical.logical_eoa(),
            claims: Vec::new(),
            problems: Vec::new(),
            coverage: None,
        };
        let physical = phase.physical_storage(source)?;
        claims
            .claims
            .extend(
                physical
                    .file()
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| PhysicalClaim {
                        range: entry.range,
                        origin: ClaimOrigin::File {
                            index,
                            kind: entry.kind,
                        },
                    }),
            );
        claims
            .claims
            .extend(
                physical
                    .metadata_capacity()
                    .iter()
                    .enumerate()
                    .map(|(index, &range)| PhysicalClaim {
                        range,
                        origin: ClaimOrigin::MetadataCapacity(index),
                    }),
            );
        claims.claims.extend(
            physical
                .non_live()
                .iter()
                .enumerate()
                .map(|(index, entry)| PhysicalClaim {
                    range: entry.range(),
                    origin: ClaimOrigin::NonLive {
                        index,
                        kind: entry.kind(),
                    },
                }),
        );
        claims
            .problems
            .extend((0..physical.problems().len()).map(ClaimProblem::File));
        let objects = match phase.topology() {
            Some(topology) => topology.indices.keys().copied().collect::<Vec<_>>(),
            None => {
                claims.problems.push(ClaimProblem::Topology);
                Vec::new()
            }
        };
        let mut collections = BTreeSet::new();
        for object in objects {
            let origin = phase.reconstruction_source(engine, object)?;
            let ownership = phase.object_ownership(&origin)?;
            claims.claims.extend(
                ownership
                    .allocations()
                    .iter()
                    .map(|allocation| PhysicalClaim {
                        range: allocation.range(),
                        origin: ClaimOrigin::Object {
                            object,
                            kind: allocation.kind(),
                        },
                    }),
            );
            if !ownership.is_complete() {
                claims.problems.push(ClaimProblem::ObjectStorage(object));
            }
            let references = phase.heap_reference_facts(&origin)?;
            collections.extend(
                references
                    .collections()
                    .iter()
                    .copied()
                    .map(PhysicalAddress::new),
            );
            if !references.is_complete() {
                claims.problems.push(ClaimProblem::References(object));
            }
        }
        for address in collections {
            match phase.heap_members(source, address) {
                Ok(contents) => claims.claims.push(PhysicalClaim {
                    range: contents.range(),
                    origin: ClaimOrigin::Heap(address),
                }),
                Err(error) => claims
                    .problems
                    .push(ClaimProblem::Collection { address, error }),
            }
        }
        Ok(claims)
    }

    /// Returns the resolved phase root, independently of topology availability.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.root
    }

    /// Returns the source image length recorded with the file-storage facts.
    pub(in crate::edit) fn physical_size(&self) -> u64 {
        self.physical_size
    }

    /// Returns the end address recorded in the session's superblock.
    pub(in crate::edit) fn logical_eoa(&self) -> u64 {
        self.logical_eoa
    }

    /// Returns whether `source` has the same phase association.
    ///
    /// The supplied object identity can differ from the phase root. Source freshness requires
    /// separate evidence.
    pub(in crate::edit) fn matches_phase(&self, source: &ReconstructionSource) -> bool {
        Arc::ptr_eq(&self.source.identity, &source.identity)
    }

    /// Borrows recorded ranges with their origins, including any overlaps.
    pub(in crate::edit) fn claims(&self) -> &[PhysicalClaim] {
        &self.claims
    }

    /// Borrows incomplete-component markers and collection-framing failures.
    pub(in crate::edit) fn problems(&self) -> &[ClaimProblem] {
        &self.problems
    }
}

/// A physical range paired with the observation that produced it.
///
/// Allocation ownership and permission to reuse storage require separate evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) struct PhysicalClaim {
    range: FileRange,
    origin: ClaimOrigin,
}

impl PhysicalClaim {
    /// Returns the recorded range in physical coordinates.
    pub(in crate::edit) fn range(self) -> FileRange {
        self.range
    }

    /// Returns the recorded observation or object associated with the range.
    pub(in crate::edit) fn origin(self) -> ClaimOrigin {
        self.origin
    }
}

/// The observation or object associated with a physical claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum ClaimOrigin {
    /// An individual observation from [`super::PhysicalStorage::file`].
    File {
        /// The observation's index, including any repeated ranges.
        index: usize,
        kind: FileStorageKind,
    },
    /// A complete global heap parent, identified by its physical collection address.
    Heap(PhysicalAddress),
    /// Capacity indexed in [`super::PhysicalStorage::metadata_capacity`].
    ///
    /// The recorded capacity can include separately observed extension or manager storage.
    MetadataCapacity(usize),
    /// A range indexed in [`super::PhysicalStorage::non_live`].
    NonLive { index: usize, kind: CapacityKind },
    /// An allocation reported for a hard-link-reachable object.
    Object {
        object: ObjectId,
        kind: AllocationKind,
    },
}

/// A failed or incomplete component of physical claim collection.
///
/// File problems contain indices into [`super::PhysicalStorage::problems`]. Object problems
/// contain identities for the local allocation and reference caches. Collection-framing failures
/// are retained with their physical addresses.
#[derive(Debug)]
pub(in crate::edit) enum ClaimProblem {
    /// A failure while collecting a referenced collection's complete framing.
    Collection {
        address: PhysicalAddress,
        error: FormatError,
    },
    /// The index of a problem in [`super::PhysicalStorage::problems`].
    File(usize),
    /// Incomplete local allocation discovery for this object.
    ObjectStorage(ObjectId),
    /// Incomplete reference interpretation for this object's content.
    References(ObjectId),
    /// Unavailable hard-link topology for the phase root.
    Topology,
}

#[cfg(test)]
mod tests;
