//! Local count projections indexed by the physical identities of prepared predecessors.

use std::collections::BTreeMap;

use crate::edit::analysis::CountPreflight;
use crate::edit::analysis::CountProjectionIdentity;
use crate::edit::object_header::reference_count::HeaderCount;
use crate::edit::object_header::reference_count::HeaderCountError;
use crate::object_graph::ObjectId;

use super::PreparedGroup;
use super::RelocatedDataset;

pub(in crate::edit::commit) use preparation::PreparedTreeCounts;
pub(super) use preparation::{CountedPreparation, TreeCountInputs};

mod preparation;

/// Checked local count updates associated with one owned preflight.
///
/// A completion proof is required to release that preflight's protected source batch.
/// Source freshness, admission, and publication permissions remain separate.
pub(super) struct OwnedCountProjection<'region> {
    updates: Vec<(HeaderCount<'region>, u32)>,
    identity: CountProjectionIdentity,
}

impl<'region> OwnedCountProjection<'region> {
    /// Prepares updates to owned count records under the preflight reconstruction decisions.
    ///
    /// Matches each changed target to a unique predecessor from the original source phase.
    /// Leaves the records unchanged and skips predecessor iteration when no counts change.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched or repeated predecessors, incompatible count records, or
    /// a changed target whose reconstruction differs from preflight.
    pub(super) fn prepare(
        plan: &CountPreflight,
        groups: impl IntoIterator<Item = &'region mut PreparedGroup>,
    ) -> Result<Self, CountProjectionError> {
        let identity = plan.projection_identity();
        let mut changes = plan
            .changes()
            .iter()
            .filter_map(|change| {
                let original = change.original().value();
                let remaining = change.remaining();
                (original != remaining).then_some((change.object(), original, remaining))
            })
            .peekable();
        let mut updates = Vec::new();
        if changes.peek().is_none() {
            return Ok(Self { updates, identity });
        }
        let mut regions = BTreeMap::new();
        for group in groups {
            let PreparedGroup {
                region,
                dense_links: _,
                replaces,
                deletes: _,
                copies: _,
                writes,
                attrs: _,
                additions: _,
            } = group;
            let predecessors = replaces
                .as_ref()
                .map(|replaces| (replaces.source(), region))
                .into_iter()
                .chain(writes.iter_mut().map(|write| {
                    let RelocatedDataset {
                        link: _,
                        source,
                        write,
                    } = write;
                    (&*source, write.region_mut())
                }));
            for (source, region) in predecessors {
                let object = source.object();
                if !plan.shares_phase(source) {
                    return Err(CountProjectionError::Phase { object });
                }
                if regions.insert(object, region).is_some() {
                    return Err(CountProjectionError::Repeated { object });
                }
            }
        }
        let mut opaque = None;
        let mut replacement_mismatch = None;
        let mut source_mismatch = None;
        for (object, original, remaining) in changes {
            let Some(region) = regions.remove(&object) else {
                if plan.reconstructed(object) {
                    source_mismatch.get_or_insert(CountProjectionError::Reconstruction {
                        object,
                        expected: true,
                        actual: false,
                    });
                }
                continue;
            };
            let projection = match region.reference_count_mut() {
                Ok(projection) => projection,
                Err(error @ (HeaderCountError::Ambiguous | HeaderCountError::Unsupported(_))) => {
                    opaque = Some(CountProjectionError::Region { object, error });
                    continue;
                }
                Err(error) => return Err(CountProjectionError::Region { object, error }),
            };
            if projection.count() != original {
                return Err(CountProjectionError::Original {
                    object,
                    expected: original,
                    actual: projection.count(),
                });
            }
            if !plan.reconstructed(object) {
                replacement_mismatch.get_or_insert(CountProjectionError::Reconstruction {
                    object,
                    expected: false,
                    actual: true,
                });
            }
            updates.push((projection, remaining));
        }
        // Framing and original-count errors take precedence over reconstruction mismatches.
        if let Some(error) = opaque.or(replacement_mismatch).or(source_mismatch) {
            return Err(error);
        }
        Ok(Self { updates, identity })
    }

    /// Applies checked local updates and returns a proof of completion.
    ///
    /// Inserting a missing local count may allocate.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`HeaderCount::set`]. Earlier local updates remain applied on error.
    pub(super) fn apply(self) -> Result<CountProjectionProof, HeaderCountError> {
        let Self { updates, identity } = self;
        for (projection, remaining) in updates {
            projection.set(remaining)?;
        }
        Ok(CountProjectionProof { identity })
    }
}

/// Successful local count application associated with its owned preflight.
///
/// For changed work, the caller must return this proof to its originating preflight. Empty or
/// entirely unchanged batches have no protected updates or identity allocation. Source freshness
/// and publication require separate proof.
pub(in crate::edit) struct CountProjectionProof {
    identity: CountProjectionIdentity,
}

impl CountProjectionProof {
    /// Returns whether `identity` has the same changed preflight association.
    ///
    /// Two absent associations compare equal.
    pub(in crate::edit) fn matches(&self, identity: &CountProjectionIdentity) -> bool {
        self.identity.same(identity)
    }
}

/// An inconsistent association or count, or invalid framing in a prepared region.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum CountProjectionError {
    /// A supported local count different from the original source-bound count.
    Original {
        object: ObjectId,
        /// The original count from the source-bound evidence.
        expected: u32,
        /// The count parsed from the owned replacement records.
        actual: u32,
    },
    /// A predecessor associated with a different analysis phase.
    Phase {
        /// The physical identity of the prepared predecessor.
        object: ObjectId,
    },
    /// A local completion proof from a different changed preflight.
    Preflight,
    /// A changed target's actual reconstruction disposition differs from preflight.
    Reconstruction {
        /// The physical identity of the changed target.
        object: ObjectId,
        /// Whether preflight excluded the target as reconstructed.
        expected: bool,
        /// Whether the target has a matched owned replacement region.
        actual: bool,
    },
    /// Invalid local framing for a matched source object.
    Region {
        object: ObjectId,
        error: HeaderCountError,
    },
    /// Multiple prepared regions for the same source object.
    Repeated {
        /// The physical identity supplied by multiple prepared predecessors.
        object: ObjectId,
    },
}
