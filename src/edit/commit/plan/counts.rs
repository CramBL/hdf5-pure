//! Local count projections indexed by the physical identities of prepared predecessors.

use core::iter;
use std::collections::BTreeMap;

use crate::edit::analysis::CountPreflight;
use crate::edit::analysis::CountProjectionIdentity;
use crate::edit::object_header::reference_count::HeaderCount;
use crate::edit::object_header::reference_count::HeaderCountError;
use crate::object_graph::ObjectId;

use super::PreparedGroup;
use super::RelocatedDataset;

pub(super) use preparation::CountedPreparation;
pub(in crate::edit::commit) use preparation::PreparedTreeCounts;
pub(super) use preparation::TreeCountInputs;

mod preparation;

/// Checked local count updates associated with one owned preflight.
///
/// A completion proof is required to release that preflight's protected source batch.
/// Source freshness, admission, and publication permissions remain separate.
pub(super) struct PreparedCounts<'region> {
    updates: Vec<(HeaderCount<'region>, u32)>,
    identity: CountProjectionIdentity,
}

impl<'region> PreparedCounts<'region> {
    /// Prepares checked local views from the preflight's original count facts.
    ///
    /// The caller applies the returned updates through [`Self::apply`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::project_owned`].
    pub(super) fn owned(
        plan: &CountPreflight,
        groups: impl IntoIterator<Item = &'region mut PreparedGroup>,
    ) -> Result<Self, CountProjectionError> {
        Self::project_owned(plan, groups)
    }

    /// Applies checked local updates and returns their completion proof.
    ///
    /// Inserting a missing local count may allocate. Produces a proof only after all selected
    /// updates succeed.
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

    /// Prepares local views and classifies the preflight's count entries.
    ///
    /// Skips `groups` for empty or entirely unchanged plans. Otherwise indexes every supplied
    /// predecessor before local queries, preserving group order and each group's write order.
    /// Preparation preserves the local record bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CountProjectionError::Phase`] or [`CountProjectionError::Repeated`] at the first
    /// incompatible indexed predecessor, checking phase before repetition per predecessor.
    /// Defers [`HeaderCountError::Ambiguous`], [`HeaderCountError::Unsupported`], and reconstruction
    /// mismatches through [`ProjectionErrors`]. Returns other [`CountProjectionError::Region`]
    /// errors and [`CountProjectionError::Original`] immediately, in changed-entry order.
    fn project_owned(
        plan: &CountPreflight,
        groups: impl IntoIterator<Item = &'region mut PreparedGroup>,
    ) -> Result<Self, CountProjectionError> {
        let mut updates = Vec::new();
        let mut changes = plan.changes().iter();
        let Some(first) = changes.find(|change| {
            if change.original().value() == change.remaining() {
                return false;
            }
            true
        }) else {
            return Ok(Self {
                updates,
                identity: plan.projection_identity(),
            });
        };
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
        let mut errors = ProjectionErrors::default();
        for change in iter::once(first).chain(changes) {
            let original = change.original().value();
            if original == change.remaining() {
                continue;
            }
            let object = change.object();
            let Some(region) = regions.remove(&object) else {
                errors.reconstruction(object, plan.reconstructed(object), false);

                continue;
            };
            let projection = match region.reference_count_mut() {
                Ok(projection) => projection,
                Err(error @ (HeaderCountError::Ambiguous | HeaderCountError::Unsupported(_))) => {
                    errors.opaque = Some(CountProjectionError::Region { object, error });
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
            errors.reconstruction(object, plan.reconstructed(object), true);
            updates.push((projection, change.remaining()));
        }
        errors.finish()?;
        Ok(Self {
            updates,
            identity: plan.projection_identity(),
        })
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

/// Deferred count errors retained until immediate preparation checks finish.
///
/// The selection order is [`Self::opaque`], [`Self::rebuilt`], then [`Self::source`].
#[derive(Default)]
struct ProjectionErrors {
    /// The last ambiguous or unsupported local-count error.
    opaque: Option<CountProjectionError>,
    /// The first reconstruction mismatch for a matched local region.
    rebuilt: Option<CountProjectionError>,
    /// The first reconstruction mismatch without a matching local region.
    source: Option<CountProjectionError>,
}

impl ProjectionErrors {
    /// Accumulates a changed target's reconstruction mismatch.
    fn reconstruction(&mut self, object: ObjectId, expected: bool, actual: bool) {
        if expected == actual {
            return;
        }
        let error = CountProjectionError::Reconstruction {
            object,
            expected,
            actual,
        };
        if actual {
            self.rebuilt.get_or_insert(error);
        } else {
            self.source.get_or_insert(error);
        }
    }

    /// Returns the highest-priority deferred failure, if present.
    ///
    /// # Errors
    ///
    /// Returns a retained [`CountProjectionError`].
    fn finish(self) -> Result<(), CountProjectionError> {
        let Self {
            opaque,
            rebuilt,
            source,
        } = self;
        match opaque.or(rebuilt).or(source) {
            Some(error) => Err(error),
            None => Ok(()),
        }
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

#[cfg(test)]
mod tests;
