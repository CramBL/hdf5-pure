//! Local count projections indexed by the physical identities of prepared predecessors.

use std::collections::BTreeMap;

use crate::edit::analysis::DeclaredCountChange;
use crate::edit::analysis::NetCountPlan;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::SourceRemovalPlan;
use crate::edit::analysis::counts::changes::{CountChange, OriginalCount};
use crate::edit::object_header::reference_count::HeaderCount;
use crate::edit::object_header::reference_count::HeaderCountError;
use crate::object_graph::ObjectId;

use super::PreparedGroup;
use super::RelocatedDataset;

/// Exclusive checked count views with planned values and retained change dispositions.
///
/// Construction leaves owned records unchanged. Application updates or inserts selected local
/// count records. Each original change value is retained in its local projection disposition.
/// Source-write and retirement permissions are established separately.
pub(super) struct PreparedCounts<'region, Change> {
    updates: Vec<(HeaderCount<'region>, u32)>,
    disposition: CountDisposition<Change>,
}

impl<'region, 'plan, 'source> PreparedCounts<'region, DeclaredCountChange<'plan, 'source>> {
    /// Matches declared count changes to checked owned replacement records in one batch.
    ///
    /// The caller supplies predecessor associations and their record regions after structural
    /// edits. Indexes each prepared predecessor once, checks phase identity and uniqueness, then
    /// looks up each known change. Skips group iteration when `plan` is absent or has no known
    /// count changes. Retains ambiguous or unsupported replacement counts separately.
    ///
    /// # Errors
    ///
    /// Returns [`CountProjectionError::Phase`] for a predecessor from another phase,
    /// [`CountProjectionError::Repeated`] for repeated predecessor identities,
    /// [`CountProjectionError::Region`] for invalid framing in a matching region, or
    /// [`CountProjectionError::Original`] if a supported local count differs from its original
    /// source-bound count. Every owned record region remains unchanged on failure.
    pub(super) fn new(
        plan: Option<&'plan SourceRemovalPlan<'_, 'source>>,
        groups: impl IntoIterator<Item = &'region mut PreparedGroup>,
    ) -> Result<Self, CountProjectionError> {
        let Some(plan) = plan else {
            return Ok(Self::empty());
        };
        Self::project(
            plan.changes().map(|change| {
                let bound = change.bound();
                (
                    CountValues {
                        object: bound.object(),
                        original: bound.original(),
                        remaining: change.remaining(),
                    },
                    change,
                )
            }),
            groups,
            |source| plan.shares_phase(source),
        )
    }
}

impl<'region, 'plan, 'source> PreparedCounts<'region, CountChange<'plan, 'source>> {
    /// Prepares local net-count updates while retaining each target's original typed evidence.
    ///
    /// Retains unchanged counts separately without querying their local records. Skips group
    /// iteration for empty or entirely unchanged work. Empty work requires no heap allocation.
    /// Retaining nonempty unchanged evidence may allocate.
    ///
    /// Matches changed targets through the shared predecessor index and checks their original
    /// counts. Construction leaves every owned record unchanged. Source writes, retirement,
    /// source freshness, and publication require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`CountProjectionError::Phase`] for a predecessor from another phase,
    /// [`CountProjectionError::Repeated`] for repeated predecessor identities,
    /// [`CountProjectionError::Region`] for invalid matching framing, or
    /// [`CountProjectionError::Original`] if a supported local count differs from its original
    /// source-bound count.
    pub(super) fn combined(
        plan: &'plan NetCountPlan<'_, 'source>,
        groups: impl IntoIterator<Item = &'region mut PreparedGroup>,
    ) -> Result<Self, CountProjectionError> {
        let mut unchanged = Vec::new();
        let changes = plan.evidence_changes().filter_map(|change| {
            let (object, original) = match change.original() {
                OriginalCount::Referenced(original) => (original.object(), original.original()),
                OriginalCount::Topology(original) => {
                    let bound = original.bound();
                    (bound.object(), bound.original())
                }
            };
            let remaining = change.remaining();
            if original == remaining {
                unchanged.push(change);
                return None;
            }
            Some((
                CountValues {
                    object,
                    original,
                    remaining,
                },
                change,
            ))
        });
        let mut prepared = Self::project(changes, groups, |source| plan.shares_phase(source))?;
        prepared.disposition.unchanged = unchanged;
        Ok(prepared)
    }
}

impl<'region, Change> PreparedCounts<'region, Change> {
    /// Creates an empty projection with no heap allocation.
    fn empty() -> Self {
        Self {
            updates: Vec::new(),
            disposition: CountDisposition {
                rebuilt: Vec::new(),
                source: Vec::new(),
                opaque: Vec::new(),
                unchanged: Vec::new(),
            },
        }
    }

    /// Prepares checked local updates and retains the supplied change values.
    ///
    /// Each entry pairs a unique change with its matching physical identity, original count, and
    /// planned count. The caller supplies `shares_phase` for the original change phase. Indexes
    /// prepared predecessors once, checks phase identity and uniqueness, and consumes each matched
    /// region at most once. Skips group iteration for an empty change iterator.
    ///
    /// Retains ambiguous or unsupported replacement counts as opaque dispositions. Construction
    /// leaves every owned record unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CountProjectionError::Phase`] or [`CountProjectionError::Repeated`] for invalid
    /// predecessor associations, [`CountProjectionError::Region`] for invalid matching framing, or
    /// [`CountProjectionError::Original`] if a matching local count differs from the original value.
    fn project(
        changes: impl IntoIterator<Item = (CountValues, Change)>,
        groups: impl IntoIterator<Item = &'region mut PreparedGroup>,
        shares_phase: impl Fn(&ReconstructionSource) -> bool,
    ) -> Result<Self, CountProjectionError> {
        let mut prepared = Self::empty();
        let mut changes = changes.into_iter().peekable();
        if changes.peek().is_none() {
            return Ok(prepared);
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
                if !shares_phase(source) {
                    return Err(CountProjectionError::Phase { object });
                }
                if regions.insert(object, region).is_some() {
                    return Err(CountProjectionError::Repeated { object });
                }
            }
        }
        for (values, change) in changes {
            let CountValues {
                object,
                original,
                remaining,
            } = values;
            let Some(region) = regions.remove(&object) else {
                prepared.disposition.source.push(change);
                continue;
            };
            let projection = match region.reference_count_mut() {
                Ok(projection) => projection,
                Err(error @ (HeaderCountError::Ambiguous | HeaderCountError::Unsupported(_))) => {
                    prepared.disposition.opaque.push((change, error));
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
            prepared.updates.push((projection, remaining));
            prepared.disposition.rebuilt.push(change);
        }
        Ok(prepared)
    }

    /// Updates the selected owned count records and returns every change's disposition.
    ///
    /// Uses each retained exclusive view, preserving unrelated record bytes and original source
    /// evidence. Inserting a missing count record may allocate.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`HeaderCount::set`]. Earlier local updates remain applied on error.
    pub(super) fn apply(self) -> Result<CountDisposition<Change>, HeaderCountError> {
        let Self {
            updates,
            disposition,
        } = self;
        for (projection, remaining) in updates {
            projection.set(remaining)?;
        }
        Ok(disposition)
    }
}

/// Original change values partitioned by their local projection disposition.
///
/// Source writes and retirement require separate permissions for every disposition.
pub(super) struct CountDisposition<Change> {
    /// Changes matched to checked replacement regions with the same original count.
    pub(super) rebuilt: Vec<Change>,
    /// Changes without a prepared replacement region for their source object.
    pub(super) source: Vec<Change>,
    /// Changes with ambiguous or unsupported replacement counts and their exact reasons.
    pub(super) opaque: Vec<(Change, HeaderCountError)>,
    /// Targets whose planned count equals the original count, without a local record query.
    pub(super) unchanged: Vec<Change>,
}

/// A physical predecessor identity with its original and planned declared counts.
struct CountValues {
    object: ObjectId,
    original: u32,
    remaining: u32,
}

/// An inconsistent association or count, or invalid framing in a prepared region.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum CountProjectionError {
    /// A supported local count different from the original source-bound count.
    Original {
        object: ObjectId,
        /// The original count from the source-bound evidence.
        expected: u32,
        /// The count parsed from the owned replacement records.
        actual: u32,
    },
    /// A predecessor associated with a different analysis phase.
    Phase { object: ObjectId },
    /// Invalid local framing for a matched source object.
    Region {
        object: ObjectId,
        error: HeaderCountError,
    },
    /// Multiple prepared regions for the same source object.
    Repeated { object: ObjectId },
}

#[cfg(test)]
mod tests;
