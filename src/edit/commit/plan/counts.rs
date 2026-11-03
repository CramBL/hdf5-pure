//! Local count projections indexed by the physical identities of prepared predecessors.

use std::collections::BTreeMap;

use crate::edit::analysis::DeclaredCountChange;
use crate::edit::analysis::SourceRemovalPlan;
use crate::edit::object_header::reference_count::HeaderCount;
use crate::edit::object_header::reference_count::HeaderCountError;
use crate::object_graph::ObjectId;

use super::PreparedGroup;
use super::RelocatedDataset;

/// Exclusive checked count views with their planned remainders and replacement dispositions.
///
/// Construction leaves owned records unchanged. Application updates or inserts selected local
/// count records. Original source-bound evidence remains borrowed from the removal plan, with
/// source-write and retirement permissions established separately.
pub(super) struct PreparedCounts<'region, 'plan, 'source> {
    updates: Vec<(HeaderCount<'region>, u32)>,
    disposition: CountDisposition<'plan, 'source>,
}

impl<'region, 'plan, 'source> PreparedCounts<'region, 'plan, 'source> {
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
        let mut prepared = Self {
            updates: Vec::new(),
            disposition: CountDisposition {
                rebuilt: Vec::new(),
                source: Vec::new(),
                opaque: Vec::new(),
            },
        };
        let Some(plan) = plan else {
            return Ok(prepared);
        };
        let mut changes = plan.changes().peekable();
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
                if !plan.shares_phase(source) {
                    return Err(CountProjectionError::Phase { object });
                }
                if regions.insert(object, region).is_some() {
                    return Err(CountProjectionError::Repeated { object });
                }
            }
        }
        for change in changes {
            let object = change.bound().object();
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
            if projection.count() != change.bound().original() {
                return Err(CountProjectionError::Original {
                    object,
                    expected: change.bound().original(),
                    actual: projection.count(),
                });
            }
            prepared.updates.push((projection, change.remaining()));
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
    pub(super) fn apply(self) -> Result<CountDisposition<'plan, 'source>, HeaderCountError> {
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

/// Declared count changes partitioned by their replacement-record disposition.
///
/// Source writes and retirement require separate permissions for every disposition.
pub(super) struct CountDisposition<'plan, 'source> {
    /// Changes matched to checked replacement regions with the same original count.
    pub(super) rebuilt: Vec<DeclaredCountChange<'plan, 'source>>,
    /// Changes without a prepared replacement region for their source object.
    pub(super) source: Vec<DeclaredCountChange<'plan, 'source>>,
    /// Changes with ambiguous or unsupported replacement counts and their exact reasons.
    pub(super) opaque: Vec<(DeclaredCountChange<'plan, 'source>, HeaderCountError)>,
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
