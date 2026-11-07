//! Complete owned count inventories, source preflight, and positive explicit subsets.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::CountProtectionError;
use super::ProtectedCounts;
use super::{CountHandoffError, CountSource, SurvivingCount, SurvivingCounts};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::counts::changes::owned::{
    CountDisposition, OwnedCountChange, UnresolvedCount,
};
use crate::edit::analysis::{NetCountPlan, ReconstructionSource};
use crate::edit::commit::CountProjectionError;
use crate::edit::commit::CountProjectionProof;
use crate::object_graph::ObjectId;

impl NetCountPlan<'_, '_> {
    /// Copies every represented count and original fact into a session-bound inventory.
    ///
    /// Checks engine identity even for an empty plan. Preserves the original physical root and
    /// phase association for nonempty work. An empty inventory retains the session association
    /// without a phase origin or root. Uses existing plan metadata without source reads, topology
    /// discovery, or count recalculation. The borrowed plan and phase may then be dropped.
    ///
    /// # Errors
    ///
    /// Returns [`CountHandoffError::Engine`] if `engine` differs from the plan's borrowed engine.
    pub(in crate::edit) fn owned_counts(
        &self,
        engine: &WriteEngine,
    ) -> Result<CountInventory, CountHandoffError> {
        if !self.same_engine(engine) {
            return Err(CountHandoffError::Engine);
        }
        let origin = self.source().map(|source| ReconstructionSource {
            identity: Arc::clone(&source.identity),
            object: source.object(),
        });
        Ok(CountInventory {
            session: engine.context.allocations.session(),
            origin,
            changes: self
                .evidence_changes()
                .map(|change| change.detach())
                .collect(),
        })
    }
}

/// Every supplied count projection and its original facts, bound to one allocation session.
///
/// Includes unchanged, zero, and implicit counts. Nonempty inventories retain the original phase
/// identity and physical root. Source freshness, original-user completeness, allocation ownership,
/// retirement, and count-write or publication permissions require separate evidence.
pub(in crate::edit) struct CountInventory {
    session: AllocationSession,
    origin: Option<ReconstructionSource>,
    changes: Vec<OwnedCountChange>,
}

impl CountInventory {
    /// Checks owned counts and source allocations against the supplied reconstruction decisions.
    ///
    /// Checks the allocation session even for empty input. For a nonempty inventory, checks the
    /// original root and publication availability. Checks reconstruction phases and uniqueness
    /// only when a count changes. Preserves every original entry and protects changed positive
    /// explicit source fields outside the reconstruction set against recorded allocator claims.
    /// A changed positive implicit count requires a matching reconstruction.
    /// Accepts [`CountDisposition::Retain`] entries without preparing zero-count writes.
    ///
    /// Uses existing count facts without source reads or topology discovery. Skips reconstruction
    /// iteration and requires no heap allocation for empty or entirely unchanged work. Retains the
    /// reconstruction set and a shared identity for changed work. Source freshness, reference
    /// completeness, admission, and publication permissions require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`CountPreflightError::OriginalZero`] for an original zero count
    /// or [`CountPreflightError::MissingField`] for a changed positive implicit count without a
    /// reconstruction. Returns [`CountPreflightError::Protection`] for an inapplicable source
    /// association, unavailable publication, missing or overlapping complete allocations, or
    /// conflicting allocator claims.
    pub(in crate::edit) fn preflight<'origin>(
        self,
        engine: &WriteEngine,
        reconstructed: impl IntoIterator<Item = &'origin ReconstructionSource>,
    ) -> Result<CountPreflight, CountPreflightError> {
        let Self {
            session,
            origin,
            changes,
        } = &self;
        if !session.same(&engine.context.allocations.session()) {
            return Err(CountPreflightError::Protection(
                CountProtectionError::Session,
            ));
        }
        let changed = changes
            .iter()
            .any(|change| change.original().value() != change.remaining());
        let replaced = match origin {
            Some(origin) => {
                let source = CountSource {
                    session: session.clone(),
                    identity: Arc::clone(&origin.identity),
                    root: origin.object(),
                };
                if changed {
                    source.reconstructions(engine, reconstructed)
                } else {
                    source.reconstructions(engine, [])
                }
                .map_err(CountPreflightError::Protection)?
            }
            None => BTreeSet::new(),
        };
        for change in changes {
            if change.original().value() == 0 {
                return Err(CountPreflightError::OriginalZero {
                    object: change.object(),
                });
            }
            match change.disposition() {
                CountDisposition::Retain(_)
                | CountDisposition::Surviving { .. }
                | CountDisposition::Unchanged(_)
                | CountDisposition::Unresolved {
                    reason: UnresolvedCount::Zero,
                    ..
                } => {}
                CountDisposition::Unresolved {
                    change,
                    reason: UnresolvedCount::MissingField,
                } if !replaced.contains(&change.object()) => {
                    return Err(CountPreflightError::MissingField {
                        object: change.object(),
                        original: change.original().value(),
                        remaining: change.remaining(),
                    });
                }
                CountDisposition::Unresolved { .. } => {}
            }
        }
        let (inventory, surviving) = self.partition().into_parts();
        let SurvivingCounts { source, updates } = surviving;
        let protected = match source {
            Some(source) => {
                let protected = ProtectedCounts::new(source, updates, &replaced)
                    .map_err(CountPreflightError::Protection)?;
                protected
                    .exclude_space(engine)
                    .map_err(CountPreflightError::Protection)?;
                protected
            }
            None => ProtectedCounts::empty(),
        };
        Ok(CountPreflight {
            inventory,
            protected,
            reconstructed: replaced,
            projection: changed.then(|| Arc::new(ProjectionIdentity)),
        })
    }

    /// Retains the complete inventory alongside its changed positive explicit count entries.
    ///
    /// Preserves the inventory's allocation and entry order. Copies eligible fields into
    /// [`SurvivingCounts`] without source reads or topology discovery. Requires no heap allocation
    /// when no eligible entries exist. Other entries remain accessible through the complete
    /// inventory and their individual classifications.
    pub(in crate::edit) fn partition(self) -> CountPartition {
        let Self {
            session,
            origin,
            changes,
        } = self;
        let surviving = origin
            .as_ref()
            .map(|origin| {
                let updates = changes
                    .iter()
                    .filter_map(|change| {
                        let CountDisposition::Surviving {
                            change,
                            field,
                            count,
                        } = change.disposition()
                        else {
                            return None;
                        };
                        Some(SurvivingCount {
                            object: change.object(),
                            field: *field,
                            count,
                            allocation: change.header_allocation(),
                        })
                    })
                    .collect::<Vec<_>>();
                SurvivingCounts {
                    source: (!updates.is_empty()).then(|| CountSource {
                        session: session.clone(),
                        identity: Arc::clone(&origin.identity),
                        root: origin.object(),
                    }),
                    updates,
                }
            })
            .unwrap_or_else(|| SurvivingCounts {
                source: None,
                updates: Vec::new(),
            });
        CountPartition {
            inventory: Self {
                session,
                origin,
                changes,
            },
            surviving,
        }
    }

    /// Returns whether the recorded allocation session matches `engine`, even for empty input.
    ///
    /// Session equality is independent of source freshness.
    pub(in crate::edit) fn same_session(&self, engine: &WriteEngine) -> bool {
        self.session.same(&engine.context.allocations.session())
    }

    /// Returns whether `source` has the same shared phase identity as this inventory's origin.
    ///
    /// Returns `false` without a phase origin. Object identity and source freshness are separate
    /// conditions.
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        self.origin
            .as_ref()
            .is_some_and(|origin| Arc::ptr_eq(&origin.identity, &source.identity))
    }

    /// Returns the original physical root, or `None` for an empty inventory.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.origin.as_ref().map(ReconstructionSource::object)
    }

    /// Borrows every projection in the order of [`NetCountPlan::evidence_changes`].
    pub(in crate::edit) fn changes(&self) -> &[OwnedCountChange] {
        &self.changes
    }
}

/// A complete original inventory alongside its changed positive explicit count subset.
///
/// Unchanged, zero, and missing-field entries remain in the complete inventory.
pub(in crate::edit) struct CountPartition {
    inventory: CountInventory,
    surviving: SurvivingCounts,
}

impl CountPartition {
    /// Returns the complete inventory and its positive explicit count batch.
    pub(in crate::edit) fn into_parts(self) -> (CountInventory, SurvivingCounts) {
        let Self {
            inventory,
            surviving,
        } = self;
        (inventory, surviving)
    }
}

/// Owned count facts, checked source allocations, and retained reconstruction decisions.
///
/// Changed positive source fields have complete disjoint allocations checked against the recorded
/// allocator state. Reconstructed targets are excluded from that batch. Releasing the batch
/// requires proof of successful local projection under the same changed preflight identity.
/// Source freshness, reference completeness, admission, and publication remain separate conditions.
pub(in crate::edit) struct CountPreflight {
    inventory: CountInventory,
    protected: ProtectedCounts,
    reconstructed: BTreeSet<ObjectId>,
    projection: Option<Arc<ProjectionIdentity>>,
}

impl CountPreflight {
    /// Returns whether `source` shares the original inventory's phase identity.
    ///
    /// Returns `false` without an original phase. Source freshness is a separate condition.
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        self.inventory.shares_phase(source)
    }

    /// Returns the original physical root, or `None` for an empty inventory.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.inventory.root()
    }

    /// Borrows every original count entry in the inventory's established order.
    pub(in crate::edit) fn changes(&self) -> &[OwnedCountChange] {
        self.inventory.changes()
    }

    /// Borrows the positive source updates and their checked complete allocations.
    pub(in crate::edit) fn protected(&self) -> &ProtectedCounts {
        &self.protected
    }

    /// Returns whether `object` was supplied in the checked reconstruction set.
    pub(in crate::edit) fn reconstructed(&self, object: ObjectId) -> bool {
        self.reconstructed.contains(&object)
    }

    /// Clones the shared identity of changed count work without allocating.
    pub(in crate::edit) fn projection_identity(&self) -> CountProjectionIdentity {
        CountProjectionIdentity(self.projection.clone())
    }

    /// Returns the original inventory and protected source batch after matching local projection.
    ///
    /// The identity allocation and protected updates are absent for empty or entirely unchanged work.
    ///
    /// # Errors
    ///
    /// Returns [`CountProjectionError::Preflight`] for a proof associated with a different changed
    /// preflight.
    pub(in crate::edit) fn into_parts(
        self,
        proof: CountProjectionProof,
    ) -> Result<(CountInventory, ProtectedCounts), CountProjectionError> {
        if !proof.matches(&self.projection_identity()) {
            return Err(CountProjectionError::Preflight);
        }
        let Self {
            inventory,
            protected,
            reconstructed: _,
            projection: _,
        } = self;
        Ok((inventory, protected))
    }
}

/// Shared identity for changed count work, absent for empty or entirely unchanged work.
///
/// Equality of this association does not establish source freshness or publication permission.
pub(in crate::edit) struct CountProjectionIdentity(Option<Arc<ProjectionIdentity>>);

impl CountProjectionIdentity {
    /// Returns whether both handles share a preflight allocation or both have no allocation.
    pub(in crate::edit) fn same(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Some(expected), Some(actual)) => Arc::ptr_eq(expected, actual),
            (None, None) => true,
            _ => false,
        }
    }
}

/// Allocation identity shared by a changed preflight and its local projection.
struct ProjectionIdentity;

/// Unsupported owned count facts or an inapplicable source-allocation association.
#[derive(Debug)]
pub(in crate::edit) enum CountPreflightError {
    /// A changed positive implicit count without a matching reconstruction.
    MissingField {
        /// The physical identity of the original target.
        object: ObjectId,
        /// The original implicit count.
        original: u32,
        /// The planned positive count.
        remaining: u32,
    },
    /// An unsupported original zero, independently of any known reference minimum.
    OriginalZero {
        /// The physical identity of the original target.
        object: ObjectId,
    },
    /// A source, reconstruction, publication, or allocator-protection failure.
    Protection(CountProtectionError),
}

#[cfg(test)]
mod tests;
