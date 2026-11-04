//! Complete owned count inventories, source preflight, and positive explicit subsets.

use std::sync::Arc;

use super::{CountHandoffError, CountSource, SurvivingCount, SurvivingCounts};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::counts::changes::owned::{CountDisposition, OwnedCountChange};
use crate::edit::analysis::{NetCountPlan, ReconstructionSource};
use crate::object_graph::ObjectId;

pub(in crate::edit) use preflight::CountPreflight;
pub(in crate::edit) use preflight::CountProjectionIdentity;

mod preflight;

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
