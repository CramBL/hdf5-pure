//! Combined count plans and owned storage summaries for copy users and hard-link removals.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;

use super::PlannedDeletion;
use super::SourceCountState;
use super::SourceRemovalPlan;
use super::added::AddedCountError;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::edit::analysis::counts::added_users::{AddedCopyUsers, CopyUsersError};
use crate::edit::analysis::counts::balance::{CountArithmeticError, CountBalance};
use crate::edit::analysis::counts::changes::{CountChange, OriginalCount};
use crate::edit::analysis::counts::referenced::{ReferencedCountError, ReferencedTarget};
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::error::FormatError;
use crate::object_graph::ObjectId;

pub(super) mod original;

impl<'source> AnalysisPhase<'source> {
    /// Plans declared net counts for supplied copy users and original hard-link removals.
    ///
    /// Requires the original superblock root for nonempty work. Checks the addition association
    /// before topology discovery or count reads. Discovers topology only for nonempty deletion
    /// input and reuses the phase's count cache and shared edge-removal traversal.
    ///
    /// Returns `None` if required topology, edge identity, or count observations are unavailable.
    /// The private plan is dropped on incompleteness or error. Independently valid phase-cache
    /// entries are preserved. Returns an engine-bound empty plan for empty inventories, with no
    /// phase origin, source reads, topology discovery, or heap allocation.
    ///
    /// Each target has one numeric projection, including copy targets untouched by removals or
    /// outside the topology. Original counts, fields, optional allocation facts, and typed evidence
    /// are retained. The caller establishes completeness and same-file applicability of the actual
    /// input inventories. Source freshness, original-user minima, ownership, retirement, and write
    /// or publication permissions remain separate requirements.
    ///
    /// # Errors
    ///
    /// Returns [`NetCountError::Association`] for incompatible engine, phase, or root associations
    /// or failed root resolution. Returns [`NetCountError::Evidence`] for original contradictions,
    /// source-count failures, or inconsistent deletion or count evidence. Returns
    /// [`NetCountError::Arithmetic`] for original-count underflow or final net-count overflow.
    pub(in crate::edit) fn net_count_plan<'deletion>(
        &mut self,
        deletions: impl IntoIterator<Item = &'deletion PlannedDeletion>,
        added: &AddedCopyUsers<'source>,
    ) -> Result<Option<NetCountPlan<'_, 'source>>, NetCountError> {
        let mut deletions = deletions.into_iter().peekable();
        let has_deletions = deletions.peek().is_some();
        let first = added.users().next();
        let engine = self.engine;
        if !has_deletions && first.is_none() {
            return Ok(Some(NetCountPlan::empty(engine)));
        }
        if let Some((object, _)) = first
            && added
                .source_for(self, object)
                .map_err(NetCountError::Association)?
                .is_none()
        {
            return Ok(None);
        }
        let root = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(|error| NetCountError::Association(CopyUsersError::Address(error)))?;
        if let Some(actual) = self.root
            && actual != root
        {
            return Err(NetCountError::Association(CopyUsersError::Root {
                expected: root,
                actual,
            }));
        }
        if has_deletions && self.topology().is_none() {
            return Ok(None);
        }
        let Some(referenced) = self.referenced_counts(added)? else {
            return Ok(None);
        };
        let source = self.reconstruction_source(engine, root)?;
        let (removed, mut queries) = self.removal_queries(deletions)?;
        if !removed.unresolved.is_empty() {
            return Ok(None);
        }
        let removals = SourceRemovalPlan::with_additions(
            removed,
            |object| queries.evidence(object),
            |object| {
                referenced
                    .get(object)
                    .map_or(0, |target| target.added().get())
            },
        )?;
        if removals.as_ref().is_some_and(|plan| {
            plan.plan
                .remaining
                .values()
                .any(|state| state.declared.is_none())
        }) {
            return Ok(None);
        }
        let mut remaining = BTreeMap::new();
        for (object, target) in referenced.into_targets() {
            if removals
                .as_ref()
                .is_some_and(|plan| plan.declared_remaining(object).is_some())
            {
                continue;
            }
            let count = CountBalance::new(target.count().original(), target.added().get())
                .remaining()
                .map_err(NetCountError::Arithmetic)?;
            remaining.insert(object, ReferencedChange { target, count });
        }
        Ok(Some(NetCountPlan {
            engine,
            source: Some(source),
            removals,
            referenced: remaining,
        }))
    }
}

/// Complete numeric projections for the supplied addition and removal inventories.
///
/// Nonempty plans retain the original-root phase association and typed original evidence. The
/// compatible removal view is retained alongside reference targets absent from that view. Empty
/// plans retain the engine association, with no phase origin or count changes.
///
/// Coverage concerns the supplied inventories. Source freshness, original-user completeness,
/// allocation ownership, retirement eligibility, and count-write or publication permission require
/// separate evidence.
pub(in crate::edit) struct NetCountPlan<'phase, 'source> {
    engine: &'source WriteEngine,
    source: Option<ReconstructionSource>,
    removals: Option<SourceRemovalPlan<'phase, 'source>>,
    referenced: BTreeMap<ObjectId, ReferencedChange<'source>>,
}

impl<'phase, 'source> NetCountPlan<'phase, 'source> {
    /// Creates an engine-bound empty plan with no phase origin or heap allocation.
    fn empty(engine: &'source WriteEngine) -> Self {
        Self {
            engine,
            source: None,
            removals: None,
            referenced: BTreeMap::new(),
        }
    }

    /// Returns whether `engine` is the engine borrowed by this plan.
    ///
    /// Pointer identity is separate from source-content validity.
    pub(in crate::edit) fn same_engine(&self, engine: &WriteEngine) -> bool {
        std::ptr::eq(self.engine, engine)
    }

    /// Borrows the original-root phase origin, or returns `None` for an empty plan.
    pub(in crate::edit) fn source(&self) -> Option<&ReconstructionSource> {
        self.source.as_ref()
    }

    /// Borrows the compatible removal view, or returns `None` for empty deletion input.
    pub(in crate::edit) fn removals(&self) -> Option<&SourceRemovalPlan<'phase, 'source>> {
        self.removals.as_ref()
    }

    /// Borrows each represented target's original facts and final declared count once.
    ///
    /// Yields removal targets in topology index order, then other referenced targets in physical
    /// address order. Includes unchanged counts and preserves the [`OriginalCount`] class.
    /// Iteration requires no source reads or allocation.
    pub(in crate::edit) fn evidence_changes(
        &self,
    ) -> impl Iterator<Item = CountChange<'_, 'source>> {
        let Self {
            engine: _,
            source: _,
            removals,
            referenced,
        } = self;
        removals
            .iter()
            .flat_map(|plan| plan.plan.remaining.values())
            .filter_map(|state| {
                let SourceCountState {
                    evidence,
                    graph: _,
                    declared,
                } = state;
                evidence
                    .as_ref()
                    .zip(*declared)
                    .map(|(evidence, remaining)| {
                        CountChange::new(OriginalCount::Topology(evidence), remaining)
                    })
            })
            .chain(referenced.values().map(|change| {
                let ReferencedChange { target, count } = change;
                CountChange::new(OriginalCount::Referenced(target.count()), *count)
            }))
    }

    /// Collects supported allocations from eligible objects and retains the other reference roots.
    ///
    /// Enumerates each zero-count object's storage once through the existing ownership collector.
    /// Preserves original physical allocation boundaries and free classes, including independently
    /// proven storage when another part of an object's ownership is unavailable.
    ///
    /// Retains touched removal-view objects ineligible for collection, followed by referenced
    /// targets outside that view. Uses the existing plan without topology discovery. An empty
    /// plan requires no source reads or heap allocation.
    pub(in crate::edit) fn collect_storage(&self) -> CountStorage {
        let Self {
            engine,
            source: _,
            removals,
            referenced,
        } = self;
        let mut allocations = Vec::new();
        if let Some(removals) = removals {
            removals.collect_storage(&mut allocations);
        }
        let reference_roots = removals
            .iter()
            .flat_map(|removals| removals.retained_objects())
            .chain(referenced.keys().copied())
            .collect();
        CountStorage {
            session: engine.context.allocations.session(),
            allocations,
            reference_roots,
        }
    }
}

/// A reference target and its checked final count outside the removal view.
struct ReferencedChange<'source> {
    target: ReferencedTarget<'source>,
    count: u32,
}

/// A count-arithmetic failure, incompatible source association, or original-evidence error.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum NetCountError {
    /// An original-count underflow or final net-count overflow.
    Arithmetic(CountArithmeticError),
    /// A mismatched engine, phase, or root, or failed root-address resolution.
    Association(CopyUsersError),
    /// A source-count failure, original contradiction, or inconsistent deletion or count evidence.
    Evidence(CountEvidenceError),
}

impl From<AddedCountError> for NetCountError {
    /// Preserves a removal-traversal arithmetic or evidence error.
    fn from(error: AddedCountError) -> Self {
        match error {
            AddedCountError::Arithmetic(error) => Self::Arithmetic(error),
            AddedCountError::Evidence(error) => Self::Evidence(error),
        }
    }
}

impl From<ReferencedCountError> for NetCountError {
    /// Preserves a referenced-count association or evidence error.
    fn from(error: ReferencedCountError) -> Self {
        match error {
            ReferencedCountError::Association(error) => Self::Association(error),
            ReferencedCountError::Evidence(error) => Self::Evidence(error),
        }
    }
}

impl From<FormatError> for NetCountError {
    /// Classifies a structural planning failure as an evidence error.
    fn from(error: FormatError) -> Self {
        Self::Evidence(error.into())
    }
}

/// Supported original allocations and retained reference roots bound to one allocation session.
///
/// The enclosing original-owner facts retain the phase and physical root in the count inventory.
/// Allocation boundaries and free classes are preserved. Source freshness, reference completeness, admission, and
/// publication permissions require separate evidence.
pub(in crate::edit) struct CountStorage {
    session: AllocationSession,
    allocations: Vec<(FileRange, FreeClass)>,
    reference_roots: Vec<ObjectId>,
}

impl CountStorage {
    /// Separates original allocations and retained reference roots without reanalysis.
    pub(in crate::edit) fn into_parts(self) -> (Vec<(FileRange, FreeClass)>, Vec<ObjectId>) {
        let Self {
            session: _,
            allocations,
            reference_roots,
        } = self;
        (allocations, reference_roots)
    }

    /// Returns whether the recorded allocation session matches `engine`, even for empty input.
    ///
    /// Session equality is independent of source freshness.
    pub(in crate::edit) fn same_session(&self, engine: &WriteEngine) -> bool {
        self.session.same(&engine.context.allocations.session())
    }

    /// Borrows complete physical allocations in object propagation and collector order.
    ///
    /// The allocation set can be incomplete when ownership of other storage is unproven. Original
    /// boundaries and free classes are preserved without sorting or coalescing.
    pub(in crate::edit) fn allocations(&self) -> &[(FileRange, FreeClass)] {
        &self.allocations
    }

    /// Borrows each retained object once for reference analysis.
    ///
    /// Touched removal-view objects are in topology index order, followed by other reference
    /// targets in physical object-header address order. Includes targets outside the topology.
    pub(in crate::edit) fn reference_roots(&self) -> &[ObjectId] {
        &self.reference_roots
    }
}
