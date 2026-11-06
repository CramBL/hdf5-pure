//! Combined count plans and owned storage summaries for copy users and hard-link removals.

use std::collections::BTreeMap;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::ObjectReferenceCountField;
use hdf5_pure_space::__private::FreeClass;

use super::PlannedDeletion;
use super::SourceCountState;
use super::SourceRemovalPlan;
use super::added::AddedCountError;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::counts::added_users::{AddedCopyUsers, CopyUsersError};
use crate::edit::analysis::counts::balance::{CountArithmeticError, CountBalance};
use crate::edit::analysis::counts::changes::{CountChange, OriginalCount};
use crate::edit::analysis::counts::referenced::{ReferencedCountError, ReferencedTarget};
use crate::edit::analysis::counts::{CountEvidenceError, HeaderCountAllocation};
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
                .map_err(|error| NetCountError::Arithmetic { object, error })?;
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

    /// Returns whether the reconstruction origin has this plan's shared phase identity.
    ///
    /// Returns `false` for an empty plan. Object identity, source freshness, and allocation
    /// ownership are separate conditions.
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        self.source
            .as_ref()
            .is_some_and(|origin| Arc::ptr_eq(&origin.identity, &source.identity))
    }

    /// Borrows the compatible removal view, or returns `None` for empty deletion input.
    pub(in crate::edit) fn removals(&self) -> Option<&SourceRemovalPlan<'phase, 'source>> {
        self.removals.as_ref()
    }

    /// Yields each represented target's original and final declared count once.
    ///
    /// Includes unchanged net counts. Removal-view targets are in topology index order, followed
    /// by other reference targets in physical object-header address order. The iterator borrows
    /// the plan and yields fixed-size numeric projections. Original typed evidence and phase
    /// association remain with the plan.
    pub(in crate::edit) fn changes(&self) -> impl Iterator<Item = NetCountChange> + '_ {
        self.evidence_changes().map(|change| {
            let remaining = change.remaining();
            match change.original() {
                OriginalCount::Topology(evidence) => {
                    let bound = evidence.bound();
                    NetCountChange {
                        object: bound.object(),
                        original: bound.original(),
                        remaining,
                        field: bound.field(),
                        allocation: bound.header_allocation(),
                    }
                }
                OriginalCount::Referenced(original) => NetCountChange {
                    object: original.object(),
                    original: original.original(),
                    remaining,
                    field: original.field(),
                    allocation: original.header_allocation(),
                },
            }
        })
    }

    /// Borrows each represented target's original facts and final declared count once.
    ///
    /// Uses the ordering and target set of [`Self::changes`], including unchanged counts. Original
    /// facts are borrowed unchanged with their [`OriginalCount`] class. Iteration requires no source
    /// reads, topology discovery, count-arithmetic recalculation, or heap allocation. Source
    /// freshness, ownership, retirement, and publication permissions remain separate requirements.
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
            source,
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
            origin: source.as_ref().map(|source| ReconstructionSource {
                identity: Arc::clone(&source.identity),
                object: source.object(),
            }),
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

/// An owned numeric projection of an original and final declared count.
///
/// Original physical fields and checked allocation geometry are optional. The originating
/// [`NetCountPlan`] retains the typed original evidence and phase association. Source freshness,
/// allocation ownership, retirement eligibility, and count-write permission require separate
/// evidence.
pub(in crate::edit) struct NetCountChange {
    object: ObjectId,
    original: u32,
    remaining: u32,
    field: Option<ObjectReferenceCountField>,
    allocation: Option<HeaderCountAllocation>,
}

impl NetCountChange {
    /// Returns the physical object-header identity of the represented target.
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }

    /// Returns the original observed declared count, including an implicit one.
    pub(in crate::edit) fn original(&self) -> u32 {
        self.original
    }

    /// Returns the final planned declared count.
    pub(in crate::edit) fn remaining(&self) -> u32 {
        self.remaining
    }

    /// Returns the original physical count field, or `None` for an implicit count.
    pub(in crate::edit) fn field(&self) -> Option<ObjectReferenceCountField> {
        self.field
    }

    /// Returns the independently checked complete allocation containing an explicit count, if known.
    pub(in crate::edit) fn header_allocation(&self) -> Option<HeaderCountAllocation> {
        self.allocation
    }
}

/// A count-arithmetic failure, incompatible source association, or original-evidence error.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum NetCountError {
    /// A checked count-arithmetic failure for a physical target.
    Arithmetic {
        /// The physical object-header identity of the affected count.
        object: ObjectId,
        /// The arithmetic failure and its original operands.
        error: CountArithmeticError,
    },
    /// A mismatched engine, phase, or root, or failed root-address resolution.
    Association(CopyUsersError),
    /// A source-count failure, original contradiction, or inconsistent deletion or count evidence.
    Evidence(CountEvidenceError),
}

impl From<AddedCountError> for NetCountError {
    /// Preserves a removal-traversal arithmetic or evidence error.
    fn from(error: AddedCountError) -> Self {
        match error {
            AddedCountError::Arithmetic { object, error } => Self::Arithmetic { object, error },
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
/// Nonempty plans retain their original phase identity and physical root. Allocation boundaries
/// and free classes are preserved. Source freshness, reference completeness, admission, and
/// publication permissions require separate evidence.
pub(in crate::edit) struct CountStorage {
    session: AllocationSession,
    origin: Option<ReconstructionSource>,
    allocations: Vec<(FileRange, FreeClass)>,
    reference_roots: Vec<ObjectId>,
}

impl CountStorage {
    /// Returns whether the recorded allocation session matches `engine`, even for empty input.
    ///
    /// Session equality is independent of source freshness.
    pub(in crate::edit) fn same_session(&self, engine: &WriteEngine) -> bool {
        self.session.same(&engine.context.allocations.session())
    }

    /// Returns whether `source` has the same shared phase identity as this batch's origin.
    ///
    /// Returns `false` without a phase origin. Object identity and source freshness are separate
    /// conditions.
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        self.origin
            .as_ref()
            .is_some_and(|origin| Arc::ptr_eq(&origin.identity, &source.identity))
    }

    /// Returns the original physical root, or `None` for an empty plan.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.origin.as_ref().map(ReconstructionSource::object)
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
