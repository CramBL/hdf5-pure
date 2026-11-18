//! Qualified global heap removals and protected parent compaction plans.

use std::collections::{BTreeMap, BTreeSet};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCompaction;
use hdf5_pure_space::__private::FreeClass;

use super::dispositions::AppliedDispositions;
use super::prospective::{ParentAbsence, ProspectiveCollections, ReferencePresence};
use super::session::SessionParentRetirement;
use super::storage::{OriginalStorage, StorageOrigin};
use crate::edit::allocation::AllocationSession;
#[cfg(test)]
use crate::edit::analysis::CollectionProvenance;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::qualification::QualifiedParentAllocation;
use crate::edit::analysis::heap::qualification::{OriginalCollections, OriginalQualification};
use crate::error::FormatError;
#[cfg(test)]
use crate::global_heap::GlobalHeapId;

/// Member and parent removal decisions with independent compaction planning outcomes.
///
/// Parent allocation release requires separate permission.
#[derive(Default)]
pub(in crate::edit::commit) struct MemberDecisions {
    #[cfg(test)]
    members: BTreeMap<(CollectionProvenance, GlobalHeapId), Option<MemberDecision>>,
    #[cfg(test)]
    parents: BTreeMap<PhysicalAddress, Option<MemberDecision>>,
    #[cfg(test)]
    planning: BTreeMap<PhysicalAddress, Result<(), CompactionUnavailable>>,
    actions: BTreeMap<PhysicalAddress, ParentDisposition>,
}

impl MemberDecisions {
    /// Derives removal decisions from original qualification and fresh prospective references.
    ///
    /// The planner preserves a qualified member or parent if a prospective reference was
    /// observed, even when reference interpretation is incomplete. Removal requires complete
    /// interpretation and root applicability. The planner groups removals by parent and preserves
    /// unobserved members, serialized reference counts, and collection capacity. It requires captured
    /// framing and an exactly matching protected parent range before constructing a plan.
    ///
    /// Returns empty decisions if original qualification is unavailable.
    pub(super) fn collect(
        original: &OriginalCollections,
        prospective: &ProspectiveCollections,
        storage: &OriginalStorage,
    ) -> Self {
        let mut result = Self::default();
        let OriginalQualification::Qualified(qualification) = original.qualification() else {
            return result;
        };
        let decision = |presence| match presence {
            ReferencePresence::ObservedLive => MemberDecision::Live,
            ReferencePresence::Unreferenced => MemberDecision::Remove,
            ReferencePresence::Unavailable => MemberDecision::Prospective,
        };
        let protected: BTreeMap<_, _> = storage
            .allocations()
            .iter()
            .filter_map(|allocation| {
                if let StorageOrigin::Parent { address, .. } = allocation.origin {
                    Some((address, allocation.range))
                } else {
                    None
                }
            })
            .collect();
        let mut removed: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
        for (origin, observations) in qualification.predecessors().into_iter().flatten() {
            for &id in observations.objects() {
                let outcome = qualification
                    .member(origin, id)
                    .then(|| decision(prospective.member_presence(id)));
                if outcome == Some(MemberDecision::Remove) {
                    removed.entry(id.collection).or_default().insert(id.index);
                }
                #[cfg(test)]
                result.members.insert((origin.clone(), id), outcome);
            }
        }
        for &address in qualification.parents().keys() {
            let parent = qualification
                .whole_parent(address)
                .then(|| decision(prospective.parent_presence(address)));
            #[cfg(test)]
            result.parents.insert(address, parent);
            let retired = (parent == Some(MemberDecision::Remove))
                .then(|| qualification.whole_parent_allocation(address))
                .flatten()
                .filter(|allocation| protected.get(&address) == Some(&allocation.range()))
                .and_then(|allocation| {
                    prospective
                        .parent_absence(address)
                        .and_then(|absence| ParentRetirement::new(allocation, absence))
                });
            if let Some(retirement) = retired {
                result
                    .actions
                    .insert(address, ParentDisposition::Retire(retirement));
                continue;
            }
            let Some(indices) = removed.get(&address) else {
                result.actions.insert(address, ParentDisposition::Retain);
                continue;
            };
            let planned = original
                .contents()
                .get(&address)
                .ok_or(CompactionUnavailable::Framing)
                .and_then(|contents| {
                    if protected.get(&address) != Some(&contents.range()) {
                        return Err(CompactionUnavailable::Protection);
                    }
                    contents
                        .framing()
                        .compact(|member| indices.contains(&member.index()))
                        .map_err(CompactionUnavailable::Planning)
                });
            let outcome = planned.map(|plan| {
                result
                    .actions
                    .insert(address, ParentDisposition::Compact(plan));
            });
            #[cfg(test)]
            result.planning.insert(address, outcome);
            #[cfg(not(test))]
            drop(outcome);
            result
                .actions
                .entry(address)
                .or_insert(ParentDisposition::Retain);
        }
        result
    }

    /// Borrows per-origin member qualification and prospective reference outcomes.
    #[cfg(test)]
    pub(in crate::edit::commit) fn members(
        &self,
    ) -> &BTreeMap<(CollectionProvenance, GlobalHeapId), Option<MemberDecision>> {
        &self.members
    }

    /// Borrows stronger whole-parent qualification and prospective reference outcomes.
    #[cfg(test)]
    pub(in crate::edit::commit) fn parents(
        &self,
    ) -> &BTreeMap<PhysicalAddress, Option<MemberDecision>> {
        &self.parents
    }

    /// Borrows planning outcomes for parents with eligible member removals.
    #[cfg(test)]
    pub(in crate::edit::commit) fn planning(
        &self,
    ) -> &BTreeMap<PhysicalAddress, Result<(), CompactionUnavailable>> {
        &self.planning
    }

    /// Borrows at most one compaction plan per protected parent.
    pub(in crate::edit::commit) fn plans(&self) -> impl Iterator<Item = &GlobalHeapCompaction> {
        self.actions.values().filter_map(|action| match action {
            ParentDisposition::Compact(plan) => Some(plan),
            ParentDisposition::Retain | ParentDisposition::Retire(_) => None,
        })
    }

    pub(super) fn dispositions(&self) -> &BTreeMap<PhysicalAddress, ParentDisposition> {
        &self.actions
    }

    pub(super) fn into_compactions(mut self) -> Self {
        self.actions
            .retain(|_, action| matches!(action, ParentDisposition::Compact(_)));
        self
    }

    pub(super) fn include_session_parents(
        &mut self,
        dispositions: &AppliedDispositions,
        prospective: &ProspectiveCollections,
    ) {
        for (&address, parent) in &dispositions.original().session_parents {
            if let Some(retirement) = parent.retirement(dispositions, prospective) {
                self.actions.insert(
                    address,
                    ParentDisposition::Retire(ParentRetirement::from_session(retirement)),
                );
            }
        }
    }
}

/// The removal eligibility of an originally qualified member or parent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum MemberDecision {
    /// A prospective reference to the member or parent was observed.
    Live,
    /// Complete prospective interpretation or root applicability is unavailable.
    Prospective,
    /// The member or parent is absent from complete prospective references.
    Remove,
}

/// An unmet prerequisite for planning a protected parent compaction.
#[derive(Debug)]
pub(in crate::edit::commit) enum CompactionUnavailable {
    /// Complete original collection framing is unavailable.
    Framing,
    /// Compaction planning failed.
    Planning(FormatError),
    /// A protected range matching the captured collection range is unavailable.
    Protection,
}

/// An exclusive prepared action for one global heap parent.
///
/// The planner stores one disposition per physical parent. Publication, allocation release, and
/// changes to storage protection require separate checks.
pub(in crate::edit::commit) enum ParentDisposition {
    /// A checked compaction plan for selected obsolete members.
    Compact(GlobalHeapCompaction),
    /// The original allocation is preserved without member mutation.
    Retain,
    /// Original whole-parent retirement evidence with matching source and session associations.
    Retire(ParentRetirement),
}

/// Original allocation evidence bound to a prospective root for whole-parent retirement.
///
/// Uses qualified original allocation and complete prospective absence, or exclusive session
/// ownership and reconciled owner replacement. Both forms retain matching allocation-session
/// and original-source associations. Publication and storage-release permission require
/// separate checks.
pub(in crate::edit::commit) struct ParentRetirement {
    allocation: ParentAllocation,
    root: PhysicalAddress,
}

impl ParentRetirement {
    /// Binds original allocation evidence to complete absence from a prospective root.
    ///
    /// Returns `None` when the address, session, or source associations differ.
    fn new(allocation: QualifiedParentAllocation, absence: ParentAbsence) -> Option<Self> {
        if allocation.range().start() != absence.address()
            || !allocation.session().same(absence.session())
            || allocation.source() != absence.source()
        {
            return None;
        }
        Some(Self {
            allocation: ParentAllocation::Qualified(allocation),
            root: absence.root(),
        })
    }

    fn from_session(retirement: SessionParentRetirement) -> Self {
        let root = retirement.root();
        Self {
            allocation: ParentAllocation::Session(retirement),
            root,
        }
    }

    /// Borrows the matched original source-phase and object association.
    pub(in crate::edit::commit) fn source(&self) -> &ReconstructionSource {
        self.allocation.source()
    }

    /// Borrows the allocation session shared by the original and prospective evidence.
    pub(in crate::edit::commit) fn session(&self) -> &AllocationSession {
        self.allocation.session()
    }

    /// Returns the prospective root associated with the retirement evidence.
    pub(in crate::edit::commit) fn root(&self) -> PhysicalAddress {
        self.root
    }

    /// Returns the complete original allocation range.
    pub(in crate::edit::commit) fn range(&self) -> FileRange {
        self.allocation.range()
    }

    /// Returns the original allocation's free-space class.
    pub(in crate::edit::commit) fn class(&self) -> FreeClass {
        self.allocation.class()
    }
}

/// Original allocation evidence retained until checked parent release.
pub(in crate::edit::commit) enum ParentAllocation {
    /// Whole-parent capacity qualified from original file observations.
    Qualified(QualifiedParentAllocation),
    /// Whole-parent capacity established by exclusive session ownership.
    Session(SessionParentRetirement),
}

impl ParentAllocation {
    /// Borrows the original root's source-phase association.
    pub(in crate::edit::commit) fn source(&self) -> &ReconstructionSource {
        match self {
            Self::Qualified(allocation) => allocation.source(),
            Self::Session(allocation) => allocation.source(),
        }
    }

    /// Borrows the allocation session matched with prospective observations.
    pub(in crate::edit::commit) fn session(&self) -> &AllocationSession {
        match self {
            Self::Qualified(allocation) => allocation.session(),
            Self::Session(allocation) => allocation.session(),
        }
    }

    /// Returns the complete original parent allocation.
    pub(in crate::edit::commit) fn range(&self) -> FileRange {
        match self {
            Self::Qualified(allocation) => allocation.range(),
            Self::Session(allocation) => allocation.range(),
        }
    }

    /// Returns the original parent allocation's free-space class.
    pub(in crate::edit::commit) fn class(&self) -> FreeClass {
        match self {
            Self::Qualified(allocation) => allocation.class(),
            Self::Session(allocation) => allocation.class(),
        }
    }
}
