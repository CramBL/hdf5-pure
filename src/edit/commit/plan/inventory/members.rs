//! Qualified global heap removals and protected parent compaction plans.

use std::collections::{BTreeMap, BTreeSet};

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCompaction;

use super::prospective::ProspectiveCollections;
use super::prospective::ReferencePresence;
use super::storage::{OriginalStorage, StorageOrigin};
use crate::edit::analysis::CollectionProvenance;
use crate::edit::analysis::heap::qualification::{
    MemberUnavailable, OriginalCollections, OriginalQualification, ParentUnavailable,
};
use crate::error::FormatError;
use crate::global_heap::GlobalHeapId;

/// Member and parent removal decisions with independent compaction planning outcomes.
///
/// Parent allocation release requires separate permission.
#[derive(Default)]
pub(in crate::edit::commit) struct MemberDecisions {
    members:
        BTreeMap<(CollectionProvenance, GlobalHeapId), Result<MemberDecision, MemberUnavailable>>,
    parents: BTreeMap<PhysicalAddress, Result<MemberDecision, ParentUnavailable>>,
    planning: BTreeMap<PhysicalAddress, Result<(), CompactionUnavailable>>,
    plans: Vec<GlobalHeapCompaction>,
    #[cfg(test)]
    physical_members: usize,
}

impl MemberDecisions {
    /// Derives removal decisions from original qualification and fresh prospective references.
    ///
    /// The planner groups origins by physical member before looking up prospective presence once
    /// per identity. It retains each origin's qualification outcome and each original parent
    /// address, including qualification failures. It preserves a qualified member or parent if a
    /// prospective reference was observed, even when other observations are incomplete. It requires
    /// original qualification, complete interpretation, and root applicability for removal.
    ///
    /// The planner groups removals by parent and preserves unobserved members, serialized reference
    /// counts, and collection capacity. It requires captured framing and an exactly matching
    /// protected parent range before constructing a plan.
    pub(super) fn collect(
        original: &OriginalCollections,
        prospective: &ProspectiveCollections,
        storage: &OriginalStorage,
    ) -> Self {
        let mut result = Self::default();
        let qualification = match original.qualification() {
            OriginalQualification::Qualified(qualification) => Some(qualification),
            OriginalQualification::Empty | OriginalQualification::Unavailable { .. } => None,
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
        let mut origins_by_member = BTreeMap::<_, Vec<_>>::new();
        for (origin, observations) in original.predecessors().into_iter().flatten() {
            for &id in observations.objects() {
                origins_by_member.entry(id).or_default().push(origin);
            }
        }
        for (id, origins) in origins_by_member {
            #[cfg(test)]
            {
                result.physical_members += 1;
            }
            let decision = MemberDecision::from(prospective.member_presence(id));
            let mut remove = false;
            for origin in origins {
                let outcome = qualification
                    .ok_or(MemberUnavailable::Qualification)
                    .and_then(|qualification| qualification.member(origin, id))
                    .map(|()| decision);
                remove |= outcome == Ok(MemberDecision::Remove);
                result.members.insert((origin.clone(), id), outcome);
            }
            if remove {
                removed.entry(id.collection).or_default().insert(id.index);
            }
        }
        for &address in original.addresses() {
            result.parents.insert(
                address,
                qualification
                    .ok_or(ParentUnavailable::Qualification)
                    .and_then(|qualification| qualification.whole_parent(address))
                    .map(|()| MemberDecision::from(prospective.parent_presence(address))),
            );
            let Some(indices) = removed.get(&address) else {
                continue;
            };
            let planned = original
                .contents()
                .get(&address)
                .and_then(|contents| contents.as_ref().ok())
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
            result
                .planning
                .insert(address, planned.map(|plan| result.plans.push(plan)));
        }
        result
    }

    /// Borrows per-origin member qualification and prospective reference outcomes.
    pub(in crate::edit::commit) fn members(
        &self,
    ) -> &BTreeMap<(CollectionProvenance, GlobalHeapId), Result<MemberDecision, MemberUnavailable>>
    {
        &self.members
    }

    /// Borrows stronger whole-parent qualification and prospective reference outcomes.
    pub(in crate::edit::commit) fn parents(
        &self,
    ) -> &BTreeMap<PhysicalAddress, Result<MemberDecision, ParentUnavailable>> {
        &self.parents
    }

    /// Borrows planning outcomes for parents with eligible member removals.
    pub(in crate::edit::commit) fn planning(
        &self,
    ) -> &BTreeMap<PhysicalAddress, Result<(), CompactionUnavailable>> {
        &self.planning
    }

    /// Borrows at most one compaction plan per protected parent.
    pub(in crate::edit::commit) fn plans(&self) -> &[GlobalHeapCompaction] {
        &self.plans
    }

    /// Returns the number of distinct physical member identities examined.
    #[cfg(test)]
    pub(in crate::edit::commit) fn physical_members(&self) -> usize {
        self.physical_members
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

/// Maps observed presence to retention and complete absence to removal.
///
/// Original qualification is checked separately by [`MemberDecisions::collect`].
impl From<ReferencePresence> for MemberDecision {
    fn from(presence: ReferencePresence) -> Self {
        match presence {
            ReferencePresence::ObservedLive => Self::Live,
            ReferencePresence::Unavailable => Self::Prospective,
            ReferencePresence::Unreferenced => Self::Remove,
        }
    }
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
