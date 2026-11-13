//! Qualified global heap member decisions and exclusive parent dispositions.

use std::collections::{BTreeMap, BTreeSet};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCompaction;
use hdf5_pure_space::__private::FreeClass;

use super::prospective::ParentAbsence;
use super::prospective::ProspectiveCollections;
use super::prospective::ReferencePresence;
use super::storage::{OriginalStorage, StorageOrigin};
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::CollectionProvenance;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::qualification::QualifiedParentAllocation;
use crate::edit::analysis::heap::qualification::{
    MemberUnavailable, OriginalCollections, OriginalQualification, ParentUnavailable,
};
use crate::error::FormatError;
use crate::global_heap::GlobalHeapId;

/// Member and parent removal decisions with one disposition per physical parent.
///
/// Compaction and whole-parent retirement are exclusive dispositions. Publication and parent
/// allocation release require separate permission.
#[derive(Default)]
pub(in crate::edit::commit) struct MemberDecisions {
    members:
        BTreeMap<(CollectionProvenance, GlobalHeapId), Result<MemberDecision, MemberUnavailable>>,
    parents: BTreeMap<PhysicalAddress, Result<MemberDecision, ParentUnavailable>>,
    planning: BTreeMap<PhysicalAddress, Result<(), CompactionUnavailable>>,
    dispositions: BTreeMap<PhysicalAddress, ParentDisposition>,
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
    /// protected parent range before constructing a plan. For a wholly unreferenced qualified
    /// parent, it binds complete allocation and absence evidence as a retirement disposition.
    /// For other parents, it selects compaction or retention. Planning restrictions are
    /// recorded independently, with the original allocation retained on failure.
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
            let parent = qualification
                .ok_or(ParentUnavailable::Qualification)
                .and_then(|qualification| qualification.whole_parent(address))
                .map(|()| MemberDecision::from(prospective.parent_presence(address)));
            result.parents.insert(address, parent);
            let disposition = if parent == Ok(MemberDecision::Remove) {
                qualification
                    .ok_or(CompactionUnavailable::Parent(
                        ParentUnavailable::Qualification,
                    ))
                    .and_then(|qualification| {
                        qualification
                            .whole_parent_allocation(address)
                            .map_err(CompactionUnavailable::Parent)
                    })
                    .and_then(|allocation| {
                        if protected.get(&address) != Some(&allocation.range()) {
                            return Err(CompactionUnavailable::Protection);
                        }
                        let absence = prospective
                            .parent_absence(address)
                            .ok_or(CompactionUnavailable::Prospective)?;
                        ParentRetirement::new(allocation, absence)
                            .map(ParentDisposition::Retire)
                            .map_err(CompactionUnavailable::Retirement)
                    })
            } else if let Some(indices) = removed.get(&address) {
                original
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
                            .map(ParentDisposition::Compact)
                            .map_err(CompactionUnavailable::Planning)
                    })
            } else {
                result
                    .dispositions
                    .insert(address, ParentDisposition::Retain);
                continue;
            };
            match disposition {
                Ok(disposition) => {
                    result.planning.insert(address, Ok(()));
                    result.dispositions.insert(address, disposition);
                }
                Err(error) => {
                    result.planning.insert(address, Err(error));
                    result
                        .dispositions
                        .insert(address, ParentDisposition::Retain);
                }
            }
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

    /// Borrows compaction and whole-parent retirement planning outcomes.
    pub(in crate::edit::commit) fn planning(
        &self,
    ) -> &BTreeMap<PhysicalAddress, Result<(), CompactionUnavailable>> {
        &self.planning
    }

    /// Borrows the exclusive disposition of each original parent in physical address order.
    pub(in crate::edit::commit) fn dispositions(
        &self,
    ) -> &BTreeMap<PhysicalAddress, ParentDisposition> {
        &self.dispositions
    }

    /// Iterates over borrowed compaction plans in physical parent-address order.
    ///
    /// The iterator yields only plans from [`ParentDisposition::Compact`].
    pub(in crate::edit::commit) fn plans(&self) -> impl Iterator<Item = &GlobalHeapCompaction> {
        self.dispositions
            .values()
            .filter_map(|disposition| match disposition {
                ParentDisposition::Compact(plan) => Some(plan),
                ParentDisposition::Retain | ParentDisposition::Retire(_) => None,
            })
    }

    /// Returns the number of distinct physical member identities examined.
    #[cfg(test)]
    pub(in crate::edit::commit) fn physical_members(&self) -> usize {
        self.physical_members
    }
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
    /// Qualified whole-allocation retirement evidence paired with prospective absence.
    Retire(ParentRetirement),
}

/// A qualified original allocation paired with matching prospective absence.
///
/// Construction matches the collection address, allocation session, and original source
/// association. The prospective root is retained. Publication and storage-release permission
/// require separate checks.
pub(in crate::edit::commit) struct ParentRetirement {
    allocation: QualifiedParentAllocation,
    root: PhysicalAddress,
}

impl ParentRetirement {
    /// Binds original allocation evidence to complete absence from a prospective root.
    ///
    /// # Errors
    ///
    /// Returns [`ParentRetirementMismatch::Address`] for different collection addresses,
    /// [`ParentRetirementMismatch::Session`] for different allocation sessions, or
    /// [`ParentRetirementMismatch::Source`] for different original source associations.
    fn new(
        allocation: QualifiedParentAllocation,
        absence: ParentAbsence,
    ) -> Result<Self, ParentRetirementMismatch> {
        if allocation.range().start() != absence.address() {
            return Err(ParentRetirementMismatch::Address {
                allocation: allocation.range().start(),
                reference: absence.address(),
            });
        }
        if !allocation.session().same(absence.session()) {
            return Err(ParentRetirementMismatch::Session);
        }
        if allocation.source() != absence.source() {
            return Err(ParentRetirementMismatch::Source {
                allocation: allocation.source().clone(),
                reference: absence.source().clone(),
            });
        }
        Ok(Self {
            allocation,
            root: absence.root(),
        })
    }

    /// Borrows the matched original source-phase and object association.
    pub(in crate::edit::commit) fn source(&self) -> &ReconstructionSource {
        self.allocation.source()
    }

    /// Borrows the allocation session shared by the original and prospective evidence.
    pub(in crate::edit::commit) fn session(&self) -> &AllocationSession {
        self.allocation.session()
    }

    /// Returns the prospective root under which the parent was proved unreferenced.
    pub(in crate::edit::commit) fn root(&self) -> PhysicalAddress {
        self.root
    }

    /// Returns the complete qualified original allocation range.
    pub(in crate::edit::commit) fn range(&self) -> FileRange {
        self.allocation.range()
    }

    /// Returns the original allocation's free-space class.
    pub(in crate::edit::commit) fn class(&self) -> FreeClass {
        self.allocation.class()
    }

    /// Consumes the evidence and returns the qualified allocation and prospective root.
    ///
    /// The caller handles allocation release and protection changes separately.
    pub(in crate::edit::commit) fn into_parts(
        self,
    ) -> (QualifiedParentAllocation, PhysicalAddress) {
        let Self { allocation, root } = self;
        (allocation, root)
    }
}

/// An incompatible association between original allocation and prospective absence.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum ParentRetirementMismatch {
    /// The qualified allocation and prospective absence refer to different collection addresses.
    Address {
        /// The original allocation's physical start.
        allocation: PhysicalAddress,
        /// The collection address inspected for prospective absence.
        reference: PhysicalAddress,
    },
    /// The allocation and absence belong to different allocation sessions.
    Session,
    /// The original source-phase or object associations differ.
    Source {
        /// The source association of the qualified original allocation.
        allocation: ReconstructionSource,
        /// The original source association retained with prospective absence.
        reference: ReconstructionSource,
    },
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

/// An unmet prerequisite for planning parent compaction or whole-parent retirement.
#[derive(Debug)]
pub(in crate::edit::commit) enum CompactionUnavailable {
    /// Complete original collection framing is unavailable.
    Framing,
    /// Whole-parent allocation qualification is unavailable.
    Parent(ParentUnavailable),
    /// Compaction planning failed.
    Planning(FormatError),
    /// Complete prospective parent absence is unavailable.
    Prospective,
    /// A protected range matching the captured collection range is unavailable.
    Protection,
    /// Original allocation and prospective absence have incompatible associations.
    Retirement(ParentRetirementMismatch),
}

#[cfg(test)]
mod tests;
