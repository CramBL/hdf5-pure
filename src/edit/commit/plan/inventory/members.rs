//! Qualified global heap member decisions and exclusive parent dispositions.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCompaction;
use hdf5_pure_space::__private::FreeClass;

use super::dispositions::AppliedDispositions;
use super::prospective::ParentAbsence;
use super::prospective::ProspectiveCollections;
use super::prospective::ReferencePresence;
use super::session::SessionParentRetirement;
use super::storage::{OriginalStorage, StorageOrigin};
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::qualification::QualifiedParentAllocation;
use crate::edit::analysis::heap::qualification::{
    MemberUnavailable, OriginalCollections, OriginalQualification, ParentUnavailable,
};

/// Member and parent removal decisions with one disposition per physical parent.
///
/// Compaction and whole-parent retirement are exclusive dispositions. Publication and parent
/// allocation release require separate permission.
#[derive(Default)]
pub(in crate::edit::commit) struct MemberDecisions {
    dispositions: BTreeMap<PhysicalAddress, ParentDisposition>,
}

impl MemberDecisions {
    /// Derives removal decisions from original qualification and fresh prospective references.
    ///
    /// The planner groups origins by physical member before looking up prospective presence once
    /// per identity. It retains one disposition per original parent address. It preserves a
    /// qualified member or parent if a prospective reference was observed, even when other
    /// observations are incomplete. It requires original qualification, complete interpretation,
    /// and root applicability for removal.
    ///
    /// The planner groups removals by parent and preserves unobserved members, serialized reference
    /// counts, and collection capacity. It requires captured framing and an exactly matching
    /// protected parent range before constructing a plan. For a wholly unreferenced qualified
    /// parent, it binds complete allocation and absence evidence as a retirement disposition.
    /// For other parents, it selects compaction or retention, with the original allocation retained
    /// on planning failure.
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
            let decision = MemberDecision::from(prospective.member_presence(id));
            let mut remove = false;
            for origin in origins {
                let outcome = qualification
                    .ok_or(MemberUnavailable::Qualification)
                    .and_then(|qualification| qualification.member(origin, id))
                    .map(|()| decision);
                remove |= outcome == Ok(MemberDecision::Remove);
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
            let disposition = if parent == Ok(MemberDecision::Remove) {
                qualification
                    .ok_or(())
                    .and_then(|qualification| {
                        qualification
                            .whole_parent_allocation(address)
                            .map_err(|_error| ())
                    })
                    .and_then(|allocation| {
                        if protected.get(&address) != Some(&allocation.range()) {
                            return Err(());
                        }
                        let absence = prospective.parent_absence(address).ok_or(())?;
                        ParentRetirement::new(allocation, absence)
                            .map(ParentDisposition::Retire)
                            .map_err(|_error| ())
                    })
            } else if let Some(indices) = removed.get(&address) {
                original
                    .contents()
                    .get(&address)
                    .and_then(|contents| contents.as_ref().ok())
                    .ok_or(())
                    .and_then(|contents| {
                        if protected.get(&address) != Some(&contents.range()) {
                            return Err(());
                        }
                        contents
                            .framing()
                            .compact(|member| indices.contains(&member.index()))
                            .map(ParentDisposition::Compact)
                            .map_err(|_error| ())
                    })
            } else {
                result
                    .dispositions
                    .insert(address, ParentDisposition::Retain);
                continue;
            };
            match disposition {
                Ok(disposition) => {
                    result.dispositions.insert(address, disposition);
                }
                Err(()) => {
                    result
                        .dispositions
                        .insert(address, ParentDisposition::Retain);
                }
            }
        }
        result
    }

    /// Selects whole-parent retirement for reconciled exclusive session overwrites.
    ///
    /// Replaces the disposition only after matching session evidence with the applied owner
    /// replacement and prospective observations.
    pub(super) fn include_session_parents(
        &mut self,
        dispositions: &AppliedDispositions,
        prospective: &ProspectiveCollections,
    ) {
        for (&address, parent) in &dispositions.original().session_parents {
            if let Some(retirement) = parent.retirement(dispositions, prospective) {
                self.dispositions.insert(
                    address,
                    ParentDisposition::Retire(ParentRetirement::from_session(retirement)),
                );
            }
        }
    }

    /// Retains compaction dispositions in the existing map and consumes all other decisions.
    ///
    /// Preserves plan ownership and physical parent-address order. The caller installs checked
    /// retirements and updates their storage exclusions before consuming the retirement evidence.
    pub(super) fn into_compactions(self) -> MemberCompactions {
        let Self { mut dispositions } = self;
        dispositions.retain(|_, disposition| matches!(disposition, ParentDisposition::Compact(_)));
        MemberCompactions {
            decisions: Self { dispositions },
        }
    }
}

impl MemberDecisions {
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
}

/// Compaction plans retained after checked parent-retirement preparation.
///
/// The internal map contains only [`ParentDisposition::Compact`] entries. Publication and member
/// mutation require separate checks.
pub(in crate::edit::commit) struct MemberCompactions {
    decisions: MemberDecisions,
}

impl MemberCompactions {
    /// Borrows the original compaction plans in physical parent-address order.
    pub(in crate::edit::commit) fn plans(&self) -> impl Iterator<Item = &GlobalHeapCompaction> {
        self.decisions.plans()
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

/// Removal eligibility from qualified references or exclusive session ownership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum MemberDecision {
    /// A prospective reference to the member or parent was observed.
    Live,
    /// Complete prospective interpretation or root applicability is unavailable.
    Prospective,
    /// Removal is qualified by complete absence or checked session ownership.
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
