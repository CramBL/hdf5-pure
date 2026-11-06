//! Original allocations excluded from placement, pending retirement, and end truncation.
//!
//! Bounds checks preserve recorded owner and parent ranges. Source freshness, mutation
//! permission, and publication require separate evidence.

use std::collections::{BTreeMap, BTreeSet};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{Extent, FreeClass, RetainedKind};

use super::dispositions::{AppliedDispositions, Disposition, ReconstructionDispositions};
use super::members::MemberDecisions;
use super::prospective::ProspectiveCollections;
use super::{AppliedEditInventory, OriginalEdits, PreparedEditInventory};
use crate::edit::WriteEngine;
use crate::edit::analysis::heap::qualification::{
    OriginalCollections, ParentStorage, ParentUnavailable,
};
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::physical::claims::{ClaimOrigin, ClaimProblem};
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::edit::commit::heap;
use crate::edit::commit::heap::HeapMutationFailure;
use crate::edit::commit::plan::PreparedTree;
use crate::edit::commit::publish::{AppliedTree, PublicationError, PublishedTree};
use crate::edit::dataset::MovingWrite;
use crate::edit::retirement::PendingRootFrees;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;

/// Owned original allocations and parent observations associated with one source phase.
///
/// Records owner storage outside eligible deletions and superseded reclamation, and full
/// candidate-parent framing. Independent ranges remain available when qualification fails.
pub(in crate::edit) struct OriginalStorage {
    source: Option<ReconstructionSource>,
    allocations: Vec<StorageAllocation>,
    parents: Result<BTreeMap<PhysicalAddress, ParentStorage>, Error>,
    unavailable: Option<StorageUnavailable>,
}

impl OriginalStorage {
    /// Collects original storage exclusions from the prepared observations and reclamation input.
    pub(super) fn collect(
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        tree: &PreparedTree,
        original: &OriginalEdits,
        dispositions: &ReconstructionDispositions,
        reclamation: &Result<Vec<(FileRange, FreeClass)>, Error>,
    ) -> Self {
        let source = dispositions.source().as_ref().ok().cloned();
        let mut result = Self {
            source,
            allocations: Vec::new(),
            parents: Ok(BTreeMap::new()),
            unavailable: None,
        };
        let Some(source) = &result.source else {
            result.unavailable = Some(StorageUnavailable::Source);
            return result;
        };
        result.parents = phase.collection_storage(source, &tree.collection_candidates);
        let claims = match phase.physical_claims(source) {
            Ok(claims) => claims,
            Err(error) => {
                result.unavailable = Some(StorageUnavailable::Claims(error));
                return result;
            }
        };
        if !original
            .owners()
            .is_some_and(|owners| owners.source() == source && owners.matches(engine, claims))
        {
            result.unavailable = Some(StorageUnavailable::Owners);
        }
        if dispositions
            .predecessors()
            .values()
            .any(|old| old.disposition != Disposition::Supersede)
        {
            result
                .unavailable
                .get_or_insert(StorageUnavailable::Dispositions);
        }
        if reclamation.is_err() {
            result
                .unavailable
                .get_or_insert(StorageUnavailable::Reclamation);
        }
        let superseded: BTreeSet<_> = dispositions
            .predecessors()
            .values()
            .filter(|old| old.disposition == Disposition::Supersede)
            .map(|old| old.source.object())
            .collect();
        let eligible: BTreeSet<_> = original
            .facts()
            .into_iter()
            .flat_map(|facts| facts.eligible_owners())
            .copied()
            .collect();
        let tails: BTreeSet<_> = tree
            .groups
            .values()
            .flat_map(|node| &node.prepared.writes)
            .filter_map(|write| match &write.write {
                MovingWrite::AppendedChunks {
                    old_tail_allocation: Some(allocation),
                    ..
                } => Some((allocation.range().start(), allocation.range().end())),
                _ => None,
            })
            .collect();
        for claim in claims.claims() {
            let ClaimOrigin::Object { object, kind } = claim.origin() else {
                continue;
            };
            let eligible = eligible.contains(&object);
            let superseded = superseded.contains(&object);
            let retired = reclamation.as_ref().is_ok_and(|retired| {
                retired
                    .binary_search_by_key(&claim.range().start(), |(range, _)| range.start())
                    .ok()
                    .is_some_and(|index| retired[index].0 == claim.range())
            }) || tails.contains(&(claim.range().start(), claim.range().end()));
            if !eligible && !(superseded && retired) {
                result.allocations.push(StorageAllocation {
                    range: claim.range(),
                    origin: StorageOrigin::Owner { object, kind },
                });
            }
        }
        if let Some(object) = claims.problems().iter().find_map(|problem| {
            if let ClaimProblem::ObjectStorage(object) = problem {
                Some(*object)
            } else {
                None
            }
        }) {
            result
                .unavailable
                .get_or_insert(StorageUnavailable::Object(object));
        }
        match &result.parents {
            Ok(parents) => {
                for (&address, parent) in parents {
                    if let Err(reason) = parent.allocation {
                        result
                            .unavailable
                            .get_or_insert(StorageUnavailable::Parent { address, reason });
                    }
                    if let Ok(range) = parent.framing {
                        result.allocations.push(StorageAllocation {
                            range,
                            origin: StorageOrigin::Parent {
                                address,
                                class: parent.original.and_then(|(_, class)| class),
                            },
                        });
                    }
                }
            }
            Err(_) => {
                result
                    .unavailable
                    .get_or_insert(StorageUnavailable::Parents);
            }
        }
        result
            .allocations
            .sort_unstable_by_key(|allocation| allocation.range.start());
        result
    }

    #[cfg(test)]
    pub(in crate::edit::commit) fn from_test_ranges(ranges: &[FileRange]) -> Self {
        Self {
            source: None,
            allocations: ranges
                .iter()
                .map(|&range| StorageAllocation {
                    range,
                    origin: StorageOrigin::Owner {
                        object: ObjectId::new(range.start()),
                        kind: AllocationKind::Data,
                    },
                })
                .collect(),
            parents: Ok(BTreeMap::new()),
            unavailable: None,
        }
    }

    /// Borrows protected ranges in ascending order of their physical start addresses.
    pub(in crate::edit::commit) fn allocations(&self) -> &[StorageAllocation] {
        &self.allocations
    }

    /// Borrows the first qualification failure, independently of recorded ranges.
    pub(in crate::edit::commit) fn unavailable(&self) -> Option<&StorageUnavailable> {
        self.unavailable.as_ref()
    }

    /// Borrows independent parent observations keyed by physical collection address.
    ///
    /// # Errors
    ///
    /// Returns the recorded error of [`AnalysisPhase::collection_storage`].
    pub(in crate::edit::commit) fn parents(
        &self,
    ) -> Result<&BTreeMap<PhysicalAddress, ParentStorage>, &Error> {
        self.parents.as_ref()
    }

    /// Checks reusable sections, retained extents, and unused aggregate capacity.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::Overlap`] if an allocator range overlaps protected storage.
    /// Returns [`StorageConflict::Source`] if an address or range cannot be resolved.
    pub(in crate::edit::commit) fn exclude_space(
        &self,
        engine: &WriteEngine,
    ) -> Result<(), StorageConflict> {
        let resolve = |extent: Extent| {
            FileRange::new(
                engine.addresses().resolve(extent.start())?,
                extent.len().get(),
            )
            .map_err(Error::from)
        };
        for extent in engine.space.reusable_sections() {
            let range = resolve(extent).map_err(StorageConflict::Source)?;
            self.exclude(range, SpaceClaim::Reusable)?;
        }
        for entry in engine.space.retained_extents() {
            let range = resolve(entry.extent()).map_err(StorageConflict::Source)?;
            self.exclude(range, SpaceClaim::Retained(entry.kind()))?;
        }
        for extent in engine
            .aggregation
            .into_iter()
            .flat_map(|space| space.unused())
        {
            let range = resolve(extent).map_err(StorageConflict::Source)?;
            self.exclude(range, SpaceClaim::Aggregate)?;
        }
        Ok(())
    }

    /// Checks a physical space claim against the recorded original allocations.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::Overlap`] with `claim` if `range` overlaps protected storage.
    pub(in crate::edit::commit) fn exclude(
        &self,
        range: FileRange,
        claim: SpaceClaim,
    ) -> Result<(), StorageConflict> {
        let index = self
            .allocations
            .partition_point(|allocation| allocation.range.end() <= range.start());
        if let Some(&allocation) = self
            .allocations
            .get(index)
            .filter(|allocation| allocation.range.overlaps(range))
        {
            return Err(StorageConflict::Overlap {
                allocation,
                range,
                claim,
            });
        }
        Ok(())
    }

    /// Checks whether a proposed physical end preserves every recorded original allocation.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::End`] if `end` is below the greatest protected range end.
    pub(in crate::edit::commit) fn check_end(
        &self,
        end: PhysicalAddress,
    ) -> Result<(), StorageConflict> {
        let required = self.maximum_end();
        if end < required {
            return Err(StorageConflict::End {
                proposed: end,
                required,
            });
        }
        Ok(())
    }

    fn maximum_end(&self) -> PhysicalAddress {
        self.allocations
            .iter()
            .map(|allocation| allocation.range.end())
            .max()
            .unwrap_or(PhysicalAddress::new(0))
    }
}

impl PreparedEditInventory {
    /// Borrows original storage evidence associated with these prepared inputs.
    pub(in crate::edit::commit) fn storage(&self) -> &OriginalStorage {
        &self.storage
    }

    /// Checks original storage exclusions before placement and records pending retirement.
    ///
    /// Checks the source-phase, root, and allocation-session associations before application.
    /// On success, retains the allocator and pending-retirement exclusion outcome in
    /// [`PendingEditInventory::exclusion`]. The caller handles rollback after application failure.
    ///
    /// # Errors
    ///
    /// Returns [`StorageApplyFailure::BeforePlacement`] with the owned prepared inputs for
    /// unavailable evidence, incompatible associations, or an allocator conflict. Returns
    /// [`StorageApplyFailure::Application`] for an error from [`Self::apply`].
    pub(in crate::edit::commit) fn apply_excluding(
        self,
        engine: &mut WriteEngine,
    ) -> Result<PendingEditInventory<'_>, StorageApplyFailure> {
        let check = if self.storage.unavailable.is_some() {
            Err(StorageApplyError::Unavailable)
        } else if !self.session.same(&engine.context.allocations.session()) {
            Err(StorageApplyError::Session)
        } else if self.storage.source.as_ref() != self.dispositions.source().as_ref().ok() {
            Err(StorageApplyError::Phase)
        } else if self
            .storage
            .source
            .as_ref()
            .map(ReconstructionSource::object)
            != ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).ok()
        {
            Err(StorageApplyError::Root)
        } else {
            engine
                .context
                .allocations
                .before_reference()
                .map_err(StorageConflict::Source)
                .and_then(|()| self.storage.exclude_space(engine))
                .map_err(StorageApplyError::Conflict)
        };
        if let Err(error) = check {
            return Err(StorageApplyFailure::BeforePlacement {
                prepared: Box::new(self),
                error,
            });
        }
        let applied = self
            .apply(engine)
            .map_err(StorageApplyFailure::Application)?;
        Ok(PendingEditInventory::new(engine, applied))
    }
}

/// Owns placed-tree observations and pending frees while borrowing the application engine.
///
/// Pending frees include deferred chunk retirement at the engine's current length. The editor
/// checks allocator and retirement ranges against original storage and retains the outcome.
/// With original candidates, the editor decodes heap references in a new phase after application
/// and retains root-applicability failures independently of reference interpretation.
/// [`Self::publish_compacting_members`] derives member removal permission from these observations.
/// Other completion callbacks require independently established mutation permission.
pub(in crate::edit::commit) struct PendingEditInventory<'a> {
    engine: &'a mut WriteEngine,
    original: OriginalEdits,
    dispositions: AppliedDispositions,
    storage: OriginalStorage,
    collections: OriginalCollections,
    prospective: ProspectiveCollections,
    pending: PendingRootFrees,
    relocations: BTreeMap<ObjectId, ObjectId>,
    exclusion: Result<(), StorageConflict>,
}

impl<'a> PendingEditInventory<'a> {
    fn new(engine: &'a mut WriteEngine, applied: AppliedEditInventory) -> Self {
        let AppliedEditInventory {
            tree,
            original,
            dispositions,
            storage,
            collections,
        } = applied;
        let AppliedTree {
            retirement,
            freed,
            relocations,
        } = tree;
        let pending = retirement.into_pending_frees(freed, engine.image.len());
        let exclusion = storage.exclude_space(engine).and_then(|()| {
            for &(range, class) in pending.batch.entries() {
                storage.exclude(range, SpaceClaim::Retirement(class))?;
            }
            Ok(())
        });
        let prospective = ProspectiveCollections::collect(
            engine,
            &original,
            &dispositions,
            &collections,
            pending.root,
        );
        Self {
            engine,
            original,
            dispositions,
            storage,
            collections,
            prospective,
            pending,
            relocations,
            exclusion,
        }
    }

    /// Publishes the pending tree and borrows its retained observations for completion.
    ///
    /// Checks placement reconciliation and current allocator claims before consuming the pending
    /// batch. Calls `complete` after publication, reference updates, root reclamation, truncation,
    /// and final storage checks succeed. The callback receives the original observations, applied
    /// dispositions, storage exclusions, and original collection qualification and framing
    /// associated with these inputs, together with prospective observations collected after
    /// application.
    ///
    /// # Errors
    ///
    /// Returns [`StoragePublicationFailure::BeforePublication`] with these pending inputs if
    /// placement reconciliation or exclusion checks fail. Returns
    /// [`StoragePublicationFailure::Publication`] with the retained observations and error if
    /// publication or completion fails. The caller handles rollback according to publication state.
    pub(in crate::edit::commit) fn publish_with_completion<T>(
        self,
        complete: impl FnOnce(
            PublishedTree<'_>,
            &OriginalEdits,
            &AppliedDispositions,
            &OriginalStorage,
            &OriginalCollections,
            &ProspectiveCollections,
        ) -> Result<T, Error>,
    ) -> Result<T, Box<StoragePublicationFailure<'a>>> {
        self.publish_with_observations(complete)
            .map(|(value, _, _)| value)
    }

    /// Retains owned collection observations after successful publication and completion.
    ///
    /// Uses the publication and error contract of [`Self::publish_with_completion`].
    ///
    /// # Errors
    ///
    /// Returns [`StoragePublicationFailure::BeforePublication`] if placement reconciliation or
    /// exclusion checks fail, or [`StoragePublicationFailure::Publication`] if publication or
    /// completion fails.
    fn publish_with_observations<T>(
        mut self,
        complete: impl FnOnce(
            PublishedTree<'_>,
            &OriginalEdits,
            &AppliedDispositions,
            &OriginalStorage,
            &OriginalCollections,
            &ProspectiveCollections,
        ) -> Result<T, Error>,
    ) -> Result<(T, OriginalCollections, ProspectiveCollections), Box<StoragePublicationFailure<'a>>>
    {
        let reason = if self.dispositions.replacements().is_err() {
            Some(StoragePublicationRejection::Placement)
        } else {
            if self.exclusion.is_ok() {
                self.exclusion = self.storage.exclude_space(self.engine);
            }
            self.exclusion
                .is_err()
                .then_some(StoragePublicationRejection::Exclusion)
        };
        if let Some(reason) = reason {
            return Err(Box::new(StoragePublicationFailure::BeforePublication {
                pending: self,
                reason,
            }));
        }
        let Self {
            engine,
            original,
            dispositions,
            storage,
            collections,
            prospective,
            pending,
            relocations,
            exclusion: _,
        } = self;
        let result = engine.publish_pending_tree_excluding(
            pending,
            relocations,
            Some(&storage),
            |published| {
                complete(
                    published,
                    &original,
                    &dispositions,
                    &storage,
                    &collections,
                    &prospective,
                )
                .map_err(PublicationError::from)
            },
        );
        match result {
            Ok(value) => Ok((value, collections, prospective)),
            Err(error) => Err(Box::new(StoragePublicationFailure::Publication {
                original,
                dispositions,
                storage,
                collections,
                prospective,
                error,
            })),
        }
    }

    /// Derives qualified removals from the owned original and fresh prospective observations.
    pub(in crate::edit::commit) fn member_decisions(&self) -> MemberDecisions {
        MemberDecisions::collect(&self.collections, &self.prospective, &self.storage)
    }

    /// Publishes the protected tree before compacting qualified global heap members.
    ///
    /// Derives plans from the owned original and prospective observations and checks publication
    /// and prospective root associations against the pending root. The writer preserves each
    /// parent allocation and applies the plans through [`heap::after_publication`].
    ///
    /// Returns the final adopted root and EOA with the observations and decisions. A heap
    /// mutation failure is recorded in [`MemberCompletion::mutation`] after publication.
    ///
    /// # Errors
    ///
    /// Returns [`StoragePublicationFailure::BeforePublication`] if placement reconciliation or
    /// storage exclusion checks fail. Returns [`StoragePublicationFailure::Publication`] if
    /// publication fails or a root association is incompatible.
    pub(in crate::edit::commit) fn publish_compacting_members(
        self,
    ) -> Result<MemberCompletion, Box<StoragePublicationFailure<'a>>> {
        let decisions = self.member_decisions();
        let requested_root = self.pending.root;
        let ((root, end, mutation), collections, prospective) =
            self.publish_with_observations(|published, _, _, _, _, prospective| {
                if published.requested_root() != requested_root
                    || prospective
                        .root()
                        .is_some_and(|root| root != requested_root)
                {
                    return Err(FormatError::Internal(
                        "heap decisions belong to another published tree".into(),
                    )
                    .into());
                }
                let root = published.root();
                let end = published.end();
                Ok((
                    root,
                    end,
                    heap::after_publication(published, decisions.plans()),
                ))
            })?;
        Ok(MemberCompletion {
            collections,
            prospective,
            decisions,
            root,
            end,
            mutation,
        })
    }

    /// Borrows heap references and root-applicability outcomes collected after application.
    pub(in crate::edit::commit) fn prospective(&self) -> &ProspectiveCollections {
        &self.prospective
    }

    /// Borrows original candidates, qualification, and framing retained until completion.
    pub(in crate::edit::commit) fn collections(&self) -> &OriginalCollections {
        &self.collections
    }

    /// Returns the recorded allocator and pending-retirement exclusion outcome.
    ///
    /// # Errors
    ///
    /// Returns the first recorded [`StorageConflict`].
    pub(in crate::edit::commit) fn exclusion(&self) -> Result<(), &StorageConflict> {
        self.exclusion.as_ref().copied()
    }

    /// Returns the greatest protected range end, or physical address zero for an empty inventory.
    pub(in crate::edit::commit) fn maximum_end(&self) -> PhysicalAddress {
        self.storage.maximum_end()
    }

    /// Checks whether a proposed end preserves every recorded original allocation.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::End`] if `end` is below [`Self::maximum_end`].
    pub(in crate::edit::commit) fn check_end(
        &self,
        end: PhysicalAddress,
    ) -> Result<(), StorageConflict> {
        self.storage.check_end(end)
    }
}

/// Owned observations and mutation outcome after successful protected tree publication.
///
/// The final root and EOA are retained even if heap compaction fails.
pub(in crate::edit::commit) struct MemberCompletion {
    /// Original qualification and collection framing used for member decisions.
    pub(in crate::edit::commit) collections: OriginalCollections,
    /// Fresh reference observations used to determine removal eligibility.
    pub(in crate::edit::commit) prospective: ProspectiveCollections,
    /// Member, parent, and planning outcomes used for this completion.
    pub(in crate::edit::commit) decisions: MemberDecisions,
    /// The final adopted root after tree publication.
    pub(in crate::edit::commit) root: PhysicalAddress,
    /// The final published EOA.
    pub(in crate::edit::commit) end: PhysicalAddress,
    /// The compaction result, including the completed prefix and allocation state on failure.
    pub(in crate::edit::commit) mutation: Result<(), HeapMutationFailure>,
}

/// The inputs or observations retained after a failed constrained tree publication.
pub(in crate::edit::commit) enum StoragePublicationFailure<'a> {
    /// Publication was rejected with the pending inventory and engine loan intact.
    BeforePublication {
        pending: PendingEditInventory<'a>,
        reason: StoragePublicationRejection,
    },
    /// The publisher consumed the pending inventory before publication or completion failed.
    Publication {
        original: OriginalEdits,
        dispositions: AppliedDispositions,
        storage: OriginalStorage,
        collections: OriginalCollections,
        prospective: ProspectiveCollections,
        error: PublicationError,
    },
}

/// A failed prerequisite for handing a pending inventory to the publisher.
#[derive(Debug)]
pub(in crate::edit::commit) enum StoragePublicationRejection {
    /// Allocator or pending-retirement ranges overlap protected storage or cannot be resolved.
    Exclusion,
    /// Applied replacement placements could not be reconciled with the prepared dispositions.
    Placement,
}

/// A protected physical range with its original owner or parent collection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) struct StorageAllocation {
    pub(in crate::edit::commit) range: FileRange,
    pub(in crate::edit::commit) origin: StorageOrigin,
}

/// The owner or global heap parent associated with a protected range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum StorageOrigin {
    /// An original object allocation identified by the physical-claims inventory.
    Owner {
        object: ObjectId,
        kind: AllocationKind,
    },
    /// Full collection framing with the candidate allocation class, if recorded.
    Parent {
        address: PhysicalAddress,
        class: Option<FreeClass>,
    },
}

/// An unmet prerequisite for complete original storage exclusions.
#[derive(Debug)]
pub(in crate::edit::commit) enum StorageUnavailable {
    /// Physical-claims collection failed.
    Claims(Error),
    /// A predecessor is excluded from complete replacement.
    Dispositions,
    /// The physical storage of an original object is incomplete.
    Object(ObjectId),
    /// Original owner qualification is absent or has incompatible associations.
    Owners,
    /// A parent has an unavailable framing, candidate-range, or allocation-class prerequisite.
    Parent {
        address: PhysicalAddress,
        reason: ParentUnavailable,
    },
    /// Parent storage collection failed.
    Parents,
    /// Original reclamation collection failed.
    Reclamation,
    /// The original source association is unavailable.
    Source,
}

/// A protected allocation overlaps a space claim or extends past a proposed end.
#[derive(Debug)]
pub(in crate::edit) enum StorageConflict {
    /// The proposed end is below the greatest protected range end.
    End {
        proposed: PhysicalAddress,
        required: PhysicalAddress,
    },
    /// A space claim overlaps protected storage.
    Overlap {
        allocation: StorageAllocation,
        range: FileRange,
        claim: SpaceClaim,
    },
    /// A space range could not be resolved or allocation confirmation failed.
    Source(Error),
}

/// The allocator or retirement role of a range checked against original storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum SpaceClaim {
    /// Unused aggregate capacity.
    Aggregate,
    /// A complete metadata reservation or publication padding range.
    Metadata,
    /// A previous persistent manager allocation pending replacement.
    OldManager,
    /// A complete reserved tail allocation proposed for recycling.
    Recycle,
    /// A retained allocator extent with its retention kind.
    Retained(RetainedKind),
    /// A pending retirement range with its allocation class.
    Retirement(FreeClass),
    /// A reusable free-space section.
    Reusable,
    /// A complete reserved root-header allocation proposed as the copy target.
    RootCopy,
}

/// The error and any prepared inputs retained after failed storage application.
pub(in crate::edit::commit) enum StorageApplyFailure {
    /// The editor consumed the inputs before returning an application error.
    Application(Error),
    /// Exclusion or association checks failed before the editor consumed the inputs.
    BeforePlacement {
        prepared: Box<PreparedEditInventory>,
        error: StorageApplyError,
    },
}

/// A failed prerequisite for placement with original storage exclusions.
#[derive(Debug)]
pub(in crate::edit::commit) enum StorageApplyError {
    /// Allocation confirmation or original storage exclusion failed.
    Conflict(StorageConflict),
    /// The storage and predecessor observations have different source phases.
    Phase,
    /// The source object differs from the engine's current root.
    Root,
    /// The preparation belongs to another allocation session.
    Session,
    /// Complete original storage evidence is unavailable.
    Unavailable,
}

#[cfg(test)]
mod tests;
