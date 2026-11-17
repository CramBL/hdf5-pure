//! Original allocations excluded from placement, pending retirement, and end truncation.
//!
//! Bounds checks preserve recorded owner and parent ranges. Source freshness, mutation
//! permission, and publication require separate evidence.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{Extent, FreeClass, RetainedKind};

use self::capture::{ClaimReleases, ResidualRelease, StorageIntegrityError};
use super::dispositions::{AppliedDispositions, Disposition, ReconstructionDispositions};
use super::members::MemberDecisions;
use super::prospective::ProspectiveCollections;
use super::{AppliedEditInventory, OriginalEdits, PreparedEditInventory};
use crate::edit::WriteEngine;
use crate::edit::analysis::heap::qualification::{OriginalCollections, ParentStorage};
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::analysis::{
    AnalysisPhase, CountProtectionError, ProtectedCounts, ReconstructionSource,
};
use crate::edit::commit::heap;
use crate::edit::commit::heap::HeapMutationFailure;
use crate::edit::commit::plan::PreparedTree;
use crate::edit::commit::publish::{AppliedTree, PublicationError, PublishedTree};
use crate::edit::dataset::MovingWrite;
use crate::edit::retirement::PendingRootFrees;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;

mod capture;
mod outcomes;
mod parents;
use crate::edit::allocation::AllocationSession;

/// Original allocations with optional source-phase and parent observations.
///
/// Records original owner allocations and any candidate-parent framing. Independent ranges
/// remain available when qualification fails.
pub(in crate::edit) struct OriginalStorage {
    source: Option<ReconstructionSource>,
    allocations: Vec<StorageAllocation>,
    parents: Result<BTreeMap<PhysicalAddress, ParentStorage>, Error>,
    integrity: Result<(), StorageIntegrityError>,
    unavailable: Option<StorageUnavailable>,
}

impl OriginalStorage {
    /// Collects complete header allocations containing updated counts for publication exclusions.
    ///
    /// Preserves the allocation order of [`ProtectedCounts`]. Records owner identities and header
    /// ranges, with source-phase and parent observations absent. The caller checks count-source
    /// applicability separately.
    pub(in crate::edit) fn from_counts(counts: &ProtectedCounts) -> Self {
        Self {
            source: None,
            allocations: counts
                .updates()
                .iter()
                .map(|update| StorageAllocation {
                    range: update.allocation().range(),
                    origin: StorageOrigin::Owner {
                        object: update.object(),
                        kind: AllocationKind::Header,
                    },
                })
                .collect(),
            parents: Ok(BTreeMap::new()),
            integrity: Ok(()),
            unavailable: None,
        }
    }

    /// Collects original storage exclusions from the prepared observations and reclamation input.
    ///
    /// Superseded allocations remain protected unless associated evidence rules out references
    /// from residual storage.
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
            integrity: Ok(()),
            unavailable: None,
        };
        let Some(source) = dispositions.source().as_ref().ok() else {
            result.unavailable = Some(StorageUnavailable::Source);
            result.integrity = Err(StorageIntegrityError::Unavailable);
            return result;
        };
        result.parents = phase.collection_storage(source, &tree.collection_candidates);
        let groups = if original.owners().is_none() {
            match phase.residual_group_domain(source) {
                Ok(groups) => Some(groups),
                Err(error) => {
                    result.unavailable = Some(StorageUnavailable::Coverage(error));
                    result.integrity = Err(StorageIntegrityError::Unavailable);
                    result.capture_parents();
                    return result;
                }
            }
        } else {
            None
        };
        let claims = match phase.physical_claims(source) {
            Ok(claims) => claims,
            Err(error) => {
                result.unavailable = Some(StorageUnavailable::Claims(error));
                result.integrity = Err(StorageIntegrityError::Unavailable);
                result.capture_parents();
                return result;
            }
        };
        let qualified_owners = original
            .owners()
            .is_some_and(|owners| owners.source() == source && owners.matches(engine, claims));
        if !qualified_owners {
            result.unavailable = Some(StorageUnavailable::Owners);
        }
        let residual = match groups.as_deref().and_then(Option::as_ref) {
            Some(groups) => ResidualRelease::Complete(groups),
            None if qualified_owners
                || matches!(
                    original,
                    OriginalEdits::Unavailable {
                        domain: Some(Ok(OwnerDomain::Available(domain))),
                        ..
                    } if domain.matches(engine, claims)
                ) =>
            {
                ResidualRelease::Absent
            }
            None => ResidualRelease::Unavailable,
        };
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
        let releases =
            ClaimReleases::new(&tree.groups, original, dispositions, reclamation, residual);
        result.capture_claims(claims, &releases);
        result.capture_parents();
        result.capture_integrity(engine, phase, source);

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
            integrity: Ok(()),
            unavailable: None,
        }
    }

    /// Withholds retirement of protected original owners and unobserved storage.
    ///
    /// Consumes deferred tail candidates after their geometry has been used to derive
    /// reclamation spans and original storage. Leaves parent requests, partial overlaps, and
    /// reclamation errors for the separate exclusion and error checks.
    pub(super) fn withhold_protected_reclamation(
        &self,
        tree: &mut PreparedTree,
        reclamation: &mut Result<Vec<(FileRange, FreeClass)>, Error>,
    ) {
        if let Ok(retired) = reclamation {
            retired.retain(|&(range, _)| !self.protects_from_retirement(range));
        }
        for write in tree
            .groups
            .values_mut()
            .flat_map(|node| &mut node.prepared.writes)
        {
            if let MovingWrite::AppendedChunks {
                old_tail_allocation,
                ..
            } = &mut write.write
            {
                *old_tail_allocation = old_tail_allocation
                    .take()
                    .filter(|allocation| !self.protects_from_retirement(allocation.range()));
            }
        }
    }

    fn protects_from_retirement(&self, range: FileRange) -> bool {
        let index = self
            .allocations
            .partition_point(|allocation| allocation.range.end() <= range.start());
        self.allocations
            .get(index)
            .is_some_and(|allocation| match allocation.origin {
                StorageOrigin::Owner { .. } => allocation.range == range,
                StorageOrigin::Parent { .. } => false,
                StorageOrigin::Unobserved => allocation.range.contains(range),
            })
    }

    /// Adds complete original count-header allocations in physical address order.
    ///
    /// Checks the batch's engine and source associations with [`ProtectedCounts::check_association`]
    /// before changing ranges. Replaces intersecting [`StorageOrigin::Unobserved`] intervals with
    /// checked header allocations, preserving the surrounding unobserved portions. Retains an
    /// identical allocation with the same owner and header kind once. Earlier insertions remain if
    /// a later range conflicts.
    ///
    /// # Errors
    ///
    /// Returns [`StorageConflict::CountAssociation`] if the batch's association check fails.
    /// Returns [`StorageConflict::Overlap`] if a count-header allocation overlaps recorded owner
    /// or parent storage, except an identical allocation with the same owner and header kind.
    /// Returns [`StorageConflict::Source`] if constructing a surrounding unobserved range fails.
    pub(in crate::edit) fn include_counts(
        &mut self,
        engine: &WriteEngine,
        counts: &ProtectedCounts,
    ) -> Result<(), StorageConflict> {
        counts
            .check_association(engine, self.source.as_ref())
            .map_err(StorageConflict::CountAssociation)?;
        for update in counts.updates() {
            let allocation = StorageAllocation {
                range: update.allocation().range(),
                origin: StorageOrigin::Owner {
                    object: update.object(),
                    kind: AllocationKind::Header,
                },
            };
            let start = self
                .allocations
                .partition_point(|existing| existing.range.end() <= allocation.range.start());
            let end = start
                + self.allocations[start..]
                    .partition_point(|existing| existing.range.start() < allocation.range.end());
            if self.allocations.get(start..end) == Some(&[allocation]) {
                continue;
            }
            if let Some(&existing) = self.allocations[start..end]
                .iter()
                .find(|existing| existing.origin != StorageOrigin::Unobserved)
            {
                return Err(StorageConflict::Overlap {
                    allocation: existing,
                    range: allocation.range,
                    claim: SpaceClaim::CountHeader,
                });
            }
            let mut refined = Vec::new();
            for &existing in &self.allocations[start..end] {
                if existing.range.start() < allocation.range.start() {
                    refined.push(StorageAllocation {
                        range: FileRange::new(
                            existing.range.start(),
                            allocation.range.start().get() - existing.range.start().get(),
                        )
                        .map_err(|error| StorageConflict::Source(error.into()))?,
                        origin: StorageOrigin::Unobserved,
                    });
                }
            }
            refined.push(allocation);
            for &existing in &self.allocations[start..end] {
                if existing.range.end() > allocation.range.end() {
                    refined.push(StorageAllocation {
                        range: FileRange::new(
                            allocation.range.end(),
                            existing.range.end().get() - allocation.range.end().get(),
                        )
                        .map_err(|error| StorageConflict::Source(error.into()))?,
                        origin: StorageOrigin::Unobserved,
                    });
                }
            }
            self.allocations.splice(start..end, refined);
        }
        Ok(())
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
    /// Returns source integrity, incompatible association, allocator conflict, or application
    /// errors. Staged inputs have already been consumed.
    pub(in crate::edit::commit) fn apply_excluding(
        mut self,
        engine: &mut WriteEngine,
    ) -> Result<PendingEditInventory<'_>, Error> {
        if let Err(error) = std::mem::replace(&mut self.storage.integrity, Ok(())) {
            return Err(outcomes::integrity_error(self, error));
        }
        if !self.session.same(&engine.context.allocations.session()) {
            return Err(FormatError::Internal(
                "incompatible prepared storage association: Session".into(),
            )
            .into());
        }
        if self.storage.source.as_ref() != self.dispositions.source().as_ref().ok() {
            return Err(FormatError::Internal(
                "incompatible prepared storage association: Phase".into(),
            )
            .into());
        }
        if self
            .storage
            .source
            .as_ref()
            .map(ReconstructionSource::object)
            != ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).ok()
        {
            return Err(FormatError::Internal(
                "incompatible prepared storage association: Root".into(),
            )
            .into());
        }
        engine.context.allocations.before_reference()?;
        self.storage.exclude_space(engine).map_err(Error::from)?;
        let applied = self.apply(engine)?;
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
    session: AllocationSession,
    counts: Option<ProtectedCounts>,
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
            counts,
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
        let session = engine.context.allocations.session();
        Self {
            engine,
            session,
            counts,
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
    /// Returns placement-reconciliation, exclusion, publication, or completion errors. The caller
    /// handles rollback according to publication state.
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
    ) -> Result<T, Error> {
        self.publish_with_observations(complete)
            .map(|(value, _, _)| value)
    }

    /// Retains owned collection observations after successful publication and completion.
    ///
    /// Uses the publication and error contract of [`Self::publish_with_completion`].
    ///
    /// # Errors
    ///
    /// Returns placement-reconciliation, exclusion, publication, or completion errors.
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
    ) -> Result<(T, OriginalCollections, ProspectiveCollections), Error> {
        self.dispositions.replacements().map_err(|_error| {
            Error::EditUnsupported("original predecessor replacement cannot be established")
        })?;
        if self.exclusion.is_ok() {
            self.exclusion = self.storage.exclude_space(self.engine);
        }
        self.exclusion.map_err(Error::from)?;
        let Self {
            engine,
            session: _,
            counts,
            original,
            dispositions,
            mut storage,
            collections,
            prospective,
            pending,
            relocations,
            exclusion: _,
        } = self;
        let completion = |published: PublishedTree<'_>, storage: &OriginalStorage| {
            complete(
                published,
                &original,
                &dispositions,
                storage,
                &collections,
                &prospective,
            )
        };
        let result = match counts {
            Some(counts) => {
                let end = engine
                    .end_of_allocation()
                    .and_then(|end| engine.addresses().resolve(end))
                    .map_err(Error::from);
                match end.and_then(|end| {
                    counts
                        .before_publication(engine, pending)
                        .map_err(Error::from)
                        .and_then(|phase| {
                            phase
                                .at_end(end)
                                .map_err(|failure| Error::from(failure.error))
                        })
                }) {
                    Ok(bounds) => {
                        bounds.with_ordered_counts_excluding(relocations, &mut storage, completion)
                    }
                    Err(error) => Err(PublicationError::Write(error)),
                }
            }
            None => engine.publish_pending_tree_excluding(
                pending,
                relocations,
                Some(&storage),
                |published| completion(published, &storage).map_err(Into::into),
            ),
        };
        match result {
            Ok(value) => Ok((value, collections, prospective)),
            Err(error) => Err(match error {
                PublicationError::Write(error) => error,
                PublicationError::Storage(error) => error.into(),
            }),
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
    /// Returns parent-preparation, placement, exclusion, publication, or root-association errors.
    pub(in crate::edit::commit) fn publish_compacting_members(
        mut self,
    ) -> Result<MemberCompletion, Error> {
        let decisions = match self.prepare_parents() {
            Ok(decisions) => decisions,
            Err(error) => return Err(outcomes::parent_error(error)),
        };
        let requested_root = self.pending.root;
        let ((_root, _end, mutation), _collections, _prospective) = self
            .publish_with_observations(|published, _, _, _, _, prospective| {
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
            #[cfg(test)]
            collections: _collections,
            #[cfg(test)]
            prospective: _prospective,
            #[cfg(test)]
            decisions,
            #[cfg(test)]
            root: _root,
            #[cfg(test)]
            end: _end,
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

/// The collection-completion result after protected tree publication succeeds.
pub(in crate::edit::commit) struct MemberCompletion {
    /// Original qualification and collection framing used for member decisions.
    #[cfg(test)]
    pub(in crate::edit::commit) collections: OriginalCollections,
    /// Fresh reference observations used to determine removal eligibility.
    #[cfg(test)]
    pub(in crate::edit::commit) prospective: ProspectiveCollections,
    /// Member, parent, and planning outcomes used for this completion.
    #[cfg(test)]
    pub(in crate::edit::commit) decisions: MemberDecisions,
    /// The final adopted root after tree publication.
    #[cfg(test)]
    pub(in crate::edit::commit) root: PhysicalAddress,
    /// The final published EOA.
    #[cfg(test)]
    pub(in crate::edit::commit) end: PhysicalAddress,
    /// The compaction result, including the completed prefix and allocation state on failure.
    pub(in crate::edit::commit) mutation: Result<(), HeapMutationFailure>,
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
    /// Source bytes outside independently known physical claims.
    Unobserved,
    /// An original object allocation with its physical owner identity.
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
    Object,
    /// Original owner qualification is absent or has incompatible associations.
    Owners,
    /// A parent has an unavailable framing, candidate-range, or allocation-class prerequisite.
    Parent,
    /// Parent storage collection failed.
    Parents,
    /// Conservative protection or coverage collection failed.
    Coverage(Error),
    /// Original reclamation collection failed.
    Reclamation,
    /// The original source association is unavailable.
    Source,
}

/// A protected allocation overlaps a space claim or extends past a proposed end.
#[derive(Debug)]
pub(in crate::edit) enum StorageConflict {
    /// Protected count facts do not match the captured source association.
    CountAssociation(CountProtectionError),
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
    /// A complete original count-header allocation.
    CountHeader,
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

#[cfg(test)]
mod tests;
