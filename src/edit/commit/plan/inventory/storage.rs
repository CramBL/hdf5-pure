//! Original allocations excluded from placement, pending retirement, and end truncation.
//!
//! Bounds checks preserve recorded owner and parent ranges. Source freshness, mutation
//! permission, and publication require separate evidence.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{Extent, FreeClass, RetainedKind};

use self::capture::{ClaimReleases, ResidualRelease, StorageIntegrityError};
use super::dispositions::{AppliedDispositions, ReconstructionDispositions};
use super::prospective::ProspectiveCollections;
use super::{AppliedEditInventory, OriginalEdits, PreparedEditInventory};
use crate::edit::WriteEngine;
use crate::edit::analysis::heap::qualification::{OriginalCollections, ParentStorage};
use crate::edit::analysis::ownership::{AllocationKind, ObjectAllocation};
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::analysis::{
    AnalysisPhase, CountProtectionError, PreservedReferences, ProtectedCounts, ReconstructionSource,
};
use crate::edit::commit::heap;
use crate::edit::commit::heap::HeapMutationFailure;
use crate::edit::commit::plan::PreparedTree;
use crate::edit::commit::publish::{PublicationError, PublishedTree};
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
    source_error: Option<Error>,
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
            source_error: None,
        }
    }

    /// Collects original storage exclusions from the prepared observations and reclamation input.
    ///
    /// Retains predecessor allocations when residual storage cannot be qualified. Replacement
    /// does not establish whether an unavailable residual group still references the old owner.
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
            source_error: None,
        };
        let Some(source) = dispositions.source().as_ref().ok() else {
            result.integrity = Err(StorageIntegrityError::Unavailable);
            return result;
        };
        result.parents = phase.collection_storage(source, &tree.collection_candidates);
        let groups = if original.owners().is_none() {
            match phase.residual_group_domain(source) {
                Ok(groups) => Some(groups),
                Err(error) => {
                    result.source_error = Some(error);
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
                result.source_error = Some(error);
                result.integrity = Err(StorageIntegrityError::Unavailable);
                result.capture_parents();
                return result;
            }
        };
        let qualified_owners = original
            .owners()
            .is_some_and(|owners| owners.source() == source && owners.matches(engine, claims));
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
        let releases =
            ClaimReleases::new(&tree.groups, original, dispositions, reclamation, residual);
        result.capture_claims(claims, &releases);
        result.capture_parents();
        result.capture_integrity(engine, phase, source, !qualified_owners);

        result
    }

    /// Selects original reference summaries whose complete object storage will be excluded
    /// from placement. Application must check these exclusions before using the summaries.
    ///
    /// Explicit in-place writes can modify existing value and index ranges. Their prepared
    /// forms do not all identify a checked owner, so they leave reuse unavailable.
    pub(super) fn preserved_references(
        &self,
        phase: &mut AnalysisPhase<'_>,
        original: &OriginalEdits,
        dispositions: &ReconstructionDispositions,
        tree: &PreparedTree,
    ) -> PreservedReferences {
        let Some(owners) = original.owners() else {
            return PreservedReferences::default();
        };
        let source = owners.source();
        if !tree.inplace_writes.is_empty()
            || self.integrity.is_err()
            || self.source.as_ref() != Some(source)
        {
            return PreservedReferences::default();
        }
        let mut references = phase.take_preserved_references(source, |object, allocations| {
            !allocations.is_empty()
                && allocations
                    .iter()
                    .all(|allocation| self.protects_owner_allocation(object, *allocation))
        });
        references.exclude(
            dispositions
                .predecessors()
                .values()
                .map(|old| old.source.object()),
        );
        references
    }

    fn protects_owner_allocation(&self, object: ObjectId, allocation: ObjectAllocation) -> bool {
        let range = allocation.range();
        self.allocations
            .binary_search_by_key(&range.start(), |protected| protected.range.start())
            .is_ok_and(|index| {
                self.allocations[index]
                    == StorageAllocation {
                        range,
                        origin: StorageOrigin::Owner {
                            object,
                            kind: allocation.kind(),
                        },
                    }
            })
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
    /// Checks original storage exclusions before placement and records pending retirement.
    ///
    /// Checks the source-phase, root, and allocation-session associations before application.
    /// On success, retains the allocator and pending-retirement exclusion outcome. The caller
    /// handles rollback after application failure.
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
/// With original candidates, the editor collects heap references in a new phase after application
/// and retains root-applicability failures independently of reference interpretation.
/// [`Self::publish_compacting_members`] derives member removal permission from these observations.
/// Other completion callbacks require independently established mutation permission.
pub(in crate::edit::commit) struct PendingEditInventory<'a> {
    engine: &'a mut WriteEngine,
    session: AllocationSession,
    counts: Option<ProtectedCounts>,
    dispositions: AppliedDispositions,
    storage: OriginalStorage,
    collections: OriginalCollections,
    prospective: ProspectiveCollections,
    pending: PendingRootFrees,
    exclusion_conflict: Option<StorageConflict>,
}

impl<'a> PendingEditInventory<'a> {
    fn new(engine: &'a mut WriteEngine, applied: AppliedEditInventory) -> Self {
        let AppliedEditInventory {
            counts,
            retirement,
            freed,
            original,
            dispositions,
            storage,
            collections,
            references,
        } = applied;
        let pending = retirement.into_pending_frees(freed, engine.image.len());
        let exclusion_conflict = storage
            .exclude_space(engine)
            .and_then(|()| {
                for &(range, class) in pending.batch.entries() {
                    storage.exclude(range, SpaceClaim::Retirement(class))?;
                }
                Ok(())
            })
            .err();
        let prospective = ProspectiveCollections::collect(
            engine,
            &original,
            &dispositions,
            &collections,
            pending.root,
            if exclusion_conflict.is_none() {
                references
            } else {
                PreservedReferences::default()
            },
        );
        let session = engine.context.allocations.session();
        Self {
            engine,
            session,
            counts,
            dispositions,
            storage,
            collections,
            prospective,
            pending,
            exclusion_conflict,
        }
    }

    /// Publishes the protected tree before compacting qualified global heap members.
    ///
    /// Derives plans from the owned original and prospective observations and checks publication
    /// and prospective root associations against the pending root. The writer preserves each
    /// parent allocation and applies the plans through [`heap::after_publication`].
    ///
    /// Returns the heap mutation result after publishing the tree.
    ///
    /// # Errors
    ///
    /// Returns parent-preparation, placement, exclusion, publication, or root-association errors.
    pub(in crate::edit::commit) fn publish_compacting_members(
        mut self,
    ) -> Result<Result<(), HeapMutationFailure>, Error> {
        let decisions = self.prepare_parents().map_err(outcomes::parent_error)?;
        let requested_root = self.pending.root;
        let prospective_root = self.prospective.root();
        let complete = |published: PublishedTree<'_>| {
            if published.requested_root() != requested_root
                || prospective_root.is_some_and(|root| root != requested_root)
            {
                return Err(FormatError::Internal(
                    "heap decisions belong to another published tree".into(),
                )
                .into());
            }
            Ok(heap::after_publication(published, decisions.plans()))
        };
        let relocations = self
            .dispositions
            .into_replacements()
            .ok_or(Error::EditUnsupported(
                "original predecessor replacement cannot be established",
            ))?;
        let Self {
            engine,
            session: _,
            counts,
            dispositions: _,
            mut storage,
            collections: _,
            prospective: _,
            pending,
            exclusion_conflict: _,
        } = self;
        let result = match counts {
            Some(counts) => {
                let end = engine
                    .end_of_allocation()
                    .and_then(|end| engine.addresses().resolve(end))
                    .map_err(Error::from);
                match end.and_then(|end| counts.before_publication(engine, pending)?.at_end(end)) {
                    Ok(bounds) => {
                        bounds.with_ordered_counts_excluding(relocations, &mut storage, complete)
                    }
                    Err(error) => Err(PublicationError::Write(error)),
                }
            }
            None => engine.publish_pending_tree_excluding(
                pending,
                relocations,
                Some(&storage),
                |published| complete(published).map_err(Into::into),
            ),
        };
        result.map_err(|error| match error {
            PublicationError::Write(error) => error,
            PublicationError::Storage(error) => error.into(),
        })
    }
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
