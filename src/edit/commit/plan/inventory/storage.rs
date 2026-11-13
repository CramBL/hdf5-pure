//! Original allocations excluded from placement, pending retirement, and end truncation.
//!
//! Bounds checks preserve recorded owner and parent ranges. Source freshness, mutation
//! permission, and publication require separate evidence.

use std::collections::{BTreeMap, BTreeSet};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{Extent, FreeClass, RetainedKind};

use self::capture::StorageIntegrityError;
use self::parents::ParentPreparationError;
use super::dispositions::{AppliedDispositions, Disposition, ReconstructionDispositions};
use super::members::MemberDecisions;
use super::members::PreparedParentRetirement;
use super::prospective::ProspectiveCollections;
use super::{AppliedEditInventory, OriginalEdits, PreparedEditInventory};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::heap::qualification::{
    OriginalCollections, ParentStorage, ParentUnavailable,
};
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::physical::claims::{ClaimOrigin, ClaimProblem};
use crate::edit::analysis::{
    AnalysisPhase, CountProtectionError, ProtectedCounts, ReconstructionSource,
};
use crate::edit::commit::CommitOutcome;
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

/// Original allocations with optional source-phase and parent observations.
///
/// Records original owner allocations and complete observed heap collections, including collections
/// without reclamation candidates. When owner qualification is unavailable, also protects source
/// intervals outside the observed claims. Records source integrity independently of qualification.
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
    /// Retains known ranges when qualification fails. For candidate parents, protects the enclosing
    /// range of the original allocation and parsed framing. Reuses cached phase observations for
    /// source-integrity checks. Omits an eligible or superseded owner's allocation only when its
    /// exact range is recorded for retirement or deferred tail replacement. Preserves original
    /// allocations reachable from accepted residual groups, including superseded owners.
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
        let residual_owners = if original.owners().is_none() {
            match phase.residual_group_domain(source) {
                Ok(groups) => groups
                    .as_ref()
                    .as_ref()
                    .map(|groups| groups.objects().clone())
                    .unwrap_or_default(),
                Err(error) => {
                    result.unavailable = Some(StorageUnavailable::Coverage(error));
                    result.integrity = Err(StorageIntegrityError::Unavailable);
                    result.capture_parents();
                    return result;
                }
            }
        } else {
            BTreeSet::new()
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
            let (object, kind) = match claim.origin() {
                ClaimOrigin::Heap(address) => {
                    result.allocations.push(StorageAllocation {
                        range: claim.range(),
                        origin: StorageOrigin::Parent {
                            address,
                            class: None,
                        },
                    });
                    continue;
                }
                ClaimOrigin::Object { object, kind } => (object, kind),
                _ => continue,
            };
            let eligible = original.owners().is_some() && eligible.contains(&object);
            let superseded = superseded.contains(&object);
            let retired = reclamation.as_ref().is_ok_and(|retired| {
                retired
                    .binary_search_by_key(&claim.range().start(), |(range, _)| range.start())
                    .ok()
                    .is_some_and(|index| retired[index].0 == claim.range())
            }) || tails.contains(&(claim.range().start(), claim.range().end()));
            if residual_owners.contains(&object) || !((eligible || superseded) && retired) {
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
        result.capture_parents();
        result.capture_integrity(engine, phase, source);

        result
    }

    /// Preserves protected owner and unobserved storage when owner qualification is unavailable.
    ///
    /// Removes generated retirements equal to a protected owner allocation or contained in a
    /// protected unobserved interval. Preserves parent requests and other ranges for exclusion
    /// checks. Leaves reclamation errors and qualified-owner inputs unchanged.
    pub(super) fn withhold_unqualified_reclamation(
        &self,
        original: &OriginalEdits,
        reclamation: &mut Result<Vec<(FileRange, FreeClass)>, Error>,
    ) {
        if original.owners().is_some() {
            return;
        }
        let Ok(retired) = reclamation else {
            return;
        };
        retired.retain(|&(range, _)| {
            let index = self
                .allocations
                .partition_point(|allocation| allocation.range.end() <= range.start());
            !self
                .allocations
                .get(index)
                .is_some_and(|allocation| match allocation.origin {
                    StorageOrigin::Owner { .. } => allocation.range == range,
                    StorageOrigin::Parent { .. } => false,
                    StorageOrigin::Unobserved => allocation.range.contains(range),
                })
        });
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

    /// Returns the recorded source and protected-range geometry outcome.
    ///
    /// Qualification and source freshness require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns the recorded [`StorageIntegrityError`] for conflicting ranges, invalid bounds,
    /// or unavailable source observations.
    pub(in crate::edit::commit) fn integrity(&self) -> Result<(), StorageIntegrityError> {
        self.integrity.clone()
    }

    /// Borrows a qualification or source-observation failure, independently of recorded ranges.
    ///
    /// Source-observation failures take precedence over qualification failures.
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
    /// Returns [`StorageConflict::Original`] if the recorded integrity outcome is an error.
    /// Returns [`StorageConflict::Overlap`] if an allocator range overlaps protected storage.
    /// Returns [`StorageConflict::Source`] if an address or range cannot be resolved.
    pub(in crate::edit::commit) fn exclude_space(
        &self,
        engine: &WriteEngine,
    ) -> Result<(), StorageConflict> {
        self.integrity().map_err(StorageConflict::Original)?;
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

    /// Extracts a recorded claims, coverage, or parent storage collection error.
    ///
    /// Prioritizes a claims or coverage error over the parent storage collection result.
    /// Qualification markers and recorded integrity geometry are handled separately.
    pub(super) fn into_capture_error(self) -> Option<Error> {
        let Self {
            source: _,
            allocations: _,
            parents,
            integrity: _,
            unavailable,
        } = self;
        match unavailable {
            Some(StorageUnavailable::Claims(error) | StorageUnavailable::Coverage(error)) => {
                Some(error)
            }
            None
            | Some(
                StorageUnavailable::Dispositions
                | StorageUnavailable::Object(_)
                | StorageUnavailable::Owners
                | StorageUnavailable::Parent { .. }
                | StorageUnavailable::Parents
                | StorageUnavailable::Reclamation
                | StorageUnavailable::Source,
            ) => parents.err(),
        }
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
    /// Permits unavailable owner or individual-parent qualification while preserving original
    /// storage exclusions. Requires complete predecessor replacement, associated parent
    /// observations, valid source geometry, and allocation publication availability.
    /// On success, retains the allocator and pending-retirement exclusion outcome in
    /// [`PendingEditInventory::exclusion`]. The caller handles rollback after application failure.
    ///
    /// # Errors
    ///
    /// Returns [`StorageApplyFailure::BeforePlacement`] with the owned prepared inputs for
    /// incompatible associations, unavailable parent associations or predecessor replacement,
    /// invalid source geometry, or an allocator conflict. Returns
    /// [`StorageApplyFailure::Application`] for an error from [`Self::apply`].
    pub(in crate::edit::commit) fn apply_excluding(
        self,
        engine: &mut WriteEngine,
    ) -> Result<PendingEditInventory<'_>, StorageApplyFailure> {
        if let Err(error) = self.before_placement(engine) {
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

    /// Checks mandatory associations, replacement dispositions, and original storage exclusions.
    ///
    /// Checks available count and storage facts independently of complete owner qualification.
    ///
    /// # Errors
    ///
    /// Returns [`StorageApplyError::Session`], [`StorageApplyError::Phase`], or
    /// [`StorageApplyError::Root`] for incompatible associations. Returns
    /// [`StorageApplyError::Unavailable`] if parent observations have incompatible associations or
    /// a predecessor is excluded from complete replacement. Returns [`StorageApplyError::Conflict`]
    /// if allocation publication, recorded integrity, or allocator exclusion checks fail.
    fn before_placement(&self, engine: &WriteEngine) -> Result<(), StorageApplyError> {
        let Self {
            prepared: _,
            original,
            session,
            dispositions,
            storage,
            collections: _,
            reclamation: _,
        } = self;
        if !session.same(&engine.context.allocations.session())
            || original.facts().is_some_and(|facts| {
                !facts.storage().same_session(engine) || !facts.counts().same_session(engine)
            })
        {
            Err(StorageApplyError::Session)
        } else if storage.source.as_ref() != dispositions.source().as_ref().ok()
            || original.source() != storage.source.as_ref()
            || original.facts().is_some_and(|facts| {
                facts.counts().root().is_some()
                    && storage
                        .source
                        .as_ref()
                        .is_none_or(|source| !facts.counts().shares_phase(source))
            })
        {
            Err(StorageApplyError::Phase)
        } else if storage.source.as_ref().map(ReconstructionSource::object)
            != ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).ok()
            || original.facts().is_some_and(|facts| {
                facts.counts().root().is_some_and(|root| {
                    storage.source.as_ref().map(ReconstructionSource::object) != Some(root)
                })
            })
        {
            Err(StorageApplyError::Root)
        } else if storage.parents.is_err()
            || dispositions
                .predecessors()
                .values()
                .any(|old| old.disposition != Disposition::Supersede)
        {
            Err(StorageApplyError::Unavailable)
        } else {
            engine
                .context
                .allocations
                .before_reference()
                .map_err(StorageConflict::Source)
                .and_then(|()| storage.exclude_space(engine))
                .map_err(StorageApplyError::Conflict)
        }
    }
}

/// Owns placed-tree observations and pending frees while borrowing the application engine.
///
/// Pending frees include deferred chunk retirement at the engine's current length. The editor
/// checks allocator and retirement ranges against original storage and retains the outcome.
/// With original candidates, the editor decodes heap references in a new phase after application
/// and retains root-applicability failures independently of reference interpretation.
/// [`Self::publish_compacting_members`] derives member removal permission from these observations.
/// Other completion callbacks require independently established mutation permission. Source count
/// updates are retained for ordered publication with the original storage exclusions.
pub(in crate::edit::commit) struct PendingEditInventory<'a> {
    engine: &'a mut WriteEngine,
    /// The allocation session retained from the prepared application inputs.
    session: AllocationSession,
    /// Protected source-count updates retained from the applied tree.
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
            session,
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
    /// Checks placement, source/root/session associations, current allocator claims, and complete
    /// parent protection before consuming whole-parent retirement evidence into the pending batch.
    /// Removes only matching retiring-parent exclusions after all retirement checks succeed.
    /// Preserves full exclusions for retained and compacted parents. Writes source-count increases
    /// before publication and decreases afterward, before
    /// `complete`. Adds complete original count-header allocations to the inventory's exclusions.
    /// Retains the inventory's exclusions even for an empty count batch. Calls `complete` after
    /// publication, reference updates, root reclamation, truncation, count writes, and final
    /// storage checks succeed. The callback receives the original observations, applied
    /// dispositions, storage exclusions, and original collection qualification and framing
    /// associated with these inputs, together with prospective observations collected after
    /// application.
    ///
    /// # Errors
    ///
    /// Returns [`StoragePublicationFailure::BeforePublication`] with these pending inputs if
    /// placement, parent preparation, association, or exclusion checks fail. Returns
    /// [`StoragePublicationFailure::Publication`] with the retained observations and error if
    /// count protection, source-count writes, publication, or completion fails. The caller handles
    /// rollback according to publication state.
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
        self.publish_with_observations(
            |published, original, dispositions, storage, collections, prospective, _| {
                complete(
                    published,
                    original,
                    dispositions,
                    storage,
                    collections,
                    prospective,
                )
            },
        )
        .map(|(value, _, _, _)| value)
    }

    /// Retains observations and consumed-retirement metadata after publication and completion.
    ///
    /// Uses the publication and error contract of [`Self::publish_with_completion`].
    ///
    /// # Errors
    ///
    /// Returns [`StoragePublicationFailure::BeforePublication`] if placement, parent preparation,
    /// association, or exclusion checks fail, or [`StoragePublicationFailure::Publication`] if
    /// publication or
    /// count protection, source-count writes, or completion fails.
    fn publish_with_observations<T>(
        mut self,
        complete: impl FnOnce(
            PublishedTree<'_>,
            &OriginalEdits,
            &AppliedDispositions,
            &OriginalStorage,
            &OriginalCollections,
            &ProspectiveCollections,
            &MemberDecisions<PreparedParentRetirement>,
        ) -> Result<T, Error>,
    ) -> Result<
        (
            T,
            OriginalCollections,
            ProspectiveCollections,
            MemberDecisions<PreparedParentRetirement>,
        ),
        Box<StoragePublicationFailure<'a>>,
    > {
        if self.dispositions.replacements().is_err() {
            return Err(Box::new(StoragePublicationFailure::BeforePublication {
                pending: self,
                reason: StoragePublicationRejection::Placement,
            }));
        }
        let decisions = match self.prepare_parents() {
            Ok(decisions) => decisions,
            Err(error) => {
                let reason = match error {
                    ParentPreparationError::Storage(error) => {
                        self.exclusion = Err(error);
                        StoragePublicationRejection::Exclusion
                    }
                    ParentPreparationError::RecordedExclusion => {
                        StoragePublicationRejection::Exclusion
                    }
                    error => StoragePublicationRejection::Parents(Box::new(error)),
                };
                return Err(Box::new(StoragePublicationFailure::BeforePublication {
                    pending: self,
                    reason,
                }));
            }
        };
        self.exclusion = Ok(());
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
                &decisions,
            )
        };
        let result = match counts {
            Some(counts) => {
                let end = PhysicalAddress::new(engine.image.len());
                counts
                    .before_publication(engine, pending)
                    .map_err(|error| PublicationError::from(Error::from(error)))
                    .and_then(|phase| {
                        phase
                            .at_end(end)
                            .map_err(|failure| PublicationError::from(Error::from(failure.error)))
                    })
                    .and_then(|bounds| {
                        bounds.with_ordered_counts_excluding(relocations, &mut storage, completion)
                    })
            }
            None => engine.publish_pending_tree_excluding(
                pending,
                relocations,
                Some(&storage),
                |published| completion(published, &storage).map_err(PublicationError::from),
            ),
        };
        match result {
            Ok(value) => Ok((value, collections, prospective, decisions)),
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

    /// Derives removals from qualified observations and checked session-owned parent evidence.
    pub(in crate::edit::commit) fn member_decisions(&self) -> MemberDecisions {
        let mut decisions =
            MemberDecisions::collect(&self.collections, &self.prospective, &self.storage);
        decisions.include_session_parents(&self.dispositions, &self.prospective);
        decisions
    }

    /// Publishes the protected tree before compacting qualified global heap members.
    ///
    /// Prepares exclusive parent dispositions and checks original and prospective associations.
    /// Transfers checked whole-parent retirements into the pending batch and removes their
    /// matching exclusions.
    /// The publisher preserves full exclusions for retained and compacted parents. After
    /// publication, applies the borrowed compaction plans through [`heap::after_publication`].
    ///
    /// Returns the final adopted root and EOA with the observations and decisions. A heap
    /// mutation failure is recorded in [`MemberCompletion::mutation`] after publication.
    ///
    /// # Errors
    ///
    /// Returns [`StoragePublicationFailure::BeforePublication`] if placement reconciliation or
    /// parent preparation, association, or storage exclusion checks fail. Returns
    /// [`StoragePublicationFailure::Publication`] if count protection, source-count writes,
    /// publication, or final root checks fail.
    pub(in crate::edit::commit) fn publish_compacting_members(
        self,
    ) -> Result<MemberCompletion, Box<StoragePublicationFailure<'a>>> {
        let requested_root = self.pending.root;
        let ((root, end, mutation), collections, prospective, decisions) = self
            .publish_with_observations(|published, _, _, _, _, prospective, decisions| {
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
    /// Member, parent, and planning outcomes with consumed whole-retirement metadata.
    pub(in crate::edit::commit) decisions: MemberDecisions<PreparedParentRetirement>,
    /// The final adopted root after tree publication.
    pub(in crate::edit::commit) root: PhysicalAddress,
    /// The final published EOA.
    pub(in crate::edit::commit) end: PhysicalAddress,
    /// The compaction result, including the completed prefix and allocation state on failure.
    pub(in crate::edit::commit) mutation: Result<(), HeapMutationFailure>,
}

impl MemberCompletion {
    /// Transfers confirmed publication and its compaction result to commit finalization.
    pub(in crate::edit::commit) fn into_outcome(self) -> CommitOutcome {
        let Self {
            collections: _,
            prospective: _,
            decisions: _,
            root: _,
            end: _,
            mutation,
        } = self;
        CommitOutcome::Published(mutation)
    }
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
    /// Parent association, protection, or retirement preparation failed.
    Parents(Box<ParentPreparationError>),
    /// Applied replacement placements could not be reconciled with the prepared dispositions.
    Placement,
}

/// A protected physical range with an observed origin or unknown ownership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) struct StorageAllocation {
    pub(in crate::edit::commit) range: FileRange,
    pub(in crate::edit::commit) origin: StorageOrigin,
}

/// The observed owner, global heap parent, or unknown ownership of a protected range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit::commit) enum StorageOrigin {
    /// An original object allocation with its physical owner identity.
    Owner {
        object: ObjectId,
        kind: AllocationKind,
    },
    /// Complete collection protection with the candidate allocation class, if recorded.
    Parent {
        address: PhysicalAddress,
        class: Option<FreeClass>,
    },
    /// A source interval outside the observed claims, protected when owner qualification fails.
    Unobserved,
}

/// A qualification or observation failure recorded during original-storage capture.
#[derive(Debug)]
pub(in crate::edit::commit) enum StorageUnavailable {
    /// Physical-claims collection failed.
    Claims(Error),
    /// Source coverage or conservative range construction failed.
    Coverage(Error),
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

/// A source or association failure, or a conflict involving protected storage.
#[derive(Debug)]
pub(in crate::edit) enum StorageConflict {
    /// An inapplicable count source or unavailable allocation publication.
    CountAssociation(CountProtectionError),
    /// The proposed end is below the greatest protected range end.
    End {
        proposed: PhysicalAddress,
        required: PhysicalAddress,
    },
    /// Original source or protected-range geometry is invalid or unavailable.
    Original(StorageIntegrityError),
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
    /// A complete original header allocation containing an updated source count.
    CountHeader,
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
    /// Allocation publication, source integrity, or original storage exclusion failed.
    Conflict(StorageConflict),
    /// Original storage, predecessor, or count observations have incompatible source associations.
    Phase,
    /// An original source or count root differs from the engine or storage root.
    Root,
    /// An original observation or prepared input has a different allocation session.
    Session,
    /// Parent associations or complete predecessor replacement are unavailable.
    Unavailable,
}

#[cfg(test)]
mod tests;
