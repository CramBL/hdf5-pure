//! Owned positive count remainders and original allocation checks against allocator claims.

use core::num::NonZeroU32;
use std::collections::BTreeSet;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::ObjectReferenceCountField;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::RetainedKind;

use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::counts::HeaderCountAllocation;
use crate::edit::analysis::deletions::TopologyIdentity;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
#[cfg(test)]
use crate::object_graph::ObjectIndex;

use super::SourceRemovalPlan;

pub(super) mod inventory;
mod publication;

impl SourceRemovalPlan<'_, '_> {
    /// Copies known positive declared remainders into an owned batch.
    ///
    /// Checks the engine identity on every call. For a nonempty batch, records the allocation
    /// session, phase identity, and root resolved from the engine's superblock context. Omits zero
    /// and unknown remainders, retaining no source association for an empty batch. Uses existing
    /// plan metadata without reading the source or discovering topology.
    ///
    /// # Errors
    ///
    /// Returns [`CountHandoffError::Engine`] for a different engine, [`CountHandoffError::Root`]
    /// if a nonempty batch's topology root differs from the engine's root, or
    /// [`CountHandoffError::Source`] for root-address errors or a positive changed count without
    /// an explicit source field.
    #[cfg(test)]
    pub(in crate::edit) fn surviving_counts(
        &self,
        engine: &WriteEngine,
    ) -> Result<SurvivingCounts, CountHandoffError> {
        let topology = self.plan.topology;
        if !std::ptr::eq(topology.engine, engine) {
            return Err(CountHandoffError::Engine);
        }
        let mut changes = self
            .changes()
            .filter_map(|change| NonZeroU32::new(change.remaining()).map(|count| (change, count)))
            .peekable();
        if changes.peek().is_none() {
            return Ok(SurvivingCounts {
                source: None,
                updates: Vec::new(),
            });
        }
        let root = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(CountHandoffError::Source)?;
        let expected = topology.graph.object(ObjectIndex::ROOT).value.id;
        if root != expected {
            return Err(CountHandoffError::Root {
                expected,
                actual: root,
            });
        }
        let updates = changes
            .map(|(change, count)| {
                let bound = change.bound();
                let field = bound.field().ok_or_else(|| {
                    CountHandoffError::Source(FormatError::Internal(
                        "a positive changed count has no explicit source field".into(),
                    ))
                })?;
                Ok(SurvivingCount {
                    object: bound.object(),
                    field,
                    count,
                    allocation: bound.header_allocation(),
                })
            })
            .collect::<Result<Vec<_>, CountHandoffError>>()?;
        Ok(SurvivingCounts {
            source: Some(CountSource {
                session: engine.context.allocations.session(),
                identity: Arc::clone(&topology.identity),
                root,
            }),
            updates,
        })
    }
}

/// Owned original fields and positive declared remainders from one source phase.
///
/// The caller may drop the borrowed plan and phase while retaining this batch. The original
/// observed count and any independent header allocation fact are preserved for each entry.
/// Source freshness, storage ownership, and permission to write require separate evidence.
pub(in crate::edit) struct SurvivingCounts {
    source: Option<CountSource>,
    updates: Vec<SurvivingCount>,
}

impl SurvivingCounts {
    /// Checks original count allocations against the engine's recorded allocator claims.
    ///
    /// Omits reconstructed objects after checking their phase identity. Requires complete,
    /// disjoint allocations for the remaining fields and checks reusable ranges, retained PAGE
    /// extents, and aggregate spare. Preserves original fields and positive planned remainders.
    /// For an empty batch, examines neither the engine nor the reconstruction iterator.
    ///
    /// Source freshness and permission to write require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`CountProtectionError::Session`] or [`CountProtectionError::Root`] if the engine
    /// differs from the recorded session or root, or [`CountProtectionError::Publication`] if
    /// allocation publication is unavailable. Returns [`CountProtectionError::Phase`] for a
    /// reconstruction from another phase or [`CountProtectionError::Repeated`] for a duplicate.
    /// Returns [`CountProtectionError::Allocation`] for missing complete geometry or
    /// [`CountProtectionError::Overlap`] for intersecting original allocations. Returns
    /// [`CountProtectionError::Reusable`], [`CountProtectionError::Retained`], or
    /// [`CountProtectionError::Aggregate`] for a conflicting allocator claim, or
    /// [`CountProtectionError::Source`] for address-conversion errors.
    ///
    /// # Performance
    ///
    /// Builds indexes for reconstruction identities and surviving allocations. Borrows allocator
    /// lists without cloning them or expanding pages. Uses existing metadata without reading the
    /// source or discovering topology.
    #[cfg(test)]
    pub(in crate::edit) fn protect<'origin>(
        self,
        engine: &WriteEngine,
        reconstructed: impl IntoIterator<Item = &'origin ReconstructionSource>,
    ) -> Result<ProtectedCounts, CountProtectionError> {
        let Self { source, updates } = self;
        let Some(source) = source else {
            return Ok(ProtectedCounts {
                source: None,
                updates: Vec::new(),
            });
        };
        let replaced = source.reconstructions(engine, reconstructed)?;
        let protected = ProtectedCounts::new(source, updates, &replaced)?;
        protected.exclude_space(engine)?;
        Ok(protected)
    }

    /// Returns whether this batch is empty or its recorded allocation session matches `engine`.
    ///
    /// Session equality is independent of source freshness.
    #[cfg(test)]
    pub(in crate::edit) fn same_session(&self, engine: &WriteEngine) -> bool {
        self.source
            .as_ref()
            .is_none_or(|source| source.session.same(&engine.context.allocations.session()))
    }

    /// Returns whether a nonempty batch and `source` share the same phase identity.
    ///
    /// Compares phase identity independently of the physical object.
    #[cfg(test)]
    pub(in crate::edit) fn shares_phase(&self, source: &ReconstructionSource) -> bool {
        self.source
            .as_ref()
            .is_some_and(|origin| Arc::ptr_eq(&origin.identity, &source.identity))
    }

    /// Returns the original physical topology root, or `None` for an empty batch.
    #[cfg(test)]
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.source.as_ref().map(|source| source.root)
    }

    /// Borrows known positive count changes in the source plan's topology-index order.
    #[cfg(test)]
    pub(in crate::edit) fn updates(&self) -> &[SurvivingCount] {
        &self.updates
    }
}

/// An original explicit field paired with its physical object and positive planned remainder.
///
/// Optional allocation geometry describes only the complete chunk containing that field.
pub(in crate::edit) struct SurvivingCount {
    object: ObjectId,
    field: ObjectReferenceCountField,
    count: NonZeroU32,
    allocation: Option<HeaderCountAllocation>,
}

impl SurvivingCount {
    /// Returns the physical identity of the original source object.
    #[cfg(test)]
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }

    /// Returns the original observed count with its physical field and complete chunk associations.
    #[cfg(test)]
    pub(in crate::edit) fn field(&self) -> ObjectReferenceCountField {
        self.field
    }

    /// Returns the positive planned declared net count.
    #[cfg(test)]
    pub(in crate::edit) fn count(&self) -> NonZeroU32 {
        self.count
    }

    /// Returns the original checked complete header allocation, if available.
    #[cfg(test)]
    pub(in crate::edit) fn header_allocation(&self) -> Option<HeaderCountAllocation> {
        self.allocation
    }
}

/// Positive count changes with disjoint complete original allocations in physical address order.
///
/// The producer checks the recorded session, root, reconstruction phase, and allocator claims.
/// These checks describe that allocator state. Source freshness, retirement exclusions, and
/// permission to write require separate evidence. An empty batch has no source association.
pub(in crate::edit) struct ProtectedCounts {
    source: Option<CountSource>,
    updates: Vec<ProtectedCount>,
}

impl ProtectedCounts {
    /// Creates an empty batch without a source association or heap allocation.
    fn empty() -> Self {
        Self {
            source: None,
            updates: Vec::new(),
        }
    }

    /// Omits reconstructed objects and checks complete allocations for the remaining fields.
    ///
    /// Sorts by physical allocation address and retains no source association for an empty result.
    ///
    /// # Errors
    ///
    /// Returns [`CountProtectionError::Allocation`] for missing complete geometry or
    /// [`CountProtectionError::Overlap`] for intersecting original allocations.
    fn new(
        source: CountSource,
        updates: Vec<SurvivingCount>,
        replaced: &BTreeSet<ObjectId>,
    ) -> Result<Self, CountProtectionError> {
        let mut updates = updates
            .into_iter()
            .filter(|update| !replaced.contains(&update.object))
            .map(|update| {
                let SurvivingCount {
                    object,
                    field,
                    count,
                    allocation,
                } = update;
                let allocation = allocation.ok_or(CountProtectionError::Allocation { object })?;
                Ok(ProtectedCount {
                    object,
                    field,
                    count,
                    allocation,
                })
            })
            .collect::<Result<Vec<_>, CountProtectionError>>()?;
        updates.sort_unstable_by_key(|update| {
            (
                update.allocation.range().start(),
                update.object.header_address(),
            )
        });
        for pair in updates.windows(2) {
            if pair[0]
                .allocation
                .range()
                .overlaps(pair[1].allocation.range())
            {
                return Err(CountProtectionError::Overlap {
                    first: (pair[0].object, pair[0].allocation.range()),
                    second: (pair[1].object, pair[1].allocation.range()),
                });
            }
        }
        if updates.is_empty() {
            return Ok(ProtectedCounts {
                source: None,
                updates,
            });
        }
        Ok(Self {
            source: Some(source),
            updates,
        })
    }

    /// Checks the recorded engine associations and an optional source root and phase.
    ///
    /// Checks the allocation session, engine root, and publication availability for a nonempty
    /// batch. If `source` is present, also checks its physical root and phase identity. For an empty
    /// batch, skips both inputs. Source freshness requires separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`CountProtectionError::Session`] for a different allocation session,
    /// [`CountProtectionError::Root`] for a different engine or supplied root, or
    /// [`CountProtectionError::Phase`] for a supplied source from another phase. Returns
    /// [`CountProtectionError::Publication`] if allocation publication is unavailable, or
    /// [`CountProtectionError::Source`] for root-address errors.
    pub(in crate::edit) fn check_association(
        &self,
        engine: &WriteEngine,
        source: Option<&ReconstructionSource>,
    ) -> Result<(), CountProtectionError> {
        let Some(origin) = &self.source else {
            return Ok(());
        };
        origin.reconstructions(engine, [])?;
        if let Some(source) = source {
            if origin.root != source.object() {
                return Err(CountProtectionError::Root {
                    expected: origin.root,
                    actual: source.object(),
                });
            }
            if !Arc::ptr_eq(&origin.identity, &source.identity) {
                return Err(CountProtectionError::Phase {
                    object: source.object(),
                });
            }
        }
        Ok(())
    }

    /// Returns the original physical topology root, or `None` for an empty batch.
    #[cfg(test)]
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.source.as_ref().map(|source| source.root)
    }

    /// Borrows positive count changes in complete-allocation address order.
    pub(in crate::edit) fn updates(&self) -> &[ProtectedCount] {
        &self.updates
    }

    /// Checks exact reusable, retained PAGE, and aggregate-spare ranges in the engine.
    ///
    /// Resolves stored allocator coordinates through the engine's address context.
    ///
    /// # Errors
    ///
    /// Returns [`CountProtectionError::Reusable`], [`CountProtectionError::Retained`], or
    /// [`CountProtectionError::Aggregate`] for an intersecting claim, or
    /// [`CountProtectionError::Source`] for address-conversion errors.
    fn exclude_space(&self, engine: &WriteEngine) -> Result<(), CountProtectionError> {
        if self.updates.is_empty() {
            return Ok(());
        }
        let resolve = |extent: Extent| {
            FileRange::new(
                engine.addresses().resolve(extent.start())?,
                extent.len().get(),
            )
        };
        self.exclude_reusable(engine.space.reusable_sections().map(resolve))?;
        self.exclude_retained(
            engine
                .space
                .retained_extents()
                .map(|entry| resolve(entry.extent()).map(|range| (range, entry.kind()))),
        )?;
        for extent in engine
            .aggregation
            .into_iter()
            .flat_map(|space| space.unused())
        {
            let spare = resolve(extent).map_err(CountProtectionError::Source)?;
            if let Some(update) = self.overlapping(spare) {
                return Err(CountProtectionError::Aggregate {
                    object: update.object,
                    allocation: update.allocation.range(),
                    spare,
                });
            }
        }
        Ok(())
    }

    /// Checks unordered retained claims against sorted original allocations.
    ///
    /// # Errors
    ///
    /// Returns [`CountProtectionError::Retained`] for an intersecting claim or
    /// [`CountProtectionError::Source`] for a supplied range error.
    fn exclude_retained(
        &self,
        retained: impl Iterator<Item = Result<(FileRange, RetainedKind), FormatError>>,
    ) -> Result<(), CountProtectionError> {
        if self.updates.is_empty() {
            return Ok(());
        }
        for entry in retained {
            let (retained, kind) = entry.map_err(CountProtectionError::Source)?;
            if let Some(update) = self.overlapping(retained) {
                return Err(CountProtectionError::Retained {
                    object: update.object,
                    allocation: update.allocation.range(),
                    retained,
                    kind,
                });
            }
        }
        Ok(())
    }

    /// Finds the first complete count allocation intersecting the physical range.
    fn overlapping(&self, range: FileRange) -> Option<&ProtectedCount> {
        let index = self
            .updates
            .partition_point(|update| update.allocation.range().end() <= range.start());
        self.updates
            .get(index)
            .filter(|update| update.allocation.range().overlaps(range))
    }

    /// Checks address-ordered reusable ranges in one forward pass.
    ///
    /// # Errors
    ///
    /// Returns [`CountProtectionError::Reusable`] for an intersecting range or
    /// [`CountProtectionError::Source`] for a supplied range error.
    fn exclude_reusable(
        &self,
        mut free: impl Iterator<Item = Result<FileRange, FormatError>>,
    ) -> Result<(), CountProtectionError> {
        if self.updates.is_empty() {
            return Ok(());
        }
        let mut current = free
            .next()
            .transpose()
            .map_err(CountProtectionError::Source)?;
        for update in &self.updates {
            let allocation = update.allocation.range();
            while let Some(range) = current {
                if range.end() <= allocation.start() {
                    current = free
                        .next()
                        .transpose()
                        .map_err(CountProtectionError::Source)?;
                    continue;
                }
                if range.overlaps(allocation) {
                    return Err(CountProtectionError::Reusable {
                        object: update.object,
                        allocation,
                        reusable: range,
                    });
                }
                break;
            }
        }
        Ok(())
    }
}

/// An original explicit count field, its positive planned remainder, and complete allocation.
pub(in crate::edit) struct ProtectedCount {
    object: ObjectId,
    field: ObjectReferenceCountField,
    count: NonZeroU32,
    allocation: HeaderCountAllocation,
}

impl ProtectedCount {
    /// Returns the physical identity of the original source object.
    pub(in crate::edit) fn object(&self) -> ObjectId {
        self.object
    }

    /// Returns the original observed count with its physical field and chunk associations.
    pub(in crate::edit) fn field(&self) -> ObjectReferenceCountField {
        self.field
    }

    /// Returns the positive planned declared net count.
    pub(in crate::edit) fn count(&self) -> NonZeroU32 {
        self.count
    }

    /// Returns the complete original header allocation containing the count field.
    pub(in crate::edit) fn allocation(&self) -> HeaderCountAllocation {
        self.allocation
    }
}

/// An engine, root, or field inconsistency while copying count evidence.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum CountHandoffError {
    /// A supplied engine different from the plan's borrowed engine.
    Engine,
    /// A nonempty handoff whose topology root differs from the engine's superblock root.
    #[cfg(test)]
    Root {
        /// The physical root of the removal topology.
        expected: ObjectId,
        /// The physical root resolved from the engine's superblock context.
        actual: ObjectId,
    },
    /// A root-address error or a positive changed count without an explicit source field.
    #[cfg(test)]
    Source(FormatError),
}

/// The allocation session, phase identity, and physical root of a nonempty batch.
struct CountSource {
    session: AllocationSession,
    identity: Arc<TopologyIdentity>,
    root: ObjectId,
}

impl CountSource {
    /// Indexes reconstruction identities from the count batch's source phase.
    ///
    /// # Errors
    ///
    /// Returns [`CountProtectionError::Session`] or [`CountProtectionError::Root`] for a different
    /// session or root, or [`CountProtectionError::Publication`] if publication is unavailable.
    /// Returns [`CountProtectionError::Phase`] for a different reconstruction phase,
    /// [`CountProtectionError::Repeated`] for a duplicate object, or
    /// [`CountProtectionError::Source`] for root-address errors.
    fn reconstructions<'origin>(
        &self,
        engine: &WriteEngine,
        reconstructed: impl IntoIterator<Item = &'origin ReconstructionSource>,
    ) -> Result<BTreeSet<ObjectId>, CountProtectionError> {
        let Self {
            session,
            identity,
            root,
        } = self;
        if !session.same(&engine.context.allocations.session()) {
            return Err(CountProtectionError::Session);
        }
        let actual = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(CountProtectionError::Source)?;
        if actual != *root {
            return Err(CountProtectionError::Root {
                expected: *root,
                actual,
            });
        }
        engine
            .context
            .allocations
            .available()
            .map_err(CountProtectionError::Publication)?;
        let mut replaced = BTreeSet::new();
        for origin in reconstructed {
            let object = origin.object();
            if !Arc::ptr_eq(identity, &origin.identity) {
                return Err(CountProtectionError::Phase { object });
            }
            if !replaced.insert(object) {
                return Err(CountProtectionError::Repeated { object });
            }
        }
        Ok(replaced)
    }
}

/// An inapplicable source association, incomplete allocation, or conflicting allocator claim.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the state retains original observations for failure and qualification diagnostics"
    )
)]
pub(in crate::edit) enum CountProtectionError {
    /// Aggregate spare intersecting a complete original allocation.
    Aggregate {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The complete original header allocation.
        allocation: FileRange,
        /// The allocator's unconsumed aggregate range.
        spare: FileRange,
    },
    /// Missing complete allocation geometry for an original count field.
    Allocation {
        /// The physical identity of the source object.
        object: ObjectId,
    },
    /// Intersecting complete allocations associated with distinct source objects.
    Overlap {
        /// The first physical object and its complete allocation range.
        first: (ObjectId, FileRange),
        /// The second physical object and its complete allocation range.
        second: (ObjectId, FileRange),
    },
    /// A reconstruction identity from a different source phase.
    Phase {
        /// The physical identity of the reconstructed object.
        object: ObjectId,
    },
    /// Unavailable allocation publication.
    Publication(Error),
    /// A repeated reconstruction identity.
    Repeated {
        /// The physical identity supplied more than once.
        object: ObjectId,
    },
    /// An exact retained PAGE claim intersecting a complete original allocation.
    Retained {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The complete original header allocation.
        allocation: FileRange,
        /// The exact recorded retained PAGE range.
        retained: FileRange,
        /// The allocator's recorded category for the intersecting range.
        kind: RetainedKind,
    },
    /// A reusable range intersecting a complete original allocation.
    Reusable {
        /// The physical identity of the source object.
        object: ObjectId,
        /// The complete original header allocation.
        allocation: FileRange,
        /// The intersecting reusable range.
        reusable: FileRange,
    },
    /// A physical root different from the original topology root.
    Root {
        /// The original topology root.
        expected: ObjectId,
        /// The root resolved from the engine or supplied in a source association.
        actual: ObjectId,
    },
    /// An engine from a different allocation session.
    Session,
    /// An error resolving a root or allocator range into physical coordinates.
    Source(FormatError),
}

#[cfg(test)]
mod tests;
