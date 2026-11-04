//! Original count fields and complete allocations checked against recorded allocator claims.

use core::num::NonZeroU32;
use std::collections::BTreeSet;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::ObjectReferenceCountField;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::RetainedKind;

use crate::edit::WriteEngine;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::counts::HeaderCountAllocation;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

use super::CountSource;
use super::SurvivingCount;
use super::SurvivingCounts;

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

    /// Returns the original physical topology root, or `None` for an empty batch.
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
            FileRange::new(engine.addresses().resolve(extent.start())?, extent.len())
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

    /// Returns the positive declared remainder after the planned removals.
    pub(in crate::edit) fn count(&self) -> NonZeroU32 {
        self.count
    }

    /// Returns the complete original header allocation containing the count field.
    pub(in crate::edit) fn allocation(&self) -> HeaderCountAllocation {
        self.allocation
    }
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
    /// A physical superblock root different from the original topology root.
    Root {
        /// The original topology root.
        expected: ObjectId,
        /// The root resolved from the engine's superblock context.
        actual: ObjectId,
    },
    /// An engine from a different allocation session.
    Session,
    /// An error resolving a root or allocator range into physical coordinates.
    Source(FormatError),
}

#[cfg(test)]
mod tests;
