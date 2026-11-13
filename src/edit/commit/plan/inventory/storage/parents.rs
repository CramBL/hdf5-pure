//! Checked transfer of whole-parent retirement evidence into pending publication.

use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use super::OriginalStorage;
use super::PendingEditInventory;
use super::SpaceClaim;
use super::StorageAllocation;
use super::StorageConflict;
use super::StorageOrigin;
use crate::edit::WriteEngine;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::qualification::OriginalQualification;
use crate::edit::commit::plan::inventory::members::MemberDecisions;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::commit::plan::inventory::members::PreparedParentRetirement;
use crate::edit::retirement::PendingRootFrees;
use crate::edit::retirement::ProvenRetirementBatch;
use crate::edit::retirement::RetirementMergeError;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

impl PendingEditInventory<'_> {
    /// Installs checked whole-parent retirements and consumes their original permissions.
    ///
    /// Checks allocation availability, session/source/root associations, full parent protection,
    /// allocator claims, and retirement bounds, classes, and overlap. Removes matching parent
    /// exclusions only after all checks pass. Preserves full exclusions for retained and compacted
    /// parents.
    /// Accepts enclosing protection for retained parents. Requires exact ranges for compaction
    /// and retirement and checks any required allocation class.
    /// If preparation fails, storage exclusions and the pending retirement batch remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ParentPreparationError`] if association, allocation, protection, retirement-batch,
    /// or recorded exclusion checks fail.
    pub(super) fn prepare_parents(
        &mut self,
    ) -> Result<MemberDecisions<PreparedParentRetirement>, ParentPreparationError> {
        self.check_parent_associations()?;
        let Self {
            engine,
            session: _,
            counts: _,
            original: _,
            dispositions: _,
            storage,
            collections,
            prospective,
            pending,
            relocations: _,
            exclusion,
        } = self;
        let decisions = MemberDecisions::collect(collections, prospective, storage);
        ParentRelease::new(engine, storage, pending, exclusion, decisions).map(ParentRelease::apply)
    }

    /// Checks allocation availability and pending source, root, and session associations.
    ///
    /// # Errors
    ///
    /// Returns allocation, address-resolution, session, original-root, source-association, or
    /// prospective-root failures as [`ParentPreparationError`].
    fn check_parent_associations(&self) -> Result<(), ParentPreparationError> {
        let Self {
            engine,
            session,
            counts: _,
            original: _,
            dispositions,
            storage,
            collections,
            prospective,
            pending,
            relocations: _,
            exclusion: _,
        } = self;
        engine
            .context
            .allocations
            .before_reference()
            .map_err(ParentPreparationError::Allocation)?;
        if !session.same(&engine.context.allocations.session()) {
            return Err(ParentPreparationError::Session);
        }
        let expected = storage.source.as_ref();
        let applied = dispositions.original().source().as_ref().ok();
        let Some(source) = expected else {
            return Err(ParentPreparationError::Source {
                expected: None,
                actual: applied.cloned(),
            });
        };
        if expected != applied {
            return Err(ParentPreparationError::Source {
                expected: expected.cloned(),
                actual: applied.cloned(),
            });
        }
        let actual = ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address)
            .map_err(ParentPreparationError::Address)?;
        if source.object() != actual {
            return Err(ParentPreparationError::OriginalRoot {
                expected: source.object(),
                actual,
            });
        }
        if let Some(actual) = prospective.source() {
            if actual != source {
                return Err(ParentPreparationError::Source {
                    expected: Some(source.clone()),
                    actual: Some(actual.clone()),
                });
            }
        }
        if let OriginalQualification::Qualified(original) = collections.qualification() {
            if original.source() != source {
                return Err(ParentPreparationError::Source {
                    expected: Some(source.clone()),
                    actual: Some(original.source().clone()),
                });
            }
        }
        if let Some(actual) = prospective.root() {
            if actual != pending.root {
                return Err(ParentPreparationError::Root {
                    expected: pending.root,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// A checked retirement batch and matching exclusions ready for an infallible transfer.
///
/// Construction leaves both borrowed inputs unchanged. Application removes only the recorded
/// parent exclusions, installs the checked batch, and consumes active retirement evidence.
struct ParentRelease<'a> {
    storage: &'a mut OriginalStorage,
    pending: &'a mut PendingRootFrees,
    decisions: MemberDecisions,
    released: BTreeSet<PhysicalAddress>,
    batch: Option<ProvenRetirementBatch>,
}

impl<'a> ParentRelease<'a> {
    /// Checks parent protection, allocator claims, and the complete merged retirement batch.
    ///
    /// Accepts a recorded exclusion conflict only for an exact matching whole-parent retirement.
    ///
    /// # Errors
    ///
    /// Returns [`ParentPreparationError`] for missing protection, incompatible associations,
    /// bounds or batch failures, recorded exclusion failures, or conflicting protected allocations.
    fn new(
        engine: &'a WriteEngine,
        storage: &'a mut OriginalStorage,
        pending: &'a mut PendingRootFrees,
        exclusion: &Result<(), StorageConflict>,
        decisions: MemberDecisions,
    ) -> Result<Self, ParentPreparationError> {
        storage
            .exclude_space(engine)
            .map_err(ParentPreparationError::Storage)?;
        if let Ok(parents) = &storage.parents {
            for (&address, parent) in parents {
                for range in [
                    parent.framing.as_ref().ok().copied(),
                    parent.original.map(|(range, _)| range),
                ]
                .into_iter()
                .flatten()
                {
                    if matches!(
                        decisions.dispositions().get(&address),
                        None | Some(ParentDisposition::Retain)
                    ) {
                        Self::protect_enclosing(storage, address, range)?;
                    } else {
                        Self::protect(
                            storage,
                            range,
                            parent.original.and_then(|(_, class)| class),
                        )?;
                    }
                }
            }
        }
        let end = PhysicalAddress::new(engine.image.len());
        let mut released = BTreeSet::new();
        let mut parents = Vec::new();
        for (&address, disposition) in decisions.dispositions() {
            match disposition {
                ParentDisposition::Retain => {}
                ParentDisposition::Compact(plan) => {
                    Self::protect(storage, plan.range(), None)?;
                    if plan.range().end() > end {
                        return Err(ParentPreparationError::Bounds {
                            range: plan.range(),
                            end,
                        });
                    }
                }
                ParentDisposition::Retire(retirement) => {
                    if !retirement
                        .session()
                        .same(&engine.context.allocations.session())
                    {
                        return Err(ParentPreparationError::Session);
                    }
                    if Some(retirement.source()) != storage.source.as_ref() {
                        return Err(ParentPreparationError::Source {
                            expected: storage.source.clone(),
                            actual: Some(retirement.source().clone()),
                        });
                    }
                    if retirement.root() != pending.root {
                        return Err(ParentPreparationError::Root {
                            expected: pending.root,
                            actual: retirement.root(),
                        });
                    }
                    Self::protect(storage, retirement.range(), Some(retirement.class()))?;
                    released.insert(address);
                    parents.push((retirement.range(), retirement.class()));
                }
            }
        }
        let batch = if parents.is_empty() {
            None
        } else {
            Some(
                pending
                    .batch
                    .including(&parents, end)
                    .map_err(ParentPreparationError::Batch)?,
            )
        };
        if let Err(error) = exclusion {
            if !Self::released_conflict(error, &released, &parents) {
                return Err(ParentPreparationError::RecordedExclusion);
            }
        }
        let entries = batch.as_ref().unwrap_or(&pending.batch).entries();
        for &(range, class) in entries {
            let start = storage
                .allocations
                .partition_point(|allocation| allocation.range.end() <= range.start());
            if let Some(&allocation) = storage.allocations[start..]
                .iter()
                .take_while(|allocation| allocation.range.start() < range.end())
                .find(|allocation| {
                    !matches!(allocation.origin, StorageOrigin::Parent { address, .. } if released.contains(&address))
                })
            {
                return Err(ParentPreparationError::Storage(StorageConflict::Overlap {
                    allocation,
                    range,
                    claim: SpaceClaim::Retirement(class),
                }));
            }
        }
        Ok(Self {
            storage,
            pending,
            decisions,
            released,
            batch,
        })
    }

    /// Removes matching parent exclusions, installs the checked batch, and consumes permissions.
    fn apply(self) -> MemberDecisions<PreparedParentRetirement> {
        let Self {
            storage,
            pending,
            decisions,
            released,
            batch,
        } = self;
        if !released.is_empty() {
            storage.allocations.retain(|allocation| {
                !matches!(allocation.origin, StorageOrigin::Parent { address, .. } if released.contains(&address))
            });
        }
        if let Some(batch) = batch {
            pending.batch = batch;
        }
        decisions.map_retirements(PreparedParentRetirement::new)
    }

    /// Checks that protection for `address` contains `range`, independently of allocation class.
    ///
    /// # Errors
    ///
    /// Returns [`ParentPreparationError::Protection`] if an enclosing exclusion for this parent
    /// is absent.
    fn protect_enclosing(
        storage: &OriginalStorage,
        address: PhysicalAddress,
        range: FileRange,
    ) -> Result<(), ParentPreparationError> {
        let index = storage
            .allocations
            .partition_point(|allocation| allocation.range.end() <= range.start());
        let actual = storage
            .allocations
            .get(index)
            .filter(|allocation| allocation.range.overlaps(range))
            .copied();
        if actual.is_some_and(|allocation| {
            matches!(allocation.origin, StorageOrigin::Parent { address: actual, .. } if actual == address)
                && allocation.range.contains(range)
        }) {
            return Ok(());
        }
        Err(ParentPreparationError::Protection {
            range,
            class: None,
            actual,
        })
    }

    /// Checks an exact parent exclusion, including its class when `class` is supplied.
    ///
    /// # Errors
    ///
    /// Returns [`ParentPreparationError::Protection`] if the full matching exclusion is absent.
    fn protect(
        storage: &OriginalStorage,
        range: FileRange,
        class: Option<FreeClass>,
    ) -> Result<(), ParentPreparationError> {
        let actual = storage
            .allocations
            .binary_search_by_key(&range.start(), |allocation| allocation.range.start())
            .ok()
            .and_then(|index| storage.allocations.get(index))
            .copied();
        if actual.is_some_and(|allocation| {
            let StorageOrigin::Parent {
                address,
                class: actual,
            } = allocation.origin
            else {
                return false;
            };
            allocation.range == range
                && address == range.start()
                && class.is_none_or(|expected| actual == Some(expected))
        }) {
            return Ok(());
        }
        Err(ParentPreparationError::Protection {
            range,
            class,
            actual,
        })
    }

    /// Returns whether an exclusion is an exact retirement of a checked released parent.
    fn released_conflict(
        error: &StorageConflict,
        released: &BTreeSet<PhysicalAddress>,
        parents: &[(FileRange, FreeClass)],
    ) -> bool {
        match error {
            StorageConflict::Overlap {
                allocation:
                    StorageAllocation {
                        range: allocation,
                        origin: StorageOrigin::Parent { address, .. },
                    },
                range,
                claim: SpaceClaim::Retirement(class),
            } => {
                released.contains(address)
                    && allocation == range
                    && parents.contains(&(*range, *class))
            }
            _ => false,
        }
    }
}

/// A failed prerequisite for consuming parent retirements and releasing their exclusions.
#[derive(Debug)]
pub(in crate::edit::commit) enum ParentPreparationError {
    /// The engine's original root address could not be resolved.
    Address(FormatError),
    /// Allocation publication is unavailable for source-reference checks.
    Allocation(Error),
    /// The merged retirement batch failed complete-allocation checks.
    Batch(RetirementMergeError),
    /// A compacted parent extends beyond the current physical image.
    Bounds {
        /// The complete compaction allocation.
        range: FileRange,
        /// The physical image end.
        end: PhysicalAddress,
    },
    /// The engine's original root differs from the captured source object.
    OriginalRoot {
        /// The captured source object.
        expected: ObjectId,
        /// The engine's resolved root object.
        actual: ObjectId,
    },
    /// A parent exclusion is absent or has incompatible bounds, origin, or class.
    Protection {
        /// The range requiring protection.
        range: FileRange,
        /// The required allocation class, if any.
        class: Option<FreeClass>,
        /// The protected allocation at the required start or first overlap, if any.
        actual: Option<StorageAllocation>,
    },
    /// A recorded exclusion is unrelated to an exact released-parent retirement.
    RecordedExclusion,
    /// Prospective absence was established for another pending root.
    Root {
        /// The root awaiting publication.
        expected: PhysicalAddress,
        /// The root associated with the prospective evidence or retirement.
        actual: PhysicalAddress,
    },
    /// The prepared inputs and current engine have different allocation sessions.
    Session,
    /// An original source association is absent or differs across captured evidence.
    Source {
        /// The original storage association, absent if it could not be established.
        expected: Option<ReconstructionSource>,
        /// The association found in the other evidence, if available.
        actual: Option<ReconstructionSource>,
    },
    /// Allocator or pending-retirement exclusion checks failed.
    Storage(StorageConflict),
}
