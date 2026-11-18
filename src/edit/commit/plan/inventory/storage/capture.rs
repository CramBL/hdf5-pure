//! Conservative source protection and independent integrity observations.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;

use super::super::OriginalEdits;
use super::super::dispositions::Disposition;
use super::super::dispositions::ReconstructionDispositions;
use super::OriginalStorage;
use super::StorageAllocation;
use super::StorageOrigin;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::OriginalOwnerInventory;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::qualification::ParentStorage;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::analysis::physical::claims::ClaimProblem;
use crate::edit::analysis::physical::claims::PhysicalClaims;
use crate::edit::analysis::physical::claims::domain::residual::ResidualGroups;
use crate::edit::commit::plan::Node;
use crate::edit::dataset::MovingWrite;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

impl OriginalStorage {
    /// Protects observed heap collections and retained object claims in claim order.
    ///
    /// Uses [`ClaimReleases`] for object-claim retention. Records heap claims with an unknown
    /// allocation class and object claims with their observed owner and kind.
    pub(super) fn capture_claims(&mut self, claims: &PhysicalClaims, releases: &ClaimReleases<'_>) {
        for claim in claims.claims() {
            let (object, kind) = match claim.origin() {
                ClaimOrigin::Heap(address) => {
                    self.allocations.push(StorageAllocation {
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
            if releases.retains(object, claim.range()) {
                self.allocations.push(StorageAllocation {
                    range: claim.range(),
                    origin: StorageOrigin::Owner { object, kind },
                });
            }
        }
    }

    /// Protects the enclosing range of each parent's original allocation and parsed framing.
    ///
    /// Retains one allocation per parent address and sorts protected ranges by physical address.
    pub(super) fn capture_parents(&mut self) {
        let Self {
            source: _,
            allocations,
            parents,
            integrity,
            source_error,
        } = self;
        allocations.sort_unstable_by_key(|allocation| allocation.range.start());
        let Ok(parents) = parents else {
            return;
        };
        for (&address, parent) in parents {
            let range = match Self::parent_range(parent) {
                Ok(Some(range)) => range,
                Ok(None) => continue,
                Err(error) => {
                    *integrity = Err(StorageIntegrityError::Unavailable);
                    *source_error = Some(error.into());
                    continue;
                }
            };
            let class = parent.original.and_then(|(_, class)| class);
            if let Some(allocation) = allocations.iter_mut().find(|allocation| matches!(allocation.origin, StorageOrigin::Parent { address: actual, .. } if actual == address)) {
                allocation.range = range;
                allocation.origin = StorageOrigin::Parent { address, class };
            } else {
                allocations.push(StorageAllocation { range, origin: StorageOrigin::Parent { address, class } });
            }
        }
        allocations.sort_unstable_by_key(|allocation| allocation.range.start());
    }

    /// Records source and protection geometry before protecting unobserved intervals.
    ///
    /// Preserves an existing integrity error. Protects residual intervals only when owner
    /// qualification is unavailable and the recorded geometry is valid. Source-observation errors
    /// take precedence over qualification failures.
    pub(super) fn capture_integrity(
        &mut self,
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        source: &ReconstructionSource,
        retain_unobserved: bool,
    ) {
        if self.integrity.is_err() {
            return;
        }
        match self.source_integrity(engine, phase, source, retain_unobserved) {
            Ok((residuals, end)) => {
                if self.integrity.is_err() {
                    return;
                }
                self.integrity = self.protected_geometry(end);
                if self.integrity.is_err() {
                    return;
                }
                if retain_unobserved {
                    if let Err(error) = self.capture_unobserved(&residuals) {
                        self.integrity = Err(StorageIntegrityError::Unavailable);
                        self.source_error = Some(error.into());
                    }
                }
            }
            Err(error) => {
                self.integrity = Err(StorageIntegrityError::Unavailable);
                self.source_error = Some(error);
            }
        }
    }

    /// Records claim and object geometry failures and returns residuals and a physical end.
    ///
    /// Returns the lesser of the source and logical allocation ends. Collects residuals only when
    /// owner qualification is unavailable.
    ///
    /// # Errors
    ///
    /// Returns source-phase association and collection errors from the phase queries, or
    /// [`FormatError::Internal`] if a coverage claim is absent from the recorded claims.
    fn source_integrity(
        &mut self,
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        source: &ReconstructionSource,
        retain_unobserved: bool,
    ) -> Result<(Vec<FileRange>, PhysicalAddress), Error> {
        let coverage = phase.physical_coverage(source)?;
        let conflict = coverage.conflict();
        let beyond_source = coverage.beyond_source();
        let beyond_allocation = coverage.beyond_eoa();
        let residuals = if retain_unobserved {
            coverage.residuals().to_vec()
        } else {
            Vec::new()
        };
        let claims = phase.physical_claims(source)?;
        let claim = |index: usize| {
            claims.claims().get(index).copied().ok_or_else(|| {
                FormatError::Internal(
                    "a coverage claim is absent from its source observations".into(),
                )
            })
        };
        self.integrity = if let Some([first, second]) = conflict {
            Err(StorageIntegrityError::ConflictingClaims {
                first: claim(first)?.range(),
                second: claim(second)?.range(),
            })
        } else if let Some(index) = beyond_source {
            Err(StorageIntegrityError::BeyondSource {
                range: claim(index)?.range(),
                end: PhysicalAddress::new(claims.physical_size()),
            })
        } else if let Some(index) = beyond_allocation {
            Err(StorageIntegrityError::BeyondAllocation {
                range: claim(index)?.range(),
                end: PhysicalAddress::new(claims.logical_eoa()),
            })
        } else if claims.logical_eoa() > claims.physical_size() {
            Err(StorageIntegrityError::TruncatedSource {
                end: PhysicalAddress::new(claims.physical_size()),
                required: PhysicalAddress::new(claims.logical_eoa()),
            })
        } else {
            Ok(())
        };
        let end = PhysicalAddress::new(claims.physical_size().min(claims.logical_eoa()));
        if self.integrity.is_ok() {
            let objects = claims
                .problems()
                .iter()
                .filter_map(|problem| match problem {
                    ClaimProblem::ObjectStorage(object) => Some(*object),
                    _ => None,
                })
                .collect::<Vec<_>>();
            self.capture_object_geometry(engine, phase, &objects)?;
        }
        Ok((residuals, end))
    }

    /// Records source bounds, overflow, and local extent failures for incomplete object claims.
    ///
    /// # Errors
    ///
    /// Returns association errors from [`AnalysisPhase::reconstruction_source`] and
    /// [`AnalysisPhase::object_ownership`].
    fn capture_object_geometry(
        &mut self,
        engine: &WriteEngine,
        phase: &mut AnalysisPhase<'_>,
        objects: &[ObjectId],
    ) -> Result<(), FormatError> {
        for &object in objects {
            let origin = phase.reconstruction_source(engine, object)?;
            let ownership = phase.object_ownership(&origin)?;
            match ownership.error() {
                Some(Error::Format(
                    error @ (FormatError::UnexpectedSourceEof { .. }
                    | FormatError::OffsetOverflow { .. }),
                )) => {
                    self.integrity = Err(StorageIntegrityError::ObjectFormat {
                        error: Box::new(error.clone()),
                    });
                    break;
                }
                Some(Error::InvalidSpaceOwnershipExtent {
                    address,
                    length,
                    previous_end,
                    logical_size,
                }) => {
                    self.integrity = Err(StorageIntegrityError::ObjectExtent {
                        address: *address,
                        length: *length,
                        previous_end: *previous_end,
                        logical_size: *logical_size,
                    });
                    break;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Checks protection ranges sorted by start address for overlap and bounds.
    ///
    /// # Errors
    ///
    /// Returns [`StorageIntegrityError::ProtectionBounds`] if a range extends past `end`, or
    /// [`StorageIntegrityError::ConflictingProtection`] if adjacent ranges overlap.
    fn protected_geometry(&self, end: PhysicalAddress) -> Result<(), StorageIntegrityError> {
        let mut previous: Option<StorageAllocation> = None;
        for &allocation in &self.allocations {
            if allocation.range.end() > end {
                return Err(StorageIntegrityError::ProtectionBounds {
                    range: allocation.range,
                    end,
                });
            }
            if let Some(first) = previous.replace(allocation) {
                if first.range.overlaps(allocation.range) {
                    return Err(StorageIntegrityError::ConflictingProtection {
                        first: first.range,
                        second: allocation.range,
                    });
                }
            }
        }
        Ok(())
    }

    /// Returns the enclosing range of the known original allocation and parsed framing.
    ///
    /// Returns [`None`] if both observations are unavailable.
    ///
    /// # Errors
    ///
    /// Returns the range-construction errors of [`FileRange::new`].
    fn parent_range(parent: &ParentStorage) -> Result<Option<FileRange>, FormatError> {
        let framing = parent.framing;
        match (framing, parent.original) {
            (Some(framing), Some((original, _))) => {
                let start = framing.start().min(original.start());
                let end = framing.end().max(original.end());
                FileRange::new(start, end.get() - start.get()).map(Some)
            }
            (Some(range), None) | (None, Some((range, _))) => Ok(Some(range)),
            (None, None) => Ok(None),
        }
    }

    /// Protects residual source intervals outside the recorded protection ranges.
    ///
    /// Requires disjoint recorded ranges sorted by start address. Preserves the original ranges
    /// and sorts the combined exclusions by start address.
    ///
    /// # Errors
    ///
    /// Returns the range-construction errors of [`FileRange::new`].
    fn capture_unobserved(&mut self, residuals: &[FileRange]) -> Result<(), FormatError> {
        let mut unobserved = Vec::new();
        for &range in residuals {
            let mut cursor = range.start();
            for allocation in self
                .allocations
                .iter()
                .filter(|allocation| allocation.range.overlaps(range))
            {
                if cursor < allocation.range.start() {
                    Self::unobserved_interval(
                        &mut unobserved,
                        cursor,
                        allocation.range.start().min(range.end()),
                    )?;
                }
                cursor = cursor.max(allocation.range.end().min(range.end()));
            }
            Self::unobserved_interval(&mut unobserved, cursor, range.end())?;
        }
        self.allocations.extend(unobserved);
        self.allocations
            .sort_unstable_by_key(|allocation| allocation.range.start());
        Ok(())
    }

    fn unobserved_interval(
        allocations: &mut Vec<StorageAllocation>,
        start: PhysicalAddress,
        end: PhysicalAddress,
    ) -> Result<(), FormatError> {
        if start < end {
            allocations.push(StorageAllocation {
                range: FileRange::new(start, end.get() - start.get())?,
                origin: StorageOrigin::Unobserved,
            });
        }
        Ok(())
    }
}

/// Prepared owner and retirement inputs for original object-claim retention.
///
/// An object claim may be omitted only for an eligible or superseded owner with an exact
/// retirement or deferred-tail range and no reference from residual storage.
pub(super) struct ClaimReleases<'a> {
    /// Qualified original owners required for count-derived eligibility.
    owners: Option<&'a OriginalOwnerInventory>,
    /// Checked coverage of references from residual storage.
    residual: ResidualRelease<'a>,
    /// Original objects with a superseding predecessor disposition.
    superseded: BTreeSet<ObjectId>,
    /// Original objects selected for count-derived retirement.
    eligible: BTreeSet<ObjectId>,
    /// Successful reclamation entries, if supplied.
    retired: Option<&'a [(FileRange, FreeClass)]>,
    /// Complete original tail ranges recorded for appended-chunk replacement.
    tails: BTreeSet<(PhysicalAddress, PhysicalAddress)>,
}

impl<'a> ClaimReleases<'a> {
    /// Borrows owner and retirement evidence and collects release identities.
    ///
    /// Collects deferred-tail ranges from appended-chunk writes in `groups`.
    /// The entries in a successful `reclamation` result must be sorted by physical start address.
    /// The caller retains any reclamation error.
    pub(super) fn new(
        groups: &BTreeMap<ObjectPathBuf, Node>,
        original: &'a OriginalEdits,
        dispositions: &ReconstructionDispositions,
        reclamation: &'a Result<Vec<(FileRange, FreeClass)>, Error>,
        residual: ResidualRelease<'a>,
    ) -> Self {
        Self {
            owners: original.owners(),
            residual,
            superseded: dispositions
                .predecessors()
                .values()
                .filter(|old| old.disposition == Disposition::Supersede)
                .map(|old| old.source.object())
                .collect(),
            eligible: original
                .facts()
                .into_iter()
                .flat_map(|facts| facts.eligible_owners())
                .copied()
                .collect(),
            retired: reclamation.as_ref().ok().map(Vec::as_slice),
            tails: groups
                .values()
                .flat_map(|node| &node.prepared.writes)
                .filter_map(|write| match &write.write {
                    MovingWrite::AppendedChunks {
                        old_tail_allocation: Some(allocation),
                        ..
                    } => Some((allocation.range().start(), allocation.range().end())),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Returns whether the object's exact range must remain protected.
    fn retains(&self, object: ObjectId, range: FileRange) -> bool {
        let Self {
            owners,
            residual,
            superseded,
            eligible,
            retired,
            tails,
        } = self;
        if residual.may_reference(object) {
            return true;
        }
        let eligible = owners.is_some() && eligible.contains(&object);
        let superseded = superseded.contains(&object);
        let retired = retired.is_some_and(|retired| {
            retired
                .binary_search_by_key(&range.start(), |(range, _)| range.start())
                .ok()
                .is_some_and(|index| retired[index].0 == range)
        }) || tails.contains(&(range.start(), range.end()));
        !((eligible || superseded) && retired)
    }
}

/// References from the original source's residual storage.
pub(super) enum ResidualRelease<'a> {
    /// Associated owner-domain evidence proves there are no residual objects.
    Absent,
    /// Complete descendants of associated residual groups.
    Complete(&'a ResidualGroups),
    /// Residual storage may reference an original object.
    Unavailable,
}

impl ResidualRelease<'_> {
    fn may_reference(&self, object: ObjectId) -> bool {
        match self {
            Self::Absent => false,
            Self::Complete(groups) => groups.objects().contains(&object),
            Self::Unavailable => true,
        }
    }
}

/// Invalid source or protected-range geometry, independent of reclamation qualification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::edit) enum StorageIntegrityError {
    /// A physical claim extends past the logical end of allocation.
    BeyondAllocation {
        range: FileRange,
        /// The logical end of allocation in physical coordinates.
        end: PhysicalAddress,
    },
    /// A physical claim extends past the source length.
    BeyondSource {
        range: FileRange,
        /// The physical source end.
        end: PhysicalAddress,
    },
    /// Two physical claims overlap.
    ConflictingClaims { first: FileRange, second: FileRange },
    /// Two recorded protection ranges overlap.
    ConflictingProtection { first: FileRange, second: FileRange },
    /// Local object allocations include an empty range or overlapping ranges.
    ObjectExtent {
        address: u64,
        length: u64,
        previous_end: u64,
        logical_size: u64,
    },
    /// Local object storage parsing failed on source bounds or address overflow.
    ObjectFormat { error: Box<FormatError> },
    /// A protection range extends past the source length or logical end of allocation.
    ProtectionBounds {
        range: FileRange,
        /// The lesser of the physical source end and logical end of allocation.
        end: PhysicalAddress,
    },
    /// The logical end of allocation exceeds the physical source end.
    TruncatedSource {
        /// The physical source end.
        end: PhysicalAddress,
        /// The logical end of allocation in physical coordinates.
        required: PhysicalAddress,
    },
    /// Required source observations or conservative protection ranges could not be collected.
    Unavailable,
}
