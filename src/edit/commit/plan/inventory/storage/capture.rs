//! Conservative source protection and independent integrity observations.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

use super::OriginalStorage;
use super::StorageAllocation;
use super::StorageOrigin;
use super::StorageUnavailable;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::heap::qualification::ParentStorage;
use crate::edit::analysis::physical::claims::ClaimProblem;
use crate::edit::analysis::physical::claims::PhysicalClaim;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

impl OriginalStorage {
    /// Protects the enclosing range of each parent's original allocation and parsed framing.
    ///
    /// Records qualification failures independently of known ranges. Retains one allocation per
    /// parent address and sorts all protected ranges by physical start address.
    pub(super) fn capture_parents(&mut self) {
        let Self {
            source: _,
            allocations,
            parents,
            integrity,
            unavailable,
        } = self;
        allocations.sort_unstable_by_key(|allocation| allocation.range.start());
        let Ok(parents) = parents else {
            unavailable.get_or_insert(StorageUnavailable::Parents);
            return;
        };
        for (&address, parent) in parents {
            if let Err(reason) = parent.allocation {
                unavailable.get_or_insert(StorageUnavailable::Parent { address, reason });
            }
            let range = match Self::parent_range(parent) {
                Ok(Some(range)) => range,
                Ok(None) => continue,
                Err(error) => {
                    *integrity = Err(StorageIntegrityError::Unavailable);
                    *unavailable = Some(StorageUnavailable::Coverage(error.into()));
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
    ) {
        if self.integrity.is_err() {
            return;
        }
        match self.source_integrity(engine, phase, source) {
            Ok((residuals, end)) => {
                if self.integrity.is_err() {
                    return;
                }
                self.integrity = self.protected_geometry(end);
                if self.integrity.is_err() {
                    return;
                }
                if matches!(self.unavailable, Some(StorageUnavailable::Owners)) {
                    if let Err(error) = self.capture_unobserved(&residuals) {
                        self.integrity = Err(StorageIntegrityError::Unavailable);
                        self.unavailable = Some(StorageUnavailable::Coverage(error.into()));
                    }
                }
            }
            Err(error) => {
                self.integrity = Err(StorageIntegrityError::Unavailable);
                self.unavailable = Some(StorageUnavailable::Coverage(error));
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
    ) -> Result<(Vec<FileRange>, PhysicalAddress), Error> {
        let coverage = phase.physical_coverage(source)?;
        let conflict = coverage.conflict();
        let beyond_source = coverage.beyond_source().first().copied();
        let beyond_allocation = coverage.beyond_eoa().first().copied();
        let residuals = if matches!(self.unavailable, Some(StorageUnavailable::Owners)) {
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
                first: claim(first)?,
                second: claim(second)?,
            })
        } else if let Some(index) = beyond_source {
            Err(StorageIntegrityError::BeyondSource {
                claim: claim(index)?,
                end: PhysicalAddress::new(claims.physical_size()),
            })
        } else if let Some(index) = beyond_allocation {
            Err(StorageIntegrityError::BeyondAllocation {
                claim: claim(index)?,
                end: PhysicalAddress::new(claims.logical_eoa()),
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
                        object,
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
                        object,
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
                return Err(StorageIntegrityError::ProtectionBounds { allocation, end });
            }
            if let Some(first) = previous.replace(allocation) {
                if first.range.overlaps(allocation.range) {
                    return Err(StorageIntegrityError::ConflictingProtection {
                        first,
                        second: allocation,
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
        let framing = parent.framing.as_ref().ok().copied();
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

/// Invalid source or protected-range geometry, independent of reclamation qualification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::edit) enum StorageIntegrityError {
    /// A physical claim extends past the logical end of allocation.
    BeyondAllocation {
        claim: PhysicalClaim,
        /// The logical end of allocation in physical coordinates.
        end: PhysicalAddress,
    },
    /// A physical claim extends past the source length.
    BeyondSource {
        claim: PhysicalClaim,
        /// The physical source end.
        end: PhysicalAddress,
    },
    /// Two physical claims overlap.
    ConflictingClaims {
        first: PhysicalClaim,
        second: PhysicalClaim,
    },
    /// Two recorded protection ranges overlap.
    ConflictingProtection {
        first: StorageAllocation,
        second: StorageAllocation,
    },
    /// Local object allocations include an empty range or overlapping ranges.
    ObjectExtent {
        object: ObjectId,
        address: u64,
        length: u64,
        previous_end: u64,
        logical_size: u64,
    },
    /// Local object storage parsing failed on source bounds or address overflow.
    ObjectFormat {
        object: ObjectId,
        error: Box<FormatError>,
    },
    /// A protection range extends past the source length or logical end of allocation.
    ProtectionBounds {
        allocation: StorageAllocation,
        /// The lesser of the physical source end and logical end of allocation.
        end: PhysicalAddress,
    },
    /// Required source observations or conservative protection ranges could not be collected.
    Unavailable,
}

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    #[cfg(test)]
    mod native_crosschecks;
}

#[cfg(test)]
mod tests;
