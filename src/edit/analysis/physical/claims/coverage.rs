//! Coverage and overlap observations for a source phase's physical claims.
//!
//! Normalized intervals describe the union of recorded claims. Source completeness, allocation
//! ownership, and permission to reuse storage require separate evidence.

use core::cmp::Reverse;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

use super::{ClaimOrigin, FileStorageKind, PhysicalClaim};
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::error::{Error, FormatError};

use self::content::ResidualContents;

pub(in crate::edit) mod content;

impl AnalysisPhase<'_> {
    /// Returns cached coverage and overlap observations for this phase's claims.
    ///
    /// Collects claims through [`Self::physical_claims`] on the first query. Constructs coverage
    /// without further source reads and retains the original claims and their support problems.
    /// The caller establishes source consistency separately.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Self::physical_claims`]. Returns [`Error::Format`] with
    /// [`FormatError::OffsetOverflow`] if an interval end exceeds [`u64::MAX`], or
    /// [`FormatError::Internal`] if a cache invariant fails.
    pub(in crate::edit) fn physical_coverage(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<&PhysicalCoverage, Error> {
        self.physical_claims(source)?;
        let claims = self
            .claims
            .as_mut()
            .ok_or_else(|| FormatError::Internal("physical claims were not retained".into()))?;
        if claims.coverage.is_none() {
            claims.coverage = Some(PhysicalCoverage::new(
                &claims.claims,
                claims.physical_size,
                claims.logical_eoa,
            )?);
        }
        claims.coverage.as_ref().ok_or_else(|| {
            FormatError::Internal("physical coverage was not retained".into()).into()
        })
    }
}

/// Disjoint residual intervals and witnesses of conflicting or out-of-bounds claims.
///
/// Residual intervals lie within the physical source and describe bytes outside every
/// recorded claim. Reuse requires independent evidence.
pub(in crate::edit) struct PhysicalCoverage {
    residuals: Vec<FileRange>,
    /// One pair of conflicting indices into the original claim slice.
    conflict: Option<[usize; 2]>,
    /// First original claim index in sweep order beyond the physical source.
    beyond_source: Option<usize>,
    /// First original claim index in sweep order beyond the recorded end of allocation.
    beyond_eoa: Option<usize>,
    /// Byte observations collected lazily over the residual intervals.
    content: Option<ResidualContents>,
}

impl PhysicalCoverage {
    /// Normalizes intervals and records the first witnesses of overlap and bounds violations.
    ///
    /// Permits extension or manager observations contained in a recorded metadata capacity.
    /// Overlapping observations of other kinds, including duplicate manager observations, conflict.
    /// Sorts claim indices and sweeps the intervals with linear scratch space.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if an interval end exceeds [`u64::MAX`].
    fn new(
        claims: &[PhysicalClaim],
        physical_size: u64,
        logical_eoa: u64,
    ) -> Result<Self, FormatError> {
        let mut coverage = Self {
            residuals: Vec::new(),
            conflict: None,
            beyond_source: None,
            beyond_eoa: None,
            content: None,
        };
        let mut ordered = (0..claims.len()).collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|&index| {
            let claim = claims[index];
            (
                claim.range().start(),
                Reverse(claim.range().end()),
                claim.origin().category(),
                index,
            )
        });
        let mut furthest: [Option<usize>; 3] = [None; 3];
        let mut covered: Vec<FileRange> = Vec::new();
        for index in ordered {
            let claim = claims[index];
            let range = claim.range();
            if range.end().get() > physical_size {
                coverage.beyond_source.get_or_insert(index);
            }
            if range.end().get() > logical_eoa {
                coverage.beyond_eoa.get_or_insert(index);
            }
            if range.length() == 0 {
                continue;
            }
            if coverage.conflict.is_none() {
                for previous in furthest.into_iter().flatten() {
                    let other = claims[previous];
                    if range.overlaps(other.range()) && !claim.metadata_alias(other) {
                        coverage.conflict = Some([previous, index]);
                        break;
                    }
                }
            }
            let slot = &mut furthest[claim.origin().category()];
            if slot.is_none_or(|previous| claims[previous].range().end() < range.end()) {
                *slot = Some(index);
            }
            if let Some(previous) = covered.last_mut()
                && range.start() <= previous.end()
            {
                *previous = FileRange::new(
                    previous.start(),
                    previous.end().max(range.end()).get() - previous.start().get(),
                )?;
            } else {
                covered.push(range);
            }
        }
        let mut cursor = 0;
        for range in &covered {
            let start = range.start().get().min(physical_size);
            if cursor < start {
                coverage.residuals.push(FileRange::new(
                    PhysicalAddress::new(cursor),
                    start - cursor,
                )?);
            }
            cursor = range.end().get().min(physical_size);
        }
        if cursor < physical_size {
            coverage.residuals.push(FileRange::new(
                PhysicalAddress::new(cursor),
                physical_size - cursor,
            )?);
        }
        Ok(coverage)
    }

    /// Borrows physical-source intervals outside the recorded union.
    pub(in crate::edit) fn residuals(&self) -> &[FileRange] {
        &self.residuals
    }

    /// Returns one conflicting pair of original claim indices, if observed.
    ///
    /// A contained extension or manager observation can alias its recorded metadata capacity.
    /// Absence of a conflict establishes neither complete source ownership nor reuse permission.
    pub(in crate::edit) fn conflict(&self) -> Option<[usize; 2]> {
        self.conflict
    }

    /// Returns the first original claim index in sweep order beyond the physical source.
    pub(in crate::edit) fn beyond_source(&self) -> Option<usize> {
        self.beyond_source
    }

    /// Returns the first original claim index in sweep order beyond the recorded allocation end.
    pub(in crate::edit) fn beyond_eoa(&self) -> Option<usize> {
        self.beyond_eoa
    }
}

impl PhysicalClaim {
    fn metadata_alias(self, other: Self) -> bool {
        let (capacity, observation) = match (self.origin().category(), other.origin().category()) {
            (0, 1) => (self, other),
            (1, 0) => (other, self),
            _ => return false,
        };
        capacity.range().contains(observation.range())
    }
}

impl ClaimOrigin {
    fn category(self) -> usize {
        match self {
            Self::MetadataCapacity(_) => 0,
            Self::File {
                kind: FileStorageKind::Extension | FileStorageKind::Manager,
                ..
            } => 1,
            _ => 2,
        }
    }
}

#[cfg(test)]
mod tests;
