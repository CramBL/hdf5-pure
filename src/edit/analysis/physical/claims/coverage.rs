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

/// Normalized claim intervals, source gaps, and independent bounds observations.
///
/// Intervals are ordered and disjoint. Adjacent claims share one covered interval. Claim ends
/// beyond the physical source remain in the covered intervals. Residual intervals lie within the
/// physical source and describe bytes outside every recorded claim, regardless of support problems.
/// A residual interval is not evidence that its bytes are free.
pub(in crate::edit) struct PhysicalCoverage {
    covered: Vec<FileRange>,
    residuals: Vec<FileRange>,
    /// One pair of conflicting indices into the original claim slice.
    conflict: Option<[usize; 2]>,
    /// Original claim indices whose ends exceed the physical source length.
    beyond_source: Vec<usize>,
    /// Original claim indices whose ends exceed the recorded end of allocation.
    beyond_eoa: Vec<usize>,
    /// Byte observations collected lazily over the residual intervals.
    content: Option<ResidualContents>,
}

impl PhysicalCoverage {
    /// Normalizes intervals and records one conflicting pair of claim indices.
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
            covered: Vec::new(),
            residuals: Vec::new(),
            conflict: None,
            beyond_source: Vec::new(),
            beyond_eoa: Vec::new(),
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
        for index in ordered {
            let claim = claims[index];
            let range = claim.range();
            if range.end().get() > physical_size {
                coverage.beyond_source.push(index);
            }
            if range.end().get() > logical_eoa {
                coverage.beyond_eoa.push(index);
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
            if let Some(previous) = coverage.covered.last_mut()
                && range.start() <= previous.end()
            {
                *previous = FileRange::new(
                    previous.start(),
                    previous.end().max(range.end()).get() - previous.start().get(),
                )?;
            } else {
                coverage.covered.push(range);
            }
        }
        let mut cursor = 0;
        for range in &coverage.covered {
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

    /// Borrows the ordered union of nonempty recorded ranges, including out-of-source claims.
    pub(in crate::edit) fn covered(&self) -> &[FileRange] {
        &self.covered
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

    /// Borrows original claim indices whose ends exceed the physical source length.
    pub(in crate::edit) fn beyond_source(&self) -> &[usize] {
        &self.beyond_source
    }

    /// Borrows original claim indices whose ends exceed the recorded end of allocation.
    pub(in crate::edit) fn beyond_eoa(&self) -> &[usize] {
        &self.beyond_eoa
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
