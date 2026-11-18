//! Byte observations over a source phase's physical residual intervals.
//!
//! A nonzero byte or failed read prevents classifying an interval as all zero.

use hdf5_pure_core::__private::FileRange;

use crate::convert::Narrow;
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::error::{Error, FormatError};
use crate::source::{PhysicalSource, Source};

impl AnalysisPhase<'_> {
    /// Returns cached content observations for this phase's physical residual intervals.
    ///
    /// Obtains intervals through [`Self::physical_coverage`] and examines their bytes on the first
    /// query. Retains read failures per interval and examines later intervals independently. The
    /// caller establishes source consistency for this phase.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Self::physical_coverage`] and [`ResidualContents::read`]. Returns
    /// [`Error::Format`] with [`FormatError::Internal`] if the cached observations cannot be
    /// retrieved.
    pub(in crate::edit) fn residual_content(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<&ResidualContents, Error> {
        self.physical_coverage(source)?;
        let engine = self.engine;
        let coverage = self
            .claims
            .as_mut()
            .and_then(|claims| claims.coverage.as_mut())
            .ok_or_else(|| FormatError::Internal("physical coverage was not retained".into()))?;
        if coverage.content.is_none() {
            let opened = engine.opened_source();
            let contents =
                ResidualContents::read(&PhysicalSource::new(&opened), &coverage.residuals)?;
            return Ok(coverage.content.insert(contents));
        }
        coverage
            .content
            .as_ref()
            .ok_or_else(|| FormatError::Internal("residual content was not retained".into()).into())
    }
}

/// Content observations for the original residual intervals of one source phase.
///
/// Allocation ownership and reuse permission require separate evidence.
pub(in crate::edit) struct ResidualContents {
    intervals: Vec<ResidualContent>,
}

impl ResidualContents {
    /// Examines the supplied intervals in order with one shared read buffer.
    ///
    /// Records source failures within their intervals and continues with later intervals.
    ///
    /// # Performance
    ///
    /// Uses one stack buffer of [`READ_BUFFER_LEN`] bytes. For `U` compared bytes and `G` intervals,
    /// byte comparison and interval bookkeeping take `O(U + G)` work. Retains `O(G)` results.
    /// Source buffering and mirroring have separate storage costs.
    ///
    /// # Errors
    ///
    /// Returns errors from [`ResidualContent::read`].
    fn read(
        source: &PhysicalSource<'_, impl Source + ?Sized>,
        ranges: &[FileRange],
    ) -> Result<Self, FormatError> {
        let mut scratch = [0_u8; READ_BUFFER_LEN];
        let intervals = ranges
            .iter()
            .map(|&range| ResidualContent::read(source, range, &mut scratch))
            .collect::<Result<_, _>>()?;
        Ok(Self { intervals })
    }

    /// Borrows content observations in the supplied interval order.
    pub(in crate::edit) fn intervals(&self) -> &[ResidualContent] {
        &self.intervals
    }
}

/// An original physical interval and its examination result.
pub(in crate::edit) struct ResidualContent {
    range: FileRange,
    outcome: ResidualOutcome,
}

impl ResidualContent {
    /// Examines `range` through its first nonzero byte or failed read.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a position exceeds [`u64::MAX`], or
    /// [`FormatError::ValueTooLargeForPlatform`] if a read length exceeds [`usize::MAX`].
    fn read(
        source: &PhysicalSource<'_, impl Source + ?Sized>,
        range: FileRange,
        scratch: &mut [u8; READ_BUFFER_LEN],
    ) -> Result<Self, FormatError> {
        let mut cursor = range.start();
        let outcome = loop {
            if cursor == range.end() {
                break ResidualOutcome::AllZero;
            }
            let length = (range.end().get() - cursor.get())
                .min(scratch.len().to_u64())
                .to_usize()?;
            let buffer = &mut scratch[..length];
            if source.read_at(cursor, buffer).is_err() {
                break ResidualOutcome::ReadFailed;
            }
            if buffer.iter().any(|&byte| byte != 0) {
                break ResidualOutcome::NonZero;
            }
            cursor = cursor.checked_add(length.to_u64())?;
        };
        Ok(Self { range, outcome })
    }

    /// Returns the original physical interval.
    pub(in crate::edit) fn range(&self) -> FileRange {
        self.range
    }

    /// Borrows the reason examination ended for this interval.
    pub(in crate::edit) fn outcome(&self) -> &ResidualOutcome {
        &self.outcome
    }
}

/// Completion or stopping condition for examination of one residual interval.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum ResidualOutcome {
    /// Every byte was successfully read and compared equal to zero.
    AllZero,
    /// A nonzero byte was encountered.
    NonZero,
    /// A source read failed.
    ReadFailed,
}

/// Length of the stack buffer shared by residual-content reads.
const READ_BUFFER_LEN: usize = 8_192;

#[cfg(test)]
mod tests;
