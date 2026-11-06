//! Byte observations over a source phase's physical residual intervals.
//!
//! Zero prefixes, nonzero witnesses, and read failures are recorded independently of allocation
//! ownership and permission to reuse storage.

use core::num::NonZeroU8;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

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
/// Original ranges, zero prefixes, and terminal outcomes are recorded per interval. Allocation
/// ownership and reuse permission require separate evidence.
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
    /// byte comparison and interval bookkeeping take `O(U + G)` work. Retains `O(G)` observations
    /// plus storage for source errors. Source reads, buffering, and storage for mirrored images
    /// require separate accounting.
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

/// An original physical interval with its verified zero prefix and examination outcome.
///
/// Examination stops at the first nonzero byte or failed read. The remaining suffix can include
/// bytes fetched in the final read but left unexamined. Source request counts include failed
/// attempts. OS and device transfers require separate measurement.
pub(in crate::edit) struct ResidualContent {
    range: FileRange,
    zero_prefix: FileRange,
    unexamined: FileRange,
    outcome: ResidualOutcome,
    read_calls: u64,
    requested_bytes: u64,
}

impl ResidualContent {
    /// Examines `range` through its first nonzero byte or failed read.
    ///
    /// After a read failure, records only the zero prefix established by completed reads.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a position or interval end exceeds [`u64::MAX`],
    /// or [`FormatError::ValueTooLargeForPlatform`] if a local read length exceeds [`usize::MAX`].
    fn read(
        source: &PhysicalSource<'_, impl Source + ?Sized>,
        range: FileRange,
        scratch: &mut [u8; READ_BUFFER_LEN],
    ) -> Result<Self, FormatError> {
        let mut cursor = range.start();
        let mut read_calls = 0;
        let mut requested_bytes = 0;
        while cursor < range.end() {
            let length = (range.end().get() - cursor.get())
                .min(scratch.len().to_u64())
                .to_usize()?;
            let buffer = &mut scratch[..length];
            read_calls += 1;
            requested_bytes += length.to_u64();
            if let Err(error) = source.read_at(cursor, buffer) {
                return Self::finish(
                    range,
                    cursor,
                    cursor,
                    ResidualOutcome::ReadFailed(error),
                    read_calls,
                    requested_bytes,
                );
            }
            if let Some((index, value)) = buffer
                .iter()
                .enumerate()
                .find_map(|(index, &byte)| NonZeroU8::new(byte).map(|value| (index, value)))
            {
                let address = cursor.checked_add(index.to_u64())?;
                return Self::finish(
                    range,
                    address,
                    address.checked_add(1)?,
                    ResidualOutcome::NonZero { address, value },
                    read_calls,
                    requested_bytes,
                );
            }
            cursor = cursor.checked_add(length.to_u64())?;
        }
        Self::finish(
            range,
            cursor,
            cursor,
            ResidualOutcome::AllZero,
            read_calls,
            requested_bytes,
        )
    }

    /// Records the zero prefix and unexamined suffix within `range`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if an interval end exceeds [`u64::MAX`].
    fn finish(
        range: FileRange,
        zero_end: PhysicalAddress,
        remaining: PhysicalAddress,
        outcome: ResidualOutcome,
        read_calls: u64,
        requested_bytes: u64,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            range,
            zero_prefix: FileRange::new(range.start(), zero_end.get() - range.start().get())?,
            unexamined: FileRange::new(remaining, range.end().get() - remaining.get())?,
            outcome,
            read_calls,
            requested_bytes,
        })
    }

    /// Returns the original physical interval.
    pub(in crate::edit) fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the prefix established as zero by successful reads.
    pub(in crate::edit) fn zero_prefix(&self) -> FileRange {
        self.zero_prefix
    }

    /// Returns the suffix excluded from the zero prefix and nonzero witness.
    ///
    /// Bytes in this suffix can have been fetched in the final read.
    pub(in crate::edit) fn unexamined(&self) -> FileRange {
        self.unexamined
    }

    /// Borrows the reason examination ended for this interval.
    pub(in crate::edit) fn outcome(&self) -> &ResidualOutcome {
        &self.outcome
    }

    /// Returns the number of source read requests, including failed attempts.
    pub(in crate::edit) fn read_calls(&self) -> u64 {
        self.read_calls
    }

    /// Returns the total buffer length requested from the source, including failed attempts.
    pub(in crate::edit) fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }
}

/// Completion or stopping condition for examination of one residual interval.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum ResidualOutcome {
    /// Every byte was successfully read and compared equal to zero.
    AllZero,
    /// The first nonzero byte, with its physical address.
    NonZero {
        address: PhysicalAddress,
        value: NonZeroU8,
    },
    /// Source read failure after any completed zero batches.
    ReadFailed(FormatError),
}

/// Length of the stack buffer shared by residual-content reads.
const READ_BUFFER_LEN: usize = 8_192;

#[cfg(test)]
mod tests;
