//! Prepared bytes shared by throughput and instruction measurements.

use std::error::Error as StdError;
use std::num::NonZeroU32;

use hdf5_pure_filter::__private as filters;
use test_util::filtered_chunks::{self, Direction, PipelineCase};

use self::steps::Steps;

#[path = "steps.rs"]
mod steps;

/// A measurement case with checked raw and encoded bytes.
pub(super) struct PreparedPipeline {
    pub(super) case: PipelineCase,
    /// The unfiltered input, independently shuffled for compressor controls.
    pub(super) raw: Vec<u8>,
    pub(super) stored: Vec<u8>,
    dims: [u64; 1],
}

impl PreparedPipeline {
    /// Encodes the input and checks complete decoded bytes before measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if the element count exceeds the chunk dimension width or a filter fails.
    ///
    /// # Panics
    ///
    /// Panics if `raw` is empty or has incomplete four-byte elements, or if decoded bytes differ
    /// from the prepared input.
    #[expect(
        clippy::panic_in_result_fn,
        reason = "the fixture checks complete bytes before measurement"
    )]
    pub(super) fn new(case: PipelineCase, raw: &[u8]) -> Result<Self, Box<dyn StdError>> {
        assert!(!raw.is_empty() && raw.len().is_multiple_of(4));
        let raw = match case {
            PipelineCase::Deflate | PipelineCase::DeflateFletcher32 | PipelineCase::Lzf => {
                filtered_chunks::shuffled_pipeline_data(raw)
            }
            _ => raw.to_vec(),
        };
        let dims = [u64::try_from(raw.len() / 4)?];
        let stored = filters::compress_chunk_with(
            &mut filters::FilterScratch::new(),
            &raw,
            &Steps(case.filters()),
            context(&dims),
        )?;
        let input = Self {
            case,
            raw,
            stored,
            dims,
        };
        assert_eq!(
            input.execute(&mut filters::FilterScratch::new(), Direction::Decode)?,
            input.raw
        );
        Ok(input)
    }

    /// Runs the selected direction with the caller's scratch.
    ///
    /// # Errors
    ///
    /// Returns the filter error if execution fails.
    #[inline]
    pub(super) fn execute(
        &self,
        scratch: &mut filters::FilterScratch,
        direction: Direction,
    ) -> Result<Vec<u8>, filters::Error> {
        let Self {
            case,
            raw,
            stored,
            dims,
        } = self;
        let pipeline = Steps(case.filters());
        match direction {
            Direction::Encode => {
                filters::compress_chunk_with(scratch, raw, &pipeline, context(dims))
            }
            Direction::Decode => {
                filters::decompress_chunk_with(scratch, stored, &pipeline, context(dims), 0)
            }
        }
    }

    pub(super) fn expected(&self, direction: Direction) -> &[u8] {
        match direction {
            Direction::Encode => &self.stored,
            Direction::Decode => &self.raw,
        }
    }
}

fn context(dims: &[u64]) -> filters::ChunkContext<'_> {
    filters::ChunkContext {
        chunk_dims: dims,
        // Four is the nonzero width of every generated input element.
        element_size: NonZeroU32::new(4).expect("four is nonzero"),
        element_type: None,
        scale_offset_type: None,
    }
}
