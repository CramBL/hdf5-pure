use crate::chunked_read::{self, ChunkFiltering};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::filters;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;

/// A chunk ready for an in-place append: the bytes the append writes, and whether a reader runs
/// them through the filter pipeline, with the mask of a filtered chunk.
#[derive(Debug)]
pub(crate) struct PreparedChunk {
    bytes: Vec<u8>,
    filtering: ChunkFiltering,
}

impl PreparedChunk {
    /// Encodes `logical`, the data of a whole chunk, in the form of `encoding`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `logical` is not the size of a whole chunk of `ctx`,
    /// and the errors of [`chunked_read::bounded_chunk_len`] for that size. Returns the errors of
    /// [`filters::compress_chunk_with`] for a filtered chunk.
    pub(crate) fn encode(
        scratch: &mut FilterScratch,
        pipeline: Option<&FilterPipeline>,
        ctx: ChunkContext<'_>,
        encoding: ChunkEncoding,
        logical: Vec<u8>,
    ) -> Result<Self, FormatError> {
        let expected = chunked_read::bounded_chunk_len(ctx.chunk_dims, ctx.element_size.narrow()?)?;
        if logical.len() != expected {
            return Err(FormatError::Internal(format!(
                "a chunk prepared for storage holds {} of its {expected} logical bytes",
                logical.len()
            )));
        }
        let (bytes, filtering) = match (encoding, pipeline) {
            (ChunkEncoding::Filtered, Some(pipeline)) => (
                filters::compress_chunk_with(scratch, &logical, pipeline, ctx)?,
                ChunkFiltering::Filtered { filter_mask: 0 },
            ),
            (ChunkEncoding::Filtered, None) => {
                (logical, ChunkFiltering::Filtered { filter_mask: 0 })
            }
            (ChunkEncoding::Raw, _) => (logical, ChunkFiltering::Raw),
        };
        Ok(Self { bytes, filtering })
    }

    /// Returns the bytes the append writes for the chunk.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the length of [`bytes`](Self::bytes), the chunk size the index record stores.
    pub(crate) fn stored_size(&self) -> u64 {
        self.bytes.len().to_u64()
    }

    /// Returns the filter mask the index record stores.
    pub(crate) fn filter_mask(&self) -> u32 {
        match self.filtering {
            ChunkFiltering::Filtered { filter_mask } => filter_mask,
            // `H5D__chunk_flush_entry` in `H5Dchunk.c` (HDF5 1.14.6) records mask zero for a
            // chunk it stores with the filters disabled.
            ChunkFiltering::Raw => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkEncoding {
    Filtered,
    Raw,
}
