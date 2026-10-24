//! The chunks a dataset creation, an append, or a chunked overwrite writes, encoded for the extent
//! the write publishes and, in an in-place append, for the extent before it.

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_core::__private::PipelineFilterMask;

use crate::chunked_read;
use crate::chunked_read::ChunkDecoding;
use crate::chunked_read::ChunkFiltering;
use crate::chunked_write;
use crate::convert::Narrow;
use crate::data_layout::ChunkedLayoutFlags;
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;

/// A chunk ready to be written: its stored bytes, and whether a reader runs them through the
/// filter pipeline, with the mask of a filtered chunk.
#[derive(Debug)]
pub(crate) struct PreparedChunk {
    bytes: Vec<u8>,
    filtering: ChunkFiltering,
}

impl PreparedChunk {
    /// Splits `raw`, the data of a dataset of extent `shape`, into whole chunks of `ctx` and
    /// encodes each one for storage, in row-major grid order.
    ///
    /// The elements of a chunk past `shape` take `fill`. Where `flags` has
    /// [`DONT_FILTER_PARTIAL_BOUND_CHUNKS`] set, a chunk that extends past `shape` is stored as its
    /// data. Every other chunk is stored as the output of `pipeline`, or as its data where
    /// `pipeline` is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if a chunk is larger than [`u32::MAX`] bytes.
    /// Returns [`FormatError::UnreadableFillValue`] if a chunk extends past `shape` and `fill` is
    /// [`FillPattern::UNKNOWN`]. Returns the errors of [`filters::compress_chunk_with`] for a
    /// filtered chunk.
    ///
    /// [`DONT_FILTER_PARTIAL_BOUND_CHUNKS`]: crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS
    pub(crate) fn encode_dataset(
        scratch: &mut FilterScratch,
        pipeline: Option<&FilterPipeline>,
        ctx: ChunkContext<'_>,
        flags: ChunkedLayoutFlags,
        shape: &[u64],
        fill: FillPattern<'_>,
        raw: &[u8],
    ) -> Result<Vec<Self>, FormatError> {
        let element_size = ctx.element_size.narrow()?;
        chunked_read::bounded_chunk_len(ctx.chunk_dims, element_size)?;
        let split =
            chunked_write::split_into_chunks(raw, shape, ctx.chunk_dims, element_size, fill)?;
        let mut offsets = flags
            .partial_edge_chunks_stored_raw()
            .then(|| vec![0; shape.len()]);
        let mut chunks = Vec::with_capacity(split.len());
        for logical in split {
            let decoding = match offsets.as_deref_mut() {
                Some(offsets) => {
                    let decoding = ChunkDecoding::at_extent(flags, ctx.chunk_dims, shape, offsets);
                    advance_row_major(ctx.chunk_dims, shape, offsets);
                    decoding
                }
                None => ChunkDecoding::Pipeline,
            };
            chunks.push(Self::encode(
                scratch,
                pipeline,
                ctx,
                ChunkEncoding::for_decoding(decoding),
                logical,
            )?);
        }
        Ok(chunks)
    }

    /// Encodes `logical`, the data of a whole chunk, in the form of `encoding`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `logical` is not the size of a whole chunk of `ctx`.
    /// Returns [`FormatError::InvalidFilterPipelineField`] if `encoding` is
    /// [`ChunkEncoding::FiltersSkipped`] and `pipeline` has more than 32 filters.
    /// Returns the errors of [`chunked_read::bounded_chunk_len`] for the chunk size, and of
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
            (ChunkEncoding::FiltersSkipped, pipeline) => {
                let filter_mask = every_filter_skipped(pipeline)?;
                (logical, ChunkFiltering::Filtered { filter_mask })
            }
            (ChunkEncoding::Raw, _) => (logical, ChunkFiltering::Raw),
        };
        Ok(Self { bytes, filtering })
    }

    /// Returns the bytes stored for the chunk.
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

/// The form in which a chunk is written, and the filter mask its index record stores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkEncoding {
    /// The output of the filter pipeline, with a filter mask of 0.
    Filtered,
    /// The chunk's data, with a filter mask that skips every filter of the pipeline.
    FiltersSkipped,
    /// The chunk's data, with a filter mask of 0.
    Raw,
}

impl ChunkEncoding {
    /// Returns the encoding that a reader at the current extent and a reader at the published
    /// extent both decode to the chunk's data.
    ///
    /// `current` is how a reader at the extent before the append decodes the chunk, or `None` for
    /// a chunk past that extent, and `published` is how a reader at the extent the append
    /// publishes decodes it. Where the two readers differ, the chunk is stored as its data with
    /// every filter skipped, which both read as it is stored.
    pub(crate) fn decodable_by(current: Option<ChunkDecoding>, published: ChunkDecoding) -> Self {
        match (current, published) {
            (Some(ChunkDecoding::Verbatim), ChunkDecoding::Pipeline)
            | (Some(ChunkDecoding::Pipeline), ChunkDecoding::Verbatim) => Self::FiltersSkipped,
            (_, published) => Self::for_decoding(published),
        }
    }

    /// Returns the encoding that a reader which decodes the chunk as `decoding` decodes to the
    /// chunk's data.
    pub(crate) fn for_decoding(decoding: ChunkDecoding) -> Self {
        match decoding {
            ChunkDecoding::Pipeline => Self::Filtered,
            ChunkDecoding::Verbatim => Self::Raw,
        }
    }
}

/// Advances `offsets`, the element offsets of a chunk in a dataset of extent `shape`, to the next
/// chunk in row-major grid order, and back to the first chunk after the last.
fn advance_row_major(chunk_dims: &[u64], shape: &[u64], offsets: &mut [u64]) {
    for ((&chunk, &dim), offset) in chunk_dims.iter().zip(shape).zip(offsets).rev() {
        *offset = offset.saturating_add(chunk);
        if *offset < dim {
            return;
        }
        *offset = 0;
    }
}

/// Returns the filter mask that skips every filter of `pipeline`.
///
/// Bit `i` of a chunk's filter mask is set when the chunk skips filter `i` of the pipeline. The
/// mask is defined in "The Extensible Array Index" of the [format specification, version
/// 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::InvalidFilterPipelineField`] if `pipeline` has more than 32 filters.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
fn every_filter_skipped(pipeline: Option<&FilterPipeline>) -> Result<u32, FormatError> {
    PipelineFilterMask::all_skipped(pipeline.map_or(0, |pipeline| pipeline.filters.len()))
        .map(|mask| mask.recorded_mask().raw_bits())
}

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::FILTER_PIPELINE_VERSION_TWO;
    use rstest::rstest;

    use super::*;
    use crate::filter_pipeline::FILTER_SHUFFLE;
    use crate::filter_pipeline::FilterDescription;

    #[rstest]
    #[case(0, 0)]
    #[case(1, 0b1)]
    #[case(2, 0b11)]
    #[case(32, u32::MAX)]
    fn a_chunk_stored_with_every_filter_skipped_has_one_mask_bit_per_filter(
        #[case] filters: usize,
        #[case] filter_mask: u32,
    ) {
        let chunk = PreparedChunk::encode(
            &mut FilterScratch::new(),
            Some(&shuffles(filters)),
            ChunkContext::basic(&[4], 4),
            ChunkEncoding::FiltersSkipped,
            vec![7; 16],
        )
        .unwrap();
        assert_eq!(
            (chunk.bytes(), chunk.stored_size(), chunk.filter_mask()),
            (&[7; 16][..], 16, filter_mask)
        );
    }

    #[rstest]
    #[case::short_raw_chunk(
        ChunkEncoding::Raw,
        shuffles(1),
        12,
        "a chunk prepared for storage holds 12 of its 16 logical bytes"
    )]
    #[case::long_filtered_chunk(
        ChunkEncoding::Filtered,
        shuffles(1),
        20,
        "a chunk prepared for storage holds 20 of its 16 logical bytes"
    )]
    fn a_chunk_of_another_length_is_an_internal_error(
        #[case] encoding: ChunkEncoding,
        #[case] pipeline: FilterPipeline,
        #[case] logical_len: usize,
        #[case] reason: &str,
    ) {
        let err = PreparedChunk::encode(
            &mut FilterScratch::new(),
            Some(&pipeline),
            ChunkContext::basic(&[4], 4),
            encoding,
            vec![7; logical_len],
        )
        .unwrap_err();
        let FormatError::Internal(detail) = &err else {
            panic!("expected Internal, got {err:?}");
        };
        assert_eq!(detail, reason);
    }

    #[test]
    fn a_pipeline_larger_than_the_mask_is_rejected_before_storage() {
        let error = PreparedChunk::encode(
            &mut FilterScratch::new(),
            Some(&shuffles(33)),
            ChunkContext::basic(&[4], 4),
            ChunkEncoding::FiltersSkipped,
            vec![7; 16],
        )
        .unwrap_err();
        assert_eq!(
            error,
            FormatError::InvalidFilterPipelineField {
                field: "filter count",
                value: 33,
                maximum: 32,
            }
        );
    }

    fn shuffles(count: usize) -> FilterPipeline {
        FilterPipeline {
            version: FILTER_PIPELINE_VERSION_TWO,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: Vec::new(),
                };
                count
            ],
        }
    }
}
