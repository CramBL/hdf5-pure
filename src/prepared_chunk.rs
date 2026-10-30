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
use crate::chunked_write::ChunkTraversal;
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
    /// Encodes the whole chunks of a dataset in dense row-major order.
    ///
    /// The encoder borrows complete contiguous input from `raw` for filtered chunks or gathers
    /// into one reusable full-chunk buffer. For chunks stored without filtering, the writer
    /// reuses gathered buffers as the stored bytes. Encoded chunks remain in the returned
    /// collection for layout planning and emission.
    ///
    /// The elements of a chunk past `shape` take `fill`. Where `flags` has
    /// [`DONT_FILTER_PARTIAL_BOUND_CHUNKS`] set, a chunk that extends past `shape` is stored as its
    /// data. Every other chunk is stored as the output of `pipeline`, or as its data where
    /// `pipeline` is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if a chunk is larger than [`u32::MAX`] bytes,
    /// the non-scalar geometry is invalid, or a derived count, stride, or offset exceeds
    /// [`u64::MAX`]. Returns [`FormatError::ValueTooLargeForPlatform`] if an element size, chunk
    /// count, or chunk byte length exceeds [`usize::MAX`]. Returns [`FormatError::Internal`] if
    /// a scalar input has a length other than the full logical chunk length.
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
        if shape.is_empty() {
            return Ok(vec![Self::encode(
                scratch,
                pipeline,
                ctx,
                ChunkEncoding::Filtered,
                raw.to_vec(),
            )?]);
        }
        let mut traversal = ChunkTraversal::new(shape, ctx.chunk_dims, element_size)?;
        let mut chunks = Vec::with_capacity(traversal.chunk_count().narrow()?);
        let mut gathered = Vec::new();
        traversal.visit(|chunk| {
            let encoding = ChunkEncoding::for_decoding(ChunkDecoding::at_extent(
                flags,
                ctx.chunk_dims,
                shape,
                chunk.offsets(),
            ));
            let prepared = match (encoding, pipeline) {
                (ChunkEncoding::Filtered, Some(pipeline)) => {
                    let logical = match chunk.contiguous(raw)? {
                        Some(logical) => logical,
                        None => {
                            gathered.resize(chunk.byte_len(), 0);
                            chunk.gather(raw, fill, &mut gathered)?;
                            &gathered
                        }
                    };
                    Self::encode_filtered(scratch, pipeline, ctx, logical)?
                }
                _ => {
                    let mut logical = vec![0; chunk.byte_len()];
                    chunk.gather(raw, fill, &mut logical)?;
                    Self::encode(scratch, pipeline, ctx, encoding, logical)?
                }
            };
            chunks.push(prepared);
            Ok(())
        })?;
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
        if let (ChunkEncoding::Filtered, Some(pipeline)) = (encoding, pipeline) {
            return Self::encode_filtered(scratch, pipeline, ctx, &logical);
        }
        Self::check_logical_length(ctx, logical.len())?;
        let filtering = match encoding {
            ChunkEncoding::Filtered => ChunkFiltering::Filtered { filter_mask: 0 },
            ChunkEncoding::FiltersSkipped => ChunkFiltering::Filtered {
                filter_mask: every_filter_skipped(pipeline)?,
            },
            ChunkEncoding::Raw => ChunkFiltering::Raw,
        };
        Ok(Self {
            bytes: logical,
            filtering,
        })
    }

    /// Encodes borrowed full-chunk bytes through `pipeline`, recording a filter mask of zero.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `logical` is not the full chunk length of `ctx`.
    /// Propagates chunk-size errors from [`chunked_read::bounded_chunk_len`] and filter errors
    /// from [`filters::compress_chunk_with`].
    pub(crate) fn encode_filtered(
        scratch: &mut FilterScratch,
        pipeline: &FilterPipeline,
        ctx: ChunkContext<'_>,
        logical: &[u8],
    ) -> Result<Self, FormatError> {
        Self::check_logical_length(ctx, logical.len())?;
        Ok(Self {
            bytes: filters::compress_chunk_with(scratch, logical, pipeline, ctx)?,
            filtering: ChunkFiltering::Filtered { filter_mask: 0 },
        })
    }

    /// Checks that `actual` equals the full logical chunk length of `ctx`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the lengths differ. Propagates chunk-size errors
    /// from [`chunked_read::bounded_chunk_len`].
    fn check_logical_length(ctx: ChunkContext<'_>, actual: usize) -> Result<(), FormatError> {
        let expected = chunked_read::bounded_chunk_len(ctx.chunk_dims, ctx.element_size.narrow()?)?;
        if actual != expected {
            return Err(FormatError::Internal(format!(
                "a chunk prepared for storage holds {actual} of its {expected} logical bytes"
            )));
        }
        Ok(())
    }

    /// Returns the bytes stored for the chunk.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the allocated byte capacity retained by this chunk.
    #[cfg(all(
        test,
        feature = "std",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    pub(crate) fn storage_capacity(&self) -> usize {
        self.bytes.capacity()
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
    #[case::no_filters(0, 0)]
    #[case::one_filter(1, 0b1)]
    #[case::two_filters(2, 0b11)]
    #[case::mask_width(32, u32::MAX)]
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

    #[rstest]
    #[case::raw(ChunkEncoding::Raw, false, 0)]
    #[case::skipped(ChunkEncoding::FiltersSkipped, true, 1)]
    #[case::unfiltered(ChunkEncoding::Filtered, false, 0)]
    fn raw_encoding_keeps_the_owned_buffer(
        #[case] encoding: ChunkEncoding,
        #[case] with_pipeline: bool,
        #[case] mask: u32,
    ) {
        let logical = vec![7; 16];
        let pointer = logical.as_ptr();
        let pipeline = with_pipeline.then(|| shuffles(1));
        let chunk = PreparedChunk::encode(
            &mut FilterScratch::new(),
            pipeline.as_ref(),
            ChunkContext::basic(&[4], 4),
            encoding,
            logical,
        )
        .unwrap();
        assert_eq!(chunk.bytes().as_ptr(), pointer);
        assert_eq!(chunk.bytes(), &[7; 16]);
        assert_eq!(chunk.filter_mask(), mask);
    }

    #[rstest]
    #[case::short(12)]
    #[case::long(20)]
    fn borrowed_encoding_rejects_a_logical_chunk_of_another_length(#[case] len: usize) {
        let error = PreparedChunk::encode_filtered(
            &mut FilterScratch::new(),
            &shuffles(1),
            ChunkContext::basic(&[4], 4),
            &vec![7; len],
        )
        .unwrap_err();
        assert_eq!(
            error,
            FormatError::Internal(format!(
                "a chunk prepared for storage holds {len} of its 16 logical bytes"
            ))
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
