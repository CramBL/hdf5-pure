//! Adapts HDF5 dataset types and filter messages to the chunk filter pipeline.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::num::NonZeroU32;

pub use hdf5_pure_filter::__private::FilterScratch;

#[cfg(test)]
use crate::FixedPointType;
use crate::error::FormatError;
use crate::filter_pipeline;
use crate::filter_pipeline::FilterPipeline;
use crate::scaleoffset::ScaleOffsetType;

mod zfp;

#[cfg(feature = "zfp")]
pub use zfp::{ZfpElementTypeWhenEnabled, zfp_element_type_from_datatype};

/// Carries a dataset's chunk shape and datatype facts to filter pipeline calls.
#[derive(Debug, Clone, Copy)]
pub struct ChunkContext<'a> {
    /// Chunk dimensions in elements (one per dataset rank).
    pub chunk_dims: &'a [u64],
    /// Size of one element in bytes (for shuffle's interleave width), proven
    /// non-zero: the chunk splitter clamps a row copy to an element boundary
    /// with `% element_size`, and the byte-oriented filters divide a buffer
    /// length by it. Carrying the proof here is what lets those sites skip a
    /// check of their own.
    pub element_size: NonZeroU32,
    /// Scalar type, required for type-aware filters like ZFP. `None` means
    /// the caller does not know or does not need it; type-aware filters
    /// will return an error.
    pub element_type: Option<ZfpElementTypeWhenEnabled>,
    /// Datatype facts the scale-offset encoder needs (class/sign/order).
    /// `None` for callers that don't have a `Datatype` or whose type isn't a
    /// scale-offset-compatible scalar; scale-offset writes then error.
    pub scale_offset_type: Option<ScaleOffsetType>,
}

/// An uninhabited scalar type used when ZFP support is disabled.
#[cfg(not(feature = "zfp"))]
pub type ZfpElementTypeWhenEnabled = core::convert::Infallible;

impl<'a> ChunkContext<'a> {
    /// Creates a test context without filter-specific scalar types.
    ///
    /// # Panics
    ///
    /// Panics if `element_size` is zero.
    #[cfg(test)]
    pub fn basic(chunk_dims: &'a [u64], element_size: u32) -> Self {
        Self {
            chunk_dims,
            element_size: NonZeroU32::new(element_size).expect("a test's element size is non-zero"),
            element_type: None,
            scale_offset_type: None,
        }
    }

    /// Derives the chunk context from a dataset's datatype.
    ///
    /// The element width and filter-specific scalar types come from the same datatype.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes per element.
    pub fn from_datatype(
        chunk_dims: &'a [u64],
        dt: &crate::datatype::Datatype,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            chunk_dims,
            element_size: dt.element_size()?,
            element_type: zfp_element_type_from_datatype(dt),
            scale_offset_type: crate::scaleoffset::scale_offset_type_from_datatype(dt),
        })
    }
}

/// Returns `None` when ZFP support is disabled.
#[cfg(not(feature = "zfp"))]
pub fn zfp_element_type_from_datatype(
    _: &crate::datatype::Datatype,
) -> Option<ZfpElementTypeWhenEnabled> {
    None
}

impl<'a> ChunkContext<'a> {
    fn filter_context(self) -> hdf5_pure_filter::__private::ChunkContext<'a> {
        hdf5_pure_filter::__private::ChunkContext {
            chunk_dims: self.chunk_dims,
            element_size: self.element_size,
            element_type: self.element_type,
            scale_offset_type: self.scale_offset_type,
        }
    }
}

/// Reverses the active filters on a stored chunk.
///
/// # Errors
///
/// Returns [`Error::UnsupportedFilter`](hdf5_pure_filter::__private::Error::UnsupportedFilter)
/// if an active filter has no decoder. Malformed filter data and output beyond a decoder's size
/// bound produce a filter-specific [`Error`](hdf5_pure_filter::__private::Error).
pub(crate) fn decompress_chunk(
    compressed: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, hdf5_pure_filter::__private::Error> {
    hdf5_pure_filter::__private::decompress_chunk(
        compressed,
        &filter_pipeline::FilterStepsRef::new(pipeline),
        ctx.filter_context(),
        filter_mask,
    )
}

/// Reverses the active filters on a stored chunk using reusable scratch state.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedFilter`] if an active filter has no decoder.
/// Malformed filter data produces a filter-specific [`FormatError`].
pub fn decompress_chunk_with(
    scratch: &mut FilterScratch,
    compressed: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, FormatError> {
    hdf5_pure_filter::__private::decompress_chunk_with(
        scratch,
        compressed,
        &filter_pipeline::FilterStepsRef::new(pipeline),
        ctx.filter_context(),
        filter_mask,
    )
    .map_err(FormatError::from)
}

/// Applies a test pipeline to a chunk with fresh scratch state.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedFilter`] if a filter has no encoder.
/// Invalid parameters produce a filter-specific [`FormatError`].
#[cfg(test)]
pub fn compress_chunk(
    data: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    hdf5_pure_filter::__private::compress_chunk_with(
        &mut FilterScratch::new(),
        data,
        &filter_pipeline::FilterStepsRef::new(pipeline),
        ctx.filter_context(),
    )
    .map_err(FormatError::from)
}

/// Applies the filters to a chunk using reusable scratch state.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedFilter`] if a filter has no encoder.
/// Invalid parameters produce a filter-specific [`FormatError`].
pub fn compress_chunk_with(
    scratch: &mut FilterScratch,
    data: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    hdf5_pure_filter::__private::compress_chunk_with(
        scratch,
        data,
        &filter_pipeline::FilterStepsRef::new(pipeline),
        ctx.filter_context(),
    )
    .map_err(FormatError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Datatype;
    use crate::datatype::byte_order::DatatypeByteOrder;

    #[test]
    fn a_context_cannot_be_built_from_a_zero_width_datatype() {
        let degenerate = Datatype::Array {
            base_type: Box::new(Datatype::FixedPoint(
                FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
            )),
            dimensions: vec![0],
        };
        assert_eq!(
            ChunkContext::from_datatype(&[4], &degenerate).unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 10 }
        );

        let ordinary = crate::datatype::Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        );
        let ctx = ChunkContext::from_datatype(&[4], &ordinary).unwrap();
        assert_eq!(ctx.element_size.get(), 4);
    }

    #[rstest::rstest]
    #[case::deflate(hdf5_pure_filter::__private::Error::DeflateSizeExceeded { limit: 16, required: 17 }, "deflate: output exceeds expected chunk size of 16 bytes (possible decompression bomb)")]
    #[case::lzf(hdf5_pure_filter::__private::Error::LzfSizeExceeded { limit: 16, required: 17 }, "lzf: output exceeds expected chunk size")]
    fn output_size_limits_preserve_legacy_filter_errors(
        #[case] error: hdf5_pure_filter::__private::Error,
        #[case] expected: &str,
    ) {
        assert_eq!(
            FormatError::from(error),
            FormatError::FilterError(expected.into())
        );
    }

    #[test]
    fn a_failed_filter_decode_preserves_the_public_error() {
        let lzf = hdf5_pure_filter::__private::decompress_lzf(&[0x1f], None).unwrap_err();
        let err = FormatError::from(lzf);
        let FormatError::FilterError(reason) = err else {
            panic!("expected FilterError, got {err:?}");
        };
        assert_eq!(reason, "lzf: truncated literal run");
    }
}
