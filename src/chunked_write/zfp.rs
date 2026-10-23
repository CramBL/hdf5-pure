//! ZFP filter metadata for chunked writes.

#![cfg(feature = "zfp")]

use hdf5_pure_filter::__private::ResolvedZfpSettings;

use crate::ZfpRate;
use crate::error::FormatError;
use crate::filter_pipeline::{FILTER_ZFP, FilterDescription};
use crate::filters::{ChunkContext, FilterScratch};

/// Builds client data and retains resolved settings when scratch state is supplied.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedZfp`] if the chunk's scalar type is missing or its
/// dimensions are unsupported. Returns the errors of [`ResolvedZfpSettings::new`] for rates
/// incompatible with the scalar type or block geometry and for chunk-size limits.
pub(super) fn filter_description(
    ctx: &ChunkContext<'_>,
    scratch: Option<&mut FilterScratch>,
    rate: ZfpRate,
    index: usize,
    flags: u16,
) -> Result<FilterDescription, FormatError> {
    let elem_ty = ctx.element_type.ok_or_else(|| {
        FormatError::UnsupportedZfp(
            "ZFP compression requires the dataset's datatype to be one \
         of f32, f64, i32, or i64"
                .into(),
        )
    })?;
    let settings =
        ResolvedZfpSettings::new(rate, elem_ty, ctx.chunk_dims).map_err(FormatError::from)?;
    let client_data = match scratch {
        Some(scratch) => scratch.zfp_client_data(settings, index),
        None => settings.client_data(),
    };
    Ok(FilterDescription {
        filter_id: FILTER_ZFP,
        name: Some("zfp".into()),
        flags,
        client_data,
    })
}
