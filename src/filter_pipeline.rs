use alloc::borrow::Cow;
#[cfg(not(feature = "std"))]
use alloc::format;

use hdf5_pure_format::__private::FilterPipelineError;

use crate::error::FormatError;

pub use hdf5_pure_filter::__private::FILTER_DEFLATE;
pub use hdf5_pure_filter::__private::FILTER_FLETCHER32;
pub use hdf5_pure_filter::__private::FILTER_LZF;
pub use hdf5_pure_filter::__private::FILTER_SCALEOFFSET;
pub use hdf5_pure_filter::__private::FILTER_SHUFFLE;
#[cfg(feature = "zfp")]
pub use hdf5_pure_filter::__private::FILTER_ZFP;
pub use hdf5_pure_filter::__private::H5Z_FLAG_OPTIONAL;
pub use hdf5_pure_format::__private::FilterDescription;
pub use hdf5_pure_format::__private::FilterPipeline;

pub(crate) fn parse_filter_pipeline(data: &[u8]) -> Result<FilterPipeline, FormatError> {
    FilterPipeline::parse(data).map_err(map_filter_pipeline_error)
}

/// An original Filter Pipeline body checked by the parser and retained for verbatim emission.
///
/// Keeps any bytes after the last declared filter.
#[derive(Clone, Debug)]
pub(crate) struct RetainedFilterPipeline<'a>(Cow<'a, [u8]>);

impl<'a> RetainedFilterPipeline<'a> {
    /// Reads the pipeline descriptions and retains their complete source body.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`parse_filter_pipeline`].
    #[cfg(any(feature = "std", test))]
    pub(crate) fn parse(body: Cow<'a, [u8]>) -> Result<(FilterPipeline, Self), FormatError> {
        let pipeline = parse_filter_pipeline(&body)?;
        Ok((pipeline, Self(body)))
    }

    /// Returns an owned body, copying the bytes if they are borrowed.
    #[cfg(feature = "std")]
    pub(crate) fn into_owned(self) -> RetainedFilterPipeline<'static> {
        RetainedFilterPipeline(Cow::Owned(self.0.into_owned()))
    }

    /// Returns the original body bytes.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

pub(crate) fn map_filter_pipeline_error(error: FilterPipelineError) -> FormatError {
    match error {
        FilterPipelineError::Format(error) => error,
        FilterPipelineError::InvalidName { filter_id, reason } => {
            FormatError::FilterError(format!("invalid name for filter {filter_id}: {reason}"))
        }
        FilterPipelineError::FieldTooLarge {
            field,
            value,
            maximum,
        } => FormatError::FilterError(format!(
            "filter pipeline {field} is {value}, above maximum {maximum}"
        )),
    }
}

/// Presents a pipeline's stored filter descriptions to the filter routines.
pub(crate) struct FilterStepsRef<'a>(&'a [FilterDescription]);

impl<'a> FilterStepsRef<'a> {
    pub(crate) fn new(pipeline: &'a FilterPipeline) -> Self {
        Self(pipeline.filters())
    }
}

impl hdf5_pure_filter::__private::FilterSteps for FilterStepsRef<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn id(&self, index: usize) -> u16 {
        self.0[index].filter_id
    }

    fn flags(&self, index: usize) -> u16 {
        self.0[index].flags
    }

    fn client_data(&self, index: usize) -> &[u32] {
        &self.0[index].client_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::invalid_name(
        FilterPipelineError::InvalidName { filter_id: 300, reason: "invalid UTF-8" },
        "invalid name for filter 300: invalid UTF-8"
    )]
    #[case::filter_count(
        FilterPipelineError::FieldTooLarge { field: "filter count", value: 33, maximum: 32 },
        "filter pipeline filter count is 33, above maximum 32"
    )]
    fn filter_pipeline_validation_errors_map_to_filter_error(
        #[case] error: FilterPipelineError,
        #[case] message: &str,
    ) {
        assert_eq!(
            map_filter_pipeline_error(error),
            FormatError::FilterError(message.into())
        );
    }
}
