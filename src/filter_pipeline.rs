use alloc::borrow::Cow;

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

/// Reads the pipeline body and returns its checked descriptions for reuse.
///
/// # Errors
///
/// Returns [`FormatError::InvalidFilterPipelineField`] if the filter count exceeds 32 or
/// [`FormatError::InvalidFilterName`] if a stored name is invalid. Propagates the other format
/// errors from [`FilterPipeline::parse`].
pub(crate) fn parse_filter_pipeline(data: &[u8]) -> Result<FilterPipeline, FormatError> {
    FilterPipeline::parse(data).map_err(FormatError::from)
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
    use alloc::vec;

    use test_util::filter_pipeline::{Pipeline, Version};

    use super::*;

    #[rstest::rstest]
    #[case::version_one(Version::One)]
    #[case::version_two(Version::Two)]
    fn retained_pipeline_keeps_the_source_suffix(#[case] version: Version) {
        let mut fixture = Pipeline::new(version);
        fixture.push(300, Some("custom"), 0x80, &[7, 13]).unwrap();
        let body = fixture.build();
        let mut source = body.clone();
        source.extend_from_slice(&[0xa5; 5]);
        let (pipeline, retained) = RetainedFilterPipeline::parse(Cow::Borrowed(&source)).unwrap();

        assert_eq!(pipeline.version(), version.byte());
        assert_eq!(
            pipeline.filters(),
            &[FilterDescription {
                filter_id: 300,
                name: Some("custom".into()),
                flags: 0x80,
                client_data: vec![7, 13],
            }]
        );
        assert_eq!(pipeline.to_bytes(), body);
        assert_eq!(retained.as_bytes(), source);
    }
}
