use hdf5_pure_filter::__private as filters;
use test_util::filtered_chunks::PipelineFilter;

/// Filter metadata with Deflate level 6 and four-byte elements.
pub(super) struct Steps<'a>(pub(super) &'a [PipelineFilter]);

impl filters::FilterSteps for Steps<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn id(&self, index: usize) -> u16 {
        match self.0[index] {
            PipelineFilter::Deflate => filters::FILTER_DEFLATE,
            PipelineFilter::Fletcher32 => filters::FILTER_FLETCHER32,
            PipelineFilter::Lzf => filters::FILTER_LZF,
            PipelineFilter::Shuffle => filters::FILTER_SHUFFLE,
        }
    }
    fn flags(&self, _index: usize) -> u16 {
        0
    }
    fn client_data(&self, index: usize) -> &[u32] {
        match self.0[index] {
            PipelineFilter::Deflate => &[6],
            PipelineFilter::Shuffle => &[4],
            PipelineFilter::Fletcher32 | PipelineFilter::Lzf => &[],
        }
    }
}
