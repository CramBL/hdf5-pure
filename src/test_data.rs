//! Filter pipeline fixtures and error messages used by the unit tests.
//!
//! Owner-local native crosschecks use the optional reference-library helpers in `test-util-hdf5`.
//! Tests without those capabilities use independent byte builders or committed fixtures.

use crate::filter_pipeline::FilterDescription;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;

/// Returns a version 2 filter pipeline of `filters`.
pub fn filter_pipeline(filters: Vec<FilterDescription>) -> FilterPipeline {
    FilterPipeline {
        version: PIPELINE_VERSION_TWO,
        filters,
    }
}

/// Returns a version 2 filter pipeline of one deflate filter at level 6.
pub fn deflate_pipeline() -> FilterPipeline {
    filter_pipeline(vec![FilterDescription {
        filter_id: crate::filter_pipeline::FILTER_DEFLATE,
        name: None,
        flags: 0,
        client_data: vec![6],
    }])
}

/// Returns a shuffle filter for elements `width` bytes wide.
pub fn shuffle(width: u32) -> FilterDescription {
    FilterDescription {
        filter_id: crate::filter_pipeline::FILTER_SHUFFLE,
        name: None,
        flags: 0,
        client_data: vec![width],
    }
}

/// Encodes `block` through `pipeline` as one chunk of one-byte elements.
pub fn encode_block(block: &[u8], pipeline: &FilterPipeline) -> Vec<u8> {
    crate::filters::compress_chunk(
        block,
        pipeline,
        ChunkContext::basic(&[u64::try_from(block.len()).unwrap()], 1),
    )
    .unwrap()
}

// "The Data Storage - Filter Pipeline Message" of the format specification, version 4.0, defines
// pipeline version 2.
const PIPELINE_VERSION_TWO: u8 = 2;
// The filter identifier bzip2 registers with The HDF Group, which this crate has no decoder for.
pub const FILTER_BZIP2: u16 = 307;
// The messages the deflate decoders of the `deflate` and `fast-deflate` features report for a
// stream without a zlib header.
pub const DEFLATE_HEADER_ERRORS: [&str; 2] = [
    "deflate: deflate decompression error",
    "deflate: deflate decompression error: incorrect header check",
];
