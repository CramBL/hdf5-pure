//! Filter pipeline fixtures and error messages used by the unit tests.
//!
//! libhdf5 is a dependency of the crosscheck package alone, so
//! a unit test that needs a C-written input reads a committed one under
//! `tests/data/c/`, written by `crates/crosscheck/tests/main/c_test_data.rs`. In the
//! other direction `src/repack/mod.rs` commits a file it writes through a
//! crate-private seam under `tests/data/pure/`, for that package to read with
//! the C library. `just test-data` rewrites both.

use crate::filter_pipeline::FilterDescription;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;

/// Set, a unit test that commits a file rewrites it instead of checking it.
pub const UPDATE: &str = "HDF5_PURE_UPDATE_TEST_DATA";

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
