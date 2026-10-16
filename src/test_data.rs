//! `tests/data/` for the unit tests: a path into it, and a copy of a file
//! there for a test that edits it.
//!
//! The reference C library is a dependency of the crosscheck package alone, so
//! a unit test that needs a C-written input reads a committed one under
//! `tests/data/c/`, written by `crates/crosscheck/tests/main/c_test_data.rs`. In the
//! other direction `src/repack.rs` commits a file it writes through a
//! crate-private seam under `tests/data/pure/`, for that package to read with
//! the C library. `just test-data` rewrites both.

use std::path::{Path, PathBuf};

use crate::filter_pipeline::FilterDescription;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;

/// Set, a unit test that commits a file rewrites it instead of checking it.
pub const UPDATE: &str = "HDF5_PURE_UPDATE_TEST_DATA";

/// A file under `tests/data`, by its path there: `path("c/1.8/v1_superblock.h5")`.
pub fn path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(relative)
}

/// A copy of a committed file under `dir`, for a test that edits it.
pub fn copy(relative: &str, dir: &Path) -> PathBuf {
    let committed = path(relative);
    assert!(
        committed.is_file(),
        "{} is missing; `just test-data::c` writes it",
        committed.display()
    );
    let copy = dir.join(committed.file_name().unwrap());
    std::fs::copy(&committed, &copy).unwrap();
    copy
}

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
// "Fractal Heap" of the format specification, version 4.0, defines block version 0 and a direct
// block's layout: the heap header address after the signature and the version, then the block
// offset, then the checksum. The fixture heaps store eight-byte offsets and two-byte block offsets.
pub const BLOCK_VERSION: u8 = 0;
pub const DIRECT_BLOCK_OWNER_AT: usize = test_util::fractal_heap::DIRECT_BLOCK_SIGNATURE.len() + 1;
pub const DIRECT_BLOCK_CHECKSUM_AT: usize = DIRECT_BLOCK_OWNER_AT + 8 + 2;
pub const DIRECT_BLOCK_DATA_AT: usize = DIRECT_BLOCK_CHECKSUM_AT + 4;
