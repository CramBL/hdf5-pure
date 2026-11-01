//! Helpers shared by the crate's integration tests and by the crosscheck
//! package, and the path to the committed test data.

#![cfg_attr(test, warn(clippy::tests_outside_test_module))]
use std::path::{Path, PathBuf};

pub mod allocation;
pub mod allocation_oracle;
pub mod attribute;
pub mod btree_v1;
pub mod btree_v2;
pub mod bytes;
pub mod checksum;
pub mod chunk_offsets;
pub mod data_layout;
pub mod dataspace;
pub mod datatype;
pub mod dense_attributes;
pub mod disk_log;
pub mod extensible_array;
pub mod file_space_info;
pub mod filter_pipeline;
pub mod filtered_chunks;
pub mod fractal_heap;
pub mod free_space;
pub mod free_space_damage;
pub mod fuzz;
pub mod generic_large;
pub mod global_heap;
pub mod group_info;
pub mod identity;
pub mod image;
pub mod legacy;
pub mod legacy_attributes;
pub mod legacy_chunked;
pub mod legacy_compound;
pub mod link_message;
pub mod local_heap;
pub mod mat_struct_array;
pub mod mcos;
pub mod modern;
pub mod native_complex;
pub mod object_header;
pub mod paged_index;
pub mod process;
pub mod publication;
pub mod publication_order;
pub mod range;
pub mod snapshot;
pub mod sohm;
pub mod superblock;
pub mod symbol_table;
pub mod temp;
pub mod userblock;
pub mod widths;

/// A file under `tests/data`, by its path there: `data("c/1.8/v1_superblock.h5")`.
pub fn data(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(relative)
}

/// Copies a committed fixture into `dir` for a test that modifies it.
///
/// The returned path has the same file name as the committed fixture.
///
/// # Panics
///
/// Panics if the committed fixture is missing, its path has no file name, or the copy fails.
pub fn copy_data(relative: &str, dir: &Path) -> PathBuf {
    let committed = data(relative);
    assert!(committed.is_file(), "{committed:?} is missing");
    let copy = dir.join(committed.file_name().unwrap());
    std::fs::copy(&committed, &copy).unwrap();
    copy
}

#[track_caller]
pub(crate) fn read_file(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}
