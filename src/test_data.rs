//! `tests/data/` for the unit tests: a path into it, and a copy of a file
//! there for a test that edits it.
//!
//! The reference C library is a dependency of the crosscheck package alone, so
//! a unit test that needs a C-written input reads a committed one under
//! `tests/data/c/`, written by `crates/crosscheck/tests/c_test_data.rs`. In the
//! other direction `src/repack.rs` commits a file it writes through a
//! crate-private seam under `tests/data/pure/`, for that package to read with
//! the C library. `just test-data` rewrites both.

use std::path::{Path, PathBuf};

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
