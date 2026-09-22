//! Helpers shared by the crate's integration tests and by the crosscheck
//! package, and the path to the committed test data.

use std::path::{Path, PathBuf};

pub mod allocation;
pub mod bytes;
pub mod checksum;
pub mod extensible_array;
pub mod heap;
pub mod image;
pub mod object_header;
pub mod superblock;
pub mod temp;
pub mod userblock;
pub mod widths;

/// A file under `tests/data`, by its path there: `data("c/1.8/v1_superblock.h5")`.
pub fn data(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(relative)
}

#[track_caller]
pub(crate) fn read_file(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}
