//! Helpers shared by the crate's integration tests and by the crosscheck
//! package, and the path to the committed test data.

use std::path::{Path, PathBuf};

pub mod allocation;
pub mod heap;
pub mod paged;
pub mod temp;

/// A file under `tests/data`, by its path there: `data("c/1.8/v1_superblock.h5")`.
pub fn data(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(relative)
}
