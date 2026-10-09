//! File-space fixture builders shared across edit and interoperability tests.

use std::path::Path;

use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

use crate::dataset::Unlimited;

/// Writes the `a`, `big`, `c` fixture used to leave an interior hole.
///
/// When `persist` is true, the file uses a persisting `FsmAggr` strategy with a
/// threshold of one byte. Dataset `big` occupies 1600 raw bytes between two live
/// datasets, so deleting it leaves reusable interior space rather than a trailing
/// extent that can be truncated.
pub fn write_interior_hole_source(path: &Path, persist: bool) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("big").with_i32_data(&[7; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    if persist {
        builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    }
    builder.write(path).unwrap();
}

/// Writes a persisting interior-hole fixture and deletes `big` in a committed edit.
pub fn write_persisted_interior_hole(path: &Path) {
    write_interior_hole_source(path, true);
    let file = File::open_rw(path).unwrap();
    file.root().delete("big").unwrap();
    file.commit().unwrap();
}

/// Writes a persisting `FsmAggr` file with an unlimited rank-1 `i32` dataset `d`.
pub fn write_persisting_unlimited_i32(path: &Path, len: i32, chunk: u64, threshold: u64) {
    let data: Vec<i32> = (0..len).collect();
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, threshold);
    Unlimited::new("d", &data, chunk).add_to(&mut builder);
    builder.write(path).unwrap();
}
