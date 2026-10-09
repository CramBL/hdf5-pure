//! File-space fixture builders shared across edit and interoperability tests.

use std::ops::Range;
use std::path::Path;

use hdf5_pure::{File, FileBuilder, FileSpaceStrategy, Layout};

use crate::dataset::Unlimited;

/// Number of `i32` elements in the middle dataset of the strategy-reuse fixture.
pub const STRATEGY_DELETED_ELEMENTS: usize = 400;

/// Byte length of the middle dataset in the strategy-reuse fixture.
pub const STRATEGY_DELETED_LEN: u64 =
    (STRATEGY_DELETED_ELEMENTS * std::mem::size_of::<i32>()) as u64;

/// Writes the `a`, `b`, `c` fixture used by file-space strategy reuse tests.
///
/// Dataset `b` is deliberately larger than its neighbors so deleting it leaves a
/// recognizable interior extent for the replacement-allocation assertions.
pub fn write_strategy_triplet(
    path: &Path,
    strategy: FileSpaceStrategy,
    persist: bool,
    threshold: u64,
) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder
        .create_dataset("b")
        .with_i32_data(&[2; STRATEGY_DELETED_ELEMENTS]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.with_file_space_strategy(strategy, persist, threshold);
    builder.write(path).unwrap();
}

/// Returns the allocated byte range of a contiguous dataset.
pub fn contiguous_extent(file: &File, path: &str) -> Range<u64> {
    let layout = file.dataset(path).unwrap().layout().unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = layout
    else {
        panic!("expected allocated contiguous storage, got {layout:?}");
    };
    address..address + size
}

#[cfg(feature = "__hdf5-1.10")]
/// Has libhdf5 write the same `a`, `b`, `c` strategy-reuse fixture.
pub fn libhdf5_write_strategy_triplet(
    path: &Path,
    strategy: hdf5::plist::file_create::FileSpaceStrategy,
) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| fcpl.file_space_strategy(strategy))
        .create(path)
        .unwrap();
    for (name, value, len) in [
        ("a", 1, 100),
        ("b", 2, STRATEGY_DELETED_ELEMENTS),
        ("c", 3, 100),
    ] {
        file.new_dataset::<i32>()
            .shape((len,))
            .create(name)
            .unwrap()
            .write(&vec![value; len])
            .unwrap();
    }
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
/// Returns the allocated byte range of a contiguous libhdf5 dataset.
pub fn libhdf5_contiguous_extent(dataset: &hdf5::Dataset) -> Range<u64> {
    let address = dataset.offset().unwrap();
    address..address + dataset.storage_size()
}

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
