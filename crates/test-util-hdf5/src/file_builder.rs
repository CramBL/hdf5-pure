//! File-builder fixtures reopened through `hdf5-pure`'s reader.

use std::path::Path;

use hdf5_pure::{File, FileBuilder};

/// Builds a file with `build`, serializes it, and opens the serialized bytes through the reader.
#[track_caller]
pub fn build_and_open(build: impl FnOnce(&mut FileBuilder)) -> File {
    let mut builder = FileBuilder::new();
    build(&mut builder);
    let bytes = builder
        .finish()
        .unwrap_or_else(|e| panic!("finish test file: {e}"));
    File::from_bytes(bytes).unwrap_or_else(|e| panic!("open serialized test file: {e}"))
}

/// Writes a one-dimensional `i32` dataset to `path`.
#[track_caller]
pub fn write_i32_dataset(path: &Path, dataset_name: &str, data: &[i32]) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset(dataset_name)
        .with_i32_data(data)
        .with_shape(&[data.len() as u64]);
    builder
        .write(path)
        .unwrap_or_else(|e| panic!("write test file {path:?}: {e}"));
}
