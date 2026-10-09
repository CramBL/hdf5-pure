//! File-builder fixtures reopened through `hdf5-pure`'s reader.

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
