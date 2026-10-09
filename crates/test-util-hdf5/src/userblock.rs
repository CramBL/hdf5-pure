//! Userblock fixtures assembled through `hdf5-pure`'s writer.

use std::path::Path;

use hdf5_pure::FileBuilder;

pub use test_util::userblock::Userblock;

/// Finishes `builder`, stamps its userblock, writes it to `path`, and returns the stamped bytes.
#[track_caller]
pub fn write(path: &Path, builder: FileBuilder, size: usize, marker: &[u8]) -> Userblock {
    let mut bytes = builder
        .finish()
        .unwrap_or_else(|e| panic!("assemble {path:?}: {e}"));
    let userblock = Userblock::stamp(&mut bytes, size, marker);
    std::fs::write(path, &bytes).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    userblock
}
