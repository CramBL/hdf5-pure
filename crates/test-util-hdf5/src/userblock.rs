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

/// Writes the standard userblock edit fixture and returns its stamped userblock.
#[track_caller]
pub fn write_edit_fixture(path: &Path, size: usize, marker: &[u8]) -> Userblock {
    let mut builder = FileBuilder::new();
    builder.with_userblock(size as u64);
    builder
        .create_dataset("alpha")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    builder.create_dataset("beta").with_i32_data(&[10, 20, 30]);
    let mut group = builder.create_group("grp");
    group.create_dataset("inner").with_f64_data(&[7.5, 8.5]);
    builder.add_group(group.finish());
    write(path, builder, size, marker)
}
