//! Userblock fixture creation and relocation.

use std::path::Path;

use hdf5_pure::UserblockSize;
use hdf5_pure::{BaseAddress, File, FileBuilder};
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_format::__private as format;
use test_util::superblock;

pub use test_util::userblock::Userblock;

/// Adds `userblock` before a fixture whose stored base address is zero.
///
/// Rewrites the superblock's base and end-of-file addresses and checksum, preserving stored
/// object addresses. The fixture must begin with its superblock signature.
///
/// # Examples
///
/// ```
/// # use hdf5_pure::{File, FileBuilder};
/// # use test_util_hdf5::userblock;
/// # let directory = tempfile::tempdir()?;
/// # let path = directory.path().join("fixture.h5");
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("values").with_i32_data(&[1, 2, 3]);
/// builder.write(&path)?;
/// let before = File::open(&path)?;
/// let stored_root = before.superblock().root_group_address;
/// let old_eof = before.superblock().eof_address;
/// drop(before);
///
/// userblock::prepend(&path, &[0xA5; 512]);
///
/// let after = File::open(&path)?;
/// assert_eq!(after.superblock().base_address.get(), 512);
/// assert_eq!(after.superblock().eof_address, old_eof + 512);
/// assert_eq!(after.superblock().root_group_address, stored_root);
/// assert_eq!(after.dataset("values")?.read_i32()?, [1, 2, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Panics
///
/// Panics if `userblock` is shorter than 512 bytes or its length is not a power of two, if the
/// fixture has a nonzero base address or signature offset, or if its framing cannot be read or
/// serialized. Also panics if an address or length overflows, or if reading or writing fails.
#[track_caller]
pub fn prepend(path: &Path, userblock: &[u8]) {
    assert!(userblock.len() >= MIN_USERBLOCK_LEN && userblock.len().is_power_of_two());
    let file = File::from_bytes(std::fs::read(path).unwrap()).unwrap();
    assert!(file.superblock().base_address.is_zero());
    assert!(file.as_bytes().starts_with(superblock::SIGNATURE));
    let mut superblock = file.superblock().clone();
    let size = u64::try_from(userblock.len()).unwrap();
    superblock.base_address = BaseAddress::new(size);
    superblock.eof_address = superblock.eof_address.checked_add(size).unwrap();
    let encoded = format::serialize_superblock(&superblock).unwrap();
    let mut bytes = Vec::with_capacity(userblock.len().checked_add(file.as_bytes().len()).unwrap());
    bytes.extend_from_slice(userblock);
    bytes.extend_from_slice(file.as_bytes());
    bytes[userblock.len()..userblock.len() + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(path, bytes).unwrap();
}

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
    builder.with_userblock(UserblockSize::try_from(size as u64).unwrap());
    builder
        .create_dataset("alpha")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    builder.create_dataset("beta").with_i32_data(&[10, 20, 30]);
    let mut group = builder.create_group("grp");
    group.create_dataset("inner").with_f64_data(&[7.5, 8.5]);
    builder.add_group(group.finish());
    write(path, builder, size, marker)
}

// A nonzero superblock offset is a power of two of at least 512 bytes, as defined in
// "Format Signature and Superblock" of the format specification, version 4.0.
const MIN_USERBLOCK_LEN: usize = 512;
