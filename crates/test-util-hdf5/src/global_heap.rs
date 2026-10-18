//! Global heap fixtures created through the reference HDF5 C library.

use std::path::Path;

use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;

/// Creates a file at `path` through libhdf5, with 8-byte addresses, lengths `sizeof_size` wide,
/// and a userblock of `userblock` bytes.
///
/// # Panics
///
/// Panics if libhdf5 cannot create `path` with 8-byte addresses, `sizeof_size`-wide lengths,
/// and a `userblock`-byte userblock.
pub fn create_libhdf5_file(path: &Path, sizeof_size: Sizeof, userblock: u64) -> hdf5::File {
    hdf5::File::with_options()
        .with_create_plist(|p| {
            p.sizes(SizeofInfo {
                sizeof_addr: Sizeof::Bytes8,
                sizeof_size,
            })
            .userblock(userblock)
        })
        .create(path)
        .unwrap()
}

/// Returns the [`STRING_COUNT`] strings the tests write, each 11 bytes long.
pub fn strings() -> Vec<String> {
    (0..STRING_COUNT)
        .map(|i| format!("string {i:04}"))
        .collect()
}

/// Number of strings written by the global heap fixtures.
pub const STRING_COUNT: usize = 300;
