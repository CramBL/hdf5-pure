//! Deflate configuration for group creation property lists.

#![cfg(feature = "__hdf5-1.10")]

use core::ffi::c_int;
use core::ffi::c_uint;

use hdf5::plist::GroupCreate;

/// Adds the deflate filter to `plist` at compression `level`.
///
/// Serializes the call with [`hdf5::sync::sync`] and returns the status from `H5Pset_deflate`,
/// nonnegative on success and negative on failure.
pub fn set_deflate(plist: &GroupCreate, level: u8) -> c_int {
    hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access, and `plist` keeps the property list live.
        // The call passes only integers and the library checks the compression level.
        unsafe { H5Pset_deflate(plist.id(), c_uint::from(level)) }
    })
}

// `H5Pset_deflate` in `H5Pocpl.c`, HDF5 1.14.6, takes a 64-bit property list ID and an unsigned level.
unsafe extern "C" {
    fn H5Pset_deflate(plist_id: i64, level: c_uint) -> c_int;
}
