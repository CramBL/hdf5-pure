//! Object-header addresses read through libhdf5.

#![cfg(feature = "__hdf5-1.10")]

use std::ffi::CString;

use hdf5_sys::h5o::H5O_INFO_BASIC;
use hdf5_sys::h5o::H5O_info1_t;
use hdf5_sys::h5p::H5P_DEFAULT;

/// Returns the address of the object header of the object at `name`.
///
/// `H5Oget_info_by_name2` reports the address, and its successor in 1.12, `H5Oget_info_by_name3`,
/// reports an object token in its place.
pub(crate) fn header_address(file: &hdf5::File, name: &str) -> u64 {
    let name = CString::new(name).unwrap();
    let mut info = H5O_info1_t::default();
    let status = hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access, `file` keeps the file open, `name` is a
        // NUL-terminated string that outlives the call, and `info` is a writable `H5O_info1_t`.
        unsafe {
            hdf5_sys::h5o::H5Oget_info_by_name2(
                file.id(),
                name.as_ptr(),
                &raw mut info,
                H5O_INFO_BASIC,
                H5P_DEFAULT,
            )
        }
    });
    assert!(status >= 0, "H5Oget_info_by_name2 failed for {name:?}");
    info.addr
}
