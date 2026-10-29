//! Object-header addresses read through libhdf5.

use std::ffi::CString;

use hdf5_sys::h5o::H5O_INFO_BASIC;
use hdf5_sys::h5o::H5O_info1_t;
use hdf5_sys::h5p::H5P_DEFAULT;

/// Returns the address of the object header of the object at `name`.
///
/// # Panics
///
/// Panics if `name` contains a NUL byte or `H5Oget_info_by_name2` fails.
pub fn header_address(file: &hdf5::File, name: &str) -> u64 {
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
