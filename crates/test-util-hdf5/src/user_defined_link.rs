//! Native construction of opaque user-defined links.

use std::ffi::{CStr, c_char, c_void};

use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5l::{self, H5L_class_t, H5L_type_t};
use hdf5_sys::h5p::H5P_DEFAULT;

/// Creates a kind-255 link with `value` as its opaque payload.
///
/// The link class is registered for the call and unregistered afterward. Traversal fails.
///
/// # Panics
///
/// Panics if a call to `H5Lregister`, `H5Lcreate_ud`, or `H5Lunregister` fails.
pub fn create(file: &hdf5::File, name: &CStr, value: &[u8]) {
    let class = H5L_class_t {
        version: h5l::H5L_LINK_CLASS_T_VERS.try_into().unwrap(),
        id: H5L_type_t::H5L_TYPE_MAX,
        trav_func: Some(traverse),
        ..H5L_class_t::default()
    };
    hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes the calls, the file identifier stays open, and
        // the class, null-terminated name and value buffer remain readable during each call.
        unsafe {
            assert!(h5l::H5Lregister(&class) >= 0);
            assert!(
                h5l::H5Lcreate_ud(
                    file.id(),
                    name.as_ptr(),
                    H5L_type_t::H5L_TYPE_MAX,
                    value.as_ptr().cast(),
                    value.len(),
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                ) >= 0
            );
            assert!(h5l::H5Lunregister(H5L_type_t::H5L_TYPE_MAX) >= 0);
        }
    });
}

extern "C" fn traverse(
    _link_name: *const c_char,
    _cur_group: hid_t,
    _data: *const c_void,
    _data_size: usize,
    _lapl: hid_t,
) -> hid_t {
    -1
}
