// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! The `__hdf5-*` feature the suite was built with names the release series
//! of the library it links. The features are passed by hand, so this is the
//! one place the two are compared.

/// The series the enabled features claim.
fn claimed() -> &'static str {
    if cfg!(feature = "__hdf5-2") {
        "2"
    } else if cfg!(feature = "__hdf5-1.14") {
        "1.14"
    } else if cfg!(feature = "__hdf5-1.12") {
        "1.12"
    } else if cfg!(feature = "__hdf5-1.10") {
        "1.10"
    } else {
        "1.8"
    }
}

/// The series of the linked library. Every 2.x release is one series.
fn linked() -> String {
    match hdf5::library_version() {
        (2, _, _) => "2".to_string(),
        (major, minor, _) => format!("{major}.{minor}"),
    }
}

#[test]
fn the_release_feature_names_the_linked_library() {
    assert_eq!(
        claimed(),
        linked(),
        "the suite was built for HDF5 {} and links {:?}: pass the feature for the linked release",
        claimed(),
        hdf5::library_version()
    );
}
