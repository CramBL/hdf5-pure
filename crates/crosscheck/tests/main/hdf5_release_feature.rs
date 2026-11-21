#![cfg(feature = "libhdf5")]
//! Checks that the selected `__libhdf5-*` feature matches the linked release series.

/// The series the enabled features claim.
fn claimed() -> &'static str {
    if cfg!(feature = "__libhdf5-2") {
        "2"
    } else if cfg!(feature = "__libhdf5-1.14") {
        "1.14"
    } else if cfg!(feature = "__libhdf5-1.12") {
        "1.12"
    } else if cfg!(feature = "__libhdf5-1.10") {
        "1.10"
    } else {
        "1.8"
    }
}

/// The series of the linked library. Every 2.x release is one series.
fn linked() -> String {
    match libhdf5::library_version() {
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
        libhdf5::library_version()
    );
}
