//! Files the reference C library creates under a chosen low bound on the format version.

#[cfg(feature = "__hdf5-1.10")]
use std::path::Path;

#[cfg(feature = "__hdf5-1.10")]
use hdf5::file::LibraryVersion;

/// A file the C library creates in the 1.8 format or newer: version 2 object
/// headers and link-message groups. HDF5 2.0 made that the default, and every
/// earlier release writes version 1 headers by default.
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_create_v18(path: &Path) -> hdf5::File {
    create_bounded(path, LibraryVersion::V18)
}

/// A file the C library creates in the 1.10 format or newer, which indexes the chunks of a
/// dataset with one unlimited dimension with an Extensible Array.
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_create_v110(path: &Path) -> hdf5::File {
    create_bounded(path, LibraryVersion::V110)
}

#[cfg(feature = "__hdf5-1.10")]
#[track_caller]
fn create_bounded(path: &Path, low: LibraryVersion) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, LibraryVersion::latest()))
        .create(path)
        .unwrap_or_else(|e| panic!("create {path:?}: {e}"))
}
