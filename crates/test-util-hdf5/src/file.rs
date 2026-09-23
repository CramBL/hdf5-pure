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
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()))
        .create(path)
        .unwrap_or_else(|e| panic!("create {path:?}: {e}"))
}
