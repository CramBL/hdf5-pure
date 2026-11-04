//! Files opened or created with libhdf5.

#[cfg(feature = "__hdf5-1.10")]
use core::num::NonZeroUsize;
use std::path::Path;

use hdf5::file::LibraryVersion;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5::plist::file_create::{Sizeof, SizeofInfo};
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileSpacePageSize;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::UserblockSize;
use hdf5_pure::{LengthWidth, OffsetWidth};

/// Opens an existing file for read-only native observations.
///
/// # Panics
///
/// Panics if libhdf5 cannot open `path`.
#[track_caller]
pub fn libhdf5_open(path: &Path) -> hdf5::File {
    hdf5::File::open(path).unwrap_or_else(|error| panic!("open {path:?}: {error}"))
}

/// Opens an existing file read-only with a native page-buffer capacity of `capacity` bytes.
///
/// # Panics
///
/// Panics if libhdf5 cannot open `path` with a page buffer of `capacity` bytes.
#[cfg(feature = "__hdf5-1.10")]
#[track_caller]
pub fn libhdf5_open_with_page_buffer(path: &Path, capacity: NonZeroUsize) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| properties.page_buffer_size(capacity.get(), 0, 0))
        .open(path)
        .unwrap_or_else(|error| panic!("open {path:?}: {error}"))
}

/// Creates a native PAGE fixture with threshold 1 and optional persistence.
///
/// # Panics
///
/// Panics if libhdf5 cannot create `path` with these settings.
#[cfg(feature = "__hdf5-1.10")]
#[track_caller]
pub fn libhdf5_create_page(
    path: &Path,
    page: FileSpacePageSize,
    userblock: UserblockSize,
    persist: bool,
) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties
                .userblock(userblock.get())
                .file_space_page_size(page.get())
                .file_space_strategy(NativeStrategy::FreeSpaceManager {
                    paged: true,
                    persist,
                    threshold: 1,
                })
        })
        .create(path)
        .unwrap_or_else(|error| panic!("create {path:?}: {error}"))
}

/// Opens an existing file for native read and write observations.
///
/// # Panics
///
/// Panics if libhdf5 cannot open `path` for writing.
#[track_caller]
pub fn libhdf5_open_rw(path: &Path) -> hdf5::File {
    hdf5::File::open_rw(path).unwrap_or_else(|error| panic!("open {path:?}: {error}"))
}

/// A file the C library creates in the 1.8 format or newer: version 2 object
/// headers and link-message groups. HDF5 2.0 made that the default, and every
/// earlier release writes version 1 headers by default.
pub fn libhdf5_create_v18(path: &Path) -> hdf5::File {
    create_bounded(path, LibraryVersion::V18)
}

/// A file the C library creates in the 1.10 format or newer, which indexes the chunks of a
/// dataset with one unlimited dimension with an Extensible Array.
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_create_v110(path: &Path) -> hdf5::File {
    create_bounded(path, LibraryVersion::V110)
}

/// A file the C library creates under a low bound of `H5F_LIBVER_EARLIEST`: a version 0
/// superblock and version 1 object headers, which no checksum covers.
pub fn libhdf5_create_earliest(path: &Path) -> hdf5::File {
    create_bounded(path, LibraryVersion::Earliest)
}

/// Creates a native file under HDF5 1.8 or newer format bounds.
///
/// Sets the address and length widths independently and reserves `userblock` bytes before the
/// superblock.
///
/// # Panics
///
/// Panics if libhdf5 rejects the creation properties or cannot create `path`.
pub fn libhdf5_create_with_widths(
    path: &Path,
    offsets: OffsetWidth,
    lengths: LengthWidth,
    userblock: u64,
) -> hdf5::File {
    create_with_widths(
        path,
        offsets,
        lengths,
        userblock,
        LibraryVersion::V18,
        LibraryVersion::latest(),
    )
}

/// Creates a native file with both format bounds set to HDF5 1.10.
///
/// Sets the address and length widths independently and reserves `userblock` bytes before the
/// superblock.
///
/// # Panics
///
/// Panics if libhdf5 rejects the creation properties or cannot create `path`.
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_create_v110_with_widths(
    path: &Path,
    offsets: OffsetWidth,
    lengths: LengthWidth,
    userblock: u64,
) -> hdf5::File {
    create_with_widths(
        path,
        offsets,
        lengths,
        userblock,
        LibraryVersion::V110,
        LibraryVersion::V110,
    )
}

fn create_with_widths(
    path: &Path,
    offsets: OffsetWidth,
    lengths: LengthWidth,
    userblock: u64,
    lower: LibraryVersion,
    upper: LibraryVersion,
) -> hdf5::File {
    let sizeof_addr = match offsets {
        OffsetWidth::Two => Sizeof::Bytes2,
        OffsetWidth::Four => Sizeof::Bytes4,
        OffsetWidth::Eight => Sizeof::Bytes8,
    };
    let sizeof_size = match lengths {
        LengthWidth::Two => Sizeof::Bytes2,
        LengthWidth::Four => Sizeof::Bytes4,
        LengthWidth::Eight => Sizeof::Bytes8,
    };
    hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(lower, upper))
        .with_fcpl(|properties| {
            properties
                .sizes(SizeofInfo {
                    sizeof_addr,
                    sizeof_size,
                })
                .userblock(userblock)
        })
        .create(path)
        .unwrap()
}

#[track_caller]
fn create_bounded(path: &Path, low: LibraryVersion) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, LibraryVersion::latest()))
        .create(path)
        .unwrap_or_else(|e| panic!("create {path:?}: {e}"))
}
