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
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::{FileSpaceSettings, FileSpaceStrategy};
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

/// Opens an existing file read-only, using a native page buffer where supported.
///
/// Under HDF5 1.14 and later, opens `path` with a page-buffer capacity of four `page`-byte pages.
/// Under earlier releases, opens `path` without a page buffer.
///
/// `H5PB_read` reads past the end of heap buffers during paged reads on releases before 1.14
/// (`H5PB.c`, HDF5 1.10.11 and 1.12.3).
///
/// # Panics
///
/// Panics if libhdf5 cannot open `path`. Under HDF5 1.14 and later, also panics if four times
/// `page` cannot be represented as a [`NonZeroUsize`].
#[cfg(feature = "__hdf5-1.10")]
#[track_caller]
pub(crate) fn libhdf5_open_page_compatible(path: &Path, page: u64) -> hdf5::File {
    if cfg!(feature = "__hdf5-1.14") {
        let capacity = usize::try_from(4 * u128::from(page))
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| panic!("page-buffer capacity for page size {page} is out of range"));
        libhdf5_open_with_page_buffer(path, capacity)
    } else {
        libhdf5_open(path)
    }
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

/// Creates a native file with the earliest lower format bound.
///
/// Sets the address and length widths independently and reserves `userblock` bytes before the
/// superblock. The upper format bound is [`LibraryVersion::latest`].
///
/// # Panics
///
/// Panics if libhdf5 rejects the creation properties or cannot create `path`.
pub fn libhdf5_create_earliest_with_widths(
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
        LibraryVersion::Earliest,
        LibraryVersion::latest(),
    )
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

/// Creates a native fixture with independent widths and the supplied file-space settings.
///
/// Sets both format bounds to HDF5 1.10 and reserves `userblock` bytes before the superblock.
///
/// # Panics
///
/// Panics if libhdf5 rejects the creation properties or cannot create `path`.
///
/// # Examples
///
/// ```
/// use hdf5::plist::file_create::Sizeof;
/// use hdf5_pure::{FileSpaceSettings, FileSpaceStrategy, LengthWidth, OffsetWidth};
/// use test_util_hdf5::file::libhdf5_create_with_widths_and_space;
///
/// # let directory = tempfile::tempdir()?;
/// let path = directory.path().join("native.h5");
/// let file = libhdf5_create_with_widths_and_space(
///     &path,
///     OffsetWidth::Four,
///     LengthWidth::Two,
///     0,
///     FileSpaceSettings::new(FileSpaceStrategy::None),
/// );
/// let sizes = file.create_plist()?.get_sizes()?;
/// assert_eq!(sizes.sizeof_addr, Sizeof::Bytes4);
/// assert_eq!(sizes.sizeof_size, Sizeof::Bytes2);
/// file.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_create_with_widths_and_space(
    path: &Path,
    offsets: OffsetWidth,
    lengths: LengthWidth,
    userblock: u64,
    settings: FileSpaceSettings,
) -> hdf5::File {
    let strategy = match settings.strategy() {
        FileSpaceStrategy::None => NativeStrategy::None,
        FileSpaceStrategy::Aggr => NativeStrategy::PageAggregation,
        FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => NativeStrategy::FreeSpaceManager {
            paged: settings.strategy() == FileSpaceStrategy::Page,
            persist: settings.persist(),
            threshold: settings.threshold(),
        },
    };
    hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::V110)
        })
        .with_fcpl(|properties| {
            properties
                .sizes(sizes(offsets, lengths))
                .userblock(userblock)
                .file_space_strategy(strategy)
                .file_space_page_size(settings.page_size().get())
        })
        .create(path)
        .unwrap()
}

fn create_with_widths(
    path: &Path,
    offsets: OffsetWidth,
    lengths: LengthWidth,
    userblock: u64,
    lower: LibraryVersion,
    upper: LibraryVersion,
) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(lower, upper))
        .with_fcpl(|properties| {
            properties
                .sizes(sizes(offsets, lengths))
                .userblock(userblock)
        })
        .create(path)
        .unwrap()
}

fn sizes(offsets: OffsetWidth, lengths: LengthWidth) -> SizeofInfo {
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
    SizeofInfo {
        sizeof_addr,
        sizeof_size,
    }
}

#[track_caller]
fn create_bounded(path: &Path, low: LibraryVersion) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, LibraryVersion::latest()))
        .create(path)
        .unwrap_or_else(|e| panic!("create {path:?}: {e}"))
}

#[cfg(all(test, feature = "__hdf5-1.10"))]
mod tests {
    use rstest::rstest;

    use crate::file;
    use crate::page_managers;
    use crate::page_managers::Geometry;

    #[rstest]
    #[case::small(Geometry::Small)]
    #[case::userblock_aligned_page(Geometry::UserblockAlignedPage)]
    #[case::large(Geometry::Large)]
    fn page_reader_uses_version_compatible_buffer(#[case] geometry: Geometry) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("page.h5");
        page_managers::write_native(&path, geometry);
        let file = file::libhdf5_open_page_compatible(&path, geometry.page_size());
        let buffer = file.fapl().unwrap().get_page_buffer_size().unwrap();
        assert_eq!(
            buffer.buf_size,
            if cfg!(feature = "__hdf5-1.14") {
                usize::try_from(4 * geometry.page_size()).unwrap()
            } else {
                0
            }
        );
        assert_eq!(buffer.min_meta_perc, 0);
        assert_eq!(buffer.min_raw_perc, 0);
        file.close().unwrap();
        directory.close().unwrap();
    }
}
