//! Provides file-space fixture builders shared across edit and interoperability tests.

#[cfg(feature = "__hdf5-1.10")]
use core::ffi::c_int;
use std::ops::Range;
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::Dataset as Hdf5Dataset;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::File as Hdf5File;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::FileBuilder as Hdf5FileBuilder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::file_create::FileSpaceStrategy as Hdf5FileSpaceStrategy;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::SyncPolicy;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::UserblockSize;
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy, Layout};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;

use crate::dataset::Unlimited;
#[cfg(feature = "__hdf5-1.10")]
use crate::file_space_oracle::OracleError;

#[cfg(feature = "__hdf5-1.10")]
use self::persisted::PersistedStrategy;

#[cfg(feature = "__hdf5-1.10")]
pub mod fragmentation;
pub mod persisted;
pub mod publication;

/// Creates a writable session after committing deletions of `holes` alternating datasets.
///
/// Creates `2 * holes + 1` datasets and deletes every other one, leaving a final survivor.
/// The free-section count depends on the strategy and metadata layout. The returned session
/// uses [`SyncPolicy::OnClose`].
///
/// # Panics
///
/// Panics if the object count overflows, or creating, opening, deleting, or committing fails.
pub fn fragmented_session(path: &Path, strategy: FileSpaceStrategy, holes: usize) -> File {
    let objects = holes
        .checked_mul(2)
        .and_then(|count| count.checked_add(1))
        .expect("the fixture object count fits usize");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    builder.write(path).unwrap();
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    for index in 0..objects {
        file.root()
            .create_dataset(&format!("data_{index:05}"), |builder| {
                builder.with_i32_data(&[7; 256]);
            })
            .unwrap();
    }
    file.commit().unwrap();
    for index in (0..holes * 2).step_by(2) {
        file.root().delete(&format!("data_{index:05}")).unwrap();
    }
    file.commit().unwrap();
    file
}

/// Returns the total zero-padding length of the persisted free-space section lists.
///
/// Requires equal used and allocated lengths for each list.
/// Compares the section records with their shortest serialization and checks the padding bytes.
///
/// # Panics
///
/// Panics if reading or parsing fails, a requirement is unmet, or padding contains a nonzero byte.
pub fn serialized_manager_padding(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    let superblock = file.superblock();
    let info = file.file_space_info().unwrap().unwrap();
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    info.manager_addrs
        .iter()
        .enumerate()
        .filter_map(|(slot, &addr)| match addr {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some((slot, stored)),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .map(|(slot, stored)| {
            let at = usize::try_from(file.resolve_address(stored).unwrap().get()).unwrap();
            let header = FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap();
            let list_addr = header.section_list_addr();
            let start = usize::try_from(file.resolve_address(list_addr).unwrap().get()).unwrap();
            let used = usize::try_from(header.section_list_used()).unwrap();
            let list = &bytes[start..start + used];
            let sections = format::parse_section_info(widths, list, stored, &header).unwrap();
            let class = match (info.strategy, slot) {
                (FileSpaceStrategy::Page, PAGE_SMALL_METADATA_SLOT | PAGE_SMALL_RAW_SLOT) => {
                    format::SECTION_CLASS_SMALL
                }
                (FileSpaceStrategy::Page, PAGE_GENERIC_LARGE_SLOT) => format::SECTION_CLASS_LARGE,
                _ => format::SECTION_CLASS_SIMPLE,
            };
            let (_, minimal) =
                format::serialize_free_space_manager(widths, &sections, stored, list_addr, class)
                    .unwrap();
            assert_eq!(header.section_list_used(), header.section_list_allocated());
            let records_end = minimal.len() - SECTION_LIST_CHECKSUM_LEN;
            assert_eq!(&list[..records_end], &minimal[..records_end]);
            assert!(
                list[records_end..used - SECTION_LIST_CHECKSUM_LEN]
                    .iter()
                    .all(|&byte| byte == 0)
            );
            u64::try_from(list.len() - minimal.len()).unwrap()
        })
        .sum()
}

// `H5O__fsinfo_encode` in `H5Ofsinfo.c`, HDF5 2.2.0, writes the `H5F_MEM_PAGE_META` address first.
const PAGE_SMALL_METADATA_SLOT: usize = 0;
// `H5O__fsinfo_encode` in `H5Ofsinfo.c`, HDF5 2.2.0, writes the `H5F_MEM_PAGE_DRAW` address third.
const PAGE_SMALL_RAW_SLOT: usize = 2;
// `H5O__fsinfo_encode` in `H5Ofsinfo.c`, HDF5 2.2.0, writes the `H5F_MEM_PAGE_GENERIC` address seventh.
const PAGE_GENERIC_LARGE_SLOT: usize = 6;

/// The checksum length of a version 0 section list, defined in "Free-space Index" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
const SECTION_LIST_CHECKSUM_LEN: usize = 4;

/// Writes the `a`, `b`, `c` fixture used by file-space strategy reuse tests.
///
/// Dataset `b` is deliberately larger than its neighbors so deleting it leaves a
/// recognizable interior extent for the replacement-allocation assertions.
pub fn write_strategy_triplet(
    path: &Path,
    strategy: FileSpaceStrategy,
    persist: bool,
    threshold: u64,
) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder
        .create_dataset("b")
        .with_i32_data(&[2; STRATEGY_DELETED_ELEMENTS]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(persist)
            .with_threshold(threshold),
    );
    builder.write(path).unwrap();
}

/// Writes the same `a`, `b`, `c` strategy-reuse fixture with the C library.
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_write_strategy_triplet(path: &Path, strategy: Hdf5FileSpaceStrategy) {
    let file = Hdf5FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| fcpl.file_space_strategy(strategy))
        .create(path)
        .unwrap();
    for (name, value, len) in [
        ("a", 1, 100),
        ("b", 2, STRATEGY_DELETED_ELEMENTS),
        ("c", 3, 100),
    ] {
        file.new_dataset::<i32>()
            .shape((len,))
            .create(name)
            .unwrap()
            .write(&vec![value; len])
            .unwrap();
    }
    file.close().unwrap();
}

/// Number of `i32` elements in the middle dataset of the strategy-reuse fixture.
pub const STRATEGY_DELETED_ELEMENTS: usize = 400;

/// Asserts that the HDF5 logical end-of-file equals the physical file length.
#[track_caller]
pub fn assert_eof_matches_physical_size(path: &Path) {
    let file = File::open(path).unwrap();
    let eof = file.superblock().eof_address;
    let actual = std::fs::metadata(path).unwrap().len();
    assert_eq!(
        eof, actual,
        "superblock EOF must match the physical file size"
    );
}

/// Byte length of the middle dataset in the strategy-reuse fixture.
pub const STRATEGY_DELETED_LEN: u64 =
    (STRATEGY_DELETED_ELEMENTS * std::mem::size_of::<i32>()) as u64;

/// Returns the external raw-data extents physically stored for a dataset.
///
/// Compact and unallocated datasets own no external raw extents. Contiguous
/// storage contributes one extent, while chunked storage contributes one extent
/// per stored chunk using each chunk's physical stored size.
pub fn raw_extents(file: &File, dataset: &str) -> Vec<Range<u64>> {
    let dataset = file.dataset(dataset).unwrap();
    match dataset.layout().unwrap() {
        Layout::Contiguous {
            address: Some(address),
            size,
        } => std::iter::once(address..address + size).collect(),
        Layout::Chunked { .. } => dataset
            .chunks()
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.address..chunk.address + chunk.storage_size)
            .collect(),
        _ => Vec::new(),
    }
}

/// Returns the allocated byte range of a contiguous dataset.
pub fn contiguous_extent(file: &File, path: &str) -> Range<u64> {
    let layout = file.dataset(path).unwrap().layout().unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = layout
    else {
        panic!("expected allocated contiguous storage, got {layout:?}");
    };
    address..address + size
}

/// Returns the allocated byte range of a contiguous dataset opened by the C library.
#[cfg(feature = "hdf5")]
pub fn libhdf5_contiguous_extent(dataset: &Hdf5Dataset) -> Range<u64> {
    let address = dataset.offset().unwrap();
    address..address + dataset.storage_size()
}

/// Enumerates free-space-manager sections through `H5Fget_free_sections`.
///
/// Returns `(stored_address, length)` pairs with base-relative addresses. The call uses
/// `H5FD_MEM_DEFAULT` to query all managers and preserves the C enumeration order. The caller can
/// construct a [`crate::file_space_oracle::SectionMap`] to sort and validate the records.
///
/// # Errors
///
/// Returns [`OracleError::C`] if a C call fails, [`OracleError::Allocation`] if the section
/// buffer cannot be allocated, or [`OracleError::Enumeration`] if the two calls report different
/// section counts.
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_free_sections(file: &Hdf5File) -> Result<Vec<(u64, u64)>, OracleError> {
    hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access and `file` keeps the file open. With a null
        // buffer and a count of zero the call writes nothing and returns the number of sections.
        let count =
            unsafe { H5Fget_free_sections(file.id(), H5FD_MEM_DEFAULT, 0, std::ptr::null_mut()) };
        let count = crate::file_space_oracle::enumeration_count(count)?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(count)
            .map_err(|_error| OracleError::Allocation { count })?;
        sections.resize(count, SectionInfo { addr: 0, size: 0 });
        // SAFETY: The library lock serializes access, `file` keeps the file open, and
        // `sections` holds `count` writable records of the layout `H5F_sect_info_t` has.
        let written = unsafe {
            H5Fget_free_sections(file.id(), H5FD_MEM_DEFAULT, count, sections.as_mut_ptr())
        };
        crate::file_space_oracle::enumeration_complete(count, written)?;
        Ok(sections
            .into_iter()
            .map(|section| (section.addr, section.size))
            .collect())
    })
}

/// Writes a libhdf5 fixture and returns its persisted free sections.
///
/// Returns `(stored_address, length)` pairs sorted by address after closing and reopening
/// the file. Uses `strategy`, `userblock`, persistence, and a one-byte threshold. Dataset `b`
/// is deleted between the surviving datasets `a` and `c`.
///
/// # Panics
///
/// Panics if creating, writing, inspecting, or closing the file fails, the section list is empty,
/// or its byte total differs from the native free-space total.
#[cfg(feature = "__hdf5-1.10")]
pub fn write_native_persisted_sections(
    path: &Path,
    strategy: PersistedStrategy,
    userblock: UserblockSize,
) -> Vec<(u64, u64)> {
    let native = Hdf5FileBuilder::new()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties.userblock(userblock.get()).file_space_strategy(
                Hdf5FileSpaceStrategy::FreeSpaceManager {
                    paged: matches!(strategy, PersistedStrategy::Page),
                    persist: true,
                    threshold: 1,
                },
            )
        })
        .create(path)
        .unwrap();
    native
        .new_dataset::<i32>()
        .shape([100])
        .create("a")
        .unwrap()
        .write_raw(&[1; 100])
        .unwrap();
    native
        .new_dataset::<i32>()
        .shape([4096])
        .create("b")
        .unwrap()
        .write_raw(&[2; 4096])
        .unwrap();
    native
        .new_dataset::<i32>()
        .shape([100])
        .create("c")
        .unwrap()
        .write_raw(&[3; 100])
        .unwrap();
    native.unlink("b").unwrap();
    native.close().unwrap();
    let native = Hdf5File::open(path).unwrap();
    let mut sections = libhdf5_free_sections(&native).unwrap();
    sections.sort_unstable();
    assert!(!sections.is_empty());
    assert_eq!(
        sections.iter().map(|(_, length)| length).sum::<u64>(),
        native.free_space()
    );
    native.close().unwrap();
    sections
}

/// Writes a persisting interior-hole fixture and deletes `big` in a committed edit.
pub fn write_persisted_interior_hole(path: &Path) {
    write_interior_hole_source(path, true);
    let file = File::open_rw(path).unwrap();
    file.root().delete("big").unwrap();
    file.commit().unwrap();
}

/// Writes the `a`, `big`, `c` fixture used to leave an interior hole.
///
/// When `persist` is true, the file uses a persisting `FsmAggr` strategy with a
/// threshold of one byte. Dataset `big` occupies 1600 raw bytes between two live
/// datasets, so deleting it leaves reusable interior space between live allocations.
pub fn write_interior_hole_source(path: &Path, persist: bool) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("big").with_i32_data(&[7; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    if persist {
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
        );
    }
    builder.write(path).unwrap();
}

/// Writes a persisting `FsmAggr` file with an unlimited rank-1 `i32` dataset `d`.
pub fn write_persisting_unlimited_i32(path: &Path, len: i32, chunk: u64, threshold: u64) {
    let data: Vec<i32> = (0..len).collect();
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(threshold),
    );
    Unlimited::new("d", &data, chunk).add_to(&mut builder);
    builder.write(path).unwrap();
}

// `H5F_sect_info_t` in `H5Fpublic.h`, HDF5 1.14.6: a `haddr_t` address and an `hsize_t` size,
// both 64-bit.
#[cfg(feature = "__hdf5-1.10")]
#[derive(Clone, Copy)]
#[repr(C)]
struct SectionInfo {
    addr: u64,
    size: u64,
}

// `H5FD_MEM_DEFAULT` of the `H5F_mem_t` enumeration in `H5Fpublic.h`, HDF5 1.14.6, which selects
// the sections of every allocation type.
#[cfg(feature = "__hdf5-1.10")]
const H5FD_MEM_DEFAULT: c_int = 0;

// `H5Fget_free_sections` in `H5F.c`, HDF5 1.14.6, takes a 64-bit file ID, an `H5F_mem_t`
// enumeration, a `size_t` count and a section buffer, and returns an `ssize_t`.
#[cfg(feature = "__hdf5-1.10")]
unsafe extern "C" {
    fn H5Fget_free_sections(
        file_id: i64,
        mem_type: c_int,
        nsects: usize,
        sect_info: *mut SectionInfo,
    ) -> isize;
}
