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
use hdf5_pure::SpaceOwnership;
use hdf5_pure::StoredAddress;
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy, Layout, SpaceAccounting};
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;

use crate::dataset::Unlimited;

/// Returns the total zero-padding length of the persisted free-space section lists.
///
/// Requires a file without a userblock and equal used and allocated lengths for each list.
/// Compares the section records with their shortest serialization and checks the padding bytes.
///
/// # Panics
///
/// Panics if reading or parsing fails, a requirement is unmet, or padding contains a nonzero byte.
pub fn serialized_manager_padding(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    let superblock = file.superblock();
    assert!(superblock.base_address.is_zero());
    let widths = FormatWidths::from_sizes(superblock.offset_size, superblock.length_size).unwrap();
    file.file_space_info()
        .unwrap()
        .manager_addrs
        .iter()
        .copied()
        .filter(|&addr| addr != u64::MAX)
        .map(|addr| {
            let at = usize::try_from(addr).unwrap();
            let header = FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap();
            let list_addr = header.section_list_addr();
            let start = usize::try_from(list_addr.get()).unwrap();
            let used = usize::try_from(header.section_list_used()).unwrap();
            let list = &bytes[start..start + used];
            let sections =
                format::parse_section_info(widths, list, StoredAddress::new(addr), &header)
                    .unwrap();
            let (_, minimal) = format::serialize_free_space_manager(
                widths,
                &sections,
                StoredAddress::new(addr),
                list_addr,
                format::SECTION_CLASS_SIMPLE,
            )
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
    builder.with_file_space_strategy(strategy, persist, threshold);
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

/// Asserts that live file-space accounting is internally consistent.
#[track_caller]
pub fn assert_space_accounting_consistent(acct: &SpaceAccounting) {
    let summed: u64 = acct.reusable_free_space.iter().map(|(_, len)| len).sum();
    assert_eq!(
        summed, acct.reusable_free_bytes,
        "reusable_free_bytes must equal the summed region lengths: {:?}",
        acct.reusable_free_space
    );
    for w in acct.reusable_free_space.windows(2) {
        assert!(
            w[0].0 + w[0].1 <= w[1].0,
            "reusable free regions must be sorted and disjoint: {:?}",
            acct.reusable_free_space
        );
    }
    for &(addr, len) in &acct.reusable_free_space {
        assert!(
            addr + len <= acct.logical_size,
            "a free region [{addr}, {}) must lie within logical_size {}",
            addr + len,
            acct.logical_size
        );
    }
}

/// Summarizes byte totals and extent counts from the storage diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpaceOwnershipSummary {
    /// The physical file length in bytes.
    pub physical_size: u64,
    /// The end-of-file position stored in the superblock.
    pub logical_size: u64,
    /// The length of the userblock, or zero for a file without one.
    pub userblock_bytes: u64,
    /// The total length of dataset payload allocations stored outside object headers.
    pub live_raw_bytes: u64,
    /// The total length of identified object headers, indexes, heaps, and file metadata.
    pub live_metadata_bytes: u64,
    /// The total length of persisted free-space-manager allocations.
    pub free_space_manager_bytes: u64,
    /// The total length of reusable free extents.
    pub reusable_free_bytes: u64,
    /// The number of unclassified bytes below the logical end of file.
    pub unclassified_bytes: u64,
    /// The number of bytes beyond the logical end of file.
    pub trailing_slack_bytes: u64,
    /// The number of dataset payload allocations stored outside object headers.
    pub live_raw_extents: usize,
    /// The number of identified object headers, index blocks, heap blocks, and file metadata blocks.
    pub live_metadata_extents: usize,
    /// The number of persisted free-space-manager allocations.
    pub free_space_manager_extents: usize,
    /// The number of reusable free extents.
    pub reusable_free_extents: usize,
    /// The number of unclassified extents below the logical end of file.
    pub unclassified_extents: usize,
    /// The number of extents beyond the logical end of file.
    pub trailing_slack_extents: usize,
}

/// Asserts complete byte accounting for a controlled fixture.
///
/// Requires complete analysis, disjoint extents, and zero unclassified bytes or trailing slack.
/// The fixture must use storage classes the diagnostic supports.
///
/// # Panics
///
/// Panics if the diagnostic returns an error or any accounting requirement is unmet.
#[track_caller]
pub fn assert_exact_space_ownership(file: &File) -> SpaceOwnershipSummary {
    let ownership = file.__space_ownership().unwrap();
    assert_ownership_extents_disjoint(&ownership);
    let SpaceOwnership {
        physical_size,
        logical_size,
        userblock_space,
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        reusable_free_space,
        unclassified_space,
        analysis_complete,
        trailing_slack_space,
    } = ownership;
    assert!(analysis_complete, "ownership analysis must be complete");

    let summary = SpaceOwnershipSummary {
        physical_size,
        logical_size,
        userblock_bytes: extent_bytes(&userblock_space),
        live_raw_bytes: extent_bytes(&live_raw_space),
        live_metadata_bytes: extent_bytes(&live_metadata_space),
        free_space_manager_bytes: extent_bytes(&free_space_manager_space),
        reusable_free_bytes: extent_bytes(&reusable_free_space),
        unclassified_bytes: extent_bytes(&unclassified_space),
        trailing_slack_bytes: extent_bytes(&trailing_slack_space),
        live_raw_extents: live_raw_space.len(),
        live_metadata_extents: live_metadata_space.len(),
        free_space_manager_extents: free_space_manager_space.len(),
        reusable_free_extents: reusable_free_space.len(),
        unclassified_extents: unclassified_space.len(),
        trailing_slack_extents: trailing_slack_space.len(),
    };
    assert_eq!(
        summary.unclassified_bytes, 0,
        "all committed bytes below logical EOF must be live storage or reusable free space: {unclassified_space:?}"
    );
    assert_eq!(
        summary.trailing_slack_bytes, 0,
        "the controlled churn fixture must not leave bytes past logical EOF: {trailing_slack_space:?}"
    );
    assert_eq!(summary.logical_size, summary.physical_size);
    assert_eq!(
        summary.userblock_bytes
            + summary.live_raw_bytes
            + summary.live_metadata_bytes
            + summary.free_space_manager_bytes
            + summary.reusable_free_bytes
            + summary.unclassified_bytes
            + summary.trailing_slack_bytes,
        summary.physical_size,
        "ownership categories must add exactly to the physical file size"
    );
    summary
}

fn assert_ownership_extents_disjoint(ownership: &SpaceOwnership) {
    let SpaceOwnership {
        physical_size,
        logical_size: _,
        userblock_space,
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        reusable_free_space,
        unclassified_space,
        analysis_complete: _,
        trailing_slack_space,
    } = ownership;
    let mut extents = userblock_space
        .iter()
        .chain(live_raw_space)
        .chain(live_metadata_space)
        .chain(free_space_manager_space)
        .chain(reusable_free_space)
        .chain(unclassified_space)
        .chain(trailing_slack_space)
        .copied()
        .collect::<Vec<_>>();
    extents.sort_unstable_by_key(|&(addr, _)| addr);
    let mut end = 0u64;
    for &(addr, len) in &extents {
        assert!(len > 0, "ownership extents must be non-empty");
        assert!(
            addr >= end,
            "ownership extents must not overlap: previous end {end}, next [{addr}, {})",
            addr + len
        );
        end = addr + len;
        assert!(
            end <= *physical_size,
            "ownership extent [{addr}, {end}) exceeds physical size {physical_size}"
        );
    }
}

fn extent_bytes(extents: &[(u64, u64)]) -> u64 {
    extents.iter().map(|&(_, len)| len).sum()
}

/// Asserts that the HDF5 logical end-of-file equals the physical file length.
#[track_caller]
pub fn assert_eof_matches_physical_size(path: &Path) {
    let file = File::open(path).unwrap();
    let eof = file.file_size();
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

/// Returns the free sections libhdf5 tracks in `file`, as `(address, size)` pairs whose addresses
/// are relative to the base address.
#[cfg(feature = "__hdf5-1.10")]
pub fn libhdf5_free_sections(file: &Hdf5File) -> Vec<(u64, u64)> {
    hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access and `file` keeps the file open. With a null
        // buffer and a count of zero the call writes nothing and returns the number of sections.
        let count =
            unsafe { H5Fget_free_sections(file.id(), H5FD_MEM_DEFAULT, 0, std::ptr::null_mut()) };
        let count = usize::try_from(count).unwrap();
        if count == 0 {
            return Vec::new();
        }
        let mut sections = vec![SectionInfo { addr: 0, size: 0 }; count];
        // SAFETY: The library lock serializes access, `file` keeps the file open, and
        // `sections` holds `count` writable records of the layout `H5F_sect_info_t` has.
        let written = unsafe {
            H5Fget_free_sections(file.id(), H5FD_MEM_DEFAULT, count, sections.as_mut_ptr())
        };
        assert_eq!(usize::try_from(written).unwrap(), count);
        sections
            .into_iter()
            .map(|section| (section.addr, section.size))
            .collect()
    })
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
        builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    }
    builder.write(path).unwrap();
}

/// Writes a persisting `FsmAggr` file with an unlimited rank-1 `i32` dataset `d`.
pub fn write_persisting_unlimited_i32(path: &Path, len: i32, chunk: u64, threshold: u64) {
    let data: Vec<i32> = (0..len).collect();
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, threshold);
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
