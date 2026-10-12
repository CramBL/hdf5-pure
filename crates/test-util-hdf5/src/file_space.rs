//! Provides file-space fixture builders shared across edit and interoperability tests.

use std::ops::Range;
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::Dataset as Hdf5Dataset;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::FileBuilder as Hdf5FileBuilder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::file_create::FileSpaceStrategy as Hdf5FileSpaceStrategy;
use hdf5_pure::SpaceOwnership;
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy, Layout, SpaceAccounting};

use crate::dataset::Unlimited;

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
        summary.live_raw_bytes
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
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        reusable_free_space,
        unclassified_space,
        analysis_complete: _,
        trailing_slack_space,
    } = ownership;
    let mut extents = live_raw_space
        .iter()
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
