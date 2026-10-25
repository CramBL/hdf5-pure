use std::ops::Range;

pub(super) use test_util_hdf5::file_space::{
    assert_eof_matches_physical_size, serialized_manager_padding,
};

use crate::edit::SpaceOwnership;
use crate::reader::{File, Layout};

/// Returns the allocated contiguous payload range of the dataset at `path`.
///
/// # Panics
///
/// Panics if the dataset or layout cannot be read or its storage is not allocated and contiguous.
pub(super) fn contiguous_extent(file: &File, path: &str) -> Range<u64> {
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

/// Returns contiguous or stored chunk payload ranges for the dataset at `path`.
///
/// Compact and unallocated storage contribute an empty list.
///
/// # Panics
///
/// Panics if the dataset, layout, or chunks cannot be read, or a range end cannot be represented.
pub(super) fn raw_extents(file: &File, path: &str) -> Vec<Range<u64>> {
    let dataset = file.dataset(path).unwrap();
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

/// Asserts a complete, disjoint classification of every file byte.
///
/// # Panics
///
/// Panics if [`assert_space_ownership`] fails with an empty list of expected unclassified extents.
pub(super) fn assert_exact_space_ownership(file: &File) -> SpaceOwnership {
    assert_space_ownership(file, &[])
}

/// Asserts a complete byte partition with exactly the unclassified extents in `expected`.
///
/// # Panics
///
/// Panics if analysis fails or [`SpaceOwnership::assert_classified`] fails.
pub(super) fn assert_space_ownership(file: &File, expected: &[(u64, u64)]) -> SpaceOwnership {
    let owned = super::ownership(file).unwrap();
    owned.assert_classified(expected);
    owned
}
