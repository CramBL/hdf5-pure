#![cfg(feature = "__hdf5-1.10")]
//! Interop for owned-handle in-place append (issue #148, phase 2): append through
//! a `File::open_rw` `Dataset` handle and confirm the reference C library
//! (`hdf5-metno`) reads the grown dataset back exactly — for unfiltered and
//! filtered datasets this crate wrote, and for a dataset the C library created.

use hdf5_pure::File;
use tempfile::tempdir;
use test_util_hdf5::dataset::{self, Filter, Unlimited};

#[test]
fn owned_append_unfiltered_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    Unlimited::new("d", &(0..5).collect::<Vec<i32>>(), 4).pure_create(&path);
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[5i32, 6, 7]).unwrap(); // any-length (unfiltered)
        ds.append(&[8i32]).unwrap();
    }
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        (0..9).collect::<Vec<_>>()
    );
}

#[test]
fn owned_append_filtered_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 4)
        .filters(&[Filter::Shuffle, Filter::Deflate(6)])
        .pure_create(&path);
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(8..12).collect::<Vec<_>>()).unwrap(); // whole chunk (filtered)
    }
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        (0..12).collect::<Vec<_>>()
    );
}

#[test]
#[cfg(target_endian = "little")]
fn owned_append_onto_c_created_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    Unlimited::new("d", &(0..5).collect::<Vec<i32>>(), 4).libhdf5_create(&path);
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[5i32, 6, 7]).unwrap();
    }
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        (0..8).collect::<Vec<_>>()
    );
}
