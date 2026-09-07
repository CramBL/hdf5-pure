// Uses the reference HDF5 C library (`hdf5-metno`), gated to 64-bit little-endian
// targets; skip on 32-bit so `cross test --target i686-...` stays pure-Rust.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! SWMR-reader (refresh) tests: hdf5-pure follows an Extensible-Array dataset
//! that an external HDF5 writer appends to.
//!
//! The reference C library acts as the writer here (create, then reopen and
//! extend); hdf5-pure opens the file with `open_swmr` and `refresh`es to observe
//! the appended rows. The append crosses the direct-data-block -> super-block
//! boundary, so the chunk index grows structurally between refreshes.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{Error, File};
use tempfile::tempdir;

/// Create a 1-D unlimited, chunked i32 dataset (latest format -> EA index) with
/// `n` rows valued 0..n.
fn c_create(path: &std::path::Path, n: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((1,))
        .shape((Extent::resizable(n),))
        .create("d")
        .unwrap();
    let data: Vec<i32> = (0..n as i32).collect();
    ds.write(&data).unwrap();
    file.close().unwrap();
}

/// Reopen `path` read-write and grow the dataset to `new_total` rows valued
/// 0..new_total.
fn c_extend(path: &std::path::Path, new_total: usize) {
    let file = hdf5::File::open_rw(path).unwrap();
    let ds = file.dataset("d").unwrap();
    ds.resize((new_total,)).unwrap();
    let full: Vec<i32> = (0..new_total as i32).collect();
    ds.write(&full).unwrap();
    file.close().unwrap();
}

#[test]
fn refresh_follows_external_appends() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("swmr.h5");

    // Writer creates 100 rows (within the direct data blocks).
    c_create(&path, 100);

    // Reader opens for SWMR and sees the initial snapshot.
    let mut file = File::open_swmr(&path).unwrap();
    {
        let ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![100]);
        assert_eq!(ds.read_i32().unwrap(), (0..100).collect::<Vec<_>>());
    }

    // Writer extends to 300 rows (now spanning the first super block).
    c_extend(&path, 300);

    // Without refresh, the reader still sees the old snapshot.
    {
        let ds = file.dataset("d").unwrap();
        assert_eq!(
            ds.shape().unwrap(),
            vec![100],
            "stale view changed before refresh"
        );
    }

    // After refresh, the reader observes the appended rows and grown index.
    file.refresh().unwrap();
    {
        let ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![300]);
        assert_eq!(ds.read_i32().unwrap(), (0..300).collect::<Vec<_>>());
    }

    // A second extension across a larger boundary (super-block nesting).
    c_extend(&path, 5000);
    file.refresh().unwrap();
    {
        let ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![5000]);
        assert_eq!(ds.read_i32().unwrap(), (0..5000).collect::<Vec<_>>());
    }
}

#[test]
fn refresh_requires_open_swmr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("plain.h5");
    c_create(&path, 50);
    let bytes = std::fs::read(&path).unwrap();
    let mut file = File::from_bytes(bytes).unwrap();
    assert!(matches!(file.refresh(), Err(Error::SwmrUnsupported)));
}
