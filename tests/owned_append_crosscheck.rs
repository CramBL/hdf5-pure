// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets; skip elsewhere so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Interop for owned-handle in-place append (issue #148, phase 2): append through
//! a `File::open_rw` `Dataset` handle and confirm the reference C library
//! (`hdf5-metno`) reads the grown dataset back exactly — for unfiltered and
//! filtered datasets this crate wrote, and for a dataset the C library created.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

fn read_c(path: &std::path::Path) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    f.dataset("d").unwrap().read_raw::<i32>().unwrap()
}

fn pure_create(path: &std::path::Path, data: &[i32], chunk: u64, filtered: bool) {
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("d")
        .with_i32_data(data)
        .with_shape(&[data.len() as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if filtered {
        ds.with_shuffle().with_deflate(6);
    }
    b.write(path).unwrap();
}

#[test]
fn owned_append_unfiltered_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    pure_create(&path, &(0..5).collect::<Vec<_>>(), 4, false);
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[5i32, 6, 7]).unwrap(); // any-length (unfiltered)
        ds.append(&[8i32]).unwrap();
    }
    assert_eq!(read_c(&path), (0..9).collect::<Vec<_>>());
}

#[test]
fn owned_append_filtered_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    pure_create(&path, &(0..8).collect::<Vec<_>>(), 4, true);
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(8..12).collect::<Vec<_>>()).unwrap(); // whole chunk (filtered)
    }
    assert_eq!(read_c(&path), (0..12).collect::<Vec<_>>());
}

#[test]
fn owned_append_onto_c_created_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(5),))
            .create("d")
            .unwrap();
        ds.write(&(0..5).collect::<Vec<_>>()).unwrap();
        file.close().unwrap();
    }
    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[5i32, 6, 7]).unwrap();
    }
    assert_eq!(read_c(&path), (0..8).collect::<Vec<_>>());
}
