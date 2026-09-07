// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets; skip elsewhere so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Interop tests for `Dataset::append_staged`: append to a filtered,
//! unlimited, Extensible-Array-indexed dataset and confirm the reference C
//! library (`hdf5-metno`) reads the grown dataset back exactly — including
//! datasets the C library itself created, whose incompressible chunks carry a
//! nonzero per-chunk filter mask that the append must preserve.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

/// A deterministic incompressible i32 stream (LCG), so deflate stores chunks
/// uncompressed and the C library records a nonzero per-chunk filter mask.
fn incompressible(seed: u32, n: usize) -> Vec<i32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x as i32
        })
        .collect()
}

fn read_pure(path: &std::path::Path) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap()
}

fn read_c(path: &std::path::Path) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    f.dataset("d").unwrap().read_raw::<i32>().unwrap()
}

/// Create a filtered (deflate+shuffle) rank-1 unlimited i32 dataset with the C
/// library under the latest format (so it gets an Extensible-Array index).
fn c_create_filtered(path: &std::path::Path, data: &[i32], chunk: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .shuffle()
        .deflate(6)
        .chunk((chunk,))
        .shape((Extent::resizable(data.len()),))
        .create("d")
        .unwrap();
    ds.write(data).unwrap();
    file.close().unwrap();
}

/// Create a filtered rank-1 unlimited i32 dataset with this crate.
fn pure_create_filtered(path: &std::path::Path, data: &[i32], chunk: u64) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(data)
        .with_shape(&[data.len() as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk])
        .with_shuffle()
        .with_deflate(6);
    b.write(path).unwrap();
}

fn pure_append(path: &std::path::Path, values: &[i32]) {
    let s = File::open_rw(path).unwrap();
    s.dataset("d")
        .unwrap()
        .append_staged(|b| {
            b.append_i32(values);
        })
        .unwrap();
    s.commit().unwrap();
}

#[test]
fn pure_creates_pure_appends_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..12).collect();
    pure_create_filtered(&path, &base, 4);
    // Aligned append (12 -> 20) then unaligned (20 -> 27).
    pure_append(&path, &(12..20).collect::<Vec<_>>());
    pure_append(&path, &(20..27).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..27).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn c_creates_pure_appends_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..20).collect();
    c_create_filtered(&path, &base, 5);
    pure_append(&path, &(20..33).collect::<Vec<_>>()); // unaligned 20 -> 33
    let expected: Vec<i32> = (0..33).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn c_incompressible_kept_chunks_filter_mask_preserved() {
    // The load-bearing interop case: the C library stores incompressible chunks
    // uncompressed and records a nonzero per-chunk filter mask. The append must
    // carry each kept chunk's original mask (not force 0) into the rebuilt index,
    // or the C library would try to inflate stored-uncompressed bytes and read
    // garbage.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base = incompressible(0xABCD_1234, 40); // 8 chunks of 5, all incompressible
    c_create_filtered(&path, &base, 5);

    let extra = incompressible(0x5555_AAAA, 17); // unaligned append (40 -> 57)
    pure_append(&path, &extra);

    let mut expected = base.clone();
    expected.extend_from_slice(&extra);
    assert_eq!(
        read_c(&path),
        expected,
        "C must read the kept incompressible chunks intact"
    );
    assert_eq!(read_pure(&path), expected);
}

#[test]
fn append_to_c_empty_extensible() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // C creates an empty (0-length) resizable filtered dataset.
    c_create_filtered(&path, &[], 4);
    pure_append(&path, &(0..10).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..10).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn pure_unfiltered_append_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..10).collect::<Vec<_>>())
        .with_shape(&[10])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();
    pure_append(&path, &(10..23).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..23).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn userblock_append_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("d")
        .with_i32_data(&(0..10).collect::<Vec<_>>())
        .with_shape(&[10])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_shuffle()
        .with_deflate(6);
    b.write(&path).unwrap();
    pure_append(&path, &(10..21).collect::<Vec<_>>()); // unaligned
    let expected: Vec<i32> = (0..21).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn large_paged_ea_append_c_reads() {
    // Append enough chunks (chunk length 1) that the rebuilt Extensible Array uses
    // super blocks / paged data blocks, then confirm the C library reads it back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..64).collect();
    pure_create_filtered(&path, &base, 1);
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&(64..4096).collect::<Vec<_>>());
            })
            .unwrap();
        s.commit().unwrap();
    }
    let expected: Vec<i32> = (0..4096).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}
