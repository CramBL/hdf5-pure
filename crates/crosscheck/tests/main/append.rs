#![cfg(feature = "__hdf5-1.10")]
//! Interop tests for `Dataset::append_staged`: append to a filtered,
//! unlimited, Extensible-Array-indexed dataset and confirm the reference C
//! library (`hdf5-metno`) reads the grown dataset back exactly — including
//! datasets the C library itself created, whose incompressible chunks carry a
//! nonzero per-chunk filter mask that the append must preserve.

use hdf5_pure::FileBuilder;
use tempfile::tempdir;
use test_util_hdf5::dataset::{self, Filter, Unlimited};

const SHUFFLE_DEFLATE: &[Filter] = &[Filter::Shuffle, Filter::Deflate(6)];

#[test]
fn pure_creates_pure_appends_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..12).collect();
    Unlimited::new("d", &base, 4)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);
    // Aligned append (12 -> 20) then unaligned (20 -> 27).
    dataset::pure_append_staged(&path, "d", &(12..20).collect::<Vec<_>>());
    dataset::pure_append_staged(&path, "d", &(20..27).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..27).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_creates_pure_appends_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..20).collect();
    Unlimited::new("d", &base, 5)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);
    dataset::pure_append_staged(&path, "d", &(20..33).collect::<Vec<_>>()); // unaligned 20 -> 33
    let expected: Vec<i32> = (0..33).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_incompressible_kept_chunks_filter_mask_preserved() {
    // The load-bearing interop case: the C library stores incompressible chunks
    // uncompressed and records a nonzero per-chunk filter mask. The append must
    // carry each kept chunk's original mask (not force 0) into the rebuilt index,
    // or the C library would try to inflate stored-uncompressed bytes and read
    // garbage.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base = dataset::incompressible(0xABCD_1234, 40); // 8 chunks of 5, all incompressible
    Unlimited::new("d", &base, 5)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);

    let extra = dataset::incompressible(0x5555_AAAA, 17); // unaligned append (40 -> 57)
    dataset::pure_append_staged(&path, "d", &extra);

    let mut expected = base.clone();
    expected.extend_from_slice(&extra);
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        expected,
        "C must read the kept incompressible chunks intact"
    );
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn append_to_c_empty_extensible() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // C creates an empty (0-length) resizable filtered dataset.
    Unlimited::<i32>::new("d", &[], 4)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);
    dataset::pure_append_staged(&path, "d", &(0..10).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..10).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn pure_unfiltered_append_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 4).pure_create(&path);
    dataset::pure_append_staged(&path, "d", &(10..23).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..23).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn userblock_append_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 4)
        .filters(SHUFFLE_DEFLATE)
        .add_to(&mut b);
    b.write(&path).unwrap();
    dataset::pure_append_staged(&path, "d", &(10..21).collect::<Vec<_>>()); // unaligned
    let expected: Vec<i32> = (0..21).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn large_paged_ea_append_c_reads() {
    // Append enough chunks (chunk length 1) that the rebuilt Extensible Array uses
    // super blocks / paged data blocks, then confirm the C library reads it back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..64).collect();
    Unlimited::new("d", &base, 1)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);
    dataset::pure_append_staged(&path, "d", &(64..4096).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..4096).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}
