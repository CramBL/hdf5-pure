#![cfg(feature = "__hdf5-1.10")]
//! Interop tests for in-place appends through an owned handle: append to a
//! filtered, unlimited,
//! Extensible-Array-indexed dataset and confirm the reference C library
//! (`hdf5-metno`) reads the grown dataset back exactly — including datasets the C
//! library itself created, whose incompressible chunks carry a nonzero per-chunk
//! filter mask that in-place appends must leave untouched. Because `File::open_rw` + `Dataset::append`
//! mutates the index in place (rather than rebuilding it), this also exercises
//! byte-for-byte-compatible incremental Extensible-Array growth.

use hdf5_pure::{File, FileAccessProperties, SyncPolicy};
use tempfile::tempdir;
use test_util_hdf5::dataset::{self, Filter, Unlimited};

const SHUFFLE_DEFLATE: &[Filter] = &[Filter::Shuffle, Filter::Deflate(6)];

/// Append `values` one element per call in a single session.
///
/// `SyncPolicy::OnClose` because these tests assert what the file *contains*,
/// not when it reached the platter, and the default `Always` costs one `fsync`
/// per call — the dominant cost of a several-thousand-append loop, and nothing
/// this test measures. The two policies write byte-identical files, which
/// `tests/main/sync_policy.rs` asserts directly. `close` still issues the closing
/// barrier, so the C library reads a fully published file either way.
fn writer_append_each(path: &std::path::Path, values: &[i32]) {
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let mut ds = file.dataset("d").unwrap();
    for &v in values {
        ds.append(&[v]).unwrap();
    }
    drop(ds);
    file.close().unwrap();
}

#[test]
fn pure_creates_writer_appends_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..12).collect();
    Unlimited::new("d", &base, 4)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);
    dataset::pure_append(&path, "d", &(12..20).collect::<Vec<_>>()); // two chunks
    dataset::pure_append(&path, "d", &(20..28).collect::<Vec<_>>()); // two chunks
    let expected: Vec<i32> = (0..28).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_creates_writer_appends_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..20).collect();
    Unlimited::new("d", &base, 5)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);
    dataset::pure_append(&path, "d", &(20..30).collect::<Vec<_>>()); // two chunks, 20 -> 30
    let expected: Vec<i32> = (0..30).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_incompressible_kept_chunks_untouched() {
    // The C library stores incompressible chunks uncompressed with a nonzero
    // per-chunk filter mask. In-place appends never touch the kept chunk elements,
    // so their masks must survive verbatim and C must still read them.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base = dataset::incompressible(0xABCD_1234, 40); // 8 chunks of 5
    Unlimited::new("d", &base, 5)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);
    let extra = dataset::incompressible(0x5555_AAAA, 15); // three chunks (40 -> 55)
    dataset::pure_append(&path, "d", &extra);
    let mut expected = base.clone();
    expected.extend_from_slice(&extra);
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        expected,
        "C must read kept incompressible chunks"
    );
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
}

#[test]
fn c_empty_unallocated_index_is_refused() {
    // The C library defers allocating an empty resizable dataset's
    // Extensible-Array index until the first chunk is written, so there is no
    // index block for in-place growth. `File::open_rw` + `Dataset::append` refuses it cleanly; the
    // batch path (`Dataset::append_staged`) materializes the index instead.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 4)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);
    let file = File::open_rw(&path).unwrap();
    let r = file
        .dataset("d")
        .unwrap()
        .append(&(0..10).collect::<Vec<_>>());
    assert!(
        matches!(r, Err(hdf5_pure::Error::AppendInPlaceUnsupported(_))),
        "expected a clean refusal, got {r:?}"
    );
    drop(file);
    // The file is unchanged and still readable by C.
    assert!(dataset::read_libhdf5::<i32>(&path, "d").is_empty());
}

#[test]
fn unfiltered_writer_append_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 4).pure_create(&path);
    dataset::pure_append(&path, "d", &(10..23).collect::<Vec<_>>()); // unaligned
    let expected: Vec<i32> = (0..23).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn streaming_many_appends_c_reads() {
    // Append chunk-length-1 elements one at a time until the Extensible Array
    // uses super blocks / paged data blocks, growing the index in place, then
    // confirm the C library reads the whole thing back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..64).collect();
    Unlimited::new("d", &base, 1)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);
    writer_append_each(&path, &(64..4096).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..4096).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn reopen_across_sessions_c_reads() {
    // Filtered appends are chunk-aligned; reopen a fresh writer each session.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..8).collect();
    Unlimited::new("d", &base, 4)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);
    dataset::pure_append(&path, "d", &(8..16).collect::<Vec<_>>()); // two chunks, session 1
    dataset::pure_append(&path, "d", &(16..24).collect::<Vec<_>>()); // two chunks, session 2
    let expected: Vec<i32> = (0..24).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_creates_writer_grows_an_unaligned_filtered_tail() {
    // A filtered dataset the C library left on a partial trailing chunk is
    // grown in place: this crate decodes that chunk with its recorded filter
    // mask, extends it, re-encodes it into a fresh allocation and repoints the
    // one index element — and the C library reads the result (issue #393).
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..7).collect(); // 7 of chunk 5 => a partial tail
    Unlimited::new("d", &base, 5)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);
    dataset::pure_append(&path, "d", &[7, 8, 9]);
    let expected: Vec<i32> = (0..10).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}
