// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets; skip elsewhere so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Interop tests for in-place appends through an owned handle: append to a
//! filtered, unlimited,
//! Extensible-Array-indexed dataset and confirm the reference C library
//! (`hdf5-metno`) reads the grown dataset back exactly — including datasets the C
//! library itself created, whose incompressible chunks carry a nonzero per-chunk
//! filter mask that in-place appends must leave untouched. Because `File::open_rw` + `Dataset::append`
//! mutates the index in place (rather than rebuilding it), this also exercises
//! byte-for-byte-compatible incremental Extensible-Array growth.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileAccessProperties, FileBuilder, SyncPolicy};
use tempfile::tempdir;

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

/// Append `values` in one call, releasing the writer's lock before returning.
fn writer_append(path: &std::path::Path, values: &[i32]) {
    let file = File::open_rw(path).unwrap();
    file.dataset("d").unwrap().append(values).unwrap();
    file.close().unwrap();
}

/// Append `values` one element per call in a single session.
///
/// `SyncPolicy::OnClose` because these tests assert what the file *contains*,
/// not when it reached the platter, and the default `Always` costs one `fsync`
/// per call — the dominant cost of a several-thousand-append loop, and nothing
/// this test measures. The two policies write byte-identical files, which
/// `tests/sync_policy.rs` asserts directly; `close` still issues the closing
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
    pure_create_filtered(&path, &base, 4);
    writer_append(&path, &(12..20).collect::<Vec<_>>()); // two chunks
    writer_append(&path, &(20..28).collect::<Vec<_>>()); // two chunks
    let expected: Vec<i32> = (0..28).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn c_creates_writer_appends_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..20).collect();
    c_create_filtered(&path, &base, 5);
    writer_append(&path, &(20..30).collect::<Vec<_>>()); // two chunks, 20 -> 30
    let expected: Vec<i32> = (0..30).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn c_incompressible_kept_chunks_untouched() {
    // The C library stores incompressible chunks uncompressed with a nonzero
    // per-chunk filter mask. In-place appends never touch the kept chunk elements,
    // so their masks must survive verbatim and C must still read them.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base = incompressible(0xABCD_1234, 40); // 8 chunks of 5
    c_create_filtered(&path, &base, 5);
    let extra = incompressible(0x5555_AAAA, 15); // three chunks (40 -> 55)
    writer_append(&path, &extra);
    let mut expected = base.clone();
    expected.extend_from_slice(&extra);
    assert_eq!(
        read_c(&path),
        expected,
        "C must read kept incompressible chunks"
    );
    assert_eq!(read_pure(&path), expected);
}

#[test]
fn c_empty_unallocated_index_is_refused() {
    // The C library defers allocating an empty resizable dataset's
    // Extensible-Array index until the first chunk is written, so there is no
    // index block for in-place growth. `File::open_rw` + `Dataset::append` refuses it cleanly; the
    // batch path (`Dataset::append_staged`) materializes the index instead.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    c_create_filtered(&path, &[], 4);
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
    assert!(read_c(&path).is_empty());
}

#[test]
fn unfiltered_writer_append_c_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..10).collect::<Vec<_>>())
        .with_shape(&[10])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();
    writer_append(&path, &(10..23).collect::<Vec<_>>()); // unaligned
    let expected: Vec<i32> = (0..23).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn streaming_many_appends_c_reads() {
    // Append chunk-length-1 elements one at a time until the Extensible Array
    // uses super blocks / paged data blocks, growing the index in place, then
    // confirm the C library reads the whole thing back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..64).collect();
    pure_create_filtered(&path, &base, 1);
    writer_append_each(&path, &(64..4096).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..4096).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn reopen_across_sessions_c_reads() {
    // Filtered appends are chunk-aligned; reopen a fresh writer each session.
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..8).collect();
    pure_create_filtered(&path, &base, 4);
    writer_append(&path, &(8..16).collect::<Vec<_>>()); // two chunks, session 1
    writer_append(&path, &(16..24).collect::<Vec<_>>()); // two chunks, session 2
    let expected: Vec<i32> = (0..24).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}

#[test]
fn c_creates_writer_grows_an_unaligned_filtered_tail() {
    // A filtered dataset the C library left on a partial trailing chunk is
    // grown in place: this crate decodes that chunk with its recorded filter
    // mask, extends it, re-encodes it into a fresh allocation and repoints the
    // one index element — and the C library reads the result (issue #393).
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    let base: Vec<i32> = (0..7).collect(); // 7 of chunk 5 => a partial tail
    c_create_filtered(&path, &base, 5);
    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("d").unwrap().append(&[7, 8, 9]).unwrap();
    }
    let expected: Vec<i32> = (0..10).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected);
}
