// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for issue #262: the two on-disk shapes this
//! change newly produces must be readable by the C library, not merely by this
//! crate's own reader.
//!
//! 1. A *filtered* dataset whose last chunk is partial, written by the immediate
//!    in-place path. Before this change that path only ever wrote whole chunks,
//!    so a partial last chunk on a filtered dataset was something only the
//!    staged, index-rebuilding path produced. The chunk is zero-padded to the
//!    full chunk size before the filter pipeline runs, and the dataspace
//!    dimension is what bounds the live elements — a reader that took the chunk
//!    size for the live length instead would read the padding back as data.
//! 2. A dataset grown by a `BufferedAppender`, every write of which is the
//!    in-place one — including, since issue #393, the first write onto a
//!    filtered dataset that starts on a partial trailing chunk.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use std::sync::{Mutex, MutexGuard};
use tempfile::tempdir;

// libhdf5 is not built thread-safe here; every test that touches the C library
// takes this guard as its first line and holds it for the whole body, so no two
// run C-library code at once. See `bounded_append_crosscheck` for the full note.
static C_LIB: Mutex<()> = Mutex::new(());

fn c_lib_guard() -> MutexGuard<'static, ()> {
    C_LIB
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A rank-1 unlimited chunked i32 dataset `d` seeded with `0..n`, shuffled and
/// deflated, written by this crate.
fn pure_create(path: &std::path::Path, n: i32, chunk: u64) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk])
        .with_shuffle()
        .with_deflate(4);
    b.write(path).unwrap();
}

fn read_c(path: &std::path::Path) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    let v = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    f.close().unwrap();
    v
}

fn read_pure(path: &std::path::Path) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap()
}

#[test]
fn c_library_reads_a_filtered_partial_last_chunk_written_in_place() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("partial.h5");
    pure_create(&path, 8, 4); // two whole chunks

    {
        let session = File::open_rw(&path).unwrap();
        // 5 elements onto a length of 8 with a chunk of 4: one whole chunk plus
        // a chunk holding a single live element and three of padding.
        session
            .dataset("d")
            .unwrap()
            .append(&(8..13i32).collect::<Vec<_>>())
            .unwrap();
    }

    let expected: Vec<i32> = (0..13).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected, "the C library disagreed");

    // The C library's own view of the shape, not just the values it hands back:
    // a padding-as-data misread would show up as a longer dataset.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("d").unwrap().shape(), vec![13]);
    f.close().unwrap();
}

#[test]
fn c_library_reads_a_buffered_appended_dataset() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("buffered.h5");
    // Start unaligned (10 of a chunk of 8), so the appender's first write
    // re-encodes the partial trailing chunk and its later ones extend from a
    // boundary — both shapes in one file.
    pure_create(&path, 10, 8);

    {
        let session = File::open_rw(&path).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        for i in 0..9i32 {
            app.append(&(10 + i * 7..17 + i * 7).collect::<Vec<_>>())
                .unwrap();
        }
        app.finish().unwrap();
    }

    let expected: Vec<i32> = (0..73).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected, "the C library disagreed");
}

#[test]
fn c_library_reads_a_buffered_append_onto_its_own_dataset() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_written.h5");

    // The C library writes the file; this crate appends to it. The latest-format
    // bounds are what make the index an Extensible Array rather than a version-1
    // B-tree, which is the index the in-place path grows.
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((8,))
            .shuffle()
            .deflate(4)
            .shape((Extent::resizable(24),))
            .create("d")
            .unwrap();
        ds.write(&(0..24i32).collect::<Vec<_>>()).unwrap();
        file.close().unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        for i in 0..10i32 {
            app.append(&(24 + i * 5..29 + i * 5).collect::<Vec<_>>())
                .unwrap();
        }
        app.finish().unwrap();
    }

    let expected: Vec<i32> = (0..74).collect();
    assert_eq!(read_pure(&path), expected);
    assert_eq!(read_c(&path), expected, "the C library disagreed");
}
