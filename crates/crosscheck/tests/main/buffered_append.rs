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

use hdf5_pure::DeflateLevel;
use hdf5_pure::File;

use test_util_hdf5::dataset::{self, Filter, Unlimited};
use test_util_hdf5::lock;

const SHUFFLE_DEFLATE: &[Filter] = &[Filter::Shuffle, Filter::Deflate(DeflateLevel::Four)];

#[test]
fn c_library_reads_a_filtered_partial_last_chunk_written_in_place() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial.h5");
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 4)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path); // two whole chunks

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
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        expected,
        "the C library disagreed"
    );

    // libhdf5's own view of the shape, not just the values it hands back:
    // a padding-as-data misread would show up as a longer dataset.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("d").unwrap().shape(), vec![13]);
    f.close().unwrap();
}

#[test]
fn c_library_reads_a_buffered_appended_dataset() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("buffered.h5");
    // Start unaligned (10 of a chunk of 8), so the appender's first write
    // re-encodes the partial trailing chunk and its later ones extend from a
    // boundary — both shapes in one file.
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);

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
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        expected,
        "the C library disagreed"
    );
}

#[test]
#[cfg(target_endian = "little")]
fn c_library_reads_a_buffered_append_onto_its_own_dataset() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_written.h5");

    // libhdf5 writes the file. This crate appends to it. The latest-format
    // bounds are what make the index an Extensible Array rather than a version-1
    // B-tree, which is the index the in-place path grows.
    Unlimited::new("d", &(0..24).collect::<Vec<i32>>(), 8)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);

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
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        expected,
        "the C library disagreed"
    );
}

#[test]
#[cfg(target_endian = "little")]
fn c_library_reads_an_owned_buffered_completion_of_its_dataset() {
    let _c = lock::libhdf5_guard();
    let path = test_util::temp::temp_path("owned_buffered.h5");
    Unlimited::new("d", &[0i32, 1], 4)
        .filters(SHUFFLE_DEFLATE)
        .libhdf5_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    drop(file);
    appender.append(&[2i32, 3, 4, 5, 6]).unwrap();
    assert_eq!(appender.buffered_elements(), 3);
    assert_eq!(
        appender.unwritten(),
        [4i32, 5, 6]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    );
    appender.finish().unwrap();
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.shape(), [7]);
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [0, 1, 2, 3, 4, 5, 6]);
    drop(dataset);
    file.close().unwrap();
}
