//! A typed whole-dataset read decodes the same values whether it sweeps the
//! dataset in row windows or reads it whole (issue #289).
//!
//! `Dataset::read_f64` and its eight siblings used to be `read_raw()` followed by
//! a decode of the entire buffer; they now sweep row windows so the stored bytes
//! beside the output are one window rather than the whole dataset. The bytes are
//! supposed to be identical either way, and the reference here says so directly:
//! **`read_f64_rows(0, n)` over every row is the old code path**, since a
//! full-range window delegates to the whole read and decodes the result in one
//! pass. Every assertion below is `sweep == whole`, so a fixture that disagrees
//! names the sweep as the thing that changed.
//!
//! Every fixture is deliberately larger than the one-mebibyte window budget, in
//! more than one window's worth of rows, and none of them divides evenly: a
//! dataset whose rows, chunk bands, and window all lined up would exercise the
//! sweep without exercising its edges.
//!
//! One case is missing from here and lives in `crates/crosscheck/tests/empty_chunked.rs`
//! instead: a chunk grid with holes in it, where a window is answered partly from
//! storage and partly from the fill value. This crate's writer allocates every
//! chunk it declares, so only the reference C library can produce that file
//! (issue #293), and a test of it has to link the library.

use hdf5_pure::{File, FileBuilder};

/// Comfortably more than the read's own window budget, and not a round number of
/// windows: 2.3 MiB of `f64` against a 1 MiB window.
const N: usize = 300_007;

/// Assert every typed reader agrees with its full-range windowed form, which is
/// the whole-read-then-decode path this sweep replaced.
fn sweep_matches_whole(ds: &hdf5_pure::Dataset, rows: u64) {
    assert_eq!(ds.read_f64().unwrap(), ds.read_f64_rows(0, rows).unwrap());
    assert_eq!(ds.read_f32().unwrap(), ds.read_f32_rows(0, rows).unwrap());
    assert_eq!(ds.read_i8().unwrap(), ds.read_i8_rows(0, rows).unwrap());
    assert_eq!(ds.read_i16().unwrap(), ds.read_i16_rows(0, rows).unwrap());
    assert_eq!(ds.read_i32().unwrap(), ds.read_i32_rows(0, rows).unwrap());
    assert_eq!(ds.read_i64().unwrap(), ds.read_i64_rows(0, rows).unwrap());
    assert_eq!(ds.read_u8().unwrap(), ds.read_u8_rows(0, rows).unwrap());
    assert_eq!(ds.read_u16().unwrap(), ds.read_u16_rows(0, rows).unwrap());
    assert_eq!(ds.read_u32().unwrap(), ds.read_u32_rows(0, rows).unwrap());
    assert_eq!(ds.read_u64().unwrap(), ds.read_u64_rows(0, rows).unwrap());
}

#[test]
fn a_swept_read_of_a_chunked_dataset_matches_a_whole_one() {
    // A chunk band that divides neither the dataset nor the window, so the last
    // window is short and the last chunk is ragged.
    let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.5).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunked.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N as u64])
        .with_chunks(&[1000]);
    b.write(&path).unwrap();

    for file in [
        File::open(&path).unwrap(),
        File::open_streaming(&path).unwrap(),
    ] {
        let ds = file.dataset("t").unwrap();
        let values = ds.read_f64().unwrap();
        assert_eq!(values.len(), N);
        assert_eq!(values, data, "the swept read returned the wrong values");
        sweep_matches_whole(&ds, N as u64);
    }
}

#[test]
fn a_swept_read_of_a_filtered_dataset_matches_a_whole_one() {
    let data: Vec<f64> = (0..N).map(|i| (i % 977) as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deflate.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N as u64])
        .with_chunks(&[3000])
        .with_deflate(4);
    b.write(&path).unwrap();

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();
    assert_eq!(ds.read_f64().unwrap(), data);
    sweep_matches_whole(&ds, N as u64);
}

#[test]
fn a_swept_read_of_a_contiguous_dataset_matches_a_whole_one() {
    // Stored narrower than every type it is read as, so the decode widens rather
    // than reinterprets — the case that rules out reusing the stored buffer.
    //
    // The row count is its own: a window is a budget in *stored* bytes, so `N`
    // two-byte elements would fit inside one and never sweep at all.
    const N: usize = 800_003;
    let data: Vec<i16> = (0..N)
        .map(|i| (i as i32 % 30_000 - 15_000) as i16)
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("contiguous.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("t")
        .with_i16_data(&data)
        .with_shape(&[N as u64]);
    b.write(&path).unwrap();

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();
    assert_eq!(ds.read_i16().unwrap(), data);
    assert_eq!(ds.read_f64().unwrap().len(), N);
    sweep_matches_whole(&ds, N as u64);
}

#[test]
fn a_swept_read_of_an_inner_split_chunk_grid_matches_a_whole_one() {
    // Chunks narrower than the dataset's inner extent, so each one scatters
    // through the N-D kernel into a window-shaped output rather than copying a
    // contiguous row band. 4.3 MiB in rows of 2 KiB: 512 rows to a window, and a
    // final window of 152 rows that ends mid-band.
    const N0: usize = 2200;
    const ROW: usize = 32 * 8;
    let data: Vec<f64> = (0..N0 * ROW).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64, 32, 8])
        .with_chunks(&[64, 16, 4]);
    b.write(&path).unwrap();

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();
    assert_eq!(ds.read_f64().unwrap(), data);
    sweep_matches_whole(&ds, N0 as u64);
}

/// A dataset with no elements at all still reports a datatype it cannot decode.
///
/// The decoders are what raise `TypeMismatch`, so a sweep that ran zero windows
/// over a zero-row dataset would never call one, and a numeric read of a string
/// dataset would come back as an empty vector instead of an error. The empty case
/// is read whole for exactly this reason.
#[test]
fn a_numeric_read_of_an_empty_string_dataset_still_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_strings.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("s").with_vlen_strings(&[]);
    b.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    let ds = file.dataset("s").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0]);
    let err = ds
        .read_f64()
        .expect_err("a string dataset must not decode as f64, empty or not");
    assert!(
        format!("{err}").contains("mismatch") || format!("{err:?}").contains("TypeMismatch"),
        "expected a type mismatch, got {err:?}"
    );
}

/// The sweep reads through the same three backends the whole read does, and
/// through a file whose objects sit past a userblock.
///
/// Every address in a layout message is stored relative to the file's base
/// address, and the reader frames the file at that base before the payload read
/// begins. A sweep issues that read once per window instead of once per dataset,
/// so a framing that were applied per read rather than per byte-range would
/// still come out right on the first window and wrong on the rest — which is the
/// shape of the base-address bugs this crate has had before.
#[test]
fn a_swept_read_agrees_across_backends_and_past_a_userblock() {
    let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.25).collect();
    let dir = tempfile::tempdir().unwrap();

    for userblock in [0u64, 512, 2048] {
        let path = dir.path().join(format!("ub{userblock}.h5"));
        let mut b = FileBuilder::new();
        if userblock > 0 {
            b.with_userblock(userblock);
        }
        b.create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[N as u64])
            .with_chunks(&[1000]);
        b.write(&path).unwrap();

        // Buffered, streaming, and the edit session's read path, which serves
        // reads from a third source again.
        let buffered = File::open(&path).unwrap();
        assert_eq!(
            buffered.dataset("t").unwrap().read_f64().unwrap(),
            data,
            "buffered read past a {userblock}-byte userblock"
        );
        drop(buffered);

        let streaming = File::open_streaming(&path).unwrap();
        assert_eq!(
            streaming.dataset("t").unwrap().read_f64().unwrap(),
            data,
            "streaming read past a {userblock}-byte userblock"
        );
        drop(streaming);

        let session = File::open_rw(&path).unwrap();
        assert_eq!(
            session.root().dataset("t").unwrap().read_f64().unwrap(),
            data,
            "edit-session read past a {userblock}-byte userblock"
        );
    }
}

/// A multi-dimensional dataset in *contiguous* storage: the window is a byte
/// range of the one run, and its rows keep their inner shape.
///
/// Every other multi-dimensional fixture here is chunked, where the window is
/// assembled from chunks rather than sliced out of a single run.
#[test]
fn a_swept_read_of_a_two_dimensional_contiguous_dataset_matches_a_whole_one() {
    const N0: usize = 3001;
    const ROW: usize = 64;
    let data: Vec<f64> = (0..N0 * ROW).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flat2d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64, ROW as u64]);
    b.write(&path).unwrap();

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();
    assert_eq!(ds.read_f64().unwrap(), data);
    sweep_matches_whole(&ds, N0 as u64);
}
