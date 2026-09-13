//! SWMR (single writer, multiple readers): append to an unlimited dataset in
//! place while a reader follows the growing file.
//!
//! The writer appends chunks and flushes durably so readers only ever observe a
//! consistent prefix; a reader calls `refresh()` to pick up new data. This
//! interoperates with the reference HDF5 C library and h5py in both directions.
//! This single-process example writes and then reads to show the mechanics; in
//! practice the writer and reader are separate processes.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example swmr
//! ```

use hdf5_pure::{File, FileBuilder, MaxExtent};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("stream.h5");

    // The dataset must have one unlimited dimension and be chunked. The latest
    // format indexes it with an Extensible Array automatically.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("log")
        .with_i32_data(&[0, 1, 2]) // initial rows
        .with_shape(&[3])
        .with_maxshape(&[MaxExtent::Unlimited]) // one unlimited dimension
        .with_chunks(&[1]);
    builder.write(&path).expect("write initial file");

    // A reader opens the file before the appends happen.
    let mut reader = File::open_swmr(&path).expect("open reader");
    let rows_before = reader.dataset("log").unwrap().shape().unwrap()[0];
    println!("reader sees {rows_before} rows initially");

    // The writer appends in place. Each call flushes durably, leaving the file
    // valid for the concurrent reader throughout.
    let writer = File::open_swmr_writer(&path).expect("open writer");
    writer.dataset("log").unwrap().append(&[3, 4, 5]).unwrap();
    writer.dataset("log").unwrap().append(&[6, 7]).unwrap();
    writer.close().unwrap(); // clears the SWMR flag; dropping the writer also works

    // The reader refreshes to observe the appended data.
    reader.refresh().unwrap();
    let ds = reader.dataset("log").unwrap();
    let rows_after = ds.shape().unwrap()[0];
    let values = ds.read_i32().unwrap();
    println!("after refresh: {rows_after} rows = {values:?}");

    assert_eq!(rows_before, 3);
    assert_eq!(rows_after, 8);
    assert_eq!(values, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    println!("\nSWMR append + refresh verified");
}
