//! Streaming in-place appends through an owned handle — open the file once with
//! [`File::open_rw`] and append many times at amortized `O(1)` index cost.
//!
//! `File::open_rw` opens an existing file for reading and writing; each
//! `Dataset::append` grows the dataset's Extensible-Array index *in place*, so it
//! never re-reads the file or rebuilds the index the way `Dataset::append_staged`
//! does. Every append is crash-atomic (the dataspace dimension is published
//! last), and the result reads back in the reference C library and h5py.
//!
//! Appends are **any-length** and may start at any length, on filtered and
//! unfiltered datasets alike: a partial trailing chunk is rewritten into a fresh
//! allocation, re-encoded through the filter pipeline when there is one.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example append_streaming
//! ```

use hdf5_pure::{File, FileBuilder, MaxExtent};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("stream.h5");

    // ---- Create an unlimited, chunked dataset ---------------------------
    // One unlimited dimension plus `with_chunks` makes this an
    // Extensible-Array-indexed dataset (this crate allocates the index eagerly,
    // so it can be grown from the very first append).
    let initial: Vec<i32> = (0..6).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("samples")
        .with_i32_data(&initial)
        .with_shape(&[initial.len() as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    builder.write(&path).expect("write initial file");

    // ---- Stream appends through one open, read-write file ---------------
    // The dataset starts at length 6 with a chunk length of 4, so its trailing
    // chunk is partial (2 of 4). Each append below is an arbitrary length: the
    // partial chunk is rewritten into a fresh allocation, its index element is
    // repointed once those bytes are on the disk, then any further whole chunks
    // are inserted. `File::open_rw` holds an exclusive lock for the file's
    // lifetime.
    {
        let file = File::open_rw(&path).expect("open for appending");
        let mut samples = file.dataset("samples").expect("open dataset handle");
        samples.append(&[6i32, 7, 8]).expect("append 1"); // -> 9
        samples.append(&[9i32, 10]).expect("append 2"); // -> 11
        samples
            .append(&(11..20).collect::<Vec<i32>>())
            .expect("append 3"); // -> 20
        file.close().expect("flush and release the lock");
    } // the lock is released here; dropping the file would release it too

    // ---- Verify ---------------------------------------------------------
    let file = File::open(&path).expect("reopen");
    let all = file.dataset("samples").unwrap().read_i32().unwrap();
    println!("dataset now holds {} samples: {all:?}", all.len());
    assert_eq!(all, (0..20).collect::<Vec<_>>());
    println!("streaming in-place append verified");
}
