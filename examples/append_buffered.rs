//! Buffering appends into whole chunks with `Dataset::buffered_appender`.
//!
//! `Dataset::append` writes on every call, which is the wrong trade for a
//! logger appending a handful of samples at a time into a chunk that holds many:
//! it pays a whole crash-atomic write sequence to add ten elements, and on a
//! filtered dataset it re-encodes the trailing chunk each time.
//!
//! A `BufferedAppender` holds appended elements in memory and writes them only
//! when they complete a chunk, so the file sees one append per chunk rather
//! than one per call, and any append length works. Buffered elements are not in
//! the file until `flush` or `finish` puts them there.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example append_buffered
//! ```

use hdf5_pure::{File, FileBuilder, MaxExtent};

const CHUNK: u64 = 64;
const BATCH: i32 = 10;
const BATCHES: i32 = 25;

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("telemetry.h5");

    // ---- A filtered, unlimited, chunked dataset, initially empty ----------
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("samples")
        .with_i32_data(&[])
        .with_shape(&[0])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK])
        .with_shuffle()
        .with_deflate(6);
    builder.write(&path).expect("write initial file");

    // ---- Append ten at a time into a chunk of sixty-four -------------------
    let mut writes = 0usize;
    {
        let session = File::open_rw(&path).expect("open for editing");
        let mut samples = session.dataset("samples").expect("open dataset");
        let mut appender = samples.buffered_appender().expect("eligible for appending");

        let mut size = session.file_size();
        for b in 0..BATCHES {
            let lo = b * BATCH;
            appender
                .append(&(lo..lo + BATCH).collect::<Vec<i32>>())
                .expect("append");
            // The file only changes where a chunk completed.
            let now = session.file_size();
            if now != size {
                writes += 1;
                size = now;
            }
        }
        println!(
            "{BATCHES} calls of {BATCH} elements into a chunk of {CHUNK}: \
             {writes} writes, {} elements still buffered",
            appender.buffered_elements()
        );
        // The buffered tail reaches the file here. `Drop` would flush too, but
        // could not report a failure.
        appender.finish().expect("flush the tail");
    }

    // Twenty-five calls of ten elements fill three chunks of sixty-four and
    // leave 250 - 192 = 58 buffered, so the loop wrote three times.
    assert_eq!(writes, 3, "expected one write per completed chunk");

    // ---- Verify -----------------------------------------------------------
    let file = File::open(&path).expect("reopen");
    let all = file.dataset("samples").unwrap().read_i32().unwrap();
    let expected: Vec<i32> = (0..BATCH * BATCHES).collect();
    assert_eq!(all, expected);
    println!("dataset holds {} samples, all accounted for", all.len());

    // ---- Resuming the log -------------------------------------------------
    // The dataset now ends mid-chunk (250 of 64). A new appender lands it back
    // on a chunk boundary with its first write, which re-encodes that trailing
    // chunk, and every write after it starts on a boundary.
    {
        let session = File::open_rw(&path).expect("reopen for editing");
        let mut samples = session.dataset("samples").expect("open dataset");
        let mut appender = samples.buffered_appender().expect("still eligible");
        for b in 0..10i32 {
            let lo = BATCH * BATCHES + b * BATCH;
            appender
                .append(&(lo..lo + BATCH).collect::<Vec<i32>>())
                .expect("append");
        }
        appender.finish().expect("flush the tail");
    }

    let file = File::open(&path).expect("reopen");
    let all = file.dataset("samples").unwrap().read_i32().unwrap();
    assert_eq!(all, (0..BATCH * BATCHES + 100).collect::<Vec<i32>>());
    println!("resumed log holds {} samples", all.len());
    println!("buffered append verified");
}
