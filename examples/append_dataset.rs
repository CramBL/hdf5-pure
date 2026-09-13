//! Appending to a filtered, unlimited dataset in place with
//! `Dataset::append_staged` — no SWMR, and without rewriting existing
//! chunks.
//!
//! A rank-1 dataset created with an unlimited dimension and a filter pipeline
//! (here shuffle + deflate) is grown along axis 0 by appending new elements.
//! Existing chunks are kept where they are; only the new chunks — plus the
//! single trailing partial chunk when the current length is not chunk-aligned —
//! are compressed and written, and the chunk index is rebuilt over the whole
//! set. The superblock is repointed last, so a failed commit leaves the
//! original dataset intact.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example append_dataset
//! ```

use hdf5_pure::{File, FileBuilder, MaxExtent};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("log.h5");

    // ---- Create a filtered, unlimited, chunked dataset ------------------
    // One unlimited dimension (`with_maxshape(&[MaxExtent::Unlimited])`) plus `with_chunks`
    // makes this an Extensible-Array-indexed dataset, which is the append
    // target. The filter pipeline is preserved across every append.
    let initial: Vec<i32> = (0..8).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("samples")
        .with_i32_data(&initial)
        .with_shape(&[initial.len() as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4])
        .with_shuffle()
        .with_deflate(6);
    builder.write(&path).expect("write initial file");

    // ---- Check eligibility up front (optional) --------------------------
    // These read-side accessors let a caller decide whether a dataset can be
    // appended to without relying on the append's refusal error.
    {
        let file = File::open(&path).expect("reopen for introspection");
        let ds = file.dataset("samples").expect("open dataset");
        assert!(ds.is_chunked());
        assert_eq!(ds.maxshape().unwrap(), Some(vec![MaxExtent::Unlimited])); // axis 0 unlimited
        assert_eq!(ds.chunk_shape().unwrap(), Some(vec![4]));
        assert_eq!(ds.filters(), vec![2, 1]); // shuffle (2) then deflate (1)
        println!(
            "eligible: chunked={}, maxshape={:?}, chunks={:?}, filters={:?}",
            ds.is_chunked(),
            ds.maxshape().unwrap(),
            ds.chunk_shape().unwrap(),
            ds.filters(),
        );
    } // the buffered reader holds no lock, but drop it before editing anyway

    // ---- Append in place ------------------------------------------------
    // Aligned append: 8 -> 16 (two whole chunks of 4).
    {
        let session = File::open_rw(&path).expect("open for editing");
        session
            .dataset("samples")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[8, 9, 10, 11, 12, 13, 14, 15]);
            })
            .unwrap();
        session.commit().expect("commit aligned append");
    } // drop the session to release its exclusive lock before reopening

    // Unaligned append: 16 -> 21. Since 21 is not a multiple of the chunk
    // length, the trailing partial chunk is read, extended, and re-encoded;
    // every earlier chunk is carried by metadata alone, so existing data is not
    // rewritten and the file does not grow by the whole dataset per append.
    {
        let session = File::open_rw(&path).expect("open for editing");
        session
            .dataset("samples")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[16, 17, 18, 19, 20]);
            })
            .unwrap();
        session.commit().expect("commit unaligned append");
    }

    // ---- Verify ---------------------------------------------------------
    let file = File::open(&path).expect("reopen");
    let all = file.dataset("samples").unwrap().read_i32().unwrap();
    println!("dataset now holds {} samples: {all:?}", all.len());
    assert_eq!(all, (0..21).collect::<Vec<_>>());
    println!("in-place append verified");
}
