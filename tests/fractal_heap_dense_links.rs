// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reading dense (fractal-heap) group link storage that the reference C library
//! writes, across group sizes that grow the heap's root indirect block past a
//! single direct block.
//!
//! Regression for a fractal-heap reader bug: the doubling-table row classifier
//! used the `start_root_rows` header field as the direct/indirect boundary
//! instead of the computed `max_direct_rows`. Once a dense group's heap grew a
//! root indirect block with more direct-block rows than that hint (empirically
//! ~150 short links), rows that are actually direct were mis-read as indirect
//! and the reader recursed into a direct block expecting an indirect-block
//! signature, failing with `InvalidFractalHeapSignature`. These groups are read
//! through both the buffered and the streaming reader.

use hdf5_pure::File;
use tempfile::tempdir;

/// Write a group of `n` single-i32 datasets `g/d{i}` with the latest format, so
/// the C library stores the links densely in a fractal heap.
fn write_dense_group(path: &std::path::Path, n: usize) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_latest())
        .create(path)
        .unwrap();
    let g = file.create_group("g").unwrap();
    for i in 0..n {
        g.new_dataset::<i32>()
            .shape((1,))
            .create(format!("d{i}").as_str())
            .unwrap()
            .write(&[i as i32])
            .unwrap();
    }
    file.close().unwrap();
}

#[test]
fn reads_c_written_dense_link_groups_across_sizes() {
    let dir = tempdir().unwrap();
    // 16: spills compact storage into a single fractal-heap direct block.
    // 150: the root grows to a 2-row indirect block — both rows direct — which
    // is the size that used to fail outright. 600: a larger root indirect block
    // with several direct-block rows, exercising the corrected boundary.
    for &n in &[16usize, 150, 600] {
        let path = dir.path().join(format!("dense_{n}.h5"));
        write_dense_group(&path, n);

        // Buffered reader: every link resolves to the right dataset value.
        let buffered = File::open(&path).unwrap();
        for i in 0..n {
            let v = buffered
                .dataset(format!("g/d{i}").as_str())
                .unwrap()
                .read_i32()
                .unwrap();
            assert_eq!(v, vec![i as i32], "buffered n={n} d{i}");
        }

        // Streaming reader: exercises the on-demand indirect-block window sizing
        // (`indirect_block_entries_len`), which shares the same row classifier.
        let streaming = File::open_streaming(&path).unwrap();
        for i in 0..n {
            let v = streaming
                .dataset(format!("g/d{i}").as_str())
                .unwrap()
                .read_i32()
                .unwrap();
            assert_eq!(v, vec![i as i32], "streaming n={n} d{i}");
        }
    }
}
