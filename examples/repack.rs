//! Reclaiming space and dropping objects with `repack`.
//!
//! Deleting an object inside a `File::open_rw` session reuses the freed space within that
//! session, but a single delete cannot shrink a file whose freed region is not
//! at the very end (the same reason the HDF5 C library ships `h5repack`).
//! `repack` reads every surviving object and rewrites the whole file compact,
//! optionally dropping objects along the way.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example repack
//! ```

use hdf5_pure::{File, FileBuilder, RepackOptions};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let input = dir.path().join("input.h5");
    let output = dir.path().join("compact.h5");

    // A file with a large "scratch" dataset we no longer need, a whole group
    // subtree to drop, and the data we want to keep.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("keep")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    builder
        .create_dataset("scratch")
        .with_f64_data(&vec![0.0; 50_000]);
    let mut aborted = builder.create_group("runs");
    aborted
        .create_dataset("aborted")
        .with_f64_data(&vec![0.0; 50_000]);
    builder.add_group(aborted.finish());
    builder.write(&input).expect("write input");

    let before = std::fs::metadata(&input).unwrap().len();

    // Rewrite compact, dropping a dataset and a whole subtree.
    let options = RepackOptions::new().drop_path("scratch").drop_path("runs");
    hdf5_pure::repack(&input, &output, &options).expect("repack");

    let after = std::fs::metadata(&output).unwrap().len();
    println!("input:  {before:>8} bytes");
    println!("output: {after:>8} bytes (scratch + runs dropped)");

    // Surviving data is reproduced exactly; dropped objects are gone.
    let file = File::open(&output).unwrap();
    let keep = file.dataset("keep").unwrap().read_f64().unwrap();
    println!("kept 'keep' = {keep:?}");
    println!(
        "'scratch' present after repack: {}",
        file.dataset("scratch").is_ok()
    );

    assert_eq!(keep, vec![1.0, 2.0, 3.0]);
    assert!(file.dataset("scratch").is_err());
    assert!(file.group("runs").is_err());
    assert!(after < before);
    println!("\nrepack shrank the file and preserved the kept data");
}
