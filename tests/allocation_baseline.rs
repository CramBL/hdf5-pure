//! Exact allocation figures for chunked creation, reading, and appending.
//!
//! This executable has one ignored test and a process-wide allocator. The measured workload
//! includes creating its HDF5 file. Allocation scaling rules run separately in
//! `tests/allocation_bounds.rs`.
//!
//! Exact figures depend on the target, toolchain, and feature set. CI runs this test on Ubuntu
//! x86_64 with the repository's pinned Rust toolchain. The test requires the exact default feature
//! set. The feature list in its `cfg` attribute is maintained by hand and must include each added
//! feature.
//!
//! Run `just soundness::heap-baseline` to compare recorded figures. Record new figures with:
//!
//! ```sh
//! just soundness::heap-baseline-record
//! ```
//!
//! Record new figures after allocation reductions too, so later increases are compared with the
//! reduced counts.
#![cfg(all(
    // Require the feature set used to record the allocation baseline.
    feature = "std",
    feature = "checksum",
    feature = "deflate",
    not(any(
        feature = "fast-deflate",
        feature = "provenance",
        feature = "zfp",
        feature = "ndarray",
        feature = "serde",
        feature = "num-complex",
    ))
))]

use hdf5_pure::{File, FileBuilder, MaxExtent};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Writes and reads an 8 MiB dataset, then appends sixteen chunks to an unlimited dataset.
#[test]
#[ignore = "Measures exact allocation counts in the pinned default-feature environment"]
fn writing_and_reading_a_chunked_dataset_matches_its_recorded_figures() {
    // The fixture's location is built *before* the profiler starts, so nothing
    // it costs is in the baseline. A temporary directory's path length follows
    // `TMPDIR`, and the figures here are exact: measured inside, a longer
    // `TMPDIR` moved `currBytes` from 177 to 672 and reported it as "some path
    // started allocating more", which is the wrong diagnosis for a machine.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("baseline.h5");

    let _profiler = heapscope::Profiler::builder()
        .no_output()
        .build()
        .unwrap_or_else(|e| {
            panic!(
                "heapscope could not start: {e}\n\
                 On x86_64 this is usually the missing frame pointers that \
                 `.cargo/config.toml` sets for that target; a build that \
                 overrides RUSTFLAGS drops them."
            )
        });

    const N0: usize = 1024 * 1024;
    const CHUNK_ELEMS: u64 = 512;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS]);
    // The appends below need an unlimited dimension, and it must be a *second*
    // dataset: `with_maxshape(&[MaxExtent::Unlimited])` is what selects an Extensible Array
    // chunk index over a Fixed Array one (`chunked_write::use_extensible`), so
    // adding it to "t" would have moved the 2,048-chunk write off the index this
    // file exists to pin and left the Fixed Array builder with no exact figures
    // at all.
    builder
        .create_dataset("growing")
        .with_f64_data(&data[..CHUNK_ELEMS as usize])
        .with_shape(&[CHUNK_ELEMS])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK_ELEMS]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let all = file.dataset("t").unwrap().read_raw().unwrap();
    assert_eq!(all.len(), N0 * 8);
    drop(all);
    drop(file);

    // Sixteen appends of a chunk each, which is the write path that runs many
    // times over one file. It is here rather than in `allocation_bounds.rs`
    // because what it costs is a *constant* per call: the rules there bound how
    // an append scales with the dataset, and no bound loose enough to hold on
    // every platform would notice a buffer that doubles its way to the batch
    // size instead of reserving it. This would, as a line in a diff.
    {
        let file = hdf5_pure::File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        let mut ds = file.dataset("growing").unwrap();
        let batch: Vec<f64> = (0..CHUNK_ELEMS).map(|i| i as f64).collect();
        for _ in 0..16 {
            ds.append(&batch).unwrap();
        }
    }

    heapscope::assert_baseline!("tests/baselines/chunked_write_read.txt");
}
