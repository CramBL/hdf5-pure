//! The exact allocation figures for one write-then-read cycle, recorded rather
//! than chosen (issue #228).
//!
//! `tests/allocation_bounds.rs` states the rules — a windowed read allocates on
//! the order of its window, a chunked read costs a constant per chunk — and those
//! run everywhere. This file is the other half: the numbers themselves, committed
//! under `tests/baselines/` so a change to them arrives in a pull request as a
//! line a reviewer reads rather than as nothing at all. A 5% creep passes every
//! rule in this suite and shows up here.
//!
//! # Why this is behind a feature, one platform, and one feature set
//!
//! An exact allocation count is a property of a target, a toolchain *and* a
//! feature set, not of this crate alone: a `Vec` growth policy, an `OsString` on
//! one platform and a `String` on another, or a new `std` release all move it
//! without anything here changing. The feature set is not hypothetical —
//! `--all-features` measured 416 bytes above the default set on the same machine,
//! in the same commit. So the figures are recorded where they can be checked (see
//! the `heap-baseline` job in `.github/workflows/soundness.yml`) and the default suite
//! runs the rules instead. Widening this to a second platform is a matter of
//! measuring it there, not of loosening anything.
//!
//! Hence the cfg below, which names the default set from both sides: the three
//! default features must be on and every other one off. That is the set the
//! figures were recorded under and the set a dependent's build most resembles.
//! Under `--all-features` — the pre-push gate — it compiles to nothing rather
//! than failing against numbers from another configuration.
//!
//! **The list is maintained by hand and nothing will tell you when it is stale.**
//! A feature added to the crate and not added here simply widens the set this
//! accepts, silently, because `--all-features` turns on `fast-deflate` and so
//! compiles this out whatever else is added. There is no version of this cfg that
//! self-corrects; adding a feature to the crate means adding it here.
//!
//! # When it fails
//!
//! Read the named figures: some path started allocating more, and the message
//! says which figure and by how much.
//!
//! The comparison is one-sided — heapscope reports only figures that *grew* — so
//! a number that went **down** never fails and never appears. That is worth
//! knowing because it is how the gate goes slack: an improvement leaves the file
//! recording the worse old number, and the difference becomes headroom a later
//! regression can spend for free. Re-record after an improvement too; nothing
//! will remind you.
//!
//! Re-record with:
//!
//! ```sh
//! HEAPSCOPE_UPDATE_BASELINE=1 cargo test --features heap-baseline --test allocation_baseline
//! ```
//!
//! and commit the diff with the change that caused it.
//!
//! # What is measured
//!
//! The whole process, which is why this binary holds exactly one `#[test]`: the
//! profiler is process-wide, and `HeapStats` cannot be built from a
//! `heapscope::Region`'s figures (it is `#[non_exhaustive]`), so a baseline is a
//! reading of the run rather than of a phase inside it. The consequence worth
//! knowing is that the fixture below is part of the measurement: changing what it
//! writes changes the numbers legitimately. The upside is that the write path is
//! pinned here too, and it allocates more than the read does.
#![cfg(all(
    feature = "heap-baseline",
    // The default set, required to be present: `--no-default-features` with
    // `heap-baseline` alone would otherwise check a different build's figures
    // against this file.
    feature = "std",
    feature = "checksum",
    feature = "deflate",
    // And required to be the whole of it.
    not(any(
        feature = "fast-deflate",
        feature = "provenance",
        feature = "zfp",
        feature = "ndarray",
        feature = "serde",
        feature = "num-complex",
        feature = "matio-crosscheck",
    ))
))]

use hdf5_pure::{File, FileBuilder};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Write an 8 MiB chunked dataset, read it back whole, and append sixteen chunks
/// to an unlimited one beside it: 2,048 chunks through the writer's chunk
/// assembly and the reader's coalescing span planner, where all but a constant
/// of the allocations in either path live, and then the in-place append path,
/// whose cost is a constant per call that no scaling rule can pin.
#[test]
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
    // dataset: `with_maxshape(&[u64::MAX])` is what selects an Extensible Array
    // chunk index over a Fixed Array one (`chunked_write::use_extensible`), so
    // adding it to "t" would have moved the 2,048-chunk write off the index this
    // file exists to pin and left the Fixed Array builder with no exact figures
    // at all.
    builder
        .create_dataset("growing")
        .with_f64_data(&data[..CHUNK_ELEMS as usize])
        .with_shape(&[CHUNK_ELEMS])
        .with_maxshape(&[u64::MAX])
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
