// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Property-based cross-validation of the file-space (`fcpl`) and bounded-mutation
//! (`fapl`) surface against the reference HDF5 C library (issue #178).
//!
//! The hand-written crosschecks in `file_space_crosscheck.rs` spot-check a few
//! fixed configurations. These properties fuzz the same invariants across a
//! generated space of **strategy**, **persist**, **page size**, **threshold**,
//! and **append patterns**, diffing every generated file against libhdf5:
//!
//! 1. **Strategy round-trip.** For any strategy / persist / page size / threshold
//!    and any set of datasets, a file hdf5-pure writes reads back byte-exact in
//!    the C library, and the C library recovers the same strategy. For a
//!    persisting *paged* file the strong invariant also holds: the C library's
//!    `H5Fget_freespace` (which can only answer by parsing our on-disk
//!    `FSHD`/`FSSE` managers) equals the sum of the free sections we wrote.
//!
//! 2. **Bounded paged mutation.** For any page size and any sequence of appends,
//!    growing a genuine paged persisting file through the bounded engine
//!    yields a file whose every row, recorded strategy, and manager free-space
//!    total the C library reads back exactly.
//!
//! A failing case is shrunk to its minimal reproducer and recorded under
//! `tests/proptest-regressions/`; commit that file so the reproducer travels
//! with the fix.

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::{File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MemoryStrategy};

/// Open with the bounded engine demanded rather than merely preferred: these
/// tests are about that engine, so a file it stops accepting must fail here
/// rather than quietly retarget the whole file at the mirror.
fn open_bounded(path: &std::path::Path) -> Result<File, hdf5_pure::Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
}

use proptest::prelude::*;
use tempfile::tempdir;

/// The four file-space strategies, each mapped to the name the C library reports.
fn strategy() -> impl Strategy<Value = FileSpaceStrategy> {
    prop_oneof![
        Just(FileSpaceStrategy::FsmAggr),
        Just(FileSpaceStrategy::Page),
        Just(FileSpaceStrategy::Aggr),
        Just(FileSpaceStrategy::None),
    ]
}

/// Valid file-space page sizes: powers of two `>= 512`, the range the paged
/// writer accepts. Under a non-paged strategy the value is only recorded.
fn page_size() -> impl Strategy<Value = u64> {
    prop_oneof![
        Just(512u64),
        Just(1024u64),
        Just(2048u64),
        Just(4096u64),
        Just(8192u64),
        Just(16384u64),
    ]
}

/// What the C library's `get_file_space_strategy` reports for a file hdf5-pure
/// wrote with `(strategy, persist, threshold)`.
fn expected_c_strategy(strategy: FileSpaceStrategy, persist: bool, threshold: u64) -> CStrategy {
    match strategy {
        FileSpaceStrategy::FsmAggr => CStrategy::FreeSpaceManager {
            paged: false,
            persist,
            threshold,
        },
        FileSpaceStrategy::Page => CStrategy::FreeSpaceManager {
            paged: true,
            persist,
            threshold,
        },
        // The C library reports AGGR under its `PageAggregation` name and does not
        // surface persist/threshold for it or for NONE.
        FileSpaceStrategy::Aggr => CStrategy::PageAggregation,
        FileSpaceStrategy::None => CStrategy::None,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// A file hdf5-pure writes with any `(strategy, persist, page_size, threshold)`
    /// and any datasets reads back byte-exact in the C library, which recovers the
    /// same strategy; and a persisting paged file's manager total matches
    /// `H5Fget_freespace` exactly.
    #[test]
    fn strategy_and_page_size_roundtrip_through_c(
        strategy in strategy(),
        persist in any::<bool>(),
        threshold in 0u64..=8,
        page_size in page_size(),
        // A mix of small (< page) and large (>= page) datasets: i32 is 4 bytes, so
        // up to 8000 elements spans several pages even at the largest page size.
        lens in prop::collection::vec(1usize..=8000usize, 1..=3),
    ) {
            let dir = tempdir().unwrap();
        let path = dir.path().join("fuzz_strategy.h5");

        let datasets: Vec<Vec<i32>> =
            lens.iter().map(|&n| (0..n as i32).collect()).collect();

        let mut b = FileBuilder::new();
        for (i, data) in datasets.iter().enumerate() {
            b.create_dataset(&format!("d{i}")).with_i32_data(data);
        }
        b.with_file_space_strategy(strategy, persist, threshold)
            .with_file_space_page_size(page_size);
        b.write(&path).unwrap();

        // hdf5-pure reads its own datasets back, and reports its tracked free space.
        let total_ours: u64 = {
            let ours = File::open(&path).unwrap();
            for (i, data) in datasets.iter().enumerate() {
                let got = ours.dataset(&format!("d{i}")).unwrap().read_i32().unwrap();
                prop_assert_eq!(&got, data);
            }
            ours.persisted_free_space().iter().map(|(_, l)| l).sum()
        };

        // The reference C library recovers the strategy and reads every dataset.
        // This includes a fresh `persist = true` non-paged file, which must record a
        // defined `eoa_pre_fsm` (issue #178) or an assertion-enabled libhdf5 aborts
        // on open (H5Fsuper.c: `fs_persist => eoa_fsm_fsalloc != UNDEF`).
        let f = hdf5::File::open(&path).unwrap();
        prop_assert_eq!(
            f.create_plist().unwrap().get_file_space_strategy().unwrap(),
            expected_c_strategy(strategy, persist, threshold)
        );
        for (i, data) in datasets.iter().enumerate() {
            let got = f.dataset(&format!("d{i}")).unwrap().read_raw::<i32>().unwrap();
            prop_assert_eq!(&got, data);
        }
        // Loading a persisting file's free-space managers requires parsing our
        // on-disk records; the C library's total must equal our own tally exactly
        // (a paged file tracks its page tails, a fresh non-paged file tracks none).
        if persist {
            let free_c = f.free_space() as i64;
            prop_assert_eq!(free_c as u64, total_ours);
        }
        drop(f);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Growing a genuine paged persisting file through the bounded backend, for any
    /// page size and any append pattern, yields a file the C library reads back
    /// row-for-row with a manager free-space total matching `H5Fget_freespace`.
    #[test]
    fn bounded_paged_append_roundtrips_through_c(
        page_size in page_size(),
        chunk_size in prop_oneof![Just(32usize), Just(64usize), Just(128usize), Just(256usize)],
        // 1..=5 append calls of 1..=1500 unfiltered (so any-length) rows each.
        appends in prop::collection::vec(1usize..=1500usize, 1..=5),
    ) {
            let dir = tempdir().unwrap();
        let path = dir.path().join("fuzz_append.h5");

        // Create a paged, persisting, unlimited chunked i32 dataset with one chunk.
        {
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..chunk_size as i32).collect::<Vec<i32>>())
                .with_shape(&[chunk_size as u64])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[chunk_size as u64]);
            b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
                .with_file_space_page_size(page_size);
            b.write(&path).unwrap();
        }

        // Grow it through the bounded backend with the generated append pattern.
        // Scope the handles so the Dataset clone and File both drop (releasing the
        // exclusive OS lock, which is mandatory on Windows) before the C open.
        let mut next = chunk_size as i32;
        {
            let file = open_bounded(&path).unwrap();
            let mut ds = file.dataset("d").unwrap();
            for &count in &appends {
                let chunk: Vec<i32> = (next..next + count as i32).collect();
                ds.append(&chunk).unwrap();
                next += count as i32;
            }
            drop(ds);
            file.close().unwrap();
        }

        let want: Vec<i32> = (0..next).collect();

        // hdf5-pure reads the full sequence back and reports its tracked free space.
        let total_ours: u64 = {
            let ours = File::open(&path).unwrap();
            let got = ours.dataset("d").unwrap().read_i32().unwrap();
            prop_assert_eq!(&got, &want);
            ours.persisted_free_space().iter().map(|(_, l)| l).sum()
        };

        // The C library recovers the paged strategy, reads every row, and its
        // free-space total equals the sum of the managers we rewrote at close.
        let f = hdf5::File::open(&path).unwrap();
        prop_assert_eq!(
            f.create_plist().unwrap().get_file_space_strategy().unwrap(),
            CStrategy::FreeSpaceManager {
                paged: true,
                persist: true,
                threshold: 0,
            }
        );
        let got = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
        prop_assert_eq!(&got, &want);
        let free_c = f.free_space() as i64;
        prop_assert_eq!(free_c as u64, total_ours);
        drop(f);
    }
}
