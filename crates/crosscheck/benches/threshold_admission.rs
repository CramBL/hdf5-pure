//! Measures complete commit time for contiguous chunk retirement beside an eligible seed.
//!
//! Each sample uses a fresh native or pure flat fixture with bounded editing and close-time sync.
//! Fixture creation, opening, deletion staging, closing, and verification are outside the timed
//! region. Each sample checks persisted accounting, surviving values, and writable recovery.

use std::fs;
use std::process::ExitCode;

use hdf5_pure::MemoryStrategy;
use test_util_hdf5::file_space::admission::Chain;
use test_util_hdf5::file_space::admission::Fixture;
use test_util_hdf5::file_space::admission::Observation;
use test_util_hdf5::file_space::admission::Origin;
use test_util_hdf5::file_space::admission::Scale;
use test_util_hdf5::file_space::admission::Threshold;

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!("commit measurements require an optimized build");
        return ExitCode::FAILURE;
    }
    println!(
        "native={:?} memory=Bounded sync=OnClose",
        hdf5::library_version()
    );
    let directory = tempfile::tempdir().unwrap();
    for origin in [Origin::Native, Origin::Pure] {
        for scale in [
            Scale::TwoThousand,
            Scale::FourThousand,
            Scale::EightThousand,
        ] {
            for threshold in [Threshold::One, Threshold::Allocation] {
                let chain = Chain::Flat { scale, threshold };
                let mut samples = Vec::new();
                for repetition in 0..3 {
                    let path = directory.path().join(format!(
                        "{origin:?}-{}-{}-{repetition}.h5",
                        chain.chunks(),
                        chain.threshold()
                    ));
                    let Observation {
                        elapsed,
                        initial_bytes,
                        final_bytes,
                        free,
                        sections,
                        manager_bytes,
                    } = Fixture::create(&path, origin, chain)
                        .commit(&path, MemoryStrategy::Bounded);
                    println!(
                        "origin={origin:?} chunks={} threshold={} repetition={repetition} initial_bytes={} final_bytes={} sections={} reusable_bytes={} manager_bytes={} commit_us={}",
                        chain.chunks(),
                        chain.threshold(),
                        initial_bytes,
                        final_bytes,
                        sections,
                        free.intervals()
                            .iter()
                            .map(|interval| interval.length())
                            .sum::<u64>(),
                        manager_bytes,
                        elapsed.as_micros()
                    );
                    samples.push(elapsed);
                    fs::remove_file(&path).unwrap();
                }
                samples.sort_unstable();
                println!(
                    "median origin={origin:?} chunks={} threshold={} commit_us={}",
                    chain.chunks(),
                    chain.threshold(),
                    samples[1].as_micros()
                );
            }
        }
    }
    ExitCode::SUCCESS
}
