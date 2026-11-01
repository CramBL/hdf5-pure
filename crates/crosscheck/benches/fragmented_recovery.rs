//! Measures bounded writable opening of native fragmented PAGE files.
//!
//! Reports three samples and their median for each fragmentation scale and recovery policy.
//! Times only opening, excluding fixture construction, inspection, close, and survivor checks.

use std::hint;
use std::process::ExitCode;
use std::time::Instant;

use hdf5_pure::File;
use hdf5_pure::PersistedFreeSpacePolicy;
use test_util_hdf5::file_space::fragmentation;
use test_util_hdf5::file_space::fragmentation::Fragmentation;

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!("writable-open measurements require an optimized build");
        return ExitCode::FAILURE;
    }
    println!("native={:?}", hdf5::library_version());
    let directory = tempfile::tempdir().unwrap();
    for scale in [
        Fragmentation::Thousand,
        Fragmentation::TwoThousand,
        Fragmentation::FourThousand,
        Fragmentation::EightThousand,
    ] {
        let path = directory.path().join(format!("{}.h5", scale.holes()));
        scale.write_native(&path);
        let before = fragmentation::persisted(&path);
        let bytes = std::fs::metadata(&path).unwrap().len();
        for policy in [
            PersistedFreeSpacePolicy::Tolerant,
            PersistedFreeSpacePolicy::Strict,
        ] {
            let mut samples = Vec::new();
            for repetition in 0..3 {
                let options = fragmentation::bounded_options(policy);
                let start = Instant::now();
                let file = File::open_rw_with_options(&path, options).unwrap();
                let elapsed = start.elapsed();
                hint::black_box(&file);
                assert!(
                    file.reusable_space_stats().unwrap().reusable_free_bytes()
                        >= scale.raw_free_bytes()
                );
                file.close().unwrap();
                assert_eq!(fragmentation::persisted(&path), before);
                samples.push(elapsed);
                println!(
                    "holes={} policy={policy:?} repetition={repetition} sections={} bytes={bytes} open_us={}",
                    scale.holes(),
                    before.free_sections().section_count(),
                    elapsed.as_micros(),
                );
            }
            scale.assert_native_survivors(&path);
            samples.sort_unstable();
            println!(
                "median holes={} policy={policy:?} sections={} bytes={bytes} open_us={}",
                scale.holes(),
                before.free_sections().section_count(),
                samples[1].as_micros(),
            );
        }
    }
    ExitCode::SUCCESS
}
