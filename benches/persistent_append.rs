//! Measures separate and batched full-chunk appends with persistent free space at threshold one.
//!
//! Times the append loop, excluding fixture construction, opening, verification, and close.

use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use test_util_hdf5::file_space::publication;

fn persistent_append(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("persistent_threshold_one_append");
    group.sample_size(10);
    for (chunks, batch) in [(2000, 1), (2000, 250), (8000, 1), (8000, 1000)] {
        let values = publication::values(chunks);
        group.bench_function(BenchmarkId::new(chunks.to_string(), batch), |bench| {
            bench.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let directory = tempfile::tempdir().unwrap();
                    let path = directory.path().join("append.h5");
                    let file = publication::session(&path);
                    let mut dataset = file.dataset("d").unwrap();
                    let start = Instant::now();
                    for values in values.chunks(batch * publication::CHUNK) {
                        dataset.append(std::hint::black_box(values)).unwrap();
                    }
                    elapsed += start.elapsed();
                    drop(dataset);
                    file.close().unwrap();
                    publication::verify(&path, chunks);
                }
                elapsed
            });
        });
    }
    group.finish();
}

criterion_group!(benches, persistent_append);
criterion_main!(benches);
