//! Filtered creation and direct pipeline throughput.
//!
//! Creation measurements exclude builder setup. Pipeline calls are timed with fresh and reused
//! scratch.

use core::time::Duration;

use criterion::BatchSize;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use test_util::filtered_chunks::WORKLOADS;
use test_util::filtered_chunks::Workload;

#[path = "filtered_chunks/pipeline.rs"]
mod pipeline;

fn filtered_creation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("filtered_creation");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    for workload in WORKLOADS {
        let data = workload.data();
        let image = builder(workload, &data).finish().unwrap();
        assert_eq!(
            File::from_bytes(image)
                .unwrap()
                .dataset("values")
                .unwrap()
                .read_u32()
                .unwrap(),
            data
        );
        group.throughput(Throughput::Bytes(u64::try_from(data.len()).unwrap() * 4));
        group.bench_function(workload.name, |bench| {
            bench.iter_batched(
                || builder(workload, &data),
                |writer| std::hint::black_box(writer.finish().unwrap()),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn builder(workload: Workload, data: &[u32]) -> FileBuilder {
    let Workload {
        name: _,
        shape,
        chunks,
        fill,
        lzf,
        random: _,
    } = workload;
    let mut builder = FileBuilder::new();
    let dataset = builder
        .create_dataset("values")
        .with_u32_data(data)
        .with_shape(&shape)
        .with_chunks(&chunks)
        .with_fill_value(fill)
        .with_shuffle();
    if lzf {
        dataset.with_lzf();
    }
    builder
}

criterion_group!(benches, filtered_creation, pipeline::benchmark);
criterion_main!(benches);
