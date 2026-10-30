//! Direct pipeline throughput with fresh or reused scratch.
//!
//! Cold cases time the first operation and scratch destruction, with empty scratch constructed
//! outside the timed routine. Reused cases retain scratch after two unmeasured calls. Input
//! preparation and destruction of returned buffers are excluded.
//!
//! Compressor controls use independently shuffled input and preserve checksum stages. Cases
//! cover cycling and incompressible input around output and dictionary size boundaries.

use criterion::BatchSize;
use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::measurement::WallTime;
use hdf5_pure_filter::__private as filters;
use test_util::filtered_chunks::{
    Direction, PIPELINE_ELEMENTS, PIPELINES, PipelinePattern, ScratchMode,
};

use self::support::PreparedPipeline;

#[path = "support.rs"]
mod support;

/// Registers encode and decode cases with fresh or reused scratch.
pub(super) fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("filter_pipeline");
    for elements in PIPELINE_ELEMENTS {
        for pattern in [PipelinePattern::Cycling, PipelinePattern::Incompressible] {
            let raw = pattern.data(elements);
            for case in PIPELINES {
                let input = PreparedPipeline::new(case, &raw)
                    .unwrap_or_else(|error| panic!("pipeline fixture failed: {error}"));
                if pattern == PipelinePattern::Incompressible {
                    assert!(input.stored.len() >= raw.len());
                }
                if let Some(control) = case.codec_control() {
                    let control = PreparedPipeline::new(control, &raw)
                        .unwrap_or_else(|error| panic!("codec control fixture failed: {error}"));
                    assert_eq!(control.stored, input.stored);
                    register(&mut group, &control, pattern);
                }
                register(&mut group, &input, pattern);
            }
        }
    }
    group.finish();
}

fn register(
    group: &mut BenchmarkGroup<'_, WallTime>,
    input: &PreparedPipeline,
    pattern: PipelinePattern,
) {
    let label = match pattern {
        PipelinePattern::Cycling => input.case.name().to_owned(),
        PipelinePattern::Incompressible => format!("{}_{}", input.case.name(), pattern.name()),
    };
    for direction in [Direction::Encode, Direction::Decode] {
        group.throughput(Throughput::Bytes(
            u64::try_from(input.raw.len())
                .unwrap_or_else(|error| panic!("invalid byte count: {error}")),
        ));
        let operation = |scratch: &mut filters::FilterScratch| {
            input
                .execute(scratch, direction)
                .unwrap_or_else(|error| panic!("pipeline operation failed: {error}"))
        };
        assert_eq!(
            operation(&mut filters::FilterScratch::new()),
            input.expected(direction)
        );
        for mode in [ScratchMode::Cold, ScratchMode::Reused] {
            group.bench_function(
                BenchmarkId::new(
                    &label,
                    format!("{}/{}/{}", direction.name(), mode.name(), input.raw.len()),
                ),
                |bench| match mode {
                    ScratchMode::Cold => bench.iter_batched(
                        filters::FilterScratch::new,
                        |mut scratch| std::hint::black_box(operation(&mut scratch)),
                        BatchSize::SmallInput,
                    ),
                    ScratchMode::Reused => {
                        let mut scratch = filters::FilterScratch::new();
                        drop(operation(&mut scratch));
                        drop(operation(&mut scratch));
                        bench.iter_batched(
                            || (),
                            |()| std::hint::black_box(operation(&mut scratch)),
                            BatchSize::SmallInput,
                        );
                    }
                },
            );
        }
    }
}
