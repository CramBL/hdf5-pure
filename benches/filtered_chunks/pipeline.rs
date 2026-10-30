//! Direct pipeline throughput with fresh or reused scratch.
//!
//! Cold cases time the first operation and scratch destruction, with empty scratch constructed
//! outside the timed routine. Reused cases retain scratch after two unmeasured calls. Input
//! preparation and destruction of returned buffers are excluded.

use core::num::NonZeroU32;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use hdf5_pure_filter::__private as filters;
use test_util::filtered_chunks::{self, Direction, PIPELINES, PipelineFilter};

/// Registers encode and decode cases with fresh or reused scratch.
pub(super) fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("filter_pipeline");
    for elements in [1024, 16_384] {
        let raw = filtered_chunks::pipeline_data(elements);
        let dims = [u64::try_from(elements).unwrap()];
        let context = filters::ChunkContext {
            chunk_dims: &dims,
            element_size: NonZeroU32::new(4).unwrap(),
            element_type: None,
            scale_offset_type: None,
        };
        for case in PIPELINES {
            let pipeline = Steps(case.filters());
            let stored = filters::compress_chunk_with(
                &mut filters::FilterScratch::new(),
                &raw,
                &pipeline,
                context,
            )
            .unwrap();
            assert_eq!(
                filters::decompress_chunk_with(
                    &mut filters::FilterScratch::new(),
                    &stored,
                    &pipeline,
                    context,
                    0
                )
                .unwrap(),
                raw
            );
            for direction in [Direction::Encode, Direction::Decode] {
                group.throughput(Throughput::Bytes(u64::try_from(raw.len()).unwrap()));
                let operation = |scratch: &mut filters::FilterScratch| match direction {
                    Direction::Encode => {
                        filters::compress_chunk_with(scratch, &raw, &pipeline, context).unwrap()
                    }
                    Direction::Decode => {
                        filters::decompress_chunk_with(scratch, &stored, &pipeline, context, 0)
                            .unwrap()
                    }
                };
                group.bench_function(
                    BenchmarkId::new(
                        case.name(),
                        format!("{}/cold/{}", direction.name(), raw.len()),
                    ),
                    |bench| {
                        bench.iter_batched(
                            filters::FilterScratch::new,
                            |mut scratch| std::hint::black_box(operation(&mut scratch)),
                            BatchSize::SmallInput,
                        );
                    },
                );
                group.bench_function(
                    BenchmarkId::new(
                        case.name(),
                        format!("{}/reused/{}", direction.name(), raw.len()),
                    ),
                    |bench| {
                        let mut scratch = filters::FilterScratch::new();
                        drop(operation(&mut scratch));
                        drop(operation(&mut scratch));
                        bench.iter_batched(
                            || (),
                            |()| std::hint::black_box(operation(&mut scratch)),
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

struct Steps<'a>(&'a [PipelineFilter]);

impl filters::FilterSteps for Steps<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn id(&self, index: usize) -> u16 {
        match self.0[index] {
            PipelineFilter::Deflate => filters::FILTER_DEFLATE,
            PipelineFilter::Fletcher32 => filters::FILTER_FLETCHER32,
            PipelineFilter::Lzf => filters::FILTER_LZF,
            PipelineFilter::Shuffle => filters::FILTER_SHUFFLE,
        }
    }
    fn flags(&self, _index: usize) -> u16 {
        0
    }
    fn client_data(&self, index: usize) -> &[u32] {
        match self.0[index] {
            PipelineFilter::Deflate => &[6],
            PipelineFilter::Shuffle => &[4],
            PipelineFilter::Fletcher32 | PipelineFilter::Lzf => &[],
        }
    }
}
