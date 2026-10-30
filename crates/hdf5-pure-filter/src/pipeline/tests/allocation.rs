//! Direct pipeline allocation measurements with cold and reused scratch.
//!
//! Cold regions retain the first output and newly initialized scratch. Reused regions retain
//! independent outputs after two warmup calls and include their descriptor container. Region
//! metrics cover allocations made inside the measured call.
//!
//! Output capacity, general spare capacity, and initialized extent are reported separately.
//! General spare capacity includes storage acquired during warmup and excludes codec state.

use core::mem;

use rstest::rstest;
use test_util::allocation;
use test_util::allocation::Measured;
use test_util::filtered_chunks::{self, Direction, PipelineCase, PipelineFilter, PipelinePattern};

use super::*;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

const REPEATS: usize = 8;

#[rstest]
#[case::shuffle(PipelineCase::Shuffle)]
#[case::fletcher32(PipelineCase::Fletcher32)]
#[case::shuffle_fletcher32(PipelineCase::ShuffleFletcher32)]
#[case::shuffle_lzf(PipelineCase::ShuffleLzf)]
#[case::shuffle_deflate(PipelineCase::ShuffleDeflate)]
#[case::shuffle_deflate_fletcher32(PipelineCase::ShuffleDeflateFletcher32)]
fn pipeline_allocations_separate_setup_outputs_and_scratch(
    #[case] case: PipelineCase,
    #[values(Direction::Encode, Direction::Decode)] direction: Direction,
    #[values(1023, 1024, 1025, 8191, 8192, 8193, 16_384)] elements: usize,
    #[values(PipelinePattern::Cycling, PipelinePattern::Incompressible)] pattern: PipelinePattern,
) {
    let raw = pattern.data(elements);
    let filters = pipeline(case, 0);
    let stored = run(
        &mut FilterScratch::new(),
        &raw,
        &[],
        &filters,
        Direction::Encode,
        0,
    );
    let expected = if direction == Direction::Encode {
        &stored
    } else {
        &raw
    };
    if pattern == PipelinePattern::Incompressible {
        assert!(stored.len() >= raw.len());
    }
    let measurement = PipelineMeasurement {
        case,
        direction,
        pattern,
        bytes: raw.len(),
    };
    let label = measurement.label();
    let ((mut scratch, first), cold) = allocation::measure(&format!("{label}_cold"), || {
        let mut scratch = FilterScratch::new();
        let output = run(&mut scratch, &raw, &stored, &filters, direction, 0);
        (scratch, output)
    });
    assert_eq!(&first, expected);
    assert!(cold.blocks > 0, "{cold}");
    let first_capacity = first.capacity();
    let retained = cold
        .live_bytes
        .checked_sub(u64::try_from(first_capacity).unwrap())
        .unwrap();
    let cold_spare = intermediate_capacity(&scratch);
    measurement.report("cold", cold, first_capacity, &scratch);
    drop(first);
    drop(run(&mut scratch, &raw, &stored, &filters, direction, 0));
    let (outputs, warm) = allocation::measure(&format!("{label}_reused"), || {
        batch(&mut scratch, &raw, &stored, &filters, direction)
    });
    assert_outputs(&outputs, expected);
    let escaping = outputs.iter().map(Vec::capacity).sum::<usize>();
    let container = outputs.capacity() * mem::size_of::<Vec<u8>>();
    let spare = intermediate_capacity(&scratch);
    measurement.report("reused", warm, escaping, &scratch);
    assert!(warm.blocks >= u64::try_from(REPEATS).unwrap(), "{warm}");
    assert!(
        warm.live_bytes <= u64::try_from(escaping + container + spare).unwrap(),
        "{warm}"
    );
    assert!(
        escaping <= REPEATS * capacity_bound(raw.len()),
        "{escaping}"
    );
    assert!(spare <= capacity_bound(raw.len()), "{spare}");
    let transient = if let Some(control) = case.codec_control() {
        let shuffled = filtered_chunks::shuffled_pipeline_data(&raw);
        let control_filters = pipeline(control, 0);
        let ((mut control_scratch, output), setup) =
            allocation::measure(&format!("{label}_control_cold"), || {
                let mut scratch = FilterScratch::new();
                let output = run(
                    &mut scratch,
                    &shuffled,
                    &stored,
                    &control_filters,
                    direction,
                    0,
                );
                (scratch, output)
            });
        let control_expected = if direction == Direction::Encode {
            &stored
        } else {
            &shuffled
        };
        assert_eq!(&output, control_expected);
        let control_measurement = PipelineMeasurement {
            case: control,
            ..measurement
        };
        control_measurement.report("cold", setup, output.capacity(), &control_scratch);
        let codec = setup.live_bytes - u64::try_from(output.capacity()).unwrap();
        assert_eq!(retained, codec + u64::try_from(cold_spare).unwrap());
        drop(output);
        drop(run(
            &mut control_scratch,
            &shuffled,
            &stored,
            &control_filters,
            direction,
            0,
        ));
        let (control_outputs, measured) =
            allocation::measure(&format!("{label}_control_reused"), || {
                batch(
                    &mut control_scratch,
                    &shuffled,
                    &stored,
                    &control_filters,
                    direction,
                )
            });
        assert_outputs(&control_outputs, control_expected);
        control_measurement.report(
            "reused",
            measured,
            control_outputs.iter().map(Vec::capacity).sum(),
            &control_scratch,
        );
        assert!(
            measured.blocks >= u64::try_from(REPEATS).unwrap(),
            "{measured}"
        );
        assert!(
            warm.blocks <= measured.blocks,
            "pipeline {warm}; codec control {measured}"
        );
        assert!(
            warm.bytes <= measured.bytes + u64::try_from(capacity_bound(raw.len())).unwrap(),
            "pipeline {warm}; codec control {measured}"
        );
        measured.peak_bytes.saturating_sub(measured.live_bytes)
    } else {
        assert_eq!(retained, u64::try_from(cold_spare).unwrap());
        let per_output =
            usize::from(case == PipelineCase::ShuffleFletcher32 && direction == Direction::Encode)
                + 1;
        assert!(
            warm.blocks <= u64::try_from(REPEATS * per_output + 1).unwrap(),
            "{warm}"
        );
        0
    };
    assert!(warm.peak_bytes >= warm.live_bytes, "{warm}");
    assert!(
        warm.peak_bytes
            <= warm.live_bytes + transient + u64::try_from(2 * capacity_bound(raw.len())).unwrap(),
        "{warm}"
    );
}

#[rstest]
#[case::shuffle(PipelineCase::Shuffle)]
#[case::fletcher32(PipelineCase::Fletcher32)]
#[case::shuffle_fletcher32(PipelineCase::ShuffleFletcher32)]
#[case::shuffle_lzf(PipelineCase::ShuffleLzf)]
#[case::shuffle_deflate(PipelineCase::ShuffleDeflate)]
#[case::shuffle_deflate_fletcher32(PipelineCase::ShuffleDeflateFletcher32)]
fn pipeline_reuse_bounds_size_changes_masks_and_errors(#[case] case: PipelineCase) {
    let filters = pipeline(case, 0);
    let schedules = [
        (16_384, 0),
        (1, 1),
        (1024, u32::MAX),
        (2, 1 << (filters.len() - 1)),
        (16_384, 0xffff_ff80),
    ];
    let fixtures: Vec<_> = schedules
        .into_iter()
        .map(|(elements, mask)| {
            let raw = filtered_chunks::pipeline_data(elements);
            let active = pipeline(case, mask);
            let stored = run(
                &mut FilterScratch::new(),
                &raw,
                &[],
                &active,
                Direction::Encode,
                0,
            );
            let encoded = run(
                &mut FilterScratch::new(),
                &raw,
                &[],
                &filters,
                Direction::Encode,
                0,
            );
            Fixture {
                raw,
                stored,
                encoded,
                mask,
            }
        })
        .collect();
    let Fixture { raw, stored, .. } = &fixtures[0];
    let (bad, expected_error, error_elements) =
        if case.filters().contains(&PipelineFilter::Fletcher32) {
            let mut bad = stored.clone();
            let computed = u32::from_le_bytes(bad[bad.len() - 4..].try_into().unwrap());
            *bad.last_mut().unwrap() ^= 0xff;
            (
                bad,
                Error::Fletcher32Mismatch {
                    expected: computed ^ 0xff00_0000,
                    computed,
                },
                raw.len() / 4,
            )
        } else if case.filters().contains(&PipelineFilter::Deflate) {
            (
                stored[..stored.len() - 4].to_vec(),
                Error::FilterError("deflate: stream ended before the chunk was complete".into()),
                raw.len() / 4,
            )
        } else if case.filters().contains(&PipelineFilter::Lzf) {
            (
                vec![0],
                Error::InvalidLzfStream("truncated literal run"),
                raw.len() / 4,
            )
        } else {
            (
                stored.clone(),
                Error::DataSizeMismatch {
                    expected: raw.len() + 4,
                    actual: raw.len(),
                },
                raw.len() / 4 + 1,
            )
        };
    let mut scratch = FilterScratch::new();
    let warmup = sequence(&mut scratch, &fixtures, &filters, &bad, error_elements);
    assert_sequence(&warmup, &fixtures, &expected_error);
    drop(warmup);
    let (observed, measured) =
        allocation::measure(&format!("pipeline_{}_varied", case.name()), || {
            sequence(&mut scratch, &fixtures, &filters, &bad, error_elements)
        });
    assert_sequence(&observed, &fixtures, &expected_error);
    let escaping = observed.outputs.iter().map(Vec::capacity).sum::<usize>();
    let bound = capacity_bound(raw.len());
    assert!(
        observed.max_spare <= bound,
        "{} > {bound}",
        observed.max_spare
    );
    assert!(intermediate_capacity(&scratch) <= bound);
    assert!(escaping <= observed.outputs.len() * bound);
    assert!(
        measured.blocks >= u64::try_from(observed.outputs.len()).unwrap(),
        "{measured}"
    );
    let error_bytes = match &observed.codec_error {
        Error::FilterError(reason) => reason.capacity(),
        _ => 0,
    };
    let container = observed.outputs.capacity() * mem::size_of::<Vec<u8>>();
    assert!(
        measured.live_bytes <= u64::try_from(escaping + container + error_bytes + bound).unwrap(),
        "{measured}"
    );
    assert!(measured.peak_bytes >= measured.live_bytes);
    assert!(
        measured.peak_bytes <= measured.live_bytes + u64::try_from(3 * bound).unwrap(),
        "{measured}"
    );
    assert!(
        measured.bytes <= u64::try_from((fixtures.len() + 4) * 3 * bound).unwrap(),
        "{measured}"
    );
    PipelineMeasurement {
        case,
        direction: Direction::Decode,
        pattern: PipelinePattern::Cycling,
        bytes: raw.len(),
    }
    .report("varied", measured, escaping, &scratch);
    eprintln!(
        "VARIED case={} max_spare_between_calls={} error_bytes={error_bytes}",
        case.name(),
        observed.max_spare
    );
    let mut encoder = FilterScratch::new();
    let mut rejected = pipeline(case, 0);
    rejected.push(FilterDescription {
        filter_id: 40_000,
        flags: 0,
        client_data: vec![],
    });
    let warmup = encode_sequence(&mut encoder, &fixtures, &filters, &rejected);
    assert_encoded_sequence(&warmup, &fixtures);
    drop(warmup);
    let (encoded, encoded_metrics) =
        allocation::measure(&format!("pipeline_{}_varied_encode", case.name()), || {
            encode_sequence(&mut encoder, &fixtures, &filters, &rejected)
        });
    assert_encoded_sequence(&encoded, &fixtures);
    let escaping = encoded.outputs.iter().map(Vec::capacity).sum::<usize>();
    let container = encoded.outputs.capacity() * mem::size_of::<Vec<u8>>();
    assert!(encoded.max_spare <= bound);
    assert!(escaping <= encoded.outputs.len() * bound);
    assert!(encoded_metrics.blocks >= u64::try_from(encoded.outputs.len()).unwrap());
    assert!(encoded_metrics.live_bytes <= u64::try_from(escaping + container + bound).unwrap());
    assert!(
        encoded_metrics.peak_bytes
            <= encoded_metrics.live_bytes + u64::try_from(3 * bound).unwrap()
    );
    assert!(encoded_metrics.bytes <= u64::try_from((fixtures.len() + 4) * 3 * bound).unwrap());
    PipelineMeasurement {
        case,
        direction: Direction::Encode,
        pattern: PipelinePattern::Cycling,
        bytes: raw.len(),
    }
    .report("varied", encoded_metrics, escaping, &encoder);
}

struct Fixture {
    raw: Vec<u8>,
    stored: Vec<u8>,
    encoded: Vec<u8>,
    mask: u32,
}

struct SequenceResult {
    outputs: Vec<Vec<u8>>,
    codec_error: Error,
    size_error: Error,
    max_spare: usize,
}

struct EncodedSequence {
    outputs: Vec<Vec<u8>>,
    error: Error,
    max_spare: usize,
}

fn encode_sequence(
    scratch: &mut FilterScratch,
    fixtures: &[Fixture],
    filters: &[FilterDescription],
    rejected: &[FilterDescription],
) -> EncodedSequence {
    let mut maximum = intermediate_capacity(scratch);
    let mut outputs = Vec::with_capacity(fixtures.len() + 1);
    for fixture in fixtures {
        outputs.push(run(
            scratch,
            &fixture.raw,
            &[],
            filters,
            Direction::Encode,
            0,
        ));
        maximum = maximum.max(intermediate_capacity(scratch));
    }
    let raw = &fixtures[0].raw;
    let dims = [u64::try_from(raw.len() / 4).unwrap()];
    let error =
        compress_chunk_with(scratch, raw, rejected, ChunkContext::basic(&dims, 4)).unwrap_err();
    maximum = maximum.max(intermediate_capacity(scratch));
    outputs.push(run(scratch, raw, &[], filters, Direction::Encode, 0));
    EncodedSequence {
        outputs,
        error,
        max_spare: maximum.max(intermediate_capacity(scratch)),
    }
}

fn assert_encoded_sequence(observed: &EncodedSequence, fixtures: &[Fixture]) {
    assert_eq!(observed.outputs.len(), fixtures.len() + 1);
    for (output, fixture) in observed.outputs.iter().zip(fixtures) {
        assert_eq!(output, &fixture.encoded);
    }
    assert_eq!(observed.outputs.last().unwrap(), &fixtures[0].encoded);
    assert_eq!(observed.error, Error::UnsupportedFilter(40_000));
}

fn sequence(
    scratch: &mut FilterScratch,
    fixtures: &[Fixture],
    filters: &[FilterDescription],
    bad: &[u8],
    error_elements: usize,
) -> SequenceResult {
    let mut maximum = intermediate_capacity(scratch);
    let mut outputs = Vec::with_capacity(fixtures.len() + 1);
    for Fixture {
        raw, stored, mask, ..
    } in fixtures
    {
        outputs.push(run(scratch, raw, stored, filters, Direction::Decode, *mask));
        maximum = maximum.max(intermediate_capacity(scratch));
    }
    let error_dims = [u64::try_from(error_elements).unwrap()];
    let error = decompress_chunk_with(
        scratch,
        bad,
        filters,
        ChunkContext::basic(&error_dims, 4),
        0,
    )
    .unwrap_err();
    maximum = maximum.max(intermediate_capacity(scratch));
    let Fixture { raw, stored, .. } = &fixtures[0];
    let rejected_dims = [u64::try_from(raw.len() / 4 + 1).unwrap()];
    let rejected = decompress_chunk_with(
        scratch,
        stored,
        filters,
        ChunkContext::basic(&rejected_dims, 4),
        0,
    )
    .unwrap_err();
    maximum = maximum.max(intermediate_capacity(scratch));
    outputs.push(run(scratch, raw, stored, filters, Direction::Decode, 0));
    maximum = maximum.max(intermediate_capacity(scratch));
    SequenceResult {
        outputs,
        codec_error: error,
        size_error: rejected,
        max_spare: maximum,
    }
}

fn assert_sequence(observed: &SequenceResult, fixtures: &[Fixture], error: &Error) {
    assert_eq!(observed.outputs.len(), fixtures.len() + 1);
    for (output, fixture) in observed.outputs.iter().zip(fixtures) {
        assert_eq!(output, &fixture.raw);
    }
    assert_eq!(observed.outputs.last().unwrap(), &fixtures[0].raw);
    assert_eq!(&observed.codec_error, error);
    assert_eq!(
        observed.size_error,
        Error::DataSizeMismatch {
            expected: fixtures[0].raw.len() + 4,
            actual: fixtures[0].raw.len()
        }
    );
}

fn batch(
    scratch: &mut FilterScratch,
    raw: &[u8],
    stored: &[u8],
    filters: &[FilterDescription],
    direction: Direction,
) -> Vec<Vec<u8>> {
    (0..REPEATS)
        .map(|_| run(scratch, raw, stored, filters, direction, 0))
        .collect()
}

fn run(
    scratch: &mut FilterScratch,
    raw: &[u8],
    stored: &[u8],
    filters: &[FilterDescription],
    direction: Direction,
    mask: u32,
) -> Vec<u8> {
    let dims = [u64::try_from(raw.len() / 4).unwrap()];
    let context = ChunkContext::basic(&dims, 4);
    match direction {
        Direction::Encode => compress_chunk_with(scratch, raw, filters, context).unwrap(),
        Direction::Decode => {
            decompress_chunk_with(scratch, stored, filters, context, mask).unwrap()
        }
    }
}

fn pipeline(case: PipelineCase, mask: u32) -> Vec<FilterDescription> {
    case.filters()
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1 << index) == 0)
        .map(|(_, filter)| {
            let (filter_id, client_data) = match filter {
                PipelineFilter::Deflate => (FILTER_DEFLATE, vec![6]),
                PipelineFilter::Fletcher32 => (FILTER_FLETCHER32, vec![]),
                PipelineFilter::Lzf => (FILTER_LZF, vec![]),
                PipelineFilter::Shuffle => (FILTER_SHUFFLE, vec![4]),
            };
            FilterDescription {
                filter_id,
                flags: 0,
                client_data,
            }
        })
        .collect()
}

fn assert_outputs(outputs: &[Vec<u8>], expected: &[u8]) {
    assert_eq!(outputs.len(), REPEATS);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

fn capacity_bound(bytes: usize) -> usize {
    2 * (bytes + 4) + 4096
}

fn intermediate_capacity(scratch: &FilterScratch) -> usize {
    scratch.spare.capacity()
}

fn intermediate_initialized_len(scratch: &FilterScratch) -> usize {
    scratch.spare.initialized_len()
}

#[derive(Clone, Copy)]
struct PipelineMeasurement {
    case: PipelineCase,
    direction: Direction,
    pattern: PipelinePattern,
    bytes: usize,
}

impl PipelineMeasurement {
    fn label(&self) -> String {
        let Self {
            case,
            direction,
            pattern,
            bytes,
        } = self;
        let pattern = match pattern {
            PipelinePattern::Cycling => "cycle",
            PipelinePattern::Incompressible => "random",
        };
        format!("{}_{}_{pattern}_{bytes}", case.name(), direction.name())
    }

    fn report(&self, phase: &str, measured: Measured, output: usize, scratch: &FilterScratch) {
        let Self {
            case,
            direction,
            pattern,
            bytes,
        } = self;
        let spare = intermediate_capacity(scratch);
        let initialized = intermediate_initialized_len(scratch);
        assert!(initialized <= spare);
        eprintln!(
            "PIPELINE case={} direction={} pattern={} bytes={bytes} phase={phase} allocations={} allocated={} live={} peak={} output_capacity={output} spare_capacity={spare} initialized_bytes={initialized}",
            case.name(),
            direction.name(),
            pattern.name(),
            measured.blocks,
            measured.bytes,
            measured.live_bytes,
            measured.peak_bytes
        );
    }
}
