//! Direct pipeline operations with independently retained outputs.
//!
//! With no arguments, the program retains eight decoded 4 KiB chunks from cycling input through
//! shuffle and Deflate, reusing scratch after two warmup calls. Pass six arguments to choose the
//! pipeline, direction, input pattern, scratch policy, element count, and repetition count.
//!
//! The profiling recipe collects instruction references inside the batch routine. Input
//! preparation, warmup, byte verification, and output destruction occur outside collection.
//! Cold operations construct and drop scratch inside the collected function.

use std::error::Error as StdError;
use std::hint;
use std::num::NonZeroUsize;

use hdf5_pure_filter::__private as filters;
use test_util::filtered_chunks::{Direction, PIPELINES, PipelinePattern, ScratchMode};

use self::support::PreparedPipeline;

#[path = "../benches/filtered_chunks/support.rs"]
mod support;

#[expect(
    clippy::panic_in_result_fn,
    reason = "the profile checks measurement validity"
)]
fn main() -> Result<(), Box<dyn StdError>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() {
        args.extend(
            [
                "shuffle_deflate",
                "decode",
                "cycling",
                "reused",
                "1024",
                "8",
            ]
            .map(String::from),
        );
    }
    let [case, direction, pattern, mode, elements, rounds] = args.as_slice() else {
        return Err(
            "usage: filter_pipeline_profile CASE DIRECTION PATTERN SCRATCH ELEMENTS ROUNDS".into(),
        );
    };
    let case = PIPELINES
        .into_iter()
        .flat_map(|case| std::iter::once(case).chain(case.codec_control()))
        .find(|candidate| candidate.name() == case)
        .ok_or_else(|| format!("unknown pipeline {case:?}"))?;
    let direction = match direction.as_str() {
        "encode" => Direction::Encode,
        "decode" => Direction::Decode,
        value => return Err(format!("unknown direction {value:?}").into()),
    };
    let pattern = match pattern.as_str() {
        "cycling" => PipelinePattern::Cycling,
        "incompressible" => PipelinePattern::Incompressible,
        value => return Err(format!("unknown pattern {value:?}").into()),
    };
    let mode = match mode.as_str() {
        "cold" => ScratchMode::Cold,
        "reused" => ScratchMode::Reused,
        value => return Err(format!("unknown scratch mode {value:?}").into()),
    };
    let elements: NonZeroUsize = elements.parse()?;
    let rounds: NonZeroUsize = rounds.parse()?;
    elements
        .get()
        .checked_mul(4)
        .ok_or("input byte count overflows")?;
    let raw = pattern.data(elements.get());
    let input = PreparedPipeline::new(case, &raw)?;
    verify_control(&input, &raw)?;
    if pattern == PipelinePattern::Incompressible {
        assert!(input.stored.len() >= raw.len());
    }
    let mut scratch = filters::FilterScratch::new();
    if mode == ScratchMode::Reused {
        drop(input.execute(&mut scratch, direction)?);
        drop(input.execute(&mut scratch, direction)?);
    }
    let mut outputs = Vec::with_capacity(rounds.get());
    profile_batch(
        &input,
        &mut scratch,
        direction,
        mode,
        rounds.get(),
        &mut outputs,
    )?;
    assert_eq!(outputs.len(), rounds.get());
    for output in &outputs {
        assert_eq!(output, input.expected(direction));
    }
    println!(
        "PROFILE case={} direction={} pattern={} scratch={} bytes={} rounds={} output_capacity={}",
        case.name(),
        direction.name(),
        pattern.name(),
        mode.name(),
        raw.len(),
        rounds,
        outputs.iter().map(Vec::capacity).sum::<usize>()
    );
    Ok(())
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "the control checks identical encoded bytes before measurement"
)]
fn verify_control(input: &PreparedPipeline, raw: &[u8]) -> Result<(), Box<dyn StdError>> {
    if let Some(control) = input.case.codec_control() {
        assert_eq!(PreparedPipeline::new(control, raw)?.stored, input.stored);
    } else if let Some(pipeline) = PIPELINES
        .into_iter()
        .find(|case| case.codec_control() == Some(input.case))
    {
        assert_eq!(PreparedPipeline::new(pipeline, raw)?.stored, input.stored);
    }
    Ok(())
}

/// Retains every output while running the selected scratch policy.
///
/// # Errors
///
/// Returns the filter error if an operation fails.
#[inline(never)]
fn profile_batch(
    input: &PreparedPipeline,
    scratch: &mut filters::FilterScratch,
    direction: Direction,
    mode: ScratchMode,
    rounds: usize,
    outputs: &mut Vec<Vec<u8>>,
) -> Result<(), filters::Error> {
    for _ in 0..rounds {
        let output = match mode {
            ScratchMode::Cold => input.execute(&mut filters::FilterScratch::new(), direction)?,
            ScratchMode::Reused => input.execute(hint::black_box(&mut *scratch), direction)?,
        };
        outputs.push(hint::black_box(output));
    }
    Ok(())
}
