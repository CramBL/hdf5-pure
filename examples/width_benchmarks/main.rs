//! Runs file-width workloads for validation or release measurements.
//!
//! For measurement, writes raw timing samples, per-phase storage and cache observations, and run
//! configuration as JSON. For validation, runs the lifecycle operations and discards timing values.
//! Each sample uses a fresh file with validated schema and contents.

use std::error::Error as StdError;
use std::fmt;
use std::io::Write;
use std::num::NonZeroUsize;
use std::time::Instant;

use hdf5_pure::Error;
use hdf5_pure::LengthWidth;
use hdf5_pure::OffsetWidth;
use test_util_hdf5::__private::width_workload::{Case, Index, Phase};
use test_util_hdf5::file_space::persisted::PersistedStrategy;

use self::report::{Label, Report, Row, Sample, Snapshot};

mod report;

fn main() -> Result<(), Box<dyn StdError>> {
    match Action::parse(std::env::args().skip(1).collect(), cfg!(debug_assertions))? {
        Action::Check => {
            let (supported, unsupported) = check()?;
            println!(
                "validated {supported} cases; {unsupported} unsupported unlimited-maximum configurations"
            );
        }
        Action::Measure(options) => {
            let report = run(options)?;
            let mut output = std::io::stdout().lock();
            serde_json::to_writer(&mut output, &report)?;
            output.write_all(b"\n")?;
        }
    }
    Ok(())
}

/// Validates all supported cases and returns the supported and excluded counts.
///
/// Uses a 65,536-byte metadata-cache budget and discards timing samples.
///
/// # Errors
///
/// Returns [`Error`] from the first failed call to [`sample`].
///
/// # Panics
///
/// Panics if fixture verification or per-pass storage or wire checks fail.
fn check() -> Result<(usize, usize), Error> {
    let (cases, unsupported) = cases();
    for &(case, _) in &cases {
        sample(case, 65_536)?;
    }
    Ok((cases.len(), unsupported.len()))
}

fn run(options: Options) -> Result<Report, Error> {
    let Options {
        samples,
        cache_bytes,
    } = options;
    let (cases, unsupported) = cases();
    let mut report = Report::new(samples.get(), cache_bytes, unsupported);
    let count = NonZeroUsize::new(cases.len());
    let Some(count) = count else {
        return Err(Error::EditUnsupported("benchmark has no supported cases"));
    };
    for &(case, _) in &cases {
        sample(case, cache_bytes)?;
    }
    for round in 0..samples.get() {
        for index in sample_order(count, round) {
            let (case, label) = cases[index];
            let sample = sample(case, cache_bytes)?;
            if round == 0 {
                report.rows.push(Row::new(label, sample));
            } else {
                // Each case is inserted during round zero before another round starts.
                let row = report
                    .rows
                    .iter_mut()
                    .find(|row| row.case == label)
                    .expect("round zero inserts every supported case");
                row.push(sample);
            }
        }
    }
    report.rows.sort_by_key(|row| row.case);
    report.finish();
    Ok(report)
}

fn cases() -> (Vec<(Case, Label)>, Vec<Label>) {
    let mut supported = Vec::new();
    let mut unsupported = Vec::new();
    for offsets in [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight] {
        for lengths in [LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight] {
            for index in [Index::Finite, Index::Extensible] {
                for strategy in [PersistedStrategy::FsmAggr, PersistedStrategy::Page] {
                    for filtered in [false, true] {
                        let label = Label::new(offsets, lengths, index, strategy, filtered);
                        match Case::new(offsets, lengths, index, strategy, filtered) {
                            Ok(case) => supported.push((case, label)),
                            Err(_) => unsupported.push(label),
                        }
                    }
                }
            }
        }
    }
    (supported, unsupported)
}

/// Measures operation intervals and observes each completed file phase.
///
/// Fixture setup, verification, staging, and close are outside the append, commit, and reopen
/// intervals. Creation includes staging and the builder's write. Warmup samples are discarded.
///
/// # Errors
///
/// Returns [`Error`] if temporary-directory setup or cleanup fails. Propagates errors from file
/// operations and snapshot measurement.
///
/// # Panics
///
/// Panics if fixture verification fails or snapshot passes disagree on storage or wire sizes.
fn sample(case: Case, cache_bytes: usize) -> Result<Sample, Error> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("widths.h5");
    let ((), create_ns) = timed(|| case.create(&path))?;
    let created = Snapshot::capture(case, &path, cache_bytes, Phase::Create)?;
    let file = case.open(&path, cache_bytes)?;
    let ((), append_ns) = timed(|| case.append(&file))?;
    file.commit()?;
    file.close()?;
    let appended = Snapshot::capture(case, &path, cache_bytes, Phase::Append)?;
    let file = case.open(&path, cache_bytes)?;
    case.stage_reclamation(&file)?;
    let ((), commit_ns) = timed(|| file.commit())?;
    file.close()?;
    let reclaimed = Snapshot::capture(case, &path, cache_bytes, Phase::Reclaim)?;
    let file = case.open(&path, cache_bytes)?;
    case.stage_reuse(&file)?;
    file.commit()?;
    file.close()?;
    let (file, reopen_ns) = timed(|| case.open(&path, cache_bytes))?;
    case.verify(&file, Phase::Reuse);
    file.close()?;
    let reused = Snapshot::capture(case, &path, cache_bytes, Phase::Reuse)?;
    directory.close()?;
    Ok(Sample {
        create_ns,
        append_ns,
        commit_ns,
        reopen_ns,
        snapshots: [created, appended, reclaimed, reused],
    })
}

fn timed<T>(operation: impl FnOnce() -> Result<T, Error>) -> Result<(T, u128), Error> {
    let start = Instant::now();
    let result = operation()?;
    Ok((result, start.elapsed().as_nanos()))
}

/// Returns a rotation of the case indices, with reverse traversal in odd-numbered rounds.
fn sample_order(count: NonZeroUsize, round: usize) -> Vec<usize> {
    let count = count.get();
    let start = round % count;
    (0..count)
        .map(|offset| {
            let offset = if round % 2 == 0 {
                offset
            } else {
                count - offset
            };
            (start + offset) % count
        })
        .collect()
}

/// The command-line operation selected from arguments and the build profile.
#[derive(Debug, Eq, PartialEq)]
enum Action {
    Check,
    Measure(Options),
}

impl Action {
    /// Selects validation for `--check` or for a debug invocation with no arguments.
    ///
    /// # Errors
    ///
    /// Returns [`Arguments`] if measurement options are invalid or timing is requested in a debug
    /// build.
    fn parse(values: Vec<String>, debug_build: bool) -> Result<Self, Arguments> {
        if values == ["--check"] || (debug_build && values.is_empty()) {
            return Ok(Self::Check);
        }
        let options = Options::parse(values)?;
        if debug_build {
            return Err(Arguments::DebugBuild);
        }
        Ok(Self::Measure(options))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Options {
    samples: NonZeroUsize,
    cache_bytes: usize,
}

impl Options {
    fn parse(values: Vec<String>) -> Result<Self, Arguments> {
        if values.len() > 2 {
            return Err(Arguments::Count {
                actual: values.len(),
            });
        }
        let samples = values.first().map_or(Ok(21), |value| {
            value.parse::<usize>().map_err(|_error| Arguments::Number {
                value: value.clone(),
            })
        })?;
        let samples = NonZeroUsize::new(samples)
            .filter(|samples| samples.get() <= 1_000)
            .ok_or(Arguments::Samples { value: samples })?;
        let cache_bytes = values.get(1).map_or(Ok(65_536), |value| {
            value.parse::<usize>().map_err(|_error| Arguments::Number {
                value: value.clone(),
            })
        })?;
        Ok(Self {
            samples,
            cache_bytes,
        })
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Arguments {
    Count { actual: usize },
    DebugBuild,
    Number { value: String },
    Samples { value: usize },
}

impl fmt::Display for Arguments {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count { actual } => {
                write!(formatter, "expected at most two arguments, got {actual}")
            }
            Self::DebugBuild => formatter.write_str("run width_benchmarks with --release"),
            Self::Number { value } => write!(formatter, "invalid integer {value:?}"),
            Self::Samples { value } => {
                write!(formatter, "sample count {value} is outside 1..=1000")
            }
        }
    }
}

impl StdError for Arguments {}

#[cfg(test)]
mod tests;
