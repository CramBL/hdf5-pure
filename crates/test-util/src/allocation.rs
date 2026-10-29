//! Measuring what one piece of work allocates, from inside a test binary.
//!
//! # What the including binary has to declare
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
//! ```
//!
//! A `#[global_allocator]` is a property of the binary, so it cannot live here.
//! Leaving it out is not a silent miss: [`measure`] refuses to start a profiler
//! that would record nothing, naming the line above.
#![allow(dead_code)]

use std::sync::OnceLock;

/// What one measured piece of work allocated.
///
/// Every figure is that work's own: allocations made on another thread, or on
/// this one outside the measured call, are not in it. That is what lets several
/// measured tests share a process — which they must, since the profiler is
/// process-wide and `cargo test` runs a binary's tests concurrently.
///
/// **It is also the way a test built on this goes quietly hollow.** Work that
/// moves to a worker thread stops being measured, and every figure here collapses
/// toward zero while an upper bound on it goes on passing. This is not
/// hypothetical for this crate: the read path has a parallel variant waiting to
/// be written. So a test that asserts a ceiling must assert a floor beside it —
/// the read's own output is in these figures, so its size is a floor that costs
/// one line and cannot be satisfied by a measurement of nothing.
#[derive(Clone, Copy, Debug)]
pub struct Measured {
    /// Allocations made, counting a reallocation as one.
    pub blocks: u64,
    /// Bytes ever allocated, freed or not.
    pub bytes: u64,
    /// The most this work held at once, in bytes.
    pub peak_bytes: u64,
    /// Bytes it allocated that were still live when it returned.
    pub live_bytes: u64,
}

impl Measured {
    /// Formats allocation metrics as an aligned report with comma-separated numbers across multiple lines.
    pub fn snapshot_report(&self) -> String {
        format!(
            "Allocation count:  {:>13}\n\
             Cumulative bytes:  {:>13}\n\
             Peak bytes:        {:>13}\n\
             Retained bytes:    {:>13}",
            grouped_decimal(self.blocks),
            grouped_decimal(self.bytes),
            grouped_decimal(self.peak_bytes),
            grouped_decimal(self.live_bytes),
        )
    }
}

impl std::fmt::Display for Measured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} allocations, {} bytes allocated, {} bytes at peak, {} still live",
            self.blocks, self.bytes, self.peak_bytes, self.live_bytes
        )
    }
}

/// The process's one profiler, started on first use and kept for the run.
static PROFILER: OnceLock<()> = OnceLock::new();

fn start_profiling() {
    PROFILER.get_or_init(|| {
        let profiler = heapscope::Profiler::builder()
            .no_output()
            .build()
            .unwrap_or_else(|e| {
                panic!(
                    "heapscope could not start: {e}\n\
                     On x86_64 this is usually the missing frame pointers that \
                     `.cargo/config.toml` sets for that target; a build that \
                     overrides RUSTFLAGS drops them."
                )
            });
        // The profiler must outlive every test in this binary, and there is
        // nowhere to put it: dropping it ends the run, and a `static` would need
        // it to be `Sync`. Nothing is lost by forgetting it — the builder above
        // asks for no output, so its drop had nothing left to do.
        #[expect(clippy::mem_forget, reason = "the profiler outlives the test binary")]
        std::mem::forget(profiler);
    });
}

/// Runs `work` inside a named region and returns what that region alone recorded.
///
/// `name` must be unique within the binary: heapscope interns region names, so
/// two measurements sharing one would be added together.
///
/// # Every reading that would be a guess panics instead
///
/// heapscope's [`heapscope::HeapStats::get`] returns an error when
/// the run cannot be trusted, and its documentation is explicit about why: a
/// getter that answered zero would turn every budget built on it into an
/// assertion that cannot fail. Reading a region goes through `Snapshot` instead,
/// which reports those conditions rather than refusing on them, so this is where
/// they have to be checked — otherwise this helper would be exactly the
/// zero-returning getter that design exists to prevent.
#[track_caller]
pub fn measure<R>(name: &str, work: impl FnOnce() -> R) -> (R, Measured) {
    start_profiling();

    measure_region(name, work)
}

/// Runs `work` inside a named region using an already running profiler.
///
/// The `_profiler` argument borrows an active profiler. Uses the same region
/// measurement and safety checks as [`measure`].
///
/// `name` must be unique within the binary: measurements with the same region
/// name are accumulated.
///
/// # Panics
///
/// Panics if the snapshot is poisoned or inexact, allocation or attribution
/// records were dropped, or the region is absent from the snapshot. Also
/// propagates panics from `work`.
#[track_caller]
pub fn measure_with_profiler<R>(
    _profiler: &heapscope::Profiler,
    name: &str,
    work: impl FnOnce() -> R,
) -> (R, Measured) {
    measure_region(name, work)
}

#[track_caller]
fn measure_region<R>(name: &str, work: impl FnOnce() -> R) -> (R, Measured) {
    let result = {
        let _region = heapscope::region(name);
        work()
    };

    let snapshot = heapscope::Snapshot::capture();
    assert!(
        !snapshot.poisoned,
        "the profiler reported an internal failure during this run, so {name:?}'s \
         figures may be missing allocations"
    );
    assert!(
        snapshot.exact,
        "the profiler could not reach a quiet point, so {name:?}'s figures are not \
         consistent with each other"
    );
    assert_eq!(
        snapshot.stats.dropped_blocks, 0,
        "the live-block table filled up, so every figure for {name:?} is missing \
         however many allocations it turned away"
    );
    assert_eq!(
        snapshot.rows_dropped, 0,
        "the profiler turned away attribution rows, and a region is one, so \
         {name:?}'s row may be short of what it should hold"
    );

    let stats = snapshot
        .regions
        .into_iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| {
            // Not the "it allocated nothing" case: a region is interned when it
            // is entered, so an empty one still has a row. This is a profiler
            // that recorded nothing at all.
            panic!("no region named {name:?} in the profile: the profiler is not recording")
        });

    let measured = Measured {
        blocks: stats.counts.total_blocks,
        bytes: stats.counts.total_bytes,
        peak_bytes: stats.counts.max_bytes,
        live_bytes: stats.counts.curr_bytes,
    };
    eprintln!("{name}: {measured}");
    (result, measured)
}

fn grouped_decimal(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Measured;

    #[rstest]
    #[case::zero(0, "0")]
    #[case::below_thousand(999, "999")]
    #[case::thousand(1_000, "1,000")]
    #[case::below_million(999_999, "999,999")]
    #[case::million(1_000_000, "1,000,000")]
    #[case::readable_example(1_300_233, "1,300,233")]
    #[case::maximum(u64::MAX, "18,446,744,073,709,551,615")]
    fn decimal_numbers_have_thousands_separators(#[case] value: u64, #[case] expected: &str) {
        assert_eq!(super::grouped_decimal(value), expected);
    }

    #[test]
    fn snapshot_report_labels_and_aligns_allocation_metrics() {
        let measured = Measured {
            blocks: 11_668,
            bytes: 1_082_754,
            peak_bytes: 1_082_720,
            live_bytes: 740_000,
        };
        assert_eq!(
            measured.snapshot_report(),
            "Allocation count:         11,668\n\
             Cumulative bytes:      1,082,754\n\
             Peak bytes:            1,082,720\n\
             Retained bytes:          740,000"
        );
    }
}
