//! Allocation measurement for a region on the calling thread.
//!
//! The including binary installs the allocator:
//!
//! ```
//! #[global_allocator]
//! static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
//! ```
//!
//! Give each measurement a unique region name within its binary, including parameter cases.
//! Allocations on other threads or outside the measured call are excluded. Keep fixture paths,
//! inputs, and assertions outside the call unless they are part of the measured operation.
//! Return owned results from the call to include their retained allocations in
//! [`Measured::live_bytes`], and check their complete values and state after measurement.
//!
//! Pair an upper bound with a known allocation floor or a separate allocating control. Check
//! the control before asserting zero allocations. [`measure`] rejects poisoned or inexact
//! snapshots, dropped allocation or attribution records, and absent regions.
//!
//! Scaling bounds run separately from pinned exact baselines. Diagnostic heap profiles support
//! manual inspection, and Criterion benchmarks measure throughput.
#![allow(dead_code)]

use std::sync::OnceLock;

/// Allocation counts and byte totals for one measured region.
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
/// # Panics
///
/// Panics if the profiler cannot start, the snapshot is poisoned or inexact,
/// allocation or attribution records were dropped, or the region is absent from
/// the snapshot. Also propagates panics from `work`.
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
