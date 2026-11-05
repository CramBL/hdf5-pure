# File address and length width measurements

The finite, unfiltered `FsmAggr` workload occupies 13,252 bytes after reuse with 2/2 widths,
compared with 15,571 bytes at 8/8. Its retained metadata-read-cache capacity is 4,930 bytes,
compared with 5,842 bytes. The median append interval is 254.803 microseconds at 2/2 and
215.532 microseconds at 8/8. The smaller file and cache footprint coincide with a slower append
median in this run.

[Raw samples and configuration](file_widths.json) cover 48 supported cases. The tables below
select finite, unfiltered `FsmAggr` cases for all nine width pairs. `O/L` denotes address and
length widths in bytes. Each comparison uses the same schema, values, filters, and file-space
settings apart from the two selected widths.

## Reproduction and configuration

Run the [benchmark example](../../examples/width_benchmarks/main.rs) from the repository root:

```sh
cargo run --locked --release --example width_benchmarks -- 21 65536 > /tmp/file_widths.json
```

The optional measurement arguments are samples per case (1 through 1,000) and metadata-read-cache
budget in bytes. Their defaults are 21 and 65,536. Timing runs require a release build.

For validation, run each supported workload once with a 65,536-byte metadata cache and print
supported and excluded counts:

```sh
cargo run --locked --example width_benchmarks -- --check
```

Pass `--check` alone in either build profile. A debug run with no arguments also selects
validation. Timing values are discarded in validation mode.

The recorded run used these settings:

- Producer revision `9f8534dcf16fda8eba1d4945a53c5c39f5076c39`, with a clean tracked tree
- Package version `0.47.0`, release profile, and default `std`, `checksum`, and `deflate` features
- Rust 1.99.1, Linux 7.0.0-42-generic, x86_64, and an Intel Core i5-1135G7
- Fresh temporary files under `/tmp` on filesystem type `tmpfs`
- Bounded editing, strict persisted-space recovery, and `SyncPolicy::OnClose`
- Persistent `FsmAggr` and PAGE strategies, a one-byte free-section threshold, and 512-byte PAGE
  size
- One discarded warmup sample per case and 21 recorded samples per case
- Rotating starting cases and alternating traversal direction between rounds

The runner records 4,032 timing intervals across 48 cases. Finite cases include all nine width
pairs, both strategies, and plain or deflate-level-six series storage. Extensible Array cases
use each address width with eight-byte lengths, both strategies, and both filter settings.
The 24 unlimited-maximum configurations with narrower lengths are listed as exclusions in the
raw report.

Each fixture contains groups with dense links and attributes, contiguous numeric data,
single-chunk and array-indexed numeric data, variable-length strings and their heap, and scratch
allocations separated by surviving data. The lifecycle appends numeric values, deletes scratch
datasets and replaces an attribute, commits reclamation, creates a replacement dataset, and
reopens for editing. `Case::verify` checks schema and contents after every phase. Native
crosschecks independently cover the shared workloads and persisted manager observations.

## Storage and cache

All values in this table are bytes. `Created` is file length after creation, and `Final` is file
length after replacement allocation and reopen. `Free` is persisted reusable space. `Remainder`
is `Final - Free`, including data, metadata, and padding outside the persisted free sections.
`Cache` is retained metadata-read-cache payload capacity after the second validation pass on a
read-only handle.

| O/L | Created | Final | Free | Remainder | Cache |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2/2 | 9,517 | 13,252 | 4,194 | 9,058 | 4,930 |
| 2/4 | 9,659 | 14,292 | 5,076 | 9,216 | 5,056 |
| 2/8 | 9,943 | 14,830 | 5,289 | 9,541 | 5,308 |
| 4/2 | 9,681 | 14,524 | 5,298 | 9,226 | 5,108 |
| 4/4 | 9,823 | 14,700 | 5,302 | 9,398 | 5,234 |
| 4/8 | 10,107 | 15,052 | 5,335 | 9,717 | 5,486 |
| 8/2 | 10,009 | 14,965 | 5,379 | 9,586 | 5,464 |
| 8/4 | 10,151 | 15,144 | 5,396 | 9,748 | 5,590 |
| 8/8 | 10,435 | 15,571 | 5,481 | 10,090 | 5,842 |

All nine final cache observations contain 84 entries. At the 65,536-byte budget, both passes
retain the same capacity, and the second pass records zero misses. The raw report also records
hits, misses, evictions, oversize reads, invalidations, and phase-specific observations. Cache
bytes exclude bookkeeping and other process allocations.

The final 2/2 file is 2,319 bytes (14.9%) smaller than 8/8, and its retained cache capacity is
912 bytes (15.6%) lower. At mixed widths, 2/8 occupies 14,830 bytes and retains 5,308 cache bytes.
The 8/2 file occupies 14,965 bytes and retains 5,464 cache bytes. The two dimensions have distinct
effects in this workload.

## Timing intervals

Each number below is the median of 21 samples, in microseconds. The JSON also retains every
sample and the minimum and maximum for each operation.

| O/L | Create | Append | Commit | Reopen |
| --- | ---: | ---: | ---: | ---: |
| 2/2 | 114.021 | 254.803 | 251.575 | 14.353 |
| 2/4 | 110.183 | 249.305 | 252.379 | 14.215 |
| 2/8 | 111.737 | 251.771 | 252.266 | 14.256 |
| 4/2 | 110.464 | 252.200 | 250.904 | 14.298 |
| 4/4 | 110.323 | 249.478 | 252.821 | 14.272 |
| 4/8 | 112.543 | 253.286 | 256.202 | 14.492 |
| 8/2 | 114.619 | 216.016 | 256.397 | 14.393 |
| 8/4 | 117.241 | 219.469 | 258.958 | 14.352 |
| 8/8 | 115.336 | 215.532 | 257.947 | 14.470 |

The intervals are defined by the following calls:

| Interval | Included work |
| --- | --- |
| Create | Fixture input staging and the builder's completed write |
| Append | Numeric series and single-chunk append calls |
| Commit | Commit after staging scratch deletion and attribute replacement |
| Reopen | Bounded editable open after completed replacement allocation |

Opening, commit, and close are outside the append interval. Staging is outside the commit
interval. Verification and close are outside the reopen interval. Every editing close and its
synchronization are outside the recorded intervals. Creation includes a closed builder write
without an explicit synchronization call.

These measurements use an in-memory filesystem. The OS cache was not flushed, and later intervals
follow earlier writes and validation reads. CPU affinity and frequency were uncontrolled.
The ranges in the raw report include execution variation. Durability-inclusive and cold-cache
costs require separate measurements.

## Field sizes and allocation geometry

The selected wire model covers the serialized superblock, version 2 dataset dataspace bodies
reconstructed from verified shapes and maxima, and the stored string descriptors. It excludes
other encoded fields and allocation padding. The descriptor component is already included in
`dataset_bytes`, so the model and dataset totals overlap.

After reuse, the 2/2 and 8/8 selected models are 166 and 328 bytes, a 162-byte difference.
Their dataset storage is 714 and 732 bytes, and manager allocations are 120 and 306 bytes.
The 2,319-byte file-length difference also includes other encoded structures, placement,
padding, free sections, and manager capacity. It cannot be assigned entirely to the selected
field model.

In the finite PAGE scenario, the unfiltered 8/2 and 8/8 files both end at 15,872 bytes despite a
120-byte selected-model reduction at 8/2. With deflate filtering, the
8/2 final file is 16,384 bytes, compared with 15,872 bytes at 8/8. The PAGE size is 512 bytes.
Shorter fields can change placement and persisted free-space geometry across these boundaries.

## Extensible Array scenario

The unlimited series is a separate scenario from the finite series. The following unfiltered
`FsmAggr` observations use eight-byte lengths and the same recorded run:

| O/L | Final bytes | Cache bytes | Append median (microseconds) |
| --- | ---: | ---: | ---: |
| 2/8 | 14,936 | 5,394 | 267.859 |
| 4/8 | 15,216 | 5,616 | 267.073 |
| 8/8 | 15,851 | 6,060 | 231.493 |

The raw report contains PAGE and filtered cases for both index scenarios, all four lifecycle
snapshots, operation ranges, and per-phase differences from each scenario's matching 8/8 case.
