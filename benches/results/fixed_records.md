# Fixed-record decoding comparison

The production implementation uses dependency-free fixed records. The `zerocopy` candidate has
lower latency in several parsing workloads, including the nested fixture and File Space Info.
The owned-value APIs and generated field declarations provide the required composition without a
production dependency. The parsing gains and borrowed views do not justify the additional derive
maintenance and build cost for these APIs. The exact `zerocopy` 0.8.63 development dependency and
experimental overlay remain available for reproduction.

## Sources and measurement conditions

The public comparison uses the same 57-case `bounded_byte_reads` harness for four sources:

| Source | Revision or preparation |
| --- | --- |
| Pre-PR | `1ff3bf482fbe6a1cd009ac1021bc6ddb791bda4e` |
| Reviewed PR | `8bf4a1d63d41c2e4b3db6025bf5dc1485db57a6c` |
| Bounded records | Revised implementation with descriptor inlining and conditional manager preparation |
| `zerocopy` records | Revised implementation with `scripts/experiments/zerocopy-records.patch` applied |

Each source uses an isolated Cargo target directory. All four dependency graphs use `zerocopy`
0.8.63 and `zerocopy-derive` 0.8.63. The derive dependencies are `proc-macro2` 1.0.106, `quote`
1.0.45, `syn` 2.0.117, and `unicode-ident` 1.0.24. Only the experimental overlay adds these to
the format crate's production dependencies.

Measurements used Rust 1.99.1, LLVM 23.1.1, `x86_64-unknown-linux-gnu`, and CPU 0 of an Intel Core
Ultra 7 255H. Public cases used Cargo's bench profile and the root crate's default features. The
private comparison used one release-profile binary for both representations with equivalent profile
settings. Frame pointers were enabled. Each case used a 500 ms warmup, a two-second measurement,
100 samples, and Criterion's 95% confidence intervals with 100,000 bootstrap draws. Accepted runs
had no concurrent builds or tests.

Fixtures and correctness assertions run outside timed iterations. Inputs, widths, results, and
written bytes pass through `black_box`. Parsed values are dropped inside each iteration. Immediate
encoding includes checked construction. Prepared encoding reuses a value constructed outside timing.
Repeated decoding allocates one output vector in both representations.

## Latency

[Public results](fixed_records_public.csv) contain every absolute timing and 95% interval for all
four sources, with changes relative to both baselines. [Private results](fixed_records_private.csv)
contain all 64 paired workloads for the two representations. Percentage intervals use 10,000
bootstrap draws of iteration/time pairs, selected independently for each source with replacement,
using seed 1236.
Negative percentages mean lower latency relative to the named baseline.

The table below gives the range of point estimates across each public workload group. The CSV
intervals describe within-run variability. Earlier valid sweeps also showed between-run variation.
Small differences within noise are not evidence of a speedup. Results apply to this compiler and
hardware configuration.

| Selected bounded records | Cases | Change from pre-PR | Change from reviewed PR |
| --- | ---: | ---: | ---: |
| Dataspace, rank zero | 6 | -37.9% to -2.0% | -34.4% to -6.3% |
| Dataspace, nonempty | 18 | -78.4% to -1.0% | -0.9% to +22.7% |
| Descriptor, parse | 3 | -73.5% to -73.0% | -58.8% to -58.0% |
| Descriptor, parse four unaligned records | 3 | -22.9% to -22.7% | -0.6% to -0.2% |
| Descriptor, prepared write | 3 | -54.0% to -52.7% | -11.8% to -8.2% |
| Descriptor, construct and write | 3 | -53.4% to -52.9% | -25.2% to -24.6% |
| Symbol table, unaligned parse | 3 | -59.5% to -52.2% | -60.2% to -43.8% |
| File Space Info, unaligned parse | 9 | -58.5% to -56.2% | -45.0% to -39.4% |
| File Space Info, serialize | 9 | -10.7% to +3.3% | -15.9% to -1.8% |

Rank-zero Dataspace parsing recovers the reviewed PR's regression. Every nonempty point estimate
is below pre-PR, with higher latency than the reviewed PR in several cases. The complete table
includes the small serialization differences without treating them as material application effects.

The `zerocopy` File Space Info parser has 11.0% to 14.0% lower latency than the bounded candidate.
The private nested parser takes 5.35 nanoseconds (95% interval 5.34–5.37 nanoseconds), compared with
9.71 nanoseconds (9.68–9.75 nanoseconds), a change of -44.9% (-45.2% to -44.6%). Nested immediate
encoding takes 1.51 nanoseconds (1.51–1.53 nanoseconds), compared with 1.52 nanoseconds (1.51–1.53
nanoseconds). That encoding difference is within noise. The private matrix includes all independent
address/length width pairs and immediate and prepared encoding, with unaligned inputs and repeated
records. The mixed-endian nested fixture tests field representation. HDF5 metadata fields remain
little-endian, and runtime datatype decoding remains separate.

## Allocations and layout

Every measurement producer checks allocations against an allocating positive control. Descriptor
operations and symbol parsing allocate zero. Rank-zero Dataspace parsing allocates zero, current
dimensions allocate one vector, and maximum dimensions allocate another when present. Transient
File Space Info parsing allocates zero, while serialization allocates one output vector. The
persisted File Space Info checks match pre-PR: three allocations totaling 448 bytes for the
manager vector.

Both private representations allocate zero for parsing and immediate or prepared writing. Repeated
four-record decoding allocates one output vector, checked against a native-size allocation floor.

| Semantic value on x86_64 | Bytes |
| --- | ---: |
| Public descriptor | 24 |
| Symbol table | 16 |
| File Space Info | 64 |
| Dataspace | 56 |
| Typed descriptor fields | 16 |
| Typed symbol fields | 16 |
| Prepared File Space Info prefix | 32 |
| Nested fixture | 24 |

Wire records have alignment one. Descriptor extents are 10, 12, or 16 bytes, symbol-table extents
are 4, 8, or 16 bytes, and File Space Info prefixes span 10 through 28 bytes. The nested fixture
occupies 16 bytes on disk.

## Generated code and build costs

The bounded descriptor has one complete extent or capacity check followed by direct field loads
or stores. A prepared value retains its address-width bound through encoding. Inlining removes
call and return-value work from immediate descriptor encoding. The File Space Info writer initializes its
96-byte manager array inside the persistence branch. Dataspace dimension loops have neither
per-element byte acquisition nor width dispatch. Rank zero exits before acquisition or allocation.
Immediate encoding writes directly into the destination.

The following text sizes come from the final public benchmark binaries. Family totals count each
symbol address once, including shared helpers present in those binaries.

| Text, bytes | Pre-PR | Reviewed PR | Bounded | `zerocopy` |
| --- | ---: | ---: | ---: | ---: |
| Descriptor parser | 381 | 243 | 246 | 288 |
| Symbol-table parser | 263 | 233 | 174 | 198 |
| File Space Info parser | 1,871 | 1,651 | 5,075 | 4,003 |
| File Space Info serializer | 1,162 | 1,097 | 8,185 | 8,185 |
| File Space Info family | 4,018 | 3,910 | 17,833 | 16,031 |
| Dataspace parser | 1,248 | 2,385 | 3,725 | 3,725 |
| Dataspace family | 4,170 | 6,106 | 7,850 | 7,850 |

Specialization increases the File Space Info and Dataspace code sizes. The `zerocopy` parser
reduces some of that growth, while the descriptor parser is larger than the bounded parser.

The format-only release build used `--no-default-features`, disabled `RUSTC_WRAPPER`, and ran
sequentially without other builds or tests. Each entry below is one observation. Empty target
directories require building dependencies. Cached-dependency observations rebuild the format crate.

| Build, seconds | Bounded | `zerocopy` |
| --- | ---: | ---: |
| Empty target directory | 3.44 | 6.54 |
| Dependencies cached | 3.07 | 3.19 |

The experimental normal dependency graph adds `zerocopy-derive`, `proc-macro2`, `quote`, `syn`,
and `unicode-ident`. Their presence elsewhere in the workspace lockfile does not remove this cost
from an isolated format build.

## Representation assessment

The experimental wire records use ordinary `repr(C)` structs, alignment-one endian integer
fields, and the `FromBytes`, `IntoBytes`, `KnownLayout`, `Immutable`, and `Unaligned` derives.
Borrowed reads use `ref_from_bytes` and `ref_from_prefix`, and direct writes use their mutable
counterparts. Repeated records use a slice view after checked extent arithmetic. Semantic tags and
normalized booleans are decoded through the shared field policies.

The [affected-version tracker][tracker] lists stable 0.8 ranges through 0.8.56. All accepted
measurements use 0.8.63, including both baselines. The [helper-name report][helpers] concerns
captured field types in generated scopes. These schemas use qualified associated field projections
and define no application type with a colliding helper name. The [enum-discriminant report][enums]
concerns `TryFromBytes` enum validation with context-dependent discriminants. These schemas use
all-bit-valid structs and decode semantic enums separately. This is an assessment of the selected
schemas and APIs. Successful tests and the absence of the reports' prerequisites do not establish
general soundness for the dependency. The reports were open when checked on 2026-11-14.

Both production candidates compile at Rust 1.89 without default features. The format crate uses
`alloc`, and the experimental wire views enable neither `std` nor `alloc` in `zerocopy`. The
existing `heapscope` development tool requires Rust 1.96, so the benchmark example uses the current
toolchain.

## Reproduction and retained evidence

Run the following from the repository root with a fresh output directory. Complete every build
before measuring, and keep other builds and tests stopped during timing.

```sh
python3 scripts/experiments/fixed_records.py prepare /tmp/fixed-record-comparison
python3 scripts/experiments/fixed_records.py build /tmp/fixed-record-comparison
python3 scripts/experiments/fixed_records.py measure /tmp/fixed-record-comparison --cpu 0
python3 scripts/experiments/fixed_records.py compare /tmp/fixed-record-comparison
```

The script retains source patches, `Cargo.lock` files, harnesses, dependency trees, compiler and
hardware settings, binary hashes, Criterion samples, absolute intervals, and comparison tables. The
accepted run artifacts are retained locally under `/tmp/hdf5-final-records/`, with final source
identities in `source-identities.json`. The private measurement reuse report records identical
executable instruction bytes and normalized disassembly after unrelated public-path changes.

The partial `invalid-overlapping-final-start` run overlapped a rebuild and is excluded. Valid
`before-inline` and `after-inline` sweeps remain as superseded candidate evidence. The accepted
public sweeps include both the descriptor inlining and conditional manager preparation.

[tracker]: https://github.com/google/zerocopy/issues/3642
[helpers]: https://github.com/google/zerocopy/issues/3633
[enums]: https://github.com/google/zerocopy/issues/3619
