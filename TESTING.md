# Testing

## Choosing the test boundary

Test observable behavior through the production entry point that owns the contract. Exercise
user-visible reader, writer, and editor behavior through the public file API. Test parser and
allocator contracts through their component interfaces. Use private unit tests for local invariants,
checked state transitions, and cases that need closer isolation.

For a file operation, check the values, stored structure, errors, recovery, and resource bounds
relevant to the change. Close and reopen the file to check persistence. Use an independent reader
to check interoperability. Cover the complete commit, append, or copy when the guarantee depends
on the complete operation. Use unit tests to isolate the parts that need their own assertions.
Reuse existing coverage when it establishes the same behavior.

## Production execution and test support

Test builds exercise the same parsing, planning, mutation, publication, and recovery code as
ordinary builds. Keep fields and counters used solely for test reports out of these algorithms.
Keep the state that enforces resource limits and the checked evidence that establishes ownership,
source identity, or permission to modify storage. Exercise the complete production operation
without test-only phase selectors or simulated success paths.

Observe effects at existing boundaries. Inject read, write, resize, or synchronization failures
at the I/O boundary so the operation runs its own error handling. Reuse an adapter that can record
or fail the chosen operation, or add one scoped to that boundary. Let the production code
construct its own plans and decide how to recover.

Private unit tests can inspect invariants using existing state. Keep assertion helpers with their
tests. Ordinary functions, test-owned fixture types, and test-gated methods can express those
assertions without adding production fields or widening visibility in ordinary builds.

Keep small helpers with their callers. Share helpers with the same contract, and extract a
module when it gives related helpers a clear home. When a test exercises a production operation,
call that operation directly so the test covers its actual sequence. Moving a gated helper
changes where readers find it. Evaluate the test's independence by how it obtains its evidence.

## Where tests and helpers belong

Keep tests with the code or API they exercise.

| Test or helper | Home |
|---|---|
| Private unit tests | Owner's `#[cfg(test)] mod tests`, inline when small, otherwise `tests.rs` or `tests/mod.rs` with domain children |
| Public API tests | Owning package's `tests/`, bundling related suites with facade/crosscheck entry points at `tests/main/main.rs` |
| Private native crosschecks | Owner-local `native_crosschecks`, gated through the helper capability macro |
| Independent bytes, malformed inputs, paths, allocation measurement | `crates/test-util` |
| Facade fixtures and reference-library helpers | `crates/test-util-hdf5`, with optional native dependencies |
| Private-access support | Owner-local test-only support, following [Production execution and test support][test-support] |

Keep `test-util` independent of implementation crates. [ARCHITECTURE.md][architecture] explains the
dependency boundary.

Keep separate targets for process-wide allocators, WASM harnesses, explicit fixture binaries, fuzzing,
benchmarks, and [API-profile checks][profiles]. `format_reexports` stays separate because the semver
helper selects it explicitly, including without `std`.

Preserve selectors, gates, imports, formatting paths, snapshots, seeds, and documentation when moving
tests. Preserve the external public API boundary and diagnostic-specific seal guards. When adding or
moving native crosschecks, ensure the [interop default filter][nextest] and applicable CI selectors
include them. Verify that the intended tests execute.

## Writing a test

Name the behavior the test proves.

Cover each public API addition and each change to a public operation's behavior through that API in
the owning package's `tests/`. An existing test satisfies this requirement if it establishes the
behavior.

Use `snake_case` names and named `rstest` cases that identify the distinguishing input. Cover meaningful
boundaries and failure cases. A registered test must be independently executable: never call it as a
helper or use it for child dispatch. Reuse ordinary functions and explicit fixture binaries instead.

Unwrap successes and check complete promised values and state. Assert discriminating error variants
and fields using [the exact-error rules][errors]. An unrelated failure must not satisfy a test.

Choose independent evidence for the behavior under test. Use independently constructed bytes or
observations from a reference library to establish format conformance. A reader and writer can
share a defect that a roundtrip misses. Use production builders for unrelated setup, and derive
the expected result independently of the operation under test. When a case depends on a particular
file layout, assert the structure that distinguishes it as well as the decoded values.

Use the low-level byte builders and mutation helpers in `test-util` for malformed inputs and
layouts the public writer cannot create. Keep fixtures built through the facade and reference
library helpers in `test-util-hdf5`. Follow the dependency directions in
[ARCHITECTURE.md][architecture].

Use parse, don't validate in Rust helpers: convert inputs once into existing domain types or small
enums/newtypes that cannot express unsupported combinations. Describe valid fixtures with checked
types and create malformed inputs from raw bytes or explicit corruption of valid bytes. Reuse
`ReadBackend` and typed dataset/filter cases where their contracts fit.

## Fixtures and temporary files

Prefer fresh construction when historical or external evidence is unnecessary.

Use independent byte builders for layout inputs and native generators for reference files. Retain
committed evidence with its consumer, invariant, producer/version, license where applicable, and
regeneration or retention rationale in the [fixture notices][fixtures]. Producer references are
regenerated through their producer. For minimized or derived regressions, document known origin,
transformation, invariant, and unknown source details. Ordinary tests never overwrite committed
fixtures.

Use `test_util::data`/`copy_data` for runtime paths. Keep intentional `include_bytes!`. Own single-file
temporary paths or a `TempDir` for several files. Release handles before destructive cleanup or
completed-file handoffs, allowing intentionally concurrent SWMR/locking handles.

## Additional coverage

Select the obligations that establish the changed behavior.

| Trigger | Essential obligation | Owner or configuration |
|---|---|---|
| Native interoperability | Fresh independent native observations of promised values and metadata. Selected tests fail on missing prerequisites or protocol errors. Preserve operation/helper synchronization. Helpers must not reacquire an explicit whole-test guard. | [Interop recipes][interop], [locking helper][locking] |
| Unsafe code or platform assumptions | Execute selected Miri paths and boundaries. Check affected feature/target combinations and preserve moved-test selection. | [Soundness][soundness], [portability][portability] |
| Allocation or performance | Assert functional results, bounds, and a positive control/floor. Follow region attribution and reject invalid measurements. Keep exact baselines, diagnostic profiles, and throughput benchmarks distinct. | [Allocation helper][allocation], [soundness recipes][soundness] |
| Generated invariants | Use bounded, shrinkable properties with explicit persistence paths. Preserve seeds and retain minimized exact regressions. | [Property suite][properties] |
| Differential fuzzing | Compare both success/failure outcomes at equivalent operations, with explicit permitted differences and bounded traversal/resources. | [Differential target][differential] |
| Snapshots | Normalize irrelevant variation, inspect changed observations, and distinguish bundled-version exact output from cross-version semantics. | [Native snapshot suite][snapshots] |
| Child processes | Use explicit fixture binaries. Bound execution/readiness and clean up children. Infrastructure failures fail the parent. Preserve runner support. | [Fixture target][child], [target declaration][targets] |
| Doctests or API contracts | Prefer self-contained executable examples. Allow `no_run` for unavailable external inputs/services or an inappropriate execution environment, with a reason beside the example. Keep external API and diagnostic checks. | [Documentation style][docstyle], [API checks][api] |

### Resource expectations

State the workload, the operation measured, and where the measurement is taken. Distinguish requests
to a source from operating-system I/O. For allocations, specify whether the assertion concerns
allocation count, cumulative bytes allocated, peak bytes, or retained bytes. Use a count or cumulative
byte bound when a regression can repeatedly allocate and free memory without changing the peak.

Compare equivalent work when changing a baseline or budget. Account for fixture size, backing,
cache state, and the measurement interval. Include publication and close when they are part of the
guarantee. Record the old and new observations and explain what changed, including any change to the
workload or measurement. If the same workload intentionally uses more resources, explain the tradeoff
and bound the new cost.

### Replacing coverage

Preserve correctness, failure, format, and resource guarantees when replacing tests.
Retire assertions about obsolete reports or exact internal operation counts when they establish no
project guarantee. Identify any consequential coverage reduction in the review. Judge a replacement
by the behavior it checks and the evidence it uses.

## Running tests

Run recipes from the repository root.

`just test` runs ordinary workspace tests with defaults. `just test-full` adds the facade's full public
feature profile. Both exclude crosschecks. Run doctests separately with `just doctest-full`.

Use `just --list <module>` for specialized recipes linked above. [CONTRIBUTING.md][contributing]
describes the PR workflow and broader gates. Focused selections must execute the intended tests.

## Enforcement

Use the existing standard gates.

[Workspace Clippy settings][lints], [Clippy configuration][clippy], and [nextest configuration][nextest]
provide lint and execution checks. Formatting, docs, and prose have separate gates. Review the
test boundary, helper ownership, independent evidence, and retained coverage.

[architecture]: ARCHITECTURE.md#test-helpers
[test-support]: #production-execution-and-test-support
[profiles]: scripts/api-profiles.toml
[errors]: CODE_STYLE.md#assert-the-exact-error
[fixtures]: tests/data/NOTICE.md
[interop]: scripts/interop.just
[locking]: crates/test-util-hdf5/src/lock.rs
[soundness]: scripts/soundness.just
[portability]: scripts/portability.just
[allocation]: crates/test-util/src/allocation.rs
[properties]: tests/main/roundtrip_properties.rs
[differential]: fuzz/fuzz_targets/streaming_differential.rs
[snapshots]: crates/crosscheck/tests/main/file_space_oracle.rs
[child]: crates/crosscheck/tests/fixtures/interrupted_writer.rs
[targets]: crates/crosscheck/Cargo.toml
[docstyle]: DOC_STYLE.md
[api]: scripts/api.just
[contributing]: CONTRIBUTING.md#submitting-pull-requests
[lints]: Cargo.toml
[clippy]: clippy.toml
[nextest]: .config/nextest.toml
