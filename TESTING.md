# Testing

## Prefer integration tests

Put new behavior and regression tests in the owning crate's `tests/` suite.
Use the interface its callers use.

- Exercise file operations through `File`, dataset, and group APIs. Use the
  owning crate's interface for format and allocation behavior.
- Use the smallest input and operation sequence that demonstrates the behavior.
- Check complete values, the stored bytes that establish the format claim, and
  the specific error expected. Close and reopen the file to check persistence.

## Organize integration tests by area

Keep ordinary integration tests in one target per crate. Use
`tests/main/main.rs` for module declarations, shared gates, and suite setup.
Group cases by behavior or format area. Give distinct behaviors their own
submodules in large areas.

| Facade test file | Responsibility |
| --- | --- |
| `tests/main/main.rs` | Declare area modules, such as `mod append;` |
| `tests/main/append.rs` | Append behavior |
| `tests/main/edit_free_space.rs` | Free-space editing |
| `tests/main/edit_reference_counts.rs` | Reference-count editing |

Area modules share the `main` target. A top-level `tests/<area>.rs` creates a
separate binary. Share the target to reduce repeated linking.

## Unit tests

Use unit tests for focused algorithms or invariants that are awkward to cover
through the crate's interface. Choose the boundary that makes the case clear
and keeps it useful when private implementation details change.

## Keep fixtures independent

Use `test-util` to construct raw file structures, including malformed inputs and
layouts the writer cannot produce. Use `test-util-hdf5` for facade fixtures and
native-library comparisons. Check format conformance against independent bytes
or a reference implementation. A roundtrip can share a reader and writer defect.

Describe valid fixtures with checked types. Follow parse, don't validate.
Represent deliberate corruption as raw bytes or explicit mutations.

## Keep test support out of production algorithms

Exercise the real operation. Inject failures at the I/O boundary so production
code handles read, write, resize, and synchronization errors. Keep test-only
fields, reports, counters, and partial-execution modes out of the algorithms.
Keep checked ownership evidence and counters that enforce resource limits.

Keep small helpers beside their callers. Share setup with the same purpose.
Call existing production operations directly. Use an ordinary function or
a test-owned type when it keeps setup or assertions clear.

## Extended tests

Use the extended suite for long operation sequences, large inputs, and broad
combinations, including free-space churn and endurance.

- Prefix each function with `extended_` and keep it in its existing area module.
- Keep short regression cases in the ordinary suite.
- Retain functional assertions, resource bounds, and per-test timeouts.

Run both suites in CI through separate nextest profiles. Use
`just test-extended` or `just test-extended-full` for the workspace and
`just interop::test-extended-hdf5 <version>` or
`just interop::test-extended-bundled` for native cases. Plain `cargo test`
also runs extended cases because it does not read nextest profiles.

## Changes to coverage and resource budgets

Preserve the behavior a test establishes when replacing it. Retire assertions
about private structures outside the contract. Explain any loss of coverage.

Compare the same workload, backing, cache state, and measurement interval when
changing a budget. Distinguish source requests from system I/O and allocation count
from cumulative, peak, and retained bytes. Explain any increase and bound the new
cost. A snapshot records the result of that decision.

## Where tests and helpers belong

Keep tests with the code or API they exercise.

| Test or helper | Home |
|---|---|
| Public API tests | Area modules in the owning crate's `tests/main/` suite |
| Private unit tests | Owner's `#[cfg(test)] mod tests`, inline when small, otherwise `tests.rs` or `tests/mod.rs` with domain children |
| Private native crosschecks | Owner-local `native_crosschecks`, gated through the helper capability macro |
| Independent bytes, malformed inputs, paths, allocation measurement | `crates/test-util` |
| Facade fixtures and reference-library helpers | `crates/test-util-hdf5`, with optional native dependencies |
| Private-access support | Owner-local test-only support, following the [unit test guidance](#unit-tests) |

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

Cover public API additions and changes in public behavior through that API in the owning crate's
`tests/`. Reuse an existing test if it establishes the behavior.

Use `snake_case` names and named `rstest` cases that identify the distinguishing input. Cover meaningful
boundaries and failure cases. A registered test must be independently executable: never call it as a
helper or use it for child dispatch. Reuse ordinary functions and explicit fixture binaries instead.

Unwrap successes and check complete promised values and state. Assert discriminating error variants
and fields using [the exact-error rules][errors]. An unrelated failure must not satisfy a test.

For a case that depends on a file layout, check the structure that distinguishes it.
Reuse `ReadBackend` and typed dataset or filter cases where their contracts fit.

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
