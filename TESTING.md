# Testing

Choose tests that establish the promised behavior. Overlapping coverage categories do not require
duplicate tests.

## Where tests and helpers belong

Keep tests with the code or API they exercise.

| Test or helper | Home |
|---|---|
| Private unit tests | Owner's `#[cfg(test)] mod tests`, inline when small, otherwise `tests.rs` or `tests/mod.rs` with domain children |
| Public API tests | Owning package's `tests/`, bundling related suites with facade/crosscheck entry points at `tests/main/main.rs` |
| Private native crosschecks | Owner-local `native_crosschecks`, gated through the helper capability macro |
| Independent bytes, malformed inputs, paths, allocation measurement | `crates/test-util` |
| Facade fixtures and reference-library helpers | `crates/test-util-hdf5`, with optional native dependencies |
| Private-access support | Owner-local test-only support |

Keep `test-util` independent of implementation crates. [ARCHITECTURE.md][architecture] explains the
dependency boundary. Share setup with the same contract. Keep small domain helpers local.

Keep separate targets for process-wide allocators, WASM harnesses, explicit fixture binaries, fuzzing,
benchmarks, and [API-profile checks][profiles]. `format_reexports` stays separate because the semver
helper selects it explicitly, including without `std`.

Preserve selectors, gates, imports, formatting paths, snapshots, seeds, and documentation when moving
tests. Preserve the external public API boundary and diagnostic-specific seal guards. When adding or
moving native crosschecks, ensure the [interop default filter][nextest] and applicable CI selectors
include them. Verify that the intended tests execute.

## Writing a test

Name the behavior the test proves.

An addition to the public API has at least one test through that API in the owning package’s `tests/`.

Use `snake_case` names and named `rstest` cases that identify the distinguishing input. Cover meaningful
boundaries and failure cases. A registered test must be independently executable: never call it as a
helper or use it for child dispatch. Reuse ordinary functions and explicit fixture binaries instead.

Unwrap successes and check complete promised values and state. Assert discriminating error variants
and fields using [the exact-error rules][errors]. An unrelated failure must not satisfy a test.

Keep evidence independent of the implementation under test. A roundtrip can share a reader/writer
defect. Use independent bytes or reference observations to establish conformance. For layout claims,
assert the structural premise as well as decoded values.

Use parse, don't validate in Rust helpers: convert inputs once into existing domain types or small
enums/newtypes that cannot express unsupported combinations. Model deliberately malformed cases
separately from valid fixture descriptions. Reuse `ReadBackend` and typed dataset/filter cases where
their contracts fit.

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
provide lint and execution checks. Formatting, docs, and prose have separate gates. Placement, helper
ownership, independent evidence, and test-entrypoint reuse remain review conventions, not completely
automated checks.

[architecture]: ARCHITECTURE.md#test-helpers
[profiles]: scripts/api-profiles.toml
[errors]: CODE_STYLE.md#assert-the-exact-error
[fixtures]: tests/data/NOTICE.md
[interop]: scripts/interop.just
[locking]: crates/test-util-hdf5/src/lock.rs
[soundness]: scripts/soundness.just
[portability]: scripts/portability.just
[allocation]: crates/test-util/src/allocation.rs
[properties]: tests/main/proptest_roundtrip.rs
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
