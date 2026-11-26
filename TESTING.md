# Testing

## Test design

Prefer integration tests through the owning crate's public API. Use unit tests for
focused coverage of internal algorithms and invariants.

Cover new or changed public behavior with tests.

Keep each test focused on one behavior or invariant. Use all the assertions
needed to establish it. Split tests that cover unrelated scenarios.

Use named `rstest` cases when inputs vary and the behavior and assertions stay
the same. Write separate tests when combining cases requires branches or
elaborate setup.

Assert specific error variants and fields using the [exact-error rules][errors].

### Test examples

- The [complex datatype tests][complex-tests] focus on one invariant at a time
  and use named `rstest` cases for component sizes and byte orders.
- The [object header continuation tests][continuation-tests] build malformed
  bytes at runtime and check the error through two read paths.
- The [dense attribute tests][dense-attribute-tests] check files written by
  libhdf5 and use libhdf5 to read independently constructed fixtures.

## Test organization

| Test | Location |
| --- | --- |
| Ordinary integration tests | One target per crate. Declare modules in `tests/main/main.rs`, for example `tests/main/append.rs` and `tests/main/edit_reference_counts.rs`. |
| Private unit tests | The owning module's `#[cfg(test)] mod tests`, inline or in a separate file. A `tests/mod.rs` follows the [module-index rule][modules]. |
| Native crosschecks | `crates/crosscheck/tests/main/` for external tests, or an owner-local `native_crosschecks` module with the existing [capability gate][architecture]. |

Keep separate targets for process-wide allocators, fixture binaries, WASM,
fuzzing, benchmarks, and [API-profile checks][profiles]. The API check selects
`format_reexports` by target name, including without `std`. Keep the
[`#[non_exhaustive]` checks][seals] outside the defining crate.

Keep test helper types that need private access in the owning test module.
Follow the [helper crate dependency rules][architecture].

### Test support in the library implementation

Keep test-only fields, reports, counters, and partial-execution modes out of
library algorithms. Keep allocation-ownership checks and resource accounting
that the operation requires.

When testing I/O failures, inject faults at the I/O boundary so the library
handles them through its usual path.

## Fixtures and format conformance

Generate ordinary regression fixtures at test runtime in Rust.

- Use `libhdf5` (the `hdf5-metno` package) to generate reference files. Put shared
  libhdf5 helpers in `test-util-hdf5`.
- Use `test-util` to construct raw HDF5 structures, unusual layouts, and
  malformed inputs.
- Use hdf5-pure to generate files when testing its writing behavior.

Describe valid fixtures with checked types. Follow parse, don't validate. Use
raw bytes or explicit mutations for deliberate corruption.

When testing on-disk format conformance, compare against independently constructed
bytes or observations from libhdf5. A roundtrip through hdf5-pure alone may miss
shared reader/writer defects.

Do not add C generators, Python or h5py scripts, or committed HDF5 binaries for
ordinary regressions. Retain historical or externally produced fixtures when
their original bytes are needed to establish compatibility or reproduce a defect.
Record where each file came from and what it tests in the [fixture notices][fixtures]. Use
`test_util::data` or `copy_data` for runtime paths to committed fixtures.

## Specialized testing

Apply only the requirements for the testing methods your change needs.

| Category | Guidance | Reference |
| --- | --- | --- |
| Native interoperability | Compare against independent observations from libhdf5 or libmatio. Selected cases fail on missing prerequisites or protocol errors. Preserve the native locking contract. Verify nextest and CI selection when adding or moving cases. | [Interop recipes][interop], [native locking][locking], [interop profiles][nextest] |
| Unsafe code and portability | Exercise affected cases under Miri and in the target and feature configurations they depend on. | [Miri][soundness], [portability][portability] |
| Allocation testing | Assert resource bounds and verify allocation measurements. Justify and bound resource increases using comparable workloads and measurement intervals. | [Allocation measurements][allocation], [baseline recipes][soundness] |
| Property testing | Use bounded, shrinkable cases with explicit persistence paths. Preserve seeds and minimized regressions. | [Property tests][properties] |
| Differential fuzzing | Compare equivalent operations, including errors. Document permitted differences and bound traversal and resources. | [Differential target][differential] |
| Snapshots | Normalize irrelevant variation and inspect changes. Keep resource-bound assertions when updating snapshots. | [Free-space snapshots][snapshots] |
| Child processes | Bound fixture startup and execution. Clean up children and fail the parent on infrastructure errors. | [Fixture binary][child], [target declaration][targets], [interrupted writer test][child-example] |
| Doctests | Prefer executable examples. Justify `no_run` beside the example. Preserve public API and diagnostic checks. | [Documentation style][docstyle], [API checks][api] |

## Extended tests

Use the extended suite for tests that would significantly increase ordinary suite
runtime, such as endurance tests. Prefix their names with `extended_` and keep
them alongside related tests.

CI runs these cases separately using the [nextest profiles][nextest] `extended`
and `extended-interop`.

## Running tests

Run from the repository root. Verify that focused selections execute the intended cases.

| Suite | Recipes |
| --- | --- |
| Ordinary workspace | `just test`, `just test-full` |
| Extended workspace | `just test-extended`, `just test-extended-full` |
| Doctests | `just doctest`, `just doctest-full` |
| Native interoperability | `just interop::test-bundled`, `just interop::test-hdf5 <version>` |
| Extended native interoperability | `just interop::test-extended-bundled`, `just interop::test-extended-hdf5 <version>` |

The `-full` recipes enable the full public feature profile. Use
`just --list <module>` for recipes in the specialized suites.

[architecture]: ARCHITECTURE.md#test-helpers
[profiles]: scripts/api-profiles.toml
[seals]: tests/main/seals.rs
[fixtures]: tests/data/NOTICE.md
[errors]: CODE_STYLE.md#assert-the-exact-error
[modules]: CODE_STYLE.md#modules
[complex-tests]: crates/hdf5-pure-core/tests/main/complex.rs
[continuation-tests]: tests/main/object_header_continuation.rs
[dense-attribute-tests]: crates/crosscheck/tests/main/dense_attributes.rs
[nextest]: .config/nextest.toml
[interop]: scripts/interop.just
[locking]: crates/test-util-hdf5/src/lock.rs
[soundness]: scripts/soundness.just
[portability]: scripts/portability.just
[allocation]: crates/test-util/src/allocation.rs
[properties]: crates/hdf5-pure-filter/src/scaleoffset/tests/properties.rs
[differential]: fuzz/streaming_differential.rs
[snapshots]: crates/crosscheck/tests/main/file_space_oracle.rs
[child]: crates/crosscheck/tests/fixtures/interrupted_writer.rs
[child-example]: crates/crosscheck/tests/main/owned_swmr.rs
[targets]: crates/crosscheck/Cargo.toml
[docstyle]: DOC_STYLE.md
[api]: scripts/api.just
