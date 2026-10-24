# Ownership fixtures

`ownership/` contains closed-file images from the bundled file-space crosschecks. The files cover
creation by either library, subsequent edits by both libraries, free-space thresholds, PAGE
allocation, userblocks, legacy file-space messages, and address widths. Crate-local tests analyze
these images through the private editor. The root crate's tests need no C library.

## Checkpoint inventory

[`ownership.tsv`](ownership.tsv) maps each checkpoint to its oracle snapshot at baseline `8841317`,
case label, and image path relative to `ownership/`. Its 524 rows cover 92 baseline snapshots. The
private test `ownership_fixture_inventory_covers_each_committed_checkpoint` compares the listed
paths with the committed image set.

| Case family | Checkpoints | Baseline snapshots | Observations |
| --- | ---: | ---: | --- |
| Round trips | 480 | 48 | Creation, Rust deletion and reuse, manager rewrites, C reuse and deletion |
| FSM subthreshold deletion | 8 | 8 | Closed files after Rust and C deletions, with and without persistence |
| PAGE thresholds | 22 | 22 | Closed files after Rust and C raw-storage or metadata deletions |
| Append reuse | 2 | 2 | Closed flat and paged files after Rust append reuse |
| Tail truncation | 2 | 2 | Closed flat and paged files after Rust deletion |
| Version 0 file-space messages | 4 | 4 | Persisted sections, address widths, and a userblock |
| Address widths | 6 | 6 | Persisted sections with Core and Sec2 drivers |

The round-trip matrix pairs six strategy and persistence settings with thresholds of 1 and 128
bytes, userblocks of 0 and 4,096 bytes, and creation by either library. Each case has ten images:
`created`, `rust-deletion`, three `rust-reuse` checkpoints, three `manager-rewrite` checkpoints,
`c-reuse`, and `c-delete`.

## Private ownership assertions

`src/storage_tests/c_origin.rs` contains the fixture tests. The parameterized test
`ownership_of_libhdf5_and_mixed_writer_checkpoints` produces one private snapshot per case directory
under `src/storage_tests/snapshots/`. The case directory is the snapshot name, and each image's stem
is its checkpoint label.

For the 518 images with supported editing widths, each row records analysis completeness, physical
and logical sizes, and byte totals and extent counts for userblock, live raw, live metadata,
free-space-manager, reusable, unclassified, and trailing storage. These values match the ownership
measurements in the baseline snapshots. The tests also assert an exact disjoint file partition,
the userblock boundary, reusable extents equal to `File::space_accounting()`, and live raw extents
enumerated independently through dataset layouts and chunks. Each observation uses a temporary copy
and checks that closing its mirrored editor preserves the complete file bytes.

Six images have two-byte or four-byte addresses. Their snapshots record the precise
`Error::EditUnsupported` reason for in-place editing, and the tests check that the failed open
preserves the file bytes. Their persisted sections remain covered by external read-only tests.
An unavailable editing session is distinct from an incomplete ownership analysis.

The test `c_origin_page_preserves_nonzero_unclassified_storage` identifies the `c-reuse` image of
`page-persist-true-page-4096-threshold-1-base-0-c-origin-true`. Its private snapshot records complete
analysis with 288 unclassified bytes in one extent and 387 manager bytes in six extents. The test
also asserts the nonzero unclassified total and an exact partition. Complete analysis permits
unclassified bytes in these fixtures. It is separate from both incomplete analysis and bytes after
logical EOF.

## External assertions

The running crosschecks continue to compare every persisted section address and length between
`File::persisted_free_space()` and `H5Fget_free_sections`, including individual adjacent boundaries.
They compare section counts and totals, and check the independent `H5Fget_freespace` total, creation
properties, stored EOF, physical size, and complete file bytes before and after read-only observation.
The workloads also retain survivor-content, allocation-reuse, threshold, and interoperability
assertions. An empty persisted map in a file without persistence describes its closed state. Session
reusable space is checked separately during edits.

The fixture inventory excludes endurance, churn, and deliberate trailing-byte cases. Their private
analyzer checks are in `src/storage_tests/endurance.rs`, `churn.rs`, and
`trailing_slack.rs`. Endurance and churn snapshots record categories at editing checkpoints and
actual closed-file reopens, including final close. Threshold-admission properties remain in
`src/storage_tests/threshold.rs`, paired with external libhdf5 checks. Incomplete-analysis fixtures
remain in `src/edit/ownership_tests.rs`.

## Regeneration and scope

`just test-data::c` runs the bundled crosscheck generators with `HDF5_PURE_UPDATE_TEST_DATA` set.
The oracle captures each image after the original read-only observation. It zeros stored version 2
object-header timestamps, updates their checksums, and writes the normalized image to both the
fixture and the workload path. The workload then continues from that image. The generator repeats
the independent oracle observation on the fixture and requires the same section maps, sizes, and
creation properties. Ordinary crosscheck runs leave the fixture corpus unchanged.

Run `just test c_origin` for the private fixture assertions and `just interop::test-hdf5-all` for
the independent crosschecks against the supported C-library versions. Fixture byte expectations
come from the bundled build and preserve the baseline's non-workload ownership observations.
Allocation choices can differ between C-library versions. The interop matrix checks persisted
section-map equality with HDF5 1.10 and later.
