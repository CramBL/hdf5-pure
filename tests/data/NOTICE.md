# Test data

Committed reference files and malformed regressions are organized by origin. Each
directory notice records producers, consumers, invariants, licensing evidence,
and regeneration instructions or retention rationale. `just test-data` lists
producer-specific commands.

- [Historical C files][historical]: frozen HDF5 1.8.23 compatibility evidence
- [LZF references][lzf] and [ZFP references][zfp]: h5py-produced vectors and pipeline files
- [MATLAB files][matlab]: native output and deliberately corrupted derivatives
- [Historical pure file][pure]: a frozen hdf5-pure 0.28.0 collision regression
- [Malformed fuzz inputs][fuzz]: ordinary regression inputs

A test reads a file from here and never writes one. The separate [mutation corpus][corpus]
records its own frozen seed origins and retention coverage.

## Dense attribute byte observations

`crates/hdf5-pure-format/src/dense_attribute_write/tests.rs` retains text observations for
`prepared_dense_output_matches_baseline_bytes`. The producer was hdf5-pure at revision
`534712d2394f8170f9d7655262c52611dce76665`, under the repository's MIT or Apache-2.0 license.
The observations are the lengths and FNV-1a hashes of complete dense-attribute blobs and
Attribute Info messages at heap address `0x8000`.

Regenerate them by constructing the test's managed, tracked, and mixed attribute sets at
that revision, calling `dense_attrs_plan` and `build`, and hashing both emitted buffers with
the test's `canonical_hash` function. Retain the recorded values to detect changes in
canonical heap placement, index ordering, and serialized bytes without committing binary
fixtures. Ordinary tests compare the observations and never regenerate them.

[historical]: c/1.8/NOTICE.md
[lzf]: h5py/lzf/NOTICE.md
[zfp]: h5py/zfp/NOTICE.md
[matlab]: matlab/NOTICE.md
[pure]: pure/NOTICE.md
[fuzz]: fuzz/NOTICE.md
[corpus]: ../../fuzz/corpus/README.md
