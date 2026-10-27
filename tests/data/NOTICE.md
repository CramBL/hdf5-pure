# Test data

Committed files the tests read, one directory per writer. Each directory says
what wrote its files and how to write them again, and `just test-data` runs the
writers this repository can run.

- `c/`: the reference HDF5 C library.
- `h5py/`: h5py, with the LZF and zfp filters and MATLAB's v7.3 layout.
- `matlab/`: MATLAB itself.
- `pure/`: this crate's own writer, through seams its public API has no
  spelling for. Each file is checked by its content, never by its bytes.
- `fuzz/`: inputs the fuzzer found.
- `unattributed/`: files from the crate's first commit whose writer was not
  recorded. A file moves out when its writer is established.

A test reads a file from here and never writes one. The writers are the
generators named in each directory.

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
