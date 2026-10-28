# Test data

Committed files the tests read, one directory per writer. Each directory records
the producer and regeneration instructions or retention rationale. `just test-data`
lists producer-specific commands.

- `c/`: frozen HDF5 1.8.23 compatibility files.
- `h5py/`: h5py, with the LZF and zfp filters.
- `matlab/`: MATLAB itself.
- `pure/`: a frozen hdf5-pure 0.28.0 collision regression.
- `fuzz/`: inputs the fuzzer found.

A test reads a file from here and never writes one.

## LZF and ZFP reference data

The [LZF generator](h5py/lzf/regen.py) uses Python 3.14, h5py 3.16.0, and NumPy 2.4.4.
The [ZFP generator](h5py/zfp/regen.py) uses Python 3.13, h5py 3.16.0, hdf5plugin 6.0.0,
and NumPy 2.4.4. The ZFP manifest records H5Z-ZFP 1.1.1 and ZFP 1.0.1. Both
generators define their numeric inputs and pin their producer dependencies.

The manifests record scalar types, dimensions, and filter parameters. Retain the raw
and compressed vectors as independent codec evidence for `src/lzf_crosscheck.rs` and
`src/zfp_crosscheck.rs`, including the integer ZFP cases. Regenerate on a little-endian
host with `just test-data::lzf` or `just test-data::zfp`. Extraction containers are
temporary, and ZFP filter parameters are retained in the manifest.

The reader tests use these pipeline containers:

| Container under `h5py/` | Invariant | Consumer |
|---|---|---|
| `lzf/u8_noise.h5` | Optional LZF is skipped for an incompressible chunk | `tests/main/lzf_roundtrip.rs` |
| `lzf/f64_shuffle_lzf.h5` | Shuffle followed by LZF | `tests/main/lzf_roundtrip.rs` |
| `lzf/i64_multichunk.h5` | Multiple LZF chunks with a partial edge | `tests/main/lzf_roundtrip.rs` |
| `zfp/f32_1d_2_partial_rate32_zfp_deflate.h5` | A partial ZFP block followed by deflate | `tests/main/zfp_roundtrip.rs` |

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
