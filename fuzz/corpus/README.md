The `parse-file` corpus supplies HDF5 inputs for parser mutations. The
[fuzz recipes][recipes] also pass it to `streaming-differential`.

These are frozen mutation starting points, retained byte for byte. Their original
producers and producer versions are unknown. No regeneration procedure or
fixture-specific license is recorded.

The five `tests/fixtures/` origins below refer to the repository's initial commit,
`d77cd9be086a5c8be812ccd0e55b835ce0ae78cd`. Those files were copied unchanged into the
corpus in `da22f10f5e5fa650054f711803adf056f353d1b4`. Each corpus copy retains those
bytes. The two SOHM inputs have separate import history.

| Seed under `parse-file/` | Known origin | Retained mutation coverage |
|---|---|---|
| `attrs.h5` | `tests/fixtures/attrs.h5` | Version 0 superblock, scalar root and dataset attributes |
| `chunked_nofilter.h5` | `tests/fixtures/chunked_nofilter.h5` | Unfiltered chunks in `raw`, shape `[50]`, chunk shape `[10]` |
| `dense_attrs.h5` | `tests/fixtures/dense_attrs.h5` | Fifty dataset attributes in dense heap and name-index storage |
| `minimal_v2.h5` | `tests/fixtures/minimal_v2.h5` | Version 0 superblock and variable-length UTF-8 root attribute `test` |
| `simple_dataset.h5` | `tests/fixtures/simple_dataset.h5` | Contiguous `data`, shape `[3]`, little-endian `f64` values |
| `sohm_list.h5` | Import `2a9f356afd7b77f931ec10625f3dfd1a5983b66c`, source and producer unknown | Shared object-header messages with a list index |
| `sohm_btree.h5` | Import `2a9f356afd7b77f931ec10625f3dfd1a5983b66c`, source and producer unknown | Shared object-header messages with a B-tree index |

[recipes]: ../../scripts/fuzz.just
