# HDF5 1.8 read fixtures

`v1_superblock.h5` and `v2_superblock.h5` were written by HDF5 1.8.23 and imported
in `e4d9e08a96c7f8f8a1a84a27a676178802bf3051`. Retain these frozen bytes as
reference-release evidence, including reader coverage without a native library.

## Producer and layout

`v1_superblock.h5` was created with `H5Pset_istore_k(64)` and
`H5Pset_sym_k(8, 16)`. Its leaf, internal, and chunk B-tree K values are 16, 8,
and 64. Parser tests check each field separately. The superblock has clean
status flags and eight-byte address and length widths.

`v2_superblock.h5` was created with `H5Pset_libver_bounds(LATEST, LATEST)`.
Its version 2 superblock is paired with a version 1 B-tree chunk index, a
combination this crate's writer does not produce.

Both files contain:

- `/values`, contiguous `f64[4] = [1.5, 2.5, 3.5, 4.5]`, with `units = "m/s"`
- `/chunked`, `i32[1000]` with values `i % 97`, 100-element chunks, and deflate level 6
- `/grp/inner`, with the same values as `/values`
- `/grp` attribute `tag = "group"` and root attribute `root_attr = "r"`

## Consumers

| File | Consumer | Retained observation |
|---|---|---|
| Both | `tests/main/c_1_8_read_compat.rs` | Complete values and attributes under both superblock versions, including non-native target execution |
| Both | `src/edit/tests/publication.rs` | Physical root publication with mirrored backing |
| `v1_superblock.h5` | `crates/hdf5-pure-format/src/superblock.rs` | Parsed K values, flags, and widths |
| `v1_superblock.h5` | `tests/main/public_api_surface.rs` | Public superblock observations |
| `v2_superblock.h5` | `src/edit/tests/publication.rs` | Bounded backing and derived user-block/base-address cases |

## Retention and licensing

The historical generator is recorded in the import commit at
`tests/fixtures/c_1_8/regen.c` and `tests/fixtures/c_1_8/regen.sh`. There is no
current regeneration recipe. Preserve the original HDF5 1.8.23 producer output.

The repository is distributed under MIT OR Apache-2.0. No separate
fixture-specific license statement is recorded. Nothing from the HDF5
software distribution is vendored here.
