# Written by this crate

Files this crate's writer produced through seams its public API has no
spelling for, so that the crosscheck package can read them with the reference
C library. Each is checked by its content: the test that writes it decodes the
committed copy and requires it to still hold what it writes, and
`just test-data::pure` rewrites it when that changes.

- `exotic_attributes.h5`: attribute encodings the C library's own API cannot
  express, a fixed-width string with null termination and a Null dataspace,
  written by `src/repack.rs` and repacked by `crates/crosscheck/tests/repack.rs`.
- `dense_attrs_colliding_v0_28_0.h5`: written by hdf5-pure 0.28.0 before the
  fix for colliding attribute names in a multi-level dense index (#225), read
  by `crates/crosscheck/tests/dense_attr_limits.rs`. Not regenerated.
