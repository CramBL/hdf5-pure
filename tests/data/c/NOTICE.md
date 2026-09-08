# Written by the reference C library

`1.8/` holds files HDF5 1.8.23 wrote, described there.

The rest are written by `crates/crosscheck/tests/c_test_data.rs` for unit tests
that need what only the C library writes, and `just test-data::c` rewrites them
with the release `hdf5-metno` bundles:

- `huge_links_12.h5`: a group of 12 links, each name so long that its link
  message is a huge heap object, for the dense-link walk in `src/group_v2.rs`.
- `paged_index.h5`: a paged, persisting file whose chunk index the C allocator
  placed as metadata, for `src/edit.rs`.
- `generic_large.h5`: a paged, persisting file with an attribute far larger
  than its page, so the generic-large manager holds a sub-page fragment, for
  `src/edit.rs`.
- `hard_link_undo.h5`: two hard links to one dataset, which this crate has no
  API to create, for `src/edit.rs`.
- `committed_datatype_v1.h5`: a committed datatype in a version 1 object
  header, linked once and used by two datasets, for `tests/named_datatypes.rs`.
