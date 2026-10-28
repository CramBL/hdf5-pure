# Written by libhdf5

`1.8/` holds files HDF5 1.8.23 wrote, described there.

The files below are reference-library outputs retained for ordinary tests. Reusable
construction lives in `crates/test-util-hdf5/src/native_fixtures.rs`, under the helper's
optional `__hdf5-1.10` capability.

The `c_test_data` suite checks current constructor output on fresh paths and retained
copies against the same contents and storage expectations. It follows each file's
encoded addresses.

`just test-data::c` explicitly rewrites these copies with the bundled `hdf5-metno`
dependency pinned in `Cargo.lock`. It selects the committed-fixture cases and sets
`HDF5_PURE_UPDATE_TEST_DATA`. Without that variable, those cases read the existing copies.

- `huge_links_12.h5`: a group of 12 links, each name so long that its link
  message is a huge heap object, for the dense-link walk in `src/group_v2.rs`.
- `paged_index.h5`: a paged, persisting file whose chunk index the C allocator
  placed as metadata, for `src/edit/`.
- `generic_large.h5`: a paged, persisting file with an attribute far larger
  than its page, so the generic-large manager holds a sub-page fragment, for
  `src/edit/`.
- `hard_link_undo.h5`: two hard links to one dataset, for `src/edit/`.
- `committed_datatype_v1.h5`: a committed datatype in a version 1 object
  header, linked once and used by two datasets, for `tests/main/named_datatypes.rs`.
