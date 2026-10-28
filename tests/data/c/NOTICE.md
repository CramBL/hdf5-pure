# Written by libhdf5

[HDF5 1.8.23 compatibility files][historical] are retained as frozen reference evidence.
Their notice records the producer, consumers, layout premises, and retention rationale.

Native interoperability tests construct fresh files through
`crates/test-util-hdf5/src/native_fixtures.rs`, under the optional `__hdf5-1.10`
capability. `crates/crosscheck/tests/main/native_fixtures.rs` checks those files
across reader backends.

[historical]: 1.8/NOTICE.md
