# HDF5 1.8 read fixtures

`v1_superblock.h5` and `v2_superblock.h5` were written by HDF5 1.8.23, the last
1.8 release. `tests/c_1_8_read_compat.rs` reads both, and the
`parse_v1_against_a_c_written_superblock` unit test in `src/superblock.rs`
parses the first. Both tests run under `cross test` on i686 and s390x, where
the `hdf5-metno` dev-dependency is not linked, and these are the only committed
fixtures with superblock versions 1 and 2.

`v1_superblock.h5` was created with `H5Pset_istore_k(64)` and
`H5Pset_sym_k(8, 16)`. The non-default chunk B-tree K is what makes 1.8.23
write a version 1 superblock, and all three K values differ from each other and
from the defaults, so a parser that permutes them reads back wrong. A defect
fixed in 0.33.0 read the chunk K and the status flags from each other's
offsets.

`v2_superblock.h5` was created with `H5Pset_libver_bounds(LATEST, LATEST)`
under 1.8, which is the version 2 superblock. Its chunked dataset is indexed by
a version 1 B-tree, a pairing this crate's writer does not produce.

Both hold the same objects:

- `/values`, contiguous `f64[4]`, with a `units` string attribute
- `/chunked`, `i32[1000]` in 100-element chunks, deflate level 6
- `/grp`, a group with a `tag` attribute holding `/grp/inner`
- a `root_attr` string attribute on the root group

The files are the ground truth the tests read and are not regenerated. Nothing
from the HDF5 distribution is vendored here.
