# Historical pure-writer fixture

`dense_attrs_colliding_v0_28_0.h5` was written by the MIT-licensed hdf5-pure 0.28.0
before the fix for colliding attribute names in a multi-level dense index (#225).

`c_repairs_a_file_written_before_the_fix` in
`crates/crosscheck/tests/main/dense_attr_limits.rs` checks that native lookups miss
half of the colliding names, the pure reader reads every value, and repacking
produces a file the native reader can search completely.

Retain the original bytes as frozen evidence for repairing the malformed index order.
This fixture has no regeneration command.
