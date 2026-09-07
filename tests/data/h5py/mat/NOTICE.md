# Synthetic MATLAB v7.3 `.mat` fixtures

Unlike `tests/data/matlab/` (genuine real-MATLAB output), the `.mat` files
here are **synthetic**: built with `h5py` to follow MATLAB's documented v7.3
on-disk conventions exactly. Each file has a committed generator script so it is
reproducible.

## `struct_array_v73.mat`

Exercises the struct-*array* layout (issue #127): a `MATLAB_class="struct"` group
whose fields are datasets of per-element HDF5 object references into `#refs#`,
rather than direct value datasets (a scalar struct). The layout matches the
`foreverallama/matio` writer (`matwriter7.write_struct_array`).

Top-level variables: `row` (1×6), `col` (6×1), `grid` (2×3), `nested` (1×2 with a
nested scalar struct field), and `scalar` (a 1×1 scalar struct, as a regression
guard). Read and asserted by `tests/mat_struct_array_read.rs`.

Regenerate with:

```
python3 tests/data/h5py/mat/gen_struct_array.py tests/data/h5py/mat/struct_array_v73.mat
```
