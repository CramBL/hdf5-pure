set working-directory := ".."

# Regenerate committed fixtures produced by the pure writer.
default: pure

# tests/data/pure: written by this crate's unit tests through crate-private seams.
pure:
    HDF5_PURE_UPDATE_TEST_DATA=1 cargo test --locked --lib

# tests/data/h5py/lzf: h5py's LZF filter.
lzf:
    uv run tests/data/h5py/lzf/regen.py

# tests/data/h5py/mat: synthetic MAT v7.3 files built with h5py.
mat:
    uv run tests/data/h5py/mat/gen_struct_array.py tests/data/h5py/mat/struct_array_v73.mat
    uv run tests/data/h5py/mat/gen_struct_array.py --mismatched tests/data/h5py/mat/struct_array_mismatched_v73.mat

# tests/data/h5py/zfp: h5py with the zfp plugin.
zfp:
    uv run tests/data/h5py/zfp/regen.py
