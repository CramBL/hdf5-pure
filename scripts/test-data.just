set working-directory := ".."

# The committed files under tests/data, rewritten by the tool that made them.
default: c pure

# tests/data/c: written by the reference C library, through the crosscheck package.
c:
    HDF5_PURE_UPDATE_TEST_DATA=1 cargo nextest run --locked -p hdf5-pure-crosscheck --features __hdf5-bundled,serde,ndarray --test c_test_data

# tests/data/pure: written by this crate's unit tests through crate-private seams.
pure:
    HDF5_PURE_UPDATE_TEST_DATA=1 cargo test --locked --lib

# tests/data/h5py/lzf: h5py's LZF filter.
lzf:
    uv run tests/data/h5py/lzf/regen.py

# tests/data/h5py/zfp: h5py with the zfp plugin.
zfp:
    uv run tests/data/h5py/zfp/regen.py
