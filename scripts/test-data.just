set working-directory := ".."

# List the remaining producer-specific fixture commands.
default:
    @{{ quote(just_executable()) }} --list test-data

# tests/data/h5py/lzf: h5py's LZF filter.
lzf:
    uv run tests/data/h5py/lzf/regen.py

# tests/data/h5py/zfp: h5py with the zfp plugin.
zfp:
    uv run tests/data/h5py/zfp/regen.py
