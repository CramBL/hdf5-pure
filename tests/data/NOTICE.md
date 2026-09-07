# Test data

Committed files the tests read, one directory per writer. Each directory says
what wrote its files and how to write them again, and `just test-data` runs the
writers this repository can run.

- `c/`: the reference HDF5 C library.
- `h5py/`: h5py, with the LZF and zfp filters and MATLAB's v7.3 layout.
- `matlab/`: MATLAB itself.
- `pure/`: this crate's own writer, through seams its public API has no
  spelling for. Each file is checked by its content, never by its bytes.
- `fuzz/`: inputs the fuzzer found.
- `unattributed/`: files from the crate's first commit whose writer was not
  recorded. A file moves out when its writer is established.

A test reads a file from here and never writes one. The writers are the
generators named in each directory.
