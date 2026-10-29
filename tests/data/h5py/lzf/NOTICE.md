# LZF reference vectors and pipeline files

## Producer and extraction

The repository-authored [generator][generator] constructs numeric arrays with
h5py 3.16.0 and NumPy 2.4.4 under Python 3.14, using h5py's built-in LZF filter.
The exact Python patch release and original HDF5 backend version are unrecorded.
The [manifest][manifest] records dimensions, chunks, pipeline order, flags,
client-data words, byte lengths, and optional-filter masks.

`<name>.raw.bin` contains the array's C-order bytes. For single-chunk LZF-only
cases, `<name>.compressed.bin` contains the bytes returned by `read_direct_chunk`.
When the optional filter is skipped, that file contains raw bytes and its
manifest mask is 1. Chained cases and cases with several chunks have no extracted stored stream.

## Files and consumers

Each case below has its named `.raw.bin` file. The stored-stream and container
columns account for all other retained binary files in this directory. Append
the listed suffixes to the case name.

| Case | Stored stream | HDF5 container | Invariant |
|---|---|---|---|
| `i16_rle` | `.compressed.bin` | None | `i16[2048]`, long runs and overlapping copies |
| `i32_ramp` | `.compressed.bin` | None | `i32[1024]`, compressible ramp |
| `i32_rank2` | `.compressed.bin` | None | `i32[16, 10]`, chunk-byte product across dimensions |
| `u8_literal_runs` | `.compressed.bin` | None | `u8[4096]`, maximum literal runs and a compressible tail |
| `u8_noise` | `.compressed.bin`, raw fallback | `.h5` | `u8[4096]`, optional LZF skipped |
| `f64_shuffle_lzf` | None | `.h5` | `f64[512]`, shuffle followed by LZF |
| `i64_multichunk` | None | `.h5` | `i64[300]`, 128-element chunks with a partial final chunk |

`src/filters/tests/lzf_reference.rs` consumes the manifest and the five stored/raw pairs. It
checks reference decoding or raw fallback, pure codec roundtrip results, and pipeline
parameters. `tests/main/lzf_roundtrip.rs` reads all three retained containers
against their raw values. It also uses `u8_noise.raw.bin` for fresh-writer
incompressibility coverage.

## Regeneration and licensing

Retain the vectors as independent decoder evidence and the containers as complete
reader pipeline evidence. Run `just test-data::lzf` on a little-endian host.
The generator's embedded dependencies and sibling requirements file pin the
producer packages. Intermediate extraction containers are temporary.

The generator and test sources use the repository's MIT OR Apache-2.0 license.
No separate fixture-specific license or external data attribution is recorded.

[generator]: regen.py
[manifest]: manifest.json
