# ZFP reference vectors and pipeline file

## Producer and extraction

The repository-authored [generator][generator] constructs numeric arrays with
h5py 3.16.0, hdf5plugin 6.0.0, and NumPy 2.4.4 under Python 3.13. The sibling
requirements file records testing with Python 3.13.5. The [manifest][manifest]
records H5Z-ZFP 1.1.1 and ZFP 1.0.1. The original HDF5 backend version is
unrecorded.

For every manifest entry named `<name>`, retain both `<name>.raw.bin` and
`<name>.compressed.bin`. The raw file contains the input array's C-order bytes.
The compressed file contains `read_direct_chunk` output from one chunk whose
shape equals the dataset shape, using the recorded fixed rate. The manifest
records scalar type, dimensions, rate, filter parameters, byte lengths, and
case notes. The generator defines the exact ramps, zeros, alternating signs,
mixed magnitudes, and partial shapes.

## Consumers and invariants

`src/zfp_crosscheck.rs` consumes every manifest pair. It compares encoded bytes
with the reference stream and checks decoding against the raw values with a
type- and rate-dependent tolerance. The manifest contains 39 pairs across
`f32`, `f64`, `i32`, and `i64`, with ranks one through four. Fourteen pairs use
integer types. Retain them as independent reference evidence. The pinned native
ZFP adapter accepts floating-point types only.

The separate `f32_1d_2_partial_rate32_zfp_deflate.h5` file contains `f32[2] = [0, 1]`
with fixed-rate ZFP at 32 bits followed by deflate level 6 and filter mask 0.
`tests/main/zfp_roundtrip.rs` reads it as a complete partial-block pipeline case.
It is not a manifest codec pair.

## Regeneration and licensing

Run `just test-data::zfp` on a little-endian host. The generator's embedded
dependencies and sibling requirements file pin the producer packages. Per-case
HDF5 extraction containers are temporary. Client-data words are retained in the
manifest, without separate parameter binaries.

The generator and test sources use the repository's MIT OR Apache-2.0 license.
No separate fixture-specific license or external data attribution is recorded.
The recorded grant in `LICENSE-ZFP` covers ported codec source.

[generator]: regen.py
[manifest]: manifest.json
