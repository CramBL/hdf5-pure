# Cargo Features

`hdf5-pure` is split into Cargo features so you only compile the parts you use. The defaults cover the common case (filesystem I/O plus the high-level reader and writer), while the optional features add MATLAB `.mat` support, alternative compression backends, N-dimensional array I/O, and data provenance. This page is the complete reference for every feature, what it pulls in, and which guide page exercises it.

For how to declare these in `Cargo.toml`, see [Installation](../getting-started/installation.md).

## At a glance

| Feature | Default | Pulls in | Implies | Description |
|---|---|---|---|---|
| `std` | yes | — | — | File I/O and the high-level reader/writer API |
| `checksum` | yes | — | — | Jenkins hash validating checksummed metadata |
| `deflate` | yes | `flate2` (rust backend) | — | Deflate (zlib) compression, pure-Rust backend |
| `serde` | no | `serde` | `std` | Serialize/deserialize MATLAB v7.3 `.mat` files via serde |
| `fast-deflate` | no | `flate2/zlib-ng` | — | zlib-ng backend for deflate |
| `ndarray` | no | `ndarray` crate | `std` | N-dimensional array I/O via the [`ndarray`](https://docs.rs/ndarray) crate |
| `num-complex` | no | `num-complex` | `serde` | `mat::ComplexElement` for `num_complex::Complex<T>`, for the bulk complex-array helpers |
| `provenance` | no | `sha2` | — | SHA-256 data provenance tracking |
| `zfp` | no | — | — | ZFP fixed-rate compression (HDF5 filter 32013), `f32`/`f64`/`i32`/`i64` x 1D-4D |
| `heap-baseline` | no | — | — | Maintainer-only: check the recorded allocation figures (see below) |

The default feature set is `std`, `checksum`, and `deflate`.

!!! note
    `serde` and `ndarray` both imply `std` because they build on the `File`, `Group`, and `Dataset` reader APIs and the `FileBuilder` writer, which require the standard library. Enabling either one enables `std` automatically.

## Default features

### `std`

Enables the standard library. This brings in the entire high-level reader and writer surface — `File`, `FileBuilder`, `Group`, `Dataset`, `File::open_rw`, `File::open_swmr_writer`, `Dataset::append`, `repack`, the `mat` module, and the in-memory, filesystem, and caller-supplied-source entry points (`FileBuilder::finish`, `File::from_bytes`, `File::open`, `File::open_streaming`, `File::from_source`, `FileBuilder::write`). The whole high-level API is `std`-gated: with `std` disabled the crate is `no_std` and exposes only the lower-level datatype and builder primitives, not `File` / `FileBuilder`. Because `std` is available on `wasm32-unknown-unknown`, a WASM build keeps it (the WASM and `no_std` section below has the details).

### `checksum`

Enables the Jenkins lookup3 hash used to validate and emit the checksums HDF5 puts on its metadata: version 2 and later object headers, the superblock, version 2 B-tree nodes, fractal heaps, and the Extensible-Array and Fixed-Array chunk indexes. It has no extra dependency. Keep this enabled for broad compatibility with files the reference HDF5 C library and h5py produce; it is also the one feature you should keep when targeting WASM (see below).

### `deflate`

Enables Deflate (zlib) compression and decompression through a pure-Rust backend (`flate2` with its `rust_backend`). This is what backs `DatasetBuilder::with_deflate`. See [Compression](../guide/compression.md) for usage.

## Optional features

### `serde`

Adds serde-based (de)serialization of MATLAB v7.3 `.mat` files through the `hdf5_pure::mat` module (`mat::to_file`, `mat::to_writer`, `mat::from_file`, `Matrix`, `Complex32`, `Complex64`). It pulls in the `serde` dependency and implies `std`. See [MATLAB v7.3 interop](../interop/matlab.md).

!!! note
    Only the serde-driven entry points are gated. The mid-level `mat::MatBuilder` — including `write_blocks`, `finish_to`, and `write` — needs `std` but not `serde`, so a writer that builds its `.mat` explicitly rather than through `Serialize` does not pay for the dependency.

!!! tip
    The `matlab_fixtures` example requires this feature and can be run with `cargo run --example matlab_fixtures --features serde`. The `mat_streaming` example needs only the defaults: `cargo run --example mat_streaming`.

### `fast-deflate`

Switches the deflate backend to zlib-ng via `flate2/zlib-ng`, trading the pure-Rust backend for the faster zlib-ng implementation. It complements (and does not replace) `deflate`; the Deflate API is unchanged. Because zlib-ng is a native dependency, this is for native builds rather than the pure-Rust WASM path.

### `ndarray`

Adds ergonomic N-dimensional array I/O via the [`ndarray`](https://docs.rs/ndarray) crate: `DatasetBuilder::with_ndarray` to write and `Dataset::read_array` / `Dataset::read_array_dyn` to read. Shape and datatype come from the array, and data is stored row-major (C order). It pulls in the `ndarray` crate and implies `std`. See the [ndarray guide](../guide/ndarray.md).

!!! tip
    The `ndarray_io` example requires this feature and can be run with `cargo run --example ndarray_io --features ndarray`.

### `num-complex`

Implements `mat::ComplexElement` for `num_complex::Complex<T>`, so a slice of the de-facto standard Rust complex type can be handed to the bulk array helpers (`mat::complex::i16_array` and friends) without a conversion pass. It implies `serde`. See [large complex arrays](../interop/matlab.md#large-complex-arrays).

!!! note
    The impls have to ship here rather than in your crate: `ComplexElement` is `unsafe` and asserts a memory layout, and the orphan rule allows the impl only from a crate that owns one of the two types. Your *own* complex type needs no feature — implement the trait for it directly.

### `provenance`

Adds SHA-256 data provenance tracking, pulling in `sha2`. `DatasetBuilder::with_provenance(creator, timestamp, source)` records provenance attributes alongside a dataset, and `Dataset::verify_provenance` recomputes the hash and returns a `VerifyResult`. The provenance attributes are stored under conventional names (`_provenance_sha256`, `_provenance_creator`, `_provenance_timestamp`, `_provenance_source`).

```rust
use hdf5_pure::{File, FileBuilder};

let mut builder = FileBuilder::new();
builder.create_dataset("measurements")
    .with_f64_data(&[1.0, 2.0, 3.0])
    .with_provenance("acquisition-rig", "2026-06-16T00:00:00Z", None);
let bytes = builder.finish().unwrap();

let file = File::from_bytes(bytes).unwrap();
let result = file.dataset("measurements").unwrap().verify_provenance().unwrap();
```

!!! note
    `verify_provenance` and `VerifyResult` require both `std` and `provenance`.

### `zfp`

Enables a pure-Rust fixed-rate port of the LLNL/zfp codec, registered as HDF5 filter ID 32013, exposed through `DatasetBuilder::with_zfp(rate)`. It supports `f32`, `f64`, `i32`, and `i64` datasets in ranks 1D through 4D in fixed-rate mode. Files written with it are byte-for-byte interoperable with the reference H5Z-ZFP plugin (`h5py` + `hdf5plugin`). It has no extra crate dependency. See [Compression](../guide/compression.md).

```rust
// Compile with `--features zfp`
let mut builder = hdf5_pure::FileBuilder::new();
builder.create_dataset("temperature")
    .with_f32_data(&data)
    .with_shape(&[ny, nx])
    .with_chunks(&[ny, nx])
    .with_zfp(16.0);  // 16 bits per value
```

## Maintainer-only features

### `heap-baseline`

`heap-baseline` is a test-only / maintainer feature. It enables `tests/allocation_baseline.rs`, which checks the crate's exact allocation counts and byte totals for one write-then-read cycle against the figures committed under `tests/baselines/`. It pulls in nothing (the heap profiler it uses, [`heapscope`](https://crates.io/crates/heapscope), is an unconditional dev-dependency), it is not a run-time dependency, and end users do not need it.

The figures it checks belong to one target, one toolchain and one feature set, so the test compiles only under the crate's *default* features and CI runs it in a single pinned job. The bounds that hold everywhere — a windowed read allocates on the order of its window, a chunked read costs a constant per chunk — are in `tests/allocation_bounds.rs` and need no feature at all. On x86_64, both need the frame pointers `.cargo/config.toml` sets.

### `matio-crosscheck`

`matio-crosscheck` is a test-only / maintainer feature. It enables a crosscheck integration test that links against the system `libmatio` (the reference MATLAB MAT file library, installed via `brew install libmatio` or `apt install libmatio-dev`) to validate `.mat` output. It implies `serde`, is not a run-time dependency, and end users do not need it.

### The crosscheck package

The tests that link the reference C library are a separate package, `hdf5-pure-crosscheck` under `crates/crosscheck/`, so that nothing else in the repository needs a C library. Its features are not this crate's. `__hdf5-bundled` builds and links the libhdf5 that `hdf5-metno` bundles, and without it the package links the installation under `HDF5_DIR`. `__hdf5-1.10`, `__hdf5-1.12`, `__hdf5-1.14` and `__hdf5-2` name the release series of the linked libhdf5, from 1.10 up. Each implies the one below, and a test gates on the lowest series that has what it needs. The interop workflow passes its row's, `__hdf5-bundled` implies the bundled release's, and `crates/crosscheck/tests/hdf5_release_feature.rs` checks the one passed against the library linked.

## WASM and `no_std`

`hdf5-pure` builds for `wasm32-unknown-unknown` with no C dependencies. Because `std` is available on that target and the high-level API is `std`-gated, a WASM build keeps the default features (which include `std`) — turning them off would compile `File` and `FileBuilder` away. Add the target and build:

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown
```

In the browser you use the in-memory entry points, `FileBuilder::finish` (returning `Vec<u8>`) and `File::from_bytes`; the path-based entry points compile but cannot reach a filesystem at runtime.

For bare-metal `no_std` (for example `thumbv7em-none-eabi`), turn the default features off and keep `checksum`:

```toml
[dependencies]
hdf5-pure = { version = "0.14", default-features = false, features = ["checksum"] }
```

The crate then compiles as `#![no_std]` with only `alloc`, but the `std`-gated `File` / `FileBuilder` API is absent — a `no_std` build exposes the lower-level primitives rather than the whole-file reader and writer. See [Portability](../interop/portability.md) for the full per-target breakdown.

!!! warning
    `fast-deflate` uses the native zlib-ng backend and is intended for native builds, not the pure-Rust WASM target.
