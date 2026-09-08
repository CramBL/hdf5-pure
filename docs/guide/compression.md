# Compression & Filters

This page covers the storage filters hdf5-pure can apply to a dataset: deflate, shuffle, scale-offset, LZF, and (behind a feature flag) ZFP. Filters shrink on-disk size while keeping the file readable by any standard HDF5 tool, because every filter here is either a built-in HDF5 filter or a registered third-party one.

!!! tip "Runnable example"
    A complete, runnable program lives at [`examples/compression.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/compression.rs). Run it with:

    ```bash
    cargo run --example compression
    ```

    It writes the same signal with several filter combinations, reports the resulting file sizes, and verifies that the lossless paths reproduce the input exactly.

## Filters require a chunked layout

Filters apply per chunk, so a dataset must use chunked storage before it can be compressed. Enable chunking with `with_chunks` and pass the chunk dimensions. Each compression method (`with_deflate`, `with_lzf`, `with_scale_offset`, `with_zfp`) implies chunked storage if you have not already set it, but choosing the chunk shape explicitly is recommended since it determines the unit of compression and of partial reads.

```rust
use hdf5_pure::FileBuilder;

let data: Vec<f64> = (0..10_000).map(|i| (i as f64 / 50.0).sin() * 1000.0).collect();

let mut builder = FileBuilder::new();
builder
    .create_dataset("signal")
    .with_f64_data(&data)
    .with_chunks(&[1000])
    .with_deflate(6);
let bytes = builder.finish().unwrap();
```

See [Writing datasets](writing.md) for the full dataset builder API.

## Deflate

`with_deflate(level)` enables zlib (DEFLATE) compression, where `level` is the standard 0–9 zlib effort setting (higher compresses harder for more CPU). This is the most broadly useful filter and the one applied by most HDF5 producers.

```rust
builder
    .create_dataset("compressed")
    .with_f64_data(&data)
    .with_chunks(&[100])
    .with_deflate(6);
```

By default the `deflate` feature is on (it is part of the crate's default features) and uses a pure-Rust backend. See [the `fast-deflate` feature](#fast-deflate-backend) below to swap in a faster backend.

## Shuffle

`with_shuffle()` enables the shuffle filter, which reorders the bytes within each chunk so that bytes of like significance sit together. Shuffle does not compress on its own; it rearranges data so a following deflate pass usually compresses better. Chain it before `with_deflate`:

```rust
builder
    .create_dataset("shuffled")
    .with_f64_data(&data)
    .with_chunks(&[100])
    .with_shuffle()
    .with_deflate(6);
```

## LZF

`with_lzf()` enables the LZF filter (id 32000), the fast lossless compressor h5py registers and ships with. It trades compression ratio for speed relative to deflate, and it is the natural choice for files exchanged with h5py, which reads and writes LZF out of the box (the plain C library needs h5py's filter plugin). Like deflate, it pairs well with shuffle; combining it with deflate on the same dataset is refused, since stacking two byte compressors is never useful.

```rust
builder
    .create_dataset("fast")
    .with_f64_data(&data)
    .with_chunks(&[100])
    .with_shuffle()
    .with_lzf();
```

## Scale-offset

Scale-offset (HDF5 filter id 6) stores each chunk's values as offsets from that chunk's minimum, packed into the fewest bits the chunk's range needs. It has two modes, selected by the [`ScaleOffset`](../reference/data-types.md) enum passed to `with_scale_offset`:

| Mode | Datatype | Loss |
|---|---|---|
| `ScaleOffset::Integer(minbits)` | signed / unsigned integers | lossless |
| `ScaleOffset::FloatDScale(decimals)` | `f32` / `f64` | lossy, to `decimals` decimal digits |

`ScaleOffset::Integer(minbits)` is lossless for integer datasets. Pass `0` to let the encoder pick the minimum bit width per chunk (the usual choice); a positive value forces a fixed minimum bit width.

```rust
use hdf5_pure::ScaleOffset;

// Integer mode is lossless. `0` lets the encoder pick the bit width per chunk.
builder
    .create_dataset("counts")
    .with_i32_data(&counts)
    .with_chunks(&[1000])
    .with_scale_offset(ScaleOffset::Integer(0));
```

`ScaleOffset::FloatDScale(decimals)` is lossy for float datasets: values are multiplied by `10^decimals`, rounded to integers, and then packed like integer mode, so the read-back is close but not exact within the retained digits. It may be followed by `with_deflate` for additional savings:

```rust
// Float D-scale is lossy: values are rounded to N decimal digits before packing.
builder
    .create_dataset("readings")
    .with_f64_data(&readings)
    .with_chunks(&[1000])
    .with_scale_offset(ScaleOffset::FloatDScale(3)) // keep 3 decimal digits
    .with_deflate(6);                               // may be followed by deflate
```

!!! note "Fill values in a scale-offset filter"
    The filter records the dataset's fill value, and an element equal to it is stored as a reserved code rather than as an offset — so a chunk that is mostly fill packs to the width of what is left. The reference C library (and h5py) records one on *every* scale-offset dataset, since an untouched fill value still counts as defined, and this crate does the same: `with_fill_value` is carried into the filter's parameters, and a dataset without one records the library default of zero. Reserving that code costs one bit per element on a chunk whose values span an exact power of two, which is what the reference pays for it too. In the lossy float D-scale mode an element counts as a fill value when it lands within one decimal quantum of it, so a value that close decodes as the fill value rather than as itself.

    Both forms are read and written, so appending to or repacking a dataset whose filter records no fill value keeps it that way.

!!! warning "Mode must match the datatype"
    The datatype class, sign, and byte order are derived from the dataset's datatype when the file is written, so the mode must match the data: integer mode on `with_i*` / `with_u*` data, float mode on `with_f32` / `with_f64` data. A mismatch makes `finish()` / `write()` return a `FormatError`. Scale-offset is mutually exclusive with ZFP and with shuffle, but may be followed by `with_deflate` or `with_lzf`.

## Filter chaining

Filters compose. Shuffle is meant to be chained before deflate, and scale-offset may be followed by deflate. The builder methods return `&mut Self`, so chain them in order:

```rust
// shuffle then deflate
builder
    .create_dataset("a")
    .with_f64_data(&data)
    .with_chunks(&[1000])
    .with_shuffle()
    .with_deflate(6);

// scale-offset then deflate
builder
    .create_dataset("b")
    .with_f64_data(&data)
    .with_chunks(&[1000])
    .with_scale_offset(ScaleOffset::FloatDScale(2))
    .with_deflate(6);
```

### Combinations that are refused

Scale-offset and ZFP each consume the raw elements and hand on something else, so anything they would sit on top of has nothing left to work with. Asking for both halves of such a pair is a contradiction, and `finish()` / `write()` returns a `FormatError::FilterError` naming the two rather than quietly dropping one:

| Requested with | Refused |
| --- | --- |
| `with_zfp` | `with_shuffle`, `with_scale_offset`, `with_deflate`, `with_lzf` |
| `with_scale_offset` | `with_shuffle`, `with_zfp` |
| `with_lzf` | `with_deflate` |

`with_lzf` + `with_deflate` is the same shape for a different reason: the two fill one byte-compressor slot, and stacking them is never useful.

## Reads are transparent

Decompression is automatic. Whatever filters a dataset was written with, the same `read_*` call returns the decoded values; there is nothing to configure on the read side, and lossless paths reproduce the input exactly.

```rust
use hdf5_pure::File;

let back = File::from_bytes(bytes)
    .unwrap()
    .dataset("signal")
    .unwrap()
    .read_f64()
    .unwrap();
assert_eq!(back, data);
```

## Portability

Deflate, shuffle, and scale-offset are all built-in HDF5 filters, so files hdf5-pure writes with them stay readable by the reference HDF5 C library, h5py, and MATLAB, and files those tools produce with the same filters are readable by hdf5-pure. LZF is h5py's registered filter rather than a built-in: h5py reads it natively, while the plain C library and MATLAB need h5py's filter plugin. `src/lzf_crosscheck.rs` decodes h5py's own streams and checks the filter pipeline hdf5-pure writes against the one h5py records; the compressed stream itself is not byte-compared, since LZF has many valid encodings of the same data, so h5py's ability to read hdf5-pure's output is verified by the read-back phase of `tests/data/h5py/lzf/regen.py`. See [Portability](../interop/portability.md) for the broader interoperability picture and what each direction covers.

## ZFP

!!! note "Requires the `zfp` feature"
    ZFP support is gated behind the `zfp` Cargo feature. Enable it in `Cargo.toml`:

    ```toml
    [dependencies]
    hdf5-pure = { version = "0.14", features = ["zfp"] }
    ```

    Or, for the runnable example, `cargo run --example compression --features zfp` if you extend it to exercise ZFP.

`with_zfp(rate)` enables fixed-rate ZFP compression, where `rate` is the number of compressed bits per value. It is a pure-Rust port of the LLNL/zfp codec, registered as HDF5 filter ID 32013, and is byte-for-byte interoperable with the reference H5Z-ZFP plugin: files hdf5-pure writes are readable by `h5py` + `hdf5plugin`, and files those tools produce are readable by hdf5-pure.

```rust
// Compile with `--features zfp`
builder
    .create_dataset("temperature")
    .with_f32_data(&data)
    .with_shape(&[ny, nx])
    .with_chunks(&[ny, nx])
    .with_zfp(16.0); // 16 bits per value
```

The supported slice is:

- Scalar types: `f32`, `f64`, `i32`, `i64`
- Ranks: 1D, 2D, 3D, 4D
- Mode: fixed-rate (`rate` bits per value)

The scalar type is derived from the dataset's datatype when the file is written, so any of `with_f32_data` / `with_f64_data` / `with_i32_data` / `with_i64_data` (or an explicit `with_dtype`) establishes it. `finish()` / `write()` returns `FormatError::UnsupportedZfp` if the dataset's datatype is not one of the four supported scalar types, or if the chunk rank is outside `1..=4`. ZFP is a standalone compressor: requesting shuffle, scale-offset, deflate, or LZF on the same dataset is refused rather than dropped.

## `fast-deflate` backend

The `fast-deflate` feature swaps the deflate backend in for the zlib-ng backend (`flate2/zlib-ng`), which is faster on supported platforms while producing standard-compatible output. The deflate API (`with_deflate`) is unchanged; only the backend differs. See the [feature reference](../reference/features.md) for the full feature matrix.
