# Portability: WASM, no_std & the C Library

`hdf5-pure` is pure Rust with no C dependencies and no build-time linkage to
libhdf5, which gives it three portability properties that the reference library
cannot offer: it compiles to WebAssembly, it compiles for `no_std` targets with
`alloc`, and the files it produces are byte-compatible with the rest of the HDF5
ecosystem. This page covers all three, including what is and is not available
without `std`.

## WebAssembly

`hdf5-pure` builds for `wasm32-unknown-unknown` with no extra toolchain. The key
fact to get right: the Rust `std` library **is** available on that target, so
you keep the default features (which include `std`). The crate's high-level
reader and writer are gated behind `std`, so turning default features off would
compile `File` and `FileBuilder` away — exactly what you do *not* want for a
WASM app.

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown
```

In the browser you use the **in-memory** path, which never touches a filesystem:
[`FileBuilder::finish`](../guide/writing.md) returns the complete file as a
`Vec<u8>` you can hand to JavaScript, and `File::from_bytes` parses bytes you
get back.

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

let bytes: Vec<u8> = builder.finish().unwrap(); // in memory, no filesystem
```

Reading is symmetric:

```rust
use hdf5_pure::File;

let file = File::from_bytes(bytes).unwrap();
let values = file.dataset("x").unwrap().read_f64().unwrap();
```

The path-based entry points (`File::open`, `FileBuilder::write`, `File::open_rw`,
`File::open_swmr_writer`) still compile for WASM, but they cannot reach a
filesystem at runtime in the browser. Build your WASM code around `finish` and
`from_bytes`.

`from_bytes` needs the whole file in memory, which a large one will not be. When
the host can serve byte ranges — a `fetch` with a `Range` header, or a WASI
import — implement `Source` over it and open the file with `File::from_source`
instead: metadata and chunks are then read on demand and peak memory tracks what
you actually read. See [Streaming](../guide/streaming.md#streaming-from-something-that-is-not-a-path).

!!! tip "Trimming the build"

    `deflate` is on by default (a pure-Rust backend, so it compiles to WASM
    fine). If you only handle uncompressed datasets you can drop it with
    `default-features = false, features = ["std", "checksum"]` — but keep `std`,
    or the high-level API disappears.

## no_std with alloc

With the default features off, the crate is `#![no_std]` and relies only on
`alloc` — it allocates `Vec`s and similar but never calls the operating system.
It compiles for freestanding targets, and CI builds `thumbv7em-none-eabi` to
keep that honest.

There is an important limitation today. The high-level, path-and-image API —
`File`, `FileBuilder`, `repack`, and
the `mat` module — is `std`-gated, so a pure-`no_std` build (`--no-default-features`)
*compiles* but does not expose the whole-file reader and writer. What stays
available without `std` is the lower-level surface: the datatype constructors
(`make_f64_type` and friends), the `DatasetBuilder` / `GroupBuilder` and the
compound/enum type builders, `ScaleOffset`, and the format primitives. So
`no_std` is a supported *compilation* target for embedding the format
machinery; building or reading a complete file still needs `std` — which, as
shown above, is available on `wasm32-unknown-unknown`.

| Capability | API | Requires `std` |
|---|---|:---:|
| Datatype & builder primitives | `make_*_type`, `DatasetBuilder`, `GroupBuilder`, `ScaleOffset` | no (`alloc` only) |
| Build a whole file in memory | `FileBuilder::new` / `FileBuilder::finish` | yes |
| Parse a file from memory | `File::from_bytes` | yes |
| Streaming read from host-served byte ranges | `File::from_source`, `Source` | yes |
| Open a file by path | `File::open` | yes |
| Streaming read by path | `File::open_streaming` | yes |
| SWMR follow read by path | `File::open_swmr` | yes |
| Write a file to a path | `FileBuilder::write` | yes |
| Edit a file in place | [`File::open_rw`](../guide/editing.md) | yes |
| Append in SWMR mode | [`File::open_swmr_writer`](../guide/swmr.md) | yes |
| Append in place (non-SWMR) | [`File::open_rw` + `Dataset::append`](../guide/editing.md#streaming-appends) | yes |
| Compact a file | [`repack`](../guide/repack.md) | yes |
| MATLAB `.mat` via serde | `mat` module | yes (`serde`) |
| N-dimensional array I/O | `with_ndarray` / `read_array` | yes (`ndarray`) |

!!! note

    The `ndarray` and `serde` features both imply `std`, because they build on
    the path-based `File` / `Dataset` reader and writer APIs. See the
    [features reference](../reference/features.md) for the full feature matrix
    and the [Installation section of the README](https://github.com/CramBL/hdf5-pure#installation)
    for dependency setup.

## Reference-library interoperability

`hdf5-pure` does not define its own dialect of HDF5: it writes and reads the
standard on-disk format. Files this crate writes are readable by the reference
HDF5 C library and by `h5py`; files those tools produce are readable here. This
holds for the format features the crate supports — multiple superblock versions,
object header layouts, contiguous and chunked storage, and the built-in deflate,
shuffle, and scale-offset filters, plus h5py's LZF.

MATLAB needs one more thing said about it, because which HDF5 library it links
has changed across releases and decides whether it can open a file at all. It
was 1.8.12 before R2021b, 1.10.7 in R2021b, 1.10.x through R2024a, and 1.14.4.3
since R2024b. A version 3 superblock is a 1.10 addition, so a file in this
crate's older default could not be opened by MATLAB before R2021b. The `mat`
writer therefore emits the HDF5 1.8 format by default
(`mat::Options::libver`), which every one of those releases reads, and
`FileBuilder::with_libver_bounds` reaches the same format for a plain `.h5`
file destined for an old reader. Around R2021b MathWorks also shipped two
libraries at once, keeping 1.8.12 on the MAT v7.3 path while `h5read`/`h5disp`
used 1.10.7 — the split behind the odd symptom of a file `h5disp` prints and
`load` refuses. That is not confined to R2021b: R2023a reports HDF5 1.10.8 and
its `load` still refuses a version 3 superblock, so the linked library version
does not tell you which formats `load` accepts (see
[MATLAB v7.3 files](matlab.md#the-on-disk-format-matlabs-load-needs)). Real
MATLAB writes an older format still: a version 0
superblock with v1 symbol-table groups, which this crate reads but does not
produce.

Interoperability is not asserted by hand. It is enforced by byte-level
crosscheck tests that compare the bytes this crate emits against fixtures
produced by the reference toolchain, so a regression in the on-disk layout fails
the test suite rather than slipping out as a quietly incompatible file. The same
discipline backs the optional [ZFP filter](../guide/compression.md)
(`src/zfp_crosscheck.rs` compares against `h5py` + `hdf5plugin`) and the MATLAB
`.mat` path. For the cross-tool story in depth, see the
[MATLAB interop page](matlab.md).

The 1.8 output format is the one claim those tests cannot make, because every
library they link is 1.10 or newer and reads both formats happily.
`crates/crosscheck/tests/libver_matrix.rs` covers it against every release the
interop workflow builds, 1.8.23 included: the 1.10 format cannot be opened at
all before 1.10, and the 1.8 format reads completely everywhere. That
measures the format boundary rather than any particular MathWorks build, which
only MATLAB itself can confirm — `examples/octave/check_format.m` asks it
directly.

The [LZF filter](../guide/compression.md) is the one place where byte-comparison
does not apply in both directions, so it is worth being precise about what is
checked. On read, `src/lzf_crosscheck.rs` decodes h5py's own compressed streams
and compares against the expected bytes. On write, it compares the filter
pipeline this crate emits — ids, order, name, optional flag, `cd_values` —
against what h5py recorded for the same dataset, but *not* the compressed
stream: LZF has many valid encodings of the same data, so matching liblzf byte
for byte is not a requirement and not a goal. That h5py decodes the streams this
crate produces is verified separately, by the read-back phase of
`tests/data/h5py/lzf/regen.py`, which needs a live h5py and so runs when the
fixtures are regenerated rather than in CI.

### Host-independent output

No property of the machine doing the writing — its architecture, pointer width,
or cache-line size — reaches the file. The same input written on `aarch64` and
on `x86_64` produces the same bytes, and a file does not grow because of the
platform that wrote it. Chunks in particular are stored back to back, with no
alignment padding; a workload that needs cache-line-aligned data should align
its own buffers, since the file carries no padding to inherit.

This is a claim about the *host*, and it holds at a fixed feature set. Two
things sit outside it:

- Output depends on the order the data is handed over. Serializing a `.mat`
  from a `HashMap` writes its fields in that map's iteration order, which the
  standard library randomizes per map, so two equal `HashMap`s can produce
  different files on one machine in one run. Use a `BTreeMap`, or a struct,
  when the field order has to be stable.
- The `fast-deflate` feature swaps the Rust deflate backend for zlib-ng, which
  dispatches on runtime CPU features. Compressed bytes are then a property of
  the machine after all. The default `deflate` backend is pure Rust and does
  not have this behavior.

### 32-bit safety

The same crosscheck discipline extends to 32-bit hosts. Every offset and length
read out of a file is narrowed through checked conversions, so a 64-bit value
that does not fit a 32-bit `usize` produces an error rather than a silent
truncation. This is why a file too large for 32-bit address space should be read
with `File::open_streaming` (see [streaming](../guide/streaming.md)) instead of
`File::open`. CI exercises the suite on `i686` under QEMU.

### Memory safety

The crate is almost entirely safe Rust, and the default feature set contains no
non-trivial `unsafe` at all. Two features introduce some, and they are mutually
exclusive rather than cumulative:

- `std` + `serde` compiles the tiled row-major/column-major transpose used by
  the MATLAB writer, which writes through a raw pointer into uninitialized
  `Vec` capacity. It is exercised under Miri with strict provenance in CI.
- A `no_std` build instead compiles a single-threaded `Mutex` replacement whose
  `Send`/`Sync` rest on the target being single-threaded rather than on a lock.
  Because `no_std` excludes the `mat` module entirely, this replaces the
  transpose rather than adding to it.

The [crate documentation on docs.rs](https://docs.rs/hdf5-pure/latest/hdf5_pure/)
covers the safety and robustness guarantees in more detail.
