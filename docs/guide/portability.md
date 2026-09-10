`hdf5-pure` is pure Rust with no C dependencies and no build-time linkage to
libhdf5, which gives it three portability properties that the reference library
cannot offer: it compiles to WebAssembly, it compiles for `no_std` targets with
`alloc`, and the files it produces are byte-compatible with the rest of the HDF5
ecosystem. This page covers all three, including what is and is not available
without `std`.

## WebAssembly

`hdf5-pure` builds for `wasm32-unknown-unknown` with no extra toolchain. The
Rust `std` library is available on that target, so a WASM build keeps the
default features, which include `std`. The crate's high-level reader and writer
are gated behind `std`: turning the default features off compiles
[`File`](crate::File) and [`FileBuilder`](crate::FileBuilder) away.

```console
$ rustup target add wasm32-unknown-unknown
$ cargo build --target wasm32-unknown-unknown
```

In the browser the in-memory path is the one to use, since a browser has no
filesystem: [`FileBuilder::finish`](crate::FileBuilder::finish) returns the
complete file as a `Vec<u8>` to hand to JavaScript, and
[`File::from_bytes`](crate::File::from_bytes) parses the bytes that come back.

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);

let bytes: Vec<u8> = builder.finish()?; // in memory, no filesystem
# assert!(hdf5_pure::is_hdf5_bytes(&bytes));
# Ok::<(), hdf5_pure::Error>(())
```

Reading is symmetric:

```rust
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);
# let bytes = builder.finish()?;
use hdf5_pure::File;

let file = File::from_bytes(bytes)?;
let values = file.dataset("x")?.read_f64()?;
# assert_eq!(values, vec![1.0, 2.0]);
# Ok::<(), hdf5_pure::Error>(())
```

The path-based entry points ([`File::open`](crate::File::open),
[`FileBuilder::write`](crate::FileBuilder::write),
[`File::open_rw`](crate::File::open_rw),
[`File::open_swmr_writer`](crate::File::open_swmr_writer)) compile for WASM, and
in the browser they have no filesystem to reach at runtime. WASM code is built
around [`finish`](crate::FileBuilder::finish) and
[`from_bytes`](crate::File::from_bytes).

[`from_bytes`](crate::File::from_bytes) needs the whole file in memory, and a
large file does not fit. Where the host can serve byte ranges, through a `fetch`
with a `Range` header or a WASI import, implement [`Source`](crate::Source) over
it and open the file with [`File::from_source`](crate::File::from_source):
metadata and chunks are then read on demand, and peak memory tracks what the
reader took. See [Streaming large
files](crate::_guide::streaming#streaming-from-something-that-is-not-a-path).

Trimming the build: `deflate` is on by default, and its backend is pure Rust, so
it compiles to WASM. A build that handles only uncompressed datasets drops it
with `default-features = false, features = ["std", "checksum"]`. Keep `std`,
which the whole high-level API is gated behind.

## `no_std` with `alloc`

With the default features off, the crate is `#![no_std]` and relies on `alloc`
alone: it allocates `Vec`s and similar without a system call. It compiles for
freestanding targets, and CI builds `thumbv7em-none-eabi` to keep that honest.

The high-level, path-and-image API is `std`-gated: [`File`](crate::File),
[`FileBuilder`](crate::FileBuilder), [`repack`](crate::repack()), and the
[`mat`](crate::mat) module. A pure-`no_std` build (`--no-default-features`)
compiles, and it exposes the lower-level surface alone: the datatype
constructors ([`make_f64_type`](crate::make_f64_type) and its siblings), the
[`DatasetBuilder`](crate::DatasetBuilder) / [`GroupBuilder`](crate::GroupBuilder)
and the compound/enum type builders, [`ScaleOffset`](crate::ScaleOffset), and the
format primitives. So `no_std` is a supported compilation target for embedding
the format machinery, and building or reading a complete file needs `std`,
which, as shown above, is available on `wasm32-unknown-unknown`.

| Capability | API | Requires `std` |
|---|---|:---:|
| Datatype & builder primitives | `make_*_type`, [`DatasetBuilder`](crate::DatasetBuilder), [`GroupBuilder`](crate::GroupBuilder), [`ScaleOffset`](crate::ScaleOffset) | no (`alloc` only) |
| Build a whole file in memory | [`FileBuilder::new`](crate::FileBuilder::new) / [`FileBuilder::finish`](crate::FileBuilder::finish) | yes |
| Parse a file from memory | [`File::from_bytes`](crate::File::from_bytes) | yes |
| Streaming read from host-served byte ranges | [`File::from_source`](crate::File::from_source), [`Source`](crate::Source) | yes |
| Open a file by path | [`File::open`](crate::File::open) | yes |
| Streaming read by path | [`File::open_streaming`](crate::File::open_streaming) | yes |
| SWMR follow read by path | [`File::open_swmr`](crate::File::open_swmr) | yes |
| Write a file to a path | [`FileBuilder::write`](crate::FileBuilder::write) | yes |
| Edit a file in place | [`File::open_rw`](crate::File::open_rw) | yes |
| Append in SWMR mode | [`File::open_swmr_writer`](crate::File::open_swmr_writer) | yes |
| Append in place (non-SWMR) | [`File::open_rw`](crate::File::open_rw) + [`Dataset::append`](crate::Dataset::append) | yes |
| Compact a file | [`repack`](crate::repack()) | yes |
| MATLAB `.mat` via serde | [`mat`](crate::mat) module | yes (`serde`) |
| N-dimensional array I/O | [`with_ndarray`](crate::DatasetBuilder::with_ndarray) / [`read_array`](crate::Dataset::read_array) | yes (`ndarray`) |

The `ndarray` and `serde` features both imply `std`, because they build on the
path-based [`File`](crate::File) / [`Dataset`](crate::Dataset) reader and writer
APIs. [Cargo features](crate#cargo-features) has the full feature matrix, and
the [Installation section of the README](https://github.com/CramBL/hdf5-pure#installation)
covers dependency setup.

## Reference-library interoperability

`hdf5-pure` writes and reads the standard on-disk format, and defines no dialect
of its own. The reference HDF5 C library and `h5py` read the files this crate
writes, and this crate reads the files those tools produce. That holds for the
format features the crate supports: multiple superblock versions, object header
layouts, contiguous and chunked storage, the built-in deflate, shuffle, and
scale-offset filters, and h5py's LZF.

MATLAB takes one more paragraph, since which HDF5 library it links has changed
across releases and decides whether it opens a file at all. It was 1.8.12 before
R2021b, 1.10.7 in R2021b, 1.10.x through R2024a, and 1.14.4.3 since R2024b. A
version 3 superblock is a 1.10 addition, so MATLAB before R2021b cannot open a
file in this crate's older default. The `mat` writer therefore emits the HDF5
1.8 format by default ([`mat::Options::libver`](crate::mat::Options)), which
every one of those releases reads, and
[`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds)
reaches the same format for a plain `.h5` file destined for an old reader.
Around R2021b MathWorks also shipped two libraries at once, keeping 1.8.12 on
the MAT v7.3 path while `h5read` and `h5disp` used 1.10.7, the split behind the
odd symptom of a file `h5disp` prints and `load` rejects. R2021b is not alone in
it: R2023a reports HDF5 1.10.8 and its `load` still rejects a version 3
superblock, so the linked library version does not identify the formats `load`
accepts, which the [`mat` module](crate::mat#the-on-disk-format-matlabs-load-needs)
covers under the on-disk format MATLAB's `load` needs. MATLAB itself writes an
older format still: a version 0 superblock with v1 symbol-table groups, which
this crate reads and does not produce.

Byte-level crosscheck tests check the interoperability. They compare the bytes
this crate emits against fixtures the reference toolchain produced, so a
regression in the on-disk layout fails the test suite. The same discipline backs
the optional [ZFP filter](crate::_guide::compression), where
`src/zfp_crosscheck.rs` compares against `h5py` with `hdf5plugin`, and the
MATLAB `.mat` path. The [`mat` module](crate::mat) has the cross-tool detail.

The 1.8 output format is the one claim those tests cannot make, since every
library they link is 1.10 or newer and reads both formats.
`crates/crosscheck/tests/libver_matrix.rs` covers it against every release the
interop workflow builds, 1.8.23 included: the 1.10 format cannot be opened at
all before 1.10, and the 1.8 format reads completely everywhere. That measures
the format boundary, which is a different thing from a particular MathWorks
build. Only MATLAB itself confirms one, and `examples/octave/check_format.m`
runs that check under it.

The [LZF filter](crate::_guide::compression) is the one place where the byte
comparison runs in one direction only. On read, `src/lzf_crosscheck.rs` decodes
h5py's own compressed streams and compares against the expected bytes. On write,
it compares the filter pipeline this crate emits, its ids, order, name, optional
flag, and `cd_values`, against what h5py recorded for the same dataset, and
leaves the compressed stream out: LZF has many valid encodings of the same data,
so matching liblzf byte for byte is neither a requirement nor a goal. That h5py
decodes the streams this crate produces is verified separately, by the read-back
phase of `tests/data/h5py/lzf/regen.py`, which needs a live h5py and runs when
the fixtures are regenerated.

### Host-independent output

No property of the machine doing the writing, its architecture, pointer width,
or cache-line size, reaches the file. The same input written on `aarch64` and on
`x86_64` produces the same bytes, and the platform that wrote a file does not
change its size. Chunks in particular are stored back to back, with no alignment
padding. A workload that needs cache-line-aligned data aligns its own buffers,
since the file carries no padding to inherit.

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
read out of a file is narrowed through a checked conversion, so a 64-bit value
that does not fit a 32-bit `usize` produces an error where a cast would
truncate. A file too large for the 32-bit address space is therefore read with
[`File::open_streaming`](crate::File::open_streaming), which [Streaming large
files](crate::_guide::streaming) covers. CI exercises the suite on `i686` under
QEMU.

### Memory safety

The crate is almost entirely safe Rust, and the default feature set contains no
non-trivial `unsafe`. Two feature sets introduce some, and a build has one of
them or the other:

- `std` with `serde` compiles the tiled row-major to column-major transpose the
  MATLAB writer uses, which writes through a raw pointer into uninitialized
  `Vec` capacity. CI exercises it under Miri with `-Zmiri-strict-provenance`.
- A `no_std` build compiles a single-threaded `Mutex` replacement whose `Send`
  and `Sync` rest on the target being single-threaded. `no_std` excludes the
  [`mat`](crate::mat) module, so such a build has the replacement and none of
  the transpose.

The [crate documentation](crate#safety-and-robustness) covers the safety and
robustness guarantees in more detail.
