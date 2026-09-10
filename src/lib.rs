//! Pure-Rust HDF5 file reading, writing, and in-place editing library.
//!
//! `hdf5-pure` is a zero-C-dependency crate for creating, reading, and editing
//! HDF5 files. It is WASM-compatible and supports `no_std` environments with
//! `alloc`.
//!
//! # Design goals
//!
//! - **Zero C dependencies.** The HDF5 binary format is implemented directly in
//!   Rust. There is no `libhdf5`, no `build.rs` linking step, and no system
//!   package to install.
//! - **Portable.** The crate compiles to `wasm32-unknown-unknown` and to bare-metal
//!   `no_std` targets (with `alloc`). The high-level filesystem API is `std`-gated.
//!   The in-memory parsing and serialization machinery is not.
//! - **Interoperable.** Files this crate writes are read by the reference HDF5 C
//!   library, h5py, and MATLAB, and vice versa. Interop is verified by crosscheck
//!   tests that compare byte-for-byte against fixtures produced by those tools.
//! - **Faithful or nothing.** Every operation that cannot reproduce data exactly
//!   fails with a named error. See [Fidelity](#fidelity-faithful-or-nothing) below.
//!
//! # Fidelity: faithful or nothing
//!
//! Editing ([`File::open_rw`]) and repacking ([`repack`](fn@repack))
//! each reject what they cannot reproduce exactly. A `File::open_rw` session that cannot reproduce
//! an object faithfully fails with [`Error::EditUnsupported`]. `repack` fails with
//! [`Error::RepackUnsupported`], which identifies the object, before it writes a byte to the
//! destination.
//!
//! # Writing files
//!
//! ```rust
//! use hdf5_pure::{AttrValue, FileBuilder};
//!
//! let mut builder = FileBuilder::new();
//! builder.create_dataset("data")
//!     .with_f64_data(&[1.0, 2.0, 3.0])
//!     .with_shape(&[3])
//!     .set_attr("unit", AttrValue::String("m/s".into()));
//! builder.set_attr("version", AttrValue::I64(2));
//! let bytes = builder.finish()?;
//! # assert!(hdf5_pure::is_hdf5_bytes(&bytes));
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! [`FileBuilder::finish`] serializes the file into a `Vec<u8>`, for a WASM build or a caller
//! that sends the bytes on, and [`FileBuilder::write`] serializes the same file to a path.
//!
//! # Reading files
//!
//! ```rust
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("data.h5");
//! # let mut builder = hdf5_pure::FileBuilder::new();
//! # builder.create_dataset("data")
//! #     .with_f64_data(&[1.0, 2.0, 3.0])
//! #     .set_attr("unit", AttrValue::String("m/s".into()));
//! # builder.set_attr("version", AttrValue::I64(2));
//! # builder.write(&path)?;
//! use hdf5_pure::{AttrValue, File};
//!
//! let file = File::open(&path)?;
//! let dataset = file.dataset("data")?;
//!
//! assert_eq!(dataset.shape()?, vec![3]);
//! assert_eq!(dataset.read_f64()?, vec![1.0, 2.0, 3.0]);
//! assert_eq!(dataset.attrs()?["unit"], AttrValue::String("m/s".into()));
//! assert_eq!(file.root().attrs()?["version"], AttrValue::I64(2));
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! [`File::from_bytes`] reads the same file from a `Vec<u8>` the caller already holds.
//!
//! # Generic over the element type
//!
//! The typed `with_*_data` / `read_*` methods have generic counterparts bounded
//! by [`H5Element`]: [`DatasetBuilder::with_data`] writes a flat slice of any
//! supported scalar and [`Dataset::read`] reads one back, so you can write code
//! generic over the stored type.
//!
//! ```rust
//! use hdf5_pure::{File, FileBuilder, H5Element};
//!
//! fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
//!     fb.create_dataset(name).with_data(values);
//! }
//!
//! let mut fb = FileBuilder::new();
//! store(&mut fb, "counts", &[1u32, 2, 3]);
//! let file = File::from_bytes(fb.finish()?)?;
//! let counts: Vec<u32> = file.dataset("counts")?.read()?;
//! assert_eq!(counts, vec![1, 2, 3]);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! # Editing files in place
//!
//! Open an existing file for reading **and** writing with [`File::open_rw`] and
//! reach every object by name through owned [`Dataset`] and [`Group`] handles
//! that add, delete, copy, or overwrite objects without rewriting the file from
//! scratch. New data and rebuilt object headers are appended at end-of-file
//! and the superblock is repointed last, so the cost is proportional to what
//! changes rather than to the file size. It edits files written by this crate,
//! the reference HDF5 C library, and h5py across all of their on-disk formats.
//! Memory follows the file, not the caller's choice of function: a latest-format
//! file with no userblock is edited without ever building a whole-file copy of it
//! (see [`MemoryStrategy`]).
//!
//! ```rust
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("data.h5");
//! # let mut builder = hdf5_pure::FileBuilder::new();
//! # builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
//! # builder.create_dataset("old").with_i32_data(&[0]);
//! # builder.write(&path)?;
//! use hdf5_pure::File;
//!
//! let file = File::open_rw(&path)?;
//! let root = file.root();
//! root.create_dataset("extra", |b| { b.with_f64_data(&[4.0, 5.0]); })?;
//! file.dataset("data")?.write(&[7.0, 8.0, 9.0])?; // H5Dwrite (overwrite)
//! root.delete("old")?;                            // H5Ldelete
//! file.commit()?;                                 // apply staged edits
//! # drop(root);
//! # drop(file);
//! # let reopened = File::open(&path)?;
//! # assert_eq!(reopened.root().datasets()?, vec!["data", "extra"]);
//! # assert_eq!(reopened.dataset("data")?.read_f64()?, vec![7.0, 8.0, 9.0]);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! A value overwrite ([`Dataset::write`]) must match the on-disk datatype and
//! shape (it is a value write, not a reshape/retype); a same-length contiguous
//! overwrite is applied straight into the existing data block without rewriting
//! any header. Immediate, amortized-`O(1)` row appends use [`Dataset::append`]
//! and need no commit.
//!
//! # Streaming large files
//!
//! [`File::open`] reads the whole file into memory. To read a file too large to
//! buffer (for example a multi-gigabyte file on a 32-bit host, where it exceeds
//! the address space), use [`File::open_streaming`], which fetches metadata and
//! dataset chunks from the file on demand instead of buffering it whole. The
//! reading API is identical; only the backing store differs. Contiguous,
//! compact, and every chunked layout read the same way, as do both group forms
//! and attributes.
//!
//! ```rust
//! # let dir = tempfile::tempdir()?;
//! # let path = dir.path().join("signals.h5");
//! # let mut builder = hdf5_pure::FileBuilder::new();
//! # builder.create_dataset("signal").with_f64_data(&[1.0, 2.0, 3.0]);
//! # builder.write(&path)?;
//! use hdf5_pure::File;
//!
//! let file = File::open_streaming(&path)?;
//! let values = file.dataset("signal")?.read_f64()?;
//! # assert_eq!(values, vec![1.0, 2.0, 3.0]);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! When the bytes are not a path at all — an object store addressed by range
//! request, a sandboxed guest handed byte ranges by its host — implement
//! [`Source`] over them and open with [`File::from_source`], which is the same
//! lazy backing store fed from somewhere else.
//!
//! # N-dimensional arrays (`ndarray` feature)
//!
//! With the `ndarray` feature, datasets can be written from and read back into
//! [`ndarray`] arrays of any rank, in row-major (C) order:
//!
//! ```rust
//! # #[cfg(feature = "ndarray")] {
//! use hdf5_pure::{File, FileBuilder};
//! use ndarray::{array, Array2};
//!
//! let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let mut fb = FileBuilder::new();
//! fb.create_dataset("m").with_ndarray(&a);
//! let bytes = fb.finish()?;
//!
//! let file = File::from_bytes(bytes)?;
//! let back: Array2<f64> = file.dataset("m")?.read_array()?;
//! assert_eq!(a, back);
//! # }
//! # Ok::<(), hdf5_pure::Error>(())
//! ```
//!
//! # Cargo features
//!
//! The crate is split into Cargo features, so a build compiles the parts it uses. The defaults
//! cover filesystem I/O with the high-level reader and writer, and the optional features add
//! MATLAB `.mat` support, a second compression backend, N-dimensional array I/O, and data
//! provenance. The [Installation section of the README][readme-install] shows how to declare them
//! in `Cargo.toml`.
//!
//! | Feature | Default | Pulls in | Implies | Description |
//! |---|---|---|---|---|
//! | `std` | yes | nothing | nothing | file I/O and the high-level reader and writer API |
//! | `checksum` | yes | nothing | nothing | the Jenkins hash that validates checksummed metadata |
//! | `deflate` | yes | `flate2`, Rust backend | nothing | deflate (zlib) compression, pure-Rust backend |
//! | `serde` | no | `serde` | `std` | serialization of MATLAB v7.3 `.mat` files through serde |
//! | `fast-deflate` | no | `flate2/zlib-ng` | nothing | the zlib-ng backend for deflate |
//! | `ndarray` | no | the `ndarray` crate | `std` | N-dimensional array I/O through the [`ndarray`](https://docs.rs/ndarray) crate |
//! | `num-complex` | no | `num-complex` | `serde` | [`mat::ComplexElement`] for `num_complex::Complex<T>`, for the bulk complex-array helpers |
//! | `provenance` | no | `sha2` | nothing | SHA-256 data provenance tracking |
//! | `zfp` | no | nothing | nothing | ZFP fixed-rate compression (HDF5 filter 32013), `f32`, `f64`, `i32` and `i64` in ranks 1 to 4 |
//! | `heap-baseline` | no | nothing | nothing | maintainer only: check the recorded allocation figures (see below) |
//! | `matio-crosscheck` | no | nothing | `serde` | maintainer only: the crosscheck against the system `libmatio` (see below) |
//!
//! The default feature set is `std`, `checksum`, and `deflate`. `serde` and `ndarray` both imply
//! `std`, since they build on the [`File`], [`Group`], and [`Dataset`] reader APIs and the
//! [`FileBuilder`] writer, which require the standard library. Enabling either one enables `std`.
//!
//! ## `std`
//!
//! Enables the standard library, and with it the whole high-level reader and writer surface:
//! [`File`], [`FileBuilder`], [`Group`], [`Dataset`], [`File::open_rw`],
//! [`File::open_swmr_writer`], [`Dataset::append`], [`repack`](fn@repack), the [`mat`] module, and
//! the in-memory, filesystem, and caller-supplied-source entry points ([`FileBuilder::finish`],
//! [`File::from_bytes`], [`File::open`], [`File::open_streaming`], [`File::from_source`],
//! [`FileBuilder::write`]). The whole high-level API is `std`-gated: with `std` disabled the crate
//! is `no_std` and exposes the lower-level datatype and builder primitives alone. `std` is
//! available on `wasm32-unknown-unknown`, so a WASM build keeps it, which [Platform
//! support](#platform-support) covers.
//!
//! ## `checksum`
//!
//! Enables the Jenkins lookup3 hash used to validate and emit the checksums HDF5 puts on its
//! metadata: version 2 and later object headers, the superblock, version 2 B-tree nodes, fractal
//! heaps, and the Extensible-Array and Fixed-Array chunk indexes. It has no extra dependency. Keep
//! it enabled for broad compatibility with the files the reference HDF5 C library and h5py
//! produce. Keep it on a WASM target too, beside `std`.
//!
//! ## `deflate`
//!
//! Enables Deflate (zlib) compression and decompression through a pure-Rust backend (`flate2` with
//! its `rust_backend`). This is what backs [`DatasetBuilder::with_deflate`]. See the
//! [compression guide](crate::_guide::compression) for usage.
//!
//! ## `serde`
//!
//! Adds serde-based (de)serialization of MATLAB v7.3 `.mat` files through the [`mat`] module
//! ([`mat::to_file`], [`mat::to_writer`], [`mat::from_file`], [`Matrix`](crate::mat::Matrix),
//! [`Complex32`](crate::mat::Complex32), [`Complex64`](crate::mat::Complex64)). It pulls in the
//! `serde` dependency and implies `std`.
//!
//! Only the serde-driven entry points are gated. The mid-level [`mat::MatBuilder`], including
//! [`write_blocks`](crate::mat::MatBuilder::write_blocks),
//! [`finish_to`](crate::mat::MatBuilder::finish_to) and [`write`](crate::mat::MatBuilder::write),
//! needs `std` and not `serde`, so a writer that builds its `.mat` explicitly does not pay for the
//! dependency.
//!
//! The `matlab_fixtures` example requires this feature and runs with `cargo run --example
//! matlab_fixtures --features serde`. The `mat_streaming` example needs only the defaults: `cargo
//! run --example mat_streaming`.
//!
//! ## `fast-deflate`
//!
//! Switches the deflate backend to zlib-ng through `flate2/zlib-ng`, which compresses faster than
//! the pure-Rust backend. It complements `deflate` and leaves the deflate API as it is. zlib-ng is
//! a native dependency, so this feature is for native builds, and the pure-Rust `deflate` backend
//! is what a WASM target compiles.
//!
//! ## `ndarray`
//!
//! Adds ergonomic N-dimensional array I/O via the [`ndarray`](https://docs.rs/ndarray) crate:
//! [`DatasetBuilder::with_ndarray`] to write and [`Dataset::read_array`] /
//! [`Dataset::read_array_dyn`] to read. Shape and datatype come from the array, and data is stored
//! row-major (C order). It pulls in the `ndarray` crate and implies `std`. See the
//! [ndarray guide](crate::_guide::ndarray).
//!
//! The `ndarray_io` example requires this feature and runs with `cargo run --example ndarray_io
//! --features ndarray`.
//!
//! ## `num-complex`
//!
//! Implements [`mat::ComplexElement`] for `num_complex::Complex<T>`, so a slice of the de-facto
//! standard Rust complex type can be handed to the bulk array helpers ([`mat::complex::i16_array`]
//! and friends) without a conversion pass. It implies `serde`. See
//! [large complex arrays](crate::mat#large-complex-arrays).
//!
//! `ComplexElement` is `unsafe` and asserts a memory layout, and the orphan rule allows an
//! implementation only from a crate that owns one of the two types, so these implementations ship
//! here. A complex type of the caller's own needs no feature: implement the trait for it directly.
//!
//! ## `provenance`
//!
//! Adds SHA-256 data provenance tracking, pulling in `sha2`.
//! [`DatasetBuilder::with_provenance(creator, timestamp, source)`](DatasetBuilder::with_provenance)
//! stores the SHA-256 digest of the data beside a dataset, together with the creator, the
//! timestamp and, where the caller passes one, the source, and [`Dataset::verify_provenance`]
//! recomputes the digest and returns a [`VerifyResult`]. The attributes have conventional names:
//! `_provenance_sha256`, `_provenance_creator`, `_provenance_timestamp` and `_provenance_source`.
//! `verify_provenance` and [`VerifyResult`] require `std` as well.
//!
//! ```rust
//! # #[cfg(feature = "provenance")] {
//! use hdf5_pure::{File, FileBuilder};
//!
//! let mut builder = FileBuilder::new();
//! builder.create_dataset("measurements")
//!     .with_f64_data(&[1.0, 2.0, 3.0])
//!     .with_provenance("acquisition-rig", "2026-06-16T00:00:00Z", None);
//! let bytes = builder.finish().unwrap();
//!
//! let file = File::from_bytes(bytes).unwrap();
//! let result = file.dataset("measurements").unwrap().verify_provenance().unwrap();
//! # assert_eq!(result, hdf5_pure::VerifyResult::Ok);
//! # }
//! ```
//!
//! ## `zfp`
//!
//! Enables a pure-Rust fixed-rate port of the LLNL/zfp codec, registered as HDF5 filter ID 32013,
//! exposed through [`DatasetBuilder::with_zfp(rate)`](DatasetBuilder::with_zfp). It supports `f32`,
//! `f64`, `i32`, and `i64` datasets in ranks 1D through 4D in fixed-rate mode. Files written with
//! it are byte-for-byte interoperable with the reference H5Z-ZFP plugin (`h5py` + `hdf5plugin`). It
//! has no extra crate dependency. See the [compression guide](crate::_guide::compression).
//!
//! ```rust
//! # #[cfg(feature = "zfp")] {
//! # let (ny, nx) = (16usize, 16usize);
//! # let data: Vec<f32> = (0..ny * nx).map(|i| i as f32).collect();
//! let mut builder = hdf5_pure::FileBuilder::new();
//! builder.create_dataset("temperature")
//!     .with_f32_data(&data)
//!     .with_shape(&[ny as u64, nx as u64])
//!     .with_chunks(&[ny as u64, nx as u64])
//!     .with_zfp(16.0);  // 16 bits per value
//! # let bytes = builder.finish().unwrap();
//! # let file = hdf5_pure::File::from_bytes(bytes).unwrap();
//! # assert_eq!(file.dataset("temperature").unwrap().shape().unwrap(), vec![16, 16]);
//! # }
//! ```
//!
//! ## `heap-baseline`
//!
//! A test-only, maintainer feature. It enables `tests/allocation_baseline.rs`, which checks the
//! crate's exact allocation counts and byte totals for one write-then-read cycle against the
//! figures committed under `tests/baselines/`. It pulls in nothing (the heap profiler it uses,
//! [`heapscope`](https://crates.io/crates/heapscope), is an unconditional dev-dependency), it is
//! not a run-time dependency, and end users do not need it.
//!
//! The figures it checks belong to one target, one toolchain and one feature set, so the test
//! compiles only under the crate's default features and CI runs it in a single pinned job. The
//! bounds that hold everywhere are in `tests/allocation_bounds.rs` and need no feature: a windowed
//! read allocates on the order of its window, and a chunked read costs a constant per chunk. On
//! x86_64, both need the frame pointers `.cargo/config.toml` sets.
//!
//! ## `matio-crosscheck`
//!
//! A test-only, maintainer feature. It enables a crosscheck integration test that links against the
//! system `libmatio` (the reference MATLAB MAT file library, installed with `brew install libmatio`
//! or `apt install libmatio-dev`) to validate `.mat` output. It implies `serde`, is not a run-time
//! dependency, and end users do not need it.
//!
//! The tests that link the reference HDF5 C library are a separate package, `hdf5-pure-crosscheck`
//! under `crates/crosscheck/`, so that nothing else in the repository needs a C library. Its
//! features are not this crate's.
//!
//! # Platform support
//!
//! The crate builds for `wasm32-unknown-unknown` with no C dependencies. `std` is available on
//! that target and the high-level API is `std`-gated, so a WASM build keeps the default features,
//! which include `std`. Turning them off compiles [`File`] and [`FileBuilder`] away. Add the
//! target and build:
//!
//! ```console
//! $ rustup target add wasm32-unknown-unknown
//! $ cargo build --target wasm32-unknown-unknown
//! ```
//!
//! In the browser the in-memory entry points are the ones to use, [`FileBuilder::finish`], which
//! returns a `Vec<u8>`, and [`File::from_bytes`]. The path-based entry points compile, and a
//! browser gives them no filesystem to reach at runtime.
//!
//! For bare-metal `no_std` (for example `thumbv7em-none-eabi`), turn the default features off and
//! keep `checksum`:
//!
//! ```toml
//! [dependencies]
//! hdf5-pure = { version = "0.44", default-features = false, features = ["checksum"] }
//! ```
//!
//! The crate then compiles as `#![no_std]` with `alloc` alone, and the `std`-gated [`File`] and
//! [`FileBuilder`] API is absent: a `no_std` build exposes the lower-level primitives, and
//! building or reading a whole file needs `std`. `fast-deflate` uses the native zlib-ng backend
//! and is for native builds, and the pure-Rust `deflate` backend is what a WASM target compiles.
//! See the [portability guide](crate::_guide::portability) for the full per-target breakdown.
//!
//! # Write paths
//!
//! Three paths put bytes on disk, with different cost models:
//!
//! | Path | Entry point | What it writes |
//! |---|---|---|
//! | **Whole-file writer** | [`FileBuilder`] | Serializes a brand-new file from scratch |
//! | **In-place editor** | [`File::open_rw`] | Appends new bytes to the existing file and patches a small, fixed set of locations |
//! | **Repack** | [`repack`](fn@repack) | Reads a source file and writes a fresh, compact copy to a separate destination through `FileBuilder` |
//!
//! `File::open_rw` is an **append-and-patch** editor. It reads and patches a latest-format file
//! with no userblock where it lies, through positioned I/O, and mirrors any other file whole in
//! memory (`O(file size)`), which is a statement about *memory*, not about what gets written
//! back. An immediate `Dataset::append` writes the new chunks, fills Extensible-Array index
//! slots, and publishes the grown dataspace dimension last under `fsync` barriers, so each append
//! is durable and crash-atomic before the call returns (`SyncPolicy::OnClose` keeps the order and
//! drops the barriers, leaving durability to `File::sync`). Every other edit is staged and applied
//! by one `commit()`: new data and new object headers are appended at the end of the file (or
//! placed into space freed by earlier commits), a rewritten header for each touched group and its
//! ancestors up to the root is appended, and the superblock is repointed at the new root **last**,
//! as the single crash-atomic commit point. A same-length value overwrite patches the existing
//! bytes where they lie. Space freed by deletions and relocations goes to a session free list for
//! reuse. A freed run reaching the end of the file is truncated, and on a file that persists its
//! free space the free list is instead serialized into on-disk free-space managers that survive
//! reopen.
//!
//! **Nothing re-serializes the file on commit.** The cost of a commit is proportional to the edit,
//! not to the file size: a staged append re-encodes at most the trailing partial chunk and carries
//! every other kept chunk into the rebuilt index by metadata alone, and no existing object moves
//! except the object headers on the edited path. A failed or interrupted commit leaves the file
//! valid: structural changes become visible only at the superblock repoint, so a crash before it
//! leaves the old object tree in place. (Same-length value overwrites patch live data blocks
//! directly and are the one staged edit outside that gate.)
//!
//! The bounded backing is the same engine minus the mirror: reads are positioned I/O through
//! bounded caches (the read-write sibling of [`File::open_streaming`]),
//! `Dataset::append` reads and patches only the metadata windows it touches with the same
//! crash-atomicity (and under the same `SyncPolicy`), and `close()` rewrites the on-disk free-space
//! managers of a file that persists them - including the per-page-type managers of a paged file,
//! whose appends stay page-homogeneous. The staged edit surface works here too: it builds what the
//! edit needs, not a copy of the file. What separates the two backings is which files they accept -
//! the bounded one requires a latest-format file with 8-byte offsets and no userblock, and
//! `open_rw` mirrors any other file.
//!
//! The SWMR writer is a restriction of the same immediate append engine - unfiltered, chunk-aligned
//! appends only - chosen so a concurrent reader never observes a torn view.
//!
//! `repack` is the one operation that rewrites a file from scratch, and it never does so in place:
//! it reads every surviving object and writes a fresh, compact copy at a separate destination path,
//! rejecting (`Error::RepackUnsupported`) anything it cannot reproduce faithfully.
//!
//! # On-disk format coverage
//!
//! The reader and editor handle the formats the reference C library and h5py
//! produce in the wild:
//!
//! - **Superblocks** version 0, 1, 2, and 3.
//! - **Object headers** version 1 (with continuation blocks) and version 2,
//!   including multi-chunk headers.
//! - **Storage layouts** - contiguous, compact, and chunked.
//! - **Chunk indexes** - B-tree v1, single chunk, implicit, fixed array (including
//!   the paged data-block layout), and extensible array (which also backs SWMR
//!   append and the in-place append paths: `Dataset::append_staged` and the
//!   owned-handle `File::open_rw` + `Dataset::append`).
//! - **Groups** - both the old symbol-table form (v0/v1) and the modern
//!   compact-link and dense (fractal-heap + v2 B-tree) forms.
//! - **Datatypes** - fixed-point, floating-point, string (fixed and
//!   variable-length), bit-field, opaque, compound, enumeration, array, and
//!   reference classes.
//!
//! The crate writes one modern format by default (the HDF5 1.10 version-3 superblock with
//! latest-format object headers), and [`FileBuilder::with_libver_bounds`] takes it back to the
//! 1.8 format at the oldest ([`LibVer::WRITER_OLDEST`]), so its output stays compact and
//! consistent while its reader remains broad.
//!
//! # Safety and robustness
//!
//! - **Almost entirely safe Rust.** The default feature set contains no non-trivial
//!   `unsafe`. Adding `serde` compiles the tiled row-major/column-major transpose
//!   used by the MATLAB writer, which is exercised under
//!   [Miri](https://github.com/rust-lang/miri) with `-Zmiri-strict-provenance` in CI.
//!   A `no_std` build has neither, and compiles a single-threaded `Mutex`
//!   replacement instead, whose `Send`/`Sync` rest on the target being
//!   single-threaded.
//! - **32-bit safe.** Every file-derived offset and length is narrowed through
//!   checked conversions, so a 64-bit value that does not fit a 32-bit `usize`
//!   errors where a cast would truncate. CI runs the suite on `i686` under QEMU and
//!   builds for `thumbv7em-none-eabi`.
//! - **Property-tested.** The write/read roundtrip and parser robustness are
//!   covered by property-based tests in addition to the example- and
//!   fixture-driven suites.
//!
//! # Origins and licenses
//!
//! The crate is licensed under MIT or Apache-2.0, at your option, and parts of it come from other
//! projects. The scale-offset filter is a port of the HDF5 library's `H5Zscaleoffset.c` and the
//! ZFP codec a port of LLNL's reference implementation, both BSD 3-Clause, and the HDF5 format
//! parsing and low-level I/O modules come from rustyhdf5 by the RustyStack project, MIT licensed.
//! The README's License section links every license text.
//!
//! [readme-install]: https://github.com/CramBL/hdf5-pure#installation

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]
// In `no_std` builds the high-level entry points that consume the parsing and
// serialization machinery — the `reader`, `writer`, and `edit` modules —
// are `std`-gated and therefore absent, leaving much of that machinery without
// an in-crate consumer. It is deliberately kept available for future `no_std`
// readers/writers rather than cfg-gating every module to `std`, so `dead_code`
// is allowed only in `no_std` builds. `std` builds (the primary target) keep
// full `dead_code` enforcement.
#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(not(feature = "std"))]
extern crate alloc;

// ---------------------------------------------------------------------------
// The guide.
// ---------------------------------------------------------------------------

#[cfg(doc)]
pub mod _guide;

// ---------------------------------------------------------------------------
// Internal modules (encapsulated as `pub(crate)`; the curated public surface is
// re-exported at the bottom of this file).
// ---------------------------------------------------------------------------

pub(crate) mod address;
pub(crate) mod attribute;
pub(crate) mod attribute_info;
pub(crate) mod btree_v1;
pub(crate) mod btree_v2;
pub(crate) mod btree_v2_write;
pub(crate) mod bytes;
pub(crate) mod checksum;
pub(crate) mod chunk_cache;
pub(crate) mod chunk_grid;
pub(crate) mod chunk_span;
pub(crate) mod chunked_read;
pub(crate) mod chunked_write;
pub(crate) mod compound;
pub(crate) mod convert;
pub(crate) mod data_layout;
pub(crate) mod data_read;
pub(crate) mod dataspace;
pub(crate) mod datatype;
pub(crate) mod display;
pub(crate) mod error;
pub(crate) mod extensible_array;
pub(crate) mod file_create_properties;
pub(crate) mod file_space_info;
pub(crate) mod file_writer;
pub(crate) mod fill_value;
pub(crate) mod filter_pipeline;
pub(crate) mod filters;
pub(crate) mod fixed_array;
pub(crate) mod fractal_heap;
pub(crate) mod fractal_heap_write;
pub(crate) mod free_space_manager;
pub(crate) mod global_heap;
pub(crate) mod group_v1;
pub(crate) mod group_v2;
pub(crate) mod layout_info;
pub(crate) mod libver;
pub(crate) mod link_info;
pub(crate) mod link_message;
pub(crate) mod local_heap;
pub(crate) mod lzf;
pub(crate) mod message_type;
pub(crate) mod object_header;
pub(crate) mod object_header_writer;
pub(crate) mod read_spec;
pub(crate) mod scaleoffset;
pub(crate) mod shared_message;
pub(crate) mod signature;
pub(crate) mod sohm;
pub(crate) mod source;
pub(crate) mod superblock;
pub(crate) mod symbol_table;
pub(crate) mod type_builders;
pub(crate) mod vl_data;
pub(crate) mod width;
#[cfg(feature = "zfp")]
pub(crate) mod zfp;

#[cfg(feature = "provenance")]
pub(crate) mod provenance;

#[cfg(not(feature = "std"))]
pub(crate) mod nosync;

// ---------------------------------------------------------------------------
// High-level modules
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub(crate) mod appender;
#[cfg(feature = "std")]
pub(crate) mod chunk_index_inplace;
#[cfg(all(test, feature = "std"))]
mod crash_replay;
#[cfg(feature = "std")]
pub(crate) mod edit;
#[cfg(feature = "std")]
pub(crate) mod file_lock;
#[cfg(feature = "std")]
pub(crate) mod free_space;
#[cfg(feature = "std")]
pub(crate) mod image;
#[cfg(feature = "std")]
pub(crate) mod reader;
#[cfg(feature = "std")]
pub(crate) mod reference_patch;
#[cfg(feature = "std")]
pub(crate) mod repack;
#[cfg(test)]
mod test_data;
#[cfg(feature = "std")]
pub(crate) mod types;
#[cfg(feature = "std")]
pub(crate) mod writer;

#[cfg(feature = "std")]
pub mod mat;

#[cfg(feature = "std")]
pub(crate) mod element;

#[cfg(feature = "ndarray")]
pub(crate) mod ndarray_support;

// ---------------------------------------------------------------------------
// Public API re-exports
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
pub use error::Error;
pub use error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};

// Reached by value through `File::superblock` and `Error::MissingMessage`, so
// exported alongside them rather than left nameless — see RELEASES.md, and
// `scripts/check-api-surface.sh` (`just api::api-surface`), which enforces it.
#[cfg(feature = "std")]
pub use address::BaseAddress;
#[cfg(feature = "std")]
pub use message_type::MessageType;
#[cfg(feature = "std")]
pub use superblock::Superblock;

#[cfg(feature = "std")]
pub use reader::{
    Dataset, DatasetAccessProperties, File, FileAccessProperties, Group, Object, StagedGroup,
    is_hdf5, is_hdf5_bytes,
};

// Curated layout / filter introspection (issue #149). Only the std-only reader
// `Dataset` produces these, so gate the re-export to match the reader block
// above; the types themselves are `alloc`-clean.
#[cfg(feature = "std")]
pub use layout_info::{Chunk, ChunkIndex, Filter, Layout};

#[cfg(feature = "std")]
pub use file_lock::{FileLocking, WriteMarkPolicy};

pub use chunk_cache::{ChunkCacheConfig, ChunkCacheStats};

pub use source::{MetadataCacheConfig, MetadataCacheStats, Source};

// The `Read + Seek` backend, for a caller who has one but no path. Exported
// beside `Source` because without it every such caller reimplements the
// seek-and-read the crate already carries.
#[cfg(feature = "std")]
pub use source::ReadSeekSource;

#[cfg(feature = "std")]
pub use vl_data::VlenStringReadOptions;

pub use libver::LibVer;

#[cfg(all(feature = "std", feature = "provenance"))]
pub use provenance::VerifyResult;

#[cfg(feature = "std")]
pub use types::{AttrValue, DType};

#[cfg(feature = "std")]
pub use writer::FileBuilder;

#[cfg(feature = "std")]
pub use appender::BufferedAppender;

#[cfg(feature = "std")]
pub use edit::{AppendBuilder, EditBacking, MemoryStrategy, SpaceAccounting, SyncPolicy};

#[cfg(feature = "std")]
pub use repack::{RepackOptions, repack};

#[cfg(feature = "std")]
pub use element::H5Element;

pub use scaleoffset::ScaleOffset;

pub use file_create_properties::FileCreateProperties;
pub use file_space_info::{FileSpaceInfo, FileSpaceStrategy};

// The HDF5 datatype handle returned by the `make_*_type` constructors and
// accepted by the compound/enum builders and `DatasetBuilder::with_dtype`.
pub use compound::{CompoundField, CompoundType};
pub use datatype::{
    CharacterSet, CompoundMember, Datatype, DatatypeByteOrder, EnumMember, ReferenceType,
    StringPadding,
};

pub use type_builders::{
    CompoundTypeBuilder, DatasetBuilder, EnumTypeBuilder, ExplicitCompoundTypeBuilder,
    FinishedGroup, GroupBuilder, make_f32_type, make_f64_type, make_i8_type, make_i16_type,
    make_i32_type, make_i64_type, make_u8_type, make_u16_type, make_u32_type, make_u64_type,
};
