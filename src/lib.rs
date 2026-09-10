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
//! # Limitations
//!
//! The crate reads and writes a broad, interoperable subset of the HDF5 format. Where it cannot
//! handle something, it returns a typed error, so every gap below is a deliberate, well-messaged
//! rejection and never a silent misread.
//!
//! The rejections fall into two kinds:
//!
//! - [Deliberately unsupported](#deliberately-unsupported): by-design constraints, and guards
//!   against file formats outside the range the crate models. These are not planned to change.
//! - [Planned support](#planned-support): what the crate rejects for now, each entry tracked by a
//!   GitHub issue. These error messages read `... not supported yet` or `... cannot be ... yet`.
//!
//! Malformed-file errors (a truncated or garbled header, an address that exceeds the platform's
//! pointer width) and API-contract errors (deleting or copying the root group, conflicting edits
//! in one commit) are ordinary runtime errors and not capability limits, so this catalog leaves
//! them out.
//!
//! ## Deliberately unsupported
//!
//! ### Reading non-modeled formats
//!
//! | Rejected | Error | Reason |
//! |---|---|---|
//! | a superblock version above 3 | [`FormatError::UnsupportedVersion`] | versions 0 to 3 are read, and the released format defines no higher one |
//! | an unrecognized object-header message flagged must-understand | [`FormatError::UnsupportedMessage`] | the format requires a reader that does not recognize a must-understand message to reject the file |
//! | a File Space Info message version other than 1 | [`FormatError::UnsupportedFileSpaceInfoVersion`] | version 1 is the only one defined for the layouts this crate writes and reads |
//!
//! These guard against files outside the format-version range the crate models, and they are not
//! features to add.
//!
//! ### Numeric element width
//!
//! | Rejected | Error | Reason |
//! |---|---|---|
//! | an integer or float element wider than 8 bytes | [`FormatError::NumericElementTooWide`] | the typed numeric readers model an element as a 64-bit word taken from its leading eight bytes, and which part of a wider element that is depends on its byte order ([#361](https://github.com/CramBL/hdf5-pure/issues/361)) |
//!
//! The rejection keeps a partial value from passing for a whole one: a 9-byte integer holding 2^64
//! would otherwise read back as `0`, and big-endian it would read back as 2^56, since the bytes
//! kept are the most significant ones. It covers datasets and attributes alike, and an attribute
//! it rejects is omitted from [`attrs`](Dataset::attrs) while
//! [`attr_datatypes`](Dataset::attr_datatypes) still reports its width.
//! [`read_raw`](Dataset::read_raw) and [`read_u8`](Dataset::read_u8) /
//! [`read_i8`](Dataset::read_i8) hand back the bytes either way. Decoding these widths is a
//! separate feature: it means 128-bit variants throughout the [`AttrValue`] and `read_*` surfaces.
//!
//! ### Compression
//!
//! | Rejected | Error | Reason |
//! |---|---|---|
//! | a filter whose backend is not compiled in | [`FormatError::UnsupportedFilter`] | enable the `deflate` or `zfp` Cargo feature, which [Cargo features](#cargo-features) covers |
//! | ZFP outside fixed-rate mode, ranks 1 to 4, and the datatypes `f32`, `f64`, `i32` and `i64` | [`FormatError::UnsupportedZfp`] | the supported scope of the bundled ZFP codec, which the [compression guide](crate::_guide::compression) covers |
//!
//! ### Producer-backed datasets
//!
//! A dataset staged with [`MatBuilder::write_blocks`](crate::mat::MatBuilder::write_blocks) is
//! always stored uncompressed, and requesting one on a builder configured for deflate is rejected
//! with
//! [`MatError::CompressionUnsupportedForBlocks`](crate::mat::MatError::CompressionUnsupportedForBlocks).
//! The writer places every object before it emits a byte, so it needs the data region's exact size
//! up front: unfiltered that is pure geometry, and compressed it is knowable only by compressing,
//! which buffers the data this path exists to avoid buffering. Supporting it takes either a
//! two-pass producer contract (compress to measure, then compress again to emit, which requires a
//! deterministic producer) or a spill file, so it is a separate design.
//!
//! A producer that fails partway leaves a partial file on the sink, which is inherent to a
//! non-seekable destination. Write to a temporary path and rename on success where the result has
//! to be all-or-nothing.
//!
//! ### External data files
//!
//! A dataset created with `H5Pset_external` keeps its elements in one or more files beside the
//! HDF5 file. Its layout message records a contiguous layout with an undefined data address,
//! byte-for-byte what a never-written dataset records, and a separate header message lists the
//! files. Reading one is rejected with [`FormatError::UnsupportedExternalStorage`], since the
//! alternative is returning the fill value for data that exists elsewhere, and
//! [`repack`](fn@repack) rejects it, since the alternative is a copy holding none of it
//! ([#331](https://github.com/CramBL/hdf5-pure/issues/331),
//! [#293](https://github.com/CramBL/hdf5-pure/issues/293)).
//!
//! Writing is rejected for the same reason, with [`Error::EditUnsupported`]: [`Dataset::write`],
//! [`write_staged`](Dataset::write_staged), and [`append_staged`](Dataset::append_staged) would
//! otherwise take the address-less layout for never-allocated storage, append the new elements to
//! the HDF5 file and point the layout at them, leaving the file disagreeing with the external
//! files about where its data lives. Delete the dataset and create it again in the same commit to
//! replace it.
//!
//! Copying is rejected too, by name: [`File::copy`] and [`File::copy_from`] reproduce a dataset
//! whose storage was never allocated as the empty storage it is, and an externally stored dataset
//! carries that same address-less layout over data that exists, so a copy that treated the two
//! alike would report success having written the schema and none of the elements
//! ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).
//!
//! What still works is reading the metadata, the shape, the datatype, and the address-less layout,
//! which is the evidence a caller needs, together with setting and removing attributes. Following
//! the external files is a separate feature: it means resolving each file the message lists
//! against the reading process's own filesystem, which is outside what a self-contained HDF5 file
//! describes.
//!
//! ### Repack faithfulness
//!
//! [`repack`](fn@repack) rewrites a file and rejects lossy filter re-encoding (lossy float
//! scale-offset, ZFP), since re-compressing lossy data would change the values. Lossless integer
//! scale-offset is the one pipeline it re-encodes faithfully, and the dataset's fill value comes
//! with it, in whichever of the two forms the source's filter recorded. Where it can, repack
//! copies already-compressed chunks verbatim, which preserves a lossy filter byte-exact.
//!
//! ### SWMR
//!
//! SWMR append requires a latest-format file (a v3 superblock) with no userblock, which mirrors
//! the HDF5 SWMR model, defined for the latest format alone. The C library rejects an older
//! superblock for SWMR writing too, and neither library reads the SWMR-write flag back on one.
//!
//! ### In-place editing
//!
//! In-place editing operates on files with 8-byte offsets and lengths, which is what the writer
//! emits and what modern files use. Other offset and length widths are not editable in place.
//!
//! ### Bounded-memory read-write
//!
//! A file [`File::open_rw`] edits [bounded](crate::_guide::editing#bounded-memory-appends) offers
//! the same edit surface as one it mirrors: reads, immediate [`Dataset::append`], and the staged
//! surface ([`write`](Dataset::write), attribute edits, `create_*` and `delete`,
//! [`copy`](File::copy), [`commit`](File::commit),
//! [`space_accounting`](File::space_accounting)), with a commit whose resident memory the edit
//! bounds and the file size does not ([`File::copy`] excepted: copying an object reads the whole
//! of it into memory). The bounded backing needs a latest-format file with 8-byte offsets and no
//! userblock, and any other file is mirrored. Its reads have the [streaming backend's
//! capabilities](crate::_guide::streaming). It grows a file that persists its free space,
//! including a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`). A paged file that does not persist
//! its free space is rejected with [`Error::EditUnsupported`] under either backing, and recreating
//! it with `persist = true` makes it editable.
//!
//! Adding an **object-reference dataset** ([`Group::create_dataset`] with
//! [`with_path_references`](DatasetBuilder::with_path_references)) resolves a path target against
//! every object this commit places, once that object has been placed and no sooner.
//! [`commit`](File::commit) processes groups deepest-first and, within a group, non-reference
//! datasets before reference ones. A target that is itself still being written when the reference
//! is resolved (an ancestor group, a same-depth sibling group ordered later, a copy destination or
//! its interior, or a [`Dataset::write`] target) is rejected, since resolving it would give a
//! stale or wrong address. A target the same commit deletes, the deleted path or anything under
//! it, is rejected too, since a deletion reclaims the whole subtree and the reference would point
//! into space the file is about to hand out again. A target supplied as an address, through
//! [`with_reference_data`](DatasetBuilder::with_reference_data) or
//! [`with_raw_data`](DatasetBuilder::with_raw_data) over a datatype that holds a reference, is
//! screened by address and not by name, wherever the commit writes one: a new dataset, a
//! [`Dataset::write_staged`] overwrite, and an in-file [`copy`](File::copy), which re-emits its
//! source's references verbatim ([#317](https://github.com/CramBL/hdf5-pure/issues/317)). A
//! supplied address is screened against both halves of what a commit can vacate, the objects it
//! deletes and the headers it rewrites elsewhere (a group it dirties, a dataset whose write
//! relocates it). The second rejection points at the path form, since the object still exists,
//! though only a dirty group is resolvable that way, once this commit has placed it. A relocating
//! dataset write is rejected by name outright, which is why that message points at separate
//! commits too. A copied address is screened against deletions alone: a target that merely moves
//! needs no rejection on either side, because the commit repoints every reachable stored reference
//! once it has published its tree ([#324](https://github.com/CramBL/hdf5-pure/issues/324)), the
//! copy's included.
//!
//! In a commit that deletes, three things are rejected outright, without that screen:
//!
//! - copying a chunked object-reference dataset, whose addresses sit inside chunks the copy path
//!   carries compressed and never decodes, the same limit that makes [`repack`](fn@repack) reject
//!   one outright
//! - copying an object carrying a shared (SOHM) attribute message, whose bytes this path cannot
//!   reach
//! - copying an object whose datatype or attribute this parser cannot read, since an unreadable
//!   datatype cannot be shown to be free of references
//!
//! A committed (`H5Tcommit`) datatype is not one of these: it is resolved, so an object with a
//! named type still copies. Separately, a datatype whose references this screen cannot read out of
//! the element bytes is rejected, since writing it would leave it unscreened: one wider than 8
//! bytes, a dataset-region reference, a variable length of references (whose addresses live in the
//! global heap the elements point at), or a compound holding one of those beside a reference the
//! screen can read. A staged dataset is rejected in any commit that rebuilds a header, which is
//! every commit except one whose only staged edit is a same-length in-place overwrite, and that
//! one vacates nothing to dangle into. A copied one is rejected only beside a deletion, as the
//! three above are. No API here builds such a type:
//! [`with_raw_data`](DatasetBuilder::with_raw_data) is one way in, and an in-file
//! [`copy`](File::copy) of a file the reference C library wrote is the other. A path the commit
//! deletes and then [replaces](crate::_guide::editing#replacing-an-object) resolves to the
//! replacement once the replacement has been placed, so the placement-order rule above still
//! governs it. A target untouched by the commit resolves against the pre-commit file, and a path
//! that resolves nowhere at all becomes an undefined reference, matching [`FileBuilder`]'s
//! resolution convention for the same builder type. This is a permanent scope line, and not a
//! `... yet` gap: reproducing the whole-file writer's two-pass dummy-then-real-address scheme
//! inside the editor's single-pass commit would be a large rewrite of the core apply loop for a
//! narrow benefit.
//!
//! ### Object header message size
//!
//! A version 2 object header describes each message's length in a 2-byte field, so no message it
//! carries may exceed 65,535 bytes. The whole-file writer rejects an oversized message and never
//! truncates one:
//!
//! - A compact attribute past the limit would be rejected with
//!   [`FormatError::AttributeMessageTooLarge`], which reports the attribute. The limit is on the
//!   message, its name, datatype, dataspace, and data together, and not on the element count.
//! - Every other oversized message, most often a Link message from a very long dataset or group
//!   name, is rejected with [`FormatError::ObjectHeaderMessageTooLarge`], which carries the message
//!   type.
//!
//! An attribute that would exceed the limit selects fractal-heap storage, where no such field
//! bounds it. See [dense attribute storage](#dense-attribute-storage) below.
//! `AttributeMessageTooLarge` is therefore a backstop no input reaches today, since the writer
//! sends exactly those attributes to a heap, kept because the limit it describes is a real
//! property of the object header. `ObjectHeaderMessageTooLarge`, which covers the messages with no
//! heap alternative, is the one reachable on size.
//!
//! The [in-place editor](crate::_guide::editing) meets the limit the same way: an attribute
//! whose message would exceed it is written to a heap, and the messages with no heap alternative
//! are rejected with [`Error::EditUnsupported`]. What no storage lifts is the 2-byte field inside
//! the attribute message that describes its name, datatype or dataspace, which is
//! [`FormatError::AttributeFieldTooLong`] on either path.
//!
//! ### Dense attribute storage
//!
//! An object stores its attributes in a fractal heap when it has more than eight of them, or when
//! any one of them is too large for an object-header message. That is the same disjunction the
//! reference C library uses, and it is why a single large attribute is written and not rejected.
//!
//! The writer emits the same heap geometry the reference C library uses for an attribute heap: a
//! doubling table of direct blocks from 1 KiB up to 64 KiB, reached through a root indirect block
//! once one block no longer holds everything, and indexed by B-trees of fixed 512-byte nodes that
//! grow internal levels as the record count rises. An attribute serializing past 65,514 bytes does
//! not fit a heap managed object, so it is written as a huge object: its bytes go outside the
//! managed blocks and a huge-objects B-tree maps a generated ID to them. An attribute's size has
//! no limit, and the number of attributes an object may carry has none either. What is still
//! rejected:
//!
//! - An attribute whose name, datatype, or dataspace serializes past 65,535 bytes is rejected with
//!   [`FormatError::AttributeFieldTooLong`], which reports the attribute and the field. Each has a
//!   2-byte length field in the attribute message, and huge storage lifts the limit on an
//!   attribute's data alone.
//! - About a terabyte of managed attributes on one object is rejected with
//!   [`FormatError::DenseAttributeHeapTooLarge`]: the heap's offsets are 40 bits wide, so its
//!   blocks cannot span more than that between them.
//!
//! The [in-place editor](crate::_guide::editing) emits the same storage through the same builder:
//! an attribute edit that outgrows the object header moves the whole set to a heap, an object
//! already using one is rebuilt around the edit, and a dataset or group created in place may carry
//! one ([#102](https://github.com/CramBL/hdf5-pure/issues/102)). Three things bound it there: an
//! attribute holding an object reference a commit could still repoint is left in the header, a
//! shared (SOHM) attribute message is not rewritten, and the heap's own bounds above apply as they
//! do to the whole-file writer. The heap a rebuild supersedes is left as dead bytes for
//! [`repack`](fn@repack), as every dense heap this editor replaces is.
//!
//! Eight is fixed, on both paths. A version 2 object header may store its own
//! `H5Pset_attr_phase_change` thresholds, and the in-place editor preserves that block across a
//! rewrite, so the reference C library still reports the pair a caller set. The switch to a heap
//! is still made at eight attributes whatever the stored pair says
//! ([#422](https://github.com/CramBL/hdf5-pure/pull/422)).
//!
//! The total is otherwise not limited, and the table adds blocks in place of rounding the whole
//! heap up to a power of two. Space is still lost per block. An attribute that does not fit what
//! remains of a block moves to the next one, and an attribute just over half of the largest
//! block's size leaves most of a block unused, so a set of such attributes can still occupy close
//! to twice its own size. The writer only ever appends to the block it filled last, and an earlier
//! block's remainder stays unused, where the reference C library's free-space manager goes back
//! for it.
//!
//! ### Group creation property list
//!
//! There is no property-list API for group creation, and none of its settings are configurable.
//! Every group the crate writes, the root group included, has one fixed shape: a new-style (v2
//! object header) group with compact link storage and no stored timestamps. An object header the
//! [in-place editor](crate::_guide::editing) rewrites keeps whatever times it already stored, with
//! the modification and change times moved to the edit, so a header created from nothing is the
//! one that gets this shape. It is equivalent to creating every group with `obj_track_times =
//! false` and never switching to old-style (symbol-table) or dense (fractal-heap) link storage,
//! whatever the file version or the child count. The reference library's GCPL defaults vary by
//! version, and this shape is fixed on purpose: it keeps output byte-for-byte reproducible, which
//! is what makes the crate a good fit for stable snapshot files. See
//! [#131](https://github.com/CramBL/hdf5-pure/issues/131).
//!
//! ## Planned support
//!
//! Rejected today with a `... yet` message, and intended to land. Each row links to its tracking
//! issue.
//!
//! ### In-place editing
//!
//! | Capability | Tracking |
//! |---|---|
//! | Dense (fractal-heap) link storage, a group with more links than it keeps compactly | [#102](https://github.com/CramBL/hdf5-pure/issues/102) |
//! | Editing across soft and external links | [#103](https://github.com/CramBL/hdf5-pure/issues/103) |
//! | Copying a version-1 object (attribute creation-order tracking is edited as of [#416](https://github.com/CramBL/hdf5-pure/issues/416)) | [#104](https://github.com/CramBL/hdf5-pure/issues/104) |
//! | Adding a link to a group that tracks link creation order, as netCDF-4 writes, once the addition would take the group past the compact-storage threshold its group-info message declares (8 links by default). Past it the links move to a fractal heap with a creation-order B-tree beside the name index, which is dense link storage, the row above, tracked by [#102](https://github.com/CramBL/hdf5-pure/issues/102). Below it such a group is edited as any other is, as of [#416](https://github.com/CramBL/hdf5-pure/issues/416): an added link takes the next creation index, a removed one leaves a gap without lowering the group's counter, and a copy of the group keeps the order it had | [#102](https://github.com/CramBL/hdf5-pure/issues/102) |
//! | Changing a shared (SOHM) message's reference count. A file with a shared-message table is read and edited, and a commit rejects what it would strand: an object whose attributes are a shared message is not rewritten, and an object header the file's shared-message index refers to is neither moved nor removed. A commit that deletes an object using a shared message leaves the count one too high, so the message stays readable and the entry is never reclaimed. [`repack`](fn@repack) rewrites the file with every message inline | [#417](https://github.com/CramBL/hdf5-pure/issues/417) |
//! | Adding chunked or extensible variable-length-string datasets | [#105](https://github.com/CramBL/hdf5-pure/issues/105) |
//! | Overwriting ([`Dataset::write_staged`]) with [`with_path_references`](DatasetBuilder::with_path_references), whose staged element bytes are placeholder addresses that only the dataset-creation path resolves. The rejection is on that builder and not on the datatype: a reference dataset still overwrites through [`with_reference_data`](DatasetBuilder::with_reference_data), which supplies resolved addresses, or [`with_raw_data`](DatasetBuilder::with_raw_data). [`with_vlen_strings`](DatasetBuilder::with_vlen_strings) overwrites as of [#321](https://github.com/CramBL/hdf5-pure/issues/321). The global heap collections it can prove are its own are reclaimed, and [`repack`](fn@repack) recovers the rest, which the [editing guide](crate::_guide::editing#staging-and-committing-edits) covers | [#321](https://github.com/CramBL/hdf5-pure/issues/321) |
//! | Cross-file copy of variable-length, reference, or shared data, including an attribute that refers to a committed datatype | [#106](https://github.com/CramBL/hdf5-pure/issues/106) |
//! | Keeping an object reference already stored in the file valid when its target's header is rewritten, where the address cannot be reached. By storage: inside a chunked dataset's chunks, in a dense (fractal-heap) attribute or one held as a shared (SOHM) record, and in an attribute of a version 1 object header (such a header's element data is reached, through the parser that reads both versions, and version 1 is what the reference C library and h5py write by default). By datatype: an object reference wider than 8 bytes, a dataset-region reference, a variable length of references (the encoding the dimension-scale attribute `DIMENSION_LIST` uses), an enumeration over one, and any compound or array reaching one of those. What a commit does repoint as its last act is the rest: a contiguous or compact dataset's elements and an attribute held in the header, holding an 8-byte object reference at any depth of compound or array nesting, including through a committed (`H5Tcommit`) datatype. An unreached reference is left as it was. Nothing is rejected for being unreachable, since every commit rebuilds its root group and would therefore reject every commit on such a file | [#324](https://github.com/CramBL/hdf5-pure/issues/324) |
//!
//! ### Repack
//!
//! | Capability | Tracking |
//! |---|---|
//! | Repack of region references, non-8-byte object references, chunked, filtered or resizable reference datasets, and unrecognized filter pipelines (time, variable-length sequences, and 8-byte object references repack faithfully, and chunked, filtered, and resizable variable-length datasets repack as of [#109](https://github.com/CramBL/hdf5-pure/issues/109)) | [#107](https://github.com/CramBL/hdf5-pure/issues/107) |
//! | Dropping a committed (`H5Tcommit`) datatype a surviving dataset or attribute still refers to, and a committed datatype no hard link reaches, both rejected because the copy would point at nothing (a committed datatype otherwise repacks: the object is recreated and its users still refer to it) | [#254](https://github.com/CramBL/hdf5-pure/issues/254) |
//!
//! ### Reading
//!
//! | Capability | Tracking |
//! |---|---|
//! | Filter-encoded fractal-heap objects | [#108](https://github.com/CramBL/hdf5-pure/issues/108) |
//! | Virtual (VDS) datasets | [#111](https://github.com/CramBL/hdf5-pure/issues/111) |
//! | Writing shared (SOHM) messages: this crate's writer stores every message privately, and [`repack`](fn@repack) rewrites a shared-message file that way. Reading one is supported as of [#417](https://github.com/CramBL/hdf5-pure/issues/417), covering the shared-message table, both index kinds, and the heap lookup, for the datatype, dataspace, fill value, filter pipeline and attribute messages `H5Pset_shared_mesg_index` can share | [#417](https://github.com/CramBL/hdf5-pure/issues/417) |
//!
//! [`repack`](fn@repack) rejects a virtual dataset as well, since it cannot relocate data living
//! outside the file. That lifts together with VDS read support
//! ([#111](https://github.com/CramBL/hdf5-pure/issues/111)).
//!
//! ### Writing
//!
//! | Capability | Tracking |
//! |---|---|
//! | Append to a variable-length dataset (writing one resizable is supported, and growing it is not) | [#109](https://github.com/CramBL/hdf5-pure/issues/109) |
//! | More than one unlimited dimension in a `maxshape`. The reference library indexes that dataspace with a version-2 B-tree, which this crate neither writes nor reads. One unlimited dimension is supported at any rank | [#299](https://github.com/CramBL/hdf5-pure/issues/299) |
//! | A `maxshape` whose chunk index would spend more than 32 MiB on elements describing no chunk. An Extensible Array allocates only the blocks its chunks land in, as the reference library does, so this applies where the slack inside those blocks is itself large. A Fixed Array is dense by format and costs the same here as in the reference library, so for a fixed `maxshape` this is a bound on what this writer holds in memory while building the index in one pass | [#299](https://github.com/CramBL/hdf5-pure/issues/299) |
//! | Add a chunked variable-length dataset to an existing file in place | [#109](https://github.com/CramBL/hdf5-pure/issues/109) |
//! | Add a dataset or attribute that refers to a committed (`H5Tcommit`) datatype to an existing file in place, where the in-place engine appends into a fixed layout and has nowhere to place the named type object. [`FileBuilder::commit_datatype`] writes one when the whole file is written, and [`repack`](fn@repack) carries them across | [#254](https://github.com/CramBL/hdf5-pure/issues/254) |
//!
//! A dataset whose storage was never allocated keeps that state through [`repack`](fn@repack)
//! ([#293](https://github.com/CramBL/hdf5-pure/issues/293)), so a schema-only file stays one and
//! is never written out full of its fill value. Three things bound that. A dataset that stores
//! only some of its chunks is unaffected: a sparse grid cannot take the verbatim path, so it falls
//! back to read-and-re-encode and the destination stores every slot, measured at one chunk in and
//! ten out. A resizable destination is given the eagerly built Extensible Array this crate gives
//! every empty resizable dataset, because an in-place append needs the index to exist before the
//! first chunk arrives. Neither form stores a chunk, and the index costs a few hundred bytes the
//! source did not spend. And the `maxshape` bound in the table above does not apply to a dataset
//! that stores nothing, since both halves of it are sized from the slots an index spans and an
//! unallocated one spans none: such a dataset is written, and rejected when its first chunk
//! arrives.
//!
//! Two gaps remain around variable-length datasets, which write chunked, filtered and resizable
//! ([#109](https://github.com/CramBL/hdf5-pure/issues/109)). A resizable one can be created and
//! not grown: [`Dataset::append`] is typed and [`H5Element`] covers numeric scalars alone, and
//! [`append_raw`](Dataset::append_raw) rejects a variable-length datatype, since it cannot encode
//! the heap references. And adding such a dataset to an existing file through the in-place edit
//! engine is rejected, since the engine appends into a fixed layout with nowhere to place the heap
//! collections ahead of the chunks.
//!
//! ### SWMR
//!
//! | Capability | Tracking |
//! |---|---|
//! | Append to multi-dimensional and filtered datasets | [#110](https://github.com/CramBL/hdf5-pure/issues/110) |
//!
//! This gap is specific to SWMR (concurrent-reader) append. Appending to a filtered 1-D unlimited
//! dataset without concurrent readers is supported at any length, through
//! [`Dataset::append_staged`] and the streaming [`Dataset::append`] alike, the streaming one under
//! a lossless pipeline, since growing a lossy dataset's trailing chunk would re-encode committed
//! values.
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
