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
