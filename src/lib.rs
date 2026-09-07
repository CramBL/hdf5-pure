//! Pure-Rust HDF5 file reading, writing, and in-place editing library.
//!
//! `hdf5-pure` is a zero-C-dependency crate for creating, reading, and editing
//! HDF5 files. It is WASM-compatible and supports `no_std` environments with
//! `alloc`.
//!
//! # Writing files
//!
//! ```rust
//! use hdf5_pure::{FileBuilder, AttrValue};
//!
//! let mut builder = FileBuilder::new();
//! builder.create_dataset("data")
//!     .with_f64_data(&[1.0, 2.0, 3.0])
//!     .with_shape(&[3])
//!     .set_attr("unit", AttrValue::String("m/s".into()));
//! let bytes = builder.finish().unwrap();
//! ```
//!
//! # Reading files
//!
//! ```rust,no_run
//! use hdf5_pure::File;
//!
//! let file = File::from_bytes(std::fs::read("output.h5").unwrap()).unwrap();
//! let ds = file.dataset("data").unwrap();
//! let values = ds.read_f64().unwrap();
//! ```
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
//! let file = File::from_bytes(fb.finish().unwrap()).unwrap();
//! let counts: Vec<u32> = file.dataset("counts").unwrap().read().unwrap();
//! assert_eq!(counts, vec![1, 2, 3]);
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
//! the reference HDF5 C library, and h5py across all of their on-disk formats,
//! and refuses — rather than silently degrade the file — anything it cannot
//! reproduce faithfully. Memory follows the file rather than the caller's choice
//! of function: a latest-format file with no userblock is edited without ever
//! building a whole-file copy of it (see [`MemoryStrategy`]).
//!
//! ```rust,no_run
//! use hdf5_pure::File;
//!
//! let file = File::open_rw("output.h5").unwrap();
//! let root = file.root();
//! root.create_dataset("extra", |b| { b.with_f64_data(&[4.0, 5.0]); }).unwrap();
//! file.dataset("data").unwrap().write(&[7.0, 8.0]).unwrap(); // H5Dwrite (overwrite)
//! root.delete("old").unwrap();                               // H5Ldelete
//! file.commit().unwrap();                                    // apply staged edits
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
//! ```rust,no_run
//! use hdf5_pure::File;
//!
//! let file = File::open_streaming("huge.h5").unwrap();
//! let values = file.dataset("signal").unwrap().read_f64().unwrap();
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
//! ```
//! # #[cfg(feature = "ndarray")] {
//! use hdf5_pure::{File, FileBuilder};
//! use ndarray::{array, Array2};
//!
//! let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let mut fb = FileBuilder::new();
//! fb.create_dataset("m").with_ndarray(&a);
//! let bytes = fb.finish().unwrap();
//!
//! let file = File::from_bytes(bytes).unwrap();
//! let back: Array2<f64> = file.dataset("m").unwrap().read_array().unwrap();
//! assert_eq!(a, back);
//! # }
//! ```

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
