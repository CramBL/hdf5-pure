//! HDF5 on-disk structures and their byte encodings.
//!
//! Besides a hidden module, the crate root holds only re-exports of `hdf5-pure-core` items, such
//! as [`Datatype`], [`Superblock`], and [`FormatError`]. The crate exports its parsers and its
//! writers to the workspace crates alone, through that module, which may change in any release.
//! File navigation, storage access, and filter execution belong to `hdf5-pure` and
//! `hdf5-pure-filter`.
//!
//! The project supports `hdf5-pure` as its API entry point. A type that `hdf5-pure` re-exports
//! follows the compatibility policy of `hdf5-pure`.

#![cfg_attr(not(test), no_std)]
#![warn(unreachable_pub)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[doc(hidden)]
pub mod __private;

mod access_mode;
mod address;
mod attribute_info;
mod attribute_message;
mod btree_v1;
mod btree_v2;
mod btree_v2_write;
mod bytes;
mod checksum;
mod chunk_record;
mod convert;
mod data_layout;
mod dataspace;
mod datatype;
mod error;
mod extensible_array;
mod file_space_info;
mod fill_value;
mod filter_pipeline;
mod fixed_array;
mod fractal_heap;
mod fractal_heap_write;
mod free_space_manager;
mod global_heap;
mod link_info;
mod link_message;
mod local_heap;
mod message_flags;
mod message_type;
mod metadata_source;
mod object_header;
mod object_header_writer;
mod shared_message;
mod signature;
mod sohm;
mod superblock;
mod symbol_table;
mod width;

pub use hdf5_pure_core::BaseAddress;
pub use hdf5_pure_core::FileSpaceInfo;
pub use hdf5_pure_core::FileSpaceStrategy;
pub use hdf5_pure_core::Superblock;

pub use hdf5_pure_core::FormatError;
pub use hdf5_pure_core::MessageType;
pub use hdf5_pure_core::OBJECT_HEADER_MESSAGE_MAX;

pub use hdf5_pure_core::CharacterSet;
pub use hdf5_pure_core::CompoundMember;
pub use hdf5_pure_core::Datatype;
pub use hdf5_pure_core::DatatypeByteOrder;
pub use hdf5_pure_core::EnumMember;
pub use hdf5_pure_core::FixedPointLayout;
pub use hdf5_pure_core::FloatingPointLayout;
pub use hdf5_pure_core::ReferenceType;
pub use hdf5_pure_core::StringPadding;

pub use hdf5_pure_core::MaxExtent;
