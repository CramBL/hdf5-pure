//! HDF5 on-disk structures and their byte encodings.
//!
//! The types describe metadata stored in HDF5 files. Parsers read message bodies
//! from byte slices, and writers produce bytes for file operations. File
//! navigation, storage access, and filter execution belong to `hdf5-pure` and
//! `hdf5-pure-filter`.
//!
//! The project supports `hdf5-pure` as its API entry point. Types re-exported there follow
//! that crate's compatibility policy. Direct use of this published support crate has no
//! independent API compatibility guarantee.

#![cfg_attr(not(test), no_std)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod address;
mod bytes;
mod checksum;
mod convert;
mod data_layout;
mod dataspace;
mod datatype;
mod error;
mod fill_value;
mod filter_pipeline;
mod message_type;
mod signature;
mod width;

pub use address::StoredAddress;
pub use bytes::ensure_len;
pub use bytes::read_length;
pub use bytes::read_length_width;
pub use bytes::read_offset;
pub use bytes::read_offset_width;
pub use bytes::read_optional_offset;
pub use bytes::read_optional_offset_width;
pub use bytes::read_uint_width;
pub use checksum::jenkins_lookup3;
pub use checksum::verify_trailing;
pub use convert::Narrow;
pub use convert::NarrowTarget;
pub use convert::is_undefined_addr;
pub use convert::slice_range;
pub use data_layout::COMPACT_DATA_OFFSET;
pub use data_layout::ChunkIndexLayout;
pub use data_layout::ChunkedLayoutFlags;
pub use data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS;
pub use data_layout::DataLayout;
pub use data_layout::FilteredSingleChunk;
pub use data_layout::SINGLE_INDEX_WITH_FILTER;
pub use dataspace::Dataspace;
pub use dataspace::DataspaceType;
pub use dataspace::Extent;
pub use dataspace::MaxExtent;
pub use datatype::CharacterSet;
pub use datatype::CompoundMember;
pub use datatype::Datatype;
pub use datatype::EnumMember;
pub use datatype::ReferenceType;
pub use datatype::StringPadding;
pub use datatype::byte_order::DatatypeByteOrder;
pub use datatype::byte_order::FixedWidthByteOrder;
pub use datatype::class_may_hold_object_address;
pub use datatype::datatype_holds_file_address;
pub use datatype::datatype_holds_object_address;
pub use datatype::element_size_usize;
pub use datatype::embedded_reference_slots;
pub use datatype::layout::FixedPointLayout;
pub use datatype::layout::FloatingPointLayout;
pub use datatype::layout::StandardNumericLayout;
pub use datatype::layout::StandardWidth;
pub use datatype::parse_datatype;
pub use datatype::serialize_datatype;
pub use datatype::stored_object_references;
pub use error::FormatError;
pub use error::OBJECT_HEADER_MESSAGE_MAX;
pub use fill_value::FillValueError;
pub use fill_value::V3_FLAGS_DEFAULT;
pub use fill_value::fill_value_is_written;
pub use fill_value::fill_value_message_v3;
pub use fill_value::parse_defined_fill_value;
pub use filter_pipeline::FilterDescription;
pub use filter_pipeline::FilterPipeline;
pub use filter_pipeline::FilterPipelineError;
#[doc(hidden)]
pub use hdf5_pure_core::__private::DISPLAY_MAX_MEMBERS;
#[doc(hidden)]
pub use hdf5_pure_core::__private::Dims;
#[doc(hidden)]
pub use hdf5_pure_core::__private::EscapedName;
#[doc(hidden)]
pub use hdf5_pure_core::__private::QuotedBytes;
#[doc(hidden)]
pub use hdf5_pure_core::__private::write_elided;
pub use message_type::MessageType;
pub use signature::HDF5_SIGNATURE;
pub use width::LengthWidth;
pub use width::OffsetWidth;
pub use width::UintWidth;
