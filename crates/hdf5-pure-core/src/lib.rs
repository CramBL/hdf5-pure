//! The portable shared types of the stable `hdf5-pure` API.
//!
//! This crate owns the types whose identity the workspace crates share, and `hdf5-pure`
//! re-exports them. Its documented public API is compatibility-checked and follows the same
//! versioning as `hdf5-pure`. Most users depend on `hdf5-pure`, which is the recommended entry
//! point. The hidden `__private` module is not part of that API.

#![cfg_attr(test, warn(clippy::tests_outside_test_module))]
#![cfg_attr(not(test), no_std)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod address;
mod address_range;
mod dataspace;
mod datatype;
mod deflate_level;
mod display;
mod error;
mod file_space_info;
mod fill_time;
mod filter_mask;
mod message_type;
mod metadata_bytes;
mod region;
mod superblock;
mod userblock_size;
mod width;
mod zfp_rate;

#[doc(hidden)]
pub mod __private;

pub use address::BaseAddress;
pub use address::DecodedStoredAddress;
pub use address::PhysicalAddress;
pub use address::StoredAddress;
pub use address_range::AddressRange;
pub use dataspace::MaxExtent;
pub use datatype::CharacterSet;
pub use datatype::CompoundMember;
pub use datatype::Datatype;
pub use datatype::EnumMember;
pub use datatype::ReferenceType;
pub use datatype::StringPadding;
pub use datatype::VariableLengthStringType;
pub use datatype::VariableLengthType;
pub use datatype::byte_order::DatatypeByteOrder;
pub use datatype::layout::BitFieldType;
pub use datatype::layout::FixedPointLayout;
pub use datatype::layout::FixedPointType;
pub use datatype::layout::FloatingPointField;
pub use datatype::layout::FloatingPointLayout;
pub use datatype::layout::FloatingPointType;
pub use datatype::layout::MantissaNormalization;
pub use datatype::layout::TimeType;
pub use datatype::layout::builder::BitFieldTypeBuilder;
pub use datatype::layout::builder::FixedPointTypeBuilder;
pub use datatype::layout::builder::FloatingPointTypeBuilder;
pub use datatype::layout::builder::NumericDatatypeProperty;
pub use deflate_level::DeflateLevel;
pub use error::FormatError;
pub use error::OBJECT_HEADER_MESSAGE_MAX;
pub use file_space_info::FileSpaceInfo;
pub use file_space_info::FileSpacePageSize;
pub use file_space_info::FileSpaceSettings;
pub use file_space_info::FileSpaceStrategy;
pub use fill_time::FillTime;
pub use filter_mask::FilterMask;
pub use message_type::MessageType;
pub use metadata_bytes::MetadataBytes;
pub use superblock::Superblock;
pub use superblock::SuperblockStatus;
pub use superblock::SuperblockStatusFlag;
pub use superblock::SuperblockStatusFlags;
pub use superblock::SuperblockVersion;
pub use userblock_size::UserblockSize;
pub use width::LengthWidth;
pub use width::OffsetWidth;
pub use zfp_rate::ZfpRate;
