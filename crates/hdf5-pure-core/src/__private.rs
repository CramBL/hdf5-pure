//! Re-exports of constructors and helpers for the workspace crates.
//!
//! This module serves the workspace crates only. It is outside the crate's compatibility
//! guarantee and may change in any release.

pub use crate::address::Address;
pub use crate::address::BaseAddressExt;
pub use crate::address::DecodedStoredAddress;
pub use crate::address::StoredAddressExt;
pub use crate::address_range::NonEmptyRange;
pub use crate::datatype::compound_member::compound_member_parts;
pub use crate::datatype::enumeration::{
    enum_member_parts, enumeration_base_offsets, enumeration_from_stored_parts,
};
pub use crate::datatype::stored_size::{
    DatatypeOffsets, array_type_size, stored_type_size, variable_length_descriptor_size,
};
pub use crate::display::DISPLAY_MAX_MEMBERS;
pub use crate::display::Dims;
pub use crate::display::EscapedName;
pub use crate::display::QuotedBytes;
pub use crate::display::write_elided;
pub use crate::file_space_info::{FileSpaceInfoFields, FileSpaceSettingsExt};
pub use crate::filter_mask::FILTER_COUNT_MAX;
pub use crate::filter_mask::PipelineFilterMask;
pub use crate::metadata_bytes::MetadataBytes;
pub use crate::metadata_bytes::MetadataBytesExt;
pub use crate::metadata_bytes::MetadataFrame;
pub use crate::quantity::ByteLength;
pub use crate::quantity::ChunkSlot;
pub use crate::quantity::ElementCount;
pub use crate::quantity::ElementSize;
pub use crate::quantity::NonZeroByteLength;
pub use crate::quantity::NonZeroElementCount;
pub use crate::quantity::SlotCount;
pub use crate::region::FileRange;
pub use crate::region::FileRegion;
pub use crate::superblock::{SuperblockFields, superblock_status_flags};
