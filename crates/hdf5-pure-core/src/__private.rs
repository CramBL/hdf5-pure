//! Constructors and helpers that the workspace crates share.
//!
//! The non-exhaustive types of this crate cannot be built with a struct literal outside it, so the
//! other workspace crates build them through the functions and the field structs here. The module
//! also exports [`StoredAddressExt`], the [`BaseAddressExt`] conversions of a [`BaseAddress`], display
//! helpers, and quantities such as [`ByteLength`], [`ElementCount`], and [`ChunkSlot`].
//!
//! This module serves the workspace crates only. It is outside the crate's compatibility
//! guarantee and may change in any release.

use alloc::vec::Vec;
use core::num::NonZeroU32;

use crate::BaseAddress;
use crate::CompoundMember;
use crate::Datatype;
use crate::DatatypeMemberName;
use crate::EnumMember;
use crate::EnumerationType;
use crate::FileSpaceInfo;
use crate::FileSpacePageSize;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::LengthWidth;
use crate::OffsetWidth;
use crate::StoredAddress;
use crate::Superblock;
use crate::SuperblockStatus;
use crate::SuperblockStatusFlags;

pub use crate::address::Address;
pub use crate::address::BaseAddressExt;
pub use crate::address::DecodedStoredAddress;
pub use crate::address::StoredAddressExt;
pub use crate::address_range::NonEmptyRange;
pub use crate::datatype::{
    DatatypeOffsets, array_type_size, stored_type_size, variable_length_descriptor_size,
};
pub use crate::display::DISPLAY_MAX_MEMBERS;
pub use crate::display::Dims;
pub use crate::display::EscapedName;
pub use crate::display::QuotedBytes;
pub use crate::display::write_elided;
pub use crate::file_space_info::FileSpaceSettingsExt;
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

/// The fields of a [`FileSpaceInfo`], for the other workspace crates to build one.
///
/// Each field holds the value of the [`FileSpaceInfo`] field of the same name.
pub struct FileSpaceInfoFields {
    pub strategy: FileSpaceStrategy,
    pub persist: bool,
    pub threshold: u64,
    pub page_size: FileSpacePageSize,
    pub page_end_meta_threshold: u16,
    pub eoa_pre_fsm: Option<DecodedStoredAddress>,
    pub manager_addrs: Vec<DecodedStoredAddress>,
}

impl FileSpaceInfoFields {
    /// Creates a [`FileSpaceInfo`] from these fields.
    pub fn build(self) -> FileSpaceInfo {
        let Self {
            strategy,
            persist,
            threshold,
            page_size,
            page_end_meta_threshold,
            eoa_pre_fsm,
            manager_addrs,
        } = self;
        FileSpaceInfo {
            strategy,
            persist,
            threshold,
            page_size,
            page_end_meta_threshold,
            eoa_pre_fsm,
            manager_addrs,
        }
    }
}

/// Borrows a compound member's name and datatype and copies its byte offset.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{__private, CompoundMember, Datatype, DatatypeMemberName, FixedPointType};
///
/// let member = CompoundMember::new(
///     DatatypeMemberName::from(c"field"), 4, Datatype::FixedPoint(FixedPointType::u16()),
/// );
/// let (name, offset, datatype) = __private::compound_member_parts(&member);
/// assert_eq!(name.as_bytes(), b"field");
/// assert_eq!(offset, 4);
/// assert_eq!(datatype, &Datatype::FixedPoint(FixedPointType::u16()));
/// ```
pub fn compound_member_parts(member: &CompoundMember) -> (&DatatypeMemberName, u64, &Datatype) {
    let CompoundMember {
        name,
        byte_offset,
        datatype,
    } = member;
    (name, *byte_offset, datatype)
}

/// Borrows an enumeration member's name and stored value bytes without copying.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{__private, DatatypeMemberName, EnumMember};
///
/// let member = EnumMember::new(DatatypeMemberName::from(c"active"), vec![1, 0]);
/// let (name, value) = __private::enum_member_parts(&member);
/// assert_eq!(name.as_bytes(), b"active");
/// assert_eq!(value, &[1, 0]);
/// ```
pub fn enum_member_parts(member: &EnumMember) -> (&DatatypeMemberName, &[u8]) {
    let EnumMember { name, value } = member;
    (name, value)
}

/// Constructs an enumeration from a decoded base and member values.
///
/// Pass the stored element size and the descriptor widths from the base description.
///
/// # Errors
///
/// Returns [`FormatError::EnumBaseSizeMismatch`] if `size` differs from the base's stored
/// size, or [`FormatError::EnumMemberValueSize`] if a member's width differs from `size`.
/// Propagates base size calculation errors.
pub fn enumeration_from_stored_parts(
    base_type: Datatype,
    size: NonZeroU32,
    offsets: DatatypeOffsets,
    members: Vec<EnumMember>,
) -> Result<EnumerationType, FormatError> {
    EnumerationType::from_stored_parts(base_type, size, offsets, members)
}

/// Returns the variable-length descriptor widths of a decoded enumeration's base.
pub const fn enumeration_base_offsets(datatype: &EnumerationType) -> DatatypeOffsets {
    datatype.offsets()
}

/// The fields of a [`Superblock`], for the other workspace crates to build one.
///
/// Each field holds the value of the [`Superblock`] field of the same name.
pub struct SuperblockFields {
    pub status: SuperblockStatus,
    pub offset_size: OffsetWidth,
    pub length_size: LengthWidth,
    pub base_address: BaseAddress,
    pub eof_address: u64,
    pub root_group_address: StoredAddress,
    pub group_leaf_node_k: Option<u16>,
    pub group_internal_node_k: Option<u16>,
    pub indexed_storage_internal_node_k: Option<u16>,
    pub free_space_address: Option<StoredAddress>,
    pub driver_info_address: Option<StoredAddress>,
    pub superblock_extension_address: Option<StoredAddress>,
    pub checksum: Option<u32>,
}

impl SuperblockFields {
    /// Creates a [`Superblock`] from these fields.
    pub fn build(self) -> Superblock {
        let Self {
            status,
            offset_size,
            length_size,
            base_address,
            eof_address,
            root_group_address,
            group_leaf_node_k,
            group_internal_node_k,
            indexed_storage_internal_node_k,
            free_space_address,
            driver_info_address,
            superblock_extension_address,
            checksum,
        } = self;
        Superblock {
            status,
            offset_size,
            length_size,
            base_address,
            eof_address,
            root_group_address,
            group_leaf_node_k,
            group_internal_node_k,
            indexed_storage_internal_node_k,
            free_space_address,
            driver_info_address,
            superblock_extension_address,
            checksum,
        }
    }
}

/// Creates status flags from an encoded field, preserving every bit.
pub const fn superblock_status_flags(bits: u32) -> SuperblockStatusFlags {
    SuperblockStatusFlags::from_bits(bits)
}
