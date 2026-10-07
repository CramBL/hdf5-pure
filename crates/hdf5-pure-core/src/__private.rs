//! Constructors and helpers that the workspace crates share.
//!
//! The non-exhaustive types of this crate cannot be built with a struct literal outside it, so the
//! other workspace crates build them through the functions and the field structs here. The module
//! also exports [`StoredAddress`], the [`BaseAddressExt`] conversions of a [`BaseAddress`], and the
//! display helpers.
//!
//! This module serves the workspace crates only. It is outside the crate's compatibility
//! guarantee and may change in any release.

use alloc::string::String;
use alloc::vec::Vec;

use crate::BaseAddress;
use crate::CompoundMember;
use crate::Datatype;
use crate::EnumMember;
use crate::FileSpaceInfo;
use crate::FileSpacePageSize;
use crate::FileSpaceStrategy;
use crate::Superblock;

pub use crate::address::BaseAddressExt;
pub use crate::address::StoredAddress;
pub use crate::display::DISPLAY_MAX_MEMBERS;
pub use crate::display::Dims;
pub use crate::display::EscapedName;
pub use crate::display::QuotedBytes;
pub use crate::display::write_elided;

/// The fields of a [`FileSpaceInfo`], for the other workspace crates to build one.
///
/// Each field holds the value of the [`FileSpaceInfo`] field of the same name.
pub struct FileSpaceInfoFields {
    pub strategy: FileSpaceStrategy,
    pub persist: bool,
    pub threshold: u64,
    pub page_size: FileSpacePageSize,
    pub page_end_meta_threshold: u16,
    pub eoa_pre_fsm: u64,
    pub manager_addrs: Vec<u64>,
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

/// Constructs a compound member from fields read from a datatype message.
pub fn compound_member(name: String, byte_offset: u64, datatype: Datatype) -> CompoundMember {
    CompoundMember {
        name,
        byte_offset,
        datatype,
    }
}

/// Constructs an enumeration member from fields read from a datatype message.
pub fn enum_member(name: String, value: Vec<u8>) -> EnumMember {
    EnumMember { name, value }
}

/// The fields of a [`Superblock`], for the other workspace crates to build one.
///
/// Each field holds the value of the [`Superblock`] field of the same name.
pub struct SuperblockFields {
    pub version: u8,
    pub offset_size: u8,
    pub length_size: u8,
    pub base_address: BaseAddress,
    pub eof_address: u64,
    pub root_group_address: u64,
    pub group_leaf_node_k: Option<u16>,
    pub group_internal_node_k: Option<u16>,
    pub indexed_storage_internal_node_k: Option<u16>,
    pub free_space_address: Option<u64>,
    pub driver_info_address: Option<u64>,
    pub consistency_flags: u32,
    pub superblock_extension_address: Option<u64>,
    pub checksum: Option<u32>,
}

impl SuperblockFields {
    /// Creates a [`Superblock`] from these fields.
    pub fn build(self) -> Superblock {
        let Self {
            version,
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
            consistency_flags,
            superblock_extension_address,
            checksum,
        } = self;
        Superblock {
            version,
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
            consistency_flags,
            superblock_extension_address,
            checksum,
        }
    }
}
