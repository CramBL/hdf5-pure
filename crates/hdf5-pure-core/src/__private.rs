//! Constructs shared member types for the format parser, and shares the display helpers.
//!
//! The member structs are non-exhaustive, so only this crate can construct
//! them with struct literals. The format parser calls these functions to build
//! members across the crate boundary.
//!
//! This module serves the workspace crates only. It is outside the crate's compatibility
//! guarantee and may change in any release.

use alloc::string::String;
use alloc::vec::Vec;

use crate::BaseAddress;
use crate::CompoundMember;
use crate::Datatype;
use crate::EnumMember;
use crate::Superblock;

pub use crate::address::BaseAddressExt;
pub use crate::address::StoredAddress;
pub use crate::display::DISPLAY_MAX_MEMBERS;
pub use crate::display::Dims;
pub use crate::display::EscapedName;
pub use crate::display::QuotedBytes;
pub use crate::display::write_elided;

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

pub fn superblock(fields: SuperblockFields) -> Superblock {
    let SuperblockFields {
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
    } = fields;
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
