//! Superblock versions 0 and 1: section `subsec_fmt4_boot_super`, version 4.0.

use crate::bytes;
use crate::superblock::{self, SIGNATURE};
use crate::widths::Widths;

/// The bytes of a version 0 or version 1 superblock, the root group's
/// symbol-table entry included.
#[derive(Clone, Copy, Debug)]
pub struct Superblock {
    version: u8,
    widths: Widths,
    consistency_flags: u32,
    group_leaf_node_k: u16,
    group_internal_node_k: u16,
    indexed_storage_internal_node_k: u16,
    base_address: u64,
    free_space_address: Option<u64>,
    eof_address: u64,
    driver_info_address: Option<u64>,
    root_link_name_offset: u64,
    root_header_address: u64,
}

impl Superblock {
    /// A version 0 superblock, the oldest the format defines.
    pub fn new(widths: Widths) -> Self {
        Self {
            version: 0,
            widths,
            consistency_flags: 0,
            group_leaf_node_k: 4,
            group_internal_node_k: 16,
            indexed_storage_internal_node_k: 32,
            base_address: 0,
            free_space_address: None,
            eof_address: 0,
            driver_info_address: None,
            root_link_name_offset: 0,
            root_header_address: 0,
        }
    }

    /// Writes version 1, which adds the indexed-storage B-tree K value and its
    /// reserved bytes between the consistency flags and the addresses.
    pub fn version_1(mut self) -> Self {
        self.version = 1;
        self
    }

    pub fn consistency_flags(mut self, flags: u32) -> Self {
        self.consistency_flags = flags;
        self
    }

    pub fn indexed_storage_internal_node_k(mut self, k: u16) -> Self {
        self.indexed_storage_internal_node_k = k;
        self
    }

    /// The offset every other address in the file is measured from.
    pub fn base_address(mut self, address: u64) -> Self {
        self.base_address = address;
        self
    }

    pub fn eof_address(mut self, address: u64) -> Self {
        self.eof_address = address;
        self
    }

    /// The address of the root group's object header, and its name's offset in
    /// the enclosing local heap.
    pub fn root_group(mut self, link_name_offset: u64, header_address: u64) -> Self {
        self.root_link_name_offset = link_name_offset;
        self.root_header_address = header_address;
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let offset = self.widths.offset;

        let mut superblock = SIGNATURE.to_vec();
        superblock.push(self.version);
        // Free-space storage, root symbol-table entry and shared-header
        // message versions, each zero in every file either library writes.
        superblock.extend_from_slice(&[0, 0, 0, 0]);
        superblock.push(superblock::width_byte(offset));
        superblock.push(superblock::width_byte(self.widths.length));
        superblock.push(0);
        superblock.extend_from_slice(&self.group_leaf_node_k.to_le_bytes());
        superblock.extend_from_slice(&self.group_internal_node_k.to_le_bytes());
        superblock.extend_from_slice(&self.consistency_flags.to_le_bytes());
        if self.version >= 1 {
            superblock.extend_from_slice(&self.indexed_storage_internal_node_k.to_le_bytes());
            superblock.extend_from_slice(&0u16.to_le_bytes());
        }

        bytes::push_uint(&mut superblock, self.base_address, offset);
        bytes::push_address(&mut superblock, self.free_space_address, offset);
        bytes::push_uint(&mut superblock, self.eof_address, offset);
        bytes::push_address(&mut superblock, self.driver_info_address, offset);

        // The root group's symbol-table entry: its name's heap offset, its
        // object header's address, a cache type, reserved bytes and the
        // scratch pad the cache type interprets.
        bytes::push_uint(&mut superblock, self.root_link_name_offset, offset);
        bytes::push_uint(&mut superblock, self.root_header_address, offset);
        superblock.extend_from_slice(&0u32.to_le_bytes());
        superblock.extend_from_slice(&0u32.to_le_bytes());
        superblock.extend_from_slice(&[0; SCRATCH_PAD]);
        superblock
    }
}

/// The scratch pad a symbol-table entry ends with, whose meaning its cache type
/// decides.
const SCRATCH_PAD: usize = 16;
