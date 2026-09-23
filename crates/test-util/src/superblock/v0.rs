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
            version: VERSION_0,
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
        self.version = VERSION_1;
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
        if self.version == VERSION_1 {
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

/// The fields of a version 0 or version 1 superblock a test follows or edits, as a file stores
/// them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fields {
    pub widths: Widths,
    pub base_address: u64,
    /// Where the end-of-file address is stored.
    pub eof_address_at: usize,
    /// The address of the root group's object header, from its symbol-table entry.
    pub root_header_address: u64,
}

impl Fields {
    /// The fields of the superblock beginning at `at` in `file`.
    #[track_caller]
    pub fn read(file: &[u8], at: usize) -> Self {
        let version = superblock::version_at(file, at);
        // Signature(8) + version(1) + three versions and a reserved byte(4), then the widths:
        // section `subsec_fmt4_boot_super`, version 4.0.
        let widths_at = at + SIGNATURE.len() + 1 + 4;
        let widths = Widths::new(
            usize::from(bytes::u8_at(file, widths_at)),
            usize::from(bytes::u8_at(file, widths_at + 1)),
        );
        // Two widths(2) + reserved(1) + two group K values(4) + consistency flags(4), same
        // section.
        let addresses_at = widths_at
            + 2
            + 1
            + 4
            + 4
            + match version {
                VERSION_0 => 0,
                VERSION_1 => INDEXED_STORAGE_K_AND_RESERVED,
                other => panic!("superblock version {other} at {at:#x} is not version 0 or 1"),
            };
        let offset = widths.offset;
        // The base, free-space, end-of-file and driver-information addresses, then the root
        // group's symbol-table entry: its link name offset, then its object header's address.
        Self {
            widths,
            base_address: bytes::uint_at(file, addresses_at, offset),
            eof_address_at: addresses_at + 2 * offset,
            root_header_address: bytes::uint_at(file, addresses_at + 5 * offset, offset),
        }
    }
}

/// The scratch pad a symbol-table entry ends with, whose meaning its cache type
/// decides.
const SCRATCH_PAD: usize = 16;

/// The indexed-storage B-tree K value(2) and its reserved bytes(2) that version 1 adds after the
/// consistency flags: section `subsec_fmt4_boot_super`, version 4.0.
const INDEXED_STORAGE_K_AND_RESERVED: usize = 4;

/// The two superblock versions this module builds and reads, same section.
const VERSION_0: u8 = 0;
const VERSION_1: u8 = 1;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::bytes;
    use crate::superblock::v0::{Fields, Superblock};
    use crate::widths::Widths;

    #[rstest]
    #[case::version_0(Superblock::new(Widths::EIGHT))]
    #[case::version_1(Superblock::new(Widths::EIGHT).version_1())]
    fn reads_back_the_fields_it_builds(#[case] superblock: Superblock) {
        let file = superblock.eof_address(4096).root_group(0, 96).build();

        let fields = Fields::read(&file, 0);
        assert_eq!(fields.widths, Widths::EIGHT);
        assert_eq!(fields.base_address, 0);
        assert_eq!(fields.root_header_address, 96);
        assert_eq!(bytes::u64_at(&file, fields.eof_address_at), 4096);
    }
}
