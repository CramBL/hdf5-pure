//! Superblock versions 2 and 3: section `subsec_fmt4_boot_super`, version 4.0.

use crate::bytes;
use crate::checksum;
use crate::superblock::{self, SIGNATURE};
use crate::widths::Widths;

/// The bytes of a version 2 or version 3 superblock, checksum included.
///
/// The two versions share a layout. Version 3 differs only in giving the
/// consistency flags meaning, for a file open under single-writer/multiple-
/// reader access.
#[derive(Clone, Copy, Debug)]
pub struct Superblock {
    version: u8,
    widths: Widths,
    consistency_flags: u8,
    base_address: u64,
    extension_address: Option<u64>,
    eof_address: u64,
    root_header_address: u64,
}

impl Superblock {
    pub fn new(widths: Widths) -> Self {
        Self {
            version: 2,
            widths,
            consistency_flags: 0,
            base_address: 0,
            extension_address: None,
            eof_address: 0,
            root_header_address: 0,
        }
    }

    pub fn version(mut self, version: u8) -> Self {
        self.version = version;
        self
    }

    pub fn consistency_flags(mut self, flags: u8) -> Self {
        self.consistency_flags = flags;
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

    pub fn root_header_address(mut self, address: u64) -> Self {
        self.root_header_address = address;
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let offset = self.widths.offset;

        let mut superblock = SIGNATURE.to_vec();
        superblock.push(self.version);
        superblock.push(superblock::width_byte(offset));
        superblock.push(superblock::width_byte(self.widths.length));
        superblock.push(self.consistency_flags);
        bytes::push_uint(&mut superblock, self.base_address, offset);
        bytes::push_address(&mut superblock, self.extension_address, offset);
        bytes::push_uint(&mut superblock, self.eof_address, offset);
        bytes::push_uint(&mut superblock, self.root_header_address, offset);
        checksum::append(&mut superblock);
        superblock
    }
}
