//! Superblock versions 2 and 3: section `subsec_fmt4_boot_super`, version 4.0.

use crate::bytes;
use crate::checksum;
use crate::superblock::{self, SIGNATURE, VERSION_2, VERSION_3};
use crate::widths::Widths;

/// A checked fixture for the bytes of a version 2 or 3 superblock, including its checksum.
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
    /// Creates a version 2 fixture with zero required addresses, zero flags, and an undefined
    /// extension.
    ///
    /// # Panics
    ///
    /// Panics if either width is other than 2, 4, or 8 bytes.
    pub fn new(widths: Widths) -> Self {
        assert!(matches!(widths.offset, 2 | 4 | 8), "invalid offset width");
        assert!(matches!(widths.length, 2 | 4 | 8), "invalid length width");
        Self {
            version: VERSION_2,
            widths,
            consistency_flags: 0,
            base_address: 0,
            extension_address: None,
            eof_address: 0,
            root_header_address: 0,
        }
    }

    /// Selects the version byte stored with the fixture's fields.
    ///
    /// # Panics
    ///
    /// Panics if `version` is outside the range 2 through 3.
    pub fn version(mut self, version: u8) -> Self {
        assert!(matches!(version, VERSION_2 | VERSION_3));
        self.version = version;
        self
    }

    /// Sets all bits of the stored consistency byte.
    pub fn consistency_flags(mut self, flags: u8) -> Self {
        self.consistency_flags = flags;
        self
    }

    /// Sets the absolute base address from which the extension and root addresses are measured.
    ///
    /// # Panics
    ///
    /// Panics if `address` is at or above the selected offset width's undefined marker.
    pub fn base_address(mut self, address: u64) -> Self {
        self.assert_defined_address(address);
        self.base_address = address;
        self
    }

    /// Sets the stored address of the superblock extension.
    ///
    /// The address is relative to [`Self::base_address`].
    ///
    /// # Panics
    ///
    /// Panics if `address` is at or above the selected offset width's undefined marker.
    pub fn extension_address(mut self, address: u64) -> Self {
        self.assert_defined_address(address);
        self.extension_address = Some(address);
        self
    }

    /// Sets the absolute end-of-file address.
    ///
    /// # Panics
    ///
    /// Panics if `address` is at or above the selected offset width's undefined marker.
    pub fn eof_address(mut self, address: u64) -> Self {
        self.assert_defined_address(address);
        self.eof_address = address;
        self
    }

    /// Sets the root group header address relative to the base address.
    ///
    /// # Panics
    ///
    /// Panics if `address` is at or above the selected offset width's undefined marker.
    pub fn root_header_address(mut self, address: u64) -> Self {
        self.assert_defined_address(address);
        self.root_header_address = address;
        self
    }

    /// Builds the encoded superblock and computes its checksum.
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

    #[track_caller]
    fn assert_defined_address(&self, address: u64) {
        let bits = 8 * self.widths.offset;
        let marker = u64::MAX >> (u64::BITS as usize - bits);
        assert!(
            address < marker,
            "address {address:#x} must fit below {marker:#x}"
        );
    }
}
