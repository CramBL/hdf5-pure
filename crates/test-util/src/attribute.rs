//! Attribute message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_attribute`, version 4.0.

use crate::bytes;
use crate::widths::Widths;

/// The body of an Attribute message.
///
/// The three versions differ in how the name, the datatype and the dataspace
/// are separated: version 1 pads each to eight bytes, version 2 drops the
/// padding and gives the flags byte meaning, and version 3 adds the name's
/// character set.
#[derive(Clone, Debug)]
pub struct Attribute<'a> {
    version: u8,
    flags: Flags,
    name: &'a str,
    character_set: u8,
    datatype: &'a [u8],
    dataspace: &'a [u8],
    data: &'a [u8],
}

impl<'a> Attribute<'a> {
    pub fn new(name: &'a str, datatype: &'a [u8], dataspace: &'a [u8], data: &'a [u8]) -> Self {
        Self {
            version: 1,
            flags: Flags::NONE,
            name,
            character_set: 0,
            datatype,
            dataspace,
            data,
        }
    }

    /// Says which of the datatype and the dataspace fields hold a reference to
    /// a shared message. Version 1 has no such byte, so setting this moves the
    /// message to version 2.
    pub fn flags(mut self, flags: Flags) -> Self {
        self.version = self.version.max(2);
        self.flags = flags;
        self
    }

    pub fn character_set(mut self, character_set: u8) -> Self {
        self.version = self.version.max(3);
        self.character_set = character_set;
        self
    }

    pub fn build(&self) -> Vec<u8> {
        // A version 1 name field counts its own terminator, and the later
        // versions do too.
        let name = {
            let mut name = self.name.as_bytes().to_vec();
            name.push(0);
            name
        };

        let mut message = vec![self.version];
        message.push(if self.version == 1 { 0 } else { self.flags.0 });
        message.extend_from_slice(&(name.len() as u16).to_le_bytes());
        message.extend_from_slice(&(self.datatype.len() as u16).to_le_bytes());
        message.extend_from_slice(&(self.dataspace.len() as u16).to_le_bytes());
        if self.version >= 3 {
            message.push(self.character_set);
        }

        let padded = self.version == 1;
        for field in [name.as_slice(), self.datatype, self.dataspace] {
            message.extend_from_slice(field);
            if padded {
                message.resize(message.len().next_multiple_of(ALIGNMENT), 0);
            }
        }
        message.extend_from_slice(self.data);
        message
    }
}

/// The body of an Attribute Info message, which states where a dense
/// attribute set's structures live.
pub fn info(fractal_heap: Option<u64>, name_index: Option<u64>, widths: Widths) -> Vec<u8> {
    let mut message = vec![INFO_VERSION, 0];
    for address in [fractal_heap, name_index] {
        bytes::push_address(&mut message, address, widths.offset);
    }
    message
}

/// Which of an attribute's fields hold a reference to a shared message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flags(pub u8);

impl Flags {
    pub const NONE: Self = Self(0x00);
    pub const SHARED_DATATYPE: Self = Self(0x01);
    pub const SHARED_DATASPACE: Self = Self(0x02);
}

impl core::ops::BitOr for Flags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// A version 1 message pads each of its three variable fields to this.
const ALIGNMENT: usize = 8;

const INFO_VERSION: u8 = 0;
