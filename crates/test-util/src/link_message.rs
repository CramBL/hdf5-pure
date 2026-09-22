//! Link message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_link`, version 4.0.

use crate::bytes;
use crate::widths::Widths;

/// The body of a Link message pointing at the object at `address`.
#[derive(Clone, Debug)]
pub struct HardLink<'n> {
    name: &'n str,
    address: u64,
    name_size_width: usize,
    creation_order: Option<u64>,
    character_set: Option<CharacterSet>,
}

impl<'n> HardLink<'n> {
    pub fn new(name: &'n str, address: u64) -> Self {
        Self {
            name,
            address,
            name_size_width: 1,
            creation_order: None,
            character_set: None,
        }
    }

    /// How many bytes the message spends on the name's length, which its flags
    /// encode as the base-two logarithm of this.
    #[track_caller]
    pub fn name_size_width(mut self, width: usize) -> Self {
        assert!(
            width.is_power_of_two() && width <= 8,
            "a name size field is 1, 2, 4 or 8 bytes wide, not {width}"
        );
        self.name_size_width = width;
        self
    }

    pub fn creation_order(mut self, creation_order: u64) -> Self {
        self.creation_order = Some(creation_order);
        self
    }

    /// Writes the character set byte, which a message omits when the name is
    /// ASCII.
    pub fn character_set(mut self, character_set: CharacterSet) -> Self {
        self.character_set = Some(character_set);
        self
    }

    pub fn build(&self, widths: Widths) -> Vec<u8> {
        let mut flags = self.name_size_width.trailing_zeros() as u8;
        if self.creation_order.is_some() {
            flags |= TRACKS_CREATION_ORDER;
        }
        if self.character_set.is_some() {
            flags |= STORES_CHARACTER_SET;
        }

        let mut message = Vec::new();
        message.push(VERSION);
        message.push(flags);
        // A hard link leaves the link-type bit clear, which keeps the link
        // type field out of the message.
        if let Some(creation_order) = self.creation_order {
            message.extend_from_slice(&creation_order.to_le_bytes());
        }
        if let Some(character_set) = self.character_set {
            message.push(character_set.0);
        }
        bytes::push_uint(&mut message, self.name.len() as u64, self.name_size_width);
        message.extend_from_slice(self.name.as_bytes());
        bytes::push_uint(&mut message, self.address, widths.offset);
        message
    }
}

/// How a link's name is encoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CharacterSet(pub u8);

impl CharacterSet {
    pub const ASCII: Self = Self(0);
    pub const UTF8: Self = Self(1);
}

/// Bit 2 of the flags byte: a creation order follows the flags.
const TRACKS_CREATION_ORDER: u8 = 0x04;

/// Bit 4: a character set byte follows the creation order.
const STORES_CHARACTER_SET: u8 = 0x10;

const VERSION: u8 = 1;
