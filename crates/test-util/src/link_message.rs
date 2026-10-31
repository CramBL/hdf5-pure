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

/// The body of a Link message with a soft or an external target.
#[derive(Clone, Debug)]
pub struct SymbolicLink<'n> {
    name: &'n str,
    link_type: u8,
    value: Vec<u8>,
    creation_order: Option<u64>,
    character_set: Option<CharacterSet>,
}

impl<'n> SymbolicLink<'n> {
    pub fn soft(name: &'n str, target: &str) -> Self {
        Self::new(name, SOFT_LINK, target.as_bytes().to_vec())
    }

    /// Writes a version 0 external link value without flags, followed by `filename` and
    /// `object_path`, each null-terminated.
    pub fn external(name: &'n str, filename: &str, object_path: &str) -> Self {
        let mut value = vec![EXTERNAL_LINK_VERSION_AND_FLAGS];
        for path in [filename, object_path] {
            value.extend_from_slice(path.as_bytes());
            value.push(0);
        }
        Self::new(name, EXTERNAL_LINK, value)
    }

    fn new(name: &'n str, link_type: u8, value: Vec<u8>) -> Self {
        Self {
            name,
            link_type,
            value,
            creation_order: None,
            character_set: None,
        }
    }

    pub fn creation_order(mut self, creation_order: u64) -> Self {
        self.creation_order = Some(creation_order);
        self
    }

    pub fn character_set(mut self, character_set: CharacterSet) -> Self {
        self.character_set = Some(character_set);
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let mut flags = STORES_LINK_TYPE;
        if self.creation_order.is_some() {
            flags |= TRACKS_CREATION_ORDER;
        }
        if self.character_set.is_some() {
            flags |= STORES_CHARACTER_SET;
        }

        let mut message = vec![VERSION, flags, self.link_type];
        if let Some(creation_order) = self.creation_order {
            message.extend_from_slice(&creation_order.to_le_bytes());
        }
        if let Some(character_set) = self.character_set {
            message.push(character_set.0);
        }
        message.push(u8::try_from(self.name.len()).expect("a one-byte link name length"));
        message.extend_from_slice(self.name.as_bytes());
        message.extend_from_slice(
            &u16::try_from(self.value.len())
                .expect("a two-byte link value length")
                .to_le_bytes(),
        );
        message.extend_from_slice(&self.value);
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

/// Bit 3: a link type byte follows the flags.
const STORES_LINK_TYPE: u8 = 0x08;

/// Bit 4: a character set byte follows the creation order.
const STORES_CHARACTER_SET: u8 = 0x10;

const SOFT_LINK: u8 = 1;
const EXTERNAL_LINK: u8 = 64;
const EXTERNAL_LINK_VERSION_AND_FLAGS: u8 = 0;

const VERSION: u8 = 1;
