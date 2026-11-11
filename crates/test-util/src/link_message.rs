//! Link message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_link`, version 4.0.

use crate::btree::v2 as btree;
use crate::bytes;
use crate::checksum;
use crate::fractal_heap;
use crate::image::Image;
use crate::widths::Widths;

/// The body of a Link message pointing at the object at `address`.
#[derive(Clone, Debug)]
pub struct HardLink<'n> {
    name: &'n [u8],
    address: u64,
    name_size_width: usize,
    creation_order: Option<u64>,
    character_set: Option<CharacterSet>,
}

impl<'n> HardLink<'n> {
    /// Starts a hard Link message with the name's UTF-8 bytes and a stored target address.
    pub fn new(name: &'n str, address: u64) -> Self {
        Self::from_bytes(name.as_bytes(), address)
    }

    /// Starts a hard Link message with raw name bytes and a stored target address.
    pub fn from_bytes(name: &'n [u8], address: u64) -> Self {
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
        message.extend_from_slice(self.name);
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
    /// Starts a soft Link message with the target's UTF-8 bytes.
    pub fn soft(name: &'n str, target: &str) -> Self {
        Self::soft_bytes(name, target.as_bytes())
    }

    /// Starts a soft Link message with raw target bytes.
    pub fn soft_bytes(name: &'n str, target: &[u8]) -> Self {
        Self::new(name, SOFT_LINK, target.to_vec())
    }

    /// Starts an external Link message with null-terminated filename and object path fields.
    pub fn external(name: &'n str, filename: &str, object_path: &str) -> Self {
        Self::external_bytes(name, filename.as_bytes(), object_path.as_bytes())
    }

    /// Starts a version 0 external link value with raw filename and object path bytes.
    ///
    /// Appends a null terminator to each field.
    pub fn external_bytes(name: &'n str, filename: &[u8], object_path: &[u8]) -> Self {
        let mut value = vec![EXTERNAL_LINK_VERSION_AND_FLAGS];
        for path in [filename, object_path] {
            value.extend_from_slice(path);
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

/// Builds a managed-heap image and version 0 Link Info body for one supplied Link message.
///
/// Uses eight-byte addresses and lengths. The caller passes `name` for the name-index hash and
/// `body` as uninterpreted object bytes.
/// The Link Info fields are defined in "The Link Info Message" of the
/// [format specification, version 4.0][spec].
///
/// # Panics
///
/// Panics if `body` exceeds the heap's 64-byte managed-object limit.
///
/// # Examples
///
/// ```
/// use test_util::link_message::{self, HardLink};
/// use test_util::widths::Widths;
///
/// let body = HardLink::new("u", 42).build(Widths::EIGHT);
/// let (image, info) = link_message::dense_storage(b"u", &body);
/// assert_eq!(&image[..4], b"FRHP");
/// assert_eq!(info.len(), 18);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
pub fn dense_storage(name: &[u8], body: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let widths = Widths::EIGHT;
    let heap = fractal_heap::Header::new(256).managed_object_count(1);
    let mut image = Image::starting_with(&heap.build(widths));
    let mut block = heap.direct_block(0, body, widths);
    let object_offset = u16::try_from(block.len() - body.len()).unwrap();
    block.resize(usize::try_from(heap.direct_block_size()).unwrap(), 0);
    image.place(256, &block);
    let mut record = checksum::lookup3(name).to_le_bytes().to_vec();
    record.extend_from_slice(&heap.managed_id(object_offset, u8::try_from(body.len()).unwrap()));
    image.place(
        512,
        &btree::Header::new(LINK_NAME_INDEX, LINK_NAME_RECORD_SIZE, 1024, 1).build(widths),
    );
    image.place(1024, &btree::leaf(LINK_NAME_INDEX, &[record]));
    let mut info = vec![LINK_INFO_VERSION, 0];
    bytes::push_address(&mut info, Some(0), widths.offset);
    bytes::push_address(&mut info, Some(512), widths.offset);
    (image.build(), info)
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

// "Version 2 B-trees" in the format specification, version 4.0, defines type 5 for link-name
// indexes.
const LINK_NAME_INDEX: u8 = 5;
// "Version 2 B-trees" in the format specification, version 4.0, defines a four-byte hash and
// seven-byte heap ID in type 5 records.
const LINK_NAME_RECORD_SIZE: u16 = 4 + 7;

// "The Link Info Message" in the format specification, version 4.0, defines version 0.
const LINK_INFO_VERSION: u8 = 0;
