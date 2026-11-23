//! Attribute message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_attribute`, version 4.0.

use crate::bytes;
use crate::datatype;
use crate::widths::Widths;

/// The Attribute message layout used by the fixture.
#[derive(Clone, Copy, Debug)]
pub enum UnpaddedVersion {
    /// Version 2 with an ASCII name.
    Two,
    /// Version 3 with a character-set byte for the name.
    Three,
}

/// An Attribute body with checked framing for version 2 or 3.
#[derive(Debug)]
pub struct UnpaddedAttribute<'a> {
    version: UnpaddedVersion,
    name: &'a str,
    datatype: DatatypeField<'a>,
    dataspace: &'a [u8],
    data: &'a [u8],
}

impl<'a> UnpaddedAttribute<'a> {
    /// Describes an attribute with inline datatype and dataspace bytes.
    ///
    /// # Panics
    ///
    /// Panics if the name is empty, contains a NUL, or is non-ASCII in version 2. Also panics if
    /// the name with its terminator, datatype, or dataspace is longer than [`u16::MAX`] bytes.
    #[track_caller]
    pub fn inline(
        version: UnpaddedVersion,
        name: &'a str,
        datatype: &'a [u8],
        dataspace: &'a [u8],
        data: &'a [u8],
    ) -> Self {
        Self::new(
            version,
            name,
            DatatypeField::Inline(datatype),
            dataspace,
            data,
        )
    }

    /// Describes an attribute that refers to a committed datatype at `address`.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`Self::inline`], using the reference length for the
    /// datatype field. Also panics if `widths.offset` is other than 2, 4, or 8 or if `address`
    /// cannot be written as a defined address at that width.
    #[track_caller]
    pub fn committed(
        version: UnpaddedVersion,
        name: &'a str,
        address: u64,
        widths: Widths,
        dataspace: &'a [u8],
        data: &'a [u8],
    ) -> Self {
        let undefined = match widths.offset {
            2 => u64::from(u16::MAX),
            4 => u64::from(u32::MAX),
            8 => u64::MAX,
            width => panic!("an offset width is 2, 4, or 8 bytes, not {width}"),
        };
        assert!(
            address < undefined,
            "a committed datatype has a defined address at its width"
        );
        let len = COMMITTED_REFERENCE_PREFIX_SIZE + widths.offset;
        Self::new(
            version,
            name,
            DatatypeField::Committed { address, len },
            dataspace,
            data,
        )
    }

    /// Writes the body and copies the supplied dataspace and element bytes.
    pub fn build(self) -> Vec<u8> {
        let Self {
            version,
            name,
            datatype,
            dataspace,
            data,
        } = self;
        let mut reference;
        let (datatype, flags) = match datatype {
            DatatypeField::Inline(datatype) => (datatype, Flags::NONE),
            DatatypeField::Committed { address, len } => {
                reference = datatype::committed_reference(address);
                reference.truncate(len);
                (reference.as_slice(), Flags::SHARED_DATATYPE)
            }
        };
        let attribute = Attribute::new(name, datatype, dataspace, data).flags(flags);
        match version {
            UnpaddedVersion::Two => attribute.build(),
            UnpaddedVersion::Three => attribute.character_set(u8::from(!name.is_ascii())).build(),
        }
    }

    #[track_caller]
    fn new(
        version: UnpaddedVersion,
        name: &'a str,
        datatype: DatatypeField<'a>,
        dataspace: &'a [u8],
        data: &'a [u8],
    ) -> Self {
        assert!(
            !name.is_empty() && !name.as_bytes().contains(&0),
            "the name must be nonempty and contain no NUL"
        );
        if matches!(version, UnpaddedVersion::Two) {
            assert!(name.is_ascii(), "a version 2 name must be ASCII");
        }
        let datatype_len = match datatype {
            DatatypeField::Inline(datatype) => datatype.len(),
            DatatypeField::Committed { len, .. } => len,
        };
        for len in [name.len() + 1, datatype_len, dataspace.len()] {
            assert!(
                u16::try_from(len).is_ok(),
                "field size {len} exceeds the two-byte limit"
            );
        }
        Self {
            version,
            name,
            datatype,
            dataspace,
            data,
        }
    }
}

#[derive(Debug)]
enum DatatypeField<'a> {
    Inline(&'a [u8]),
    Committed { address: u64, len: usize },
}

// The shared reference version and location type precede the address in
// "Data Object Header Messages" in version 4.0.
const COMMITTED_REFERENCE_PREFIX_SIZE: usize = 2;

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

/// Encodes compact Attribute Info with tracked creation order.
///
/// Writes `next` as the creation counter and undefined heap and index addresses. Includes
/// the creation-order index address when `indexed` is true.
/// Version 0 fields are defined in "The Attribute Info Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
pub fn tracked_info(next: u16, indexed: bool, widths: Widths) -> Vec<u8> {
    let flags = INFO_TRACKS_CREATION_ORDER
        | if indexed {
            INFO_INDEXES_CREATION_ORDER
        } else {
            0
        };
    let mut message = vec![INFO_VERSION, flags];
    message.extend_from_slice(&next.to_le_bytes());
    bytes::push_undefined_address(&mut message, widths.offset);
    bytes::push_undefined_address(&mut message, widths.offset);
    if indexed {
        bytes::push_undefined_address(&mut message, widths.offset);
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

// "The Attribute Info Message" of the format specification, version 4.0, defines the tracking bit.
const INFO_TRACKS_CREATION_ORDER: u8 = 0x01;
// "The Attribute Info Message" of the format specification, version 4.0, defines the indexing bit.
const INFO_INDEXES_CREATION_ORDER: u8 = 0x02;
