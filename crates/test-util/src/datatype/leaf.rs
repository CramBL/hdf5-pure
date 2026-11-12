//! Independent atomic datatype-message bodies with selected versions and properties.

use super::Class;

/// A version from 1 through 5 for atomic datatype-message fixtures.
///
/// The version field is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum Version {
    /// Datatype message version 1.
    One = 1,
    /// Datatype message version 2.
    Two = 2,
    /// Datatype message version 3.
    Three = 3,
    /// Datatype message version 4.
    Four = 4,
    /// Datatype message version 5.
    Five = 5,
}

impl Version {
    /// Returns the version number for the message header.
    pub const fn get(self) -> u8 {
        self as u8
    }
}

/// The class properties of an independent atomic datatype fixture.
#[derive(Clone, Debug)]
pub enum Leaf {
    /// A signed big-endian integer with low and high padding bits set.
    Integer,
    /// An unsigned little-endian integer with low and high padding bits set.
    LittleInteger,
    /// A floating-point description with leading-one normalization and all padding bits set.
    Float(Order),
    /// A floating-point description with normalization code 0 and clear padding bits.
    UnnormalizedFloat(Order),
    /// A floating-point description with an implied leading mantissa bit and clear padding bits.
    ImpliedFloat(Order),
    /// A floating-point description with only its low padding bit set.
    LowPaddedFloat(Order),
    /// A floating-point description with only its high padding bit set.
    HighPaddedFloat(Order),
    /// A floating-point description with only its internal padding bit set.
    InternallyPaddedFloat(Order),
    /// A big-endian time description with 24 significant bits.
    Time,
    /// A little-endian time description with 24 significant bits.
    LittleTime,
    /// A seven-byte fixed-length string with selected metadata.
    String {
        /// The treatment of unused bytes.
        padding: Padding,
        /// The encoding of the string bytes.
        charset: Charset,
    },
    /// A big-endian bitfield with low and high padding bits set.
    Bitfield,
    /// A little-endian bitfield with low and high padding bits set.
    LittleBitfield,
    /// A three-byte opaque element with the selected logical tag.
    Opaque(OpaqueTag),
}

impl Leaf {
    /// Selects a seven-byte UTF-8 string with space padding.
    pub fn string() -> Self {
        Self::String {
            padding: Padding::SpacePad,
            charset: Charset::Utf8,
        }
    }

    /// Builds the message body with the selected version.
    pub fn bytes(&self, version: Version) -> Vec<u8> {
        let (class, flags, size, payload) = match self {
            Self::Integer => (Class::FIXED_POINT, [0x0f, 0, 0], 4, vec![3, 0, 25, 0]),
            Self::LittleInteger => (Class::FIXED_POINT, [0x06, 0, 0], 4, vec![3, 0, 25, 0]),
            Self::Float(order) => floating_body(*order, 0x1e),
            Self::UnnormalizedFloat(order) => floating_body(*order, 0),
            Self::ImpliedFloat(order) => floating_body(*order, 0x20),
            Self::LowPaddedFloat(order) => floating_body(*order, 0x12),
            Self::HighPaddedFloat(order) => floating_body(*order, 0x14),
            Self::InternallyPaddedFloat(order) => floating_body(*order, 0x18),
            Self::Time => (Class::TIME, [1, 0, 0], 4, vec![24, 0]),
            Self::LittleTime => (Class::TIME, [0, 0, 0], 4, vec![24, 0]),
            Self::String { padding, charset } => (
                Class::STRING,
                [(*padding as u8) | ((*charset as u8) << 4), 0, 0],
                7,
                Vec::new(),
            ),
            Self::Bitfield => (Class::BIT_FIELD, [7, 0, 0], 4, vec![3, 0, 17, 0]),
            Self::LittleBitfield => (Class::BIT_FIELD, [6, 0, 0], 4, vec![3, 0, 17, 0]),
            Self::Opaque(tag) => {
                let mut payload = tag.bytes();
                payload.resize(usize::from(tag.encoded_len()), 0);
                (Class::OPAQUE, [tag.encoded_len(), 0, 0], 3, payload)
            }
        };
        let mut bytes = super::header(class, version.get(), flags, size);
        bytes.extend(payload);
        bytes
    }
}

/// The byte order of a non-VAX floating-point fixture.
#[derive(Clone, Copy, Debug)]
pub enum Order {
    /// Little-endian order.
    Little,
    /// Big-endian order.
    Big,
}

impl Order {
    fn bits(self) -> u8 {
        match self {
            Self::Little => 0,
            Self::Big => 1,
        }
    }
}

/// A version from 3 through 5 for VAX floating-point fixtures.
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum VaxVersion {
    /// Datatype message version 3.
    Three = 3,
    /// Datatype message version 4.
    Four = 4,
    /// Datatype message version 5.
    Five = 5,
}

impl VaxVersion {
    /// Returns the version number for the message header.
    pub const fn get(self) -> u8 {
        self as u8
    }
}

/// Builds a VAX-ordered floating-point message at version 3, 4, or 5.
pub fn vax_bytes(version: VaxVersion) -> Vec<u8> {
    let mut bytes = Leaf::Float(Order::Little).bytes(Version::Three);
    bytes[0] = (version.get() << 4) | Class::FLOATING_POINT.0;
    bytes[1] |= VAX_ORDER;
    bytes
}

/// Both floating-point byte-order bits, defined as VAX in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// `H5O__dtype_encode_helper` requires version 3 or later (`H5Odtype.c`, HDF5 2.0.0).
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VAX_ORDER: u8 = 0x41;

fn floating_body(order: Order, flags: u8) -> (Class, [u8; 3], u32, Vec<u8>) {
    (
        Class::FLOATING_POINT,
        [order.bits() | flags, 63, 0],
        8,
        vec![3, 0, 61, 0, 52, 11, 3, 49, 0xff, 3, 0, 0],
    )
}

/// The stored string padding code.
///
/// String padding codes for datatype message versions 1 through 5 are defined in "The Datatype
/// Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum Padding {
    /// NUL-terminated content.
    NullTerminate = 0,
    /// NUL padding.
    NullPad = 1,
    /// Space padding.
    SpacePad = 2,
}

/// The stored string character-set code.
///
/// String character-set codes for datatype message versions 1 through 5 are defined in "The Datatype
/// Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum Charset {
    /// ASCII text.
    Ascii = 0,
    /// UTF-8 text.
    Utf8 = 1,
}

/// A logical opaque tag of up to 247 bytes.
#[derive(Clone, Copy, Debug)]
pub enum OpaqueTag {
    /// Eight logical bytes, requiring another alignment unit for the terminator.
    Aligned,
    /// An empty tag with no property bytes.
    Empty,
    /// The maximum writable logical length of 247 bytes.
    Maximum,
    /// Two logical bytes including a non-ASCII byte.
    NonAscii,
}

impl OpaqueTag {
    /// Returns the logical tag bytes without a terminator or padding.
    pub fn bytes(self) -> Vec<u8> {
        match self {
            Self::Empty => Vec::new(),
            Self::NonAscii => vec![0xff, b't'],
            Self::Aligned => vec![b't'; 8],
            Self::Maximum => vec![b't'; 247],
        }
    }

    /// Returns the aligned property length, including a terminator for a nonempty tag.
    pub fn encoded_len(self) -> u8 {
        match self {
            Self::Empty => 0,
            Self::NonAscii => 8,
            Self::Aligned => 16,
            Self::Maximum => 248,
        }
    }
}

/// An opaque property filled with tag bytes without a NUL terminator.
#[derive(Clone, Copy, Debug)]
pub enum FullOpaqueProperty {
    /// One eight-byte alignment unit.
    Eight,
    /// The largest aligned property length of 248 bytes.
    Maximum,
}

impl FullOpaqueProperty {
    /// Builds an opaque message with the selected version and full tag property.
    pub fn bytes(self, version: Version) -> Vec<u8> {
        let length = self.length();
        let mut bytes = super::header(Class::OPAQUE, version.get(), [length, 0, 0], 3);
        bytes.extend(vec![b't'; usize::from(length)]);
        bytes
    }

    /// Returns the property length in bytes.
    pub fn length(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Maximum => 248,
        }
    }
}
