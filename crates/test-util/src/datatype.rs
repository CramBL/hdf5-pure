//! Datatype message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.

/// The eight bytes every datatype message begins with: its class and version
/// packed into one byte, the three class bit fields, and the element size.
pub fn header(class: Class, version: u8, bit_fields: [u8; 3], size: u32) -> Vec<u8> {
    let mut datatype = vec![(class.0 & 0x0F) | ((version & 0x0F) << 4)];
    datatype.extend_from_slice(&bit_fields);
    datatype.extend_from_slice(&size.to_le_bytes());
    datatype
}

/// A little-endian IEEE 754 double, laid out as the reference library writes
/// one.
pub fn f64_le() -> Vec<u8> {
    let mut datatype = header(Class::FLOATING_POINT, 1, [0x00, 0x00, 0x02], 8);
    // Bit offset(2) and bit precision(2), then the exponent's location and
    // size, the mantissa's, and the exponent bias.
    datatype.extend_from_slice(&0u16.to_le_bytes());
    datatype.extend_from_slice(&64u16.to_le_bytes());
    datatype.extend_from_slice(&[52, 11, 0, 52]);
    datatype.extend_from_slice(&1023u32.to_le_bytes());
    datatype
}

/// A fixed-length string of `size` bytes, null-padded and ASCII.
pub fn fixed_string(size: u32) -> Vec<u8> {
    header(Class::STRING, 1, [0x01, 0, 0], size)
}

/// A reference to a committed datatype, which stands where an encoding
/// usually is when the message's shared flag is set.
pub fn committed_reference(address: u64) -> Vec<u8> {
    let mut reference = vec![SHARED_MESSAGE_VERSION, COMMITTED];
    reference.extend_from_slice(&address.to_le_bytes());
    reference
}

/// What a datatype describes, which its first byte's low nibble holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Class(pub u8);

impl Class {
    pub const FIXED_POINT: Self = Self(0);
    pub const FLOATING_POINT: Self = Self(1);
    pub const TIME: Self = Self(2);
    pub const STRING: Self = Self(3);
    pub const BIT_FIELD: Self = Self(4);
    pub const OPAQUE: Self = Self(5);
    pub const COMPOUND: Self = Self(6);
    pub const REFERENCE: Self = Self(7);
    pub const ENUMERATED: Self = Self(8);
    pub const VARIABLE_LENGTH: Self = Self(9);
    pub const ARRAY: Self = Self(10);
}

/// The version of the shared message reference the reference library writes
/// for a committed datatype.
const SHARED_MESSAGE_VERSION: u8 = 2;

/// The shared message type whose location is an object header.
const COMMITTED: u8 = 2;
