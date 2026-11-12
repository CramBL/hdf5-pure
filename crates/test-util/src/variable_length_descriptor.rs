//! Independent byte construction for variable-length value descriptors.

/// A stored collection address whose integer type fixes its encoded width.
#[derive(Clone, Copy, Debug)]
pub enum Address {
    /// A two-byte address.
    Two(u16),
    /// A four-byte address.
    Four(u32),
    /// An eight-byte address.
    Eight(u64),
}

impl Address {
    /// Encodes `count`, this address, and `index` as a little-endian descriptor.
    ///
    /// `count` is the number of base elements, independent of the payload's byte length.
    ///
    /// # Examples
    ///
    /// ```
    /// use test_util::variable_length_descriptor::Address;
    ///
    /// assert_eq!(Address::Two(0x1234).descriptor(2, 7), [2, 0, 0, 0, 0x34, 0x12, 7, 0, 0, 0]);
    /// ```
    pub fn descriptor(self, count: u32, index: u32) -> Vec<u8> {
        let mut bytes = count.to_le_bytes().to_vec();
        match self {
            Self::Two(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            Self::Four(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            Self::Eight(value) => bytes.extend_from_slice(&value.to_le_bytes()),
        }
        bytes.extend_from_slice(&index.to_le_bytes());
        bytes
    }

    /// Returns the stored integer without changing its value.
    pub fn get(self) -> u64 {
        match self {
            Self::Two(value) => u64::from(value),
            Self::Four(value) => u64::from(value),
            Self::Eight(value) => value,
        }
    }

    /// Returns the number of bytes used to encode the address.
    pub fn width(self) -> u8 {
        match self {
            Self::Two(_) => TWO_BYTE_ADDRESS_WIDTH,
            Self::Four(_) => FOUR_BYTE_ADDRESS_WIDTH,
            Self::Eight(_) => EIGHT_BYTE_ADDRESS_WIDTH,
        }
    }
}

// Collection addresses use this width in "Global Heap" of the format specification, version 4.0.
const TWO_BYTE_ADDRESS_WIDTH: u8 = 2;

// Collection addresses use this width in "Global Heap" of the format specification, version 4.0.
const FOUR_BYTE_ADDRESS_WIDTH: u8 = 4;

// Collection addresses use this width in "Global Heap" of the format specification, version 4.0.
const EIGHT_BYTE_ADDRESS_WIDTH: u8 = 8;
