//! Datatype message versions and the byte orders each permits for floating-point properties.

use hdf5_pure_core::{DatatypeByteOrder, FormatError};

use crate::datatype::header::FLOATING_POINT_CLASS;

/// A datatype message version from 1 through 5.
///
/// The version occupies the high four bits of the class-and-version byte, defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum DatatypeVersion {
    /// Datatype message version 1.
    One = DATATYPE_VERSION_ONE,
    /// Datatype message version 2.
    Two = DATATYPE_VERSION_TWO,
    /// Datatype message version 3.
    Three = DATATYPE_VERSION_THREE,
    /// Datatype message version 4.
    Four = DATATYPE_VERSION_FOUR,
    /// Datatype message version 5.
    Five = DATATYPE_VERSION_FIVE,
}

impl DatatypeVersion {
    /// Returns the version number stored in the message header.
    pub const fn get(self) -> u8 {
        self as u8
    }

    /// Returns this version if it meets the class layout's minimum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] if this version is below `minimum`.
    pub(super) fn for_class(self, class: u8, minimum: Self) -> Result<Self, FormatError> {
        if self < minimum {
            return Err(FormatError::InvalidDatatypeVersion {
                class,
                version: self.get(),
            });
        }
        Ok(self)
    }

    /// Checks the child's version against its containing message's version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DatatypeChildVersion`] if the child's version is higher.
    pub(super) fn check_child(self, class: u8, child: Self) -> Result<(), FormatError> {
        if child > self {
            return Err(FormatError::DatatypeChildVersion {
                class,
                version: self.get(),
                child_version: child.get(),
            });
        }
        Ok(())
    }
}

impl TryFrom<u8> for DatatypeVersion {
    type Error = FormatError;

    /// Converts a stored version number to a supported message version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeMessageVersion`] if `value` is outside `1..=5`.
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            DATATYPE_VERSION_ONE => Ok(Self::One),
            DATATYPE_VERSION_TWO => Ok(Self::Two),
            DATATYPE_VERSION_THREE => Ok(Self::Three),
            DATATYPE_VERSION_FOUR => Ok(Self::Four),
            DATATYPE_VERSION_FIVE => Ok(Self::Five),
            value => Err(FormatError::InvalidDatatypeMessageVersion(value)),
        }
    }
}

/// A floating-point byte order paired with a compatible datatype message version.
///
/// Little-endian and big-endian order support versions 1 through 5. VAX order requires
/// version 3 or later. The byte-order table in "The Datatype Message" of the [format
/// specification, version 4.0][spec] defines bits 6 and 0 together. Its generic reserved-bit
/// row also lists bit 6, which the byte-order table uses for VAX.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatingPointByteOrder {
    version: DatatypeVersion,
    order: DatatypeByteOrder,
}

impl FloatingPointByteOrder {
    /// Pairs `order` with a compatible message version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] for VAX order before version 3.
    pub fn new(version: DatatypeVersion, order: DatatypeByteOrder) -> Result<Self, FormatError> {
        if order == DatatypeByteOrder::Vax {
            version.for_class(FLOATING_POINT_CLASS, DatatypeVersion::Three)?;
        }
        Ok(Self { version, order })
    }

    /// Parses the byte order and reserved bits from a floating-point class bit field.
    ///
    /// Bits 6 and 0 select little-endian (`0x00`), big-endian (`0x01`), or VAX (`0x41`) order.
    /// The parser checks reserved bits 7 and 16 through 23. Padding, normalization, and sign
    /// location are separate properties.
    ///
    /// The parser rejects bit 6 in versions 1 and 2. `H5O__dtype_decode_helper` ignores that bit
    /// in those versions (`H5Odtype.c`, HDF5 2.0.0). `H5O__dtype_encode_helper` writes VAX as
    /// `0x41` with a minimum version of 3 in the same release.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] if a reserved class bit is set,
    /// [`FormatError::InvalidByteOrder`] for the reserved order pattern `0x40`, and
    /// [`FormatError::InvalidDatatypeVersion`] for VAX order before version 3.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeByteOrder, FormatError};
    /// use hdf5_pure_format::__private::{DatatypeVersion, FloatingPointByteOrder};
    ///
    /// let order = FloatingPointByteOrder::parse(DatatypeVersion::Three, [0x41, 31, 0])?;
    /// assert_eq!(order.order(), DatatypeByteOrder::Vax);
    /// assert_eq!(order.bits(), 0x41);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn parse(version: DatatypeVersion, flags: [u8; 3]) -> Result<Self, FormatError> {
        let bits = u32::from(flags[0]) | (u32::from(flags[1]) << 8) | (u32::from(flags[2]) << 16);
        if bits & FLOAT_RESERVED_MASK != 0 {
            return Err(FormatError::InvalidDatatypeFlags {
                class: FLOATING_POINT_CLASS,
                version: version.get(),
                flags: bits,
            });
        }
        let order = flags[0] & (FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT);
        let order = match order {
            0 => DatatypeByteOrder::LittleEndian,
            FLOAT_BYTE_ORDER_LOW_BIT => DatatypeByteOrder::BigEndian,
            FLOAT_BYTE_ORDER_HIGH_BIT => return Err(FormatError::InvalidByteOrder(order)),
            _ => DatatypeByteOrder::Vax,
        };
        Self::new(version, order)
    }

    /// Returns the message version associated with this byte order.
    pub const fn version(self) -> DatatypeVersion {
        self.version
    }

    /// Returns the decoded byte order.
    pub const fn order(self) -> DatatypeByteOrder {
        self.order
    }

    /// Returns bits 6 and 0 for this byte order, with all other bits clear.
    pub const fn bits(self) -> u8 {
        Self::bits_for(self.order)
    }

    /// Encodes the byte order in bits 6 and 0. The caller selects a compatible version.
    pub(super) const fn bits_for(order: DatatypeByteOrder) -> u8 {
        match order {
            DatatypeByteOrder::LittleEndian => 0,
            DatatypeByteOrder::BigEndian => FLOAT_BYTE_ORDER_LOW_BIT,
            DatatypeByteOrder::Vax => FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT,
        }
    }
}

// The format specification, version 4.0, defines these version codes in
// "The Datatype Message".
pub(in crate::datatype) const DATATYPE_VERSION_ONE: u8 = 1;
pub(in crate::datatype) const DATATYPE_VERSION_TWO: u8 = 2;
pub(in crate::datatype) const DATATYPE_VERSION_THREE: u8 = 3;
/// Datatype message version 4, defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_FOUR: u8 = 4;
pub(in crate::datatype) const DATATYPE_VERSION_FIVE: u8 = 5;

/// The mask for reserved bits 7 and 16 through 23 of a floating-point class bit field.
///
/// The bits for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_RESERVED_MASK: u32 = 0xFF_0080;

// The floating-point byte-order table in "The Datatype Message" of the format
// specification, version 4.0, defines these bits together. See `FloatingPointByteOrder`
// for how the table's use of bit 6 relates to the reserved-bit row.

/// Bit 0 of a floating-point class bit field, the low bit of its byte order.
pub(in crate::datatype) const FLOAT_BYTE_ORDER_LOW_BIT: u8 = 0x01;

/// Bit 6 of a floating-point class bit field, the high bit of its byte order.
pub(in crate::datatype) const FLOAT_BYTE_ORDER_HIGH_BIT: u8 = 0x40;
