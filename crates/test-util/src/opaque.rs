//! Independent datatype messages with opaque members and array bases.
//!
//! The layouts are defined in "The Datatype Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use crate::datatype;
use crate::datatype::Class;

/// A compound layout with an opaque member followed by a one-byte string member.
#[derive(Clone, Copy, Debug)]
pub enum CompoundLayout {
    /// Version 1 with a scalar opaque member.
    VersionOneScalar,
    /// Version 1 with a two-element opaque array in the legacy dimension block.
    VersionOneArray,
    /// Version 2 with padded member names and four-byte offsets.
    VersionTwo,
    /// Version 3 with a 255-byte element and one-byte offsets.
    VersionThreeOneByteLimit,
    /// Version 3 with a 256-byte element and two-byte offsets.
    VersionThreeTwoByteStart,
    /// Version 3 with a 65,536-byte element and three-byte offsets.
    VersionThreeThreeByteStart,
    /// Version 4 with unpadded member names and one-byte offsets.
    VersionFour,
    /// Version 5 with unpadded member names and one-byte offsets.
    VersionFive,
}

impl CompoundLayout {
    /// Builds the complete compound message with a four-byte opaque type tagged `b"BLOB"`.
    ///
    /// The first member starts at byte zero and the second at the element's final byte.
    pub fn build(self) -> Vec<u8> {
        let size = self.size();
        let mut bytes = datatype::header(Class::COMPOUND, self.version(), [2, 0, 0], size);
        let offset_width = self.offset_width();
        bytes.extend_from_slice(if self.has_padded_names() {
            b"blob\0\0\0\0"
        } else {
            b"blob\0"
        });
        bytes.extend_from_slice(&0u32.to_le_bytes()[..offset_width]);
        if self.has_dimensions() {
            bytes.extend_from_slice(&[u8::from(self.is_array()), 0, 0, 0]);
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            for dimension in [2u32, 0, 0, 0] {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&datatype::header(
            Class::OPAQUE,
            DATATYPE_VERSION_ONE,
            [8, 0, 0],
            4,
        ));
        bytes.extend_from_slice(b"BLOB\0\0\0\0");
        bytes.extend_from_slice(if self.has_padded_names() {
            b"next\0\0\0\0"
        } else {
            b"next\0"
        });
        bytes.extend_from_slice(&(size - 1).to_le_bytes()[..offset_width]);
        if self.has_dimensions() {
            bytes.extend_from_slice(&[0; COMPOUND_V1_DIMENSION_BLOCK_LEN]);
        }
        bytes.extend_from_slice(&datatype::fixed_string(1));
        bytes
    }

    /// Returns whether the first member uses the version 1 array dimension block.
    pub fn is_array(self) -> bool {
        matches!(self, Self::VersionOneArray)
    }

    fn version(self) -> u8 {
        match self {
            Self::VersionOneScalar | Self::VersionOneArray => DATATYPE_VERSION_ONE,
            Self::VersionTwo => DATATYPE_VERSION_TWO,
            Self::VersionThreeOneByteLimit
            | Self::VersionThreeTwoByteStart
            | Self::VersionThreeThreeByteStart => DATATYPE_VERSION_THREE,
            Self::VersionFour => DATATYPE_VERSION_FOUR,
            Self::VersionFive => DATATYPE_VERSION_FIVE,
        }
    }

    fn size(self) -> u32 {
        match self {
            Self::VersionOneScalar | Self::VersionTwo | Self::VersionFour | Self::VersionFive => 5,
            Self::VersionOneArray => 9,
            Self::VersionThreeOneByteLimit => 255,
            Self::VersionThreeTwoByteStart => 256,
            Self::VersionThreeThreeByteStart => 65_536,
        }
    }

    fn offset_width(self) -> usize {
        match self {
            Self::VersionOneScalar | Self::VersionOneArray | Self::VersionTwo => {
                LEGACY_COMPOUND_OFFSET_LEN
            }
            Self::VersionThreeOneByteLimit | Self::VersionFour | Self::VersionFive => 1,
            Self::VersionThreeTwoByteStart => 2,
            Self::VersionThreeThreeByteStart => 3,
        }
    }

    fn has_padded_names(self) -> bool {
        matches!(
            self,
            Self::VersionOneScalar | Self::VersionOneArray | Self::VersionTwo
        )
    }

    fn has_dimensions(self) -> bool {
        matches!(self, Self::VersionOneScalar | Self::VersionOneArray)
    }
}

/// The datatype message version for a two-element array of four-byte opaque values
/// tagged `b"BLOB"`.
#[derive(Clone, Copy, Debug)]
pub enum ArrayVersion {
    /// Version 2 with a dimension permutation field.
    Two,
    /// Version 3 with compact dimension metadata.
    Three,
    /// Version 4 with compact dimension metadata.
    Four,
    /// Version 5 with compact dimension metadata.
    Five,
}

impl ArrayVersion {
    /// Builds the complete array message with a version 1 opaque base.
    pub fn build(self) -> Vec<u8> {
        let mut bytes = datatype::header(Class::ARRAY, self.number(), [0; 3], 8);
        if matches!(self, Self::Two) {
            bytes.extend_from_slice(&[1, 0, 0, 0]);
        } else {
            bytes.push(1);
        }
        bytes.extend_from_slice(&2u32.to_le_bytes());
        if matches!(self, Self::Two) {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        bytes.extend_from_slice(&datatype::header(
            Class::OPAQUE,
            DATATYPE_VERSION_ONE,
            [8, 0, 0],
            4,
        ));
        bytes.extend_from_slice(b"BLOB\0\0\0\0");
        bytes
    }

    fn number(self) -> u8 {
        match self {
            Self::Two => DATATYPE_VERSION_TWO,
            Self::Three => DATATYPE_VERSION_THREE,
            Self::Four => DATATYPE_VERSION_FOUR,
            Self::Five => DATATYPE_VERSION_FIVE,
        }
    }
}

/// Datatype message version 1, with the legacy compound dimension block.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_ONE: u8 = 1;
/// Datatype message version 2, with a separate array class.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_TWO: u8 = 2;
/// Datatype message version 3, with compact compound and array properties.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_THREE: u8 = 3;
/// Datatype message version 4, with revised references.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_FOUR: u8 = 4;
/// Datatype message version 5, with the complex class.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION_FIVE: u8 = 5;
/// The member offset width in version 1 and 2 compound messages.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const LEGACY_COMPOUND_OFFSET_LEN: usize = 4;
/// The dimension block size in a version 1 compound member.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPOUND_V1_DIMENSION_BLOCK_LEN: usize = 28;
