//! Independent native complex datatype messages and contiguous dataset images.

use crate::attribute::Attribute;
use crate::data_layout;
use crate::dataspace;
use crate::datatype::{self, Class};
use crate::image::Image;
use crate::link_message::HardLink;
use crate::object_header::v2::Header;
use crate::object_header::{Message, MessageType};
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// An IEEE floating-point component layout for complex fixtures.
#[derive(Clone, Copy, Debug)]
pub enum Component {
    /// A big-endian IEEE binary32 component.
    F32Be,
    /// A little-endian IEEE binary32 component.
    F32Le,
    /// A big-endian IEEE binary64 component.
    F64Be,
    /// A little-endian IEEE binary64 component.
    F64Le,
}

impl Component {
    /// Returns the byte width of one real or imaginary component.
    pub fn size(self) -> u32 {
        match self {
            Self::F32Le | Self::F32Be => 4,
            Self::F64Le | Self::F64Be => 8,
        }
    }

    /// Returns whether component values are encoded most-significant byte first.
    pub fn big_endian(self) -> bool {
        matches!(self, Self::F32Be | Self::F64Be)
    }

    /// Returns two encoded complex values, `(1.25, -2.5)` and `(3.5, 4.25)`.
    pub fn values(self) -> Vec<u8> {
        match self {
            Self::F32Le => [1.25f32, -2.5, 3.5, 4.25]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            Self::F32Be => [1.25f32, -2.5, 3.5, 4.25]
                .into_iter()
                .flat_map(f32::to_be_bytes)
                .collect(),
            Self::F64Le => [1.25f64, -2.5, 3.5, 4.25]
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect(),
            Self::F64Be => [1.25f64, -2.5, 3.5, 4.25]
                .into_iter()
                .flat_map(f64::to_be_bytes)
                .collect(),
        }
    }

    fn float_message(self) -> Vec<u8> {
        let (sign, precision, exponent, bias) = match self {
            Self::F32Le | Self::F32Be => (31, 32u16, [23, 8, 0, 23], 127u32),
            Self::F64Le | Self::F64Be => (63, 64u16, [52, 11, 0, 52], 1023u32),
        };
        let mut encoded = datatype::header(
            Class::FLOATING_POINT,
            FLOAT_VERSION,
            [IMPLIED_NORMALIZATION | u8::from(self.big_endian()), sign, 0],
            self.size(),
        );
        encoded.extend_from_slice(&0u16.to_le_bytes());
        encoded.extend_from_slice(&precision.to_le_bytes());
        encoded.extend_from_slice(&exponent);
        encoded.extend_from_slice(&bias.to_le_bytes());
        encoded
    }
}

/// A datatype shape for the fixture's two complex values.
#[derive(Clone, Copy, Debug)]
pub enum Container {
    /// One array element containing two native complex values.
    Array,
    /// Two compound elements, each with a native complex member named `value`.
    Compound,
    /// Two compound elements with floating-point members named `real` and `imag`.
    MatlabCompound,
    /// Two native complex elements.
    Native,
}

impl Container {
    /// Returns the number of dataset elements encoded by [`Self::file`].
    pub fn count(self) -> u64 {
        match self {
            Self::Array => 1,
            Self::Native | Self::Compound | Self::MatlabCompound => 2,
        }
    }

    /// Builds a datatype message for the selected component layout.
    pub fn message(self, component: Component) -> Vec<u8> {
        let size = 2 * component.size();
        let mut native =
            datatype::header(COMPLEX_CLASS, COMPLEX_VERSION, [HOMOGENEOUS, 0, 0], size);
        native.extend_from_slice(&component.float_message());
        match self {
            Self::Native => native,
            Self::Array => {
                let mut encoded = datatype::header(Class::ARRAY, COMPLEX_VERSION, [0; 3], size * 2);
                encoded.push(1);
                encoded.extend_from_slice(&2u32.to_le_bytes());
                encoded.extend_from_slice(&native);
                encoded
            }
            Self::Compound => {
                let mut encoded =
                    datatype::header(Class::COMPOUND, COMPLEX_VERSION, [1, 0, 0], size);
                encoded.extend_from_slice(b"value\0\0");
                encoded.extend_from_slice(&native);
                encoded
            }
            Self::MatlabCompound => {
                let mut encoded =
                    datatype::header(Class::COMPOUND, COMPOUND_VERSION, [2, 0, 0], size);
                encoded.extend_from_slice(b"real\0\0");
                encoded.extend_from_slice(&component.float_message());
                encoded.extend_from_slice(b"imag\0");
                encoded.push(u8::try_from(component.size()).unwrap());
                encoded.extend_from_slice(&component.float_message());
                encoded
            }
        }
    }

    /// Builds a contiguous `values` dataset with matching dataset and root attributes.
    ///
    /// The dataset and both attributes contain the bytes returned by [`Component::values`].
    pub fn file(self, component: Component) -> Vec<u8> {
        let mut image = Image::starting_with(&[0; SUPERBLOCK_SIZE]);
        let raw = component.values();
        let data_address = u64::try_from(image.append(&raw)).unwrap();
        let datatype = self.message(component);
        let dataspace = dataspace::v2(
            1,
            dataspace::Flags::NONE,
            dataspace::Kind::SIMPLE,
            &[self.count()],
            None,
        );
        let attr = Attribute::new("values", &datatype, &dataspace, &raw).build();
        let dataset = u64::try_from(
            image.append(
                &Header::new()
                    .messages([
                        Message::new(MessageType::DATASPACE, &dataspace),
                        Message::new(MessageType::DATATYPE, &datatype),
                        Message::new(
                            MessageType::DATA_LAYOUT,
                            &data_layout::contiguous(
                                data_address,
                                u64::try_from(raw.len()).unwrap(),
                            ),
                        ),
                        Message::new(MessageType::ATTRIBUTE, &attr),
                    ])
                    .build(),
            ),
        )
        .unwrap();
        let mut link_info = vec![LINK_INFO_VERSION, 0];
        link_info.extend_from_slice(&u64::MAX.to_le_bytes());
        link_info.extend_from_slice(&u64::MAX.to_le_bytes());
        let root = u64::try_from(
            image.append(
                &Header::new()
                    .messages([
                        Message::new(MessageType::LINK_INFO, &link_info),
                        Message::new(MessageType::GROUP_INFO, &[GROUP_INFO_VERSION, 0]),
                        Message::new(
                            MessageType::LINK,
                            &HardLink::new("values", dataset).build(Widths::EIGHT),
                        ),
                        Message::new(MessageType::ATTRIBUTE, &attr),
                    ])
                    .build(),
            ),
        )
        .unwrap();
        let superblock = Superblock::new(Widths::EIGHT)
            .root_header_address(root)
            .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
            .build();
        image.place(0, &superblock);
        image.build()
    }
}

/// The native complex class code in datatype message version 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_CLASS: Class = Class(11);
/// The datatype message version for native complex values.
///
/// Version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_VERSION: u8 = 5;
/// The datatype message version used for the floating-point components.
///
/// Version 1 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_VERSION: u8 = 1;
/// The datatype message version used for the compound complex convention.
///
/// Version 3 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPOUND_VERSION: u8 = 3;
/// The flag for a shared floating-point component type in a version 5 complex datatype.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const HOMOGENEOUS: u8 = 0x01;
/// The normalization bits for an implied leading mantissa bit in a version 1 floating-point type.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const IMPLIED_NORMALIZATION: u8 = 0x20;

/// The byte length of a version 2 superblock with eight-byte addresses and lengths.
///
/// Defined in "Format Signature and Superblock" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
const SUPERBLOCK_SIZE: usize = 48;

/// The version of the Link Info message.
///
/// Version 0 is defined in "The Link Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
const LINK_INFO_VERSION: u8 = 0;
/// The version of the Group Info message.
///
/// Version 0 is defined in "The Group Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
const GROUP_INFO_VERSION: u8 = 0;
