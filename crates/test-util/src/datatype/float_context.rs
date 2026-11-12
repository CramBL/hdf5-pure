//! Direct and recursive framing for eight-byte floating-point parent descriptions.
//!
//! Legacy and modern property layouts are defined in "The Datatype Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use super::Class;
use super::containers;
use super::containers::CompoundField;
use super::containers::MemberOffset;
use super::leaf::Version;

/// The framing around an unchecked eight-byte parent description.
#[derive(Clone, Copy, Debug)]
pub enum Context {
    /// A version 3 array with one element.
    ArrayThree,
    /// A version 2 array with one element and identity permutation.
    ArrayTwo,
    /// A version 5 homogeneous rectangular complex description.
    Complex,
    /// A version 1 compound with one scalar member.
    CompoundOne,
    /// A version 3 compound with one member at offset zero.
    CompoundThree,
    /// A version 2 compound with one member at offset zero.
    CompoundTwo,
    /// The parent description without a containing description.
    Direct,
    /// A version 1 enumeration with one zero-valued member.
    EnumerationOne,
    /// A version 3 enumeration with one zero-valued member.
    EnumerationThree,
    /// A version 2 enumeration with one zero-valued member.
    EnumerationTwo,
    /// A version 5 sequence with an eight-byte heap address.
    Sequence,
}

impl Context {
    /// Copies `parent` into the selected framing.
    ///
    /// Accepts unchecked parent bytes. Container sizes assume an eight-byte element.
    /// The sequence descriptor occupies 16 bytes, including an eight-byte heap address.
    /// Property layouts are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Examples
    ///
    /// ```
    /// use test_util::datatype::{float_context::Context, leaf::{Leaf, Order, Version}};
    ///
    /// let parent = Leaf::Float(Order::Big).bytes(Version::One);
    /// let bytes = Context::ArrayTwo.bytes(&parent);
    /// assert_eq!(&bytes[..8], &[0x2a, 0, 0, 0, 8, 0, 0, 0]);
    /// assert_eq!(&bytes[8..20], &[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    /// assert_eq!(&bytes[20..], parent);
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    pub fn bytes(self, parent: &[u8]) -> Vec<u8> {
        match self {
            Self::Direct => parent.to_vec(),
            Self::CompoundOne => crate::legacy_compound::raw_member(8, 0, 0, 0, [0; 4], parent),
            Self::CompoundTwo => {
                let mut bytes = super::header(Class::COMPOUND, Version::Two.get(), [1, 0, 0], 8);
                bytes.extend_from_slice(b"x\0\0\0\0\0\0\0");
                bytes.extend_from_slice(&0u32.to_le_bytes());
                bytes.extend_from_slice(parent);
                bytes
            }
            Self::CompoundThree => containers::compound(
                Version::Three.get(),
                [1, 0, 0],
                8,
                MemberOffset::One,
                &[CompoundField {
                    name: b"x",
                    offset: 0,
                    datatype: parent,
                }],
            ),
            Self::EnumerationOne | Self::EnumerationTwo | Self::EnumerationThree => {
                let version = match self {
                    Self::EnumerationOne => Version::One,
                    Self::EnumerationTwo => Version::Two,
                    _ => Version::Three,
                };
                let mut bytes = super::header(Class::ENUMERATED, version.get(), [1, 0, 0], 8);
                bytes.extend_from_slice(parent);
                if matches!(self, Self::EnumerationThree) {
                    bytes.extend_from_slice(b"x\0");
                } else {
                    bytes.extend_from_slice(b"x\0\0\0\0\0\0\0");
                }
                bytes.extend_from_slice(&[0; 8]);
                bytes
            }
            Self::ArrayTwo => {
                let mut bytes = super::header(Class::ARRAY, Version::Two.get(), [0; 3], 8);
                bytes.extend_from_slice(&[1, 0, 0, 0]);
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&0u32.to_le_bytes());
                bytes.extend_from_slice(parent);
                bytes
            }
            Self::ArrayThree => super::array(Version::Three.get(), [0; 3], 8, 1, &[1], parent),
            Self::Sequence => super::variable_length(Version::Five.get(), [0; 3], 16, parent),
            Self::Complex => super::complex(Version::Five.get(), COMPLEX_HOMOGENEOUS, 16, parent),
        }
    }
}

/// The homogeneous rectangular flags for datatype message version 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_HOMOGENEOUS: [u8; 3] = [1, 0, 0];
