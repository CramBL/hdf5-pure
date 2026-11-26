use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroU32;

use super::complex::ComplexType;
use super::compound_member::CompoundMember;
use super::enumeration::EnumerationType;
use super::layout::{BitFieldType, FixedPointType, FloatingPointType, TimeType};
use super::reference::ReferenceType;
use super::string::{CharacterSet, StringPadding};
use super::variable_length::VariableLengthType;
use crate::FormatError;

/// The element datatype described by an HDF5 datatype message.
///
/// [`type_size`](Self::type_size) gives the element size in bytes. The fixed-point,
/// floating-point, time, and bit-field payloads contain checked sizes and bit ranges.
/// Metadata for a numeric type wider than eight bytes or with an unsupported floating-point
/// encoding is available even when numeric value conversions cannot decode its values.
///
/// "The Datatype Message" of the [format specification, version 4.0][spec] defines datatype
/// message versions 1 through 5. [`Complex`](Self::Complex) represents the homogeneous
/// rectangular form of class 11.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Datatype {
    /// Fixed-point integers (class 0).
    FixedPoint(FixedPointType),
    /// Floating-point numbers (class 1).
    FloatingPoint(FloatingPointType),
    /// Time values (class 2).
    Time(TimeType),
    /// Fixed-length strings (class 3).
    String {
        /// Width of one string, in bytes.
        size: u32,
        /// Treatment of unused bytes.
        padding: StringPadding,
        /// Encoding of the string bytes.
        charset: CharacterSet,
    },
    /// Bit fields (class 4).
    BitField(BitFieldType),
    /// Opaque data (class 5).
    Opaque {
        /// Size of one element, in bytes.
        size: u32,
        /// Application-defined logical tag bytes, excluding the terminator and padding.
        ///
        /// An empty tag denotes an untagged type. Writing accepts at most 247 bytes and rejects
        /// embedded NULs. Non-ASCII bytes are preserved. Reading also accepts an aligned property
        /// filled entirely by tag bytes. `H5O__dtype_encode_helper` writes that form for tags whose
        /// logical length is a multiple of eight (H5Odtype.c, HDF5 2.2.0). A 248-byte tag can be read
        /// but exceeds the writing limit.
        tag: Vec<u8>,
    },
    /// Compound elements with named fields (class 6).
    Compound {
        /// Total size of one element, in bytes.
        size: u32,
        /// Named fields within an element.
        members: Vec<CompoundMember>,
    },
    /// Object or dataset region references (class 7).
    Reference {
        /// Size of one reference, in bytes.
        size: u32,
        /// Kind of object or region referenced.
        ref_type: ReferenceType,
    },
    /// Enumerations with value widths checked by [`EnumerationType`] (class 8).
    Enumeration(EnumerationType),
    /// Variable-length sequences and strings (class 9).
    VariableLength(VariableLengthType),
    /// Fixed-size arrays (class 10).
    Array {
        /// Datatype of one array element.
        base_type: Box<Datatype>,
        /// Extent of each array dimension.
        dimensions: Vec<u32>,
    },
    /// Homogeneous rectangular complex numbers (class 11).
    Complex(ComplexType),
}

impl Datatype {
    /// Returns the size in bytes of one element.
    ///
    /// Uses 16 bytes for a variable-length descriptor, the size with an eight-byte file offset.
    /// For arrays, multiplies the element size by the dimension extents using saturating
    /// arithmetic.
    pub fn type_size(&self) -> u32 {
        match self {
            Datatype::FixedPoint(datatype) => datatype.size().get(),
            Datatype::FloatingPoint(datatype) => datatype.size().get(),
            Datatype::Time(datatype) => datatype.size().get(),
            Datatype::String { size, .. } => *size,
            Datatype::BitField(datatype) => datatype.size().get(),
            Datatype::Opaque { size, .. } => *size,
            Datatype::Compound { size, .. } => *size,
            Datatype::Reference { size, .. } => *size,
            Datatype::Enumeration(datatype) => datatype.size().get(),
            Datatype::VariableLength(_) => 16,
            Datatype::Complex(datatype) => datatype.size().get(),
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                let elem_count: u32 = dimensions
                    .iter()
                    .copied()
                    .fold(1u32, |a, b| a.saturating_mul(b));
                base_type.type_size().saturating_mul(elem_count)
            }
        }
    }

    /// Returns the class code stored in the low four bits of a datatype message's first byte.
    ///
    /// "The Datatype Message" of the format specification, version 4.0, defines the class field.
    fn class_code(&self) -> u8 {
        match self {
            Datatype::FixedPoint(..) => 0,
            Datatype::FloatingPoint(..) => 1,
            Datatype::Time(..) => 2,
            Datatype::String { .. } => 3,
            Datatype::BitField(..) => 4,
            Datatype::Opaque { .. } => 5,
            Datatype::Compound { .. } => 6,
            Datatype::Reference { .. } => 7,
            Datatype::Enumeration(_) => 8,
            Datatype::VariableLength(_) => 9,
            Datatype::Array { .. } => 10,
            Datatype::Complex(_) => COMPLEX_CLASS,
        }
    }

    /// Returns the size in bytes of one element.
    ///
    /// Uses the same size as [`Self::type_size`]. The [`NonZeroU32`] result can serve as a divisor.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if the element type occupies
    /// zero bytes.
    pub fn element_size(&self) -> Result<NonZeroU32, FormatError> {
        match self {
            Self::FixedPoint(datatype) => Ok(datatype.size()),
            Self::FloatingPoint(datatype) => Ok(datatype.size()),
            Self::Time(datatype) => Ok(datatype.size()),
            Self::BitField(datatype) => Ok(datatype.size()),
            Self::Enumeration(datatype) => Ok(datatype.size()),
            Self::Complex(datatype) => Ok(datatype.size()),
            _ => NonZeroU32::new(self.type_size()).ok_or(FormatError::ZeroSizedDatatype {
                class: self.class_code(),
            }),
        }
    }
}

/// The native complex datatype class code.
///
/// The code for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_CLASS: u8 = 0x0B;
