//! The fixed header shared by both datatype parsers.
//!
//! The common header for datatype message versions 1 through 5 is defined in "The Datatype
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

#[cfg(test)]
use alloc::vec;
#[cfg(test)]
use alloc::vec::Vec;

#[cfg(test)]
use byteorder::{ByteOrder, LittleEndian};

use super::message::DatatypeVersion;
use crate::bytes::ByteCursor;
use crate::error::FormatError;

/// An eight-byte datatype header with a checked class and version.
pub(super) struct Header {
    pub(super) class: DatatypeClass,
    pub(super) version: DatatypeVersion,
    /// The three class bit field bytes in stored order.
    pub(super) flags: [u8; 3],
    /// The declared byte size of one element.
    pub(super) size: u32,
}

impl Header {
    /// Reads the complete header before checking the class or version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] for an incomplete header, with the header's end
    /// offset as `expected` and the input length as `available`.
    ///
    /// For [`Context::Description`], checks the version before the class, returning
    /// [`FormatError::InvalidDatatypeMessageVersion`] for a version outside `1..=5` or
    /// [`FormatError::InvalidDatatypeClass`] for an unsupported class. For
    /// [`Context::ComplexParent`], checks the class first, returning
    /// [`FormatError::InvalidComplexParent`] for a class other than floating point or
    /// [`FormatError::InvalidDatatypeVersion`] for a version outside `1..=5`.
    pub(super) fn parse(
        cursor: &mut ByteCursor<'_>,
        context: Context,
    ) -> Result<Self, FormatError> {
        let [
            class_and_version,
            low,
            middle,
            high,
            size0,
            size1,
            size2,
            size3,
        ] = cursor.array::<DATATYPE_HEADER_SIZE>()?;
        let class = class_and_version & DATATYPE_CLASS_MASK;
        if context == Context::ComplexParent && class != FLOATING_POINT_CLASS {
            return Err(FormatError::InvalidComplexParent { class });
        }
        let version = class_and_version >> DATATYPE_VERSION_SHIFT;
        let version = DatatypeVersion::try_from(version).map_err(|error| match context {
            Context::ComplexParent => FormatError::InvalidDatatypeVersion { class, version },
            Context::Description => error,
        })?;
        Ok(Self {
            class: DatatypeClass::parse(class)?,
            version,
            flags: [low, middle, high],
            size: u32::from_le_bytes([size0, size1, size2, size3]),
        })
    }

    /// Interprets the class bit field bytes in little-endian order.
    pub(super) fn flag_bits(&self) -> u32 {
        self.fields().flag_bits()
    }

    /// Copies the fields used after class dispatch.
    pub(super) fn fields(&self) -> HeaderFields {
        let &Self {
            class: _,
            version,
            flags,
            size,
        } = self;
        HeaderFields {
            version,
            flags,
            size,
        }
    }
}

/// The version, class bit field, and element size passed to a property decoder.
#[derive(Clone, Copy)]
pub(super) struct HeaderFields {
    pub(super) version: DatatypeVersion,
    pub(super) flags: [u8; 3],
    pub(super) size: u32,
}

impl HeaderFields {
    /// Checks the class bit field against a reserved-bit mask.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] with the complete class bit field if any
    /// bit in `mask` is set.
    pub(super) fn reject_reserved_flags(&self, class: u8, mask: u32) -> Result<(), FormatError> {
        let flags = self.flag_bits();
        if flags & mask != 0 {
            return Err(FormatError::InvalidDatatypeFlags {
                class,
                version: self.version.get(),
                flags,
            });
        }
        Ok(())
    }

    fn flag_bits(&self) -> u32 {
        u32::from_le_bytes([self.flags[0], self.flags[1], self.flags[2], 0])
    }
}

/// The position of a datatype header in a description.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Context {
    /// The floating-point description within a complex datatype.
    ComplexParent,
    /// An outer description or a child of another datatype class.
    Description,
}

impl Context {
    /// Returns [`FormatError::InvalidComplexParentFlags`] with the version and class bit field
    /// for [`Self::ComplexParent`], or the supplied `error` for [`Self::Description`].
    pub(super) fn map_floating_point_flags_error(
        self,
        fields: HeaderFields,
        error: FormatError,
    ) -> FormatError {
        match self {
            Self::ComplexParent => FormatError::InvalidComplexParentFlags {
                version: fields.version.get(),
                flags: fields.flag_bits(),
            },
            Self::Description => error,
        }
    }
}

/// A supported class from the low four bits of the header's first byte.
#[derive(Clone, Copy)]
pub(super) enum DatatypeClass {
    Atomic(AtomicClass),
    Compound,
    Reference,
    Enumeration,
    VariableLength,
    Array,
    Complex,
}

impl DatatypeClass {
    /// Parses a class code from the common header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeClass`] for an unknown code.
    fn parse(class: u8) -> Result<Self, FormatError> {
        Ok(match class {
            FIXED_POINT_CLASS => Self::Atomic(AtomicClass::FixedPoint),
            FLOATING_POINT_CLASS => Self::Atomic(AtomicClass::FloatingPoint),
            TIME_CLASS => Self::Atomic(AtomicClass::Time),
            STRING_CLASS => Self::Atomic(AtomicClass::String),
            BIT_FIELD_CLASS => Self::Atomic(AtomicClass::BitField),
            OPAQUE_CLASS => Self::Atomic(AtomicClass::Opaque),
            COMPOUND_CLASS => Self::Compound,
            REFERENCE_CLASS => Self::Reference,
            ENUMERATION_CLASS => Self::Enumeration,
            VARIABLE_LENGTH_CLASS => Self::VariableLength,
            ARRAY_CLASS => Self::Array,
            COMPLEX_CLASS => Self::Complex,
            class => return Err(FormatError::InvalidDatatypeClass(class)),
        })
    }

    /// Returns the class code as stored in the header.
    pub(super) const fn code(self) -> u8 {
        match self {
            Self::Atomic(class) => class as u8,
            Self::Compound => COMPOUND_CLASS,
            Self::Reference => REFERENCE_CLASS,
            Self::Enumeration => ENUMERATION_CLASS,
            Self::VariableLength => VARIABLE_LENGTH_CLASS,
            Self::Array => ARRAY_CLASS,
            Self::Complex => COMPLEX_CLASS,
        }
    }
}

/// The datatype classes with atomic property layouts.
#[derive(Clone, Copy)]
#[repr(u8)]
pub(super) enum AtomicClass {
    FixedPoint = FIXED_POINT_CLASS,
    FloatingPoint = FLOATING_POINT_CLASS,
    Time = TIME_CLASS,
    String = STRING_CLASS,
    BitField = BIT_FIELD_CLASS,
    Opaque = OPAQUE_CLASS,
}

/// The byte length of the common datatype message header.
///
/// The header for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const DATATYPE_HEADER_SIZE: usize = 8;
/// The mask for the class code in the common datatype message header.
///
/// The class field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const DATATYPE_CLASS_MASK: u8 = 0x0F;
/// The bit position of the version in the common datatype message header.
///
/// The version field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const DATATYPE_VERSION_SHIFT: u8 = 4;

// The format specification, version 4.0, defines these class codes
// in "The Datatype Message".
pub(super) const FIXED_POINT_CLASS: u8 = 0;
pub(super) const FLOATING_POINT_CLASS: u8 = 1;
pub(super) const TIME_CLASS: u8 = 2;
pub(super) const STRING_CLASS: u8 = 3;
pub(super) const BIT_FIELD_CLASS: u8 = 4;
pub(super) const OPAQUE_CLASS: u8 = 5;
pub(super) const COMPOUND_CLASS: u8 = 6;
pub(super) const REFERENCE_CLASS: u8 = 7;
pub(super) const ENUMERATION_CLASS: u8 = 8;
pub(super) const VARIABLE_LENGTH_CLASS: u8 = 9;
pub(super) const ARRAY_CLASS: u8 = 10;
pub(super) const COMPLEX_CLASS: u8 = 11;

/// Writes an eight-byte test header from its raw fields.
#[cfg(test)]
pub(super) fn build_datatype_header(class: u8, version: u8, flags: [u8; 3], size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
    buf[1] = flags[0];
    buf[2] = flags[1];
    buf[3] = flags[2];
    LittleEndian::write_u32(&mut buf[4..8], size);
    buf
}

/// The four-byte Size field in "The Datatype Message" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[cfg(test)]
pub(super) const DATATYPE_SIZE_BYTES: u8 = 4;
