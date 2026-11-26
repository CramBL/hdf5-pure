//! Property layouts and field readers shared by the semantic and retained-version parsers.
//!
//! The layouts for datatype message versions 1 through 5 are defined in "The Datatype
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use alloc::vec::Vec;
use core::ffi::CStr;
use core::num::{NonZeroU32, NonZeroUsize};

use hdf5_pure_core::__private;
use hdf5_pure_core::__private::DatatypeOffsets;
use hdf5_pure_core::{ComplexType, DatatypeMemberName, EnumMember, FormatError};

use super::atomic;
use super::compound_layout::CompoundFields;
use super::description::{self, NameLayout};
use super::{DatatypeVersion, MessageDepth};
use crate::bytes::ByteCursor;
use crate::convert::Narrow;
use crate::datatype::header::{
    ARRAY_CLASS, AtomicClass, COMPLEX_CLASS, Context, DatatypeClass, Header,
};
use crate::datatype::legacy_member::CompoundV1Shape;
use crate::datatype::variable_length::{self, VariableLengthMetadata};
use crate::width::OffsetWidth;

/// A parsed datatype paired with the nonzero element size from its message.
pub(in crate::datatype) struct Decoded<T> {
    pub(in crate::datatype) datatype: T,
    /// The size in bytes of one stored element.
    pub(in crate::datatype) stored_size: NonZeroU32,
    /// The offset widths inferred from variable-length descriptors in the description.
    pub(in crate::datatype) offsets: DatatypeOffsets,
}

/// Whether to decode a semantic datatype or retain its message versions.
#[derive(Clone, Copy, PartialEq)]
pub(in crate::datatype) enum Projection {
    /// Keeps message versions and requires the layouts representable by a retained description.
    Retained,
    /// Produces a semantic datatype, including legacy container layouts.
    Semantic,
}

/// The property layout selected by a datatype header and parser.
pub(in crate::datatype) enum Layout {
    Atomic(AtomicClass),
    Compound(Compound),
    Reference,
    Enumeration(Enumeration),
    VariableLength(VariableLengthMetadata, OffsetWidth),
    Array(Array),
    Complex(Complex),
}

impl Layout {
    /// Selects the property layout before reading the properties or nested descriptions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] if this parser requires a later version
    /// for the class, or [`FormatError::InvalidDatatypeFlags`] for set
    /// reserved compound, enumeration, or array flags in a retained description. Propagates
    /// variable-length metadata, descriptor size, and complex form errors.
    pub(in crate::datatype) fn from_header(
        header: &Header,
        projection: Projection,
    ) -> Result<Self, FormatError> {
        let &Header {
            class,
            version,
            flags,
            size,
        } = header;
        Ok(match class {
            DatatypeClass::Atomic(class) => Self::Atomic(class),
            DatatypeClass::Compound => {
                let fields = CompoundFields::for_version(version, size);
                let names = MemberNames::from_header(header, projection, fields.name_layout())?;
                Self::Compound(Compound { names, fields })
            }
            DatatypeClass::Reference => Self::Reference,
            DatatypeClass::Enumeration => Self::Enumeration(Enumeration(MemberNames::from_header(
                header,
                projection,
                NameLayout::for_version(version),
            )?)),
            DatatypeClass::VariableLength => {
                let metadata = variable_length::parse_variable_length_metadata(flags)?;
                Self::VariableLength(metadata, variable_length::descriptor_width(size)?)
            }
            DatatypeClass::Array => Self::Array(Array::from_header(header, projection)?),
            DatatypeClass::Complex => Self::Complex(Complex::from_header(header)?),
        })
    }
}

/// The name and offset layout for all members of one compound datatype.
pub(in crate::datatype) struct Compound {
    names: MemberNames,
    fields: CompoundFields,
}

impl Compound {
    /// Returns the count stored in the header's class bit field.
    pub(in crate::datatype) fn member_count(&self) -> usize {
        usize::from(self.names.count)
    }

    /// Reads the fields preceding a compound member's nested datatype.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete. Propagates
    /// version 1 shape errors.
    pub(in crate::datatype) fn member<'a>(
        &self,
        cursor: &mut ByteCursor<'a>,
    ) -> Result<Member<'a>, FormatError> {
        let name = self.names.read(cursor)?;
        let (offset, shape) = match self.fields {
            CompoundFields::LegacyWithShape => {
                let offset = u64::from(cursor.u32()?);
                let reject_reserved = self.names.projection == Projection::Retained;
                (
                    offset,
                    Some(CompoundV1Shape::parse(cursor, reject_reserved)?),
                )
            }
            CompoundFields::Legacy => (u64::from(cursor.u32()?), None),
            CompoundFields::Packed(width) => (cursor.uint(width)?, None),
        };
        Ok(Member {
            name,
            offset,
            shape,
        })
    }
}

/// A compound member's fields before its nested datatype.
pub(in crate::datatype) struct Member<'a> {
    /// The borrowed member name with its NUL terminator and without padding.
    pub(in crate::datatype) name: &'a CStr,
    /// The byte offset within the containing element.
    pub(in crate::datatype) offset: u64,
    /// The version 1 scalar or array shape when present.
    pub(in crate::datatype) shape: Option<CompoundV1Shape>,
}

/// The member count and name encoding of an enumeration.
pub(in crate::datatype) struct Enumeration(MemberNames);

impl Enumeration {
    /// Reads member names and values after checking the enumeration size against the base size.
    ///
    /// The cursor starts immediately after the base datatype. The reader reads all names before
    /// acquiring the complete value list and pairs each `value_width`-byte value with its name in
    /// stored order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::EnumBaseSizeMismatch`] if `declared_size` differs from `value_width`,
    /// [`FormatError::UnexpectedEof`] if a name or the value list is incomplete,
    /// or [`FormatError::AllocationFailed`] if a member list reservation fails. Propagates
    /// platform-conversion errors for the value width or list length.
    pub(in crate::datatype) fn members(
        self,
        cursor: &mut ByteCursor<'_>,
        declared_size: u32,
        value_width: NonZeroU32,
    ) -> Result<Vec<EnumMember>, FormatError> {
        description::check_enum_base_size(declared_size, value_width.get())?;
        let count = usize::from(self.0.count);
        let mut members = Vec::new();
        for _ in 0..count {
            let name = DatatypeMemberName::from(self.0.read(cursor)?);
            push_member(&mut members, count, EnumMember::new(name, Vec::new()))?;
        }
        let length = u64::from(self.0.count) * u64::from(value_width.get());
        let values = cursor.slice(length.to_usize()?)?;
        let value_width = value_width.narrow::<NonZeroUsize>()?;
        for (member, value) in members
            .iter_mut()
            .zip(values.chunks_exact(value_width.get()))
        {
            member.value = value.to_vec();
        }
        Ok(members)
    }
}

/// Appends a member, limiting each reservation to the remaining declared members.
///
/// Callers pass a declared `count` greater than the number of members already appended.
///
/// # Errors
///
/// Returns [`FormatError::AllocationFailed`] if reserving space for more members fails.
pub(in crate::datatype) fn push_member<T>(
    members: &mut Vec<T>,
    count: usize,
    member: T,
) -> Result<(), FormatError> {
    let index = members.len();
    if members.capacity() == index {
        let additional = index.max(4).min(count - index);
        members
            .try_reserve_exact(additional)
            .map_err(|source| FormatError::AllocationFailed {
                values: index + additional,
                source,
            })?;
    }
    members.push(member);
    Ok(())
}

struct MemberNames {
    count: u16,
    layout: NameLayout,
    projection: Projection,
}

impl MemberNames {
    /// Reads the member count and records the supplied layout for the name fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] if a retained description has reserved
    /// class bits set.
    fn from_header(
        header: &Header,
        projection: Projection,
        layout: NameLayout,
    ) -> Result<Self, FormatError> {
        let &Header {
            class,
            flags: [low, high, _],
            ..
        } = header;
        if projection == Projection::Retained {
            header
                .fields()
                .reject_reserved_flags(class.code(), MEMBER_COUNT_RESERVED)?;
        }
        Ok(Self {
            count: u16::from_le_bytes([low, high]),
            layout,
            projection,
        })
    }

    /// Borrows a NUL-terminated name after acquiring its complete field, including any padding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the terminator or required padding is missing.
    fn read<'a>(&self, cursor: &mut ByteCursor<'a>) -> Result<&'a CStr, FormatError> {
        let remaining = cursor.remaining();
        let offset = cursor.position();
        let Ok(name) = CStr::from_bytes_until_nul(remaining) else {
            let available = offset + remaining.len();
            let expected = match self.projection {
                Projection::Retained => available,
                Projection::Semantic => offset,
            }
            .saturating_add(1);
            return Err(FormatError::UnexpectedEof {
                expected,
                available,
            });
        };
        cursor.take(self.layout.field_len(name.to_bytes().len()))?;
        Ok(name)
    }
}

// "The Datatype Message" in format specification version 4.0 includes permutation indices
// only in the version 2 array property layout.
/// The layout of the rank and dimensions preceding an array's base datatype.
pub(in crate::datatype) enum Array {
    /// A four-byte rank block followed by dimensions and permutation indices.
    Legacy,
    /// A one-byte rank followed by dimensions.
    Packed,
}

impl Array {
    /// Selects the array layout from its message version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] before version 2 for a semantic
    /// description or before version 3 for a retained description. Returns
    /// [`FormatError::InvalidDatatypeFlags`] if a retained description has reserved class
    /// bits set.
    fn from_header(header: &Header, projection: Projection) -> Result<Self, FormatError> {
        let minimum = match projection {
            Projection::Retained => DatatypeVersion::Three,
            Projection::Semantic => DatatypeVersion::Two,
        };
        header.version.for_class(ARRAY_CLASS, minimum)?;
        if projection == Projection::Retained {
            header
                .fields()
                .reject_reserved_flags(ARRAY_CLASS, ARRAY_RESERVED)?;
        }
        Ok(if header.version == DatatypeVersion::Two {
            Self::Legacy
        } else {
            Self::Packed
        })
    }

    /// Reads the rank and dimensions before the base datatype.
    ///
    /// Acquires any version 2 permutation indices with the dimension list and returns
    /// the dimensions in stored order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the rank block or the list of dimensions
    /// and any permutation indices is incomplete.
    pub(in crate::datatype) fn dimensions(
        self,
        cursor: &mut ByteCursor<'_>,
    ) -> Result<Vec<u32>, FormatError> {
        let (rank, has_permutation) = match self {
            Self::Legacy => {
                let &[rank, _, _, _] = cursor.take_array()?;
                (usize::from(rank), true)
            }
            Self::Packed => (usize::from(cursor.u8()?), false),
        };
        let count = if has_permutation { rank * 2 } else { rank };
        let fields = cursor.take_arrays::<4>(count)?;
        Ok(fields
            .iter()
            .take(rank)
            .copied()
            .map(u32::from_le_bytes)
            .collect())
    }
}

/// Checks the declared array size against the stored base size times the dimension extents.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if an extent is zero,
/// [`FormatError::LengthTooLarge`] if the product exceeds the size field, or
/// [`FormatError::InvalidDatatypeSize`] if the declared size differs from the product.
pub(in crate::datatype) fn array_size(
    declared_size: u32,
    base_size: NonZeroU32,
    dimensions: &[u32],
) -> Result<NonZeroU32, FormatError> {
    let expected = super::positive_size(
        ARRAY_CLASS,
        __private::array_type_size(base_size.get(), dimensions)?,
    )?;
    if declared_size != expected.get() {
        return Err(FormatError::InvalidDatatypeSize {
            class: ARRAY_CLASS,
            size: declared_size,
            expected: expected.get(),
        });
    }
    Ok(expected)
}

/// The declared size and parser for a homogeneous rectangular complex type.
pub(in crate::datatype) struct Complex {
    size: u32,
}

impl Complex {
    /// Checks the version and form required to read a complex parent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] before version 5,
    /// [`FormatError::InvalidComplexFlags`] if reserved bits are set, or
    /// [`FormatError::UnsupportedComplexForm`] if the form is not homogeneous rectangular.
    fn from_header(header: &Header) -> Result<Self, FormatError> {
        let &Header {
            class: _,
            version,
            flags: [form_flags, _, _],
            size,
        } = header;
        version.for_class(COMPLEX_CLASS, DatatypeVersion::Five)?;
        let flags = header.flag_bits();
        if flags & COMPLEX_RESERVED_MASK != 0 {
            return Err(FormatError::InvalidComplexFlags(flags));
        }
        let homogeneous = form_flags & COMPLEX_HOMOGENEOUS_BIT != 0;
        let form = (form_flags & COMPLEX_FORM_MASK) >> COMPLEX_FORM_SHIFT;
        if !homogeneous || form != COMPLEX_RECTANGULAR_FORM {
            return Err(FormatError::UnsupportedComplexForm { homogeneous, form });
        }
        Ok(Self { size })
    }

    /// Reads the floating-point parent and returns the complex type and parent version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] before reading a parent beyond the nesting
    /// limit. Returns [`FormatError::UnexpectedEof`] if the parent is incomplete, with byte
    /// counts relative to the cursor's original input. Returns [`FormatError::InvalidComplexSize`]
    /// if the declared size differs from twice the component size. Propagates errors reading
    /// the parent and checking the component size.
    pub(in crate::datatype) fn component(
        self,
        cursor: &mut ByteCursor<'_>,
        depth: MessageDepth,
    ) -> Result<(ComplexType, DatatypeVersion), FormatError> {
        depth.child()?;
        let context = Context::ComplexParent;
        let parent = Header::parse(cursor, context)?;
        let component = atomic::decode_floating_point(cursor, parent.fields(), context)?;
        let complex = ComplexType::new(component)?;
        if self.size != complex.size().get() {
            return Err(FormatError::InvalidComplexSize {
                size: self.size,
                component_size: component.size().get(),
            });
        }
        Ok((complex, parent.version))
    }
}

// "The Datatype Message" in format specification version 4.0 defines no array class bits.
const ARRAY_RESERVED: u32 = 0xff_ffff;
// "The Datatype Message" in format specification version 4.0 reserves bits 16 through 23
// of compound and enumeration class bit fields.
const MEMBER_COUNT_RESERVED: u32 = 0xff_0000;

/// The homogeneous flag in a native complex datatype's class bit field.
///
/// The flag for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const COMPLEX_HOMOGENEOUS_BIT: u8 = 0x01;
/// The mask for the form in a native complex datatype's class bit field.
///
/// The field for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_FORM_MASK: u8 = 0x06;
/// The bit position of the form in a native complex datatype's class bit field.
///
/// The field for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_FORM_SHIFT: u8 = 1;
/// The rectangular form code of a native complex datatype.
///
/// The code for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_RECTANGULAR_FORM: u8 = 0;
/// The mask for reserved bits 3 through 23 of a complex class bit field.
///
/// The bits for datatype message version 5 are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_RESERVED_MASK: u32 = 0xFF_FFF8;
