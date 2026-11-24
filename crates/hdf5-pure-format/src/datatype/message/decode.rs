//! Property layouts and field readers shared by the semantic and retained-version parsers.
//!
//! The layouts for datatype message versions 1 through 5 are defined in "The Datatype
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use alloc::string::String;
use alloc::vec::Vec;
use core::num::{NonZeroU32, NonZeroUsize};
use core::str;

use hdf5_pure_core::__private;
use hdf5_pure_core::{ComplexType, EnumMember, FormatError};

use super::DatatypeVersion;
use super::atomic;
use crate::bytes::ByteCursor;
use crate::convert::Narrow;
use crate::datatype::header::{AtomicClass, Context, DatatypeClass, Header};
use crate::datatype::legacy_member::MemberShape;
use crate::datatype::{
    self, ARRAY_CLASS, COMPLEX_CLASS, COMPLEX_FORM_MASK, COMPLEX_FORM_SHIFT,
    COMPLEX_HOMOGENEOUS_BIT, COMPLEX_RECTANGULAR_FORM, COMPLEX_RESERVED_MASK, COMPOUND_CLASS,
    ENUMERATION_CLASS, VariableLengthMetadata,
};
use crate::width::OffsetWidth;

/// A parsed datatype paired with the nonzero element size from its message.
pub(in crate::datatype) struct Decoded<T> {
    pub(in crate::datatype) datatype: T,
    /// The size in bytes of one stored element.
    pub(in crate::datatype) stored_size: NonZeroU32,
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
    VariableLength(VariableLengthMetadata),
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
    /// variable-length metadata, descriptor size, compound size, and complex form errors.
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
                let names = MemberNames::from_header(header, projection)?;
                let fields = match version {
                    DatatypeVersion::One => CompoundFields::LegacyWithShape,
                    DatatypeVersion::Two => CompoundFields::Legacy,
                    DatatypeVersion::Three | DatatypeVersion::Four | DatatypeVersion::Five => {
                        CompoundFields::Packed(datatype::offset_bytes_for_size(size))
                    }
                };
                if projection == Projection::Retained {
                    super::positive_size(COMPOUND_CLASS, size)?;
                }
                Self::Compound(Compound { names, fields })
            }
            DatatypeClass::Reference => Self::Reference,
            DatatypeClass::Enumeration => {
                Self::Enumeration(Enumeration(MemberNames::from_header(header, projection)?))
            }
            DatatypeClass::VariableLength => {
                let metadata = datatype::parse_variable_length_metadata(flags)?;
                descriptor_width(size)?;
                Self::VariableLength(metadata)
            }
            DatatypeClass::Array => Self::Array(Array::from_header(header, projection)?),
            DatatypeClass::Complex => Self::Complex(Complex::from_header(header, projection)?),
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
    /// member name and version 1 shape errors.
    pub(in crate::datatype) fn member(
        &self,
        cursor: &mut ByteCursor<'_>,
    ) -> Result<Member, FormatError> {
        let name = self.names.read(cursor, COMPOUND_CLASS)?;
        let (offset, shape) = match self.fields {
            CompoundFields::LegacyWithShape => {
                let offset = u64::from(cursor.u32()?);
                (offset, Some(MemberShape::parse(cursor)?))
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

// "The Datatype Message" in format specification version 4.0 defines compound offsets
// and the version 1 dimension block.
enum CompoundFields {
    Legacy,
    LegacyWithShape,
    Packed(usize),
}

/// A compound member's fields before its nested datatype.
pub(in crate::datatype) struct Member {
    pub(in crate::datatype) name: String,
    /// The byte offset within the containing element.
    pub(in crate::datatype) offset: u64,
    /// The version 1 scalar or array shape when present.
    pub(in crate::datatype) shape: Option<MemberShape>,
}

/// The member count and name encoding of an enumeration.
pub(in crate::datatype) struct Enumeration(MemberNames);

impl Enumeration {
    /// Reads all member names, then acquires the complete list of stored values.
    ///
    /// The cursor starts immediately after the base datatype. Each value occupies
    /// `value_width` bytes and is paired with its name in stored order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a name or the value list is incomplete,
    /// [`FormatError::UnsupportedDatatypeMemberEncoding`] if a retained name is not UTF-8,
    /// or [`FormatError::AllocationFailed`] if a member list reservation fails. Propagates
    /// platform-conversion errors for the value width or list length.
    pub(in crate::datatype) fn members(
        self,
        cursor: &mut ByteCursor<'_>,
        value_width: NonZeroU32,
    ) -> Result<Vec<EnumMember>, FormatError> {
        let count = usize::from(self.0.count);
        let mut members = Vec::new();
        for index in 0..count {
            let name = self.0.read(cursor, ENUMERATION_CLASS)?;
            if members.capacity() == index {
                let additional = index.max(4).min(count - index);
                members.try_reserve_exact(additional).map_err(|source| {
                    FormatError::AllocationFailed {
                        values: index + additional,
                        source,
                    }
                })?;
            }
            members.push(__private::enum_member(name, Vec::new()));
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

struct MemberNames {
    count: u16,
    encoding: NameEncoding,
}

impl MemberNames {
    /// Selects the member name encoding for the message version and parser.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] for a retained description before
    /// version 3, or [`FormatError::InvalidDatatypeFlags`] if that description has reserved
    /// class bits set.
    fn from_header(header: &Header, projection: Projection) -> Result<Self, FormatError> {
        let &Header {
            class,
            version,
            flags: [low, high, _],
            size: _,
        } = header;
        let class = class.code();
        let encoding = match projection {
            Projection::Retained => {
                version.for_class(class, DatatypeVersion::Three)?;
                header
                    .fields()
                    .reject_reserved_flags(class, MEMBER_COUNT_RESERVED)?;
                NameEncoding::StrictUtf8
            }
            Projection::Semantic if version < DatatypeVersion::Three => NameEncoding::Padded,
            Projection::Semantic => NameEncoding::Unpadded,
        };
        Ok(Self {
            count: u16::from_le_bytes([low, high]),
            encoding,
        })
    }

    /// Reads one NUL-terminated name and any padding in its name slot.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the terminator or required padding is
    /// missing, or [`FormatError::UnsupportedDatatypeMemberEncoding`] if a retained
    /// name is not UTF-8.
    fn read(&self, cursor: &mut ByteCursor<'_>, class: u8) -> Result<String, FormatError> {
        let remaining = cursor.remaining();
        let offset = cursor.position();
        let length = remaining
            .iter()
            .position(|&byte| byte == 0)
            .ok_or_else(|| {
                let available = offset + remaining.len();
                let expected = match self.encoding {
                    NameEncoding::StrictUtf8 => available,
                    NameEncoding::Padded | NameEncoding::Unpadded => offset,
                }
                .saturating_add(1);
                FormatError::UnexpectedEof {
                    expected,
                    available,
                }
            })?;
        let name = cursor.take(length)?;
        cursor.take_array::<1>()?;
        match self.encoding {
            NameEncoding::Padded => {
                let padding = (NAME_ALIGNMENT - (length + 1) % NAME_ALIGNMENT) % NAME_ALIGNMENT;
                cursor.take(padding)?;
                Ok(String::from_utf8_lossy(name).into_owned())
            }
            NameEncoding::Unpadded => Ok(String::from_utf8_lossy(name).into_owned()),
            NameEncoding::StrictUtf8 => str::from_utf8(name).map(String::from).map_err(|source| {
                FormatError::UnsupportedDatatypeMemberEncoding {
                    class,
                    offset,
                    source,
                }
            }),
        }
    }
}

enum NameEncoding {
    /// Semantic names padded to a multiple of eight bytes after their terminator.
    Padded,
    /// Retained names with a terminator and strict UTF-8 decoding.
    StrictUtf8,
    /// Semantic names with a terminator and no padding.
    Unpadded,
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
        datatype::array_type_size(base_size.get(), dimensions)?,
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
    projection: Projection,
}

impl Complex {
    /// Checks the version and form required to read a complex parent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] before version 5,
    /// [`FormatError::InvalidComplexFlags`] if reserved bits are set, or
    /// [`FormatError::UnsupportedComplexForm`] if the form is not homogeneous rectangular.
    fn from_header(header: &Header, projection: Projection) -> Result<Self, FormatError> {
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
        Ok(Self { size, projection })
    }

    /// Reads the floating-point parent and returns the complex type and parent version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the parent is incomplete. For an incomplete
    /// property block in the semantic representation, the byte counts are relative to the
    /// parent header. Returns [`FormatError::InvalidComplexSize`] if the declared size differs
    /// from twice the component size, and propagates errors reading the parent and checking
    /// the component size.
    pub(in crate::datatype) fn component(
        self,
        cursor: &mut ByteCursor<'_>,
    ) -> Result<(ComplexType, DatatypeVersion), FormatError> {
        let Self { size, projection } = self;
        let context = Context::ComplexParent;
        let start = cursor.position();
        let parent = Header::parse(cursor, context)?;
        let component = atomic::decode_floating_point(cursor, parent.fields(), context).map_err(
            |error| match (projection, error) {
                (
                    Projection::Semantic,
                    FormatError::UnexpectedEof {
                        expected,
                        available,
                    },
                ) => {
                    // The complete parent header lies before the attempted property extent.
                    FormatError::UnexpectedEof {
                        expected: expected - start,
                        available: available - start,
                    }
                }
                (_, error) => error,
            },
        )?;
        let complex = ComplexType::new(component)?;
        if size != complex.size().get() {
            return Err(FormatError::InvalidComplexSize {
                size,
                component_size: component.size().get(),
            });
        }
        Ok((complex, parent.version))
    }
}

/// Derives the heap address width from the variable-length descriptor's declared size.
///
/// # Errors
///
/// Returns [`FormatError::InvalidVariableLengthSize`] if the descriptor size is not 10, 12,
/// or 16 bytes.
fn descriptor_width(size: u32) -> Result<OffsetWidth, FormatError> {
    let width = size
        .checked_sub(datatype::SEQUENCE_LENGTH_SIZE + datatype::HEAP_INDEX_SIZE)
        .and_then(|width| u8::try_from(width).ok())
        .ok_or(FormatError::InvalidVariableLengthSize { size })?;
    OffsetWidth::try_from(width)
        .map_err(|_invalid_width| FormatError::InvalidVariableLengthSize { size })
}

// "The Datatype Message" in format specification version 4.0 pads terminated compound and
// enumeration names to a multiple of eight bytes in versions 1 and 2.
const NAME_ALIGNMENT: usize = 8;
// "The Datatype Message" in format specification version 4.0 defines no array class bits.
const ARRAY_RESERVED: u32 = 0xff_ffff;
// "The Datatype Message" in format specification version 4.0 reserves bits 16 through 23
// of compound and enumeration class bit fields.
const MEMBER_COUNT_RESERVED: u32 = 0xff_0000;
