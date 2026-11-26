use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU32;

use self::byte_order::DatatypeByteOrder;
pub use self::compound_member::CompoundMember;
use self::layout::{BitFieldType, FixedPointType, FloatingPointType, TimeType};
pub use self::member_name::DatatypeMemberName;
use crate::display;
use crate::display::DISPLAY_MAX_MEMBERS;
use crate::display::Dims;
use crate::display::QuotedBytes;
use crate::{FormatError, OffsetWidth};

pub mod byte_order;
mod compound_member;
pub mod layout;
mod member_name;

/// String padding type.
#[derive(Debug, Clone, PartialEq)]
pub enum StringPadding {
    /// Terminates a shortened string with a null byte.
    NullTerminate,
    /// Pads unused bytes with null bytes.
    NullPad,
    /// Pads unused bytes with spaces.
    SpacePad,
}

/// Character set encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum CharacterSet {
    /// Uses ASCII encoding.
    Ascii,
    /// Uses UTF-8 encoding.
    Utf8,
}

/// Reference type.
///
/// The format also defines attribute references in "The Datatype Message" of
/// the [format specification, version 4.0][spec], so this enum is non-exhaustive.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReferenceType {
    /// Refers to an object in the file.
    Object,
    /// Refers to a region of a dataset.
    DatasetRegion,
}

/// A member of an enumeration datatype.
///
/// The struct is non-exhaustive.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EnumMember {
    /// The stored name bytes.
    pub name: DatatypeMemberName,
    /// The encoded value bytes. [`EnumerationType::new`] checks their length against the base
    /// datatype's width.
    pub value: Vec<u8>,
}

impl EnumMember {
    /// Constructs a named member and stores its value bytes as supplied.
    pub fn new(name: DatatypeMemberName, value: Vec<u8>) -> Self {
        Self { name, value }
    }
}

/// An enumeration whose named values have the stored width of its base datatype.
///
/// [Construction](Self::new) requires a fixed-point base. Decoding retains the stored base,
/// including supported floating-point datatypes. In libhdf5, `H5Tenum_create` requires an integer
/// base (`H5Tenum.c`, HDF5 2.2.0). The message decoder checks that the parent size matches the
/// enumeration size and accepts other parent classes (`H5O__dtype_decode_helper` in `H5Odtype.c`,
/// HDF5 2.2.0).
///
/// The writer preserves the member names and value bytes. The offset width of each variable-length
/// descriptor in a decoded base must match the destination's offset width.
///
/// Class 8 of the datatype message is defined in "The Datatype Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub struct EnumerationType {
    base_type: Box<Datatype>,
    size: NonZeroU32,
    members: Vec<EnumMember>,
    offsets: DatatypeOffsets,
}

impl EnumerationType {
    /// Constructs an enumeration with a fixed-point base.
    ///
    /// Pass each value in the base type's byte order. Preserves the supplied bytes and
    /// member order, and accepts an empty member list.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::EnumMemberValueSize`] if a value has a different byte length than
    /// the base type. The error reports the first such member.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{Datatype, DatatypeMemberName, EnumMember, EnumerationType, FixedPointType};
    ///
    /// let base = FixedPointType::u16();
    /// let name = DatatypeMemberName::new("active")?;
    /// let member = EnumMember::new(name, 1_u16.to_le_bytes().to_vec());
    /// let enumeration = EnumerationType::new(base, vec![member])?;
    /// assert_eq!(enumeration.base_type(), &Datatype::FixedPoint(base));
    /// assert_eq!(enumeration.size().get(), 2);
    /// assert_eq!(enumeration.members()[0].name.as_bytes(), b"active");
    /// assert_eq!(enumeration.members()[0].value, [1, 0]);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn new(base_type: FixedPointType, members: Vec<EnumMember>) -> Result<Self, FormatError> {
        Self::from_parts(
            Datatype::FixedPoint(base_type),
            base_type.size(),
            members,
            DatatypeOffsets::default(),
        )
    }

    pub(crate) fn from_stored_parts(
        base_type: Datatype,
        size: NonZeroU32,
        offsets: DatatypeOffsets,
        members: Vec<EnumMember>,
    ) -> Result<Self, FormatError> {
        let base_size = offsets.stored_type_size(&base_type)?;
        if base_size != size.get() {
            return Err(FormatError::EnumBaseSizeMismatch {
                size: size.get(),
                base_size,
            });
        }
        Self::from_parts(base_type, size, members, offsets)
    }

    fn from_parts(
        base_type: Datatype,
        size: NonZeroU32,
        mut members: Vec<EnumMember>,
        offsets: DatatypeOffsets,
    ) -> Result<Self, FormatError> {
        if let Some(index) = members
            .iter()
            .position(|member| u32::try_from(member.value.len()) != Ok(size.get()))
        {
            let EnumMember { name, value } = members.swap_remove(index);
            return Err(FormatError::EnumMemberValueSize(
                name,
                size.get(),
                value.len(),
            ));
        }
        Ok(Self {
            base_type: Box::new(base_type),
            size,
            members,
            offsets,
        })
    }

    /// Returns the datatype that describes each member's encoded value.
    pub const fn base_type(&self) -> &Datatype {
        &self.base_type
    }

    /// Returns the stored size of one value in bytes.
    pub const fn size(&self) -> NonZeroU32 {
        self.size
    }

    /// Borrows the members in the order supplied at construction.
    pub fn members(&self) -> &[EnumMember] {
        &self.members
    }

    /// Returns the descriptor widths of the base as stored in the message.
    pub(crate) const fn offsets(&self) -> DatatypeOffsets {
        self.offsets
    }
}

/// The offset widths of variable-length descriptors in a datatype description.
///
/// The element size depends on the outermost descriptor's width for a variable-length datatype
/// or an array with a variable-length element. The writer also checks widths deeper in a
/// decoded enumeration's base to preserve the layout of its stored values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DatatypeOffsets {
    /// The descriptor width used to compute the stored element size, if one applies.
    element: Option<OffsetWidth>,
    /// The set of descriptor widths used anywhere in the description.
    used: u8,
}

impl DatatypeOffsets {
    /// Adds a descriptor width to the widths from its base and uses it for element sizing.
    pub const fn descriptor(width: OffsetWidth, base: Self) -> Self {
        Self {
            element: Some(width),
            used: base.used | width.get(),
        }
    }

    /// Includes the widths from a compound member. The compound has a declared element size.
    pub fn include_field(&mut self, field: Self) {
        self.used |= field.used;
    }

    /// Keeps the widths from the base. The enumeration has a declared element size.
    pub const fn in_enumeration(self) -> Self {
        Self {
            element: None,
            used: self.used,
        }
    }

    fn stored_type_size(self, datatype: &Datatype) -> Result<u32, FormatError> {
        stored_type_size_with_offsets(datatype, self.element)
    }

    /// Checks that the stored descriptors use the destination's offset width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::EnumBaseOffsetWidthMismatch`] with the smallest stored width
    /// that differs from `destination`, if any.
    pub fn check_destination(self, destination: OffsetWidth) -> Result<(), FormatError> {
        for source in [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight] {
            if source != destination && self.used & source.get() != 0 {
                return Err(FormatError::EnumBaseOffsetWidthMismatch {
                    source_offset_size: source.get(),
                    destination_offset_size: destination.get(),
                });
            }
        }
        Ok(())
    }
}

/// Represents the class-specific fields of an HDF5 Datatype message.
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
    /// Class 0: Fixed-point (integer) types.
    FixedPoint(FixedPointType),
    /// Class 1: Floating-point types.
    FloatingPoint(FloatingPointType),
    /// Class 2: Time type (rarely used).
    Time(TimeType),
    /// Class 3: Fixed-length string.
    String {
        /// Width of one string, in bytes.
        size: u32,
        /// Treatment of unused bytes.
        padding: StringPadding,
        /// Encoding of the string bytes.
        charset: CharacterSet,
    },
    /// Class 4: Bit field.
    BitField(BitFieldType),
    /// Class 5: Opaque data.
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
    /// Class 6: Compound type.
    Compound {
        /// Total size of one element, in bytes.
        size: u32,
        /// Named fields within an element.
        members: Vec<CompoundMember>,
    },
    /// Class 7: Reference type.
    Reference {
        /// Size of one reference, in bytes.
        size: u32,
        /// Kind of object or region referenced.
        ref_type: ReferenceType,
    },
    /// Class 8: Enumeration with value widths checked by [`EnumerationType`].
    Enumeration(EnumerationType),
    /// Class 9: Variable-length type.
    VariableLength(VariableLengthType),
    /// Class 10: Array type.
    Array {
        /// Datatype of one array element.
        base_type: Box<Datatype>,
        /// Extent of each array dimension.
        dimensions: Vec<u32>,
    },
    /// Class 11: Homogeneous rectangular complex numbers.
    Complex(ComplexType),
}

impl Datatype {
    /// Returns the size in bytes of one element of this type.
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

    /// The class code this type encodes as, the low nibble of a datatype message's first byte.
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
    /// The size is returned as a [`NonZeroU32`], so callers can divide by it or
    /// divide it into a byte count without a separate zero check. Prefer this
    /// over [`type_size`](Self::type_size) whenever the result feeds a division.
    ///
    /// The zero-size check is here because [`type_size`](Self::type_size) is
    /// computed: an [`Array`](Self::Array) with a zero dimension reports a zero
    /// size even though its header field is non-zero, and the public variants let a
    /// caller build a zero-sized value directly.
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

impl fmt::Display for DatatypeByteOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::LittleEndian => "le",
            Self::BigEndian => "be",
            Self::Vax => "vax",
        })
    }
}

impl fmt::Display for StringPadding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::NullTerminate => "null-term",
            Self::NullPad => "null-pad",
            Self::SpacePad => "space-pad",
        })
    }
}

impl fmt::Display for CharacterSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Ascii => "ascii",
            Self::Utf8 => "utf8",
        })
    }
}

impl fmt::Display for ReferenceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Object => "object_ref",
            Self::DatasetRegion => "region_ref",
        })
    }
}

/// The width in bits of a `size`-byte type.
///
/// Widens first: `size` is an on-disk `u32`, so a crafted size near [`u32::MAX`]
/// would overflow a `u32` multiply.
fn bit_width(size: u32) -> u64 {
    u64::from(size) * 8
}

/// The bit span, written only when it is narrower than the whole type.
fn write_bit_span(
    f: &mut fmt::Formatter<'_>,
    size: u32,
    bit_offset: u16,
    bit_precision: u16,
) -> fmt::Result {
    if bit_offset != 0 || u64::from(bit_precision) != bit_width(size) {
        let end = u64::from(bit_offset) + u64::from(bit_precision);
        write!(f, "(bits {bit_offset}..{end})")?;
    }
    Ok(())
}

/// The byte order, written only when it is not little-endian.
fn write_byte_order(f: &mut fmt::Formatter<'_>, byte_order: &DatatypeByteOrder) -> fmt::Result {
    if *byte_order != DatatypeByteOrder::LittleEndian {
        write!(f, " {byte_order}")?;
    }
    Ok(())
}

impl fmt::Display for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FixedPoint(datatype) => {
                let layout = datatype.layout();
                let size = datatype.size().get();
                let sign = if layout.signed() { 'i' } else { 'u' };
                write!(f, "{sign}{}", bit_width(size))?;
                write_bit_span(f, size, layout.bit_offset(), layout.bit_precision())?;
                write_byte_order(f, &datatype.byte_order())
            }
            Self::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                let size = datatype.size().get();
                write!(f, "f{}", bit_width(size))?;
                write_bit_span(f, size, layout.bit_offset(), layout.bit_precision())?;
                write_byte_order(f, &datatype.byte_order())
            }
            Self::Time(datatype) => {
                let size = datatype.size().get();
                write!(f, "time{}", bit_width(size))?;
                write_bit_span(f, size, 0, datatype.bit_precision().get())?;
                write_byte_order(f, &datatype.byte_order())
            }
            Self::String {
                size,
                padding,
                charset,
            } => write!(f, "string[{size}] {charset} {padding}"),
            Self::BitField(datatype) => {
                let size = datatype.size().get();
                write!(f, "bitfield{}", bit_width(size))?;
                write_bit_span(
                    f,
                    size,
                    datatype.bit_offset(),
                    datatype.bit_precision().get(),
                )?;
                write_byte_order(f, &datatype.byte_order())
            }
            Self::Opaque { size, tag } => {
                write!(f, "opaque[{size}]")?;
                if !tag.is_empty() {
                    write!(f, " {}", QuotedBytes(tag))?;
                }
                Ok(())
            }
            Self::Compound { members, .. } => {
                f.write_str("compound{")?;
                for (i, member) in members.iter().take(DISPLAY_MAX_MEMBERS).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}: {}", member.name, member.datatype)?;
                }
                display::write_elided(f, members.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                f.write_str("}")
            }
            Self::Reference { ref_type, .. } => write!(f, "{ref_type}"),
            Self::Enumeration(datatype) => {
                write!(f, "enum<{}>[", datatype.base_type())?;
                let members = datatype.members();
                for (i, member) in members.iter().take(DISPLAY_MAX_MEMBERS).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", member.name)?;
                }
                display::write_elided(f, members.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                f.write_str("]")
            }
            Self::VariableLength(VariableLengthType::Sequence { element_type }) => {
                write!(f, "vlen<{element_type}>")
            }
            Self::VariableLength(VariableLengthType::String(string)) => {
                write!(f, "vlen_string {}", string.charset())
            }
            Self::Array {
                base_type,
                dimensions,
            } => write!(f, "array<{base_type}, {}>", Dims(dimensions)),
            Self::Complex(datatype) => {
                write!(f, "complex<{}>", Self::FloatingPoint(datatype.component()))
            }
        }
    }
}

/// Returns the size in bytes of one stored element, including zero-sized types.
///
/// Uses `offset_width` for variable-length descriptors and computes array sizes recursively.
/// Uses [`Datatype::type_size`] for other types, including the retained width of an enumeration.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte datatype
/// size field.
pub fn stored_type_size(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<u32, FormatError> {
    stored_type_size_with_offsets(datatype, Some(offset_width))
}

/// Returns the stored element size in bytes.
///
/// Supply `offset_width` for a variable-length datatype or an array whose innermost element
/// is variable-length. Uses the declared element size for compounds and enumerations.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if a required offset width is absent, or
/// [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte datatype size field.
fn stored_type_size_with_offsets(
    datatype: &Datatype,
    offset_width: Option<OffsetWidth>,
) -> Result<u32, FormatError> {
    match datatype {
        Datatype::VariableLength(_) => {
            let offset_width = offset_width.ok_or_else(|| {
                FormatError::Internal("missing the stored variable-length descriptor width".into())
            })?;
            Ok(variable_length_descriptor_size(offset_width))
        }
        Datatype::Array {
            base_type,
            dimensions,
        } => array_type_size(
            stored_type_size_with_offsets(base_type, offset_width)?,
            dimensions,
        ),
        _ => Ok(datatype.type_size()),
    }
}

/// Computes the stored array size in bytes, returning zero if any dimension is zero.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if the size exceeds the four-byte datatype size field.
pub fn array_type_size(base_size: u32, dimensions: &[u32]) -> Result<u32, FormatError> {
    if dimensions.contains(&0) {
        return Ok(0);
    }
    dimensions.iter().try_fold(base_size, |size, &dimension| {
        size.checked_mul(dimension)
            .ok_or(FormatError::LengthTooLarge {
                length: u64::from(size) * u64::from(dimension),
                length_size: DATATYPE_SIZE_BYTES,
            })
    })
}

/// Returns the size in bytes of a sequence count and a global heap ID at `offset_width`.
pub const fn variable_length_descriptor_size(offset_width: OffsetWidth) -> u32 {
    SEQUENCE_LENGTH_SIZE + offset_width.get() as u32 + HEAP_INDEX_SIZE
}

// `H5T__vlen_disk_write` writes this count before the heap ID (`H5Tvlen.c`, HDF5 2.2.0).
const SEQUENCE_LENGTH_SIZE: u32 = 4;
// "Global Heap" of the format specification, version 4.0, defines the object index in the heap ID.
const HEAP_INDEX_SIZE: u32 = 4;
// "The Datatype Message" of the format specification, version 4.0, defines the four-byte size field.
const DATATYPE_SIZE_BYTES: u8 = 4;

/// A homogeneous rectangular complex datatype with a checked element size.
///
/// The real and imaginary components share one [`FloatingPointType`], including its layout
/// and byte order. Each complex element occupies twice the component size, within [`u32`]'s
/// range. This describes the native complex class, separate from [`Datatype::Compound`]
/// conventions with named real and imaginary fields.
///
/// Class 11 of datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComplexType {
    component: FloatingPointType,
    size: NonZeroU32,
}

impl ComplexType {
    /// Constructs a complex datatype whose two components have the supplied floating-point type.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ComplexSizeOverflow`] if twice the component size exceeds
    /// [`u32::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{ComplexType, Datatype, DatatypeByteOrder, FloatingPointType};
    ///
    /// let component = FloatingPointType::ieee754_binary32(DatatypeByteOrder::BigEndian);
    /// let complex = ComplexType::new(component)?;
    /// assert_eq!(complex.component(), component);
    /// assert_eq!(complex.size().get(), 8);
    /// assert_eq!(Datatype::Complex(complex).type_size(), 8);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn new(component: FloatingPointType) -> Result<Self, FormatError> {
        let size = component
            .size()
            .get()
            .checked_mul(2)
            .and_then(NonZeroU32::new)
            .ok_or(FormatError::ComplexSizeOverflow {
                component_size: component.size().get(),
            })?;
        Ok(Self { component, size })
    }

    /// Returns the floating-point datatype shared by both components.
    pub const fn component(self) -> FloatingPointType {
        self.component
    }

    /// Returns the combined byte size of both components.
    pub const fn size(self) -> NonZeroU32 {
        self.size
    }
}

/// A variable-length sequence or string.
///
/// Strings have required padding and character-set metadata. Sequences hold their element
/// datatype.
///
/// The variable-length class for datatype message versions 1 through 5 is defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{CharacterSet, Datatype, StringPadding, VariableLengthType};
///
/// let element = Datatype::String {
///     size: 1,
///     padding: StringPadding::NullTerminate,
///     charset: CharacterSet::Ascii,
/// };
/// let sequence = VariableLengthType::sequence(element);
/// let VariableLengthType::Sequence { element_type } = sequence else {
///     panic!("expected a sequence");
/// };
/// assert_eq!(element_type.type_size(), 1);
///
/// let datatype = Datatype::VariableLength(VariableLengthType::string(
///     StringPadding::NullTerminate,
///     CharacterSet::Utf8,
/// ));
/// let Datatype::VariableLength(VariableLengthType::String(string)) = datatype else {
///     panic!("expected a variable-length string");
/// };
/// assert_eq!(string.charset(), &CharacterSet::Utf8);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub enum VariableLengthType {
    /// A sequence whose elements share one datatype.
    Sequence {
        /// The datatype of one element.
        element_type: Box<Datatype>,
    },
    /// A string with padding and character-set metadata.
    String(VariableLengthStringType),
}

impl VariableLengthType {
    /// Constructs a sequence with the supplied element datatype.
    pub fn sequence(element_type: Datatype) -> Self {
        Self::Sequence {
            element_type: Box::new(element_type),
        }
    }

    /// Constructs a string with an unsigned one-byte integer parent datatype.
    pub fn string(padding: StringPadding, charset: CharacterSet) -> Self {
        Self::String(VariableLengthStringType::new(padding, charset))
    }

    /// Returns the element datatype of a sequence or the parent datatype of a string.
    pub fn base_type(&self) -> &Datatype {
        match self {
            Self::Sequence { element_type } => element_type,
            Self::String(string) => string.base_type(),
        }
    }
}

/// A variable-length string's padding, character set, and parent datatype.
///
/// The parent datatype has a positive element size. [`Self::new`] uses an unsigned one-byte
/// integer parent, and [`Self::with_base_type`] retains a supplied parent datatype.
///
/// The variable-length class for datatype message versions 1 through 5 is defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub struct VariableLengthStringType {
    padding: StringPadding,
    charset: CharacterSet,
    base_type: Box<Datatype>,
}

impl VariableLengthStringType {
    /// Constructs a string with an unsigned one-byte integer parent datatype.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{CharacterSet, StringPadding, VariableLengthStringType};
    ///
    /// let string = VariableLengthStringType::new(StringPadding::SpacePad, CharacterSet::Ascii);
    /// assert_eq!(string.padding(), &StringPadding::SpacePad);
    /// assert_eq!(string.charset(), &CharacterSet::Ascii);
    /// assert_eq!(string.base_type().type_size(), 1);
    /// ```
    pub fn new(padding: StringPadding, charset: CharacterSet) -> Self {
        Self {
            padding,
            charset,
            base_type: Box::new(Datatype::FixedPoint(FixedPointType::u8())),
        }
    }

    /// Constructs a string with the supplied parent datatype.
    ///
    /// The parent datatype is retained as part of the string metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `base_type` occupies zero bytes per element.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{
    ///     CharacterSet, Datatype, FormatError, StringPadding, VariableLengthStringType,
    /// };
    ///
    /// let base = Datatype::String {
    ///     size: 1,
    ///     padding: StringPadding::NullPad,
    ///     charset: CharacterSet::Ascii,
    /// };
    /// let string = VariableLengthStringType::with_base_type(
    ///     StringPadding::SpacePad,
    ///     CharacterSet::Utf8,
    ///     base.clone(),
    /// )?;
    /// assert_eq!(string.base_type(), &base);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn with_base_type(
        padding: StringPadding,
        charset: CharacterSet,
        base_type: Datatype,
    ) -> Result<Self, FormatError> {
        // H5O__dtype_decode_helper in H5Odtype.c, libhdf5 1.14.6, rejects a zero-sized datatype.
        base_type.element_size()?;
        Ok(Self {
            padding,
            charset,
            base_type: Box::new(base_type),
        })
    }

    /// Returns the treatment of unused string bytes.
    pub fn padding(&self) -> &StringPadding {
        &self.padding
    }

    /// Returns the encoding of the string bytes.
    pub fn charset(&self) -> &CharacterSet {
        &self.charset
    }

    /// Returns the parent datatype stored with the string metadata.
    pub fn base_type(&self) -> &Datatype {
        &self.base_type
    }
}

/// The native complex datatype class code.
///
/// The code for datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const COMPLEX_CLASS: u8 = 0x0B;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::CharacterSet;
    use super::Datatype;
    use super::StringPadding;
    use super::VariableLengthStringType;
    use crate::FormatError;

    #[rstest]
    #[case::zero_width_string(Datatype::String {
        size: 0,
        padding: StringPadding::NullTerminate,
        charset: CharacterSet::Ascii,
    }, STRING_CLASS)]
    #[case::zero_width_opaque(Datatype::Opaque { size: 0, tag: Vec::new() }, OPAQUE_CLASS)]
    #[case::zero_element_array(Datatype::Array {
        base_type: Box::new(Datatype::String {
            size: 1,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        }),
        dimensions: vec![0],
    }, ARRAY_CLASS)]
    fn rejects_zero_sized_string_bases(#[case] base: Datatype, #[case] class: u8) {
        assert_eq!(
            VariableLengthStringType::with_base_type(
                StringPadding::NullTerminate,
                CharacterSet::Utf8,
                base,
            )
            .unwrap_err(),
            FormatError::ZeroSizedDatatype { class },
        );
    }

    const STRING_CLASS: u8 = 0x03;
    const OPAQUE_CLASS: u8 = 0x05;
    const ARRAY_CLASS: u8 = 0x0A;
}
