//! Datatype-message parsing and serialization with retained versions.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::str;

use hdf5_pure_core::{
    BitFieldType, CharacterSet, ComplexType, DatatypeByteOrder, FixedPointType, FloatingPointType,
    FormatError, MantissaNormalization, StringPadding, TimeType,
};

use super::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
    FloatingPointByteOrder,
};
use crate::bytes::Fields;
use crate::datatype::{
    self, ARRAY_CLASS, BIT_FIELD_CLASS, COMPLEX_CLASS, COMPLEX_FORM_MASK, COMPLEX_FORM_SHIFT,
    COMPLEX_HOMOGENEOUS_BIT, COMPLEX_RECTANGULAR_FORM, COMPLEX_RESERVED_MASK, COMPOUND_CLASS,
    DATATYPE_CLASS_MASK, DATATYPE_HEADER_SIZE, DATATYPE_VERSION_SHIFT, ENUMERATION_CLASS,
    FIXED_POINT_CLASS, FLOAT_INTERNAL_PADDING_BIT, FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE,
    FLOAT_NORMALIZATION_LEADING_ONE, FLOAT_NORMALIZATION_MASK, FLOAT_NORMALIZATION_SHIFT,
    FLOAT_NORMALIZATION_UNNORMALIZED, FLOATING_POINT_CLASS, HIGH_PADDING_BIT, LOW_PADDING_BIT,
    OPAQUE_CLASS, OpaqueTagWire, STRING_CLASS, TIME_CLASS, VARIABLE_LENGTH_CLASS,
    VARIABLE_LENGTH_PADDING_SHIFT, VARIABLE_LENGTH_SEQUENCE, VARIABLE_LENGTH_STRING,
    VariableLengthMetadata,
};
use crate::width::OffsetWidth;

impl DatatypeMessage {
    /// Parses a datatype message and returns the number of bytes consumed.
    ///
    /// Supports fixed-point, floating-point, time, string, bitfield, opaque, and variable-length
    /// descriptions at versions 1 through 5, compounds and arrays at versions 3 through 5,
    /// and homogeneous rectangular complex descriptions at version 5. Each nested description
    /// retains its own version, at most the containing description's version. Complex parents
    /// are floating-point descriptions. VAX order requires version 3 or later. Floating-point
    /// order and reserved bits are checked through [`FloatingPointByteOrder::parse`].
    ///
    /// Variable-length descriptor sizes must be 10, 12, or 16 bytes, corresponding to heap
    /// address widths of 2, 4, or 8 bytes. The parser checks array element sizes against the
    /// recorded parent size and dimensions, and complex sizes against twice the component size.
    /// Compound member ranges are checked using their declared source sizes. A compound must
    /// have at least one member, with unique UTF-8 names.
    /// The nesting limit is 64 descriptions on one parent chain, including the outer description.
    ///
    /// The parser accepts an opaque property filled with tag bytes without a NUL terminator.
    /// The writer uses terminated tags, as described in [`serialize`](Self::serialize).
    /// The properties are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete,
    /// [`FormatError::InvalidDatatypeMessageVersion`] if the version is outside `1..=5`, and
    /// [`FormatError::InvalidDatatypeClass`] for a class outside the supported descriptions.
    /// Returns [`FormatError::InvalidDatatypeFlags`] or [`FormatError::InvalidOpaqueReservedBits`]
    /// if reserved atomic, compound, or array class bits are set. Returns
    /// [`FormatError::InvalidDatatypeVersion`] for compound or array versions 1 or 2, complex
    /// versions 1 through 4, or a complex parent version outside `1..=5`.
    ///
    /// Returns [`FormatError::InvalidVariableLengthSize`] for an unsupported descriptor size,
    /// [`FormatError::DatatypeChildVersion`] if a parent has a higher version than its container,
    /// and [`FormatError::NestingDepthExceeded`] if the parent chain exceeds the nesting limit.
    /// Returns [`FormatError::InvalidDatatypeSize`] if an array size differs from its properties,
    /// and [`FormatError::LengthTooLarge`] if the array dimension product exceeds the size field.
    ///
    /// Returns [`FormatError::InvalidCompoundMemberCount`] for a compound without members,
    /// [`FormatError::DuplicateCompoundField`] for duplicate names, and
    /// [`FormatError::CompoundFieldOutOfBounds`] for a member outside its compound element.
    /// Returns [`FormatError::UnsupportedDatatypeMemberEncoding`] if a member name is not
    /// UTF-8, retaining the name's byte offset and the original [`str::Utf8Error`].
    ///
    /// Returns [`FormatError::InvalidComplexFlags`] for reserved complex class bits,
    /// [`FormatError::UnsupportedComplexForm`] for heterogeneous or non-rectangular descriptions,
    /// [`FormatError::InvalidComplexParent`] for a non-floating parent, and
    /// [`FormatError::InvalidComplexSize`] if the declared size differs from twice the component
    /// size. Returns [`FormatError::InvalidComplexParentFlags`] for a floating-point parent's
    /// reserved class bits, reserved byte order, or VAX order before version 3.
    /// Propagates numeric layout, floating-point byte-order, string metadata, opaque-tag,
    /// variable-length metadata, component-size, and element-size errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::FormatError;
    /// use hdf5_pure_format::__private::{DatatypeMessage, DatatypeVersion, OffsetWidth};
    ///
    /// let bytes = [0x50, 0x08, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0, 0xff];
    /// let (message, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(message.version(), DatatypeVersion::Five);
    /// assert_eq!(consumed, 12);
    /// assert_eq!(message.serialize(OffsetWidth::Eight)?, bytes[..consumed]);
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    pub fn parse(data: &[u8]) -> Result<(Self, usize), FormatError> {
        let Parsed {
            message,
            end,
            size: _,
        } = Self::parse_at(data, Context::Description, 0, 0)?;
        Ok((message, end))
    }

    /// Parses a description and its declared element size at `position`.
    ///
    /// The returned end position is relative to `data`. Complex-parent class checks and
    /// diagnostics are applied according to `context`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`parse`](Self::parse). The nesting limit includes `depth`
    /// enclosing descriptions.
    fn parse_at(
        data: &[u8],
        context: Context,
        position: usize,
        depth: u8,
    ) -> Result<Parsed, FormatError> {
        if depth >= MAX_MESSAGE_DEPTH {
            return Err(FormatError::NestingDepthExceeded);
        }
        let mut fields = Fields::new(data, position);
        let header = Header::parse(&mut fields, context)?;
        let (properties, consumed) = match header.class {
            FIXED_POINT_CLASS => {
                header.reserved(FIXED_RESERVED)?;
                let datatype = FixedPointType::new(
                    header.size,
                    binary_order(header.flags[0]),
                    header.flags[0] & SIGNED_BIT != 0,
                    fields.u16()?,
                    fields.u16()?,
                )?
                .with_padding(
                    header.flags[0] & LOW_PADDING_BIT != 0,
                    header.flags[0] & HIGH_PADDING_BIT != 0,
                );
                (DatatypeProperties::FixedPoint(datatype), fields.pos())
            }
            FLOATING_POINT_CLASS => {
                let order = FloatingPointByteOrder::parse(header.version, header.flags)
                    .map_err(|error| context.float_error(&header, error))?;
                let normalization = match (header.flags[0] >> FLOAT_NORMALIZATION_SHIFT)
                    & FLOAT_NORMALIZATION_MASK
                {
                    FLOAT_NORMALIZATION_UNNORMALIZED => MantissaNormalization::Unnormalized,
                    FLOAT_NORMALIZATION_LEADING_ONE => MantissaNormalization::LeadingOne,
                    FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE => {
                        MantissaNormalization::ImpliedLeadingOne
                    }
                    value => return Err(FormatError::InvalidMantissaNormalization(value)),
                };
                let datatype = FloatingPointType::new(
                    header.size,
                    order.order(),
                    normalization,
                    header.flags[1],
                    fields.u16()?,
                    fields.u16()?,
                    fields.u8()?,
                    fields.u8()?,
                    fields.u8()?,
                    fields.u8()?,
                    fields.u32()?,
                )?
                .with_padding(
                    header.flags[0] & LOW_PADDING_BIT != 0,
                    header.flags[0] & HIGH_PADDING_BIT != 0,
                    header.flags[0] & FLOAT_INTERNAL_PADDING_BIT != 0,
                );
                (DatatypeProperties::FloatingPoint(datatype), fields.pos())
            }
            TIME_CLASS => {
                header.reserved(TIME_RESERVED)?;
                (
                    DatatypeProperties::Time(TimeType::new(
                        header.size,
                        binary_order(header.flags[0]),
                        fields.u16()?,
                    )?),
                    fields.pos(),
                )
            }
            STRING_CLASS => {
                header.reserved(STRING_RESERVED)?;
                (
                    DatatypeProperties::String {
                        size: header.size,
                        padding: datatype::parse_string_padding(
                            header.flags[0] & STRING_PADDING_MASK,
                        )?,
                        charset: datatype::parse_charset(header.flags[0] >> STRING_CHARSET_SHIFT)?,
                    },
                    fields.pos(),
                )
            }
            BIT_FIELD_CLASS => {
                header.reserved(BITFIELD_RESERVED)?;
                let datatype = BitFieldType::new(
                    header.size,
                    binary_order(header.flags[0]),
                    fields.u16()?,
                    fields.u16()?,
                )?
                .with_padding(
                    header.flags[0] & LOW_PADDING_BIT != 0,
                    header.flags[0] & HIGH_PADDING_BIT != 0,
                );
                (DatatypeProperties::BitField(datatype), fields.pos())
            }
            OPAQUE_CLASS => {
                let reserved = u16::from_le_bytes([header.flags[1], header.flags[2]]);
                if reserved != 0 {
                    return Err(FormatError::InvalidOpaqueReservedBits { bits: reserved });
                }
                let tag = OpaqueTagWire::parse(header.flags[0], &data[fields.pos()..])?;
                (
                    DatatypeProperties::Opaque {
                        size: header.size,
                        tag: tag.logical.to_vec(),
                    },
                    fields.pos() + usize::from(tag.encoded_len),
                )
            }
            COMPOUND_CLASS => {
                let count = header.modern_member_count()?;
                super::positive_size(COMPOUND_CLASS, header.size)?;
                let offset_bytes = datatype::offset_bytes_for_size(header.size);
                let mut members = Vec::new();
                for _ in 0..count {
                    let (name, end) = member_name(data, COMPOUND_CLASS, fields.pos())?;
                    fields = Fields::new(data, end);
                    let offset = fields.uint(offset_bytes)?;
                    let Parsed {
                        message: datatype,
                        end,
                        size: field_size,
                    } = Self::parse_at(data, Context::Description, fields.pos(), depth + 1)?;
                    if offset
                        .checked_add(u64::from(field_size))
                        .is_none_or(|end| end > u64::from(header.size))
                    {
                        return Err(FormatError::CompoundFieldOutOfBounds {
                            name,
                            offset,
                            field_size,
                            compound_size: header.size,
                        });
                    }
                    members.push(DatatypeMessageMember::new(name, offset, datatype)?);
                    fields = Fields::new(data, end);
                }
                (
                    DatatypeProperties::Compound {
                        size: header.size,
                        members,
                    },
                    fields.pos(),
                )
            }
            VARIABLE_LENGTH_CLASS => {
                let metadata = datatype::parse_variable_length_metadata(header.flags)?;
                descriptor_width(header.size)?;
                let Parsed {
                    message: base_type,
                    end: consumed,
                    size: _,
                } = Self::parse_at(data, Context::Description, fields.pos(), depth + 1)?;
                let base_type = Box::new(base_type);
                let properties = match metadata {
                    VariableLengthMetadata::Sequence => {
                        DatatypeProperties::VariableLengthSequence { base_type }
                    }
                    VariableLengthMetadata::String { padding, charset } => {
                        DatatypeProperties::VariableLengthString {
                            padding,
                            charset,
                            base_type,
                        }
                    }
                };
                (properties, consumed)
            }
            ARRAY_CLASS => {
                header
                    .version
                    .for_class(ARRAY_CLASS, DatatypeVersion::Three)?;
                header.reserved(ARRAY_RESERVED)?;
                let rank = fields.u8()?;
                let dimensions = (0..rank)
                    .map(|_| fields.u32())
                    .collect::<Result<Vec<_>, _>>()?;
                let Parsed {
                    message: base_type,
                    end,
                    size: base_size,
                } = Self::parse_at(data, Context::Description, fields.pos(), depth + 1)?;
                let expected = datatype::array_type_size(base_size, &dimensions)?;
                if header.size != expected {
                    return Err(FormatError::InvalidDatatypeSize {
                        class: ARRAY_CLASS,
                        size: header.size,
                        expected,
                    });
                }
                (
                    DatatypeProperties::Array {
                        base_type: Box::new(base_type),
                        dimensions,
                    },
                    end,
                )
            }
            COMPLEX_CLASS => {
                header
                    .version
                    .for_class(COMPLEX_CLASS, DatatypeVersion::Five)?;
                let flags = header.flag_bits();
                if flags & COMPLEX_RESERVED_MASK != 0 {
                    return Err(FormatError::InvalidComplexFlags(flags));
                }
                let homogeneous = header.flags[0] & COMPLEX_HOMOGENEOUS_BIT != 0;
                let form = (header.flags[0] & COMPLEX_FORM_MASK) >> COMPLEX_FORM_SHIFT;
                if !homogeneous || form != COMPLEX_RECTANGULAR_FORM {
                    return Err(FormatError::UnsupportedComplexForm { homogeneous, form });
                }
                let Parsed {
                    message: parent,
                    end,
                    size: _,
                } = Self::parse_at(data, Context::ComplexParent, fields.pos(), depth + 1)?;
                let DatatypeProperties::FloatingPoint(component) = parent.properties() else {
                    return Err(FormatError::Internal(
                        "a checked complex parent is not floating point".into(),
                    ));
                };
                let complex = ComplexType::new(*component)?;
                if header.size != complex.size().get() {
                    return Err(FormatError::InvalidComplexSize {
                        size: header.size,
                        component_size: component.size().get(),
                    });
                }
                (
                    DatatypeProperties::Complex {
                        component: Box::new(parent),
                    },
                    end,
                )
            }
            class => return Err(FormatError::InvalidDatatypeClass(class)),
        };
        Ok(Parsed {
            message: Self::new(header.version, properties)?,
            end: consumed,
            size: header.size,
        })
    }

    /// Serializes a datatype description with its retained message versions.
    ///
    /// Serializes the classes accepted by [`parse`](Self::parse). The writer derives each
    /// variable-length descriptor size from `offset_width`, including nested descriptions.
    /// Array sizes are derived from the encoded parent size and dimensions at that width.
    /// Atomic element sizes are independent of it.
    /// Compound sizes and member offsets are retained. The writer checks member ranges using
    /// their encoded sizes at that width and writes each UTF-8 name with a NUL terminator.
    /// Member offsets occupy the minimum number of bytes needed for the compound size.
    /// The nesting limit is 64 descriptions on one parent chain, including the outer description.
    ///
    /// The writer appends a NUL terminator to each nonempty opaque tag and pads its property
    /// to a multiple of eight bytes. The maximum writable logical tag length is 247 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeClass`] for a class outside the supported descriptions,
    /// [`FormatError::OpaqueTagTooLong`] for a tag longer than 247 bytes, and
    /// [`FormatError::InvalidOpaqueTagByte`] for a tag containing a NUL byte.
    /// Returns [`FormatError::NestingDepthExceeded`] if the parent chain exceeds the nesting
    /// limit, and [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte size
    /// field at `offset_width`. Propagates errors from checking floating-point version/order
    /// compatibility and calculating the stored element size. Returns
    /// [`FormatError::CompoundFieldOutOfBounds`] if a member exceeds its compound element
    /// at `offset_width`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let message = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::FixedPoint(FixedPointType::u8()),
    /// )?;
    /// assert_eq!(
    ///     message.serialize(OffsetWidth::Eight)?,
    ///     [0x50, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0],
    /// );
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// Variable-length parents retain their own versions:
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let parent = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FixedPoint(FixedPointType::i16()),
    /// )?;
    /// let sequence = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::VariableLengthSequence { base_type: Box::new(parent) },
    /// )?;
    /// let bytes = sequence.serialize(OffsetWidth::Two)?;
    /// assert_eq!(&bytes[..8], &[0x59, 0, 0, 0, 10, 0, 0, 0]);
    /// assert_eq!(bytes[8], 0x20);
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, sequence);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// Arrays and complex descriptions retain their component versions:
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeByteOrder, FloatingPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let component = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary32(
    ///         DatatypeByteOrder::BigEndian,
    ///     )),
    /// )?;
    /// let complex = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::Complex { component: Box::new(component) },
    /// )?;
    /// let array = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::Array { base_type: Box::new(complex), dimensions: vec![2] },
    /// )?;
    /// let bytes = array.serialize(OffsetWidth::Eight)?;
    /// assert_eq!(&bytes[..8], &[0x5a, 0, 0, 0, 16, 0, 0, 0]);
    /// assert_eq!(bytes[13], 0x5b);
    /// assert_eq!(bytes[21], 0x21);
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, array);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// Compound members retain their offsets and individual versions:
    ///
    /// ```
    /// use hdf5_pure_core::{FixedPointType, FormatError};
    /// use hdf5_pure_format::__private::{
    ///     DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion, OffsetWidth,
    /// };
    ///
    /// let integer = DatatypeMessage::new(
    ///     DatatypeVersion::Two,
    ///     DatatypeProperties::FixedPoint(FixedPointType::u8()),
    /// )?;
    /// let compound = DatatypeMessage::new(
    ///     DatatypeVersion::Five,
    ///     DatatypeProperties::Compound {
    ///         size: 4,
    ///         members: vec![DatatypeMessageMember::new("x".into(), 3, integer)?],
    ///     },
    /// )?;
    /// let bytes = compound.serialize(OffsetWidth::Eight)?;
    /// assert_eq!(&bytes[..11], &[0x56, 1, 0, 0, 4, 0, 0, 0, b'x', 0, 3]);
    /// assert_eq!(bytes[11], 0x20);
    /// let (parsed, consumed) = DatatypeMessage::parse(&bytes)?;
    /// assert_eq!(parsed, compound);
    /// assert_eq!(consumed, bytes.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, FormatError> {
        let mut output = Vec::new();
        self.encode_into(&mut output, offset_width, 0)?;
        Ok(output)
    }

    /// Appends this description and its nested parents, returning the stored element size.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`serialize`](Self::serialize). The nesting limit includes
    /// `depth` enclosing descriptions.
    fn encode_into(
        &self,
        output: &mut Vec<u8>,
        offset_width: OffsetWidth,
        depth: u8,
    ) -> Result<u32, FormatError> {
        if depth >= MAX_MESSAGE_DEPTH {
            return Err(FormatError::NestingDepthExceeded);
        }
        let flags = self.class_flags()?;
        let start = output.len();
        let size = match self.properties {
            DatatypeProperties::Array { .. } => 0,
            _ => self.stored_size(offset_width)?,
        };
        let mut header = datatype::append_header(
            output,
            self.properties.class(),
            self.version.get(),
            flags,
            size,
        );
        match &self.properties {
            DatatypeProperties::FixedPoint(datatype) => {
                output.extend_from_slice(&datatype.layout().bit_offset().to_le_bytes());
                output.extend_from_slice(&datatype.layout().bit_precision().to_le_bytes());
            }
            DatatypeProperties::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                output.extend_from_slice(&layout.bit_offset().to_le_bytes());
                output.extend_from_slice(&layout.bit_precision().to_le_bytes());
                output.extend_from_slice(&[
                    layout.exponent_location(),
                    layout.exponent_size(),
                    layout.mantissa_location(),
                    layout.mantissa_size(),
                ]);
                output.extend_from_slice(&layout.exponent_bias().to_le_bytes());
            }
            DatatypeProperties::Time(datatype) => {
                output.extend_from_slice(&datatype.bit_precision().get().to_le_bytes())
            }
            DatatypeProperties::BitField(datatype) => {
                output.extend_from_slice(&datatype.bit_offset().to_le_bytes());
                output.extend_from_slice(&datatype.bit_precision().get().to_le_bytes());
            }
            DatatypeProperties::Opaque { tag, .. } => {
                output.extend_from_slice(tag);
                output.resize(start + DATATYPE_HEADER_SIZE + usize::from(flags[0]), 0);
            }
            DatatypeProperties::String { .. } => {}
            DatatypeProperties::Compound { size, members } => {
                let offset_bytes = datatype::offset_bytes_for_size(*size);
                for DatatypeMessageMember {
                    name,
                    byte_offset: offset,
                    datatype,
                } in members
                {
                    let offset = *offset;
                    output.extend_from_slice(name.as_bytes());
                    output.push(0);
                    output.extend_from_slice(offset.to_le_bytes().get(..offset_bytes).ok_or_else(
                        || {
                            FormatError::Internal(
                                "a compound offset field exceeds eight bytes".into(),
                            )
                        },
                    )?);
                    let field_size = datatype.encode_into(output, offset_width, depth + 1)?;
                    if offset
                        .checked_add(u64::from(field_size))
                        .is_none_or(|end| end > u64::from(*size))
                    {
                        return Err(FormatError::CompoundFieldOutOfBounds {
                            name: name.clone(),
                            offset,
                            field_size,
                            compound_size: *size,
                        });
                    }
                }
            }
            DatatypeProperties::Enumeration { .. } => {
                return Err(FormatError::InvalidDatatypeClass(ENUMERATION_CLASS));
            }
            DatatypeProperties::VariableLengthSequence { base_type }
            | DatatypeProperties::VariableLengthString { base_type, .. } => {
                base_type.encode_into(output, offset_width, depth + 1)?;
            }
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => {
                output.push(u8::try_from(dimensions.len()).map_err(|_err| {
                    FormatError::Internal("a checked array rank exceeds u8".into())
                })?);
                for dimension in dimensions {
                    output.extend_from_slice(&dimension.to_le_bytes());
                }
                let base_size = base_type.encode_into(output, offset_width, depth + 1)?;
                header.type_size = datatype::array_type_size(base_size, dimensions)?;
            }
            DatatypeProperties::Complex { component } => {
                component.encode_into(output, offset_width, depth + 1)?;
            }
        }
        Ok(header.finish(output).type_size)
    }

    /// Encodes the three class-specific flag bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeClass`] for a class outside the supported descriptions.
    /// Propagates floating-point version/order and opaque-tag encoding errors.
    fn class_flags(&self) -> Result<[u8; 3], FormatError> {
        Ok(match &self.properties {
            DatatypeProperties::FixedPoint(datatype) => [
                binary_order_bit(datatype.byte_order())
                    | datatype::padding_bits(
                        datatype.layout().low_padding(),
                        datatype.layout().high_padding(),
                    )
                    | if datatype.layout().signed() {
                        SIGNED_BIT
                    } else {
                        0
                    },
                0,
                0,
            ],
            DatatypeProperties::FloatingPoint(datatype) => {
                let layout = datatype.layout();
                let normalization = match layout.mantissa_normalization() {
                    MantissaNormalization::Unnormalized => FLOAT_NORMALIZATION_UNNORMALIZED,
                    MantissaNormalization::LeadingOne => FLOAT_NORMALIZATION_LEADING_ONE,
                    MantissaNormalization::ImpliedLeadingOne => {
                        FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE
                    }
                };
                [
                    FloatingPointByteOrder::new(self.version, datatype.byte_order())?.bits()
                        | datatype::padding_bits(layout.low_padding(), layout.high_padding())
                        | if layout.internal_padding() {
                            FLOAT_INTERNAL_PADDING_BIT
                        } else {
                            0
                        }
                        | (normalization << FLOAT_NORMALIZATION_SHIFT),
                    layout.sign_location(),
                    0,
                ]
            }
            DatatypeProperties::Time(datatype) => [binary_order_bit(datatype.byte_order()), 0, 0],
            DatatypeProperties::String {
                padding, charset, ..
            } => {
                let padding = padding_code(padding);
                let charset = charset_code(charset);
                [padding | (charset << STRING_CHARSET_SHIFT), 0, 0]
            }
            DatatypeProperties::BitField(datatype) => [
                binary_order_bit(datatype.byte_order())
                    | datatype::padding_bits(datatype.low_padding(), datatype.high_padding()),
                0,
                0,
            ],
            DatatypeProperties::Opaque { tag, .. } => {
                [OpaqueTagWire::encode(tag)?.encoded_len, 0, 0]
            }
            DatatypeProperties::Compound { members, .. } => member_flags(members.len())?,
            DatatypeProperties::Enumeration { .. } => {
                return Err(FormatError::InvalidDatatypeClass(ENUMERATION_CLASS));
            }
            DatatypeProperties::VariableLengthSequence { .. } => [VARIABLE_LENGTH_SEQUENCE, 0, 0],
            DatatypeProperties::VariableLengthString {
                padding, charset, ..
            } => [
                VARIABLE_LENGTH_STRING | (padding_code(padding) << VARIABLE_LENGTH_PADDING_SHIFT),
                charset_code(charset),
                0,
            ],
            DatatypeProperties::Array { .. } => [0; 3],
            DatatypeProperties::Complex { .. } => [COMPLEX_HOMOGENEOUS_BIT, 0, 0],
        })
    }
}

/// The common datatype header with a checked message version.
struct Header {
    class: u8,
    version: DatatypeVersion,
    flags: [u8; 3],
    size: u32,
}

impl Header {
    /// Parses the common header and checks its message version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a required field is incomplete, or
    /// [`FormatError::InvalidDatatypeMessageVersion`] if the version is outside `1..=5`.
    /// In [`Context::ComplexParent`], returns [`FormatError::InvalidComplexParent`] for a
    /// non-floating class and [`FormatError::InvalidDatatypeVersion`] for an unsupported version.
    fn parse(fields: &mut Fields<'_>, context: Context) -> Result<Self, FormatError> {
        let class_and_version = fields.u8()?;
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
            class,
            version,
            flags: fields.array()?,
            size: fields.u32()?,
        })
    }

    /// Checks the class flags against a mask of reserved bits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeFlags`] if any masked bit is set.
    fn reserved(&self, mask: u32) -> Result<(), FormatError> {
        let flags = self.flag_bits();
        if flags & mask != 0 {
            return Err(FormatError::InvalidDatatypeFlags {
                class: self.class,
                version: self.version.get(),
                flags,
            });
        }
        Ok(())
    }

    /// Parses the member count for the property layouts of versions 3 through 5.
    ///
    /// The layouts are defined in "The Datatype Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDatatypeVersion`] for versions 1 or 2, and
    /// [`FormatError::InvalidDatatypeFlags`] if bits 16 through 23 are set.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    fn modern_member_count(&self) -> Result<u16, FormatError> {
        let Self {
            class,
            version,
            flags,
            size: _,
        } = self;
        version.for_class(*class, DatatypeVersion::Three)?;
        self.reserved(MEMBER_COUNT_RESERVED)?;
        Ok(u16::from_le_bytes([flags[0], flags[1]]))
    }

    /// Returns the complete 24-bit class bit field.
    fn flag_bits(&self) -> u32 {
        u32::from_le_bytes([self.flags[0], self.flags[1], self.flags[2], 0])
    }
}

/// A checked description with its source end position and declared element size.
struct Parsed {
    /// The description, including retained versions at each node.
    message: DatatypeMessage,
    /// The first byte after this description within the input slice.
    end: usize,
    /// The element size from the source header, in bytes.
    size: u32,
}

/// The role of a description in a recursive parse.
#[derive(Clone, Copy, PartialEq)]
enum Context {
    /// A floating-point parent within a complex description.
    ComplexParent,
    /// A description without a required parent class.
    Description,
}

impl Context {
    /// Maps a floating-point flag error to the diagnostic for this context.
    ///
    /// Returns [`FormatError::InvalidComplexParentFlags`] for [`Self::ComplexParent`],
    /// using the header's version and flags. Returns `error` for [`Self::Description`].
    fn float_error(self, header: &Header, error: FormatError) -> FormatError {
        match self {
            Self::ComplexParent => FormatError::InvalidComplexParentFlags {
                version: header.version.get(),
                flags: header.flag_bits(),
            },
            Self::Description => error,
        }
    }
}

/// Encodes a checked member count in the low two class flag bytes.
///
/// The count field is defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if `count` exceeds [`u16::MAX`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn member_flags(count: usize) -> Result<[u8; 3], FormatError> {
    let [low, high] = super::member_count(count)
        .map_err(|_err| {
            FormatError::Internal("a checked datatype member count exceeds u16".into())
        })?
        .to_le_bytes();
    Ok([low, high, 0])
}

/// Parses an unpadded UTF-8 name and the position after its NUL terminator.
///
/// The unpadded NUL-terminated name field is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec]. This representation requires UTF-8 text.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the terminator is missing, and
/// [`FormatError::UnsupportedDatatypeMemberEncoding`] if the name is not UTF-8.
/// The encoding error includes `class`, `position`, and the original [`str::Utf8Error`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn member_name(data: &[u8], class: u8, position: usize) -> Result<(String, usize), FormatError> {
    let rest = data.get(position..).ok_or(FormatError::UnexpectedEof {
        expected: position.saturating_add(1),
        available: data.len(),
    })?;
    let length = rest
        .iter()
        .position(|&byte| byte == 0)
        .ok_or(FormatError::UnexpectedEof {
            expected: data.len().saturating_add(1),
            available: data.len(),
        })?;
    let bytes = rest.get(..length).ok_or_else(|| {
        FormatError::Internal("a terminated member name exceeds its input".into())
    })?;
    let name =
        str::from_utf8(bytes).map_err(|source| FormatError::UnsupportedDatatypeMemberEncoding {
            class,
            offset: position,
            source,
        })?;
    let end = position
        .checked_add(length)
        .and_then(|end| end.checked_add(1))
        .ok_or(FormatError::UnexpectedEof {
            expected: usize::MAX,
            available: data.len(),
        })?;
    Ok((String::from(name), end))
}

/// Parses the heap-address width from a variable-length descriptor size.
///
/// # Errors
///
/// Returns [`FormatError::InvalidVariableLengthSize`] if `size` is not 10, 12, or 16 bytes.
fn descriptor_width(size: u32) -> Result<OffsetWidth, FormatError> {
    let width = size
        .checked_sub(datatype::SEQUENCE_LENGTH_SIZE + datatype::HEAP_INDEX_SIZE)
        .ok_or(FormatError::InvalidVariableLengthSize { size })?;
    let width =
        u8::try_from(width).map_err(|_err| FormatError::InvalidVariableLengthSize { size })?;
    OffsetWidth::try_from(width).map_err(|_err| FormatError::InvalidVariableLengthSize { size })
}

fn padding_code(padding: &StringPadding) -> u8 {
    match padding {
        StringPadding::NullTerminate => NULL_TERMINATE,
        StringPadding::NullPad => NULL_PAD,
        StringPadding::SpacePad => SPACE_PAD,
    }
}

fn charset_code(charset: &CharacterSet) -> u8 {
    match charset {
        CharacterSet::Ascii => ASCII_CHARSET,
        CharacterSet::Utf8 => UTF8_CHARSET,
    }
}

fn binary_order(flags: u8) -> DatatypeByteOrder {
    if flags & BINARY_ORDER_BIT == 0 {
        DatatypeByteOrder::LittleEndian
    } else {
        DatatypeByteOrder::BigEndian
    }
}

fn binary_order_bit(order: DatatypeByteOrder) -> u8 {
    if order == DatatypeByteOrder::BigEndian {
        BINARY_ORDER_BIT
    } else {
        0
    }
}

/// The mask for reserved fixed-point class bits 4 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FIXED_RESERVED: u32 = 0xff_fff0;
/// The mask for reserved time class bits 1 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const TIME_RESERVED: u32 = 0xff_fffe;
/// The mask for reserved string class bits 8 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const STRING_RESERVED: u32 = 0xff_ff00;
/// The mask for reserved bitfield class bits 3 through 23.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const BITFIELD_RESERVED: u32 = 0xff_fff8;
/// Bit 0, the byte-order bit for fixed-point, time, and bitfield descriptions.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const BINARY_ORDER_BIT: u8 = 0x01;
/// Bit 3, the signed-integer flag for fixed-point descriptions.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const SIGNED_BIT: u8 = 0x08;
/// The mask for string padding bits 0 through 3.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const STRING_PADDING_MASK: u8 = 0x0f;
/// The position of the string character-set field at bit 4.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const STRING_CHARSET_SHIFT: u8 = 4;

/// The string padding code for NUL termination.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const NULL_TERMINATE: u8 = 0;
/// The string padding code for NUL padding.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const NULL_PAD: u8 = 1;
/// The string padding code for space padding.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const SPACE_PAD: u8 = 2;
/// The ASCII string character-set code.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const ASCII_CHARSET: u8 = 0;
/// The UTF-8 string character-set code.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const UTF8_CHARSET: u8 = 1;

/// The maximum number of descriptions on a parsing or serialization path.
const MAX_MESSAGE_DEPTH: u8 = 64;

/// The mask for reserved array class bits 0 through 23.
///
/// Array descriptions at versions 3 through 5 have no class flags, as defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const ARRAY_RESERVED: u32 = 0xff_ffff;

/// The mask for reserved compound and enumeration class bits 16 through 23.
///
/// The fields are defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const MEMBER_COUNT_RESERVED: u32 = 0xff_0000;
