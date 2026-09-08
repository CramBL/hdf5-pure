//! HDF5 Datatype message parsing (message type 0x0003).
//!
//! Supports all 12 HDF5 type classes (0–11) with recursive parsing
//! for compound, enumeration, variable-length, and array types.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec, vec::Vec};

use core::fmt;
use core::num::{NonZeroU32, NonZeroUsize};

use byteorder::{ByteOrder, LittleEndian};

use crate::bytes::ensure_len;
use crate::display::{DISPLAY_MAX_MEMBERS, Dims, EscapedName, QuotedBytes, write_elided};
use crate::error::FormatError;

/// Byte order of numeric data.
#[derive(Debug, Clone, PartialEq)]
pub enum DatatypeByteOrder {
    LittleEndian,
    BigEndian,
    Vax,
}

/// String padding type.
#[derive(Debug, Clone, PartialEq)]
pub enum StringPadding {
    NullTerminate,
    NullPad,
    SpacePad,
}

/// Character set encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum CharacterSet {
    Ascii,
    Utf8,
}

/// Reference type.
///
/// Non-exhaustive: the format has gained reference kinds since (HDF5 1.12 added
/// attribute references), so match with a `_` arm.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReferenceType {
    Object,
    DatasetRegion,
}

/// A member of a compound datatype.
///
/// Non-exhaustive: parsed from a datatype message, and
/// [`CompoundTypeBuilder`](crate::CompoundTypeBuilder) builds one over an
/// arbitrary offset and member datatype, so nothing needs to construct this
/// directly.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset within the compound.
    pub byte_offset: u64,
    /// Member datatype.
    pub datatype: Datatype,
}

/// A member of an enumeration datatype.
///
/// Non-exhaustive: parsed from a datatype message, and
/// [`EnumTypeBuilder`](crate::EnumTypeBuilder) builds one over any integer base
/// type (`with_base` plus `raw_value`), so nothing needs to construct this
/// directly.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EnumMember {
    /// Member name.
    pub name: String,
    /// Raw value bytes (length = base type size).
    pub value: Vec<u8>,
}

/// Parsed HDF5 datatype.
///
/// Non-exhaustive: the format's class set is not closed (HDF5 1.14.6 added a
/// complex-number class), so match with a `_` arm. Only the *class* set is
/// sealed — the variants stay open, so an exotic type this crate has no
/// constructor for can still be built as a literal, and surfacing a format field
/// this crate currently discards (a fixed-point type's padding bits, say) would
/// still be a breaking change.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Datatype {
    /// Class 0: Fixed-point (integer) types.
    FixedPoint {
        size: u32,
        byte_order: DatatypeByteOrder,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// Class 1: Floating-point types.
    FloatingPoint {
        size: u32,
        byte_order: DatatypeByteOrder,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    },
    /// Class 2: Time type (rarely used).
    Time {
        size: u32,
        byte_order: DatatypeByteOrder,
        bit_precision: u16,
    },
    /// Class 3: Fixed-length string.
    String {
        size: u32,
        padding: StringPadding,
        charset: CharacterSet,
    },
    /// Class 4: Bit field.
    BitField {
        size: u32,
        byte_order: DatatypeByteOrder,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// Class 5: Opaque data.
    Opaque { size: u32, tag: Vec<u8> },
    /// Class 6: Compound type.
    Compound {
        size: u32,
        members: Vec<CompoundMember>,
    },
    /// Class 7: Reference type.
    Reference { size: u32, ref_type: ReferenceType },
    /// Class 8: Enumeration type.
    Enumeration {
        size: u32,
        base_type: Box<Datatype>,
        members: Vec<EnumMember>,
    },
    /// Class 9: Variable-length type.
    VariableLength {
        is_string: bool,
        padding: Option<StringPadding>,
        charset: Option<CharacterSet>,
        base_type: Box<Datatype>,
    },
    /// Class 10: Array type.
    Array {
        base_type: Box<Datatype>,
        dimensions: Vec<u32>,
    },
}

// ---- Display ----
//
// These types land in error messages, so `Display` is the short form: the width
// and class, plus the fields that depart from the ordinary — a big-endian order,
// a bit span narrower than the type. A string always names its charset and
// padding, ordinary or not, because they decide how its bytes read. `Debug`
// keeps the full record.

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
/// would overflow a `u32` multiply (issue #140).
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
            Self::FixedPoint {
                size,
                byte_order,
                signed,
                bit_offset,
                bit_precision,
            } => {
                let sign = if *signed { 'i' } else { 'u' };
                write!(f, "{sign}{}", bit_width(*size))?;
                write_bit_span(f, *size, *bit_offset, *bit_precision)?;
                write_byte_order(f, byte_order)
            }
            Self::FloatingPoint {
                size,
                byte_order,
                bit_offset,
                bit_precision,
                ..
            } => {
                write!(f, "f{}", bit_width(*size))?;
                write_bit_span(f, *size, *bit_offset, *bit_precision)?;
                write_byte_order(f, byte_order)
            }
            Self::Time {
                size,
                byte_order,
                bit_precision,
            } => {
                write!(f, "time{}", bit_width(*size))?;
                write_bit_span(f, *size, 0, *bit_precision)?;
                write_byte_order(f, byte_order)
            }
            Self::String {
                size,
                padding,
                charset,
            } => write!(f, "string[{size}] {charset} {padding}"),
            Self::BitField {
                size,
                byte_order,
                bit_offset,
                bit_precision,
            } => {
                write!(f, "bitfield{}", bit_width(*size))?;
                write_bit_span(f, *size, *bit_offset, *bit_precision)?;
                write_byte_order(f, byte_order)
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
                    write!(f, "{}: {}", EscapedName(&member.name), member.datatype)?;
                }
                write_elided(f, members.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                f.write_str("}")
            }
            Self::Reference { ref_type, .. } => write!(f, "{ref_type}"),
            Self::Enumeration {
                base_type, members, ..
            } => {
                write!(f, "enum<{base_type}>[")?;
                for (i, member) in members.iter().take(DISPLAY_MAX_MEMBERS).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", EscapedName(&member.name))?;
                }
                write_elided(f, members.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                f.write_str("]")
            }
            Self::VariableLength {
                is_string,
                charset,
                base_type,
                ..
            } => {
                if *is_string {
                    f.write_str("vlen_string")?;
                    if let Some(charset) = charset {
                        write!(f, " {charset}")?;
                    }
                    Ok(())
                } else {
                    write!(f, "vlen<{base_type}>")
                }
            }
            Self::Array {
                base_type,
                dimensions,
            } => write!(f, "array<{base_type}, {}>", Dims(dimensions)),
        }
    }
}

fn parse_string_padding(val: u8) -> Result<StringPadding, FormatError> {
    match val {
        0 => Ok(StringPadding::NullTerminate),
        1 => Ok(StringPadding::NullPad),
        2 => Ok(StringPadding::SpacePad),
        _ => Err(FormatError::InvalidStringPadding(val)),
    }
}

fn parse_charset(val: u8) -> Result<CharacterSet, FormatError> {
    match val {
        0 => Ok(CharacterSet::Ascii),
        1 => Ok(CharacterSet::Utf8),
        _ => Err(FormatError::InvalidCharacterSet(val)),
    }
}

/// Read a null-terminated string from `data` starting at `offset`.
/// Returns (string, bytes_consumed including the null terminator).
fn read_null_terminated_string(data: &[u8], offset: usize) -> Result<(String, usize), FormatError> {
    if offset >= data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: offset + 1,
            available: data.len(),
        });
    }
    let remaining = &data[offset..];
    let null_pos = remaining
        .iter()
        .position(|&b| b == 0)
        .ok_or(FormatError::UnexpectedEof {
            expected: offset + 1,
            available: data.len(),
        })?;
    let name = String::from_utf8_lossy(&remaining[..null_pos]).into_owned();
    Ok((name, null_pos + 1))
}

/// Determine how many bytes are needed to encode `compound_size` as a byte offset (v3).
fn offset_bytes_for_size(compound_size: u32) -> usize {
    if compound_size <= 0xFF {
        1
    } else if compound_size <= 0xFFFF {
        2
    } else {
        4
    }
}

/// Read an unsigned integer of 1, 2, 4, or 8 bytes (LE).
fn read_uint(data: &[u8], offset: usize, nbytes: usize) -> Result<u64, FormatError> {
    ensure_len(data, offset, nbytes)?;
    let slice = &data[offset..offset + nbytes];
    Ok(match nbytes {
        1 => slice[0] as u64,
        2 => LittleEndian::read_u16(slice) as u64,
        4 => LittleEndian::read_u32(slice) as u64,
        8 => LittleEndian::read_u64(slice),
        _ => {
            return Err(FormatError::UnexpectedEof {
                expected: offset + nbytes,
                available: data.len(),
            });
        }
    })
}

impl Datatype {
    /// Parse a datatype message from raw bytes.
    ///
    /// Returns `(Datatype, bytes_consumed)` for recursive parsing.
    ///
    /// Crate-internal: no public API hands out datatype-message bytes to feed it.
    /// Read a dataset's type with [`Dataset::datatype`](crate::Dataset::datatype).
    ///
    /// A parsed type always has a non-zero [`type_size`](Self::type_size): no HDF5
    /// type occupies zero bytes per element, and every reader divides raw bytes by
    /// that size to recover an element count. Refusing it here — the one place an
    /// untrusted datatype message becomes a `Datatype` — holds that invariant for
    /// every reader of a file instead of asking each one to re-check it.
    ///
    /// It says nothing about a `Datatype` a caller builds and hands to the writer,
    /// which never passes through here. `CompoundTypeBuilder::build` over no fields
    /// yields a zero-size compound today, and the write path divides by the element
    /// size just as the read path does.
    pub(crate) fn parse(data: &[u8]) -> Result<(Datatype, usize), FormatError> {
        // Minimum header: 4 bytes (class_and_version + 3 bytes bit field) + 4 bytes size = 8
        ensure_len(data, 0, 8)?;

        let class_and_version = data[0];
        let class_id = class_and_version & 0x0F;
        let version = (class_and_version >> 4) & 0x0F;

        // 24-bit class bit field (little-endian)
        let bf0 = data[1];
        let bf1 = data[2];
        let bf2 = data[3];
        let _bit_field_24 = (bf0 as u32) | ((bf1 as u32) << 8) | ((bf2 as u32) << 16);

        let size = LittleEndian::read_u32(&data[4..8]);
        let mut pos = 8;

        let parsed = match class_id {
            0 => {
                // Fixed-Point
                ensure_len(data, pos, 4)?;
                let byte_order = if bf0 & 0x01 == 0 {
                    DatatypeByteOrder::LittleEndian
                } else {
                    DatatypeByteOrder::BigEndian
                };
                let signed = (bf0 >> 3) & 0x01 == 1;
                let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
                let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
                pos += 4;
                Ok((
                    Datatype::FixedPoint {
                        size,
                        byte_order,
                        signed,
                        bit_offset,
                        bit_precision,
                    },
                    pos,
                ))
            }
            1 => {
                // Floating-Point
                ensure_len(data, pos, 12)?;
                let bo_low = bf0 & 0x01;
                let bo_high = (bf0 >> 6) & 0x01;
                let byte_order = match (bo_high, bo_low) {
                    (0, 0) => DatatypeByteOrder::LittleEndian,
                    (0, 1) => DatatypeByteOrder::BigEndian,
                    (1, 0) => DatatypeByteOrder::Vax,
                    (1, 1) => DatatypeByteOrder::Vax,
                    _ => unreachable!(),
                };
                let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
                let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
                let exponent_location = data[pos + 4];
                let exponent_size = data[pos + 5];
                let mantissa_location = data[pos + 6];
                let mantissa_size = data[pos + 7];
                let exponent_bias = LittleEndian::read_u32(&data[pos + 8..pos + 12]);
                pos += 12;
                Ok((
                    Datatype::FloatingPoint {
                        size,
                        byte_order,
                        bit_offset,
                        bit_precision,
                        exponent_location,
                        exponent_size,
                        mantissa_location,
                        mantissa_size,
                        exponent_bias,
                    },
                    pos,
                ))
            }
            2 => {
                // Time
                ensure_len(data, pos, 2)?;
                let byte_order = if bf0 & 0x01 == 0 {
                    DatatypeByteOrder::LittleEndian
                } else {
                    DatatypeByteOrder::BigEndian
                };
                let bit_precision = LittleEndian::read_u16(&data[pos..pos + 2]);
                pos += 2;
                Ok((
                    Datatype::Time {
                        size,
                        byte_order,
                        bit_precision,
                    },
                    pos,
                ))
            }
            3 => {
                // String
                let padding_val = bf0 & 0x0F;
                let charset_val = (bf0 >> 4) & 0x0F;
                let padding = parse_string_padding(padding_val)?;
                let charset = parse_charset(charset_val)?;
                Ok((
                    Datatype::String {
                        size,
                        padding,
                        charset,
                    },
                    pos,
                ))
            }
            4 => {
                // Bit Field
                ensure_len(data, pos, 4)?;
                let byte_order = if bf0 & 0x01 == 0 {
                    DatatypeByteOrder::LittleEndian
                } else {
                    DatatypeByteOrder::BigEndian
                };
                let bit_offset = LittleEndian::read_u16(&data[pos..pos + 2]);
                let bit_precision = LittleEndian::read_u16(&data[pos + 2..pos + 4]);
                pos += 4;
                Ok((
                    Datatype::BitField {
                        size,
                        byte_order,
                        bit_offset,
                        bit_precision,
                    },
                    pos,
                ))
            }
            5 => {
                // Opaque
                let tag_len = bf0 as usize;
                ensure_len(data, pos, tag_len)?;
                let tag = data[pos..pos + tag_len].to_vec();
                // Tags are padded to multiple of 8 bytes
                let padded = (tag_len + 7) & !7;
                let pos = 8 + padded; // from start of properties
                Ok((Datatype::Opaque { size, tag }, pos))
            }
            6 => {
                // Compound
                let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
                let mut members = Vec::with_capacity(num_members as usize);

                if version == 3 || version == 4 {
                    let ob = offset_bytes_for_size(size);
                    for _ in 0..num_members {
                        let (name, name_len) = read_null_terminated_string(data, pos)?;
                        pos += name_len;
                        let byte_offset = read_uint(data, pos, ob)?;
                        pos += ob;
                        let (member_dt, consumed) = Datatype::parse(&data[pos..])?;
                        pos += consumed;
                        members.push(CompoundMember {
                            name,
                            byte_offset,
                            datatype: member_dt,
                        });
                    }
                } else if version == 1 || version == 2 {
                    // v1 and v2: the member name is NUL-terminated and padded with
                    // additional NULs to a multiple of 8 bytes, followed by a
                    // 4-byte member byte offset. v1 then carries a fixed 28-byte
                    // dimension block — dimensionality(1) + reserved(3) +
                    // dimension permutation(4) + reserved(4) + dimension sizes(16)
                    // — before the member datatype message; v2 drops that block.
                    for _ in 0..num_members {
                        let (name, name_len) = read_null_terminated_string(data, pos)?;
                        let padded = (name_len + 7) & !7;
                        pos += padded;
                        ensure_len(data, pos, 4)?;
                        let byte_offset = LittleEndian::read_u32(&data[pos..pos + 4]) as u64;
                        pos += 4;
                        if version == 1 {
                            ensure_len(data, pos, 28)?;
                            pos += 28;
                        }
                        let (member_dt, consumed) = Datatype::parse(&data[pos..])?;
                        pos += consumed;
                        members.push(CompoundMember {
                            name,
                            byte_offset,
                            datatype: member_dt,
                        });
                    }
                } else {
                    return Err(FormatError::InvalidDatatypeVersion {
                        class: class_id,
                        version,
                    });
                }

                Ok((Datatype::Compound { size, members }, pos))
            }
            7 => {
                // Reference
                let ref_type_val = bf0 & 0x0F;
                let ref_type = match ref_type_val {
                    0 => ReferenceType::Object,
                    1 => ReferenceType::DatasetRegion,
                    _ => return Err(FormatError::InvalidReferenceType(ref_type_val)),
                };
                Ok((Datatype::Reference { size, ref_type }, pos))
            }
            8 => {
                // Enumeration
                let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
                // Parse base type
                let (base_type, base_consumed) = Datatype::parse(&data[pos..])?;
                pos += base_consumed;
                let base_size = base_type.type_size();
                let mut members = Vec::with_capacity(num_members as usize);
                // Enum layout: base_type, then all names (null-terminated), then all values
                // v1/v2: names are padded to 8-byte boundaries
                // v3: names are just null-terminated
                let mut member_names = Vec::with_capacity(num_members as usize);
                for _ in 0..num_members {
                    let (name, name_len) = read_null_terminated_string(data, pos)?;
                    if version < 3 {
                        let padded = (name_len + 7) & !7;
                        pos += padded;
                    } else {
                        pos += name_len;
                    }
                    member_names.push(name);
                }
                // Now values
                for name in &member_names {
                    ensure_len(data, pos, base_size as usize)?;
                    let value = data[pos..pos + base_size as usize].to_vec();
                    pos += base_size as usize;
                    members.push(EnumMember {
                        name: name.clone(),
                        value,
                    });
                }
                Ok((
                    Datatype::Enumeration {
                        size,
                        base_type: Box::new(base_type),
                        members,
                    },
                    pos,
                ))
            }
            9 => {
                // Variable-Length
                let vl_type = bf0 & 0x0F;
                let is_string = vl_type == 1;
                let padding = if is_string {
                    let pad_val = (bf0 >> 4) & 0x0F;
                    Some(parse_string_padding(pad_val)?)
                } else {
                    None
                };
                let charset = if is_string {
                    let cs_val = bf1 & 0x0F;
                    Some(parse_charset(cs_val)?)
                } else {
                    None
                };
                let (base_type, consumed) = Datatype::parse(&data[pos..])?;
                pos += consumed;
                Ok((
                    Datatype::VariableLength {
                        is_string,
                        padding,
                        charset,
                        base_type: Box::new(base_type),
                    },
                    pos,
                ))
            }
            10 => {
                // Array
                if version == 2 {
                    ensure_len(data, pos, 4)?;
                    let ndims = data[pos] as usize;
                    pos += 4; // ndims(1) + reserved(3)
                    ensure_len(data, pos, ndims * 4 + ndims * 4)?;
                    let mut dimensions = Vec::with_capacity(ndims);
                    for _ in 0..ndims {
                        dimensions.push(LittleEndian::read_u32(&data[pos..pos + 4]));
                        pos += 4;
                    }
                    // skip permutation indices
                    pos += ndims * 4;
                    let (base_type, consumed) = Datatype::parse(&data[pos..])?;
                    pos += consumed;
                    Ok((
                        Datatype::Array {
                            base_type: Box::new(base_type),
                            dimensions,
                        },
                        pos,
                    ))
                } else if version == 3 {
                    ensure_len(data, pos, 1)?;
                    let ndims = data[pos] as usize;
                    pos += 1;
                    ensure_len(data, pos, ndims * 4)?;
                    let mut dimensions = Vec::with_capacity(ndims);
                    for _ in 0..ndims {
                        dimensions.push(LittleEndian::read_u32(&data[pos..pos + 4]));
                        pos += 4;
                    }
                    let (base_type, consumed) = Datatype::parse(&data[pos..])?;
                    pos += consumed;
                    Ok((
                        Datatype::Array {
                            base_type: Box::new(base_type),
                            dimensions,
                        },
                        pos,
                    ))
                } else {
                    Err(FormatError::InvalidDatatypeVersion {
                        class: class_id,
                        version,
                    })
                }
            }
            11 => {
                // Complex number — store as compound of two floats internally
                // Parse like compound with version 3 and 2 members
                // But actually class 11 has no special properties beyond class 6 compound.
                // It's just recognized as a separate class. For now parse the 2 members
                // as compound.
                let num_members = (bf0 as u16) | ((bf1 as u16) << 8);
                let mut members = Vec::with_capacity(num_members as usize);
                let ob = offset_bytes_for_size(size);
                for _ in 0..num_members {
                    let (name, name_len) = read_null_terminated_string(data, pos)?;
                    pos += name_len;
                    let byte_offset = read_uint(data, pos, ob)?;
                    pos += ob;
                    let (member_dt, consumed) = Datatype::parse(&data[pos..])?;
                    pos += consumed;
                    members.push(CompoundMember {
                        name,
                        byte_offset,
                        datatype: member_dt,
                    });
                }
                Ok((Datatype::Compound { size, members }, pos))
            }
            _ => Err(FormatError::InvalidDatatypeClass(class_id)),
        };

        // The declared size is checked through `type_size` rather than the header
        // field, because the two differ: an array type derives its size from its
        // base type and dimensions, so a zero dimension yields a zero-byte element
        // from a non-zero header field.
        let (datatype, consumed) = parsed?;
        if datatype.type_size() == 0 {
            return Err(FormatError::ZeroSizedDatatype { class: class_id });
        }
        Ok((datatype, consumed))
    }

    /// Serialize datatype to HDF5 message bytes.
    ///
    /// Crate-internal: hand a `Datatype` to
    /// [`DatasetBuilder::with_dtype`](crate::DatasetBuilder::with_dtype) and the
    /// writer encodes it. Widening this again is additive if a caller ever needs
    /// the raw encoding.
    pub(crate) fn serialize(&self) -> Vec<u8> {
        match self {
            Datatype::FixedPoint {
                size,
                byte_order,
                signed,
                bit_offset,
                bit_precision,
            } => {
                let mut bf0 = 0u8;
                if matches!(byte_order, DatatypeByteOrder::BigEndian) {
                    bf0 |= 0x01;
                }
                if *signed {
                    bf0 |= 0x08;
                }
                let mut buf = Self::build_header(0, 1, [bf0, 0, 0], *size);
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf
            }
            Datatype::FloatingPoint {
                size,
                byte_order,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            } => {
                let mut bf0 = 0x20u8; // bit 5: sign location bit (standard IEEE 754)
                match byte_order {
                    DatatypeByteOrder::BigEndian => {
                        bf0 |= 0x01;
                    }
                    DatatypeByteOrder::Vax => {
                        bf0 |= 0x40;
                    }
                    _ => {}
                }
                // bf[1] = sign bit location (bit position of sign in the value)
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "size is an element byte size; *8-1 is a bit index that fits in a u8 (at most 63 for an 8-byte element)"
                )]
                let bf1 = (*size * 8 - 1) as u8;
                let mut buf = Self::build_header(1, 1, [bf0, bf1, 0], *size);
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf.push(*exponent_location);
                buf.push(*exponent_size);
                buf.push(*mantissa_location);
                buf.push(*mantissa_size);
                buf.extend_from_slice(&exponent_bias.to_le_bytes());
                buf
            }
            Datatype::String {
                size,
                padding,
                charset,
            } => {
                let pad_val = match padding {
                    StringPadding::NullTerminate => 0,
                    StringPadding::NullPad => 1,
                    StringPadding::SpacePad => 2,
                };
                let cs_val = match charset {
                    CharacterSet::Ascii => 0,
                    CharacterSet::Utf8 => 1,
                };
                let bf0 = pad_val | (cs_val << 4);
                Self::build_header(3, 1, [bf0, 0, 0], *size)
            }
            Datatype::VariableLength {
                is_string,
                padding,
                charset,
                base_type,
            } => {
                let mut bf0 = if *is_string { 0x01u8 } else { 0x00 };
                if *is_string && let Some(p) = padding {
                    let pv = match p {
                        StringPadding::NullTerminate => 0,
                        StringPadding::NullPad => 1,
                        StringPadding::SpacePad => 2,
                    };
                    bf0 |= pv << 4;
                }
                let bf1 = if *is_string {
                    charset.as_ref().map_or(0, |c| match c {
                        CharacterSet::Ascii => 0,
                        CharacterSet::Utf8 => 1,
                    })
                } else {
                    0
                };
                let mut buf = Self::build_header(9, 1, [bf0, bf1, 0], 16);
                buf.extend_from_slice(&base_type.serialize());
                buf
            }
            Datatype::Compound { size, members } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "compound member count is written into the 2-byte member-count field of the datatype message"
                )]
                let num = members.len() as u16;
                let bf0 = (num & 0xFF) as u8;
                let bf1 = ((num >> 8) & 0xFF) as u8;
                let mut buf = Self::build_header(6, 3, [bf0, bf1, 0], *size);
                let ob = offset_bytes_for_size(*size);
                for m in members {
                    // Null-terminated name
                    buf.extend_from_slice(m.name.as_bytes());
                    buf.push(0);
                    // Byte offset (variable-width)
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "ob is the offset-byte width chosen to hold byte_offset, so each arm casts to a width that fits by construction"
                    )]
                    match ob {
                        1 => buf.push(m.byte_offset as u8),
                        2 => buf.extend_from_slice(&(m.byte_offset as u16).to_le_bytes()),
                        _ => buf.extend_from_slice(&(m.byte_offset as u32).to_le_bytes()),
                    }
                    // Recursively serialize member datatype
                    buf.extend_from_slice(&m.datatype.serialize());
                }
                buf
            }
            Datatype::Enumeration {
                size,
                base_type,
                members,
            } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "enumeration member count is written into the 2-byte member-count field of the datatype message"
                )]
                let num = members.len() as u16;
                let bf0 = (num & 0xFF) as u8;
                let bf1 = ((num >> 8) & 0xFF) as u8;
                let mut buf = Self::build_header(8, 3, [bf0, bf1, 0], *size);
                // Base type
                buf.extend_from_slice(&base_type.serialize());
                // All names (null-terminated)
                for m in members {
                    buf.extend_from_slice(m.name.as_bytes());
                    buf.push(0);
                }
                // All values
                for m in members {
                    buf.extend_from_slice(&m.value);
                }
                buf
            }
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                let mut buf = Self::build_header(10, 3, [0, 0, 0], self.type_size());
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "array rank is written into the 1-byte dimensionality field; HDF5 caps array rank well below 255"
                )]
                buf.push(dimensions.len() as u8);
                for &d in dimensions {
                    buf.extend_from_slice(&d.to_le_bytes());
                }
                buf.extend_from_slice(&base_type.serialize());
                buf
            }
            Datatype::Reference { size, ref_type } => {
                let bf0 = match ref_type {
                    ReferenceType::Object => 0,
                    ReferenceType::DatasetRegion => 1,
                };
                Self::build_header(7, 1, [bf0, 0, 0], *size)
            }
            Datatype::Time {
                size,
                byte_order,
                bit_precision,
            } => {
                // bf0 bit 0 is the byte order (0 = little-endian, 1 = big-endian).
                let bf0 = if matches!(byte_order, DatatypeByteOrder::BigEndian) {
                    0x01u8
                } else {
                    0
                };
                let mut buf = Self::build_header(2, 1, [bf0, 0, 0], *size);
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf
            }
            Datatype::BitField {
                size,
                byte_order,
                bit_offset,
                bit_precision,
            } => {
                let bf0 = if matches!(byte_order, DatatypeByteOrder::BigEndian) {
                    0x01u8
                } else {
                    0
                };
                let mut buf = Self::build_header(4, 1, [bf0, 0, 0], *size);
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf
            }
            Datatype::Opaque { size, tag } => {
                // bf0 carries the ASCII tag length; the tag is padded with zero
                // bytes to a multiple of 8, mirroring `parse`.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "opaque tag length is written into the 1-byte tag-length bit field (bf0)"
                )]
                let bf0 = tag.len() as u8;
                let mut buf = Self::build_header(5, 1, [bf0, 0, 0], *size);
                buf.extend_from_slice(tag);
                let padded = (tag.len() + 7) & !7;
                buf.resize(buf.len() + (padded - tag.len()), 0);
                buf
            }
        }
    }

    fn build_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
        buf[1] = bf[0];
        buf[2] = bf[1];
        buf[3] = bf[2];
        buf[4..8].copy_from_slice(&size.to_le_bytes());
        buf
    }

    /// Return the size in bytes of one element of this type.
    pub fn type_size(&self) -> u32 {
        match self {
            Datatype::FixedPoint { size, .. } => *size,
            Datatype::FloatingPoint { size, .. } => *size,
            Datatype::Time { size, .. } => *size,
            Datatype::String { size, .. } => *size,
            Datatype::BitField { size, .. } => *size,
            Datatype::Opaque { size, .. } => *size,
            Datatype::Compound { size, .. } => *size,
            Datatype::Reference { size, .. } => *size,
            Datatype::Enumeration { size, .. } => *size,
            Datatype::VariableLength { .. } => 16, // typically pointer + length
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

    /// The class code this type encodes as, the low nibble of a datatype
    /// message's first byte.
    ///
    /// Kept beside [`type_size`](Self::type_size) rather than read back out of
    /// [`serialize`](Self::serialize), so naming the class in an error costs no
    /// encoding.
    pub(crate) fn class_code(&self) -> u8 {
        match self {
            Datatype::FixedPoint { .. } => 0,
            Datatype::FloatingPoint { .. } => 1,
            Datatype::Time { .. } => 2,
            Datatype::String { .. } => 3,
            Datatype::BitField { .. } => 4,
            Datatype::Opaque { .. } => 5,
            Datatype::Compound { .. } => 6,
            Datatype::Reference { .. } => 7,
            Datatype::Enumeration { .. } => 8,
            Datatype::VariableLength { .. } => 9,
            Datatype::Array { .. } => 10,
        }
    }

    /// The element size in bytes, proven non-zero.
    ///
    /// Prefer this to [`type_size`](Self::type_size) for any element size that
    /// is about to be divided or divided *by*: it returns the size as a
    /// [`NonZeroU32`], so the value carries its own proof and the code it is
    /// handed to cannot divide by zero. Every such site in this crate takes a
    /// non-zero size rather than re-checking one.
    ///
    /// The refusal has to live here rather than in the type because
    /// `type_size()` is *computed*: an [`Array`](Self::Array) reports its base
    /// type times its dimensions, so a zero dimension yields a zero-width
    /// element behind a header that claims otherwise, and the variants are
    /// deliberately open for a caller to build as a literal. A type read out of
    /// a file is already refused when its message is decoded; this is the same
    /// refusal for a constructed one, on the way into a writer.
    ///
    /// # Errors
    ///
    /// [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes per
    /// element.
    pub fn element_size(&self) -> Result<NonZeroU32, FormatError> {
        NonZeroU32::new(self.type_size()).ok_or(FormatError::ZeroSizedDatatype {
            class: self.class_code(),
        })
    }

    /// The element size in bytes as a non-zero `usize`, for the byte arithmetic
    /// that indexes an in-memory buffer.
    ///
    /// The narrowing is the one [`convert`](crate::convert) describes: a `u32`
    /// fits `usize` on every target this crate supports, and the conversion is
    /// routed through a checked one anyway.
    ///
    /// # Errors
    ///
    /// [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes per
    /// element, or [`FormatError::ValueTooLargeForPlatform`] if the size does
    /// not fit this target's `usize`.
    pub(crate) fn element_size_usize(&self) -> Result<NonZeroUsize, FormatError> {
        crate::convert::nonzero_usize_from(self.element_size()?)
    }
}
/// Whether a datatype of this encoded class *could* hold an object address,
/// decided from the first byte of a datatype message rather than by parsing it.
///
/// A **necessary** condition for [`datatype_holds_object_address`] and never a
/// sufficient one: a compound of two integers has a qualifying class and holds
/// no address at all. It exists so a walk over every object in a file can reject
/// the overwhelmingly common cases — a fixed-point, floating-point, string, or
/// opaque dataset — without allocating a parsed [`Datatype`] for each. The
/// classes it admits are exactly the ones `datatype_holds_object_address`
/// recurses through, plus the reference itself; `class_predicate_admits_every_
/// reference_holding_type` in this module's tests is what holds the two together.
pub(crate) fn class_may_hold_object_address(class_and_version: u8) -> bool {
    matches!(
        class_and_version & 0x0F,
        COMPOUND_CLASS | REFERENCE_CLASS | ENUMERATION_CLASS | VARIABLE_LENGTH_CLASS | ARRAY_CLASS
    )
}

/// Datatype message class ids, as the low nibble of a datatype message's first
/// byte. Only the classes that can carry an object address downward are named.
const COMPOUND_CLASS: u8 = 6;
const REFERENCE_CLASS: u8 = 7;
const ENUMERATION_CLASS: u8 = 8;
const VARIABLE_LENGTH_CLASS: u8 = 9;
const ARRAY_CLASS: u8 = 10;

/// Whether `dt` reaches an **object address** anywhere in its structure — an
/// object or dataset-region reference, directly or through a compound member,
/// array entry, enumeration base, or the contents of a variable-length
/// sequence.
///
/// The paired half of [`embedded_reference_slots`], which locates the ones it
/// can address. This one recognises an object reference of any width and in any
/// position; that one maps only the 8-byte form reachable through compound
/// members and array entries. The gap between them is not an oversight but the
/// point: a datatype this accepts and that cannot map is one whose addresses
/// cannot be read, which callers must refuse rather than pass over. Their fall-
/// through arm consults this function so the two cannot drift apart.
///
/// A variable-length datatype counts only when what it *holds* is an object
/// reference. The heap itself is not at risk — a deletion frees object headers
/// and dataset storage and never a global heap collection, so a variable-length
/// string keeps pointing at data that is still there — but a `H5T_VLEN` of
/// `H5T_STD_REF_OBJ`, which the reference library writes, keeps its addresses in
/// the heap *contents*, where the element bytes hold only a heap id.
pub(crate) fn datatype_holds_object_address(dt: &Datatype) -> bool {
    match dt {
        // Both reference kinds name an object. An object reference *is* the
        // header address; a dataset-region reference is a global-heap id whose
        // heap object holds the address and a selection, so the address is one
        // indirection further out — out of reach of a screen that reads element
        // bytes, which is what makes it unmappable rather than absent.
        Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_object_address(&m.datatype)),
        Datatype::Array { base_type, .. }
        | Datatype::Enumeration { base_type, .. }
        | Datatype::VariableLength { base_type, .. } => datatype_holds_object_address(base_type),
        _ => false,
    }
}

/// Whether `dt`'s element bytes carry a **file-absolute address** at any depth:
/// a variable-length element (a global-heap collection address and index) or a
/// reference (an object address, or for a dataset-region reference a heap id),
/// directly or through a compound member, array entry, or enumeration base.
///
/// The union of the two addresses an element can hold, and deliberately not a
/// finer answer than that — both callers ask only whether an address is in
/// there at all:
///
/// - a **cross-file copy** (`reject_foreign_addresses`) refuses such a
///   datatype, since an address into the source file cannot be translated into
///   another one;
/// - the **heap-collection provenance** of a variable-length overwrite (issue
///   #321) gives up its record when a raw-bytes write could name a collection a
///   second time.
///
/// Both are one-sided: answering `true` too often costs a refusal or a reclaim,
/// answering `false` too often would cost correctness.
///
/// Distinct from [`datatype_holds_object_address`], which asks specifically
/// whether an *object header* address is reachable — so it answers `false` for
/// a variable-length string and `true` for a variable length *of* references,
/// where this one answers `true` for both.
pub(crate) fn datatype_holds_file_address(dt: &Datatype) -> bool {
    match dt {
        Datatype::VariableLength { .. } | Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_file_address(&m.datatype)),
        Datatype::Array { base_type, .. } | Datatype::Enumeration { base_type, .. } => {
            datatype_holds_file_address(base_type)
        }
        _ => false,
    }
}

/// Every 8-byte object reference `datatype` reaches through a compound member or
/// array entry, as byte offsets within one element, in declaration order.
///
/// Mirrors [`embedded_vlen_slots`](crate::vl_data::embedded_vlen_slots) for the
/// other kind of address a rewrite invalidates. A datatype that *is* an object
/// reference yields the single slot at offset 0, so callers handling that case
/// separately should test for it first.
///
/// Returns `None` when the element bytes cannot be walked safely: the offsets
/// found do not fit the datatype's declared element size, or the type reaches an
/// object reference this walker cannot address (see
/// [`datatype_holds_object_address`]). Both mean the same thing to a caller —
/// the addresses are not readable from here — so neither is reported as an empty
/// slot list, which would read as "this type holds none".
pub(crate) fn embedded_reference_slots(datatype: &Datatype) -> Option<Vec<usize>> {
    /// Returns `false` when the datatype cannot be walked on this target, for the
    /// reasons [`embedded_vlen_slots`]' walker documents.
    fn collect(datatype: &Datatype, base: usize, capacity: usize, out: &mut Vec<usize>) -> bool {
        if out.len() > capacity {
            return true;
        }
        match datatype {
            Datatype::Reference {
                ref_type: ReferenceType::Object,
                size: 8,
            } => {
                out.push(base);
                true
            }
            Datatype::Compound { members, .. } => {
                for m in members {
                    let Some(at) = usize::try_from(m.byte_offset)
                        .ok()
                        .and_then(|off| base.checked_add(off))
                    else {
                        return false;
                    };
                    if !collect(&m.datatype, at, capacity, out) {
                        return false;
                    }
                }
                true
            }
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                // As in `embedded_vlen_slots`: probe once so that entries which can
                // never contribute do not drive a walk over huge declared
                // dimensions, and so every iteration below pushes at least one slot.
                // Walked once and translated per entry, for the reason
                // `embedded_vlen_slots` documents: re-walking is exponential in
                // nesting depth.
                let mut probe = Vec::new();
                if !collect(base_type, 0, capacity, &mut probe) {
                    return false;
                }
                if probe.is_empty() {
                    return true;
                }
                let count = dimensions
                    .iter()
                    .copied()
                    .fold(1u64, |a, b| a.saturating_mul(u64::from(b)));
                // As in `embedded_vlen_slots`: more entries than the element has
                // room for cannot fit, so reject without walking them.
                if count > capacity as u64 {
                    return false;
                }
                let entries = usize::try_from(count).unwrap_or(usize::MAX);
                let stride = base_type.type_size() as usize;
                for i in 0..entries {
                    let Some(at) = i.checked_mul(stride).and_then(|off| base.checked_add(off))
                    else {
                        return false;
                    };
                    for &slot in &probe {
                        let Some(off) = at.checked_add(slot) else {
                            return false;
                        };
                        out.push(off);
                        if out.len() > capacity {
                            return true;
                        }
                    }
                }
                true
            }
            // Anything this walker does not map. A type that nonetheless
            // reaches an object reference — a width other than 8, an
            // enumeration over one, a variable-length sequence *of* them — is
            // one whose addresses cannot be located in the element bytes, so
            // say so rather than report "no slots here" and let a caller read
            // that as "nothing to check". Asking the predicate rather than
            // restating its arms is what keeps the pair honest as either grows.
            _ => !datatype_holds_object_address(datatype),
        }
    }

    let element_size = datatype.type_size() as usize;
    let capacity = element_size / 8;
    let mut slots = Vec::new();
    if !collect(datatype, 0, capacity, &mut slots) {
        return None;
    }
    // `checked_add`: an offset near the top of the address space would otherwise
    // wrap here and read as "fits".
    if slots.len() > capacity
        || slots
            .iter()
            .any(|&s| s.checked_add(8).is_none_or(|end| end > element_size))
    {
        return None;
    }
    Some(slots)
}

/// Every 8-byte object reference stored in `raw`, as
/// `(byte offset within raw, the address stored there)`.
///
/// `slots` is [`embedded_reference_slots`] for the datatype `raw` holds elements
/// of, and `element_size` its `type_size`. Callers differ in what they do with
/// an address — screen it against what a commit vacates, rewrite it to where the
/// object moved — but not in how they find one, and this is the one place that
/// walk lives. A second copy of it would be free to disagree about the element
/// stride, about a trailing partial element, or about which slots exist.
///
/// A trailing run shorter than one element is skipped: `chunks_exact` yields
/// whole elements only, which is the same thing every reader of these bytes does
/// with a truncated tail.
pub(crate) fn stored_object_references<'a>(
    raw: &'a [u8],
    element_size: usize,
    slots: &'a [usize],
) -> impl Iterator<Item = (usize, u64)> + 'a {
    raw.chunks_exact(element_size.max(1))
        .enumerate()
        .flat_map(move |(i, element)| {
            slots.iter().map(move |&at| {
                let stored = u64::from_le_bytes(element[at..at + 8].try_into().expect(
                    "embedded_reference_slots keeps every slot 8 bytes inside the element",
                ));
                (i * element_size + at, stored)
            })
        })
}

/// Build a datatype header (8 bytes) for testing.
#[cfg(test)]
fn build_dt_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
    buf[1] = bf[0];
    buf[2] = bf[1];
    buf[3] = bf[2];
    LittleEndian::write_u32(&mut buf[4..8], size);
    buf
}

#[cfg(test)]
mod tests {

    /// Every datatype that reaches an object address must have an encoded class
    /// [`class_may_hold_object_address`] admits.
    ///
    /// The two are a pair with one job between them: the class predicate is the
    /// cheap gate a whole-file walk applies before it will parse a datatype at
    /// all (`crate::reference_patch`), and the type predicate is the answer it
    /// gates. A type the gate rejects is never parsed, so if the gate ever
    /// rejected one that holds an address, the walk would pass over a reference
    /// in silence — no error, no refusal, just a stored address left dangling.
    /// Nothing in either function's code says the other exists; this is what
    /// says it.
    #[test]
    fn the_class_gate_admits_every_reference_holding_datatype() {
        let object_ref = || Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        };
        let i32_le = || Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let holds_an_address = [
            ("a bare object reference", object_ref()),
            (
                "a dataset-region reference",
                Datatype::Reference {
                    size: 12,
                    ref_type: ReferenceType::DatasetRegion,
                },
            ),
            (
                "a compound holding one",
                Datatype::Compound {
                    size: 12,
                    members: vec![
                        CompoundMember {
                            name: "r".into(),
                            byte_offset: 0,
                            datatype: object_ref(),
                        },
                        CompoundMember {
                            name: "i".into(),
                            byte_offset: 8,
                            datatype: i32_le(),
                        },
                    ],
                },
            ),
            (
                "an array of them",
                Datatype::Array {
                    base_type: Box::new(object_ref()),
                    dimensions: vec![2],
                },
            ),
            (
                "a variable length of them",
                Datatype::VariableLength {
                    is_string: false,
                    padding: None,
                    charset: None,
                    base_type: Box::new(object_ref()),
                },
            ),
            (
                "an enumeration over one",
                Datatype::Enumeration {
                    size: 8,
                    base_type: Box::new(object_ref()),
                    members: vec![EnumMember {
                        name: "a".into(),
                        value: vec![0; 8],
                    }],
                },
            ),
            (
                "one nested two deep",
                Datatype::Array {
                    base_type: Box::new(Datatype::Compound {
                        size: 8,
                        members: vec![CompoundMember {
                            name: "r".into(),
                            byte_offset: 0,
                            datatype: object_ref(),
                        }],
                    }),
                    dimensions: vec![3],
                },
            ),
        ];
        for (what, dt) in holds_an_address {
            assert!(
                datatype_holds_object_address(&dt),
                "{what} holds an object address"
            );
            let encoded = dt.serialize();
            assert!(
                class_may_hold_object_address(encoded[0]),
                "{what} encodes as class {}, which the gate rejects — a walk would \
                 never parse it and would pass over the address inside it",
                encoded[0] & 0x0F
            );
        }
    }

    /// The gate is a *necessary* condition and nothing more: it admits types
    /// that hold no address, and that is not a defect. Stated so a later reading
    /// of it as "this type holds a reference" has something to contradict it.
    #[test]
    fn the_class_gate_is_necessary_and_not_sufficient() {
        let ints = Datatype::Compound {
            size: 8,
            members: vec![CompoundMember {
                name: "a".into(),
                byte_offset: 0,
                datatype: Datatype::FixedPoint {
                    size: 8,
                    byte_order: DatatypeByteOrder::LittleEndian,
                    signed: true,
                    bit_offset: 0,
                    bit_precision: 64,
                },
            }],
        };
        assert!(!datatype_holds_object_address(&ints));
        assert!(
            class_may_hold_object_address(ints.serialize()[0]),
            "a compound of integers is admitted by the class gate and holds no address"
        );
    }

    use super::*;

    // Helper to build a fixed-point datatype message
    fn build_fixed_point(
        size: u32,
        be: bool,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    ) -> Vec<u8> {
        let bf0 = if be { 0x01 } else { 0x00 } | if signed { 0x08 } else { 0x00 };
        let mut buf = build_dt_header(0, 1, [bf0, 0, 0], size);
        let mut props = [0u8; 4];
        LittleEndian::write_u16(&mut props[0..2], bit_offset);
        LittleEndian::write_u16(&mut props[2..4], bit_precision);
        buf.extend_from_slice(&props);
        buf
    }

    // Helper to build a floating-point datatype message
    fn build_float(
        size: u32,
        exp_loc: u8,
        exp_size: u8,
        mant_loc: u8,
        mant_size: u8,
        exp_bias: u32,
    ) -> Vec<u8> {
        // LE byte order: bo_low=0, bo_high=0
        let bf0 = 0x00u8;
        let bf1 = 0x00u8;
        // mantissa norm = 2 (MSB not stored) in bits 24-31... wait, that's bf2
        let bf2 = 0x02u8; // norm = 2
        let mut buf = build_dt_header(1, 1, [bf0, bf1, bf2], size);
        let mut props = [0u8; 12];
        LittleEndian::write_u16(&mut props[0..2], 0); // bit_offset
        LittleEndian::write_u16(&mut props[2..4], (size * 8) as u16); // bit_precision
        props[4] = exp_loc;
        props[5] = exp_size;
        props[6] = mant_loc;
        props[7] = mant_size;
        LittleEndian::write_u32(&mut props[8..12], exp_bias);
        buf.extend_from_slice(&props);
        buf
    }

    #[test]
    fn test_fixed_point_u8() {
        let data = build_fixed_point(1, false, false, 0, 8);
        let (dt, consumed) = Datatype::parse(&data).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(
            dt,
            Datatype::FixedPoint {
                size: 1,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: false,
                bit_offset: 0,
                bit_precision: 8,
            }
        );
    }

    #[test]
    fn test_fixed_point_i16_le() {
        let data = build_fixed_point(2, false, true, 0, 16);
        let (dt, _) = Datatype::parse(&data).unwrap();
        assert_eq!(
            dt,
            Datatype::FixedPoint {
                size: 2,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 16,
            }
        );
    }

    #[test]
    fn test_fixed_point_u32_be() {
        let data = build_fixed_point(4, true, false, 0, 32);
        let (dt, _) = Datatype::parse(&data).unwrap();
        match &dt {
            Datatype::FixedPoint {
                byte_order,
                signed,
                size,
                ..
            } => {
                assert_eq!(*byte_order, DatatypeByteOrder::BigEndian);
                assert!(!signed);
                assert_eq!(*size, 4);
            }
            _ => panic!("expected FixedPoint"),
        }
    }

    #[test]
    fn test_fixed_point_i64_le() {
        let data = build_fixed_point(8, false, true, 0, 64);
        let (dt, _) = Datatype::parse(&data).unwrap();
        assert_eq!(
            dt,
            Datatype::FixedPoint {
                size: 8,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 64,
            }
        );
    }

    #[test]
    fn test_float_f32_le() {
        // IEEE 754 f32: exp=8 bits at bit 23, mant=23 bits at bit 0, bias=127
        let data = build_float(4, 23, 8, 0, 23, 127);
        let (dt, consumed) = Datatype::parse(&data).unwrap();
        assert_eq!(consumed, 20);
        assert_eq!(
            dt,
            Datatype::FloatingPoint {
                size: 4,
                byte_order: DatatypeByteOrder::LittleEndian,
                bit_offset: 0,
                bit_precision: 32,
                exponent_location: 23,
                exponent_size: 8,
                mantissa_location: 0,
                mantissa_size: 23,
                exponent_bias: 127,
            }
        );
    }

    #[test]
    fn test_float_f64_le() {
        let data = build_float(8, 52, 11, 0, 52, 1023);
        let (dt, _) = Datatype::parse(&data).unwrap();
        assert_eq!(
            dt,
            Datatype::FloatingPoint {
                size: 8,
                byte_order: DatatypeByteOrder::LittleEndian,
                bit_offset: 0,
                bit_precision: 64,
                exponent_location: 52,
                exponent_size: 11,
                mantissa_location: 0,
                mantissa_size: 52,
                exponent_bias: 1023,
            }
        );
    }

    #[test]
    fn test_string_null_terminated_ascii() {
        let buf = build_dt_header(3, 1, [0x00, 0, 0], 10); // padding=0(nullterm), charset=0(ascii)
        let (dt, consumed) = Datatype::parse(&buf).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(
            dt,
            Datatype::String {
                size: 10,
                padding: StringPadding::NullTerminate,
                charset: CharacterSet::Ascii,
            }
        );
    }

    #[test]
    fn test_string_space_padded_utf8() {
        // padding=2(space pad), charset=1(utf8) → bf0 = 0x12
        let buf = build_dt_header(3, 1, [0x12, 0, 0], 32);
        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(
            dt,
            Datatype::String {
                size: 32,
                padding: StringPadding::SpacePad,
                charset: CharacterSet::Utf8,
            }
        );
    }

    #[test]
    fn test_opaque() {
        // tag_len = 4, tag = "BLOB"
        let mut buf = build_dt_header(5, 1, [4, 0, 0], 64);
        buf.extend_from_slice(b"BLOB");
        // Pad to 8 bytes
        buf.extend_from_slice(&[0, 0, 0, 0]);
        let (dt, consumed) = Datatype::parse(&buf).unwrap();
        assert_eq!(consumed, 16); // 8 header + 8 padded tag
        assert_eq!(
            dt,
            Datatype::Opaque {
                size: 64,
                tag: b"BLOB".to_vec(),
            }
        );
    }

    #[test]
    fn test_compound_v3_two_members() {
        // Compound with size=12, 2 members: "x" u32 at offset 0, "y" f64 at offset 4
        // Size=12, so offset_bytes=1
        let mut buf = build_dt_header(6, 3, [2, 0, 0], 12); // 2 members
        // Member "x": name "x\0", offset=0, then u32 LE datatype
        buf.extend_from_slice(b"x\0");
        buf.push(0); // byte_offset = 0
        buf.extend_from_slice(&build_fixed_point(4, false, false, 0, 32));
        // Member "y": name "y\0", offset=4, then f64 LE datatype
        buf.extend_from_slice(b"y\0");
        buf.push(4); // byte_offset = 4
        buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 12);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "x");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "y");
                assert_eq!(members[1].byte_offset, 4);
                match &members[0].datatype {
                    Datatype::FixedPoint {
                        size: 4,
                        signed: false,
                        ..
                    } => {}
                    other => panic!("expected u32, got {other:?}"),
                }
                match &members[1].datatype {
                    Datatype::FloatingPoint { size: 8, .. } => {}
                    other => panic!("expected f64, got {other:?}"),
                }
            }
            _ => panic!("expected Compound"),
        }
    }

    #[test]
    fn test_compound_v1_complex_matlab_layout() {
        // MATLAB stores a complex value as a version-1 compound of two f64
        // members named "real" and "imag" at offsets 0 and 8. v1 members pad
        // the NUL-terminated name to a multiple of 8 bytes and carry a fixed
        // 28-byte dimension block — dimensionality(1) + reserved(3) +
        // dimension permutation(4) + reserved(4) + dimension sizes(16) —
        // between the byte offset and the member datatype message. Regression
        // test for a stride bug that skipped only 24 bytes (omitting the second
        // reserved field) and so misread every real-MATLAB complex compound.
        let mut buf = build_dt_header(6, 1, [2, 0, 0], 16); // v1, 2 members, size 16
        for (name, offset) in [(&b"real\0\0\0\0"[..], 0u32), (&b"imag\0\0\0\0"[..], 8)] {
            buf.extend_from_slice(name); // NUL-terminated, padded to 8
            let mut off = [0u8; 4];
            LittleEndian::write_u32(&mut off, offset);
            buf.extend_from_slice(&off);
            buf.extend_from_slice(&[0u8; 28]); // v1 dimension block
            buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));
        }

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 16);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "real");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "imag");
                assert_eq!(members[1].byte_offset, 8);
                for m in &members {
                    assert!(
                        matches!(m.datatype, Datatype::FloatingPoint { size: 8, .. }),
                        "expected f64 member, got {:?}",
                        m.datatype
                    );
                }
            }
            _ => panic!("expected Compound"),
        }
    }

    #[test]
    fn test_reference_object() {
        let buf = build_dt_header(7, 1, [0, 0, 0], 8);
        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(
            dt,
            Datatype::Reference {
                size: 8,
                ref_type: ReferenceType::Object,
            }
        );
    }

    #[test]
    fn test_reference_region() {
        let buf = build_dt_header(7, 1, [1, 0, 0], 12);
        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(
            dt,
            Datatype::Reference {
                size: 12,
                ref_type: ReferenceType::DatasetRegion,
            }
        );
    }

    #[test]
    fn test_enumeration() {
        // Enum with base type i32 LE, 3 members
        let mut buf = build_dt_header(8, 3, [3, 0, 0], 4); // 3 members
        // Base type: i32 LE
        buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));
        // Names: "RED\0", "GREEN\0", "BLUE\0"
        buf.extend_from_slice(b"RED\0");
        buf.extend_from_slice(b"GREEN\0");
        buf.extend_from_slice(b"BLUE\0");
        // Values: 0, 1, 2 (as i32 LE)
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Enumeration {
                size,
                base_type,
                members,
            } => {
                assert_eq!(size, 4);
                assert_eq!(members.len(), 3);
                assert_eq!(members[0].name, "RED");
                assert_eq!(members[0].value, 0i32.to_le_bytes().to_vec());
                assert_eq!(members[1].name, "GREEN");
                assert_eq!(members[1].value, 1i32.to_le_bytes().to_vec());
                assert_eq!(members[2].name, "BLUE");
                assert_eq!(members[2].value, 2i32.to_le_bytes().to_vec());
                match *base_type {
                    Datatype::FixedPoint {
                        signed: true,
                        size: 4,
                        ..
                    } => {}
                    other => panic!("expected i32, got {other:?}"),
                }
            }
            _ => panic!("expected Enumeration"),
        }
    }

    #[test]
    fn test_variable_length_string_utf8() {
        // VL string: type=1, padding=0(null term), charset=1(utf8)
        // bf0: bits 0-3 = 1 (string), bits 4-7 = 0 (null term) → 0x01
        // bf1: bits 0-3 = 1 (utf8) → 0x01
        let mut buf = build_dt_header(9, 1, [0x01, 0x01, 0], 16);
        // Base type: u8 (class 0, unsigned, size 1)
        buf.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::VariableLength {
                is_string,
                padding,
                charset,
                base_type,
            } => {
                assert!(is_string);
                assert_eq!(padding, Some(StringPadding::NullTerminate));
                assert_eq!(charset, Some(CharacterSet::Utf8));
                assert_eq!(base_type.type_size(), 1);
            }
            _ => panic!("expected VariableLength"),
        }
    }

    #[test]
    fn test_variable_length_sequence_f32() {
        // VL sequence: type=0
        // bf0 = 0x00
        let mut buf = build_dt_header(9, 1, [0x00, 0x00, 0], 16);
        // Base type: f32 LE
        buf.extend_from_slice(&build_float(4, 23, 8, 0, 23, 127));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::VariableLength {
                is_string,
                padding,
                charset,
                base_type,
            } => {
                assert!(!is_string);
                assert_eq!(padding, None);
                assert_eq!(charset, None);
                assert_eq!(base_type.type_size(), 4);
            }
            _ => panic!("expected VariableLength"),
        }
    }

    /// HDF5 2.0 writes every datatype message with version 5 under its latest
    /// library bounds. For a compound the layout is version 3's: unpadded
    /// member names and offsets in the fewest bytes the size needs.
    #[test]
    fn a_version_5_compound_parses_like_version_3() {
        let members = |version: u8| {
            let mut buf = build_dt_header(6, version, [2, 0, 0], 16);
            buf.extend_from_slice(b"re\0");
            buf.push(0);
            buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));
            buf.extend_from_slice(b"im\0");
            buf.push(8);
            buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));
            buf
        };
        let (expected, expected_len) = Datatype::parse(&members(3)).unwrap();
        let (parsed, len) = Datatype::parse(&members(5)).unwrap();
        assert_eq!((parsed, len), (expected, expected_len));
    }

    /// The same for an array: version 5 keeps version 3's layout.
    #[test]
    fn a_version_5_array_parses_like_version_3() {
        let array = |version: u8| {
            let mut buf = build_dt_header(10, version, [0, 0, 0], 48);
            buf.push(2);
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));
            buf
        };
        let (expected, expected_len) = Datatype::parse(&array(3)).unwrap();
        let (parsed, len) = Datatype::parse(&array(5)).unwrap();
        assert_eq!((parsed, len), (expected, expected_len));
    }

    #[test]
    fn test_array_2d() {
        // Array [3][4] of i32 LE, version 3
        let mut buf = build_dt_header(10, 3, [0, 0, 0], 48); // 3*4*4=48
        buf.push(2); // ndims=2
        buf.extend_from_slice(&3u32.to_le_bytes()); // dim 0
        buf.extend_from_slice(&4u32.to_le_bytes()); // dim 1
        // Base type: i32 LE
        buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                assert_eq!(dimensions, vec![3, 4]);
                match *base_type {
                    Datatype::FixedPoint {
                        size: 4,
                        signed: true,
                        ..
                    } => {}
                    other => panic!("expected i32, got {other:?}"),
                }
            }
            _ => panic!("expected Array"),
        }
    }

    /// Nothing in HDF5 occupies zero bytes per element, and the readers divide by
    /// the element size, so a declared zero is refused where an untrusted message
    /// becomes a `Datatype` rather than at each division (issue #268).
    #[test]
    fn a_zero_width_element_type_is_refused() {
        let buf = build_dt_header(3, 1, [0x01, 0, 0], 0); // fixed-length string of 0 bytes
        assert_eq!(
            Datatype::parse(&buf).unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 3 }
        );
    }

    /// An array's element size is its base type across its dimensions, not the
    /// size the header declares, and the two disagree: a zero dimension is a
    /// zero-width element behind a header that claims 48 bytes. Reading the
    /// declared field instead of the computed one lets this one through.
    #[test]
    fn an_array_with_a_zero_dimension_is_refused_despite_its_header_size() {
        let mut buf = build_dt_header(10, 3, [0, 0, 0], 48);
        buf.push(2); // ndims=2
        buf.extend_from_slice(&0u32.to_le_bytes()); // dim 0 — no elements
        buf.extend_from_slice(&4u32.to_le_bytes()); // dim 1
        buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));

        assert_eq!(
            Datatype::parse(&buf).unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 10 }
        );
    }

    /// The refusal reaches a nested type too: a compound member is parsed through
    /// the same entry, so a zero-width member is caught where it is decoded rather
    /// than becoming a member whose size no reader can use.
    #[test]
    fn a_zero_width_compound_member_is_refused() {
        let member = build_dt_header(3, 1, [0x01, 0, 0], 0);
        let mut buf = build_dt_header(6, 3, [1, 0, 0], 8); // one member
        buf.extend_from_slice(b"s\0");
        buf.push(0); // byte offset, one byte for a size-8 compound
        buf.extend_from_slice(&member);

        assert_eq!(
            Datatype::parse(&buf).unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 3 }
        );
    }

    /// The accessor the rest of the crate uses agrees with `type_size` for an
    /// ordinary type. The point of the pair is that one of them carries a proof
    /// and the other does not — not that they report different widths.
    #[test]
    fn element_size_matches_type_size_for_a_type_that_has_one() {
        let dt = Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        assert_eq!(dt.element_size().unwrap().get(), dt.type_size());
    }

    /// A caller-built literal never passes through `parse`, so `element_size` is
    /// the only thing standing between a degenerate type and the writers. The
    /// `Array` case is the one that matters: its width is *computed* from its
    /// dimensions, so this cannot be caught by inspecting a stored size field.
    #[test]
    fn element_size_refuses_a_constructed_array_with_a_zero_dimension() {
        let dt = Datatype::Array {
            base_type: Box::new(Datatype::FixedPoint {
                size: 4,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            }),
            dimensions: vec![0, 4],
        };
        assert_eq!(dt.type_size(), 0);
        assert_eq!(
            dt.element_size().unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 10 }
        );
        assert_eq!(
            dt.element_size_usize().unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 10 }
        );
    }

    /// The class in the error names the type that was refused, not the base type
    /// underneath it, so a report points at the message the writer was handed.
    #[test]
    fn element_size_reports_the_refused_types_own_class() {
        let dt = Datatype::Compound {
            size: 0,
            members: vec![],
        };
        assert_eq!(
            dt.element_size().unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 6 }
        );
    }

    #[test]
    fn test_bitfield() {
        let mut buf = build_dt_header(4, 1, [0, 0, 0], 2); // 16-bit LE bitfield
        let mut props = [0u8; 4];
        LittleEndian::write_u16(&mut props[0..2], 0);
        LittleEndian::write_u16(&mut props[2..4], 16);
        buf.extend_from_slice(&props);

        let (dt, _) = Datatype::parse(&buf).unwrap();
        assert_eq!(
            dt,
            Datatype::BitField {
                size: 2,
                byte_order: DatatypeByteOrder::LittleEndian,
                bit_offset: 0,
                bit_precision: 16,
            }
        );
    }

    #[test]
    fn test_time() {
        let mut buf = build_dt_header(2, 1, [0, 0, 0], 8);
        let mut props = [0u8; 2];
        LittleEndian::write_u16(&mut props[0..2], 64);
        buf.extend_from_slice(&props);

        let (dt, consumed) = Datatype::parse(&buf).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(
            dt,
            Datatype::Time {
                size: 8,
                byte_order: DatatypeByteOrder::LittleEndian,
                bit_precision: 64,
            }
        );
    }

    #[test]
    fn test_time_byte_order_roundtrips() {
        // A big-endian time type must serialize and re-parse with its byte order
        // preserved (bf0 bit 0), so repack can reproduce it faithfully.
        for (be, order) in [
            (0u8, DatatypeByteOrder::LittleEndian),
            (1u8, DatatypeByteOrder::BigEndian),
        ] {
            let mut buf = build_dt_header(2, 1, [be, 0, 0], 4);
            buf.extend_from_slice(&32u16.to_le_bytes());
            let (dt, _) = Datatype::parse(&buf).unwrap();
            assert_eq!(
                dt,
                Datatype::Time {
                    size: 4,
                    byte_order: order.clone(),
                    bit_precision: 32,
                }
            );
            // serialize -> parse must round-trip the byte order.
            let (reparsed, _) = Datatype::parse(&dt.serialize()).unwrap();
            assert_eq!(reparsed, dt);
        }
    }

    #[test]
    fn test_nested_compound_array_enum() {
        // Compound containing a single member "data" which is an Array[2] of Enum(i32, 2 values)
        // Build the enum first
        let mut enum_bytes = build_dt_header(8, 3, [2, 0, 0], 4); // 2 members
        enum_bytes.extend_from_slice(&build_fixed_point(4, false, true, 0, 32)); // base i32
        enum_bytes.extend_from_slice(b"A\0");
        enum_bytes.extend_from_slice(b"B\0");
        enum_bytes.extend_from_slice(&0i32.to_le_bytes());
        enum_bytes.extend_from_slice(&1i32.to_le_bytes());

        // Build array[2] of that enum, version 3
        let mut array_bytes = build_dt_header(10, 3, [0, 0, 0], 8); // 2*4=8
        array_bytes.push(1); // ndims=1
        array_bytes.extend_from_slice(&2u32.to_le_bytes()); // dim[0]=2
        array_bytes.extend_from_slice(&enum_bytes);

        // Build compound with 1 member, size=8
        let mut buf = build_dt_header(6, 3, [1, 0, 0], 8); // 1 member
        buf.extend_from_slice(b"data\0");
        buf.push(0); // byte_offset = 0 (size=8, so 1 byte offsets)
        buf.extend_from_slice(&array_bytes);

        let (dt, _) = Datatype::parse(&buf).unwrap();
        match dt {
            Datatype::Compound { members, .. } => {
                assert_eq!(members.len(), 1);
                assert_eq!(members[0].name, "data");
                match &members[0].datatype {
                    Datatype::Array {
                        dimensions,
                        base_type,
                    } => {
                        assert_eq!(dimensions, &[2]);
                        match base_type.as_ref() {
                            Datatype::Enumeration { members, .. } => {
                                assert_eq!(members.len(), 2);
                                assert_eq!(members[0].name, "A");
                                assert_eq!(members[1].name, "B");
                            }
                            other => panic!("expected Enum, got {other:?}"),
                        }
                    }
                    other => panic!("expected Array, got {other:?}"),
                }
            }
            _ => panic!("expected Compound"),
        }
    }

    #[test]
    fn test_error_invalid_class() {
        let buf = build_dt_header(13, 1, [0, 0, 0], 4);
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidDatatypeClass(13));
    }

    #[test]
    fn test_error_truncated_data() {
        let buf = [0u8; 4]; // too short for header
        let err = Datatype::parse(&buf).unwrap_err();
        match err {
            FormatError::UnexpectedEof { .. } => {}
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
    }

    #[test]
    fn test_error_invalid_string_padding() {
        let buf = build_dt_header(3, 1, [0x03, 0, 0], 10); // padding=3 invalid
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidStringPadding(3));
    }

    #[test]
    fn test_error_invalid_charset() {
        let buf = build_dt_header(3, 1, [0x20, 0, 0], 10); // charset=2 invalid
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidCharacterSet(2));
    }

    #[test]
    fn test_error_invalid_reference_type() {
        let buf = build_dt_header(7, 1, [5, 0, 0], 8);
        let err = Datatype::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidReferenceType(5));
    }

    #[test]
    fn serialize_parse_compound_roundtrip() {
        let dt = Datatype::Compound {
            size: 20,
            members: vec![
                CompoundMember {
                    name: "x".to_string(),
                    byte_offset: 0,
                    datatype: Datatype::FloatingPoint {
                        size: 8,
                        byte_order: DatatypeByteOrder::LittleEndian,
                        bit_offset: 0,
                        bit_precision: 64,
                        exponent_location: 52,
                        exponent_size: 11,
                        mantissa_location: 0,
                        mantissa_size: 52,
                        exponent_bias: 1023,
                    },
                },
                CompoundMember {
                    name: "y".to_string(),
                    byte_offset: 8,
                    datatype: Datatype::FloatingPoint {
                        size: 8,
                        byte_order: DatatypeByteOrder::LittleEndian,
                        bit_offset: 0,
                        bit_precision: 64,
                        exponent_location: 52,
                        exponent_size: 11,
                        mantissa_location: 0,
                        mantissa_size: 52,
                        exponent_bias: 1023,
                    },
                },
                CompoundMember {
                    name: "id".to_string(),
                    byte_offset: 16,
                    datatype: Datatype::FixedPoint {
                        size: 4,
                        byte_order: DatatypeByteOrder::LittleEndian,
                        signed: true,
                        bit_offset: 0,
                        bit_precision: 32,
                    },
                },
            ],
        };
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn serialize_parse_enum_roundtrip() {
        let dt = Datatype::Enumeration {
            size: 4,
            base_type: Box::new(Datatype::FixedPoint {
                size: 4,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            }),
            members: vec![
                EnumMember {
                    name: "RED".to_string(),
                    value: 0i32.to_le_bytes().to_vec(),
                },
                EnumMember {
                    name: "GREEN".to_string(),
                    value: 1i32.to_le_bytes().to_vec(),
                },
                EnumMember {
                    name: "BLUE".to_string(),
                    value: 2i32.to_le_bytes().to_vec(),
                },
            ],
        };
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    /// Fixed-point base type for enum round-trip tests.
    fn enum_base_fp(size: u32, be: bool, signed: bool) -> Datatype {
        Datatype::FixedPoint {
            size,
            byte_order: if be {
                DatatypeByteOrder::BigEndian
            } else {
                DatatypeByteOrder::LittleEndian
            },
            signed,
            bit_offset: 0,
            #[expect(
                clippy::cast_possible_truncation,
                reason = "test builds byte-width base types; size*8 is well within u16"
            )]
            bit_precision: (size * 8) as u16,
        }
    }

    /// Build an enum datatype over `base`, storing each member value truncated to
    /// the base width (the value blob is opaque bytes, so any content round-trips).
    fn make_enum(base: Datatype, members: &[(&str, i64)]) -> Datatype {
        let size = base.type_size();
        let width = size as usize;
        Datatype::Enumeration {
            size,
            base_type: Box::new(base),
            members: members
                .iter()
                .map(|(name, v)| EnumMember {
                    name: (*name).to_string(),
                    value: v.to_le_bytes()[..width].to_vec(),
                })
                .collect(),
        }
    }

    #[test]
    fn serialize_parse_enum_base_type_variety() {
        // The i32 base is already covered above; here u8, big-endian i16, and i64
        // bases all round-trip through the enum wrapper.
        for base in [
            enum_base_fp(1, false, false), // u8
            enum_base_fp(2, true, true),   // i16 big-endian
            enum_base_fp(8, false, true),  // i64
        ] {
            let dt = make_enum(base.clone(), &[("A", 0), ("B", 1), ("NEG", -1)]);
            let bytes = dt.serialize();
            let (parsed, consumed) = Datatype::parse(&bytes).unwrap();
            assert_eq!(parsed, dt, "round-trip failed for base {base:?}");
            assert_eq!(consumed, bytes.len());
        }
    }

    #[test]
    fn serialize_parse_enum_large_member_count() {
        // More than 256 members exercises the 2-byte member-count field, which is
        // split across bf0/bf1 in the datatype message header.
        let owned: Vec<(String, i64)> = (0..300).map(|i| (format!("M{i}"), i)).collect();
        let members: Vec<(&str, i64)> = owned.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let dt = make_enum(enum_base_fp(4, false, true), &members);
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
        match parsed {
            Datatype::Enumeration { members, .. } => {
                assert_eq!(members.len(), 300);
                assert_eq!(members[299].name, "M299");
            }
            other => panic!("expected Enumeration, got {other:?}"),
        }
    }

    #[test]
    fn enum_value_width_is_not_validated_against_base_size() {
        // `EnumTypeBuilder::build`/`Datatype::Enumeration` take the element size
        // from the base type only, with no check that member value blobs match it.
        // A 4-byte value on a 1-byte base therefore serializes in full but parses
        // back reading just `base_size` (1) byte per member, silently truncating.
        // This documents the current permissiveness; it is NOT a supported
        // round-trip, and the assertion guards against a silent change either way.
        let dt = Datatype::Enumeration {
            size: 1,
            base_type: Box::new(enum_base_fp(1, false, false)),
            members: vec![EnumMember {
                name: "X".to_string(),
                value: 5i32.to_le_bytes().to_vec(), // 4 bytes on a 1-byte base
            }],
        };
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_ne!(
            parsed, dt,
            "a value wider than the base silently truncates on parse"
        );
        match parsed {
            Datatype::Enumeration { members, .. } => assert_eq!(members[0].value, vec![5]),
            other => panic!("expected Enumeration, got {other:?}"),
        }
    }

    #[test]
    fn serialize_parse_array_roundtrip() {
        let dt = Datatype::Array {
            base_type: Box::new(Datatype::FloatingPoint {
                size: 8,
                byte_order: DatatypeByteOrder::LittleEndian,
                bit_offset: 0,
                bit_precision: 64,
                exponent_location: 52,
                exponent_size: 11,
                mantissa_location: 0,
                mantissa_size: 52,
                exponent_bias: 1023,
            }),
            dimensions: vec![3],
        };
        let bytes = dt.serialize();
        let (parsed, _) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn serialize_parse_time_roundtrip() {
        let dt = Datatype::Time {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_precision: 64,
        };
        let bytes = dt.serialize();
        let (parsed, consumed) = Datatype::parse(&bytes).unwrap();
        assert_eq!(parsed, dt);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn serialize_parse_bitfield_roundtrip() {
        for byte_order in [
            DatatypeByteOrder::LittleEndian,
            DatatypeByteOrder::BigEndian,
        ] {
            let dt = Datatype::BitField {
                size: 4,
                byte_order,
                bit_offset: 3,
                bit_precision: 17,
            };
            let bytes = dt.serialize();
            let (parsed, consumed) = Datatype::parse(&bytes).unwrap();
            assert_eq!(parsed, dt);
            assert_eq!(consumed, bytes.len());
        }
    }

    #[test]
    fn serialize_parse_opaque_roundtrip() {
        // Tag lengths that do and do not land on an 8-byte boundary, to exercise
        // the zero padding both ways.
        for tag in [
            b"abc".to_vec(),         // 3 bytes -> padded to 8
            b"12345678".to_vec(),    // 8 bytes -> no padding
            b"sensor-id\0".to_vec(), // 10 bytes -> padded to 16, embedded NUL preserved
        ] {
            let dt = Datatype::Opaque { size: 16, tag };
            let bytes = dt.serialize();
            // The property section (after the 8-byte header) must be a multiple
            // of 8, matching what the reference library expects.
            assert_eq!((bytes.len() - 8) % 8, 0);
            let (parsed, consumed) = Datatype::parse(&bytes).unwrap();
            assert_eq!(parsed, dt);
            assert_eq!(consumed, bytes.len());
        }
    }

    #[test]
    fn test_type_size() {
        let dt = Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        assert_eq!(dt.type_size(), 4);

        let dt = Datatype::Array {
            base_type: Box::new(Datatype::FixedPoint {
                size: 4,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            }),
            dimensions: vec![3, 4],
        };
        assert_eq!(dt.type_size(), 48);
    }
}

#[cfg(all(test, feature = "std"))]
mod display_tests {
    use super::*;

    #[test]
    fn ordinary_numeric_types_read_as_their_rust_names() {
        let int = Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        assert_eq!(int.to_string(), "i32");

        let float = Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        assert_eq!(float.to_string(), "f64");
    }

    /// Every width a message writes is `size * 8` over an on-disk `u32`, so a
    /// crafted size near [`u32::MAX`] overflows a `u32` multiply and panics a
    /// debug build (issue #140). [`bit_width`] widens first; this holds each
    /// class that calls it to that, rather than reaching one of them through
    /// whatever `classify_datatype` happens to route here.
    #[test]
    fn a_crafted_size_writes_its_width_instead_of_overflowing() {
        let bits = u64::from(u32::MAX) * 8;
        let cases = [
            (
                Datatype::FixedPoint {
                    size: u32::MAX,
                    byte_order: DatatypeByteOrder::LittleEndian,
                    signed: true,
                    bit_offset: 0,
                    bit_precision: 0,
                },
                format!("i{bits}(bits 0..0)"),
            ),
            (
                Datatype::FloatingPoint {
                    size: u32::MAX,
                    byte_order: DatatypeByteOrder::LittleEndian,
                    bit_offset: 0,
                    bit_precision: 0,
                    exponent_location: 0,
                    exponent_size: 0,
                    mantissa_location: 0,
                    mantissa_size: 0,
                    exponent_bias: 0,
                },
                format!("f{bits}(bits 0..0)"),
            ),
            (
                Datatype::Time {
                    size: u32::MAX,
                    byte_order: DatatypeByteOrder::LittleEndian,
                    bit_precision: 0,
                },
                format!("time{bits}(bits 0..0)"),
            ),
            (
                Datatype::BitField {
                    size: u32::MAX,
                    byte_order: DatatypeByteOrder::LittleEndian,
                    bit_offset: 0,
                    bit_precision: 0,
                },
                format!("bitfield{bits}(bits 0..0)"),
            ),
        ];

        for (dtype, expected) in cases {
            assert_eq!(dtype.to_string(), expected);
        }
    }

    /// The bit span adds two `u16`s, which is the other place a crafted field
    /// could wrap. Both widen, so the end is 131,070 rather than 65,534.
    #[test]
    fn a_crafted_bit_span_does_not_wrap() {
        let dtype = Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: u16::MAX,
            bit_precision: u16::MAX,
        };
        assert_eq!(dtype.to_string(), "u8(bits 65535..131070)");
    }

    /// Only what departs from the ordinary is written, since that is what the
    /// reader of the message is looking for.
    #[test]
    fn unusual_fields_are_written_and_ordinary_ones_are_not() {
        let big_endian = Datatype::FixedPoint {
            size: 2,
            byte_order: DatatypeByteOrder::BigEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 16,
        };
        assert_eq!(big_endian.to_string(), "u16 be");

        let narrow = Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 24,
        };
        assert_eq!(narrow.to_string(), "i32(bits 0..24)");
    }

    #[test]
    fn nested_types_recurse_through_their_members() {
        let compound = Datatype::Compound {
            size: 12,
            members: vec![
                CompoundMember {
                    name: "x".into(),
                    byte_offset: 0,
                    datatype: Datatype::FloatingPoint {
                        size: 4,
                        byte_order: DatatypeByteOrder::LittleEndian,
                        bit_offset: 0,
                        bit_precision: 32,
                        exponent_location: 23,
                        exponent_size: 8,
                        mantissa_location: 0,
                        mantissa_size: 23,
                        exponent_bias: 127,
                    },
                },
                CompoundMember {
                    name: "n".into(),
                    byte_offset: 4,
                    datatype: Datatype::FixedPoint {
                        size: 8,
                        byte_order: DatatypeByteOrder::LittleEndian,
                        signed: true,
                        bit_offset: 0,
                        bit_precision: 64,
                    },
                },
            ],
        };
        assert_eq!(compound.to_string(), "compound{x: f32, n: i64}");

        let array = Datatype::Array {
            base_type: Box::new(Datatype::FixedPoint {
                size: 1,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: false,
                bit_offset: 0,
                bit_precision: 8,
            }),
            dimensions: vec![2, 3],
        };
        assert_eq!(
            array.to_string(),
            "array<u8, 2x3>",
            "the shape is spelled `2x3`, never a `Debug` slice"
        );
    }

    /// The leaf enums format through `Formatter::pad`, so a caller lining these
    /// up in a column gets the width it asked for rather than having it
    /// silently dropped.
    #[test]
    fn a_leaf_enum_honors_the_width_it_is_given() {
        assert_eq!(format!("{:>8}", CharacterSet::Ascii), "   ascii");
        assert_eq!(format!("{:<8}|", DatatypeByteOrder::BigEndian), "be      |");
        assert_eq!(format!("{}", StringPadding::NullPad), "null-pad");
    }

    #[test]
    fn a_string_carries_its_width_charset_and_padding() {
        let string = Datatype::String {
            size: 16,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Utf8,
        };
        assert_eq!(string.to_string(), "string[16] utf8 null-pad");
    }

    /// The tag is arbitrary file bytes, so it cannot reach a message unescaped.
    #[test]
    fn an_opaque_tag_is_quoted_and_escaped() {
        let opaque = Datatype::Opaque {
            size: 4,
            tag: b"a\"b\x00".to_vec(),
        };
        assert_eq!(opaque.to_string(), "opaque[4] \"a\\\"b\\x00\"");
    }

    /// A member name comes from the file by way of `from_utf8_lossy`, which
    /// rejects nothing, so it is escaped for the same reason an opaque tag is.
    #[test]
    fn a_member_name_cannot_carry_a_control_character_into_a_message() {
        let compound = Datatype::Compound {
            size: 4,
            members: vec![CompoundMember {
                name: "a\nb\u{1b}[31m".into(),
                byte_offset: 0,
                datatype: u32_datatype(),
            }],
        };
        let shown = compound.to_string();
        assert!(!shown.chars().any(char::is_control), "{shown}");
        assert_eq!(shown, "compound{a\\nb\\u{1b}[31m: u32}");

        let enumeration = Datatype::Enumeration {
            size: 4,
            base_type: Box::new(u32_datatype()),
            members: vec![EnumMember {
                name: "red\u{0}".into(),
                value: vec![0, 0, 0, 0],
            }],
        };
        let shown = enumeration.to_string();
        assert!(!shown.chars().any(char::is_control), "{shown}");
        assert_eq!(shown, "enum<u32>[red\\0]");
    }

    /// The member count is an on-disk `u16`, so the list a file can ask for is
    /// far longer than a message can carry. Both member-bearing variants elide,
    /// so both are checked.
    #[test]
    fn a_long_member_list_is_elided_and_reports_the_remainder() {
        let over_cap = DISPLAY_MAX_MEMBERS + 3;

        let compound = Datatype::Compound {
            size: (over_cap * 4) as u32,
            members: (0..over_cap)
                .map(|i| CompoundMember {
                    name: format!("m{i}"),
                    byte_offset: (i * 4) as u64,
                    datatype: u32_datatype(),
                })
                .collect(),
        };
        let enumeration = Datatype::Enumeration {
            size: 4,
            base_type: Box::new(u32_datatype()),
            members: (0..over_cap)
                .map(|i| EnumMember {
                    name: format!("m{i}"),
                    value: vec![0, 0, 0, 0],
                })
                .collect(),
        };

        for (datatype, close) in [(compound, "}"), (enumeration, "]")] {
            let shown = datatype.to_string();
            assert!(shown.ends_with(&format!(", … 3 more{close}")), "{shown}");
            assert!(shown.contains("m0"), "{shown}");
            assert!(
                !shown.contains(&format!("m{DISPLAY_MAX_MEMBERS}")),
                "{shown}"
            );
        }
    }

    /// The boundary: exactly the cap is written whole, with no "0 more".
    #[test]
    fn a_member_list_at_exactly_the_cap_is_not_elided() {
        let members: Vec<_> = (0..DISPLAY_MAX_MEMBERS)
            .map(|i| EnumMember {
                name: format!("m{i}"),
                value: vec![0, 0, 0, 0],
            })
            .collect();
        let shown = Datatype::Enumeration {
            size: 4,
            base_type: Box::new(u32_datatype()),
            members,
        }
        .to_string();

        assert!(!shown.contains('…'), "{shown}");
        assert!(
            shown.ends_with(&format!("m{}]", DISPLAY_MAX_MEMBERS - 1)),
            "{shown}"
        );
    }

    fn u32_datatype() -> Datatype {
        Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 32,
        }
    }
}
