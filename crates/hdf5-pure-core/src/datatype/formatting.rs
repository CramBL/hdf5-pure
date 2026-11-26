use core::fmt;

use super::byte_order::DatatypeByteOrder;
use super::definition::Datatype;
use super::variable_length::VariableLengthType;
use crate::display::{self, DISPLAY_MAX_MEMBERS, Dims, QuotedBytes};

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

/// The width in bits of a `size`-byte type.
///
/// Uses a `u64` product to represent the bit width of any `u32` byte size.
fn bit_width(size: u32) -> u64 {
    u64::from(size) * 8
}

/// Writes the significant bit range when it occupies part of the element.
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

/// Writes a byte-order suffix for big-endian or VAX datatypes.
fn write_byte_order(f: &mut fmt::Formatter<'_>, byte_order: &DatatypeByteOrder) -> fmt::Result {
    if *byte_order != DatatypeByteOrder::LittleEndian {
        write!(f, " {byte_order}")?;
    }
    Ok(())
}
