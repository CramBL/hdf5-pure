//! Simplified type representations for the high-level API.

use std::collections::HashMap;
use std::fmt;

#[cfg(test)]
use crate::VariableLengthStringType;
use crate::VariableLengthType;
use crate::address::AddressContext;
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
#[cfg(test)]
use crate::{FixedPointType, FloatingPointType, TimeType};

use crate::display::{self, DISPLAY_MAX_MEMBERS, Dims, EscapedName};
use crate::width::FormatWidths;

pub use crate::file_writer::AttrValue;

/// Simplified datatype enum for the high-level API.
///
/// `DType` maps a parsed [`Datatype`] onto a coarser enum and implements
/// [`Display`](core::fmt::Display), which gives a short summary of a dataset's type without a
/// match over the full one.
///
/// | Variant | Meaning |
/// |---|---|
/// | [`F32`](Self::F32) / [`F64`](Self::F64) | 4-/8-byte float |
/// | [`I8`](Self::I8) / [`I16`](Self::I16) / [`I32`](Self::I32) / [`I64`](Self::I64) | signed integers |
/// | [`U8`](Self::U8) / [`U16`](Self::U16) / [`U32`](Self::U32) / [`U64`](Self::U64) | unsigned integers |
/// | [`String`](Self::String) | fixed-length string |
/// | [`VariableLengthString`](Self::VariableLengthString) | variable-length string |
/// | [`ObjectReference`](Self::ObjectReference) | HDF5 object reference (8-byte address) |
/// | [`Compound`](Self::Compound) | compound with classified fields |
/// | [`Enum`](Self::Enum) | enumeration with member names |
/// | [`Array`](Self::Array) | fixed-size array of a base type |
/// | [`Other`](Self::Other) | anything not classified above, carrying the type itself |
///
/// `DType` is the quick human-readable summary, and
/// [`Dataset::datatype`](crate::Dataset::datatype) reports exact field offsets, bit precision, and
/// byte order. A type nested in a compound field or an array base has no such fallback, which is
/// why [`Other`](Self::Other) carries the [`Datatype`] itself.
///
/// Non-exhaustive: variants are added as this crate supports more datatypes, so
/// match with a `_` arm.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    String,
    Compound(Vec<(std::string::String, DType)>),
    Enum(Vec<std::string::String>),
    Array(Box<DType>, Vec<u32>),
    VariableLengthString,
    /// HDF5 object reference (8-byte address).
    ObjectReference,
    /// A datatype outside the simplified variants.
    ///
    /// Retains the complete [`Datatype`] at the top level and inside compound
    /// fields or arrays.
    Other(Box<Datatype>),
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::String => write!(f, "string"),
            DType::VariableLengthString => write!(f, "vlen_string"),
            DType::ObjectReference => write!(f, "object_ref"),
            DType::Compound(fields) => {
                write!(f, "compound{{")?;
                for (i, (name, dt)) in fields.iter().take(DISPLAY_MAX_MEMBERS).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {dt}", EscapedName(name))?;
                }
                display::write_elided(f, fields.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                write!(f, "}}")
            }
            DType::Enum(names) => {
                write!(f, "enum[")?;
                for (i, name) in names.iter().take(DISPLAY_MAX_MEMBERS).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", EscapedName(name))?;
                }
                display::write_elided(f, names.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                write!(f, "]")
            }
            DType::Array(base, dims) => write!(f, "array<{base}, {}>", Dims(dims)),
            DType::Other(dt) => write!(f, "other({dt})"),
        }
    }
}

/// Classifies a parsed [`Datatype`] into the simplified [`DType`] variants.
pub(crate) fn classify_datatype(dt: &Datatype) -> DType {
    match dt {
        Datatype::FloatingPoint(numeric) if numeric.size().get() == 4 => DType::F32,
        Datatype::FloatingPoint(numeric) if numeric.size().get() == 8 => DType::F64,
        Datatype::FixedPoint(numeric) => match (numeric.size().get(), numeric.layout().signed()) {
            (1, true) => DType::I8,
            (2, true) => DType::I16,
            (4, true) => DType::I32,
            (8, true) => DType::I64,
            (1, false) => DType::U8,
            (2, false) => DType::U16,
            (4, false) => DType::U32,
            (8, false) => DType::U64,
            _ => DType::Other(Box::new(dt.clone())),
        },
        Datatype::String { .. } => DType::String,
        Datatype::VariableLength(VariableLengthType::String(_)) => DType::VariableLengthString,
        Datatype::Compound { members, .. } => {
            let fields = members
                .iter()
                .map(|m| (m.name.clone(), classify_datatype(&m.datatype)))
                .collect();
            DType::Compound(fields)
        }
        Datatype::Enumeration { members, .. } => {
            let names = members.iter().map(|m| m.name.clone()).collect();
            DType::Enum(names)
        }
        Datatype::Array {
            base_type,
            dimensions,
        } => DType::Array(Box::new(classify_datatype(base_type)), dimensions.clone()),
        Datatype::Reference {
            ref_type: crate::datatype::ReferenceType::Object,
            ..
        } => DType::ObjectReference,
        // The type itself, not a rendering of it: this is the only view a
        // caller gets of a member or an array base that the curated set has no
        // name for.
        _ => DType::Other(Box::new(dt.clone())),
    }
}

/// Decodes attribute messages into a name-to-value map.
///
/// Attributes that cannot be decoded are omitted. `addresses` resolves the stored
/// collection addresses of variable-length data to physical positions in `source`.
pub(crate) fn attrs_to_map<S: crate::source::Source + ?Sized>(
    attrs: &[crate::attribute::AttributeMessage],
    source: &S,
    widths: FormatWidths,
    addresses: AddressContext,
) -> HashMap<std::string::String, AttrValue> {
    let mut map = HashMap::new();
    for attr in attrs {
        if let Some(val) = decode_attr_value(attr, source, widths, addresses) {
            map.insert(attr.name.clone(), val);
        }
    }
    map
}

/// Decodes an attribute's datatype, dataspace, and raw bytes into an [`AttrValue`].
///
/// Scalar dataspaces use scalar variants, and other dataspaces use array variants.
/// One-element arrays remain arrays. Character-set metadata selects the ASCII
/// or UTF-8 string variants. Integer variants preserve 1-, 2-, 4-, and 8-byte
/// widths. Other supported integer widths use a 64-bit variant. Four-byte floats
/// use [`AttrValue::F32`], and eight-byte floats use [`AttrValue::F64`].
///
/// The decoded value omits some encoding details:
///
/// - byte order and numeric precision, which are rewritten at the variant's full width
/// - enumeration member names, with values decoded through the integer base type
/// - array rank, with elements flattened into one-dimensional variants
/// - fixed-string padding, which is rewritten as null padding
/// - null dataspace identity, which has the same empty array value as an empty simple dataspace
///
/// Fixed-string variants preserve stored widths larger than their decoded content.
/// Lossy decoding can expand the text beyond its stored width, in which case the
/// plain variant is used. A scalar sequence of one-byte ASCII strings also uses
/// the plain fixed-string variant. Its array form uses
/// [`AttrValue::VarLenAsciiCharArray`]. Standard variable-length strings retain
/// their variable-length representation for both scalars and arrays.
///
/// Numeric arrays contain the complete elements decoded from the supplied bytes.
/// An incomplete element, an empty numeric scalar, or an unsupported datatype
/// returns `None`. Numeric elements wider than eight bytes are unsupported.
/// A zero-size fixed-string scalar decodes to an empty string.
fn decode_attr_value<S: crate::source::Source + ?Sized>(
    attr: &crate::attribute::AttributeMessage,
    source: &S,
    widths: FormatWidths,
    addresses: AddressContext,
) -> Option<AttrValue> {
    use crate::dataspace::DataspaceType;
    use crate::datatype::Datatype;

    // A scalar dataspace and a 1-element simple dataspace are different on
    // disk (v1 carries rank 0, v2 a type byte), and the write side picks
    // between them per variant, so this is what makes the round trip faithful
    // at length one.
    let scalar = attr.dataspace.space_type == DataspaceType::Scalar;

    // Enumeration values decode through their integer base type.
    match crate::data_read::effective_numeric(&attr.datatype) {
        Datatype::FloatingPoint(numeric) if numeric.size().get() == 4 => {
            let vals = crate::data_read::read_as_f64(&attr.raw_data, &attr.datatype).ok()?;
            if scalar {
                Some(AttrValue::F32(narrow_f32(*vals.first()?)))
            } else {
                Some(AttrValue::F32Array(
                    vals.into_iter().map(narrow_f32).collect(),
                ))
            }
        }
        Datatype::FloatingPoint(..) => {
            let vals = crate::data_read::read_as_f64(&attr.raw_data, &attr.datatype).ok()?;
            if scalar {
                Some(AttrValue::F64(*vals.first()?))
            } else {
                Some(AttrValue::F64Array(vals))
            }
        }
        Datatype::FixedPoint(numeric) if numeric.layout().signed() => signed_attr_value(
            crate::data_read::read_as_i64(&attr.raw_data, &attr.datatype).ok()?,
            scalar,
            numeric.size().get(),
        ),
        Datatype::FixedPoint(numeric) => unsigned_attr_value(
            crate::data_read::read_as_u64(&attr.raw_data, &attr.datatype).ok()?,
            scalar,
            numeric.size().get(),
        ),
        Datatype::String { charset, size, .. } => {
            let strings = crate::data_read::read_as_strings(&attr.raw_data, &attr.datatype).ok()?;
            // Preserve empty scalar strings and stored widths larger than the content.
            Some(if scalar {
                crate::type_builders::decoded_fixed_string(one_or_empty(strings), *size, charset)
            } else {
                crate::type_builders::decoded_fixed_string_array(strings, *size, charset)
            })
        }
        Datatype::VariableLength(kind)
            if crate::vl_data::is_vlen_string_datatype(&attr.datatype) =>
        {
            let strings = crate::vl_data::read_vl_strings_from_source(
                source,
                &attr.raw_data,
                attr.dataspace.num_elements(),
                widths,
                addresses,
                crate::vl_data::VlenStringReadOptions::default(),
            )
            .ok()?;
            // Standard variable-length strings retain their representation in scalars and arrays.
            // The ASCII character-sequence representation has only an array variant.
            match (vlen_string_shape(kind), scalar) {
                (VlenStringShape::AsciiCharSequence, true) => {
                    Some(AttrValue::AsciiString(one_or_empty(strings)))
                }
                (VlenStringShape::AsciiCharSequence, false) => {
                    Some(AttrValue::VarLenAsciiCharArray(strings))
                }
                (VlenStringShape::Ascii, true) => {
                    Some(AttrValue::VarLenAsciiString(one_or_empty(strings)))
                }
                (VlenStringShape::Ascii, false) => Some(AttrValue::VarLenAsciiStringArray(strings)),
                (VlenStringShape::Utf8, true) => {
                    Some(AttrValue::VarLenString(one_or_empty(strings)))
                }
                (VlenStringShape::Utf8, false) => Some(AttrValue::VarLenStringArray(strings)),
            }
        }
        _ => None,
    }
}

/// Converts an exactly widened binary32 value back to [`f32`].
///
/// A binary32 value widens exactly to [`f64`], so the conversion recovers its value.
#[expect(
    clippy::cast_possible_truncation,
    reason = "narrows a value the reader widened from the 4 bytes the file holds, which is exact for the IEEE layout"
)]
fn narrow_f32(value: f64) -> f32 {
    value as f32
}

/// Selects a signed attribute variant for `width`, with a 64-bit fallback.
///
/// Widths of 1, 2, and 4 bytes use their variants if the first scalar element
/// or every array element fits. Other widths and failed conversions use the
/// 64-bit variant. An empty scalar returns `None`.
fn signed_attr_value(values: Vec<i64>, scalar: bool, width: u32) -> Option<AttrValue> {
    let narrowed = match width {
        1 => narrow_elements(&values, scalar, AttrValue::I8, AttrValue::I8Array),
        2 => narrow_elements(&values, scalar, AttrValue::I16, AttrValue::I16Array),
        4 => narrow_elements(&values, scalar, AttrValue::I32, AttrValue::I32Array),
        _ => None,
    };
    match narrowed {
        Some(value) => Some(value),
        None if scalar => Some(AttrValue::I64(*values.first()?)),
        None => Some(AttrValue::I64Array(values)),
    }
}

/// Selects an unsigned attribute variant using the rules of [`signed_attr_value`].
fn unsigned_attr_value(values: Vec<u64>, scalar: bool, width: u32) -> Option<AttrValue> {
    let narrowed = match width {
        1 => narrow_elements(&values, scalar, AttrValue::U8, AttrValue::U8Array),
        2 => narrow_elements(&values, scalar, AttrValue::U16, AttrValue::U16Array),
        4 => narrow_elements(&values, scalar, AttrValue::U32, AttrValue::U32Array),
        _ => None,
    };
    match narrowed {
        Some(value) => Some(value),
        None if scalar => Some(AttrValue::U64(*values.first()?)),
        None => Some(AttrValue::U64Array(values)),
    }
}

/// Converts the first element for a scalar or every element for an array.
///
/// Returns `None` if a scalar is empty or conversion of a selected element fails.
/// The scalar constructor receives the converted first element, and the array
/// constructor receives all converted elements.
fn narrow_elements<S: Copy, T: TryFrom<S>>(
    values: &[S],
    scalar: bool,
    one: fn(T) -> AttrValue,
    many: fn(Vec<T>) -> AttrValue,
) -> Option<AttrValue> {
    if scalar {
        return Some(one(T::try_from(*values.first()?).ok()?));
    }
    let narrowed: Vec<T> = values
        .iter()
        .map(|&v| T::try_from(v).ok())
        .collect::<Option<Vec<T>>>()?;
    Some(many(narrowed))
}

/// The [`AttrValue`] representation for a variable-length string datatype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VlenStringShape {
    /// A sequence of one-byte ASCII strings, written by [`AttrValue::VarLenAsciiCharArray`].
    AsciiCharSequence,
    /// A true variable-length ASCII string (`H5T_STRING`, `STRSIZE = VAR`),
    /// written by [`AttrValue::VarLenAsciiString`] and its array form.
    Ascii,
    /// A true variable-length UTF-8 string,
    /// written by [`AttrValue::VarLenString`] and its array form.
    Utf8,
}

/// Returns the attribute string representation for a variable-length kind.
///
/// Strings are classified by their character-set metadata. Sequences use the ASCII
/// character-sequence representation.
fn vlen_string_shape(kind: &VariableLengthType) -> VlenStringShape {
    match kind {
        VariableLengthType::Sequence { .. } => VlenStringShape::AsciiCharSequence,
        VariableLengthType::String(string) => match string.charset() {
            CharacterSet::Ascii => VlenStringShape::Ascii,
            CharacterSet::Utf8 => VlenStringShape::Utf8,
        },
    }
}

/// Returns the first decoded string, or an empty string if there are no elements.
///
/// A zero-size fixed-string datatype decodes to an empty vector. The resulting
/// empty string preserves that attribute in [`attrs_to_map`].
fn one_or_empty(strings: Vec<std::string::String>) -> std::string::String {
    strings.into_iter().next().unwrap_or_default()
}

#[cfg(test)]
mod tests;
