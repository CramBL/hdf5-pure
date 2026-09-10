//! Simplified type representations for the high-level API.

use std::collections::HashMap;
use std::fmt;

use crate::address::BaseAddress;
use crate::datatype::Datatype;
use crate::display::{DISPLAY_MAX_MEMBERS, Dims, EscapedName, write_elided};

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
    /// A type this curated view has no name for, carrying the type itself.
    ///
    /// Reached through a compound field or an array base as well as at the top
    /// level, where [`Dataset::datatype`](crate::Dataset::datatype) is not an
    /// escape hatch, so what lands here has to be enough to work with on its
    /// own.
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
                write_elided(f, fields.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
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
                write_elided(f, names.len().saturating_sub(DISPLAY_MAX_MEMBERS))?;
                write!(f, "]")
            }
            DType::Array(base, dims) => write!(f, "array<{base}, {}>", Dims(dims)),
            DType::Other(dt) => write!(f, "other({dt})"),
        }
    }
}

/// Convert a low-level `Datatype` to a simplified `DType`.
pub(crate) fn classify_datatype(dt: &Datatype) -> DType {
    match dt {
        Datatype::FloatingPoint { size: 4, .. } => DType::F32,
        Datatype::FloatingPoint { size: 8, .. } => DType::F64,
        Datatype::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => DType::I8,
        Datatype::FixedPoint {
            size: 2,
            signed: true,
            ..
        } => DType::I16,
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => DType::I32,
        Datatype::FixedPoint {
            size: 8,
            signed: true,
            ..
        } => DType::I64,
        Datatype::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => DType::U8,
        Datatype::FixedPoint {
            size: 2,
            signed: false,
            ..
        } => DType::U16,
        Datatype::FixedPoint {
            size: 4,
            signed: false,
            ..
        } => DType::U32,
        Datatype::FixedPoint {
            size: 8,
            signed: false,
            ..
        } => DType::U64,
        Datatype::String { .. } => DType::String,
        Datatype::VariableLength {
            is_string: true, ..
        } => DType::VariableLengthString,
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

/// Read attribute messages into a `HashMap<String, AttrValue>`.
///
/// Best-effort: attributes that can't be decoded are silently skipped.
/// `base_address` is the file-level userblock offset — needed so that
/// variable-length attribute data (stored in global heap collections with
/// addresses relative to the base) can be located correctly.
pub(crate) fn attrs_to_map<S: crate::source::Source + ?Sized>(
    attrs: &[crate::attribute::AttributeMessage],
    source: &S,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> HashMap<std::string::String, AttrValue> {
    let mut map = HashMap::new();
    for attr in attrs {
        if let Some(val) = decode_attr_value(attr, source, offset_size, length_size, base_address) {
            map.insert(attr.name.clone(), val);
        }
    }
    map
}

/// Decode one attribute message into the [`AttrValue`] variant that describes
/// what the file holds.
///
/// The datatype and the dataspace together determine the variant, so a value
/// this crate wrote reads back as the variant it was written from. The
/// dataspace kind — not the element count — decides scalar against array, so a
/// one-element array stays an array. Charset selects the `Ascii*` variants.
///
/// A number keeps the width it is stored at. An integer's 1-, 2-, 4- and 8-byte
/// datatypes take the [`AttrValue`] variant of that width, signed or unsigned as
/// the datatype says (issue #350), and a 4-byte float takes
/// [`F32`](AttrValue::F32) rather than widening to `f64` (issue #354). An
/// integer width the format allows but Rust has no integer for — 3 bytes, say —
/// widens to the 64-bit variant, as does a value that does not fit the width its
/// own datatype declares, which takes a precision wider than that width to
/// reach. Both carry the value unchanged; only the variant is wider than the
/// file. A width *above* 8 bytes has no variant at all: the numeric readers
/// model an element as a 64-bit word and refuse a wider one, so the attribute is
/// reported undecodable rather than decoding to part of its value — `attrs`
/// omits it, `attr_datatypes` still reports its type (issue #361).
///
/// What is still not recoverable, because [`AttrValue`] has no way to express
/// it — each of these reads correctly but would be rewritten differently:
///
/// - **Byte order and precision.** Every integer variant writes back
///   little-endian at its full width, so a big-endian attribute, or one storing
///   fewer bits than its bytes hold, reads correctly and would be re-encoded in
///   this crate's own layout. The same holds for a float whose exponent or
///   mantissa is not laid out the way IEEE 754 lays it out.
/// - **Enumeration members.** An enum attribute decodes through its integer base
///   type, so its codes survive and the member names do not. This is how h5py's
///   `np.bool_` attributes arrive, written as `enum[FALSE, TRUE]`: as `0`/`1` of
///   the base type's width.
/// - **A scalar in MATLAB's sequence-of-one-byte-strings shape.** Its array
///   form reads as [`VarLenAsciiCharArray`](AttrValue::VarLenAsciiCharArray), which is
///   the form MATLAB writes; a scalar in that shape has no variant and reads as
///   [`AsciiString`](AttrValue::AsciiString), which writes a fixed-width slot.
///   A *standard* variable-length string keeps its datatype at either arity
///   (issue #383).
/// - **Rank.** Every array variant is one-dimensional, so a rank-2 attribute
///   reads as its elements flattened.
/// - **String padding.** A fixed-width string reports its content and the
///   `STRSIZE` it was stored at, but not whether the bytes past the content were
///   null-terminated, null-padded or space-padded: every variant writes back
///   `NULLPAD`. A slot *narrower* than its decoded text loses its width too,
///   since the decoder is lossy and the text can outgrow the slot it came from.
/// - **Null dataspaces.** These read as an empty array variant.
///
/// A numeric attribute whose message holds fewer bytes than its dataspace
/// promises decodes to the elements its bytes do hold, and a *scalar* left with
/// none of them is reported undecodable (`None`) rather than defaulted, since no
/// value would be truthful. So is any attribute whose bytes stop part-way
/// through an element. An empty *string* is different: its zero-size datatype
/// legitimately decodes to no elements, and the empty string is the value, so it
/// is kept.
fn decode_attr_value<S: crate::source::Source + ?Sized>(
    attr: &crate::attribute::AttributeMessage,
    source: &S,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Option<AttrValue> {
    use crate::dataspace::DataspaceType;
    use crate::datatype::Datatype;

    // A scalar dataspace and a 1-element simple dataspace are different on
    // disk (v1 carries rank 0, v2 a type byte), and the write side picks
    // between them per variant, so this is what makes the round trip faithful
    // at length one.
    let scalar = attr.dataspace.space_type == DataspaceType::Scalar;

    // An enumeration is stored as values of its integer base type, so it decodes
    // through that base — the same view the numeric readers take of an enum dataset.
    match crate::data_read::effective_numeric(&attr.datatype) {
        Datatype::FloatingPoint { size: 4, .. } => {
            let vals = attr.read_as_f64().ok()?;
            if scalar {
                Some(AttrValue::F32(narrow_f32(*vals.first()?)))
            } else {
                Some(AttrValue::F32Array(
                    vals.into_iter().map(narrow_f32).collect(),
                ))
            }
        }
        Datatype::FloatingPoint { .. } => {
            let vals = attr.read_as_f64().ok()?;
            if scalar {
                Some(AttrValue::F64(*vals.first()?))
            } else {
                Some(AttrValue::F64Array(vals))
            }
        }
        Datatype::FixedPoint {
            signed: true, size, ..
        } => signed_attr_value(attr.read_as_i64().ok()?, scalar, *size),
        Datatype::FixedPoint {
            signed: false,
            size,
            ..
        } => unsigned_attr_value(attr.read_as_u64().ok()?, scalar, *size),
        Datatype::String { charset, size, .. } => {
            let strings = attr.read_as_strings().ok()?;
            // A zero-size string datatype decodes to no elements at all, so a
            // scalar takes the empty string rather than reporting the whole
            // attribute undecodable — `attrs_to_map` drops what this returns
            // `None` for, and an empty string attribute must not disappear.
            //
            // `size` is the stored `STRSIZE`, which the value alone does not
            // recover: a 64-byte slot holding "ok" and a 2-byte one holding it
            // read back the same string. The variant carries the width when the
            // slot is wider than its content (issue #359), so that writing the
            // value back reproduces the slot's *width* rather than shrinking it.
            // Not its padding rule: a `SPACEPAD` or `NULLTERM` source comes back
            // `NULLPAD`, which is the one encoding these variants write.
            Some(if scalar {
                crate::type_builders::decoded_fixed_string(one_or_empty(strings), *size, charset)
            } else {
                crate::type_builders::decoded_fixed_string_array(strings, *size, charset)
            })
        }
        Datatype::VariableLength {
            is_string,
            base_type,
            charset,
            ..
        } if *is_string || is_ascii_char_vlen_base(base_type) => {
            // Two MATLAB-relevant encodings share the same on-disk byte
            // layout (length + heap ref + object index per element; heap
            // object holds raw bytes without terminator):
            //   - is_string: true             — H5T_STRING{STRSIZE=VAR}
            //   - VLEN of H5T_STRING{SIZE=1}  — what matio / MATLAB emit
            //
            // The reader resolves each element from the global heap, adding
            // `base_address` to the (relative) collection addresses.
            let strings = crate::vl_data::read_vl_strings_from_source(
                source,
                &attr.raw_data,
                attr.dataspace.num_elements(),
                offset_size,
                length_size,
                base_address,
                crate::vl_data::VlenStringReadOptions::default(),
            )
            .ok()?;
            // Each shape decodes into the variant that writes it back as it was
            // found, so reading an attribute and setting it again keeps its
            // datatype rather than rewriting a variable-length string as a
            // fixed-width one.
            //
            // The one exception is a *scalar* in MATLAB's sequence-of-chars
            // shape, which has no variant: that shape exists for `MATLAB_fields`
            // and its neighbours, which are always arrays, and giving it one
            // would mean a scalar variant no builder needs.
            match (
                vlen_string_shape(*is_string, base_type, charset.as_ref()),
                scalar,
            ) {
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

/// A 4-byte float's value, as the `f32` the file holds.
///
/// The decoders model every float as an `f64`, and an IEEE 4-byte value widens
/// into one exactly, so narrowing it back is the identity rather than a
/// rounding step — the cast can only lose something for a 4-byte float laid out
/// some other way, whose value is reconstructed first and whose 24 or fewer
/// mantissa bits an `f32` still holds.
#[expect(
    clippy::cast_possible_truncation,
    reason = "narrows a value the reader widened from the 4 bytes the file holds, which is exact for the IEEE layout"
)]
fn narrow_f32(value: f64) -> f32 {
    value as f32
}

/// The variant a signed integer attribute takes, chosen from the width its
/// datatype declares so the value keeps that width (issue #350).
///
/// Falling back to the 64-bit variant is what covers a width no Rust integer
/// has — the format allows 3 bytes — and an element outside the range the
/// declared width holds, which needs a datatype whose precision exceeds its own
/// size to reach. Neither changes the value. A width above 8 bytes never lands
/// here: the reader refuses it, so this is not reached at all (issue #361).
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

/// The variant an unsigned integer attribute takes. The signed rule, in the
/// unsigned variants: see [`signed_attr_value`].
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

/// Every element as the narrower integer `T`, in the variant `one` or `many`
/// names, or `None` if any element does not fit `T` — the whole attribute then
/// keeps the wider variant rather than one element wrapping.
///
/// A scalar with no elements is `None` too, and that answer must survive the
/// caller's fallback rather than becoming a widened zero — which is why the
/// fallback re-reads the first element through `?` rather than defaulting. The
/// scalar's value is taken from that element without building a vector, since
/// most attributes are scalars and the one element is the whole value.
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

/// Which family of `AttrValue` variants a variable-length string attribute
/// belongs to, decided from its datatype alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VlenStringShape {
    /// A VLEN *sequence* of 1-byte ASCII strings — the encoding MATLAB and matio
    /// use, and the one this crate writes for [`AttrValue::VarLenAsciiCharArray`].
    AsciiCharSequence,
    /// A true variable-length ASCII string (`H5T_STRING`, `STRSIZE = VAR`),
    /// written by [`AttrValue::VarLenAsciiString`] and its array form.
    Ascii,
    /// A true variable-length UTF-8 string, or one whose charset is unstated,
    /// written by [`AttrValue::VarLenString`] and its array form.
    Utf8,
}

/// Classify a variable-length string datatype.
///
/// `is_string` is what separates a true variable-length string from a sequence
/// of 1-byte strings, and it has to be consulted: libhdf5 writes a VL string's
/// base type as a 1-byte *integer*, so the base type alone happens to be enough
/// for the files it produces — but nothing in the format stops a writer from
/// giving a VL string a 1-byte *string* base, which is byte-identical to the
/// MATLAB sequence's base. Trusting the base type alone would then report the
/// MATLAB encoding for a value that is not in it, and rewrite it as a different
/// datatype class.
fn vlen_string_shape(
    is_string: bool,
    base_type: &crate::datatype::Datatype,
    charset: Option<&crate::datatype::CharacterSet>,
) -> VlenStringShape {
    use crate::datatype::CharacterSet;
    if !is_string && is_ascii_char_vlen_base(base_type) {
        return VlenStringShape::AsciiCharSequence;
    }
    // A variable-length string states its own charset, and the base type must
    // not be consulted for it: a writer may give one the same 1-byte string base
    // the sequence uses, and reading the charset off that base would report
    // ASCII for a datatype whose own field says UTF-8 — rewriting the value in a
    // charset it was not stored in. A sequence carries its charset on the base
    // type instead, which is why the branch above reads it there.
    if charset == Some(&CharacterSet::Ascii) {
        VlenStringShape::Ascii
    } else {
        VlenStringShape::Utf8
    }
}

/// The single string a scalar attribute holds, or the empty string when the
/// datatype decoded to no elements at all.
///
/// A zero-size string datatype yields no elements (`read_as_strings` returns an
/// empty vec), which is how an empty string attribute is stored. Reporting the
/// attribute undecodable there would drop it from `attrs()` entirely, since
/// `attrs_to_map` keeps only what decodes.
fn one_or_empty(strings: Vec<std::string::String>) -> std::string::String {
    strings.into_iter().next().unwrap_or_default()
}

/// Recognize the MATLAB-style VLEN encoding where the base type is a 1-byte
/// ASCII string (`H5T_VLEN { H5T_STRING { STRSIZE 1, ..., CSET ASCII } }`).
/// Other VLEN sequences of strings may exist but we only auto-decode this
/// specific shape as a string array.
fn is_ascii_char_vlen_base(base: &crate::datatype::Datatype) -> bool {
    use crate::datatype::{CharacterSet, Datatype};
    matches!(
        base,
        Datatype::String {
            size: 1,
            charset: CharacterSet::Ascii,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::{Datatype, DatatypeByteOrder};

    /// A width the curated set has no name for reaches the caller as the type
    /// itself, and writing it must not overflow the `size * 8` bit-width
    /// computation (issue #140) — `size` is an on-disk `u32`, so a crafted one
    /// near [`u32::MAX`] is what a file can hold.
    #[test]
    fn an_unusual_size_arrives_whole_and_writes_its_width() {
        let int = Datatype::FixedPoint {
            size: u32::MAX,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 0,
        };
        let float = Datatype::FloatingPoint {
            size: u32::MAX,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 0,
            exponent_location: 0,
            exponent_size: 0,
            mantissa_location: 0,
            mantissa_size: 0,
            exponent_bias: 0,
        };

        let bits = u64::from(u32::MAX) * 8;
        for (dt, prefix) in [(&int, 'i'), (&float, 'f')] {
            let classified = classify_datatype(dt);
            assert_eq!(classified, DType::Other(Box::new(dt.clone())));

            // Exact, not a prefix: the right width is a prefix of every wider
            // wrong one, so `starts_with` would pass a `size * 80`.
            assert_eq!(
                classified.to_string(),
                format!("other({prefix}{bits}(bits 0..0))")
            );
        }
    }

    /// Why `Other` carries the type rather than a rendering of it: a type
    /// reached by recursion has no [`Dataset::datatype`](crate::Dataset::datatype)
    /// to fall back on, so what lands here is the caller's only view of it
    /// (issue #243). Both recursion sites, since either can nest one.
    #[test]
    fn an_unclassified_type_reaches_the_caller_whole_through_either_recursion() {
        let opaque = Datatype::Opaque {
            size: 3,
            tag: b"rgb".to_vec(),
        };
        let carried = DType::Other(Box::new(opaque.clone()));

        let compound = Datatype::Compound {
            size: 3,
            members: vec![crate::datatype::CompoundMember {
                name: "pixel".into(),
                byte_offset: 0,
                datatype: opaque.clone(),
            }],
        };
        let DType::Compound(fields) = classify_datatype(&compound) else {
            panic!("a compound classifies as one");
        };
        assert_eq!(fields, vec![("pixel".to_string(), carried.clone())]);

        let array = Datatype::Array {
            base_type: Box::new(opaque),
            dimensions: vec![2, 3],
        };
        assert_eq!(
            classify_datatype(&array),
            DType::Array(Box::new(carried), vec![2, 3])
        );
    }

    /// Write one attribute per case, read every attribute back, and return the
    /// values keyed by name. Goes through the real writer and reader, in
    /// memory, so what it measures is the file rather than a constructed
    /// message.
    fn round_trip(cases: &[(&str, AttrValue)]) -> HashMap<std::string::String, AttrValue> {
        let mut builder = crate::writer::FileBuilder::new();
        for (name, value) in cases {
            builder.set_attr(name, value.clone());
        }
        // A dataset gives the global heap a reason to exist for the
        // variable-length cases, matching how these files are really built.
        builder.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = builder.finish().unwrap();
        crate::File::from_bytes(bytes)
            .unwrap()
            .root()
            .attrs()
            .unwrap()
    }

    /// The variant a string attribute was written from is the variant it reads
    /// back as. The one-element arrays are the point: charset and dataspace
    /// kind are both on disk, so neither collapses into the scalar form.
    #[test]
    fn every_string_variant_round_trips_to_itself() {
        let cases = vec![
            ("utf8_scalar", AttrValue::String("m/s".into())),
            ("utf8_one", AttrValue::StringArray(vec!["m/s".into()])),
            (
                "utf8_two",
                AttrValue::StringArray(vec!["m/s".into(), "kg".into()]),
            ),
            ("ascii_scalar", AttrValue::AsciiString("double".into())),
            (
                "ascii_one",
                AttrValue::AsciiStringArray(vec!["double".into()]),
            ),
            (
                "ascii_two",
                AttrValue::AsciiStringArray(vec!["double".into(), "int16".into()]),
            ),
            (
                "vlen_one",
                AttrValue::VarLenAsciiCharArray(vec!["x".into()]),
            ),
            (
                "vlen_three",
                AttrValue::VarLenAsciiCharArray(vec!["x".into(), "y".into(), "velocity".into()]),
            ),
            // The standard variable-length string, whose datatype the MATLAB
            // shape above does not write (#383). Charset and arity separate the
            // four the same way they separate the fixed-width variants.
            ("vlen_utf8_scalar", AttrValue::VarLenString("mètre".into())),
            (
                "vlen_utf8_one",
                AttrValue::VarLenStringArray(vec!["mètre".into()]),
            ),
            (
                "vlen_ascii_scalar",
                AttrValue::VarLenAsciiString("double".into()),
            ),
            (
                "vlen_ascii_two",
                AttrValue::VarLenAsciiStringArray(vec!["x".into(), "yy".into()]),
            ),
            // A declared width is stored and recovered too, or a slot would
            // shrink to its content the next time it was written (#359).
            (
                "ascii_sized",
                AttrValue::ascii_string_sized("ok", 64).unwrap(),
            ),
            (
                "ascii_array_sized",
                AttrValue::ascii_string_array_sized(vec!["north".into(), "s".into()], 16).unwrap(),
            ),
            ("utf8_sized", AttrValue::string_sized("mètre", 32).unwrap()),
            (
                "utf8_array_sized",
                AttrValue::string_array_sized(vec!["m/s".into()], 12).unwrap(),
            ),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(read.get(*name), Some(written), "attribute {name}");
        }
    }

    /// A width that is the one the content implies reads back as the *plain*
    /// variant, not the sized one.
    ///
    /// The two write identical bytes at that width, so reporting the sized
    /// variant for them would be a distinction with nothing behind it — and
    /// would move every string attribute this crate has ever written onto a
    /// variant its readers do not match. The sized variant is reserved for the
    /// widths the content cannot recover (#359).
    #[test]
    fn a_width_the_content_implies_reads_back_as_the_plain_variant() {
        let cases = vec![
            ("exact", AttrValue::ascii_string_sized("double", 6).unwrap()),
            ("empty", AttrValue::ascii_string_sized("", 1).unwrap()),
            (
                "longest",
                AttrValue::string_array_sized(vec!["m/s".into(), "kg".into()], 3).unwrap(),
            ),
        ];
        let read = round_trip(&cases);
        assert_eq!(
            read.get("exact"),
            Some(&AttrValue::AsciiString("double".into()))
        );
        assert_eq!(
            read.get("empty"),
            Some(&AttrValue::AsciiString(String::new()))
        );
        assert_eq!(
            read.get("longest"),
            Some(&AttrValue::StringArray(vec!["m/s".into(), "kg".into()]))
        );
    }

    /// Decode one attribute of an arbitrary datatype, which the writer cannot
    /// stage because [`AttrValue`] has no variant to write it from.
    fn decode_raw(
        datatype: Datatype,
        raw_data: Vec<u8>,
        dimensions: Vec<u64>,
    ) -> Option<AttrValue> {
        use crate::dataspace::{Dataspace, DataspaceType};

        let scalar = dimensions.is_empty();
        let attr = crate::attribute::AttributeMessage {
            name: "a".into(),
            datatype,
            dataspace: Dataspace {
                space_type: if scalar {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                },
                #[expect(clippy::cast_possible_truncation)]
                rank: dimensions.len() as u8,
                dimensions,
                max_dimensions: None,
            },
            raw_data,
            datatype_location: crate::shared_message::DatatypeLocation::Inline,
        };
        decode_attr_value(
            &attr,
            &crate::source::BytesSource::new(Vec::new()),
            8,
            8,
            BaseAddress::ZERO,
        )
    }

    /// A slot narrower than the text it decodes to keeps the plain variant.
    ///
    /// The string decoder is lossy: a byte that is not valid UTF-8 becomes
    /// `U+FFFD`, which is *three* bytes, so a two-byte Latin-1 slot — `0xB0`
    /// `0x43`, "°C" in a fixed ASCII attribute, which is ordinary in real files
    /// — decodes to four bytes of text. Declaring the stored width over that
    /// text would build a sized variant holding a value it cannot fit, the exact
    /// pair `AttrValue::ascii_string_sized` refuses; the `width` field would be a
    /// false claim about the bytes beside it, and writing the value back would
    /// declare 4 while the variant said 2.
    ///
    /// This is why `declared_width` compares with `>` and not `!=` (#359).
    #[test]
    fn a_slot_narrower_than_its_decoded_text_is_not_reported_as_padded() {
        let value = decode_raw(
            Datatype::String {
                size: 2,
                padding: crate::datatype::StringPadding::NullPad,
                charset: crate::datatype::CharacterSet::Ascii,
            },
            vec![0xB0, b'C'],
            vec![],
        )
        .expect("a fixed ASCII attribute decodes");

        assert_eq!(
            value,
            AttrValue::AsciiString("\u{FFFD}C".into()),
            "a slot its own text does not fit must not claim a width"
        );
        assert!(
            value.as_str().is_some_and(|s| s.len() > 2),
            "the case only bites because the decoded text outgrew the slot"
        );
    }

    /// An enum attribute decodes through its integer base type rather than being
    /// dropped. h5py writes every `np.bool_` attribute this way, so before this the
    /// booleans in an h5py-written file were missing from `attrs()` entirely (#248).
    #[test]
    fn an_enum_attribute_decodes_as_its_base_type() {
        let h5py_bool =
            crate::type_builders::EnumTypeBuilder::with_base(crate::type_builders::make_i8_type())
                .value("FALSE", 0)
                .value("TRUE", 1)
                .build()
                .unwrap();

        // The base is one byte wide, so the code arrives as an 8-bit value (#350).
        assert_eq!(
            decode_raw(h5py_bool.clone(), vec![1], vec![]),
            Some(AttrValue::I8(1))
        );
        // Array-ness comes from the dataspace here as it does for every other type.
        assert_eq!(
            decode_raw(h5py_bool, vec![1, 0, 1], vec![3]),
            Some(AttrValue::I8Array(vec![1, 0, 1]))
        );
    }

    /// The base type's signedness and width carry through, so an enum over an
    /// unsigned 16-bit base lands in the unsigned variant of that width with its
    /// value intact.
    #[test]
    fn an_unsigned_enum_attribute_keeps_its_base_signedness_and_width() {
        let mode =
            crate::type_builders::EnumTypeBuilder::with_base(crate::type_builders::make_u16_type())
                .value("low", 1)
                .value("high", 40_000)
                .build()
                .unwrap();

        assert_eq!(
            decode_raw(mode, 40_000_u16.to_le_bytes().to_vec(), vec![]),
            Some(AttrValue::U16(40_000))
        );
    }

    /// Array-ness survives at length one for numbers too, and a float keeps its
    /// width: a 4-byte attribute reads back as `F32`, not widened to `F64`
    /// (#354). The integers say the same thing at every width in
    /// [`every_integer_width_round_trips_to_itself`].
    ///
    /// The `f32` values are the ones a narrowing that rounded would get wrong:
    /// the extremes of the range, and a subnormal, which is where the exponent
    /// handling of a widen-then-narrow round trip breaks if it breaks at all.
    #[test]
    fn every_float_variant_round_trips_to_itself() {
        let cases = vec![
            ("f64_scalar", AttrValue::F64(1.5)),
            ("f64_one", AttrValue::F64Array(vec![1.5])),
            ("f64_two", AttrValue::F64Array(vec![1.5, 2.5])),
            ("f32_scalar", AttrValue::F32(1.5)),
            ("f32_one", AttrValue::F32Array(vec![1.5])),
            ("f32_two", AttrValue::F32Array(vec![f32::MIN, f32::MAX])),
            (
                "f32_edges",
                AttrValue::F32Array(vec![
                    f32::MIN_POSITIVE,
                    f32::from_bits(1),
                    f32::EPSILON,
                    -0.0,
                    f32::INFINITY,
                ]),
            ),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(read.get(*name), Some(written), "attribute {name}");
        }
    }

    /// An unsigned value above `i64::MAX` reads back as itself. It used to be
    /// reinterpreted into a negative `i64` for want of a `U64Array` variant, so
    /// this is the case that variant exists for — and the accessors report the
    /// value rather than a wrapped one at every length.
    #[test]
    fn a_full_range_unsigned_value_survives_at_every_length() {
        let read = round_trip(&[
            ("scalar", AttrValue::U64(u64::MAX)),
            ("one", AttrValue::U64Array(vec![u64::MAX])),
            ("two", AttrValue::U64Array(vec![u64::MAX, 1])),
        ]);
        assert_eq!(read.get("scalar"), Some(&AttrValue::U64(u64::MAX)));
        assert_eq!(read.get("one"), Some(&AttrValue::U64Array(vec![u64::MAX])));
        assert_eq!(
            read.get("two"),
            Some(&AttrValue::U64Array(vec![u64::MAX, 1]))
        );
        // Read through the accessors: the full-range value is reported as
        // unsigned, and asking for it as `i64` refuses rather than wrapping.
        for (name, expected) in [
            ("scalar", vec![u64::MAX]),
            ("one", vec![u64::MAX]),
            ("two", vec![u64::MAX, 1]),
        ] {
            let value = read.get(name).expect("present");
            assert_eq!(value.to_u64s(), Some(expected), "{name} must read unsigned");
            assert_eq!(
                value.to_i64s(),
                None,
                "{name} does not fit an i64 and must not wrap"
            );
        }
        assert_eq!(read["scalar"].as_u64(), Some(u64::MAX));
        assert_eq!(read["one"].as_u64(), Some(u64::MAX));
        assert_eq!(
            read["two"].as_u64(),
            None,
            "two elements are not a single value"
        );
    }

    /// The MATLAB sequence-of-ASCII-chars encoding and a true variable-length
    /// ASCII string have the same base type once a writer chooses a 1-byte
    /// string base for the latter; `is_string` is the only thing separating
    /// them. libhdf5 writes an integer base, so no file it produces reaches the
    /// ambiguous case — which is exactly why this is asserted here rather than
    /// left to the C crosscheck, where the branch cannot be reached.
    #[test]
    fn only_a_sequence_of_ascii_chars_claims_the_matlab_shape() {
        use crate::datatype::{CharacterSet, Datatype, DatatypeByteOrder, StringPadding};

        let char_base = Datatype::String {
            size: 1,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        };
        let int_base = Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        };

        // What MATLAB and matio write, and what this crate writes for
        // `AttrValue::VarLenAsciiCharArray`.
        assert_eq!(
            vlen_string_shape(false, &char_base, None),
            VlenStringShape::AsciiCharSequence
        );
        // The same base type, but flagged a string: a true variable-length
        // string, which must not claim the sequence variant.
        assert_eq!(
            vlen_string_shape(true, &char_base, Some(&CharacterSet::Ascii)),
            VlenStringShape::Ascii
        );
        // And its charset comes from its own field, not from that base: reading
        // it off the base reported ASCII here, which rewrote a UTF-8 attribute
        // as an ASCII one.
        assert_eq!(
            vlen_string_shape(true, &char_base, Some(&CharacterSet::Utf8)),
            VlenStringShape::Utf8
        );
        // What libhdf5 actually writes for a variable-length string.
        assert_eq!(
            vlen_string_shape(true, &int_base, Some(&CharacterSet::Ascii)),
            VlenStringShape::Ascii
        );
        assert_eq!(
            vlen_string_shape(true, &int_base, Some(&CharacterSet::Utf8)),
            VlenStringShape::Utf8
        );
        // An unstated charset reads as UTF-8, which is the lossless assumption:
        // every ASCII string is valid UTF-8, so nothing is corrupted by it.
        assert_eq!(
            vlen_string_shape(true, &int_base, None),
            VlenStringShape::Utf8
        );
    }

    /// An empty string attribute stays present and stays empty.
    ///
    /// A zero-length string is stored with a zero-size datatype, which decodes
    /// to no elements; treating that as undecodable dropped the attribute out of
    /// `attrs()` entirely, because `attrs_to_map` keeps only what decodes.
    #[test]
    fn an_empty_string_attribute_is_not_dropped() {
        let cases = vec![
            ("utf8", AttrValue::String(std::string::String::new())),
            ("ascii", AttrValue::AsciiString(std::string::String::new())),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(
                read.get(*name),
                Some(written),
                "attribute {name} must survive with its empty value"
            );
        }
    }

    /// Every integer width round-trips as itself, rather than every one of them
    /// arriving as 64-bit (#350).
    ///
    /// Both ends are under test at once: the writer stores each variant at its
    /// own width, and the reader picks the variant back out of that width. A
    /// writer that stored `I16` in four bytes would be read as `I32` and fail
    /// here, so the pair cannot drift together — and `crates/crosscheck/tests/attr_width.rs`
    /// is what says the width means the same thing outside this crate, with the
    /// reference C library reading the bytes.
    ///
    /// The extremes are the elements that matter: a value at the edge of its
    /// width is what a decode that widened through the wrong signedness, or
    /// narrowed with a wrap, gets wrong.
    #[test]
    fn every_integer_width_round_trips_to_itself() {
        let cases = vec![
            ("i8", AttrValue::I8(-7)),
            ("i8_one", AttrValue::I8Array(vec![-7])),
            ("i8_two", AttrValue::I8Array(vec![i8::MIN, i8::MAX])),
            ("i16", AttrValue::I16(-7)),
            ("i16_one", AttrValue::I16Array(vec![-7])),
            ("i16_two", AttrValue::I16Array(vec![i16::MIN, i16::MAX])),
            ("i32", AttrValue::I32(-7)),
            ("i32_one", AttrValue::I32Array(vec![-7])),
            ("i32_two", AttrValue::I32Array(vec![i32::MIN, i32::MAX])),
            ("u8", AttrValue::U8(7)),
            ("u8_one", AttrValue::U8Array(vec![7])),
            ("u8_two", AttrValue::U8Array(vec![0, u8::MAX])),
            ("u16", AttrValue::U16(7)),
            ("u16_one", AttrValue::U16Array(vec![7])),
            ("u16_two", AttrValue::U16Array(vec![0, u16::MAX])),
            ("u32", AttrValue::U32(7)),
            ("u32_one", AttrValue::U32Array(vec![7])),
            ("u32_two", AttrValue::U32Array(vec![0, u32::MAX])),
            ("i64", AttrValue::I64(-7)),
            ("i64_one", AttrValue::I64Array(vec![-7])),
            ("i64_two", AttrValue::I64Array(vec![i64::MIN, i64::MAX])),
            ("u64", AttrValue::U64(7)),
            ("u64_one", AttrValue::U64Array(vec![7])),
            ("u64_two", AttrValue::U64Array(vec![0, u64::MAX])),
        ];
        let read = round_trip(&cases);
        for (name, written) in &cases {
            assert_eq!(read.get(*name), Some(written), "attribute {name}");
        }
    }

    /// A width no Rust integer has — the format allows any byte count — widens
    /// to the 64-bit variant rather than losing the attribute. Three bytes is
    /// what h5py's `np.dtype("u3")`-style types and some instrument writers
    /// produce, and the value must still arrive.
    #[test]
    fn a_width_with_no_rust_integer_widens() {
        let three_byte = Datatype::FixedPoint {
            size: 3,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 24,
        };
        assert_eq!(
            decode_raw(three_byte.clone(), vec![0xFF, 0xFF, 0xFF], vec![]),
            Some(AttrValue::U64(0x00FF_FFFF))
        );
        assert_eq!(
            decode_raw(three_byte, vec![1, 0, 0, 2, 0, 0], vec![2]),
            Some(AttrValue::U64Array(vec![1, 2]))
        );
    }

    /// A value outside the range its own width holds keeps the 64-bit variant
    /// rather than being wrapped into the narrow one.
    ///
    /// It takes a datatype whose *precision* exceeds its size to get there: one
    /// stored byte holding `0xFF`, sign-extended from bit 16 rather than bit 8,
    /// decodes to 255 — which no `i8` holds. No writer produces such a type, but
    /// a file is not this crate's to trust, and `-1` in place of `255` would be
    /// a silent lie about what the file says.
    #[test]
    fn a_value_outside_its_declared_width_is_not_wrapped() {
        let overwide = Datatype::FixedPoint {
            size: 1,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        };
        assert_eq!(
            decode_raw(overwide.clone(), vec![0xFF], vec![]),
            Some(AttrValue::I64(255)),
            "a scalar past `i8` must widen, not wrap"
        );
        assert_eq!(
            decode_raw(overwide, vec![0x01, 0xFF], vec![2]),
            Some(AttrValue::I64Array(vec![1, 255])),
            "one element past `i8` widens the whole array, since a mixed answer \
             would carry two different meanings for one attribute"
        );
    }

    /// The accessors are what a consumer should use, and they read every shape
    /// above as the same logical value — which is the reason the reader is free
    /// to be faithful about the variant.
    #[test]
    fn accessors_span_the_shapes_the_reader_now_distinguishes() {
        let read = round_trip(&[
            ("scalar", AttrValue::AsciiString("double".into())),
            ("one", AttrValue::StringArray(vec!["double".into()])),
            (
                "vlen",
                AttrValue::VarLenAsciiCharArray(vec!["double".into()]),
            ),
            ("vlen_std", AttrValue::VarLenAsciiString("double".into())),
            (
                "vlen_std_one",
                AttrValue::VarLenStringArray(vec!["double".into()]),
            ),
        ]);
        for name in ["scalar", "one", "vlen", "vlen_std", "vlen_std_one"] {
            assert_eq!(
                read.get(name).and_then(AttrValue::as_str),
                Some("double"),
                "attribute {name}"
            );
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod display_tests {
    use super::*;
    use crate::datatype::{CharacterSet, Datatype, DatatypeByteOrder, StringPadding};

    #[test]
    fn an_array_shape_is_not_a_debug_slice() {
        let dtype = DType::Array(Box::new(DType::F32), vec![2, 3]);
        assert_eq!(dtype.to_string(), "array<f32, 2x3>");
        assert_eq!(
            DType::Array(Box::new(DType::U8), vec![4]).to_string(),
            "array<u8, 4>"
        );
    }

    /// An unclassified datatype carries the type, and writes as the summary of
    /// it. The whole `Debug` record is unreadable in the message that quotes it.
    #[test]
    fn an_unclassified_type_carries_the_type_and_writes_a_summary() {
        let vax = Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::Vax,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        };
        // Classification keys off size alone, so this stays `F32`; the point is
        // the fallback below.
        assert_eq!(classify_datatype(&vax), DType::F32);

        let time = Datatype::Time {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_precision: 32,
        };
        let classified = classify_datatype(&time);
        assert_eq!(classified, DType::Other(Box::new(time.clone())));
        assert_eq!(classified.to_string(), "other(time32)");

        let opaque = Datatype::Opaque {
            size: 3,
            tag: b"rgb".to_vec(),
        };
        assert_eq!(
            classify_datatype(&opaque).to_string(),
            "other(opaque[3] \"rgb\")",
            "not `other(Opaque {{ size: 3, tag: [114, 103, 98] }})`"
        );
    }

    /// The curated view quotes the same file-recorded names as the detailed
    /// one, so it escapes them by the same rule — in both member-bearing
    /// variants, not just whichever one a test happened to reach for.
    #[test]
    fn a_curated_member_name_is_escaped_in_either_variant() {
        let compound = DType::Compound(vec![("a\nb".into(), DType::I32)]).to_string();
        assert!(!compound.chars().any(char::is_control), "{compound}");
        assert_eq!(compound, "compound{a\\nb: i32}");

        let enumeration = DType::Enum(vec!["a\u{1b}[31mb".into()]).to_string();
        assert!(!enumeration.chars().any(char::is_control), "{enumeration}");
        assert_eq!(enumeration, "enum[a\\u{1b}[31mb]");
    }

    /// Likewise for the cap: a file can declare far more members than a message
    /// can carry, in either variant.
    #[test]
    fn a_curated_member_list_is_elided_in_either_variant() {
        let over_cap = DISPLAY_MAX_MEMBERS + 2;
        let names: Vec<String> = (0..over_cap).map(|i| format!("m{i}")).collect();

        let compound = DType::Compound(
            names
                .iter()
                .map(|name| (name.clone(), DType::I32))
                .collect(),
        );
        let enumeration = DType::Enum(names);

        for (dtype, close) in [(compound, "}"), (enumeration, "]")] {
            let shown = dtype.to_string();
            assert!(shown.ends_with(&format!(", … 2 more{close}")), "{shown}");
            assert!(
                !shown.contains(&format!("m{DISPLAY_MAX_MEMBERS}")),
                "{shown}"
            );
        }
    }

    /// The two views describe the same file, so a type that classifies to a
    /// named [`DType`] is spelled the same way by both. Where they differ, the
    /// `Datatype` is the longer of the two, never a different word: it carries
    /// the on-disk detail `DType` drops.
    #[test]
    fn dtype_and_datatype_agree_on_the_names_they_share() {
        let identical = [
            Datatype::FixedPoint {
                size: 4,
                byte_order: DatatypeByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            },
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
            },
        ];
        for datatype in identical {
            assert_eq!(
                classify_datatype(&datatype).to_string(),
                datatype.to_string()
            );
        }

        // A fixed-width string is the case where they differ: `DType` names the
        // class, `Datatype` adds the width, charset and padding that decide how
        // the bytes read.
        let string = Datatype::String {
            size: 8,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        };
        assert_eq!(classify_datatype(&string).to_string(), "string");
        assert_eq!(string.to_string(), "string[8] ascii null-pad");
        assert!(
            string
                .to_string()
                .starts_with(&classify_datatype(&string).to_string()),
            "the longer spelling still opens with the shorter one"
        );
    }
}
