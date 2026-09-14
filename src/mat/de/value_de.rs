//! `serde::Deserializer` implementations for `MatValue`.
//!
//! Entry: [`MatValueDeserializer::new`] wraps a [`MatValue`] and implements
//! `Deserializer`. Helper access types (struct maps, sequences) live below.

use std::collections::VecDeque;

use serde::de::{
    DeserializeSeed, Deserializer, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};
use serde::forward_to_deserialize_any;

use crate::mat::complex::complex_tag_for_sentinel;
use crate::mat::de::mcos_reader::TABLE_META_KEY;
use crate::mat::error::MatError;
use crate::mat::matrix::{MATRIX_SENTINEL, complex_tag_for_matrix_sentinel};
use crate::mat::table::{MATCOLUMN_SENTINEL, MATTABLE_SENTINEL};
use crate::mat::value::{ComplexNum, ComplexVec, MatValue, NumVec, ScalarNum};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) struct MatValueDeserializer {
    value: MatValue,
}

impl MatValueDeserializer {
    pub(crate) fn new(value: MatValue) -> Self {
        Self { value }
    }
}

impl<'de> Deserializer<'de> for MatValueDeserializer {
    type Error = MatError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        dispatch_any(self.value, visitor)
    }

    // ----- primitives (coerce from Scalar) -----

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Scalar(ScalarNum::Bool(b)) => visitor.visit_bool(b),
            MatValue::Scalar(ScalarNum::U8(x)) => visitor.visit_bool(x != 0),
            other => mismatch("bool", other),
        }
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i8")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deserialize_i8 narrows the MAT scalar to the i8 the visitor requested; lossy by the serde contract"
        )]
        visitor.visit_i8(v as i8)
    }
    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i16")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deserialize_i16 narrows the MAT scalar to the i16 the visitor requested; lossy by the serde contract"
        )]
        visitor.visit_i16(v as i16)
    }
    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i32")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deserialize_i32 narrows the MAT scalar to the i32 the visitor requested; lossy by the serde contract"
        )]
        visitor.visit_i32(v as i32)
    }
    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_i64(self.value, "i64")?;
        visitor.visit_i64(v)
    }
    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u8")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deserialize_u8 narrows the MAT scalar to the u8 the visitor requested; lossy by the serde contract"
        )]
        visitor.visit_u8(v as u8)
    }
    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u16")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deserialize_u16 narrows the MAT scalar to the u16 the visitor requested; lossy by the serde contract"
        )]
        visitor.visit_u16(v as u16)
    }
    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u32")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deserialize_u32 narrows the MAT scalar to the u32 the visitor requested; lossy by the serde contract"
        )]
        visitor.visit_u32(v as u32)
    }
    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_u64(self.value, "u64")?;
        visitor.visit_u64(v)
    }
    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_f64(self.value, "f32")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deserialize_f32 narrows the MAT scalar to the f32 the visitor requested; lossy by the serde contract"
        )]
        visitor.visit_f32(v as f32)
    }
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let v = to_f64(self.value, "f64")?;
        visitor.visit_f64(v)
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => {
                let mut iter = s.chars();
                let c = iter.next().ok_or_else(|| {
                    MatError::Custom("expected single char, got empty string".into())
                })?;
                if iter.next().is_some() {
                    return Err(MatError::Custom(
                        "expected single char, got multi-char string".into(),
                    ));
                }
                visitor.visit_char(c)
            }
            other => mismatch("char", other),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        self.deserialize_string(visitor)
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => visitor.visit_string(s),
            other => mismatch("string", other),
        }
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        self.deserialize_byte_buf(visitor)
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Vec1D(NumVec::U8(v)) => visitor.visit_byte_buf(v),
            other => mismatch("bytes", other),
        }
    }

    // ----- option / unit -----

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            // `EmptyStructArray` is the on-disk encoding of a missing value
            // (`struct([])`), which the writer emits. Reading it back as `None`
            // keeps `Vec<Option<T>>` round-trips sound.
            MatValue::EmptyStructArray => visitor.visit_none(),
            other => visitor.visit_some(MatValueDeserializer::new(other)),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            // Mirroring `deserialize_option` above, a unit that lowered to
            // `struct([])` in a sequence round-trips back to `()`.
            MatValue::EmptyStructArray => visitor.visit_unit(),
            other => mismatch("unit", other),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        visitor.visit_newtype_struct(self)
    }

    // ----- collections -----

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Vec1D(v) => visitor.visit_seq(Vec1DSeq::new(v)),
            // A degenerate 2-D matrix (1×N or N×1) flattens to a sequence so
            // that `Vec<T>` deserializes from either orientation.
            MatValue::Matrix { rows, cols, vec } if rows == 1 || cols == 1 => {
                visitor.visit_seq(Vec1DSeq::new(vec))
            }
            MatValue::Matrix { rows, cols, vec } => {
                visitor.visit_seq(MatrixRowsSeq::new(rows, cols, vec))
            }
            MatValue::ComplexVec1D(pairs) => visitor.visit_seq(ComplexPairsSeq::new(pairs)),
            // A degenerate 2-D complex matrix (1×N or N×1) flattens to a
            // sequence so `Vec<Complex*>` deserializes from either
            // orientation, matching the numeric `Matrix` flatten above.
            MatValue::ComplexMatrix { rows, cols, pairs } if rows == 1 || cols == 1 => {
                visitor.visit_seq(ComplexPairsSeq::new(pairs))
            }
            MatValue::ComplexMatrix { rows, cols, pairs } => {
                visitor.visit_seq(ComplexMatrixRowsSeq::new(rows, cols, pairs))
            }
            MatValue::Cell(elements) => visitor.visit_seq(CellSeq::new(elements)),
            // A struct array deserializes like `Matrix`: a `1×N`/`N×1` array
            // flattens to a sequence of structs (`Vec<T>`); a true `M×N` array
            // yields a sequence of rows (`Vec<Vec<T>>`).
            MatValue::StructArray {
                rows,
                cols,
                elements,
            } => visitor.visit_seq(struct_array_seq(rows, cols, elements)),
            // `deserialize_seq` is called when the target IS a sequence (e.g.
            // `Vec<T>`): yield zero elements rather than failing, so a sender
            // that emitted `struct([])` for an empty optional sequence still
            // deserializes cleanly. `dispatch_any` uses `visit_none` for the
            // same value because the target there is unknown and a missing
            // optional is the more common interpretation.
            MatValue::EmptyStructArray => visitor.visit_seq(CellSeq::new(Vec::new())),
            // A scalar can deserialize as a length-1 Vec.
            MatValue::Scalar(s) => {
                let v = NumVec::from_single(s);
                visitor.visit_seq(Vec1DSeq::new(v))
            }
            // …and so can a complex scalar, which is what a one-element
            // complex array reads back as.
            MatValue::ComplexScalar(n) => {
                visitor.visit_seq(ComplexPairsSeq::new(ComplexVec::from_single(n)))
            }
            other => mismatch("sequence", other),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, MatError> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::Struct(fields) => visitor.visit_map(StructMap::new(fields)),
            // An opaque object's properties form a map keyed by property name,
            // so `MatDatetime`/`MatCategorical`/… and any struct with matching
            // field names deserialize from it directly. A decoded table/timetable
            // carries a reserved metadata entry that ordinary targets must not
            // see (only `MatTable`/`MatTimetable` read it, via their sentinel).
            MatValue::Opaque { class_name, fields } => {
                visitor.visit_map(StructMap::new(strip_table_meta(&class_name, fields)))
            }
            other => mismatch("map/struct", other),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, MatError> {
        // A complex sentinel names one component class, and only a value of
        // that class satisfies it. Reading an `int16` complex dataset into a
        // `Complex64` (or the reverse) is refused rather than converted: the
        // component width is what the file says it holds, so silently changing
        // it would make the Rust type a poor witness of the file.
        if let Some(tag) = complex_tag_for_sentinel(name) {
            let got = self.value.kind();
            let wanted = || {
                MatError::Custom(format!(
                    "expected a complex {} value, got {got}",
                    tag.class().as_str()
                ))
            };
            return match self.value {
                MatValue::ComplexScalar(n) if n.tag() == tag => {
                    visitor.visit_map(ComplexStructMap::new(n))
                }
                // A real value of the same class reads as complex with a zero
                // imaginary part.
                MatValue::Scalar(s) => match ComplexNum::from_real(tag, s) {
                    Some(n) => visitor.visit_map(ComplexStructMap::new(n)),
                    None => Err(wanted()),
                },
                _ => Err(wanted()),
            };
        }
        match name {
            // Every Matrix sentinel (numeric and the per-class complex ones)
            // consumes the same set of `MatValue` shapes; the sentinel name
            // only matters at serialize time, where it preserves the element
            // class for an empty Vec.
            _ if name == MATRIX_SENTINEL || complex_tag_for_matrix_sentinel(name).is_some() => {
                match self.value {
                    MatValue::Matrix { rows, cols, vec } => {
                        visitor.visit_map(MatrixStructMap::numeric(rows, cols, vec))
                    }
                    // Allow deserializing a scalar or 1-D vec as a Matrix
                    // (treating it as rows=1 or single-element).
                    MatValue::Vec1D(v) => {
                        let len = v.len();
                        visitor.visit_map(MatrixStructMap::numeric(1, len, v))
                    }
                    MatValue::Scalar(s) => {
                        let v = NumVec::from_single(s);
                        visitor.visit_map(MatrixStructMap::numeric(1, 1, v))
                    }
                    MatValue::ComplexMatrix { rows, cols, pairs } => {
                        visitor.visit_map(MatrixStructMap::complex(rows, cols, pairs))
                    }
                    MatValue::ComplexVec1D(pairs) => {
                        let len = pairs.len();
                        visitor.visit_map(MatrixStructMap::complex(1, len, pairs))
                    }
                    MatValue::ComplexScalar(n) => visitor.visit_map(MatrixStructMap::complex(
                        1,
                        1,
                        ComplexVec::from_single(n),
                    )),
                    other => mismatch("Matrix", other),
                }
            }
            // A `MatColumn` asks for the underlying table-column value tagged by
            // kind, so the right variant can be built with full type fidelity.
            MATCOLUMN_SENTINEL => {
                let (kind, payload) = classify_column(self.value);
                let fields = vec![
                    ("kind".to_owned(), MatValue::String(kind.to_owned())),
                    ("value".to_owned(), payload),
                ];
                visitor.visit_map(StructMap::new(fields))
            }
            // `MatTable`/`MatTimetable` request the table's full field map,
            // including the reserved metadata entry that other targets never see.
            MATTABLE_SENTINEL => match self.value {
                MatValue::Opaque { fields, .. } | MatValue::Struct(fields) => {
                    visitor.visit_map(StructMap::new(fields))
                }
                other => mismatch("table", other),
            },
            _ => {
                // Plain struct: deserialize from a MATLAB struct group.
                let _ = fields;
                self.deserialize_map(visitor)
            }
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => visitor.visit_enum(UnitVariantAccess::Name(s)),
            // The variant's index, as written under
            // `UnitVariantEncoding::Index`. Serde's derived `Deserialize`
            // resolves an index to a variant itself.
            MatValue::Scalar(s) => match variant_index(&s) {
                Some(i) => visitor.visit_enum(UnitVariantAccess::Index(i)),
                None => Err(MatError::UnsupportedType(
                    "enum variant from a non-index scalar",
                )),
            },
            other => Err(MatError::UnsupportedType(match other.kind() {
                "struct" => "struct enum variant (not supported in v1)",
                _ => "non-unit enum variant",
            })),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        match self.value {
            MatValue::String(s) => visitor.visit_string(s),
            MatValue::Scalar(s) => match variant_index(&s) {
                Some(i) => visitor.visit_u64(i),
                None => mismatch("identifier", MatValue::Scalar(s)),
            },
            other => mismatch("identifier", other),
        }
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        let _ = self.value;
        visitor.visit_unit()
    }
}

// ---------------------------------------------------------------------------
// Dispatch for deserialize_any
// ---------------------------------------------------------------------------

fn dispatch_any<'de, V: Visitor<'de>>(value: MatValue, visitor: V) -> Result<V::Value, MatError> {
    match value {
        MatValue::Scalar(s) => match s {
            ScalarNum::Bool(b) => visitor.visit_bool(b),
            ScalarNum::F64(x) => visitor.visit_f64(x),
            ScalarNum::F32(x) => visitor.visit_f32(x),
            ScalarNum::I64(x) => visitor.visit_i64(x),
            ScalarNum::I32(x) => visitor.visit_i32(x),
            ScalarNum::I16(x) => visitor.visit_i16(x),
            ScalarNum::I8(x) => visitor.visit_i8(x),
            ScalarNum::U64(x) => visitor.visit_u64(x),
            ScalarNum::U32(x) => visitor.visit_u32(x),
            ScalarNum::U16(x) => visitor.visit_u16(x),
            ScalarNum::U8(x) => visitor.visit_u8(x),
        },
        MatValue::String(s) => visitor.visit_string(s),
        MatValue::Vec1D(v) => visitor.visit_seq(Vec1DSeq::new(v)),
        // Match `deserialize_seq`: a degenerate 2-D matrix (1×N or N×1)
        // flattens so callers going through `deserialize_any` (untagged
        // enums, custom Visitor impls, serde-derive's `Content` round-trip)
        // see the same orientation-agnostic sequence as `Vec<T>` callers.
        MatValue::Matrix { rows, cols, vec } if rows == 1 || cols == 1 => {
            visitor.visit_seq(Vec1DSeq::new(vec))
        }
        MatValue::Matrix { rows, cols, vec } => {
            visitor.visit_seq(MatrixRowsSeq::new(rows, cols, vec))
        }
        MatValue::ComplexScalar(n) => visitor.visit_map(ComplexStructMap::new(n)),
        MatValue::ComplexVec1D(pairs) => visitor.visit_seq(ComplexPairsSeq::new(pairs)),
        MatValue::ComplexMatrix { rows, cols, pairs } if rows == 1 || cols == 1 => {
            visitor.visit_seq(ComplexPairsSeq::new(pairs))
        }
        MatValue::ComplexMatrix { rows, cols, pairs } => {
            visitor.visit_seq(ComplexMatrixRowsSeq::new(rows, cols, pairs))
        }
        MatValue::Struct(fields) => visitor.visit_map(StructMap::new(fields)),
        MatValue::Cell(elements) => visitor.visit_seq(CellSeq::new(elements)),
        // Mirror `deserialize_seq`: a struct array presents as a sequence so
        // callers going through `deserialize_any` (untagged enums, custom
        // `Visitor`s, serde-derive's `Content` round-trip) see the same shape as
        // `Vec<T>` callers.
        MatValue::StructArray {
            rows,
            cols,
            elements,
        } => visitor.visit_seq(struct_array_seq(rows, cols, elements)),
        MatValue::EmptyStructArray => visitor.visit_none(),
        // A decoded opaque object presents as a map over its (decoded or raw)
        // properties, identical to a struct — so `datetime`/`categorical`/… and
        // unknown opaque classes alike deserialize into a matching Rust struct.
        // The reserved table/timetable metadata entry is hidden from this
        // general path (only `MatTable`/`MatTimetable` read it via their sentinel).
        MatValue::Opaque { class_name, fields } => {
            visitor.visit_map(StructMap::new(strip_table_meta(&class_name, fields)))
        }
    }
}

/// Drop the reserved table/timetable metadata entry (see `TABLE_META_KEY`) so it
/// never surfaces as a field to an ordinary struct/map deserialization target.
/// Mirrors how the reader hides MATLAB's reserved `#refs#` / `#subsystem#`
/// groups.
///
/// Gated on the opaque `class_name`: only `table` / `timetable` carry this
/// reserved entry (the decoder adds it). Other opaque classes are left untouched
/// because their field names can be user-controlled — a `containers.Map` re-keyed
/// by its string keys could legitimately have a key equal to `TABLE_META_KEY`,
/// which must be preserved rather than silently dropped.
fn strip_table_meta(class_name: &str, fields: Vec<(String, MatValue)>) -> Vec<(String, MatValue)> {
    if class_name != "table" && class_name != "timetable" {
        return fields;
    }
    let mut fields = fields;
    fields.retain(|(name, _)| name != TABLE_META_KEY);
    fields
}

// ---------------------------------------------------------------------------
// Helper coercions
// ---------------------------------------------------------------------------

fn mismatch<T>(expected: &'static str, got: MatValue) -> Result<T, MatError> {
    Err(MatError::Custom(format!(
        "expected {expected}, got {}",
        got.kind()
    )))
}

/// Classify a table-column value into a [`MatColumn`](crate::mat::MatColumn)
/// kind tag plus the payload to deserialize for that variant. Numeric widths
/// collapse to a single `numeric` kind (read as `f64`); `datetime` / `duration`
/// / `categorical` are recognized by the opaque class name; everything this
/// enum does not model (`cell`, `struct`, user objects) is `other`.
fn classify_column(value: MatValue) -> (&'static str, MatValue) {
    let kind = match &value {
        MatValue::Opaque { class_name, .. } => match class_name.as_str() {
            "datetime" => "datetime",
            "duration" => "duration",
            "categorical" => "categorical",
            _ => "other",
        },
        MatValue::Scalar(ScalarNum::Bool(_)) | MatValue::Vec1D(NumVec::Bool(_)) => "logical",
        MatValue::Matrix {
            vec: NumVec::Bool(_),
            ..
        } => "logical",
        MatValue::Scalar(_) | MatValue::Vec1D(_) | MatValue::Matrix { .. } => "numeric",
        MatValue::String(_) => "text",
        MatValue::Cell(elems) if elems.iter().all(|e| matches!(e, MatValue::String(_))) => "text",
        _ => "other",
    };
    // A single-string column must present as a one-element list so the `text`
    // variant's `Vec<String>` deserializes.
    let payload = match (kind, value) {
        ("text", MatValue::String(s)) => MatValue::Cell(vec![MatValue::String(s)]),
        (_, other) => other,
    };
    (kind, payload)
}

fn to_i64(v: MatValue, expected: &'static str) -> Result<i64, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::I8(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::I16(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::I32(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::I64(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as i64),
        #[expect(
            clippy::cast_possible_wrap,
            reason = "the `x <= i64::MAX` guard keeps the U64 in i64 range, so `x as i64` cannot wrap"
        )]
        MatValue::Scalar(ScalarNum::U64(x)) if x <= i64::MAX as u64 => Ok(x as i64),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coercing a MAT F64 scalar to the requested i64; float->int truncation is the intended numeric conversion"
        )]
        MatValue::Scalar(ScalarNum::F64(x)) => Ok(x as i64),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coercing a MAT F32 scalar to the requested i64; float->int truncation is the intended numeric conversion"
        )]
        MatValue::Scalar(ScalarNum::F32(x)) => Ok(x as i64),
        MatValue::Scalar(ScalarNum::Bool(b)) => Ok(i64::from(b)),
        other => mismatch(expected, other),
    }
}

fn to_u64(v: MatValue, expected: &'static str) -> Result<u64, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as u64),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as u64),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as u64),
        MatValue::Scalar(ScalarNum::U64(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::I8(x)) if x >= 0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::I16(x)) if x >= 0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::I32(x)) if x >= 0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::I64(x)) if x >= 0 => Ok(x as u64),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the `x >= 0.0` guard keeps the F64 non-negative; float->int truncation to u64 is the intended numeric conversion"
        )]
        MatValue::Scalar(ScalarNum::F64(x)) if x >= 0.0 => Ok(x as u64),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the `x >= 0.0` guard keeps the F32 non-negative; float->int truncation to u64 is the intended numeric conversion"
        )]
        MatValue::Scalar(ScalarNum::F32(x)) if x >= 0.0 => Ok(x as u64),
        MatValue::Scalar(ScalarNum::Bool(b)) => Ok(u64::from(b)),
        other => mismatch(expected, other),
    }
}

fn to_f64(v: MatValue, expected: &'static str) -> Result<f64, MatError> {
    match v {
        MatValue::Scalar(ScalarNum::F64(x)) => Ok(x),
        MatValue::Scalar(ScalarNum::F32(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I64(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I32(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I16(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::I8(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U64(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as f64),
        MatValue::Scalar(ScalarNum::Bool(b)) => Ok(if b { 1.0 } else { 0.0 }),
        other => mismatch(expected, other),
    }
}

// ---------------------------------------------------------------------------
// Vec1D SeqAccess
// ---------------------------------------------------------------------------

/// `SeqAccess` over a numeric vector. The source `NumVec` is already a
/// contiguous `Vec<Copy>`, so we walk it with a cursor and build exactly one
/// `MatValue::Scalar` per element on demand rather than pre-expanding the whole
/// vector into a `VecDeque<MatValue>` of widened, boxed scalars.
struct Vec1DSeq {
    vec: NumVec,
    cursor: usize,
}

impl Vec1DSeq {
    fn new(v: NumVec) -> Self {
        Self { vec: v, cursor: 0 }
    }
}

impl<'de> SeqAccess<'de> for Vec1DSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        match self.vec.get(self.cursor) {
            Some(scalar) => {
                self.cursor += 1;
                seed.deserialize(MatValueDeserializer::new(MatValue::Scalar(scalar)))
                    .map(Some)
            }
            None => Ok(None),
        }
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.vec.len() - self.cursor)
    }
}

/// Build a [`SeqAccess`] over a struct array, mirroring how a numeric
/// [`MatValue::Matrix`] flattens. `elements` is row-major.
///
/// - A `1×N` / `N×1` array (one dimension is 1) yields each element as a
///   [`MatValue::Struct`], so the array deserializes as a flat `Vec<T>`.
/// - A true `M×N` array yields each row as its own `1×cols` struct array, so it
///   deserializes as `Vec<Vec<T>>` — the same row-of-rows shape `Matrix` uses.
fn struct_array_seq(rows: usize, cols: usize, elements: Vec<Vec<(String, MatValue)>>) -> CellSeq {
    if rows <= 1 || cols <= 1 {
        return CellSeq::new(elements.into_iter().map(MatValue::Struct).collect());
    }
    let mut rows_vec: Vec<MatValue> = Vec::with_capacity(rows);
    let mut it = elements.into_iter();
    for _ in 0..rows {
        let row: Vec<Vec<(String, MatValue)>> = it.by_ref().take(cols).collect();
        rows_vec.push(MatValue::StructArray {
            rows: 1,
            cols,
            elements: row,
        });
    }
    CellSeq::new(rows_vec)
}

// ---------------------------------------------------------------------------
// Cell SeqAccess: yields each element verbatim
// ---------------------------------------------------------------------------

struct CellSeq {
    items: VecDeque<MatValue>,
}

impl CellSeq {
    fn new(elements: Vec<MatValue>) -> Self {
        Self {
            items: elements.into(),
        }
    }
}

impl<'de> SeqAccess<'de> for CellSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        match self.items.pop_front() {
            Some(v) => seed.deserialize(MatValueDeserializer::new(v)).map(Some),
            None => Ok(None),
        }
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.items.len())
    }
}

// ---------------------------------------------------------------------------
// MatrixRows SeqAccess: yields each row as a Vec1D
// ---------------------------------------------------------------------------

struct MatrixRowsSeq {
    rows_remaining: usize,
    cols: usize,
    row_major_iter: NumVecIter,
}

impl MatrixRowsSeq {
    fn new(rows: usize, cols: usize, vec: NumVec) -> Self {
        Self {
            rows_remaining: rows,
            cols,
            row_major_iter: NumVecIter::new(vec),
        }
    }
}

impl<'de> SeqAccess<'de> for MatrixRowsSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        if self.rows_remaining == 0 {
            return Ok(None);
        }
        self.rows_remaining -= 1;
        let row = self.row_major_iter.take_n(self.cols);
        seed.deserialize(MatValueDeserializer::new(MatValue::Vec1D(row)))
            .map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.rows_remaining)
    }
}

/// Iterator-by-vec-tag: pops the front `n` elements each call.
struct NumVecIter {
    inner: NumVec,
}
impl NumVecIter {
    fn new(v: NumVec) -> Self {
        Self { inner: v }
    }

    fn take_n(&mut self, n: usize) -> NumVec {
        macro_rules! split {
            ($variant:ident, $vec:expr) => {{
                let tail = $vec.split_off(n);
                let head = std::mem::replace($vec, tail);
                NumVec::$variant(head)
            }};
        }
        match &mut self.inner {
            NumVec::Bool(v) => split!(Bool, v),
            NumVec::F64(v) => split!(F64, v),
            NumVec::F32(v) => split!(F32, v),
            NumVec::I64(v) => split!(I64, v),
            NumVec::I32(v) => split!(I32, v),
            NumVec::I16(v) => split!(I16, v),
            NumVec::I8(v) => split!(I8, v),
            NumVec::U64(v) => split!(U64, v),
            NumVec::U32(v) => split!(U32, v),
            NumVec::U16(v) => split!(U16, v),
            NumVec::U8(v) => split!(U8, v),
        }
    }
}

// ---------------------------------------------------------------------------
// ComplexPairs SeqAccess
// ---------------------------------------------------------------------------

/// Yields each pair of a complex vector as a complex scalar, so `Vec<T>`
/// deserializes element by element without materializing per-element values.
struct ComplexPairsSeq {
    pairs: ComplexVec,
    next: usize,
}

impl ComplexPairsSeq {
    fn new(pairs: ComplexVec) -> Self {
        Self { pairs, next: 0 }
    }
}

impl<'de> SeqAccess<'de> for ComplexPairsSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        match self.pairs.get(self.next) {
            Some(n) => {
                self.next += 1;
                seed.deserialize(MatValueDeserializer::new(MatValue::ComplexScalar(n)))
                    .map(Some)
            }
            None => Ok(None),
        }
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.pairs.len() - self.next)
    }
}

// ---------------------------------------------------------------------------
// ComplexMatrixRows SeqAccess: yields each row as a complex vector
// ---------------------------------------------------------------------------

struct ComplexMatrixRowsSeq {
    rows_remaining: usize,
    cols: usize,
    /// Row-major, consumed front-to-back one row per element.
    row_major: ComplexVec,
}

impl ComplexMatrixRowsSeq {
    fn new(rows: usize, cols: usize, row_major: ComplexVec) -> Self {
        Self {
            rows_remaining: rows,
            cols,
            row_major,
        }
    }
}

impl<'de> SeqAccess<'de> for ComplexMatrixRowsSeq {
    type Error = MatError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, MatError> {
        if self.rows_remaining == 0 {
            return Ok(None);
        }
        self.rows_remaining -= 1;
        let head = self.row_major.split_off_front(self.cols);
        seed.deserialize(MatValueDeserializer::new(MatValue::ComplexVec1D(head)))
            .map(Some)
    }
}

// ---------------------------------------------------------------------------
// Struct MapAccess
// ---------------------------------------------------------------------------

struct StructMap {
    fields: VecDeque<(String, MatValue)>,
    pending_value: Option<MatValue>,
}

impl StructMap {
    fn new(fields: Vec<(String, MatValue)>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
            pending_value: None,
        }
    }
}

impl<'de> MapAccess<'de> for StructMap {
    type Error = MatError;
    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, MatError> {
        match self.fields.pop_front() {
            Some((k, v)) => {
                self.pending_value = Some(v);
                seed.deserialize(StringRefDe(k)).map(Some)
            }
            None => Ok(None),
        }
    }
    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, MatError> {
        let v = self
            .pending_value
            .take()
            .ok_or_else(|| MatError::Custom("next_value before next_key".into()))?;
        seed.deserialize(MatValueDeserializer::new(v))
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.fields.len())
    }
}

/// Small helper Deserializer that yields a String as identifier/str.
struct StringRefDe(String);

impl<'de> Deserializer<'de> for StringRefDe {
    type Error = MatError;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        visitor.visit_string(self.0)
    }
    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

// ---------------------------------------------------------------------------
// Matrix sentinel MapAccess: yields {rows, cols, data}
// ---------------------------------------------------------------------------

struct MatrixStructMap {
    state: MatrixState,
    rows: usize,
    cols: usize,
    data: Option<MatrixData>,
}

enum MatrixData {
    Numeric(NumVec),
    Complex(ComplexVec),
}

enum MatrixState {
    NeedRowsKey,
    NeedRowsValue,
    NeedColsKey,
    NeedColsValue,
    NeedDataKey,
    NeedDataValue,
    Done,
}

impl MatrixStructMap {
    fn numeric(rows: usize, cols: usize, vec: NumVec) -> Self {
        Self {
            state: MatrixState::NeedRowsKey,
            rows,
            cols,
            data: Some(MatrixData::Numeric(vec)),
        }
    }

    fn complex(rows: usize, cols: usize, pairs: ComplexVec) -> Self {
        Self {
            state: MatrixState::NeedRowsKey,
            rows,
            cols,
            data: Some(MatrixData::Complex(pairs)),
        }
    }
}

impl<'de> MapAccess<'de> for MatrixStructMap {
    type Error = MatError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, MatError> {
        let (key, next_state) = match self.state {
            MatrixState::NeedRowsKey => ("rows", MatrixState::NeedRowsValue),
            MatrixState::NeedColsKey => ("cols", MatrixState::NeedColsValue),
            MatrixState::NeedDataKey => ("data", MatrixState::NeedDataValue),
            MatrixState::Done => return Ok(None),
            _ => return Err(MatError::Custom("matrix map state desync".into())),
        };
        self.state = next_state;
        seed.deserialize(StringRefDe(key.to_string())).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, MatError> {
        match self.state {
            MatrixState::NeedRowsValue => {
                self.state = MatrixState::NeedColsKey;
                seed.deserialize((self.rows as u64).into_deserializer())
            }
            MatrixState::NeedColsValue => {
                self.state = MatrixState::NeedDataKey;
                seed.deserialize((self.cols as u64).into_deserializer())
            }
            MatrixState::NeedDataValue => {
                self.state = MatrixState::Done;
                let value = match self.data.take().unwrap() {
                    MatrixData::Numeric(v) => MatValue::Vec1D(v),
                    MatrixData::Complex(pairs) => MatValue::ComplexVec1D(pairs),
                };
                seed.deserialize(MatValueDeserializer::new(value))
            }
            _ => Err(MatError::Custom("matrix map value before key".into())),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(match self.state {
            MatrixState::NeedRowsKey | MatrixState::NeedRowsValue => 3,
            MatrixState::NeedColsKey | MatrixState::NeedColsValue => 2,
            MatrixState::NeedDataKey | MatrixState::NeedDataValue => 1,
            MatrixState::Done => 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Complex sentinel MapAccess
// ---------------------------------------------------------------------------

/// Presents a complex scalar as the `{real, imag}` map the `Complex*`
/// newtypes deserialize from. One type for every component class: each
/// component goes back through the value deserializer as a tagged scalar, so
/// the newtype's own field type decides how it is read.
struct ComplexStructMap {
    state: ComplexState,
    re: ScalarNum,
    im: ScalarNum,
}

enum ComplexState {
    NeedRealKey,
    NeedRealValue,
    NeedImagKey,
    NeedImagValue,
    Done,
}

impl ComplexStructMap {
    fn new(n: ComplexNum) -> Self {
        let (re, im) = n.components();
        Self {
            state: ComplexState::NeedRealKey,
            re,
            im,
        }
    }
}

impl<'de> MapAccess<'de> for ComplexStructMap {
    type Error = MatError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, MatError> {
        let (key, next) = match self.state {
            ComplexState::NeedRealKey => ("real", ComplexState::NeedRealValue),
            ComplexState::NeedImagKey => ("imag", ComplexState::NeedImagValue),
            ComplexState::Done => return Ok(None),
            _ => return Err(MatError::Custom("complex map state desync".into())),
        };
        self.state = next;
        seed.deserialize(StringRefDe(key.to_string())).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, MatError> {
        let component = match self.state {
            ComplexState::NeedRealValue => {
                self.state = ComplexState::NeedImagKey;
                self.re.clone()
            }
            ComplexState::NeedImagValue => {
                self.state = ComplexState::Done;
                self.im.clone()
            }
            _ => return Err(MatError::Custom("complex map value before key".into())),
        };
        seed.deserialize(MatValueDeserializer::new(MatValue::Scalar(component)))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(match self.state {
            ComplexState::NeedRealKey | ComplexState::NeedRealValue => 2,
            ComplexState::NeedImagKey | ComplexState::NeedImagValue => 1,
            ComplexState::Done => 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Enum unit variant
// ---------------------------------------------------------------------------

/// A fieldless variant identified either by name or by index, since
/// [`UnitVariantEncoding`](crate::mat::UnitVariantEncoding) writes one or the
/// other. Serde's derived `Deserialize` resolves both.
enum UnitVariantAccess {
    Name(String),
    Index(u64),
}

impl<'de> EnumAccess<'de> for UnitVariantAccess {
    type Error = MatError;
    type Variant = UnitVariantOnly;
    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, UnitVariantOnly), MatError> {
        let value = match self {
            UnitVariantAccess::Name(s) => seed.deserialize(StringRefDe(s))?,
            UnitVariantAccess::Index(i) => seed.deserialize(VariantIndexDe(i))?,
        };
        Ok((value, UnitVariantOnly))
    }
}

/// Reads as a variant index and nothing else. The counterpart to
/// [`StringRefDe`] for the numeric encoding.
struct VariantIndexDe(u64);

impl<'de> Deserializer<'de> for VariantIndexDe {
    type Error = MatError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, MatError> {
        visitor.visit_u64(self.0)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

/// A scalar read as a fieldless variant's index.
///
/// Lenient on width and class, because the file may not have come from this
/// crate: MATLAB's default numeric class is `double`, so an index typed by
/// hand arrives as one. Anything that is not a non-negative whole number is
/// refused rather than rounded.
fn variant_index(s: &ScalarNum) -> Option<u64> {
    /// 2^64, exactly representable. One past the largest `u64`.
    const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the guard admits only a non-negative whole f64 below 2^64, which \
                  is exactly the range u64 represents, so the cast is lossless"
    )]
    fn whole(x: f64) -> Option<u64> {
        // Strict `<`: `u64::MAX as f64` rounds *up* to 2^64, so `<=` would
        // admit exactly 2^64 and then saturate it to `u64::MAX` on the cast,
        // which is the rounding this refuses everywhere else.
        (x >= 0.0 && x.fract() == 0.0 && x < TWO_POW_64).then_some(x as u64)
    }
    match *s {
        ScalarNum::U64(x) => Some(x),
        ScalarNum::U32(x) => Some(x as u64),
        ScalarNum::U16(x) => Some(x as u64),
        ScalarNum::U8(x) => Some(x as u64),
        ScalarNum::I64(x) => u64::try_from(x).ok(),
        ScalarNum::I32(x) => u64::try_from(x).ok(),
        ScalarNum::I16(x) => u64::try_from(x).ok(),
        ScalarNum::I8(x) => u64::try_from(x).ok(),
        ScalarNum::F64(x) => whole(x),
        ScalarNum::F32(x) => whole(x as f64),
        ScalarNum::Bool(_) => None,
    }
}

struct UnitVariantOnly;

impl<'de> VariantAccess<'de> for UnitVariantOnly {
    type Error = MatError;
    fn unit_variant(self) -> Result<(), MatError> {
        Ok(())
    }
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, _seed: T) -> Result<T::Value, MatError> {
        Err(MatError::UnsupportedType("newtype enum variant"))
    }
    fn tuple_variant<V: Visitor<'de>>(
        self,
        _len: usize,
        _visitor: V,
    ) -> Result<V::Value, MatError> {
        Err(MatError::UnsupportedType("tuple enum variant"))
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, MatError> {
        Err(MatError::UnsupportedType("struct enum variant"))
    }
}

// ---------------------------------------------------------------------------
// NumVec helper: build a 1-element NumVec from a scalar
// ---------------------------------------------------------------------------

impl NumVec {
    pub(crate) fn from_single(s: ScalarNum) -> NumVec {
        match s {
            ScalarNum::Bool(b) => NumVec::Bool(vec![b]),
            ScalarNum::F64(x) => NumVec::F64(vec![x]),
            ScalarNum::F32(x) => NumVec::F32(vec![x]),
            ScalarNum::I64(x) => NumVec::I64(vec![x]),
            ScalarNum::I32(x) => NumVec::I32(vec![x]),
            ScalarNum::I16(x) => NumVec::I16(vec![x]),
            ScalarNum::I8(x) => NumVec::I8(vec![x]),
            ScalarNum::U64(x) => NumVec::U64(vec![x]),
            ScalarNum::U32(x) => NumVec::U32(vec![x]),
            ScalarNum::U16(x) => NumVec::U16(vec![x]),
            ScalarNum::U8(x) => NumVec::U8(vec![x]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    fn opaque(class_name: &str, fields: Vec<(String, MatValue)>) -> MatValue {
        MatValue::Opaque {
            class_name: class_name.to_owned(),
            fields,
        }
    }

    // A `containers.Map` re-keys by its (user-controlled) string keys, so a key
    // equal to the reserved table metadata string must NOT be stripped — only
    // `table` / `timetable` carry that reserved entry.
    #[test]
    fn reserved_meta_key_is_preserved_for_non_tabular_opaque() {
        let value = opaque(
            "containers.Map",
            vec![
                (
                    TABLE_META_KEY.to_owned(),
                    MatValue::Scalar(ScalarNum::F64(1.0)),
                ),
                ("other".to_owned(), MatValue::Scalar(ScalarNum::F64(2.0))),
            ],
        );
        let map: BTreeMap<String, f64> =
            Deserialize::deserialize(MatValueDeserializer::new(value)).unwrap();
        assert_eq!(map.get(TABLE_META_KEY), Some(&1.0));
        assert_eq!(map.get("other"), Some(&2.0));
    }

    // A `table`'s reserved metadata entry is still hidden from ordinary targets.
    #[test]
    fn reserved_meta_key_is_stripped_for_tabular_opaque() {
        let value = opaque(
            "table",
            vec![
                (TABLE_META_KEY.to_owned(), MatValue::Struct(Vec::new())),
                ("Var1".to_owned(), MatValue::Scalar(ScalarNum::F64(3.0))),
            ],
        );
        let map: BTreeMap<String, f64> =
            Deserialize::deserialize(MatValueDeserializer::new(value)).unwrap();
        assert!(!map.contains_key(TABLE_META_KEY));
        assert_eq!(map.get("Var1"), Some(&3.0));
    }
}
