//! The core [`serde::Serializer`], which lowers any serializable input to a
//! [`Value`].
//!
//! The serializer makes one pass, collecting the input into the value tree, and
//! an emitter then walks the tree and writes the file.

use core::iter;
use core::mem;

use serde::ser::{
    Impossible, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeTuple,
    SerializeTupleStruct, Serializer,
};

use super::value::{Leaf, Value};
use crate::mat::complex::{complex_tag_for_array_sentinel, complex_tag_for_sentinel};
use crate::mat::error::MatError;
use crate::mat::matrix::{MATRIX_SENTINEL, complex_tag_for_matrix_sentinel};
use crate::mat::options::{EmptySequencePolicy, NullPolicy, Options, UnitVariantEncoding};
use crate::mat::value::{ComplexNum, ComplexTag, ComplexVec, NumVec, ScalarNum, ScalarTag};

// ---------------------------------------------------------------------------
// Public entry: serialize a value into a Value
// ---------------------------------------------------------------------------

pub(crate) fn to_value<T: Serialize + ?Sized>(
    value: &T,
    options: &Options,
) -> Result<Value, MatError> {
    value.serialize(ValueSerializer::new(options))
}

/// Lower a `None` / unit / unit-struct per [`Options::null_policy`].
///
/// The root serializer routes through this too, so `NullPolicy::Error` refuses a
/// root null with the same message it uses everywhere else, rather than the root
/// being the one slot the policy does not reach.
pub(super) fn null_value(opts: &Options) -> Result<Value, MatError> {
    match opts.null_policy {
        NullPolicy::EmptyStructArray => Ok(Value::Leaf(Leaf::EmptyStructArray)),
        NullPolicy::Omit => Ok(Value::Omit),
        NullPolicy::Error => Err(MatError::UnsupportedType(
            "null value under NullPolicy::Error",
        )),
    }
}

// ---------------------------------------------------------------------------
// ValueSerializer
// ---------------------------------------------------------------------------

/// Borrows the caller's [`Options`] so that the handful of decisions with no
/// single right answer (null lowering, unit-variant encoding, the class of an
/// empty sequence) are made where the input is still in view, rather than
/// being baked into the value tree and second-guessed by the emitter.
#[derive(Clone, Copy)]
pub(crate) struct ValueSerializer<'a> {
    opts: &'a Options,
}

impl<'a> ValueSerializer<'a> {
    pub(crate) fn new(opts: &'a Options) -> Self {
        Self { opts }
    }
}

impl<'a> Serializer for ValueSerializer<'a> {
    type Ok = Value;
    type Error = MatError;

    type SerializeSeq = SeqSer<'a>;
    type SerializeTuple = SeqSer<'a>;
    type SerializeTupleStruct = SeqSer<'a>;
    type SerializeTupleVariant = Impossible<Value, MatError>;
    type SerializeMap = MapSer<'a>;
    type SerializeStruct = StructSer<'a>;
    type SerializeStructVariant = Impossible<Value, MatError>;

    // ----- primitives -----

    fn serialize_bool(self, v: bool) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::Bool(v))))
    }
    fn serialize_i8(self, v: i8) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::I8(v))))
    }
    fn serialize_i16(self, v: i16) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::I16(v))))
    }
    fn serialize_i32(self, v: i32) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::I32(v))))
    }
    fn serialize_i64(self, v: i64) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::I64(v))))
    }
    fn serialize_i128(self, _v: i128) -> Result<Value, MatError> {
        Err(MatError::UnsupportedType(
            "i128 (MATLAB has no 128-bit integer)",
        ))
    }
    fn serialize_u8(self, v: u8) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::U8(v))))
    }
    fn serialize_u16(self, v: u16) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::U16(v))))
    }
    fn serialize_u32(self, v: u32) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::U32(v))))
    }
    fn serialize_u64(self, v: u64) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::U64(v))))
    }
    fn serialize_u128(self, _v: u128) -> Result<Value, MatError> {
        Err(MatError::UnsupportedType("u128"))
    }
    fn serialize_f32(self, v: f32) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::F32(v))))
    }
    fn serialize_f64(self, v: f64) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Scalar(ScalarNum::F64(v))))
    }

    fn serialize_char(self, v: char) -> Result<Value, MatError> {
        let mut buf = [0u8; 4];
        Ok(Value::Leaf(Leaf::String(
            v.encode_utf8(&mut buf).to_string(),
        )))
    }

    fn serialize_str(self, v: &str) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::String(v.to_owned())))
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::Vec1D(NumVec::U8(v.to_vec()))))
    }

    // ----- option / unit / newtype -----

    fn serialize_none(self) -> Result<Value, MatError> {
        null_value(self.opts)
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Value, MatError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Value, MatError> {
        // Unit lowers exactly like `serialize_none` above. The common way to
        // hit it is `serde_json::Value::Null`, which serializes via
        // `serialize_unit`; routing both through `null_policy` means the two
        // spellings of "no value" cannot come to disagree.
        null_value(self.opts)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value, MatError> {
        null_value(self.opts)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        idx: u32,
        variant: &'static str,
    ) -> Result<Value, MatError> {
        match self.opts.unit_variant_encoding {
            UnitVariantEncoding::Name => Ok(Value::Leaf(Leaf::String(variant.to_owned()))),
            UnitVariantEncoding::Index => Ok(Value::Leaf(Leaf::Scalar(ScalarNum::U32(idx)))),
        }
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Value, MatError> {
        // A bulk complex array: `mat::complex`'s array helpers hand the whole
        // slice over as one byte payload rather than paying a serializer
        // dispatch per sample. The result is the value the element-wise path
        // would have built, so the file is unchanged either way.
        if let Some(tag) = complex_tag_for_array_sentinel(name) {
            return value.serialize(ComplexArraySer { tag });
        }
        // Transparent newtype — pass through.
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Value, MatError> {
        Err(MatError::UnsupportedType("newtype enum variant"))
    }

    // ----- sequences -----

    fn serialize_seq(self, len: Option<usize>) -> Result<SeqSer<'a>, MatError> {
        Ok(SeqSer::new(len, self.opts))
    }
    fn serialize_tuple(self, len: usize) -> Result<SeqSer<'a>, MatError> {
        Ok(SeqSer::new(Some(len), self.opts))
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<SeqSer<'a>, MatError> {
        Ok(SeqSer::new(Some(len), self.opts))
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, MatError> {
        Err(MatError::UnsupportedType("tuple enum variant"))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<MapSer<'a>, MatError> {
        Ok(MapSer::new(self.opts))
    }

    fn serialize_struct(self, name: &'static str, len: usize) -> Result<StructSer<'a>, MatError> {
        let kind = if let Some(tag) = complex_tag_for_sentinel(name) {
            StructKind::Complex(tag, ComplexFields::default())
        } else if let Some(tag) = complex_tag_for_matrix_sentinel(name) {
            StructKind::Matrix(MatrixFields::default(), MatrixKind::Complex(tag))
        } else {
            match name {
                MATRIX_SENTINEL => StructKind::Matrix(MatrixFields::default(), MatrixKind::Numeric),
                // serde supplies the exact field count, so the field Vec can be
                // sized once instead of growing by reallocation.
                _ => StructKind::Plain(PlainStructFields::with_capacity(len)),
            }
        };
        Ok(StructSer {
            opts: self.opts,
            kind,
        })
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, MatError> {
        Err(MatError::UnsupportedType("struct enum variant"))
    }
}

// ---------------------------------------------------------------------------
// Bulk complex array: the payload of a `mat::complex` array sentinel
// ---------------------------------------------------------------------------

/// Turns one borrowed byte payload into a whole [`Leaf::ComplexVec1D`].
///
/// Only `serialize_bytes` is reachable. The array sentinels are `pub(crate)`
/// and `mat::complex`'s helpers are their only writer, so every other method
/// here is a bug in this crate rather than something a caller can provoke —
/// hence one shared message instead of a tailored one per type.
struct ComplexArraySer {
    tag: ComplexTag,
}

impl ComplexArraySer {
    fn wrong_payload() -> MatError {
        MatError::Custom("a complex array sentinel must carry its samples as bytes".to_owned())
    }
}

/// The scalar entry points, all of them refusals.
macro_rules! reject_payload {
    ($($method:ident($ty:ty)),* $(,)?) => {
        $(fn $method(self, _v: $ty) -> Result<Value, MatError> {
            Err(Self::wrong_payload())
        })*
    };
}

impl Serializer for ComplexArraySer {
    type Ok = Value;
    type Error = MatError;

    type SerializeSeq = Impossible<Value, MatError>;
    type SerializeTuple = Impossible<Value, MatError>;
    type SerializeTupleStruct = Impossible<Value, MatError>;
    type SerializeTupleVariant = Impossible<Value, MatError>;
    type SerializeMap = Impossible<Value, MatError>;
    type SerializeStruct = Impossible<Value, MatError>;
    type SerializeStructVariant = Impossible<Value, MatError>;

    fn serialize_bytes(self, v: &[u8]) -> Result<Value, MatError> {
        Ok(Value::Leaf(Leaf::ComplexVec1D(
            ComplexVec::from_native_bytes(self.tag, v)?,
        )))
    }

    reject_payload! {
        serialize_bool(bool),
        serialize_i8(i8), serialize_i16(i16), serialize_i32(i32), serialize_i64(i64),
        serialize_u8(u8), serialize_u16(u16), serialize_u32(u32), serialize_u64(u64),
        serialize_f32(f32), serialize_f64(f64),
        serialize_char(char), serialize_str(&str),
        serialize_unit_struct(&'static str),
    }

    fn serialize_none(self) -> Result<Value, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, _v: &T) -> Result<Value, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_unit(self) -> Result<Value, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
    ) -> Result<Value, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _v: &T,
    ) -> Result<Value, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _v: &T,
    ) -> Result<Value, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, MatError> {
        Err(Self::wrong_payload())
    }
}

// ---------------------------------------------------------------------------
// Sequence serializer: handles Vec<T>, [T; N], tuples, tuple structs
// ---------------------------------------------------------------------------

pub(crate) struct SeqSer<'a> {
    accum: SeqAccum,
    opts: &'a Options,
}

impl<'a> SeqSer<'a> {
    fn new(len: Option<usize>, opts: &'a Options) -> Self {
        Self {
            accum: SeqAccum::Empty {
                cap: len.unwrap_or(0),
            },
            opts,
        }
    }

    fn push<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), MatError> {
        let value = v.serialize(ValueSerializer::new(self.opts))?;
        let accum = mem::replace(&mut self.accum, SeqAccum::Empty { cap: 0 });
        self.accum = accum.pushed(value)?;
        Ok(())
    }

    fn finish(self) -> Result<Value, MatError> {
        self.accum.finish(self.opts)
    }
}

impl SerializeSeq for SeqSer<'_> {
    type Ok = Value;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<Value, MatError> {
        self.finish()
    }
}

impl SerializeTuple for SeqSer<'_> {
    type Ok = Value;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<Value, MatError> {
        self.finish()
    }
}

impl SerializeTupleStruct for SeqSer<'_> {
    type Ok = Value;
    type Error = MatError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<Value, MatError> {
        self.finish()
    }
}

/// How a sequence's elements are held while it is being collected, and with
/// that what the sequence has turned out to be: same-class scalars are a
/// numeric or complex vector, same-class same-length vectors are the rows of a
/// matrix, and anything else is a cell array. A sequence with no elements takes
/// the class [`EmptySequencePolicy`] names.
///
/// A flat numeric or complex array is the common large case, and holding it as
/// one `Value` per element costs 56 bytes for what packs into 4. So the
/// accumulator stays packed for as long as the elements agree, and only falls
/// back to one value per element when one of them breaks the pattern.
enum SeqAccum {
    /// Every element so far is a complex vector of one class and length.
    ComplexRows {
        tag: ComplexTag,
        cols: usize,
        rows: Vec<ComplexVec>,
    },
    /// Every element so far is a complex scalar of one class.
    ComplexScalars(ComplexVec),
    /// Nothing pushed yet. The first element picks the state.
    Empty { cap: usize },
    /// Anything else, one value per element.
    Mixed(Vec<Value>),
    /// Every element so far is a numeric vector of one class and length.
    Rows {
        tag: ScalarTag,
        cols: usize,
        rows: Vec<NumVec>,
    },
    /// Every element so far is a numeric scalar of one class.
    Scalars(NumVec),
}

impl SeqAccum {
    /// Returns the state a first element establishes.
    fn started(cap: usize, value: Value) -> Result<Self, MatError> {
        Ok(match value {
            Value::Leaf(Leaf::ComplexScalar(c)) => {
                let mut packed = ComplexVec::with_capacity_for_tag(c.tag(), cap);
                packed.push(c)?;
                SeqAccum::ComplexScalars(packed)
            }
            Value::Leaf(Leaf::ComplexVec1D(v)) => {
                let (tag, cols) = (v.tag(), v.len());
                let mut rows = Vec::with_capacity(cap);
                rows.push(v);
                SeqAccum::ComplexRows { tag, cols, rows }
            }
            Value::Leaf(Leaf::Scalar(s)) => {
                let mut packed = NumVec::with_capacity_for_tag(s.tag(), cap);
                packed.push(s)?;
                SeqAccum::Scalars(packed)
            }
            Value::Leaf(Leaf::Vec1D(v)) => {
                let (tag, cols) = (v.tag(), v.len());
                let mut rows = Vec::with_capacity(cap);
                rows.push(v);
                SeqAccum::Rows { tag, cols, rows }
            }
            other => {
                let mut values = Vec::with_capacity(cap);
                values.push(other);
                SeqAccum::Mixed(values)
            }
        })
    }

    /// Returns the state after `value`: appended to the packed run, or
    /// [`SeqAccum::Mixed`] when `value` breaks the pattern.
    fn pushed(self, value: Value) -> Result<Self, MatError> {
        Ok(match self {
            SeqAccum::ComplexRows {
                tag,
                cols,
                mut rows,
            } => match value {
                Value::Leaf(Leaf::ComplexVec1D(v)) if v.tag() == tag && v.len() == cols => {
                    rows.push(v);
                    SeqAccum::ComplexRows { tag, cols, rows }
                }
                other => SeqAccum::Mixed(
                    rows.into_iter()
                        .map(|v| Value::Leaf(Leaf::ComplexVec1D(v)))
                        .chain(iter::once(other))
                        .collect(),
                ),
            },
            SeqAccum::ComplexScalars(mut packed) => match value {
                Value::Leaf(Leaf::ComplexScalar(c)) if c.tag() == packed.tag() => {
                    packed.push(c)?;
                    SeqAccum::ComplexScalars(packed)
                }
                other => SeqAccum::Mixed(
                    packed
                        .into_pairs()
                        .map(|c| Value::Leaf(Leaf::ComplexScalar(c)))
                        .chain(iter::once(other))
                        .collect(),
                ),
            },
            SeqAccum::Empty { cap } => SeqAccum::started(cap, value)?,
            SeqAccum::Mixed(mut values) => {
                values.push(value);
                SeqAccum::Mixed(values)
            }
            SeqAccum::Rows {
                tag,
                cols,
                mut rows,
            } => match value {
                Value::Leaf(Leaf::Vec1D(v)) if v.tag() == tag && v.len() == cols => {
                    rows.push(v);
                    SeqAccum::Rows { tag, cols, rows }
                }
                other => SeqAccum::Mixed(
                    rows.into_iter()
                        .map(|v| Value::Leaf(Leaf::Vec1D(v)))
                        .chain(iter::once(other))
                        .collect(),
                ),
            },
            SeqAccum::Scalars(mut packed) => match value {
                Value::Leaf(Leaf::Scalar(s)) if s.tag() == packed.tag() => {
                    packed.push(s)?;
                    SeqAccum::Scalars(packed)
                }
                other => SeqAccum::Mixed(
                    packed
                        .into_scalars()
                        .map(|s| Value::Leaf(Leaf::Scalar(s)))
                        .chain(iter::once(other))
                        .collect(),
                ),
            },
        })
    }

    /// Returns the value the collected elements amount to.
    fn finish(self, opts: &Options) -> Result<Value, MatError> {
        Ok(match self {
            SeqAccum::ComplexRows { tag, cols, rows } => {
                let mut pairs = ComplexVec::with_capacity_for_tag(tag, rows.len() * cols);
                let count = rows.len();
                for row in rows {
                    pairs.extend(row)?;
                }
                Value::Leaf(Leaf::ComplexMatrix {
                    rows: count,
                    cols,
                    pairs,
                })
            }
            SeqAccum::ComplexScalars(packed) => Value::Leaf(Leaf::ComplexVec1D(packed)),
            // No element revealed its type, so the MATLAB class is the
            // caller's to pick. See `EmptySequencePolicy`.
            SeqAccum::Empty { .. } => match opts.empty_sequence_policy {
                EmptySequencePolicy::DoubleArray => {
                    Value::Leaf(Leaf::Vec1D(NumVec::F64(Vec::new())))
                }
                EmptySequencePolicy::Cell => Value::Cell(Vec::new()),
            },
            SeqAccum::Mixed(values) => Value::Cell(values.into_iter().map(cell_element).collect()),
            SeqAccum::Rows { tag, cols, rows } => {
                let mut vec = NumVec::with_capacity_for_tag(tag, rows.len() * cols);
                let count = rows.len();
                for row in rows {
                    vec.extend(row)?;
                }
                Value::Leaf(Leaf::Matrix {
                    rows: count,
                    cols,
                    vec,
                })
            }
            SeqAccum::Scalars(packed) => Value::Leaf(Leaf::Vec1D(packed)),
        })
    }
}

/// Returns `value` as a cell-array element.
///
/// `Omit` has no element form: a `None` inside a sequence holds its slot as
/// MATLAB's `struct([])`.
fn cell_element(value: Value) -> Value {
    match value {
        Value::Omit => Value::Leaf(Leaf::EmptyStructArray),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Map serializer: HashMap<String, T> → struct
// ---------------------------------------------------------------------------

pub(crate) struct MapSer<'a> {
    fields: Vec<(String, Value)>,
    pending_key: Option<String>,
    opts: &'a Options,
}

impl<'a> MapSer<'a> {
    fn new(opts: &'a Options) -> Self {
        Self {
            fields: Vec::new(),
            pending_key: None,
            opts,
        }
    }
}

impl SerializeMap for MapSer<'_> {
    type Ok = Value;
    type Error = MatError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), MatError> {
        let key_val = key.serialize(ValueSerializer::new(self.opts))?;
        let key_str = match key_val {
            Value::Leaf(Leaf::String(s)) => s,
            other => {
                return Err(MatError::UnsupportedType(match other.kind() {
                    "struct" => "map with non-string keys (struct as key)",
                    _ => "map with non-string keys",
                }));
            }
        };
        self.pending_key = Some(key_str);
        Ok(())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        let key = self.pending_key.take().ok_or_else(|| {
            MatError::Custom("serialize_value called before serialize_key".into())
        })?;
        let val = value.serialize(ValueSerializer::new(self.opts))?;
        if !matches!(val, Value::Omit) {
            self.fields.push((key, val));
        }
        Ok(())
    }

    fn end(self) -> Result<Value, MatError> {
        Ok(Value::Struct(self.fields))
    }
}

// ---------------------------------------------------------------------------
// Struct serializer: dispatches between Matrix sentinel, Complex sentinels,
// and a plain MATLAB-struct group.
// ---------------------------------------------------------------------------

pub(crate) struct StructSer<'a> {
    opts: &'a Options,
    kind: StructKind,
}

pub(crate) enum StructKind {
    Matrix(MatrixFields, MatrixKind),
    Complex(ComplexTag, ComplexFields),
    Plain(PlainStructFields),
}

/// Element class hint for a `Matrix<T>` sentinel. Carried through from the
/// chosen sentinel name (see `matrix::MatElement`) so that empty matrices,
/// where the inner `Vec<T>` cannot reveal `T`, still emit with the right
/// MATLAB class.
#[derive(Clone, Copy)]
pub(crate) enum MatrixKind {
    Numeric,
    Complex(ComplexTag),
}

#[derive(Default)]
pub(crate) struct MatrixFields {
    rows: Option<usize>,
    cols: Option<usize>,
    data: Option<Value>,
}

/// The two fields of a complex sentinel, held as tagged scalars: the component
/// class comes from the sentinel, and `end` checks the fields against it.
#[derive(Default)]
pub(crate) struct ComplexFields {
    real: Option<ScalarNum>,
    imag: Option<ScalarNum>,
}

#[derive(Default)]
pub(crate) struct PlainStructFields {
    fields: Vec<(String, Value)>,
}

impl PlainStructFields {
    fn with_capacity(n: usize) -> Self {
        Self {
            fields: Vec::with_capacity(n),
        }
    }
}

impl SerializeStruct for StructSer<'_> {
    type Ok = Value;
    type Error = MatError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), MatError> {
        let vs = ValueSerializer::new(self.opts);
        match &mut self.kind {
            StructKind::Matrix(fields, _) => match key {
                "rows" => {
                    let v = value.serialize(vs)?;
                    fields.rows = Some(expect_usize(v, "Matrix::rows")?);
                }
                "cols" => {
                    let v = value.serialize(vs)?;
                    fields.cols = Some(expect_usize(v, "Matrix::cols")?);
                }
                "data" => {
                    // `Matrix::data` is the sentinel's own payload, not a
                    // sequence the caller wrote, so `empty_sequence_policy` has
                    // no business reaching it. Under `Cell` an empty matrix
                    // lowered to a cell array, and `matrix_from_fields` then had
                    // no `Vec1D` left to recover the element class from, which
                    // made every empty `Matrix<T>` unserializable. Pin the policy
                    // rather than teaching the sentinel handler to accept a shape
                    // it should never receive.
                    let pinned = Options {
                        empty_sequence_policy: EmptySequencePolicy::DoubleArray,
                        ..self.opts.clone()
                    };
                    let v = value.serialize(ValueSerializer::new(&pinned))?;
                    fields.data = Some(v);
                }
                other => {
                    return Err(MatError::Custom(format!(
                        "unexpected field {other:?} on Matrix sentinel"
                    )));
                }
            },
            StructKind::Complex(tag, fields) => match key {
                "real" => fields.real = Some(expect_component(value.serialize(vs)?)?),
                "imag" => fields.imag = Some(expect_component(value.serialize(vs)?)?),
                other => {
                    return Err(MatError::Custom(format!(
                        "unexpected field {other:?} on the complex {} sentinel",
                        tag.class().as_str()
                    )));
                }
            },
            StructKind::Plain(ps) => {
                let v = value.serialize(vs)?;
                ps.fields.push((key.to_owned(), v));
            }
        }
        Ok(())
    }

    fn end(self) -> Result<Value, MatError> {
        match self.kind {
            StructKind::Plain(ps) => Ok(Value::Struct(ps.fields)),
            StructKind::Matrix(fields, kind) => matrix_from_fields(fields, kind),
            StructKind::Complex(tag, fields) => {
                let re = fields
                    .real
                    .ok_or_else(|| MatError::MissingField("real".into()))?;
                let im = fields
                    .imag
                    .ok_or_else(|| MatError::MissingField("imag".into()))?;
                let n = ComplexNum::from_components(tag, re, im).ok_or_else(|| {
                    MatError::Custom(format!(
                        "complex {} fields must both be {}",
                        tag.class().as_str(),
                        tag.class().as_str()
                    ))
                })?;
                Ok(Value::Leaf(Leaf::ComplexScalar(n)))
            }
        }
    }
}

fn matrix_from_fields(fields: MatrixFields, kind: MatrixKind) -> Result<Value, MatError> {
    let rows = fields
        .rows
        .ok_or_else(|| MatError::MissingField("rows".into()))?;
    let cols = fields
        .cols
        .ok_or_else(|| MatError::MissingField("cols".into()))?;
    let data = fields
        .data
        .ok_or_else(|| MatError::MissingField("data".into()))?;
    // `rows` and `cols` arrive from the serialized input, so the product can
    // overflow. Refuse it here: a wrapped total would agree with a short data
    // vector and let the pair through to a writer that transposes through a raw
    // pointer sized from that same product.
    let total = rows.checked_mul(cols).ok_or_else(|| {
        MatError::Custom(format!(
            "Matrix dimensions {rows}x{cols} overflow the address space"
        ))
    })?;
    let length_check = |actual: usize| -> Result<(), MatError> {
        if actual != total {
            return Err(MatError::Custom(format!(
                "Matrix::data length {} does not match rows*cols = {}",
                actual, total
            )));
        }
        Ok(())
    };
    match (kind, data) {
        // Numeric Matrix<T>: data unifies to Vec1D of T's tag. Matrix<Complex*>
        // does not land here; it carries `T::SENTINEL = MATRIX_COMPLEX{32,64}_SENTINEL`
        // and routes to the dedicated Complex64 / Complex32 arms below.
        (MatrixKind::Numeric, Value::Leaf(Leaf::Vec1D(vec))) => {
            length_check(vec.len())?;
            Ok(Value::Leaf(Leaf::Matrix { rows, cols, vec }))
        }
        // Matrix<Complex*>: a ComplexVec1D of the sentinel's class in the data
        // slot, OR an empty Vec1D (the f64-default that an empty
        // `Vec<Complex*>` collapses to in the seq path: with no elements
        // observed, the seq unification can't recover T). The dedicated
        // sentinel here lets us recover the class.
        (MatrixKind::Complex(tag), Value::Leaf(Leaf::ComplexVec1D(pairs)))
            if pairs.tag() == tag =>
        {
            length_check(pairs.len())?;
            Ok(Value::Leaf(Leaf::ComplexMatrix { rows, cols, pairs }))
        }
        (MatrixKind::Complex(tag), Value::Leaf(Leaf::Vec1D(vec))) if vec.is_empty() => {
            length_check(0)?;
            Ok(Value::Leaf(Leaf::ComplexMatrix {
                rows,
                cols,
                pairs: ComplexVec::empty_with_tag(tag),
            }))
        }
        (_, other) => Err(MatError::Custom(format!(
            "Matrix::data must be a Vec<T>, got {}",
            other.kind()
        ))),
    }
}

fn expect_usize(v: Value, field: &str) -> Result<usize, MatError> {
    match v {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "serde scalar accessor; conversion of a user-level MAT scalar (e.g. a Matrix rows/cols field) to usize, not a file-derived size"
        )]
        Value::Leaf(Leaf::Scalar(ScalarNum::U64(x))) => Ok(x as usize),
        Value::Leaf(Leaf::Scalar(ScalarNum::U32(x))) => Ok(x as usize),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the `x >= 0` guard keeps the I64 non-negative; serde scalar accessor converting a user-level MAT scalar to usize, not a file-derived size"
        )]
        Value::Leaf(Leaf::Scalar(ScalarNum::I64(x))) if x >= 0 => Ok(x as usize),
        Value::Leaf(Leaf::Scalar(ScalarNum::I32(x))) if x >= 0 => Ok(x as usize),
        Value::Leaf(Leaf::Scalar(ScalarNum::U16(x))) => Ok(x as usize),
        Value::Leaf(Leaf::Scalar(ScalarNum::U8(x))) => Ok(x as usize),
        other => Err(MatError::Custom(format!(
            "{field} must be an unsigned integer, got {}",
            other.kind()
        ))),
    }
}

/// A complex sentinel's `real`/`imag` field, kept at the width it was
/// serialized at. `end` checks it against the sentinel's class.
fn expect_component(v: Value) -> Result<ScalarNum, MatError> {
    match v {
        Value::Leaf(Leaf::Scalar(s)) => Ok(s),
        other => Err(MatError::Custom(format!(
            "a complex field must be a numeric scalar, got {}",
            other.kind()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use crate::mat::complex::Complex64;

    use super::*;

    #[test]
    fn scalars_of_one_class_become_a_vector() {
        assert_eq!(
            to_value(&vec![1u8, 2, 3], &Options::default()).unwrap(),
            Value::Leaf(Leaf::Vec1D(NumVec::U8(vec![1, 2, 3]))),
        );
    }

    #[test]
    fn scalars_of_two_classes_become_a_cell_array() {
        assert_eq!(
            to_value(&(1u8, 2u16), &Options::default()).unwrap(),
            Value::Cell(vec![
                Value::Leaf(Leaf::Scalar(ScalarNum::U8(1))),
                Value::Leaf(Leaf::Scalar(ScalarNum::U16(2))),
            ]),
        );
    }

    #[test]
    fn rows_of_one_class_and_length_become_a_matrix() {
        assert_eq!(
            to_value(&vec![vec![1u8, 2], vec![3, 4]], &Options::default()).unwrap(),
            Value::Leaf(Leaf::Matrix {
                rows: 2,
                cols: 2,
                vec: NumVec::U8(vec![1, 2, 3, 4]),
            }),
        );
    }

    #[test]
    fn ragged_rows_become_a_cell_array() {
        assert_eq!(
            to_value(&vec![vec![1u8, 2], vec![3]], &Options::default()).unwrap(),
            Value::Cell(vec![
                Value::Leaf(Leaf::Vec1D(NumVec::U8(vec![1, 2]))),
                Value::Leaf(Leaf::Vec1D(NumVec::U8(vec![3]))),
            ]),
        );
    }

    #[test]
    fn a_scalar_after_a_row_spills_the_rows_into_the_cell_array() {
        assert_eq!(
            to_value(&(vec![1u8, 2], 3u8), &Options::default()).unwrap(),
            Value::Cell(vec![
                Value::Leaf(Leaf::Vec1D(NumVec::U8(vec![1, 2]))),
                Value::Leaf(Leaf::Scalar(ScalarNum::U8(3))),
            ]),
        );
    }

    #[test]
    fn complex_rows_of_one_class_and_length_become_a_complex_matrix() {
        assert_eq!(
            to_value(
                &vec![
                    vec![Complex64::new(1.0, 2.0)],
                    vec![Complex64::new(3.0, 4.0)],
                ],
                &Options::default(),
            )
            .unwrap(),
            Value::Leaf(Leaf::ComplexMatrix {
                rows: 2,
                cols: 1,
                pairs: ComplexVec::F64(vec![(1.0, 2.0), (3.0, 4.0)]),
            }),
        );
    }

    #[test]
    fn an_omitted_element_holds_its_slot_in_the_cell_array() {
        let opts = Options {
            null_policy: NullPolicy::Omit,
            ..Options::default()
        };
        assert_eq!(
            to_value(&(Some(1.0f64), None::<f64>), &opts).unwrap(),
            Value::Cell(vec![
                Value::Leaf(Leaf::Scalar(ScalarNum::F64(1.0))),
                Value::Leaf(Leaf::EmptyStructArray),
            ]),
        );
    }
}
