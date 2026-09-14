//! `ValueSerializer`: the core `serde::Serializer` that produces a
//! [`MatValue`] from any serializable input.
//!
//! The serializer makes one pass to collect everything into the intermediate
//! tree, then the emitter walks the tree to build the HDF5 file.

use core::iter;
use core::mem;

use serde::ser::{
    Impossible, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeTuple,
    SerializeTupleStruct, Serializer,
};

use crate::mat::complex::{complex_tag_for_array_sentinel, complex_tag_for_sentinel};
use crate::mat::error::MatError;
use crate::mat::matrix::{MATRIX_SENTINEL, complex_tag_for_matrix_sentinel};
use crate::mat::options::{EmptySequencePolicy, NullPolicy, Options, UnitVariantEncoding};

use crate::mat::value::{
    ComplexNum, ComplexTag, ComplexVec, MatValue, NumVec, ScalarNum, ScalarTag,
};

// ---------------------------------------------------------------------------
// Public entry: serialize a value into a MatValue
// ---------------------------------------------------------------------------

pub(crate) fn to_value<T: Serialize + ?Sized>(
    value: &T,
    options: &Options,
) -> Result<MatValue, MatError> {
    value.serialize(ValueSerializer::new(options))
}

/// Lower a `None` / unit / unit-struct per [`Options::null_policy`].
///
/// The root serializer routes through this too, so `NullPolicy::Error` refuses a
/// root null with the same message it uses everywhere else, rather than the root
/// being the one slot the policy does not reach.
pub(super) fn null_value(opts: &Options) -> Result<MatValue, MatError> {
    match opts.null_policy {
        NullPolicy::EmptyStructArray => Ok(MatValue::EmptyStructArray),
        NullPolicy::Omit => Ok(MatValue::Omit),
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
    type Ok = MatValue;
    type Error = MatError;

    type SerializeSeq = SeqSer<'a>;
    type SerializeTuple = SeqSer<'a>;
    type SerializeTupleStruct = SeqSer<'a>;
    type SerializeTupleVariant = Impossible<MatValue, MatError>;
    type SerializeMap = MapSer<'a>;
    type SerializeStruct = StructSer<'a>;
    type SerializeStructVariant = Impossible<MatValue, MatError>;

    // ----- primitives -----

    fn serialize_bool(self, v: bool) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::Bool(v)))
    }
    fn serialize_i8(self, v: i8) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I8(v)))
    }
    fn serialize_i16(self, v: i16) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I16(v)))
    }
    fn serialize_i32(self, v: i32) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I32(v)))
    }
    fn serialize_i64(self, v: i64) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::I64(v)))
    }
    fn serialize_i128(self, _v: i128) -> Result<MatValue, MatError> {
        Err(MatError::UnsupportedType(
            "i128 (MATLAB has no 128-bit integer)",
        ))
    }
    fn serialize_u8(self, v: u8) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U8(v)))
    }
    fn serialize_u16(self, v: u16) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U16(v)))
    }
    fn serialize_u32(self, v: u32) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U32(v)))
    }
    fn serialize_u64(self, v: u64) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::U64(v)))
    }
    fn serialize_u128(self, _v: u128) -> Result<MatValue, MatError> {
        Err(MatError::UnsupportedType("u128"))
    }
    fn serialize_f32(self, v: f32) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::F32(v)))
    }
    fn serialize_f64(self, v: f64) -> Result<MatValue, MatError> {
        Ok(MatValue::Scalar(ScalarNum::F64(v)))
    }

    fn serialize_char(self, v: char) -> Result<MatValue, MatError> {
        let mut buf = [0u8; 4];
        Ok(MatValue::String(v.encode_utf8(&mut buf).to_string()))
    }

    fn serialize_str(self, v: &str) -> Result<MatValue, MatError> {
        Ok(MatValue::String(v.to_owned()))
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<MatValue, MatError> {
        Ok(MatValue::Vec1D(NumVec::U8(v.to_vec())))
    }

    // ----- option / unit / newtype -----

    fn serialize_none(self) -> Result<MatValue, MatError> {
        null_value(self.opts)
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<MatValue, MatError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<MatValue, MatError> {
        // Unit lowers exactly like `serialize_none` above. The common way to
        // hit it is `serde_json::Value::Null`, which serializes via
        // `serialize_unit`; routing both through `null_policy` means the two
        // spellings of "no value" cannot come to disagree.
        null_value(self.opts)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<MatValue, MatError> {
        null_value(self.opts)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        idx: u32,
        variant: &'static str,
    ) -> Result<MatValue, MatError> {
        match self.opts.unit_variant_encoding {
            UnitVariantEncoding::Name => Ok(MatValue::String(variant.to_owned())),
            UnitVariantEncoding::Index => Ok(MatValue::Scalar(ScalarNum::U32(idx))),
        }
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<MatValue, MatError> {
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
    ) -> Result<MatValue, MatError> {
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

/// Turns one borrowed byte payload into a whole [`MatValue::ComplexVec1D`].
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
        $(fn $method(self, _v: $ty) -> Result<MatValue, MatError> {
            Err(Self::wrong_payload())
        })*
    };
}

impl Serializer for ComplexArraySer {
    type Ok = MatValue;
    type Error = MatError;

    type SerializeSeq = Impossible<MatValue, MatError>;
    type SerializeTuple = Impossible<MatValue, MatError>;
    type SerializeTupleStruct = Impossible<MatValue, MatError>;
    type SerializeTupleVariant = Impossible<MatValue, MatError>;
    type SerializeMap = Impossible<MatValue, MatError>;
    type SerializeStruct = Impossible<MatValue, MatError>;
    type SerializeStructVariant = Impossible<MatValue, MatError>;

    fn serialize_bytes(self, v: &[u8]) -> Result<MatValue, MatError> {
        Ok(MatValue::ComplexVec1D(ComplexVec::from_native_bytes(
            self.tag, v,
        )?))
    }

    reject_payload! {
        serialize_bool(bool),
        serialize_i8(i8), serialize_i16(i16), serialize_i32(i32), serialize_i64(i64),
        serialize_u8(u8), serialize_u16(u16), serialize_u32(u32), serialize_u64(u64),
        serialize_f32(f32), serialize_f64(f64),
        serialize_char(char), serialize_str(&str),
        serialize_unit_struct(&'static str),
    }

    fn serialize_none(self) -> Result<MatValue, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, _v: &T) -> Result<MatValue, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_unit(self) -> Result<MatValue, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
    ) -> Result<MatValue, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _v: &T,
    ) -> Result<MatValue, MatError> {
        Err(Self::wrong_payload())
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _v: &T,
    ) -> Result<MatValue, MatError> {
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

    fn finish(self) -> Result<MatValue, MatError> {
        self.accum.finish(self.opts)
    }
}

impl SerializeSeq for SeqSer<'_> {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

impl SerializeTuple for SeqSer<'_> {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

impl SerializeTupleStruct for SeqSer<'_> {
    type Ok = MatValue;
    type Error = MatError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        self.push(value)
    }
    fn end(self) -> Result<MatValue, MatError> {
        self.finish()
    }
}

/// How a sequence's elements are held while it is being collected, and with
/// that what the sequence has turned out to be: same-class scalars are a
/// numeric or complex vector, same-class same-length vectors are the rows of a
/// matrix, and anything else is a cell array.
///
/// A flat numeric or complex array is the common large case, and holding it as
/// one `MatValue` per element costs 56 bytes for what packs into 4. So the
/// accumulator stays packed for as long as the elements agree, and only falls
/// back to one value per element when one of them breaks the pattern. That
/// fallback is where the sequence was always going to end up anyway, so the
/// cost lands on the sequences that were never packed to begin with.
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
    Mixed(Vec<MatValue>),
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
    /// The state one element on its own establishes.
    fn started(cap: usize, value: MatValue) -> Result<Self, MatError> {
        Ok(match value {
            MatValue::ComplexScalar(c) => {
                let mut packed = ComplexVec::with_capacity_for_tag(c.tag(), cap);
                packed.push(c)?;
                SeqAccum::ComplexScalars(packed)
            }
            MatValue::ComplexVec1D(v) => {
                let (tag, cols) = (v.tag(), v.len());
                let mut rows = Vec::with_capacity(cap);
                rows.push(v);
                SeqAccum::ComplexRows { tag, cols, rows }
            }
            MatValue::Scalar(s) => {
                let mut packed = NumVec::with_capacity_for_tag(s.tag(), cap);
                packed.push(s)?;
                SeqAccum::Scalars(packed)
            }
            MatValue::Vec1D(v) => {
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

    /// Take `value` into the state the elements before it established, or spill
    /// to [`SeqAccum::Mixed`] when it breaks the pattern.
    fn pushed(self, value: MatValue) -> Result<Self, MatError> {
        Ok(match self {
            SeqAccum::ComplexRows {
                tag,
                cols,
                mut rows,
            } => match value {
                MatValue::ComplexVec1D(v) if v.tag() == tag && v.len() == cols => {
                    rows.push(v);
                    SeqAccum::ComplexRows { tag, cols, rows }
                }
                broke => SeqAccum::Mixed(
                    rows.into_iter()
                        .map(MatValue::ComplexVec1D)
                        .chain(iter::once(broke))
                        .collect(),
                ),
            },
            SeqAccum::ComplexScalars(mut packed) => match value {
                MatValue::ComplexScalar(c) if c.tag() == packed.tag() => {
                    packed.push(c)?;
                    SeqAccum::ComplexScalars(packed)
                }
                broke => SeqAccum::Mixed(
                    packed
                        .into_pairs()
                        .map(MatValue::ComplexScalar)
                        .chain(iter::once(broke))
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
                MatValue::Vec1D(v) if v.tag() == tag && v.len() == cols => {
                    rows.push(v);
                    SeqAccum::Rows { tag, cols, rows }
                }
                broke => SeqAccum::Mixed(
                    rows.into_iter()
                        .map(MatValue::Vec1D)
                        .chain(iter::once(broke))
                        .collect(),
                ),
            },
            SeqAccum::Scalars(mut packed) => match value {
                MatValue::Scalar(s) if s.tag() == packed.tag() => {
                    packed.push(s)?;
                    SeqAccum::Scalars(packed)
                }
                broke => SeqAccum::Mixed(
                    packed
                        .into_scalars()
                        .map(MatValue::Scalar)
                        .chain(iter::once(broke))
                        .collect(),
                ),
            },
        })
    }

    /// The value the collected elements amount to.
    fn finish(self, opts: &Options) -> Result<MatValue, MatError> {
        Ok(match self {
            SeqAccum::ComplexRows { tag, cols, rows } => {
                let mut pairs = ComplexVec::with_capacity_for_tag(tag, rows.len() * cols);
                let count = rows.len();
                for row in rows {
                    pairs.extend(row)?;
                }
                MatValue::ComplexMatrix {
                    rows: count,
                    cols,
                    pairs,
                }
            }
            SeqAccum::ComplexScalars(packed) => MatValue::ComplexVec1D(packed),
            // No element revealed its type, so the MATLAB class is the
            // caller's to pick. See `EmptySequencePolicy`.
            SeqAccum::Empty { .. } => match opts.empty_sequence_policy {
                EmptySequencePolicy::DoubleArray => MatValue::Vec1D(NumVec::F64(Vec::new())),
                EmptySequencePolicy::Cell => MatValue::Cell(Vec::new()),
            },
            SeqAccum::Mixed(values) => {
                MatValue::Cell(values.into_iter().map(cell_element).collect())
            }
            SeqAccum::Rows { tag, cols, rows } => {
                let mut vec = NumVec::with_capacity_for_tag(tag, rows.len() * cols);
                let count = rows.len();
                for row in rows {
                    vec.extend(row)?;
                }
                MatValue::Matrix {
                    rows: count,
                    cols,
                    vec,
                }
            }
            SeqAccum::Scalars(packed) => MatValue::Vec1D(packed),
        })
    }
}

/// A cell array element. `Omit` has no element form: a `None` inside a
/// sequence holds its slot, as MATLAB's `struct([])`.
fn cell_element(value: MatValue) -> MatValue {
    match value {
        MatValue::Omit => MatValue::EmptyStructArray,
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Map serializer: HashMap<String, T> → struct
// ---------------------------------------------------------------------------

pub(crate) struct MapSer<'a> {
    fields: Vec<(String, MatValue)>,
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
    type Ok = MatValue;
    type Error = MatError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), MatError> {
        let key_val = key.serialize(ValueSerializer::new(self.opts))?;
        let key_str = match key_val {
            MatValue::String(s) => s,
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
        if !matches!(val, MatValue::Omit) {
            self.fields.push((key, val));
        }
        Ok(())
    }

    fn end(self) -> Result<MatValue, MatError> {
        Ok(MatValue::Struct(self.fields))
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
    data: Option<MatValue>,
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
    fields: Vec<(String, MatValue)>,
}

impl PlainStructFields {
    fn with_capacity(n: usize) -> Self {
        Self {
            fields: Vec::with_capacity(n),
        }
    }
}

impl SerializeStruct for StructSer<'_> {
    type Ok = MatValue;
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

    fn end(self) -> Result<MatValue, MatError> {
        match self.kind {
            StructKind::Plain(ps) => Ok(MatValue::Struct(ps.fields)),
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
                Ok(MatValue::ComplexScalar(n))
            }
        }
    }
}

fn matrix_from_fields(fields: MatrixFields, kind: MatrixKind) -> Result<MatValue, MatError> {
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
        (MatrixKind::Numeric, MatValue::Vec1D(vec)) => {
            length_check(vec.len())?;
            Ok(MatValue::Matrix { rows, cols, vec })
        }
        // Matrix<Complex*>: a ComplexVec1D of the sentinel's class in the data
        // slot, OR an empty Vec1D (the f64-default that an empty
        // `Vec<Complex*>` collapses to in the seq path: with no elements
        // observed, the seq unification can't recover T). The dedicated
        // sentinel here lets us recover the class.
        (MatrixKind::Complex(tag), MatValue::ComplexVec1D(pairs)) if pairs.tag() == tag => {
            length_check(pairs.len())?;
            Ok(MatValue::ComplexMatrix { rows, cols, pairs })
        }
        (MatrixKind::Complex(tag), MatValue::Vec1D(vec)) if vec.is_empty() => {
            length_check(0)?;
            Ok(MatValue::ComplexMatrix {
                rows,
                cols,
                pairs: ComplexVec::empty_with_tag(tag),
            })
        }
        (_, other) => Err(MatError::Custom(format!(
            "Matrix::data must be a Vec<T>, got {}",
            other.kind()
        ))),
    }
}

fn expect_usize(v: MatValue, field: &str) -> Result<usize, MatError> {
    match v {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "serde scalar accessor; conversion of a user-level MAT scalar (e.g. a Matrix rows/cols field) to usize, not a file-derived size"
        )]
        MatValue::Scalar(ScalarNum::U64(x)) => Ok(x as usize),
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(x as usize),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the `x >= 0` guard keeps the I64 non-negative; serde scalar accessor converting a user-level MAT scalar to usize, not a file-derived size"
        )]
        MatValue::Scalar(ScalarNum::I64(x)) if x >= 0 => Ok(x as usize),
        MatValue::Scalar(ScalarNum::I32(x)) if x >= 0 => Ok(x as usize),
        MatValue::Scalar(ScalarNum::U16(x)) => Ok(x as usize),
        MatValue::Scalar(ScalarNum::U8(x)) => Ok(x as usize),
        other => Err(MatError::Custom(format!(
            "{field} must be an unsigned integer, got {}",
            other.kind()
        ))),
    }
}

/// A complex sentinel's `real`/`imag` field, kept at the width it was
/// serialized at. `end` checks it against the sentinel's class.
fn expect_component(v: MatValue) -> Result<ScalarNum, MatError> {
    match v {
        MatValue::Scalar(s) => Ok(s),
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
            MatValue::Vec1D(NumVec::U8(vec![1, 2, 3])),
        );
    }

    #[test]
    fn scalars_of_two_classes_become_a_cell_array() {
        assert_eq!(
            to_value(&(1u8, 2u16), &Options::default()).unwrap(),
            MatValue::Cell(vec![
                MatValue::Scalar(ScalarNum::U8(1)),
                MatValue::Scalar(ScalarNum::U16(2)),
            ]),
        );
    }

    #[test]
    fn rows_of_one_class_and_length_become_a_matrix() {
        assert_eq!(
            to_value(&vec![vec![1u8, 2], vec![3, 4]], &Options::default()).unwrap(),
            MatValue::Matrix {
                rows: 2,
                cols: 2,
                vec: NumVec::U8(vec![1, 2, 3, 4]),
            },
        );
    }

    #[test]
    fn ragged_rows_become_a_cell_array() {
        assert_eq!(
            to_value(&vec![vec![1u8, 2], vec![3]], &Options::default()).unwrap(),
            MatValue::Cell(vec![
                MatValue::Vec1D(NumVec::U8(vec![1, 2])),
                MatValue::Vec1D(NumVec::U8(vec![3])),
            ]),
        );
    }

    #[test]
    fn a_scalar_after_a_row_spills_the_rows_into_the_cell_array() {
        assert_eq!(
            to_value(&(vec![1u8, 2], 3u8), &Options::default()).unwrap(),
            MatValue::Cell(vec![
                MatValue::Vec1D(NumVec::U8(vec![1, 2])),
                MatValue::Scalar(ScalarNum::U8(3)),
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
            MatValue::ComplexMatrix {
                rows: 2,
                cols: 1,
                pairs: ComplexVec::F64(vec![(1.0, 2.0), (3.0, 4.0)]),
            },
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
            MatValue::Cell(vec![
                MatValue::Scalar(ScalarNum::F64(1.0)),
                MatValue::EmptyStructArray,
            ]),
        );
    }
}
