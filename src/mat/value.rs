//! Intermediate value tree the reader builds out of a `.mat` file, and the
//! tagged scalar and array types it shares with the serializer.

use crate::mat::class::MatClass;
use crate::mat::error::MatError;
use crate::mat::transpose::transpose_pairs;

/// A scalar numeric value tagged by its Rust type.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ScalarNum {
    Bool(bool),
    F64(f64),
    F32(f32),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
}

impl ScalarNum {
    pub(crate) fn tag(&self) -> ScalarTag {
        match self {
            ScalarNum::Bool(_) => ScalarTag::Bool,
            ScalarNum::F64(_) => ScalarTag::F64,
            ScalarNum::F32(_) => ScalarTag::F32,
            ScalarNum::I64(_) => ScalarTag::I64,
            ScalarNum::I32(_) => ScalarTag::I32,
            ScalarNum::I16(_) => ScalarTag::I16,
            ScalarNum::I8(_) => ScalarTag::I8,
            ScalarNum::U64(_) => ScalarTag::U64,
            ScalarNum::U32(_) => ScalarTag::U32,
            ScalarNum::U16(_) => ScalarTag::U16,
            ScalarNum::U8(_) => ScalarTag::U8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarTag {
    Bool,
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

/// A typed 1-D array of a single primitive class.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NumVec {
    Bool(Vec<bool>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I16(Vec<i16>),
    I8(Vec<i8>),
    U64(Vec<u64>),
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

impl NumVec {
    pub(crate) fn len(&self) -> usize {
        match self {
            NumVec::Bool(v) => v.len(),
            NumVec::F64(v) => v.len(),
            NumVec::F32(v) => v.len(),
            NumVec::I64(v) => v.len(),
            NumVec::I32(v) => v.len(),
            NumVec::I16(v) => v.len(),
            NumVec::I8(v) => v.len(),
            NumVec::U64(v) => v.len(),
            NumVec::U32(v) => v.len(),
            NumVec::U16(v) => v.len(),
            NumVec::U8(v) => v.len(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the scalar at `index`, or `None` if out of bounds. Lets callers
    /// walk a numeric vector element-by-element without materializing a
    /// `Vec<MatValue>` of boxed scalars.
    pub(crate) fn get(&self, index: usize) -> Option<ScalarNum> {
        match self {
            NumVec::Bool(v) => v.get(index).copied().map(ScalarNum::Bool),
            NumVec::F64(v) => v.get(index).copied().map(ScalarNum::F64),
            NumVec::F32(v) => v.get(index).copied().map(ScalarNum::F32),
            NumVec::I64(v) => v.get(index).copied().map(ScalarNum::I64),
            NumVec::I32(v) => v.get(index).copied().map(ScalarNum::I32),
            NumVec::I16(v) => v.get(index).copied().map(ScalarNum::I16),
            NumVec::I8(v) => v.get(index).copied().map(ScalarNum::I8),
            NumVec::U64(v) => v.get(index).copied().map(ScalarNum::U64),
            NumVec::U32(v) => v.get(index).copied().map(ScalarNum::U32),
            NumVec::U16(v) => v.get(index).copied().map(ScalarNum::U16),
            NumVec::U8(v) => v.get(index).copied().map(ScalarNum::U8),
        }
    }

    /// Consume the vector as a stream of scalars. For a caller that is
    /// unpacking the whole thing, unlike [`get`](Self::get)'s cursor walk.
    pub(crate) fn into_scalars(self) -> Box<dyn Iterator<Item = ScalarNum>> {
        macro_rules! stream {
            ($v:expr, $ctor:path) => {
                Box::new($v.into_iter().map($ctor))
            };
        }
        match self {
            NumVec::Bool(v) => stream!(v, ScalarNum::Bool),
            NumVec::F64(v) => stream!(v, ScalarNum::F64),
            NumVec::F32(v) => stream!(v, ScalarNum::F32),
            NumVec::I64(v) => stream!(v, ScalarNum::I64),
            NumVec::I32(v) => stream!(v, ScalarNum::I32),
            NumVec::I16(v) => stream!(v, ScalarNum::I16),
            NumVec::I8(v) => stream!(v, ScalarNum::I8),
            NumVec::U64(v) => stream!(v, ScalarNum::U64),
            NumVec::U32(v) => stream!(v, ScalarNum::U32),
            NumVec::U16(v) => stream!(v, ScalarNum::U16),
            NumVec::U8(v) => stream!(v, ScalarNum::U8),
        }
    }

    pub(crate) fn tag(&self) -> ScalarTag {
        match self {
            NumVec::Bool(_) => ScalarTag::Bool,
            NumVec::F64(_) => ScalarTag::F64,
            NumVec::F32(_) => ScalarTag::F32,
            NumVec::I64(_) => ScalarTag::I64,
            NumVec::I32(_) => ScalarTag::I32,
            NumVec::I16(_) => ScalarTag::I16,
            NumVec::I8(_) => ScalarTag::I8,
            NumVec::U64(_) => ScalarTag::U64,
            NumVec::U32(_) => ScalarTag::U32,
            NumVec::U16(_) => ScalarTag::U16,
            NumVec::U8(_) => ScalarTag::U8,
        }
    }

    pub(crate) fn empty_with_tag(tag: ScalarTag) -> Self {
        match tag {
            ScalarTag::Bool => NumVec::Bool(Vec::new()),
            ScalarTag::F64 => NumVec::F64(Vec::new()),
            ScalarTag::F32 => NumVec::F32(Vec::new()),
            ScalarTag::I64 => NumVec::I64(Vec::new()),
            ScalarTag::I32 => NumVec::I32(Vec::new()),
            ScalarTag::I16 => NumVec::I16(Vec::new()),
            ScalarTag::I8 => NumVec::I8(Vec::new()),
            ScalarTag::U64 => NumVec::U64(Vec::new()),
            ScalarTag::U32 => NumVec::U32(Vec::new()),
            ScalarTag::U16 => NumVec::U16(Vec::new()),
            ScalarTag::U8 => NumVec::U8(Vec::new()),
        }
    }

    /// Like [`empty_with_tag`](Self::empty_with_tag) but reserves `cap` elements
    /// up front, so a known-size fill via [`push`](Self::push) /
    /// [`extend`](Self::extend) never reallocates.
    pub(crate) fn with_capacity_for_tag(tag: ScalarTag, cap: usize) -> Self {
        match tag {
            ScalarTag::Bool => NumVec::Bool(Vec::with_capacity(cap)),
            ScalarTag::F64 => NumVec::F64(Vec::with_capacity(cap)),
            ScalarTag::F32 => NumVec::F32(Vec::with_capacity(cap)),
            ScalarTag::I64 => NumVec::I64(Vec::with_capacity(cap)),
            ScalarTag::I32 => NumVec::I32(Vec::with_capacity(cap)),
            ScalarTag::I16 => NumVec::I16(Vec::with_capacity(cap)),
            ScalarTag::I8 => NumVec::I8(Vec::with_capacity(cap)),
            ScalarTag::U64 => NumVec::U64(Vec::with_capacity(cap)),
            ScalarTag::U32 => NumVec::U32(Vec::with_capacity(cap)),
            ScalarTag::U16 => NumVec::U16(Vec::with_capacity(cap)),
            ScalarTag::U8 => NumVec::U8(Vec::with_capacity(cap)),
        }
    }

    /// Push a scalar into this vec, requiring matching tags.
    pub(crate) fn push(&mut self, v: ScalarNum) -> Result<(), MatError> {
        match (self, v) {
            (NumVec::Bool(vec), ScalarNum::Bool(x)) => vec.push(x),
            (NumVec::F64(vec), ScalarNum::F64(x)) => vec.push(x),
            (NumVec::F32(vec), ScalarNum::F32(x)) => vec.push(x),
            (NumVec::I64(vec), ScalarNum::I64(x)) => vec.push(x),
            (NumVec::I32(vec), ScalarNum::I32(x)) => vec.push(x),
            (NumVec::I16(vec), ScalarNum::I16(x)) => vec.push(x),
            (NumVec::I8(vec), ScalarNum::I8(x)) => vec.push(x),
            (NumVec::U64(vec), ScalarNum::U64(x)) => vec.push(x),
            (NumVec::U32(vec), ScalarNum::U32(x)) => vec.push(x),
            (NumVec::U16(vec), ScalarNum::U16(x)) => vec.push(x),
            (NumVec::U8(vec), ScalarNum::U8(x)) => vec.push(x),
            _ => return Err(MatError::MixedSequenceElementTypes),
        }
        Ok(())
    }

    /// Append another vec of the same tag.
    pub(crate) fn extend(&mut self, other: NumVec) -> Result<(), MatError> {
        match (self, other) {
            (NumVec::Bool(a), NumVec::Bool(b)) => a.extend(b),
            (NumVec::F64(a), NumVec::F64(b)) => a.extend(b),
            (NumVec::F32(a), NumVec::F32(b)) => a.extend(b),
            (NumVec::I64(a), NumVec::I64(b)) => a.extend(b),
            (NumVec::I32(a), NumVec::I32(b)) => a.extend(b),
            (NumVec::I16(a), NumVec::I16(b)) => a.extend(b),
            (NumVec::I8(a), NumVec::I8(b)) => a.extend(b),
            (NumVec::U64(a), NumVec::U64(b)) => a.extend(b),
            (NumVec::U32(a), NumVec::U32(b)) => a.extend(b),
            (NumVec::U16(a), NumVec::U16(b)) => a.extend(b),
            (NumVec::U8(a), NumVec::U8(b)) => a.extend(b),
            _ => return Err(MatError::MixedSequenceElementTypes),
        }
        Ok(())
    }
}

/// The complex counterparts of [`ScalarNum`] / [`NumVec`], generated from one
/// list so a component width is added in a single place.
///
/// MATLAB stores a complex array as a `{real, imag}` compound whose
/// `MATLAB_class` names the *component* class, so the set of component classes
/// is the numeric classes: every one that [`NumVec`] carries except `logical`,
/// which has no complex form. Keeping the component class in a tag rather than
/// in the variant name is what lets an empty complex array still know what it
/// is — the pairs are gone, the tag is not.
macro_rules! complex_kinds {
    ($($variant:ident => $ty:ty, $class:ident),* $(,)?) => {
        /// The component class of a complex value.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum ComplexTag {
            $($variant,)*
        }

        impl ComplexTag {
            /// The MATLAB class a complex array of this component reports in
            /// its `MATLAB_class` attribute.
            pub(crate) fn class(self) -> MatClass {
                match self {
                    $(ComplexTag::$variant => MatClass::$class,)*
                }
            }

            /// The component tag for `class`, or `None` for a class that has no
            /// complex form (`char`, `logical`, `struct`, `cell`).
            pub(crate) fn from_class(class: MatClass) -> Option<Self> {
                match class {
                    $(MatClass::$class => Some(ComplexTag::$variant),)*
                    _ => None,
                }
            }
        }

        /// A complex scalar tagged by its component class.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub(crate) enum ComplexNum {
            $($variant($ty, $ty),)*
        }

        impl ComplexNum {
            pub(crate) fn tag(&self) -> ComplexTag {
                match self {
                    $(ComplexNum::$variant(..) => ComplexTag::$variant,)*
                }
            }

            /// The real and imaginary parts as tagged scalars, so a caller can
            /// hand each to serde without naming the component type.
            pub(crate) fn components(self) -> (ScalarNum, ScalarNum) {
                match self {
                    $(ComplexNum::$variant(re, im) => {
                        (ScalarNum::$variant(re), ScalarNum::$variant(im))
                    })*
                }
            }

            /// Rebuild a complex value from its two components, requiring both
            /// to carry `tag`'s class.
            ///
            /// Returns `None` on any mismatch rather than converting: a
            /// component is stored at the width the caller asked for, so a
            /// coercion here would be a silent change of what the file says it
            /// holds.
            pub(crate) fn from_components(
                tag: ComplexTag,
                re: ScalarNum,
                im: ScalarNum,
            ) -> Option<Self> {
                match (tag, re, im) {
                    $((ComplexTag::$variant, ScalarNum::$variant(re), ScalarNum::$variant(im)) => {
                        Some(ComplexNum::$variant(re, im))
                    })*
                    _ => None,
                }
            }

            /// Promote a real scalar to a complex value with a zero imaginary
            /// part, when the scalar's class is `tag`'s. `None` for a class
            /// mismatch, and for `logical`, which has no complex form.
            pub(crate) fn from_real(tag: ComplexTag, re: ScalarNum) -> Option<Self> {
                match (tag, re) {
                    $((ComplexTag::$variant, ScalarNum::$variant(re)) => {
                        Some(ComplexNum::$variant(re, <$ty>::default()))
                    })*
                    _ => None,
                }
            }
        }

        /// A typed 1-D array of complex pairs, laid out `[(re, im), ...]`.
        #[derive(Debug, Clone, PartialEq)]
        pub(crate) enum ComplexVec {
            $($variant(Vec<($ty, $ty)>),)*
        }

        impl ComplexVec {
            pub(crate) fn len(&self) -> usize {
                match self {
                    $(ComplexVec::$variant(v) => v.len(),)*
                }
            }

            pub(crate) fn tag(&self) -> ComplexTag {
                match self {
                    $(ComplexVec::$variant(_) => ComplexTag::$variant,)*
                }
            }

            pub(crate) fn empty_with_tag(tag: ComplexTag) -> Self {
                match tag {
                    $(ComplexTag::$variant => ComplexVec::$variant(Vec::new()),)*
                }
            }

            pub(crate) fn with_capacity_for_tag(tag: ComplexTag, cap: usize) -> Self {
                match tag {
                    $(ComplexTag::$variant => ComplexVec::$variant(Vec::with_capacity(cap)),)*
                }
            }

            pub(crate) fn from_single(n: ComplexNum) -> Self {
                match n {
                    $(ComplexNum::$variant(re, im) => ComplexVec::$variant(vec![(re, im)]),)*
                }
            }

            /// Rebuild a vector of `tag`'s class from interleaved `(re, im)`
            /// components in **native**-endian order.
            ///
            /// Native, not little-endian, because the bytes are a live Rust
            /// slice the caller handed over whole (see `mat::complex`'s array
            /// helpers) rather than anything read from a file. The writer
            /// re-encodes to little-endian on the way out, so a big-endian
            /// host produces the same file a little-endian one does.
            pub(crate) fn from_native_bytes(
                tag: ComplexTag,
                bytes: &[u8],
            ) -> Result<Self, MatError> {
                match tag {
                    $(ComplexTag::$variant => {
                        const W: usize = core::mem::size_of::<$ty>();
                        if bytes.len() % (2 * W) != 0 {
                            return Err(MatError::Custom(format!(
                                "complex {} array payload of {} bytes is not a whole \
                                 number of {}-byte pairs",
                                ComplexTag::$variant.class().as_str(),
                                bytes.len(),
                                2 * W,
                            )));
                        }
                        let v: Vec<($ty, $ty)> = bytes
                            .chunks_exact(2 * W)
                            .map(|pair| {
                                let (re, im) = pair.split_at(W);
                                (
                                    <$ty>::from_ne_bytes(re.try_into().expect("half of a pair")),
                                    <$ty>::from_ne_bytes(im.try_into().expect("half of a pair")),
                                )
                            })
                            .collect();
                        Ok(ComplexVec::$variant(v))
                    })*
                }
            }

            /// Consume the vector as a stream of pairs. For a caller that is
            /// unpacking the whole thing, unlike [`get`](Self::get)'s cursor
            /// walk.
            pub(crate) fn into_pairs(self) -> Box<dyn Iterator<Item = ComplexNum>> {
                match self {
                    $(ComplexVec::$variant(v) => {
                        Box::new(v.into_iter().map(|(re, im)| ComplexNum::$variant(re, im)))
                    })*
                }
            }

            /// The pair at `index`, or `None` if out of bounds.
            pub(crate) fn get(&self, index: usize) -> Option<ComplexNum> {
                match self {
                    $(ComplexVec::$variant(v) => {
                        v.get(index).map(|&(re, im)| ComplexNum::$variant(re, im))
                    })*
                }
            }

            /// Push a pair, requiring a matching tag.
            pub(crate) fn push(&mut self, n: ComplexNum) -> Result<(), MatError> {
                match (self, n) {
                    $((ComplexVec::$variant(v), ComplexNum::$variant(re, im)) => v.push((re, im)),)*
                    _ => return Err(MatError::MixedSequenceElementTypes),
                }
                Ok(())
            }

            /// Append another vec of the same tag.
            pub(crate) fn extend(&mut self, other: ComplexVec) -> Result<(), MatError> {
                match (self, other) {
                    $((ComplexVec::$variant(a), ComplexVec::$variant(b)) => a.extend(b),)*
                    _ => return Err(MatError::MixedSequenceElementTypes),
                }
                Ok(())
            }

            /// Split the first `n` pairs off the front, leaving the rest.
            pub(crate) fn split_off_front(&mut self, n: usize) -> ComplexVec {
                match self {
                    $(ComplexVec::$variant(v) => {
                        let tail = v.split_off(n);
                        ComplexVec::$variant(core::mem::replace(v, tail))
                    })*
                }
            }

            /// Reinterpret `rows × cols` row-major pairs as column-major, the
            /// order MATLAB stores a matrix in.
            pub(crate) fn transposed(self, rows: usize, cols: usize) -> Self {
                match self {
                    $(ComplexVec::$variant(v) => {
                        ComplexVec::$variant(transpose_pairs(rows, cols, &v))
                    })*
                }
            }
        }
    };
}

complex_kinds! {
    F64 => f64, Double,
    F32 => f32, Single,
    I64 => i64, Int64,
    I32 => i32, Int32,
    I16 => i16, Int16,
    I8 => i8, Int8,
    U64 => u64, UInt64,
    U32 => u32, UInt32,
    U16 => u16, UInt16,
    U8 => u8, UInt8,
}

/// A node of the tree the reader builds out of a file.
///
/// The serializer has a value tree of its own, [`Value`]. A variant here
/// describes what a `.mat` file holds, and the reader alone produces it.
///
/// [`Value`]: crate::mat::ser::value::Value
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MatValue {
    /// Numeric / logical scalar.
    Scalar(ScalarNum),
    /// 1-D numeric array → stored as `[1, N]`.
    Vec1D(NumVec),
    /// 2-D numeric array in row-major order → stored column-major with shape
    /// `[cols, rows]`.
    Matrix {
        rows: usize,
        cols: usize,
        vec: NumVec,
    },
    /// UTF-16 `char` string → stored as `uint16 [1, N]`.
    String(String),
    /// Complex scalar → stored as a `[1, 1]` `{real, imag}` compound.
    ComplexScalar(ComplexNum),
    /// Complex 1-D array → stored as `[1, N]`.
    ComplexVec1D(ComplexVec),
    /// Complex 2-D matrix in row-major order → stored column-major with shape
    /// `[cols, rows]`, mirroring [`MatValue::Matrix`].
    ComplexMatrix {
        rows: usize,
        cols: usize,
        pairs: ComplexVec,
    },
    /// Ordered, named fields. Serialized as a MATLAB struct group.
    Struct(Vec<(String, MatValue)>),
    /// A cell array (`MATLAB_class = "cell"`), whose elements live under
    /// `#refs#` and whose dataset stores object references in element order.
    ///
    /// The node carries no shape: the deserializer presents a cell array of any
    /// shape as a 1-D sequence, in storage order.
    Cell(Vec<MatValue>),
    /// Empty struct array placeholder for `None` inside a sequence. Renders
    /// as MATLAB's `struct([])` (a `[0, 0]` empty marker with
    /// `MATLAB_class="struct"` and `MATLAB_empty=1`).
    EmptyStructArray,
    /// A MATLAB struct *array* (`1×N` / `N×1` / `M×N` struct with fields).
    ///
    /// On disk MATLAB stores a struct array as a `MATLAB_class="struct"` group
    /// whose every field is a dataset of object references — one reference per
    /// array element — i.e. a struct-of-arrays. The reader transposes that into
    /// this array-of-structs: `elements` lists each element's fields in
    /// row-major order, and `rows`/`cols` carry the array shape so the
    /// deserializer mirrors [`MatValue::Matrix`] (a `1×N`/`N×1` array flattens
    /// to a sequence of structs → `Vec<T>`; a true `M×N` array yields a
    /// sequence of rows → `Vec<Vec<T>>`).
    ///
    /// A `Vec<Struct>` goes out as a cell array (see [`MatValue::Cell`]), so
    /// this layout comes from a file alone.
    StructArray {
        rows: usize,
        cols: usize,
        elements: Vec<Vec<(String, MatValue)>>,
    },
    /// A decoded MATLAB MCOS opaque object (`MATLAB_object_decode = 3`).
    ///
    /// Aside from the modern `string` class (which lowers to [`MatValue::String`]
    /// / [`MatValue::Cell`]), MATLAB stores `datetime`, `duration`,
    /// `categorical`, `table`, `containers.Map`, `dictionary`, user `classdef`
    /// instances, … as opaque objects in the hidden `#subsystem#/MCOS` store.
    /// This variant carries the MATLAB class name and the object's resolved
    /// properties in declaration order:
    ///
    /// - For a class with a dedicated decoder (`datetime`, `duration`,
    ///   `categorical`) `fields` holds the decoded logical components (e.g.
    ///   datetime's `millis_utc` / `sub_ms`), which deserialize into the
    ///   matching public type ([`MatDatetime`](crate::mat::MatDatetime), …) or
    ///   any struct with the same field names.
    /// - For every other opaque class `fields` holds the raw property values,
    ///   so the object is still losslessly readable as a struct rather than
    ///   failing the whole file.
    ///
    /// Of the MCOS classes, this crate writes `string` alone.
    Opaque {
        /// The MATLAB class name (`"datetime"`, `"categorical"`, `"table"`, …).
        class_name: String,
        /// Resolved properties in declaration order.
        fields: Vec<(String, MatValue)>,
    },
}

impl MatValue {
    /// Return a short human-readable description for error messages.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            MatValue::Scalar(_) => "scalar",
            MatValue::Vec1D(_) => "1-D vector",
            MatValue::Matrix { .. } => "2-D matrix",
            MatValue::String(_) => "string",
            MatValue::ComplexScalar(_) => "complex scalar",
            MatValue::ComplexVec1D(_) => "complex vector",
            MatValue::ComplexMatrix { .. } => "complex matrix",
            MatValue::Struct(_) => "struct",
            MatValue::Cell(_) => "cell array",
            MatValue::EmptyStructArray => "empty struct array",
            MatValue::StructArray { .. } => "struct array",
            MatValue::Opaque { .. } => "opaque object",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Native rather than little-endian is the whole reason a big-endian host
    /// writes the same file: the components come out of live memory, and the
    /// writer swaps them on the way to disk.
    ///
    /// Written with `to_ne_bytes` on both sides, so it holds either way — but
    /// that also means it cannot catch `from_ne_bytes` being changed to
    /// `from_le_bytes`, and no CI target is big-endian. Reason about this one
    /// rather than trusting it to fail.
    #[test]
    fn native_bytes_decode_at_the_host_order() {
        let pair = [1i16.to_ne_bytes(), (-2i16).to_ne_bytes()].concat();
        assert_eq!(
            ComplexVec::from_native_bytes(ComplexTag::I16, &pair).unwrap(),
            ComplexVec::I16(vec![(1, -2)]),
        );
    }

    /// A payload that does not divide into whole pairs would otherwise lose
    /// its tail silently, since `chunks_exact` ignores a short remainder.
    #[test]
    fn a_partial_pair_is_refused_rather_than_truncated() {
        let err = ComplexVec::from_native_bytes(ComplexTag::I16, &[0, 1, 2, 3, 4]).unwrap_err();
        assert!(
            format!("{err}").contains("whole number of 4-byte pairs"),
            "unexpected message: {err}"
        );
    }

    /// The tag is the only thing an empty payload carries, and losing it is
    /// what makes an empty array write as an untyped `double`.
    #[test]
    fn an_empty_payload_keeps_its_class() {
        assert_eq!(
            ComplexVec::from_native_bytes(ComplexTag::U8, &[]).unwrap(),
            ComplexVec::U8(Vec::new()),
        );
    }
}
