//! Writes a serializer [`Value`] tree as HDF5 bytes through the [`MatBuilder`]
//! mid-level API.
//!
//! The entry points that take [`Options`] come here, and [`super::emit`] serves
//! the ones that do not.

use super::value::{Leaf, Value};
use crate::mat::builder::{CellWriter, MatBuilder, StructWriter};
use crate::mat::error::MatError;
use crate::mat::options::{Options, StringClass};
use crate::mat::transpose::transpose_scalars;
use crate::mat::value::{ComplexVec, NumVec, ScalarNum};

/// The complex dispatchers for the three write scopes, from one list.
///
/// Each expands to a `match` over every component class, which is where a
/// class becomes a concrete Rust type; a class added to [`ComplexVec`] fails to
/// compile here until it is listed, the same compile-time check `ser::emit`
/// relies on.
macro_rules! complex_dispatchers {
    ($($variant:ident => $write:ident, $push:ident),* $(,)?) => {
        fn write_complex_at_builder(
            mb: &mut MatBuilder,
            name: &str,
            dims: &[usize],
            pairs: ComplexVec,
        ) -> Result<(), MatError> {
            match pairs {
                $(ComplexVec::$variant(v) => mb.$write(name, dims, &v).map(|_| ()),)*
            }
        }

        fn write_complex_at_struct(
            sw: &mut StructWriter,
            name: &str,
            dims: &[usize],
            pairs: ComplexVec,
        ) -> Result<(), MatError> {
            match pairs {
                $(ComplexVec::$variant(v) => sw.$write(name, dims, &v).map(|_| ()),)*
            }
        }

        fn push_complex(
            cw: &mut CellWriter,
            dims: &[usize],
            pairs: ComplexVec,
        ) -> Result<(), MatError> {
            match pairs {
                $(ComplexVec::$variant(v) => cw.$push(dims, &v).map(|_| ()),)*
            }
        }
    };
}

complex_dispatchers! {
    F64 => write_complex_f64, push_complex_f64,
    F32 => write_complex_f32, push_complex_f32,
    I64 => write_complex_i64, push_complex_i64,
    I32 => write_complex_i32, push_complex_i32,
    I16 => write_complex_i16, push_complex_i16,
    I8 => write_complex_i8, push_complex_i8,
    U64 => write_complex_u64, push_complex_u64,
    U32 => write_complex_u32, push_complex_u32,
    U16 => write_complex_u16, push_complex_u16,
    U8 => write_complex_u8, push_complex_u8,
}

/// Walk top-level fields and emit through a `MatBuilder`.
pub(crate) fn emit_file_with_options(
    fields: Vec<(String, Value)>,
    options: &Options,
) -> Result<Vec<u8>, MatError> {
    build_with_options(fields, options)?.finish()
}

/// Same file as [`emit_file_with_options`], streamed to `w` instead of returned.
pub(crate) fn emit_file_with_options_to<W: std::io::Write>(
    fields: Vec<(String, Value)>,
    options: &Options,
    w: W,
) -> Result<(), MatError> {
    build_with_options(fields, options)?.finish_to(w)
}

/// Stage every field into a `MatBuilder`, ready to be finished either way.
fn build_with_options(
    fields: Vec<(String, Value)>,
    options: &Options,
) -> Result<MatBuilder, MatError> {
    let mut mb = MatBuilder::new(options.clone());
    for (name, value) in fields {
        if matches!(value, Value::Omit) {
            continue;
        }
        emit_at_root(&mut mb, &name, value)?;
    }
    Ok(mb)
}

fn emit_at_root(mb: &mut MatBuilder, name: &str, value: Value) -> Result<(), MatError> {
    match value {
        Value::Cell(elements) => {
            // A cell built from a sequence is a 1-D value, so its shape is the
            // one `dims::vector_dims` gives every other 1-D value: oriented by
            // `one_dimensional_mode`, and `0x0` when empty.
            let dims = mb.vector_dims(elements.len());
            mb.cell(name, &dims, |cw| emit_cell_elements(cw, elements))
                .map(|_| ())
        }
        Value::Leaf(leaf) => emit_leaf_at_builder(mb, name, leaf),
        Value::Omit => Ok(()),
        Value::Struct(fields) => mb
            .struct_(name, |sw| emit_struct_fields(sw, fields))
            .map(|_| ()),
    }
}

fn emit_struct_fields(sw: &mut StructWriter, fields: Vec<(String, Value)>) -> Result<(), MatError> {
    for (name, value) in fields {
        if matches!(value, Value::Omit) {
            continue;
        }
        emit_at_struct(sw, &name, value)?;
    }
    Ok(())
}

fn emit_at_struct(sw: &mut StructWriter, name: &str, value: Value) -> Result<(), MatError> {
    match value {
        Value::Cell(elements) => {
            let dims = sw.vector_dims(elements.len());
            sw.cell(name, &dims, |cw| emit_cell_elements(cw, elements))
                .map(|_| ())
        }
        Value::Leaf(leaf) => emit_leaf_at_struct(sw, name, leaf),
        Value::Omit => Ok(()),
        Value::Struct(fields) => sw
            .struct_(name, |inner| emit_struct_fields(inner, fields))
            .map(|_| ()),
    }
}

fn emit_cell_elements(cw: &mut CellWriter, elements: Vec<Value>) -> Result<(), MatError> {
    for value in elements {
        emit_cell_element(cw, value)?;
    }
    Ok(())
}

fn emit_cell_element(cw: &mut CellWriter, value: Value) -> Result<(), MatError> {
    match value {
        Value::Cell(elements) => {
            let dims = cw.vector_dims(elements.len());
            cw.push_cell(&dims, |inner| emit_cell_elements(inner, elements))
                .map(|_| ())
        }
        Value::Leaf(leaf) => emit_leaf_at_cell(cw, leaf),
        Value::Omit => cw.push_empty_struct_array().map(|_| ()),
        Value::Struct(fields) => cw
            .push_struct(|sw| emit_struct_fields(sw, fields))
            .map(|_| ()),
    }
}

fn emit_leaf_at_cell(cw: &mut CellWriter, leaf: Leaf) -> Result<(), MatError> {
    match leaf {
        Leaf::ComplexMatrix { rows, cols, pairs } => {
            let col_major = pairs.transposed(rows, cols);
            push_complex(cw, &[rows, cols], col_major)
        }
        Leaf::ComplexScalar(n) => push_complex(cw, &[1, 1], ComplexVec::from_single(n)),
        Leaf::ComplexVec1D(pairs) => {
            let dims = cw.vector_dims(pairs.len());
            push_complex(cw, &dims, pairs)
        }
        Leaf::EmptyStructArray => cw.push_empty_struct_array().map(|_| ()),
        Leaf::Matrix { rows, cols, vec } => emit_cell_matrix(cw, rows, cols, vec),
        Leaf::Scalar(n) => emit_cell_scalar(cw, n),
        Leaf::String(s) => cw.push_char(&s).map(|_| ()),
        Leaf::Vec1D(v) => emit_cell_vec(cw, v),
    }
}

fn emit_cell_scalar(cw: &mut CellWriter, scalar: ScalarNum) -> Result<(), MatError> {
    match scalar {
        ScalarNum::Bool(b) => {
            cw.push_scalar_logical(b)?;
        }
        ScalarNum::F64(x) => {
            cw.push_scalar_f64(x)?;
        }
        ScalarNum::F32(x) => {
            cw.push_scalar_f32(x)?;
        }
        ScalarNum::I64(x) => {
            cw.push_scalar_i64(x)?;
        }
        ScalarNum::I32(x) => {
            cw.push_scalar_i32(x)?;
        }
        ScalarNum::I16(x) => {
            cw.push_scalar_i16(x)?;
        }
        ScalarNum::I8(x) => {
            cw.push_scalar_i8(x)?;
        }
        ScalarNum::U64(x) => {
            cw.push_scalar_u64(x)?;
        }
        ScalarNum::U32(x) => {
            cw.push_scalar_u32(x)?;
        }
        ScalarNum::U16(x) => {
            cw.push_scalar_u16(x)?;
        }
        ScalarNum::U8(x) => {
            cw.push_scalar_u8(x)?;
        }
    }
    Ok(())
}

fn emit_cell_vec(cw: &mut CellWriter, v: NumVec) -> Result<(), MatError> {
    // 1-D cell-element vectors honor the configured OneDimensionalMode
    // (default ColumnVector → MATLAB shape `[N, 1]`).
    let dims = cw.vector_dims(v.len());
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            cw.push_logical(&dims, &bytes)?;
        }
        NumVec::F64(vec) => {
            cw.push_f64(&dims, &vec)?;
        }
        NumVec::F32(vec) => {
            cw.push_f32(&dims, &vec)?;
        }
        NumVec::I64(vec) => {
            cw.push_i64(&dims, &vec)?;
        }
        NumVec::I32(vec) => {
            cw.push_i32(&dims, &vec)?;
        }
        NumVec::I16(vec) => {
            cw.push_i16(&dims, &vec)?;
        }
        NumVec::I8(vec) => {
            cw.push_i8(&dims, &vec)?;
        }
        NumVec::U64(vec) => {
            cw.push_u64(&dims, &vec)?;
        }
        NumVec::U32(vec) => {
            cw.push_u32(&dims, &vec)?;
        }
        NumVec::U16(vec) => {
            cw.push_u16(&dims, &vec)?;
        }
        NumVec::U8(vec) => {
            cw.push_u8(&dims, &vec)?;
        }
    }
    Ok(())
}

fn emit_cell_matrix(
    cw: &mut CellWriter,
    rows: usize,
    cols: usize,
    v: NumVec,
) -> Result<(), MatError> {
    let dims = [rows, cols];
    match v {
        NumVec::Bool(vec) => {
            let col_major = transpose_scalars(rows, cols, &vec);
            let bytes: Vec<u8> = col_major.into_iter().map(u8::from).collect();
            cw.push_logical(&dims, &bytes)?;
        }
        NumVec::F64(vec) => {
            cw.push_f64(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::F32(vec) => {
            cw.push_f32(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I64(vec) => {
            cw.push_i64(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I32(vec) => {
            cw.push_i32(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I16(vec) => {
            cw.push_i16(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::I8(vec) => {
            cw.push_i8(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U64(vec) => {
            cw.push_u64(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U32(vec) => {
            cw.push_u32(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U16(vec) => {
            cw.push_u16(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
        NumVec::U8(vec) => {
            cw.push_u8(&dims, &transpose_scalars(rows, cols, &vec))?;
        }
    }
    Ok(())
}

fn emit_leaf_at_builder(mb: &mut MatBuilder, name: &str, leaf: Leaf) -> Result<(), MatError> {
    match leaf {
        Leaf::ComplexMatrix { rows, cols, pairs } => {
            let col_major = pairs.transposed(rows, cols);
            write_complex_at_builder(mb, name, &[rows, cols], col_major)
        }
        Leaf::ComplexScalar(n) => {
            write_complex_at_builder(mb, name, &[1, 1], ComplexVec::from_single(n))
        }
        Leaf::ComplexVec1D(pairs) => {
            let dims = mb.vector_dims(pairs.len());
            write_complex_at_builder(mb, name, &dims, pairs)
        }
        Leaf::EmptyStructArray => mb.write_empty_struct_array(name).map(|_| ()),
        Leaf::Matrix { rows, cols, vec } => emit_matrix_at_builder(mb, name, rows, cols, vec),
        Leaf::Scalar(n) => emit_scalar_at_builder(mb, name, n),
        Leaf::String(s) => emit_string_at_builder(mb, name, &s),
        Leaf::Vec1D(v) => emit_vec_at_builder(mb, name, v),
    }
}

fn emit_leaf_at_struct(sw: &mut StructWriter, name: &str, leaf: Leaf) -> Result<(), MatError> {
    match leaf {
        Leaf::ComplexMatrix { rows, cols, pairs } => {
            let col_major = pairs.transposed(rows, cols);
            write_complex_at_struct(sw, name, &[rows, cols], col_major)
        }
        Leaf::ComplexScalar(n) => {
            write_complex_at_struct(sw, name, &[1, 1], ComplexVec::from_single(n))
        }
        Leaf::ComplexVec1D(pairs) => {
            let dims = sw.vector_dims(pairs.len());
            write_complex_at_struct(sw, name, &dims, pairs)
        }
        Leaf::EmptyStructArray => sw.write_empty_struct_array(name).map(|_| ()),
        Leaf::Matrix { rows, cols, vec } => emit_matrix_at_struct(sw, name, rows, cols, vec),
        Leaf::Scalar(n) => emit_scalar_at_struct(sw, name, n),
        Leaf::String(s) => emit_string_at_struct(sw, name, &s),
        Leaf::Vec1D(v) => emit_vec_at_struct(sw, name, v),
    }
}

fn emit_scalar_at_builder(
    mb: &mut MatBuilder,
    name: &str,
    scalar: ScalarNum,
) -> Result<(), MatError> {
    match scalar {
        ScalarNum::Bool(b) => mb.write_scalar_logical(name, b),
        ScalarNum::F64(x) => mb.write_scalar_f64(name, x),
        ScalarNum::F32(x) => mb.write_scalar_f32(name, x),
        ScalarNum::I64(x) => mb.write_scalar_i64(name, x),
        ScalarNum::I32(x) => mb.write_scalar_i32(name, x),
        ScalarNum::I16(x) => mb.write_scalar_i16(name, x),
        ScalarNum::I8(x) => mb.write_scalar_i8(name, x),
        ScalarNum::U64(x) => mb.write_scalar_u64(name, x),
        ScalarNum::U32(x) => mb.write_scalar_u32(name, x),
        ScalarNum::U16(x) => mb.write_scalar_u16(name, x),
        ScalarNum::U8(x) => mb.write_scalar_u8(name, x),
    }
    .map(|_| ())
}

fn emit_scalar_at_struct(
    sw: &mut StructWriter,
    name: &str,
    scalar: ScalarNum,
) -> Result<(), MatError> {
    match scalar {
        ScalarNum::Bool(b) => sw.write_scalar_logical(name, b),
        ScalarNum::F64(x) => sw.write_scalar_f64(name, x),
        ScalarNum::F32(x) => sw.write_scalar_f32(name, x),
        ScalarNum::I64(x) => sw.write_scalar_i64(name, x),
        ScalarNum::I32(x) => sw.write_scalar_i32(name, x),
        ScalarNum::I16(x) => sw.write_scalar_i16(name, x),
        ScalarNum::I8(x) => sw.write_scalar_i8(name, x),
        ScalarNum::U64(x) => sw.write_scalar_u64(name, x),
        ScalarNum::U32(x) => sw.write_scalar_u32(name, x),
        ScalarNum::U16(x) => sw.write_scalar_u16(name, x),
        ScalarNum::U8(x) => sw.write_scalar_u8(name, x),
    }
    .map(|_| ())
}

fn emit_vec_at_builder(mb: &mut MatBuilder, name: &str, v: NumVec) -> Result<(), MatError> {
    let dims = mb.vector_dims(v.len());
    if v.is_empty() {
        return mb.write_empty(name, v.tag().class(), &dims).map(|_| ());
    }
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            mb.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => mb.write_f64(name, &dims, &vec),
        NumVec::F32(vec) => mb.write_f32(name, &dims, &vec),
        NumVec::I64(vec) => mb.write_i64(name, &dims, &vec),
        NumVec::I32(vec) => mb.write_i32(name, &dims, &vec),
        NumVec::I16(vec) => mb.write_i16(name, &dims, &vec),
        NumVec::I8(vec) => mb.write_i8(name, &dims, &vec),
        NumVec::U64(vec) => mb.write_u64(name, &dims, &vec),
        NumVec::U32(vec) => mb.write_u32(name, &dims, &vec),
        NumVec::U16(vec) => mb.write_u16(name, &dims, &vec),
        NumVec::U8(vec) => mb.write_u8(name, &dims, &vec),
    }
    .map(|_| ())
}

fn emit_vec_at_struct(sw: &mut StructWriter, name: &str, v: NumVec) -> Result<(), MatError> {
    let dims = sw.vector_dims(v.len());
    if v.is_empty() {
        return sw.write_empty(name, v.tag().class(), &dims).map(|_| ());
    }
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            sw.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => sw.write_f64(name, &dims, &vec),
        NumVec::F32(vec) => sw.write_f32(name, &dims, &vec),
        NumVec::I64(vec) => sw.write_i64(name, &dims, &vec),
        NumVec::I32(vec) => sw.write_i32(name, &dims, &vec),
        NumVec::I16(vec) => sw.write_i16(name, &dims, &vec),
        NumVec::I8(vec) => sw.write_i8(name, &dims, &vec),
        NumVec::U64(vec) => sw.write_u64(name, &dims, &vec),
        NumVec::U32(vec) => sw.write_u32(name, &dims, &vec),
        NumVec::U16(vec) => sw.write_u16(name, &dims, &vec),
        NumVec::U8(vec) => sw.write_u8(name, &dims, &vec),
    }
    .map(|_| ())
}

fn emit_matrix_at_builder(
    mb: &mut MatBuilder,
    name: &str,
    rows: usize,
    cols: usize,
    v: NumVec,
) -> Result<(), MatError> {
    let dims = [rows, cols];
    match v {
        NumVec::Bool(vec) => {
            let col_major = transpose_scalars(rows, cols, &vec);
            let bytes: Vec<u8> = col_major.into_iter().map(u8::from).collect();
            mb.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => mb.write_f64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::F32(vec) => mb.write_f32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I64(vec) => mb.write_i64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I32(vec) => mb.write_i32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I16(vec) => mb.write_i16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I8(vec) => mb.write_i8(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U64(vec) => mb.write_u64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U32(vec) => mb.write_u32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U16(vec) => mb.write_u16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U8(vec) => mb.write_u8(name, &dims, &transpose_scalars(rows, cols, &vec)),
    }
    .map(|_| ())
}

fn emit_matrix_at_struct(
    sw: &mut StructWriter,
    name: &str,
    rows: usize,
    cols: usize,
    v: NumVec,
) -> Result<(), MatError> {
    let dims = [rows, cols];
    match v {
        NumVec::Bool(vec) => {
            let col_major = transpose_scalars(rows, cols, &vec);
            let bytes: Vec<u8> = col_major.into_iter().map(u8::from).collect();
            sw.write_logical(name, &dims, &bytes)
        }
        NumVec::F64(vec) => sw.write_f64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::F32(vec) => sw.write_f32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I64(vec) => sw.write_i64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I32(vec) => sw.write_i32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I16(vec) => sw.write_i16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::I8(vec) => sw.write_i8(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U64(vec) => sw.write_u64(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U32(vec) => sw.write_u32(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U16(vec) => sw.write_u16(name, &dims, &transpose_scalars(rows, cols, &vec)),
        NumVec::U8(vec) => sw.write_u8(name, &dims, &transpose_scalars(rows, cols, &vec)),
    }
    .map(|_| ())
}

fn emit_string_at_builder(mb: &mut MatBuilder, name: &str, s: &str) -> Result<(), MatError> {
    match mb.options().string_class {
        StringClass::Char => mb.write_char(name, s).map(|_| ()),
        StringClass::String => mb
            .write_string_object(name, &[s.to_owned()], &[1, 1])
            .map(|_| ()),
    }
}

fn emit_string_at_struct(sw: &mut StructWriter, name: &str, s: &str) -> Result<(), MatError> {
    match sw.string_class() {
        StringClass::Char => sw.write_char(name, s).map(|_| ()),
        StringClass::String => sw
            .write_string_object(name, &[s.to_owned()], &[1, 1])
            .map(|_| ()),
    }
}
