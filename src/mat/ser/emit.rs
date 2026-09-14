//! Writes a serializer [`Value`] tree as an HDF5 file under MATLAB v7.3
//! conventions.
//!
//! The entry points that take no [`Options`] come here, and
//! [`super::emit_with_builder`] serves the ones that do.

use std::collections::VecDeque;

use super::value::{Leaf, Value};
use crate::file_writer::AttrValue;
use crate::mat::builder::{RefsH5Path, refs_h5path};
use crate::mat::class::MatClass;
use crate::mat::dims;
use crate::mat::error::MatError;
use crate::mat::options::Options;
use crate::mat::transpose::transpose_scalars;
use crate::mat::userblock::{self, USERBLOCK_SIZE};
use crate::mat::utf16;
use crate::mat::value::{ComplexVec, NumVec, ScalarNum};
use crate::type_builders::{DatasetBuilder, FinishedGroup, GroupBuilder};
use crate::writer::FileBuilder;

/// Hidden MATLAB conventional group that holds the targets of object
/// references. Cell-array elements live here, addressed by absolute path.
const REFS_GROUP: &str = "#refs#";

/// Allocator + queue for cell-array element interning. Cells stash each
/// element under a fresh `ref_{id:016x}` name and the emitter drains the
/// queue at file-build time to materialize the `#refs#` group. Draining is
/// itself emit-aware: a Cell whose elements include nested cells will push
/// new entries onto the queue while it is being drained.
struct RefsAccumulator {
    next_id: u64,
    pending: VecDeque<(String, Value)>,
}

impl RefsAccumulator {
    fn new() -> Self {
        Self {
            next_id: 0,
            pending: VecDeque::new(),
        }
    }

    /// Reserve a fresh name and queue `value` for later emission. Returns the
    /// absolute path the parent dataset's reference should resolve to.
    fn intern(&mut self, value: Value) -> String {
        let name = format!("ref_{:016x}", self.next_id);
        self.next_id += 1;
        let path = format!("{REFS_GROUP}/{name}");
        self.pending.push_back((name, value));
        path
    }

    fn pop_front(&mut self) -> Option<(String, Value)> {
        self.pending.pop_front()
    }

    fn has_any(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// Turn a list of top-level `(name, value)` pairs into a MAT 7.3 file.
pub(crate) fn emit_file(fields: Vec<(String, Value)>) -> Result<Vec<u8>, MatError> {
    build_file(fields)?.finish().map_err(MatError::Hdf5)
}

/// Same file as [`emit_file`], streamed to `w` instead of returned. Assembly is
/// front-to-back, so the whole file is never resident.
pub(crate) fn emit_file_to<W: std::io::Write>(
    fields: Vec<(String, Value)>,
    w: W,
) -> Result<(), MatError> {
    build_file(fields)?.finish_to(w).map_err(MatError::Hdf5)
}

/// Stage every field into a [`FileBuilder`] carrying the MAT userblock, ready to
/// be finished either way. Shared so the buffered and streaming entry points
/// cannot come to describe different files.
fn build_file(fields: Vec<(String, Value)>) -> Result<FileBuilder, MatError> {
    let mut builder = FileBuilder::new();
    builder.with_userblock(USERBLOCK_SIZE);
    // This emitter serves the no-options entry points, which are defined to
    // produce what `Options::default()` produces, so every file-level setting the
    // options carry has to be applied here too — the two emitters are only
    // byte-identical while both are configured the same way. See
    // `Options::libver` for why the default is the HDF5 1.8 format.
    builder.with_libver_bounds(crate::LibVer::Earliest, Options::default().libver);
    let mut refs = RefsAccumulator::new();

    for (name, value) in fields {
        if matches!(value, Value::Omit) {
            continue;
        }
        emit_at_root(&mut builder, &name, value, &mut refs)?;
    }

    if refs.has_any() {
        let mut refs_group = builder.create_group(REFS_GROUP);
        // Drain in FIFO order; emitting one entry may itself queue more
        // (nested cells), which the loop will pick up on later iterations.
        while let Some((name, value)) = refs.pop_front() {
            emit_into_group(&mut refs_group, &name, value, &mut refs, RefsH5Path::Own)?;
        }
        builder.add_group(refs_group.finish());
    }

    builder.with_userblock_content(&userblock::header_block(userblock::DEFAULT_DESCRIPTION));
    Ok(builder)
}

/// Emit a single named value at the file root.
fn emit_at_root(
    builder: &mut FileBuilder,
    name: &str,
    value: Value,
    refs: &mut RefsAccumulator,
) -> Result<(), MatError> {
    match value {
        Value::Cell(elements) => {
            apply_cell(builder.create_dataset(name), elements, refs);
            Ok(())
        }
        Value::Leaf(leaf) => {
            apply_leaf_to_dataset(builder.create_dataset(name), leaf);
            Ok(())
        }
        Value::Omit => Ok(()),
        Value::Struct(fields) => {
            let group = build_struct_group(name, fields, refs, RefsH5Path::Omit)?;
            builder.add_group(group);
            Ok(())
        }
    }
}

/// Emit a value as a child of a group.
fn emit_into_group(
    group: &mut GroupBuilder,
    name: &str,
    value: Value,
    refs: &mut RefsAccumulator,
    h5path: RefsH5Path,
) -> Result<(), MatError> {
    match value {
        Value::Cell(elements) => {
            apply_cell(dataset_for(group, name, h5path), elements, refs);
            Ok(())
        }
        Value::Leaf(leaf) => {
            apply_leaf_to_dataset(dataset_for(group, name, h5path), leaf);
            Ok(())
        }
        Value::Omit => Ok(()),
        Value::Struct(fields) => {
            let sub = build_struct_group(name, fields, refs, h5path)?;
            group.add_group(sub);
            Ok(())
        }
    }
}

/// Returns the dataset to write `name` into, and sets the `H5PATH` attribute
/// that addresses a `#refs#` member.
fn dataset_for<'a>(
    group: &'a mut GroupBuilder,
    name: &str,
    h5path: RefsH5Path,
) -> &'a mut DatasetBuilder {
    let ds = group.create_dataset(name);
    if h5path == RefsH5Path::Own {
        ds.set_attr("H5PATH", refs_h5path(name));
    }
    ds
}

/// Build a `FinishedGroup` representing a MATLAB struct.
fn build_struct_group(
    name: &str,
    fields: Vec<(String, Value)>,
    refs: &mut RefsAccumulator,
    h5path: RefsH5Path,
) -> Result<FinishedGroup, MatError> {
    let mut group = new_group_builder(name);
    if h5path == RefsH5Path::Own {
        // Before the struct's own attributes: MATLAB's order on a `#refs#`
        // struct group is H5PATH, then MATLAB_class, then MATLAB_fields.
        group.set_attr("H5PATH", refs_h5path(name));
    }
    // Filter out Omit fields and record the surviving order.
    let mut live_names: Vec<String> = Vec::with_capacity(fields.len());
    for (fname, value) in fields {
        if matches!(value, Value::Omit) {
            continue;
        }
        // A field of a struct is not itself a `#refs#` member, whatever the
        // struct is.
        emit_into_group(&mut group, &fname, value, refs, RefsH5Path::Omit)?;
        // `emit_into_group` only borrows the name, so move it in afterward
        // rather than cloning.
        live_names.push(fname);
    }
    group.set_attr(
        "MATLAB_class",
        AttrValue::AsciiString(MatClass::Struct.as_str().into()),
    );
    group.set_attr("MATLAB_fields", AttrValue::VarLenAsciiCharArray(live_names));
    Ok(group.finish())
}

fn new_group_builder(name: &str) -> GroupBuilder {
    // Same as FileBuilder::create_group but without needing a FileBuilder.
    // GroupBuilder::new is crate-visible.
    GroupBuilder::new(name)
}

/// Writes `leaf` into `ds` as data, shape, and the `MATLAB_class` attribute.
fn apply_leaf_to_dataset(ds: &mut DatasetBuilder, leaf: Leaf) {
    match leaf {
        Leaf::ComplexMatrix { rows, cols, pairs } => {
            let col_major = pairs.transposed(rows, cols);
            apply_complex(ds, col_major, &[cols as u64, rows as u64]);
        }
        Leaf::ComplexScalar(n) => apply_complex(ds, ComplexVec::from_single(n), &[1, 1]),
        Leaf::ComplexVec1D(pairs) => {
            // `[0, 0]` when empty, the answer `dims::vector_dims` gives the
            // options emitter — read the note there on why the two must not
            // drift. An empty complex vector only became reachable with the
            // `mat::complex` array helpers: before them no input produced this
            // arm with no pairs, so the hardcoded `[1, n]` was never wrong.
            let shape = match pairs.len() as u64 {
                0 => [0, 0],
                n => [1, n],
            };
            apply_complex(ds, pairs, &shape);
        }
        Leaf::EmptyStructArray => apply_empty_struct_array(ds),
        Leaf::Matrix { rows, cols, vec } => apply_matrix(ds, rows, cols, vec),
        Leaf::Scalar(n) => apply_scalar(ds, n),
        Leaf::String(s) => apply_char_string(ds, &s),
        Leaf::Vec1D(v) => apply_vec_1d(ds, v),
    }
}

/// Stashes each element under `#refs#` and writes the parent dataset as a vector
/// of object references, oriented the way [`dims::vector_dims`] orients every
/// other 1-D value. An empty cell has no reference to store: its dataset is the
/// `[0, 0]` empty marker [`MatBuilder::cell`] writes.
///
/// The orientation is the default one, since this emitter has no [`Options`] of
/// its own.
///
/// [`MatBuilder::cell`]: crate::mat::MatBuilder::cell
fn apply_cell(ds: &mut DatasetBuilder, elements: Vec<Value>, refs: &mut RefsAccumulator) {
    let paths: Vec<String> = elements.into_iter().map(|el| refs.intern(el)).collect();
    let matlab_dims = dims::vector_dims(paths.len(), Options::default().one_dimensional_mode);
    if paths.is_empty() {
        emit_empty(ds, MatClass::Cell, &matlab_dims);
        return;
    }
    let path_refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    ds.with_path_references(&path_refs)
        .with_shape(&dims::storage_dims_u64(&matlab_dims));
    set_class(ds, MatClass::Cell);
}

/// The empty-marker encoding this emitter writes.
///
/// It has no [`Options`] of its own, so it takes the default's — read from
/// [`Options::default`] rather than written here as a constant, because the same
/// value written through `to_bytes` and through
/// `to_bytes_with_options(&Options::default())` has to produce the same bytes,
/// and a constant is what let those two come apart when the default moved.
fn default_empty_encoding() -> crate::mat::options::EmptyMarkerEncoding {
    Options::default().empty_marker_encoding
}

/// Empty-struct-array marker (MATLAB `struct([])`). What `Option::None` lowers
/// to under the default [`NullPolicy::EmptyStructArray`], both as a struct
/// field and inside a sequence.
///
/// [`NullPolicy::EmptyStructArray`]: crate::mat::NullPolicy::EmptyStructArray
fn apply_empty_struct_array(ds: &mut DatasetBuilder) {
    emit_empty(ds, MatClass::Struct, &[0, 0]);
}

/// Write `pairs` as a `{real, imag}` compound dataset of the given HDF5 shape,
/// tagged with the component's MATLAB class.
///
/// The `match` is the one place the component width becomes a concrete Rust
/// type; a class added to [`ComplexVec`] fails to compile here until it is
/// listed, which is the point.
fn apply_complex(ds: &mut DatasetBuilder, pairs: ComplexVec, shape: &[u64]) {
    let class = pairs.tag().class();
    macro_rules! arms {
        ($($variant:ident),* $(,)?) => {
            match pairs {
                $(ComplexVec::$variant(v) => {
                    ds.with_complex_data(&v).with_shape(shape);
                })*
            }
        };
    }
    arms!(F64, F32, I64, I32, I16, I8, U64, U32, U16, U8);
    set_class(ds, class);
}

fn apply_scalar(ds: &mut DatasetBuilder, n: ScalarNum) {
    match n {
        ScalarNum::Bool(b) => {
            ds.with_u8_data(&[u8::from(b)]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Logical);
            set_logical_decode(ds);
        }
        ScalarNum::F64(x) => {
            ds.with_f64_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Double);
        }
        ScalarNum::F32(x) => {
            ds.with_f32_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Single);
        }
        ScalarNum::I64(x) => {
            ds.with_i64_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int64);
        }
        ScalarNum::I32(x) => {
            ds.with_i32_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int32);
        }
        ScalarNum::I16(x) => {
            ds.with_i16_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int16);
        }
        ScalarNum::I8(x) => {
            ds.with_i8_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::Int8);
        }
        ScalarNum::U64(x) => {
            ds.with_u64_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt64);
        }
        ScalarNum::U32(x) => {
            ds.with_u32_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt32);
        }
        ScalarNum::U16(x) => {
            ds.with_u16_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt16);
        }
        ScalarNum::U8(x) => {
            ds.with_u8_data(&[x]).with_shape(&[1, 1]);
            set_class(ds, MatClass::UInt8);
        }
    }
}

fn apply_vec_1d(ds: &mut DatasetBuilder, v: NumVec) {
    let n = v.len() as u64;
    if n == 0 {
        emit_empty(ds, v.tag().class(), &[0, 0]);
        return;
    }
    let shape = [1u64, n];
    match v {
        NumVec::Bool(vec) => {
            let bytes: Vec<u8> = vec.into_iter().map(u8::from).collect();
            ds.with_u8_data(&bytes).with_shape(&shape);
            set_class(ds, MatClass::Logical);
            set_logical_decode(ds);
        }
        NumVec::F64(vec) => {
            ds.with_f64_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Double);
        }
        NumVec::F32(vec) => {
            ds.with_f32_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Single);
        }
        NumVec::I64(vec) => {
            ds.with_i64_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int64);
        }
        NumVec::I32(vec) => {
            ds.with_i32_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int32);
        }
        NumVec::I16(vec) => {
            ds.with_i16_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int16);
        }
        NumVec::I8(vec) => {
            ds.with_i8_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::Int8);
        }
        NumVec::U64(vec) => {
            ds.with_u64_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt64);
        }
        NumVec::U32(vec) => {
            ds.with_u32_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt32);
        }
        NumVec::U16(vec) => {
            ds.with_u16_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt16);
        }
        NumVec::U8(vec) => {
            ds.with_u8_data(&vec).with_shape(&shape);
            set_class(ds, MatClass::UInt8);
        }
    }
}

fn apply_matrix(ds: &mut DatasetBuilder, rows: usize, cols: usize, vec: NumVec) {
    debug_assert_eq!(vec.len(), rows * cols);
    // An empty matrix is a marker, not a zero-element array of its class, and it
    // keeps the MATLAB shape it was given — `Matrix::from_row_major(0, 3, [])`
    // records `[0, 3]`, not `[0, 0]`. `MatBuilder::write_array_inner` routes the
    // same value to `write_empty` with the same dims; the two emitters have to
    // agree byte for byte under default options.
    if rows * cols == 0 {
        emit_empty(ds, vec.tag().class(), &[rows, cols]);
        return;
    }
    // HDF5 shape for a MATLAB [rows × cols] matrix is [cols, rows].
    let shape = [cols as u64, rows as u64];
    match vec {
        NumVec::Bool(row_major) => {
            // Fuse the column-major transpose with the bool->u8 conversion into
            // a single pass, dropping the intermediate Vec<bool>. Iteration order
            // and index expression match `transpose_scalars` exactly so the
            // column-major byte layout is identical.
            let mut bytes = Vec::with_capacity(rows * cols);
            for c in 0..cols {
                for r in 0..rows {
                    bytes.push(u8::from(row_major[r * cols + c]));
                }
            }
            ds.with_u8_data(&bytes).with_shape(&shape);
            set_class(ds, MatClass::Logical);
            set_logical_decode(ds);
        }
        NumVec::F64(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_f64_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Double);
        }
        NumVec::F32(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_f32_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Single);
        }
        NumVec::I64(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i64_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int64);
        }
        NumVec::I32(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i32_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int32);
        }
        NumVec::I16(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i16_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int16);
        }
        NumVec::I8(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_i8_data(&col).with_shape(&shape);
            set_class(ds, MatClass::Int8);
        }
        NumVec::U64(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u64_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt64);
        }
        NumVec::U32(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u32_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt32);
        }
        NumVec::U16(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u16_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt16);
        }
        NumVec::U8(row_major) => {
            let col = transpose_scalars(rows, cols, &row_major);
            ds.with_u8_data(&col).with_shape(&shape);
            set_class(ds, MatClass::UInt8);
        }
    }
}

fn apply_char_string(ds: &mut DatasetBuilder, s: &str) {
    let units = utf16::encode_utf16(s);
    let n = units.len() as u64;
    if n == 0 {
        emit_empty(ds, MatClass::Char, &[0, 0]);
        return;
    }
    // MATLAB strings are row vectors: MATLAB shape [1, N] → HDF5 [N, 1]
    // (column-major on-disk). This matches libmatio's output and lets
    // MATLAB `strcmp` work without transposing.
    ds.with_u16_data(&units).with_shape(&[n, 1]);
    set_class(ds, MatClass::Char);
    set_char_decode(ds);
}

/// Writes the empty marker for a value of `class` at MATLAB shape
/// `matlab_dims`. The counterpart of [`MatBuilder::write_empty`], which the
/// with-options emitter reaches for the same values.
///
/// No `MATLAB_int_decode`, for any class: it says how to read the stored
/// integers back as `char` or `logical` values, and an empty marker's payload is
/// a `uint64` dimension vector rather than data of the marked class. MATLAB
/// agrees — of the 352 empty datasets in `tests/data/matlab`, not one
/// carries it. `MatBuilder::write_empty` states the same rule; the two have to
/// hold it identically or the emitters diverge for an empty logical.
///
/// [`MatBuilder::write_empty`]: crate::mat::MatBuilder::write_empty
fn emit_empty(ds: &mut DatasetBuilder, class: MatClass, matlab_dims: &[usize]) {
    crate::mat::builder::emit_empty_storage(ds, default_empty_encoding(), class, matlab_dims);
    set_class(ds, class);
    ds.set_attr("MATLAB_empty", AttrValue::U32(1));
}

fn set_class(ds: &mut DatasetBuilder, class: MatClass) {
    ds.set_attr(
        "MATLAB_class",
        AttrValue::AsciiString(class.as_str().into()),
    );
}

/// MATLAB writes logical datasets as uint8 storage with `MATLAB_int_decode = 1`
/// in addition to `MATLAB_class = "logical"`. Without this attribute matio
/// (and MATLAB itself) report the variable as an empty/unknown class.
fn set_logical_decode(ds: &mut DatasetBuilder) {
    ds.set_attr("MATLAB_int_decode", AttrValue::I32(1));
}

/// `char` datasets are uint16 storage; MATLAB also expects
/// `MATLAB_int_decode = 2` so the library decodes the uint16 code units as
/// UTF-16 characters rather than a numeric array.
fn set_char_decode(ds: &mut DatasetBuilder) {
    ds.set_attr("MATLAB_int_decode", AttrValue::I32(2));
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::mat::ser::emit_with_builder;
    use crate::type_builders::AttrSpec;

    /// The rule [`MatBuilder::write_empty`] states — an empty marker carries
    /// `MATLAB_class` and `MATLAB_empty` and nothing else — held for one emitter
    /// and not the other: this one kept `MATLAB_int_decode` on the `logical`
    /// class after the builder dropped it.
    ///
    /// Asserted here rather than through `to_bytes` because no serde value
    /// reaches this arm: the sequence accumulator lowers an empty
    /// `Vec<bool>` to `NumVec::F64` under [`EmptySequencePolicy::DoubleArray`],
    /// so an end-to-end test would pass with the defect in place. A change to
    /// that policy is exactly what would deliver a typed empty logical here.
    ///
    /// [`MatBuilder::write_empty`]: crate::mat::MatBuilder::write_empty
    /// [`EmptySequencePolicy::DoubleArray`]: crate::mat::EmptySequencePolicy::DoubleArray
    #[rstest]
    #[case(MatClass::Double)]
    #[case(MatClass::Single)]
    #[case(MatClass::Int8)]
    #[case(MatClass::Int16)]
    #[case(MatClass::Int32)]
    #[case(MatClass::Int64)]
    #[case(MatClass::UInt8)]
    #[case(MatClass::UInt16)]
    #[case(MatClass::UInt32)]
    #[case(MatClass::UInt64)]
    #[case(MatClass::Char)]
    #[case(MatClass::Logical)]
    #[case(MatClass::Struct)]
    #[case(MatClass::Cell)]
    fn an_empty_marker_carries_only_class_and_empty(#[case] class: MatClass) {
        let mut ds = DatasetBuilder::new("x");
        emit_empty(&mut ds, class, &[0, 0]);
        let mut names: Vec<&str> = ds.attrs.iter().map(|(n, _)| n.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["MATLAB_class", "MATLAB_empty"],
            "{class:?} empty marker carries a non-MATLAB attribute set"
        );
    }

    /// An empty matrix keeps the shape it was given. `MatBuilder`'s
    /// `write_array_inner` forwards `matlab_dims` to `write_empty` unchanged, so
    /// collapsing every empty to `0x0` here would make the two emitters disagree
    /// for `Matrix::from_row_major(0, 3, vec![])`.
    #[test]
    fn an_empty_matrix_keeps_its_matlab_dims() {
        for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
            let mut ds = DatasetBuilder::new("m");
            apply_matrix(&mut ds, rows, cols, NumVec::F64(Vec::new()));
            assert_eq!(
                ds.data.as_deref(),
                Some(
                    [rows as u64, cols as u64]
                        .iter()
                        .flat_map(|d| d.to_le_bytes())
                        .collect::<Vec<u8>>()
                        .as_slice()
                ),
                "{rows}x{cols} empty matrix records the wrong dimension vector"
            );
            assert!(
                ds.attrs.iter().any(|(n, _)| n == "MATLAB_empty"),
                "{rows}x{cols} empty matrix is not marked empty"
            );
        }
    }

    #[test]
    fn an_empty_cell_writes_the_zero_by_zero_marker() {
        let mut ds = DatasetBuilder::new("c");
        apply_cell(&mut ds, Vec::new(), &mut RefsAccumulator::new());
        assert_eq!(
            ds.data.as_deref(),
            Some(
                [0u64, 0]
                    .iter()
                    .flat_map(|d| d.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .as_slice()
            ),
            "an empty cell records the wrong dimension vector"
        );
        let attrs: Vec<(&str, AttrValue)> = ds
            .attrs
            .iter()
            .map(|(name, spec)| {
                let AttrSpec::Value(value) = spec else {
                    panic!("an empty cell carries a pre-encoded {name:?} attribute");
                };
                (name.as_str(), value.clone())
            })
            .collect();
        assert_eq!(
            attrs,
            [
                ("MATLAB_class", AttrValue::AsciiString("cell".to_owned())),
                ("MATLAB_empty", AttrValue::U32(1)),
            ]
        );
    }

    #[test]
    fn an_empty_cell_writes_what_the_with_options_emitter_writes() {
        let fields = || vec![("c".to_owned(), Value::Cell(Vec::new()))];
        assert_eq!(
            emit_file(fields()).unwrap(),
            emit_with_builder::emit_file_with_options(fields(), &Options::default()).unwrap(),
        );
    }
}
