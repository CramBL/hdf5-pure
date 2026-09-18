//! Read an HDF5 file with MATLAB conventions into a `MatValue` tree.

use std::collections::HashMap;

use crate::convert::Narrow;
use crate::display::EscapedName;
use crate::error::FormatError;
use crate::file_writer::AttrValue;
use crate::mat::class::MatClass;
use crate::mat::de::mcos_reader::{self, Mcos, PropValue};
use crate::mat::error::MatError;
use crate::mat::string_object::MCOS_MAGIC_NUMBER;
use crate::mat::utf16;
use crate::mat::value::{ComplexTag, ComplexVec, MatValue, NumVec, ScalarNum};
use crate::reader::{Dataset, File, Group, Object};
use crate::type_builders::ComplexComponent;
use crate::types::DType;

/// Maximum struct/cell nesting depth followed when reading. Real MATLAB data
/// nests a handful of levels deep; this bound is generous headroom that still
/// turns a malformed file — e.g. a cell whose object reference points back at
/// itself, or a cyclic group link — into a typed error instead of unbounded
/// recursion and a stack overflow.
const MAX_NESTING_DEPTH: usize = 256;

/// Parse a MAT v7.3 file into an ordered list of `(name, value)` fields
/// rooted at the file's top level.
pub(crate) fn read_file(bytes: &[u8]) -> Result<Vec<(String, MatValue)>, MatError> {
    // Accept raw HDF5 too — we don't require the MATLAB signature to be
    // present to deserialize successfully.
    let file = File::from_bytes(bytes.to_vec()).map_err(MatError::Hdf5)?;
    // Parse the MCOS opaque-object store once; opaque datasets (modern `string`,
    // …) anywhere in the file resolve their payloads against it. `None` when the
    // file has no `#subsystem#` group.
    let mcos = Mcos::parse(&file)?;
    let root = file.root();
    read_group(&root, mcos.as_ref(), false, 0)
}

/// Read a struct group's `(name, value)` fields.
///
/// `in_heap` marks reads that descend from an MCOS opaque object's property
/// heap (see [`read_dataset`]); the top level passes `false`.
fn read_group(
    group: &Group,
    mcos: Option<&Mcos>,
    in_heap: bool,
    depth: usize,
) -> Result<Vec<(String, MatValue)>, MatError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(MatError::Format(FormatError::NestingDepthExceeded));
    }
    let mut out = Vec::new();

    // Collect datasets first, then subgroups. Preserve iteration order from
    // the HDF5 link table. Skip MATLAB's reserved internal entries — `#refs#`
    // (cell/object-reference payloads) and `#subsystem#` (the MCOS opaque-class
    // store): MATLAB hides them, no variable name can begin with `#`, and their
    // contents are reached by reference from the variables that use them, not
    // by walking the top level.
    for name in group.datasets().map_err(MatError::Hdf5)? {
        if name.starts_with('#') {
            continue;
        }
        let ds = group.dataset(&name).map_err(MatError::Hdf5)?;
        out.push((name, read_dataset(&ds, mcos, in_heap, depth)?));
    }
    for name in group.groups().map_err(MatError::Hdf5)? {
        if name.starts_with('#') {
            continue;
        }
        let sub = group.group(&name).map_err(MatError::Hdf5)?;
        out.push((name, read_group_as_value(&sub, mcos, in_heap, depth + 1)?));
    }
    Ok(out)
}

fn read_group_as_value(
    group: &Group,
    mcos: Option<&Mcos>,
    in_heap: bool,
    depth: usize,
) -> Result<MatValue, MatError> {
    // An MCOS enumeration instance is stored as a group of metadata datasets,
    // not as an ordinary struct: a top-level enum variable (with
    // `MATLAB_object_decode=3` and the enum class as `MATLAB_class`), or an
    // enum reached as a struct/heap cell (`MATLAB_class="struct"`). Both forms
    // are the same group shape, and every group read funnels through here, so
    // detect and decode the enumeration before falling back to a struct.
    if let Some(mcos) = mcos {
        if let Some(value) = try_decode_enum_instance(group, mcos, depth)? {
            return Ok(value);
        }
    }
    // A struct *array* stores each field as a dataset of per-element object
    // references rather than as a direct value; detect that layout and transpose
    // it into an array-of-structs before falling back to reading the group as a
    // scalar struct.
    if is_struct_array(group)? {
        return read_struct_array(group, mcos, in_heap, depth);
    }
    let fields = read_group(group, mcos, in_heap, depth)?;
    Ok(MatValue::Struct(fields))
}

/// Detect MATLAB's struct-*array* on-disk layout: a `MATLAB_class="struct"`
/// group whose every field is a dataset of object references (one reference per
/// array element), as opposed to a *scalar* struct whose fields are direct value
/// datasets or nested-struct subgroups.
///
/// Mirrors matio's `is_struct_matrix` heuristic — no subgroups, and no field
/// dataset carries a `MATLAB_class` attribute — with an added object-reference
/// dtype guard. The guard matters because the reader also accepts raw (non-MAT)
/// HDF5: a plain group of numeric datasets has no `MATLAB_class` either, and
/// must still read as an ordinary struct rather than be misread as a struct
/// array (and then fail trying to dereference non-references). A *cell* field of
/// a scalar struct is a reference dataset too, but carries `MATLAB_class="cell"`,
/// so the class-attribute check keeps that case a scalar struct.
fn is_struct_array(group: &Group) -> Result<bool, MatError> {
    // A nested-struct field of a scalar struct is stored as a subgroup, so any
    // subgroup means this is a scalar struct, not a struct array.
    if !group.groups().map_err(MatError::Hdf5)?.is_empty() {
        return Ok(false);
    }
    let mut saw_field = false;
    for name in group.datasets().map_err(MatError::Hdf5)? {
        if name.starts_with('#') {
            continue;
        }
        saw_field = true;
        let ds = group.dataset(&name).map_err(MatError::Hdf5)?;
        // Check the (cheaper, value-excluding) dtype first: a scalar struct's
        // numeric/char field is not a reference, so this short-circuits without
        // parsing attributes.
        if ds.dtype().map_err(MatError::Hdf5)? != DType::ObjectReference {
            return Ok(false);
        }
        if ds
            .attrs()
            .map_err(MatError::Hdf5)?
            .contains_key("MATLAB_class")
        {
            return Ok(false);
        }
    }
    Ok(saw_field)
}

/// Read a struct-array group into a [`MatValue::StructArray`].
///
/// Each field is a dataset of object references — one per array element — so the
/// fields form a struct-of-arrays on disk. This dereferences and decodes each
/// element's field value through the normal read path (so nested structs, cells,
/// and opaque objects all decode), then transposes the column-major storage into
/// a row-major array-of-structs, matching how numeric matrices present their
/// elements.
fn read_struct_array(
    group: &Group,
    mcos: Option<&Mcos>,
    in_heap: bool,
    depth: usize,
) -> Result<MatValue, MatError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(MatError::Format(FormatError::NestingDepthExceeded));
    }
    // Field order follows the group's dataset link order, matching how a scalar
    // struct's fields are read (serde matches fields by name, so order is only
    // cosmetic).
    let field_names: Vec<String> = group
        .datasets()
        .map_err(MatError::Hdf5)?
        .into_iter()
        .filter(|n| !n.starts_with('#'))
        .collect();
    let first_name = field_names
        .first()
        .ok_or_else(|| MatError::Custom("struct array group has no fields".into()))?;

    // Every field's reference dataset shares the struct array's dimensions.
    let first = group.dataset(first_name).map_err(MatError::Hdf5)?;
    let (rows, cols, total) = shape_decomposition(&first.shape().map_err(MatError::Hdf5)?)?;

    // Decode each field's per-element values in column-major storage order. Each
    // value is wrapped in `Option` so the transpose below can move it out exactly
    // once without cloning potentially large field values.
    let mut columns: Vec<(String, Vec<Option<MatValue>>)> = Vec::with_capacity(field_names.len());
    for name in &field_names {
        let ds = group.dataset(name).map_err(MatError::Hdf5)?;
        let members = ds.dereference().map_err(MatError::Hdf5)?;
        if members.len() != total {
            return Err(MatError::Custom(format!(
                "struct array field {name:?} has {} elements; expected {total} to match the array shape",
                members.len()
            )));
        }
        let mut col = Vec::with_capacity(total);
        for member in &members {
            col.push(Some(read_object(member, mcos, in_heap, depth + 1)?));
        }
        columns.push((name.clone(), col));
    }

    // Transpose column-major storage (`storage = c * rows + r`) into row-major
    // array-of-structs. For a `1×N` / `N×1` array one of `rows`/`cols` is 1, so
    // storage order already equals linear order and this is an identity walk.
    let mut elements: Vec<Vec<(String, MatValue)>> = Vec::with_capacity(total);
    for r in 0..rows {
        for c in 0..cols {
            let storage = c * rows + r;
            let mut fields = Vec::with_capacity(columns.len());
            for (name, col) in columns.iter_mut() {
                let value = col[storage]
                    .take()
                    .expect("each storage slot is consumed exactly once");
                fields.push((name.clone(), value));
            }
            elements.push(fields);
        }
    }

    Ok(MatValue::StructArray {
        rows,
        cols,
        elements,
    })
}

/// Detect and decode an MCOS enumeration-instance group, returning `Ok(None)`
/// when `group` is an ordinary struct rather than an enumeration.
///
/// An enumeration instance is a group of six metadata datasets:
/// - `EnumerationInstanceTag` — a scalar `uint32` whose value is the MCOS magic
///   number; this is the unambiguous marker that the group is an enumeration.
/// - `ClassName` — the enum class as a 1-based MCOS class-table id.
/// - `ValueNames` — the member-name pool, as 1-based name-heap indices.
/// - `ValueIndices` — the per-element 0-based index into that pool, carrying the
///   array's MATLAB shape.
/// - `Values` — the underlying value backing each member (not surfaced).
/// - `BuiltinClassName` — the storage class of the underlying value (unused).
///
/// Only the absent tag (or a tag that is not the magic number) means "ordinary
/// group, not an enumeration". Once the magic tag matches, `ClassName`,
/// `ValueNames`, and `ValueIndices` are required: their absence, an unresolvable
/// class id, or an out-of-range member index is treated as a corrupt enumeration
/// and surfaced as an error. This deliberately differs from the optional-property
/// opaque decoders (`datetime` / `duration` / `categorical`), which substitute
/// empty defaults — there MATLAB genuinely omits a property for an empty value,
/// whereas a valid enumeration never omits these datasets.
fn try_decode_enum_instance(
    group: &Group,
    mcos: &Mcos,
    depth: usize,
) -> Result<Option<MatValue>, MatError> {
    // The tag dataset's absence means an ordinary group, so a failed lookup is
    // "not an enumeration", not an error.
    let tag_ds = match group.dataset("EnumerationInstanceTag") {
        Ok(ds) => ds,
        Err(_) => return Ok(None),
    };
    let tag = tag_ds.read_u32().map_err(MatError::Hdf5)?;
    if tag.first() != Some(&MCOS_MAGIC_NUMBER) {
        return Ok(None);
    }

    // Resolve the fully qualified class from its class id, falling back to the
    // group's `MATLAB_class` attribute if the id does not resolve (it is
    // `"struct"` for an in-heap enum cell, which the fallback discards). Both the
    // top-level and in-heap cases normally resolve via the class id; the error
    // is only reached when the id is out of range *and* no usable attribute
    // remains, i.e. a corrupt instance.
    let class_id = read_enum_scalar_u32(group, "ClassName")?;
    let class_name = mcos
        .class_name(class_id)
        .ok()
        .filter(|name| !name.is_empty())
        .or_else(|| enum_class_from_attr(group))
        .ok_or_else(|| {
            MatError::Custom(format!(
                "enum ClassName id {class_id} did not resolve and no MATLAB_class fallback was usable"
            ))
        })?;

    // The member-name pool: each `ValueNames` entry is a 1-based name index.
    let pool: Vec<String> = read_enum_u32_vec(group, "ValueNames")?
        .iter()
        .map(|&idx| mcos.name(idx))
        .collect::<Result<_, _>>()?;

    // Map each element's pool index to its member name, row-major. An
    // out-of-range index is corruption, not a silently-empty name.
    let names = enum_member_names(&pool, &read_enum_indices(group, depth)?)?;

    Ok(Some(mcos_reader::build_enum_value(class_name, names)))
}

/// Map each element's 0-based `ValueIndices` entry to its member name in the
/// `ValueNames` pool. An out-of-range index is a corrupt enumeration, surfaced
/// as an error rather than a silently-empty name (matching how `ValueNames`'
/// own name-heap indices are resolved, and the out-of-range handling throughout
/// the MCOS reader).
fn enum_member_names(pool: &[String], indices: &[usize]) -> Result<Vec<String>, MatError> {
    indices
        .iter()
        .map(|&i| {
            pool.get(i).cloned().ok_or_else(|| {
                MatError::Custom(format!(
                    "enum ValueIndices index {i} out of range (pool has {} members)",
                    pool.len()
                ))
            })
        })
        .collect()
}

/// Open a required enumeration-metadata dataset. The magic tag has already
/// matched, so a missing dataset is a corrupt enumeration; map the HDF5 lookup
/// error to a clear message rather than a bare "link not found".
fn enum_dataset(group: &Group, field: &str) -> Result<Dataset, MatError> {
    group.dataset(field).map_err(|e| {
        MatError::Custom(format!(
            "malformed enum instance: cannot read required dataset {field:?} ({e})"
        ))
    })
}

/// Read an enumeration metadata field as a single `uint32` (its first element,
/// or `0` if the dataset is empty).
fn read_enum_scalar_u32(group: &Group, field: &str) -> Result<u32, MatError> {
    let ds = enum_dataset(group, field)?;
    Ok(ds
        .read_u32()
        .map_err(MatError::Hdf5)?
        .first()
        .copied()
        .unwrap_or(0))
}

/// Read an enumeration metadata field as a flat `uint32` vector. The member-name
/// pool is a `1×N` row, so its storage order is its presentation order.
fn read_enum_u32_vec(group: &Group, field: &str) -> Result<Vec<u32>, MatError> {
    let ds = enum_dataset(group, field)?;
    ds.read_u32().map_err(MatError::Hdf5)
}

/// Read the `ValueIndices` array as a flat `Vec<usize>` in the crate's row-major
/// presentation order.
fn read_enum_indices(group: &Group, depth: usize) -> Result<Vec<usize>, MatError> {
    let ds = enum_dataset(group, "ValueIndices")?;
    // Reject a non-`uint32` index array up front with a dtype-specific message
    // (mirrors the saveobj-payload datatype guard), so corruption surfaces
    // clearly rather than as a vaguer downstream shape mismatch.
    let dtype = ds.dtype().map_err(MatError::Hdf5)?;
    if dtype != DType::U32 {
        return Err(MatError::Custom(format!(
            "enum ValueIndices has datatype {dtype}; expected uint32"
        )));
    }
    // Reuse the numeric read path so the column-major HDF5 buffer is transposed
    // into row-major, matching how other opaque arrays present their elements.
    let value = read_dataset(&ds, None, false, depth)?;
    enum_indices_to_usize(&value)
}

/// Flatten a decoded `ValueIndices` value (a `uint32` scalar, vector, or matrix)
/// into row-major `usize` indices.
fn enum_indices_to_usize(value: &MatValue) -> Result<Vec<usize>, MatError> {
    match value {
        MatValue::Scalar(ScalarNum::U32(x)) => Ok(vec![x.to_usize().map_err(MatError::Format)?]),
        MatValue::Vec1D(NumVec::U32(v))
        | MatValue::Matrix {
            vec: NumVec::U32(v),
            ..
        } => v
            .iter()
            .map(|&x| x.to_usize().map_err(MatError::Format))
            .collect(),
        other => Err(MatError::Custom(format!(
            "enum ValueIndices is not a uint32 array (got {})",
            other.kind()
        ))),
    }
}

/// The group's `MATLAB_class` attribute, unless it is the generic `"struct"`
/// (an in-heap enum cell carries no real class name there).
fn enum_class_from_attr(group: &Group) -> Option<String> {
    let attrs = group.attrs().ok()?;
    let class = mcos_reader::raw_matlab_class(&attrs)?;
    (class != "struct").then_some(class)
}

/// Read a single dataset into a `MatValue` using its MATLAB_class attribute
/// (if present) and its HDF5 shape. `depth` bounds struct/cell nesting.
///
/// `in_heap` is `true` when this dataset is reached through an MCOS opaque
/// object's property heap (a property value, or an element of a cell / field of
/// a struct nested under one). Inside that heap a property that is *itself* an
/// opaque object is stored as a bare `uint32` metadata array (the same
/// `[MAGIC, ndims, dims…, object ids…, class id]` layout an opaque parent
/// dataset carries) **without** a `MATLAB_object_decode` attribute. Such
/// embedded references are resolved here; a top-level dataset (`in_heap ==
/// false`) is never treated this way, so ordinary `uint32` data cannot be
/// misread as a reference.
fn read_dataset(
    ds: &Dataset,
    mcos: Option<&Mcos>,
    in_heap: bool,
    depth: usize,
) -> Result<MatValue, MatError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(MatError::Format(FormatError::NestingDepthExceeded));
    }
    let attrs = ds.attrs().map_err(MatError::Hdf5)?;

    // An `mxOPAQUE_CLASS` object (`MATLAB_object_decode` set) stores its data in
    // the `#subsystem#` MCOS store, not on this dataset. Decode it there before
    // the builtin-class path — its `MATLAB_class` (e.g. `string`) is not a
    // `MatClass` variant.
    if let Some(decode) = mcos_reader::matlab_object_decode(&attrs) {
        return decode_opaque(ds, &attrs, decode, mcos, depth);
    }

    let class = matlab_class_from_attrs(&attrs)?;
    let shape = ds.shape().map_err(MatError::Hdf5)?;
    let dtype = ds.dtype().map_err(MatError::Hdf5)?;
    let is_empty = is_empty_attr(&attrs) || shape.contains(&0);

    // Embedded MCOS object reference: a non-empty `uint32` heap value whose
    // contents parse as opaque metadata names one or more nested objects (e.g. a
    // `string`/`datetime`/`categorical` column inside a table's `data` cell).
    // The exact-length validation in `parse_opaque_metadata`, plus the
    // reserved `MAGIC` first word, keeps genuine `uint32` data from matching.
    // Retain the probe's decoded buffer: when the payload turns out to be a
    // plain uint32 array rather than an object reference, it is reused below
    // instead of reading (and decompressing) the same dataset a second time.
    let mut probed_u32: Option<Vec<u32>> = None;
    if in_heap && !is_empty && dtype == DType::U32 {
        if let Some(mcos) = mcos {
            let raw = ds.read_u32().map_err(MatError::Hdf5)?;
            if let Ok(meta) = mcos_reader::parse_opaque_metadata(&raw) {
                if !meta.object_ids.is_empty() {
                    return decode_object_ids(mcos, &meta.object_ids, depth);
                }
            }
            probed_u32 = Some(raw);
        }
    }

    let class = class.unwrap_or_else(|| class_from_dtype(&dtype));

    if is_empty {
        // Complex compound types preserve their shape and class so a 0×0
        // `Matrix<Complex*>` round-trips back to a 0×0 complex matrix
        // rather than collapsing to a numeric empty vec.
        if is_complex_dtype(&dtype)
            && let Some(tag) = ComplexTag::from_class(class)
        {
            return empty_complex_value(tag, &shape);
        }
        // An empty struct-classed *dataset* (`MATLAB_empty=1`) is `struct([])`,
        // the marker the serializer writes for `Option::None` inside a
        // sequence. Surface it as `EmptyStructArray` so an `Option` field
        // round-trips to `None` (an empty struct *group*, by contrast, reads as
        // an empty struct via `read_group_as_value`).
        if class == MatClass::Struct {
            return Ok(MatValue::EmptyStructArray);
        }
        // Empty numeric/char: produce an empty 1-D vec of the correct tag.
        return Ok(empty_value_for_class(class));
    }

    // The object-reference probe above already read this uint32 payload. When
    // the value is a plain uint32 array (the resolved class is `UInt32`, so the
    // `UInt32` match arm below would take the non-complex `read_numeric` path),
    // reuse that buffer rather than reading the dataset again.
    if class == MatClass::UInt32 {
        if let Some(raw) = probed_u32 {
            return numeric_value_from_flat(NumVec::U32(raw), &shape);
        }
    }

    match class {
        MatClass::Char => {
            let units = ds.read_u16().map_err(MatError::Hdf5)?;
            let s = utf16::decode_utf16(&units)?;
            Ok(MatValue::String(s))
        }
        MatClass::Logical => read_numeric(ds, &shape, class),
        MatClass::Double
        | MatClass::Single
        | MatClass::Int8
        | MatClass::Int16
        | MatClass::Int32
        | MatClass::Int64
        | MatClass::UInt8
        | MatClass::UInt16
        | MatClass::UInt32
        | MatClass::UInt64 => {
            if is_complex_dtype(&dtype) {
                read_complex(ds, &shape, class, &dtype)
            } else {
                read_numeric(ds, &shape, class)
            }
        }
        MatClass::Struct => Err(MatError::Custom(
            "dataset has MATLAB_class='struct'; expected a group".into(),
        )),
        MatClass::Cell => read_cell(ds, mcos, in_heap, depth),
    }
}

/// Read a cell-array dataset: a vector of HDF5 object references, each pointing
/// at a member object (struct group, numeric/char dataset, nested cell, …)
/// interned under `#refs#`. Members are decoded in storage order and collected
/// into a flat [`MatValue::Cell`], matching the column-vector layout the
/// serializer writes (the deserializer flattens a cell to a sequence).
fn read_cell(
    ds: &Dataset,
    mcos: Option<&Mcos>,
    in_heap: bool,
    depth: usize,
) -> Result<MatValue, MatError> {
    let members = ds.dereference().map_err(MatError::Hdf5)?;
    let mut elems = Vec::with_capacity(members.len());
    for member in members {
        elems.push(read_object(&member, mcos, in_heap, depth + 1)?);
    }
    Ok(MatValue::Cell(elems))
}

/// Decode a dereferenced object (a `#refs#`/`#subsystem#` dataset or struct
/// group) into a `MatValue`, dispatching on whether it is a dataset or group.
fn read_object(
    obj: &Object,
    mcos: Option<&Mcos>,
    in_heap: bool,
    depth: usize,
) -> Result<MatValue, MatError> {
    match obj {
        Object::Dataset(d) => read_dataset(d, mcos, in_heap, depth),
        Object::Group(g) => read_group_as_value(g, mcos, in_heap, depth),
    }
}

/// Decode an `mxOPAQUE_CLASS` dataset (one with `MATLAB_object_decode` set).
///
/// The modern `string` class keeps its dedicated saveobj decoder; every other
/// opaque class resolves its object id(s) from the parent `uint32` metadata and
/// decodes each object through the `#subsystem#/MCOS` FileWrapper property
/// tables — to a typed value (`datetime`, `duration`, `categorical`) or a
/// lossless [`MatValue::Opaque`] for classes without a dedicated decoder.
fn decode_opaque(
    ds: &Dataset,
    attrs: &HashMap<String, AttrValue>,
    decode: i64,
    mcos: Option<&Mcos>,
    depth: usize,
) -> Result<MatValue, MatError> {
    let class = mcos_reader::raw_matlab_class(attrs).ok_or_else(|| {
        MatError::Custom("opaque object (MATLAB_object_decode set) has no MATLAB_class".into())
    })?;
    // Only `MATLAB_object_decode == 3` (mxOPAQUE_CLASS / MCOS) is decoded here.
    // `1` (function handle) and `2` (legacy object) have different on-disk
    // layouts; refuse them by name rather than misread.
    if !mcos_reader::is_mcos_decode(decode) {
        return Err(MatError::UnsupportedMatlabClass(class));
    }
    let mcos = mcos.ok_or_else(|| {
        MatError::Custom(
            "file references an MCOS opaque object but has no #subsystem# store".into(),
        )
    })?;

    if mcos_reader::is_string_class(&class, decode) {
        return mcos.decode_string(ds);
    }

    let meta = ds.read_u32().map_err(MatError::Hdf5)?;
    let parsed = mcos_reader::parse_opaque_metadata(&meta)?;
    if parsed.object_ids.is_empty() {
        // An empty opaque array names no objects; resolve its declared class
        // from the parent metadata's trailing class id (falling back to the
        // dataset's `MATLAB_class`) and surface it with no properties so the
        // read still succeeds rather than erroring.
        let class_name = mcos.class_name(parsed.class_id).unwrap_or(class);
        return Ok(MatValue::Opaque {
            class_name,
            fields: Vec::new(),
        });
    }

    decode_object_ids(mcos, &parsed.object_ids, depth)
}

/// Decode one or more MCOS object ids (from an opaque parent dataset or an
/// embedded heap reference) into a value. A single id is the object itself; a
/// genuine object array becomes a cell of the decoded objects.
fn decode_object_ids(mcos: &Mcos, object_ids: &[u32], depth: usize) -> Result<MatValue, MatError> {
    let mut values = Vec::with_capacity(object_ids.len());
    for &object_id in object_ids {
        values.push(decode_opaque_object(mcos, object_id, depth)?);
    }
    if values.len() == 1 {
        Ok(values.pop().expect("len checked"))
    } else {
        Ok(MatValue::Cell(values))
    }
}

/// Assemble one MCOS object's resolved properties (reading heap cells through
/// the full read path so nested objects/cells decode too) and finish it via a
/// typed decoder or the lossless `Opaque` fallback.
fn decode_opaque_object(mcos: &Mcos, object_id: u32, depth: usize) -> Result<MatValue, MatError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(MatError::Format(FormatError::NestingDepthExceeded));
    }
    let class_name = mcos.object_class_name(object_id)?;
    let mut fields = Vec::new();
    for prop in mcos.properties(object_id)? {
        // Property values descend from the opaque object's heap, so read them
        // with `in_heap = true`: a property (or a cell element / struct field
        // beneath it) that is itself an opaque object decodes instead of
        // surfacing as its raw `uint32` reference metadata.
        let value = match prop.value {
            PropValue::Heap(v) => read_object(mcos.heap_object(v)?, Some(mcos), true, depth + 1)?,
            PropValue::Inline(n) => MatValue::Scalar(mcos_reader::inline_to_scalar(n)),
            PropValue::Name(s) => MatValue::String(s),
        };
        fields.push((prop.name, value));
    }
    mcos_reader::decode_object_fields(class_name, fields)
}

/// Extract and parse the `MATLAB_class` attribute value, if present.
fn matlab_class_from_attrs(
    attrs: &HashMap<String, AttrValue>,
) -> Result<Option<MatClass>, MatError> {
    let raw = match attrs.get("MATLAB_class") {
        None => None,
        Some(value) => Some(
            value
                .as_str()
                .ok_or_else(|| {
                    MatError::Custom(format!(
                        "MATLAB_class attribute has unexpected type: {value:?}"
                    ))
                })?
                .to_string(),
        ),
    };
    match raw {
        Some(s) => Ok(Some(MatClass::parse(&s)?)),
        None => Ok(None),
    }
}

fn is_empty_attr(attrs: &HashMap<String, AttrValue>) -> bool {
    // Through the accessor rather than per-variant arms: writers differ on the
    // width and on whether the flag is a scalar or a one-element array, and
    // the reader reports each of those faithfully.
    attrs
        .get("MATLAB_empty")
        .and_then(AttrValue::as_i64)
        .is_some_and(|v| v != 0)
}

/// Build an empty `ComplexMatrix` whose `(rows, cols)` matches the dataspace
/// shape, so deserializing into `Matrix<Complex*>` recovers the original
/// shape (0×0, 0×N, N×0) instead of collapsing to 1×0.
fn empty_complex_value(tag: ComplexTag, shape: &[u64]) -> Result<MatValue, MatError> {
    let (rows, cols, _total) = shape_decomposition(shape)?;
    Ok(MatValue::ComplexMatrix {
        rows,
        cols,
        pairs: ComplexVec::empty_with_tag(tag),
    })
}

fn empty_value_for_class(class: MatClass) -> MatValue {
    use crate::mat::value::ScalarTag;
    match class {
        MatClass::Char => MatValue::String(String::new()),
        MatClass::Logical => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::Bool)),
        MatClass::Double => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::F64)),
        MatClass::Single => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::F32)),
        MatClass::Int8 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I8)),
        MatClass::Int16 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I16)),
        MatClass::Int32 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I32)),
        MatClass::Int64 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::I64)),
        MatClass::UInt8 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U8)),
        MatClass::UInt16 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U16)),
        MatClass::UInt32 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U32)),
        MatClass::UInt64 => MatValue::Vec1D(NumVec::empty_with_tag(ScalarTag::U64)),
        MatClass::Struct => MatValue::Struct(Vec::new()),
        MatClass::Cell => MatValue::Cell(Vec::new()),
    }
}

/// Infer a `MatClass` from the raw `DType` when no `MATLAB_class` attribute
/// is present (so non-MAT files still work).
fn class_from_dtype(dtype: &DType) -> MatClass {
    match dtype {
        DType::F64 => MatClass::Double,
        DType::F32 => MatClass::Single,
        DType::I8 => MatClass::Int8,
        DType::I16 => MatClass::Int16,
        DType::I32 => MatClass::Int32,
        DType::I64 => MatClass::Int64,
        DType::U8 => MatClass::UInt8,
        DType::U16 => MatClass::UInt16,
        DType::U32 => MatClass::UInt32,
        DType::U64 => MatClass::UInt64,
        DType::String => MatClass::Char,
        DType::VariableLengthString => MatClass::Char,
        _ => MatClass::Double, // fallback guess
    }
}

fn is_complex_dtype(dtype: &DType) -> bool {
    match dtype {
        DType::Compound(fields) => {
            fields.len() == 2
                && fields.iter().any(|(n, _)| n == "real")
                && fields.iter().any(|(n, _)| n == "imag")
        }
        _ => false,
    }
}

/// The component tag a stored datatype decodes as, or `None` for anything that
/// is not one of the ten numeric component classes.
///
/// Deliberately exhaustive with no fallback arm: [`class_from_dtype`] guesses
/// `Double` for an unrecognized type, which is the right shape for naming a
/// class but the wrong one for deciding how to slice raw bytes.
fn complex_component_tag(dtype: &DType) -> Option<ComplexTag> {
    Some(match dtype {
        DType::F64 => ComplexTag::F64,
        DType::F32 => ComplexTag::F32,
        DType::I64 => ComplexTag::I64,
        DType::I32 => ComplexTag::I32,
        DType::I16 => ComplexTag::I16,
        DType::I8 => ComplexTag::I8,
        DType::U64 => ComplexTag::U64,
        DType::U32 => ComplexTag::U32,
        DType::U16 => ComplexTag::U16,
        DType::U8 => ComplexTag::U8,
        _ => return None,
    })
}

/// Check that a complex compound is laid out the way `tag` says it is before
/// any of its bytes are decoded.
///
/// The component width comes from the `MATLAB_class` attribute, but the bytes
/// come from the compound, and nothing forces the two to agree: a file can
/// claim `int16` over a pair of `f64` members, or carry no class attribute at
/// all and fall back to `double` over a pair of `int64`s at the identical
/// element size. Decoding either one succeeds and returns numbers that are
/// entirely wrong, so the disagreement is refused here instead.
fn validate_complex_dtype(dtype: &DType, tag: ComplexTag) -> Result<(), MatError> {
    let DType::Compound(fields) = dtype else {
        return Err(MatError::Custom(format!(
            "complex {} dataset has a non-compound datatype {dtype}",
            tag.class().as_str()
        )));
    };
    // Real first, imaginary second: `parse_complex_pairs` reads the component
    // at offset 0 as the real part, so the declared order is load-bearing and
    // the reversed layout has to be refused rather than silently swapped.
    let (Some((first, first_ty)), Some((second, second_ty))) = (fields.first(), fields.get(1))
    else {
        return Err(MatError::Custom(format!(
            "complex {} compound has {} fields, expected 2",
            tag.class().as_str(),
            fields.len()
        )));
    };
    if first != "real" || second != "imag" {
        // The file names these members, so they are escaped for the same reason
        // `DType` escapes them: `{member}` on the next message already is, and
        // a raw name beside it could carry a newline into the same string.
        return Err(MatError::Custom(format!(
            "complex {} compound is ordered {{{}, {}}}, expected {{real, imag}}",
            tag.class().as_str(),
            EscapedName(first),
            EscapedName(second)
        )));
    }
    for (name, member) in [(first, first_ty), (second, second_ty)] {
        if complex_component_tag(member) != Some(tag) {
            return Err(MatError::Custom(format!(
                "complex {} dataset stores its {} component as {member}, \
                 which is a different class than MATLAB_class claims",
                tag.class().as_str(),
                EscapedName(name)
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Numeric reading
// ---------------------------------------------------------------------------

fn read_numeric(ds: &Dataset, shape: &[u64], class: MatClass) -> Result<MatValue, MatError> {
    let (_, _, total) = shape_decomposition(shape)?;

    // For a single-element dataset we emit a Scalar of the appropriate class.
    if total == 1 {
        return Ok(MatValue::Scalar(read_scalar(ds, class)?));
    }

    // Read all elements as the MATLAB class's native type, then apply the shape
    // dispatch shared with the reuse path in `read_dataset`.
    let flat = read_all_elements(ds, class)?;
    numeric_value_from_flat(flat, shape)
}

/// Build a numeric [`MatValue`] from an already-decoded column-major flat
/// vector and the dataset's HDF5 shape. Shared by [`read_numeric`] and the
/// uint32 object-reference probe reuse path so the two cannot diverge: a single
/// element becomes a `Scalar`, a 1-D shape a `Vec1D`, and a 2-D shape a
/// row-major `Matrix` (skipping the no-op transpose when a dimension is 1).
fn numeric_value_from_flat(flat: NumVec, shape: &[u64]) -> Result<MatValue, MatError> {
    let (rows, cols, total) = shape_decomposition(shape)?;

    if total == 1 {
        return flat
            .get(0)
            .map(MatValue::Scalar)
            .ok_or_else(|| MatError::Custom("single-element dataset had no element".into()));
    }

    // A 1-D HDF5 dataset (no recorded cols/rows split) is treated as a flat
    // Vec1D. Files produced by MATLAB/this library are always 2-D, but some
    // external tools write true 1-D shapes.
    if shape.len() <= 1 {
        return Ok(MatValue::Vec1D(flat));
    }

    // 2-D dataset. Preserve the MATLAB [rows, cols] shape even when one dim
    // is 1 — Matrix<T> needs this to distinguish row vs column vectors. The
    // deserializer flattens to a plain sequence for Vec<T> callers.
    // Skip the transpose when one dim is 1: column-major and row-major
    // orderings are identical for 1×N / N×1, so the call would be a no-op
    // copy.
    let matrix = if rows == 1 || cols == 1 {
        flat
    } else {
        transpose_col_major_to_row_major(flat, rows, cols)?
    };
    Ok(MatValue::Matrix {
        rows,
        cols,
        vec: matrix,
    })
}

fn shape_decomposition(shape: &[u64]) -> Result<(usize, usize, usize), MatError> {
    // HDF5 shape for MATLAB data is [cols, rows] (column-major storage of a
    // [rows, cols] matrix). For a 1-D variant like [1, N] or [N, 1], treat as
    // a vector of length N. The dimensions come from the file, so each is
    // narrowed to `usize` through the checked conversion that errors (rather
    // than truncating) if a dimension exceeds the platform's pointer width,
    // and the product is taken with `checked_mul` for the same reason: a
    // declared shape whose product wraps would otherwise yield an element
    // count far below the largest index the transpose derives from `rows`.
    Ok(match shape.len() {
        0 => (1, 1, 1),
        1 => {
            let n = shape[0].to_usize()?;
            (1, n, n)
        }
        2 => {
            let cols_hdf5 = shape[0].to_usize()?;
            let rows_hdf5 = shape[1].to_usize()?;
            // MATLAB matrix has rows = rows_hdf5, cols = cols_hdf5.
            let total = checked_element_count(cols_hdf5, rows_hdf5, shape)?;
            (rows_hdf5, cols_hdf5, total)
        }
        _ => {
            let mut total: usize = 1;
            for &d in shape {
                total = checked_element_count(total, d.to_usize()?, shape)?;
            }
            (1, total, total)
        }
    })
}

/// Multiply two file-derived extents, refusing a product that would wrap.
fn checked_element_count(a: usize, b: usize, shape: &[u64]) -> Result<usize, MatError> {
    a.checked_mul(b).ok_or_else(|| {
        MatError::Custom(format!(
            "shape {shape:?} declares more elements than the address space holds"
        ))
    })
}

fn read_all_elements(ds: &Dataset, class: MatClass) -> Result<NumVec, MatError> {
    Ok(match class {
        MatClass::Double => NumVec::F64(ds.read_f64().map_err(MatError::Hdf5)?),
        MatClass::Single => NumVec::F32(ds.read_f32().map_err(MatError::Hdf5)?),
        MatClass::Int8 => NumVec::I8(ds.read_i8().map_err(MatError::Hdf5)?),
        MatClass::Int16 => NumVec::I16(ds.read_i16().map_err(MatError::Hdf5)?),
        MatClass::Int32 => NumVec::I32(ds.read_i32().map_err(MatError::Hdf5)?),
        MatClass::Int64 => NumVec::I64(ds.read_i64().map_err(MatError::Hdf5)?),
        MatClass::UInt8 => NumVec::U8(ds.read_u8().map_err(MatError::Hdf5)?),
        MatClass::UInt16 => NumVec::U16(ds.read_u16().map_err(MatError::Hdf5)?),
        MatClass::UInt32 => NumVec::U32(ds.read_u32().map_err(MatError::Hdf5)?),
        MatClass::UInt64 => NumVec::U64(ds.read_u64().map_err(MatError::Hdf5)?),
        MatClass::Logical => {
            let bytes = ds.read_u8().map_err(MatError::Hdf5)?;
            NumVec::Bool(bytes.into_iter().map(|b| b != 0).collect())
        }
        _ => return Err(MatError::Custom(format!("read_numeric: class {class:?}"))),
    })
}

fn read_scalar(ds: &Dataset, class: MatClass) -> Result<ScalarNum, MatError> {
    Ok(match class {
        MatClass::Double => ScalarNum::F64(ds.read_f64().map_err(MatError::Hdf5)?[0]),
        MatClass::Single => ScalarNum::F32(ds.read_f32().map_err(MatError::Hdf5)?[0]),
        MatClass::Int8 => ScalarNum::I8(ds.read_i8().map_err(MatError::Hdf5)?[0]),
        MatClass::Int16 => ScalarNum::I16(ds.read_i16().map_err(MatError::Hdf5)?[0]),
        MatClass::Int32 => ScalarNum::I32(ds.read_i32().map_err(MatError::Hdf5)?[0]),
        MatClass::Int64 => ScalarNum::I64(ds.read_i64().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt8 => ScalarNum::U8(ds.read_u8().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt16 => ScalarNum::U16(ds.read_u16().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt32 => ScalarNum::U32(ds.read_u32().map_err(MatError::Hdf5)?[0]),
        MatClass::UInt64 => ScalarNum::U64(ds.read_u64().map_err(MatError::Hdf5)?[0]),
        MatClass::Logical => ScalarNum::Bool(ds.read_u8().map_err(MatError::Hdf5)?[0] != 0),
        _ => return Err(MatError::Custom(format!("read_scalar: class {class:?}"))),
    })
}

fn transpose_col_major_to_row_major(
    col_major: NumVec,
    rows: usize,
    cols: usize,
) -> Result<NumVec, MatError> {
    debug_assert_eq!(col_major.len(), rows * cols);

    fn transpose<T: Copy>(v: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                out.push(v[c * rows + r]);
            }
        }
        out
    }

    Ok(match col_major {
        NumVec::F64(v) => NumVec::F64(transpose(v, rows, cols)),
        NumVec::F32(v) => NumVec::F32(transpose(v, rows, cols)),
        NumVec::I8(v) => NumVec::I8(transpose(v, rows, cols)),
        NumVec::I16(v) => NumVec::I16(transpose(v, rows, cols)),
        NumVec::I32(v) => NumVec::I32(transpose(v, rows, cols)),
        NumVec::I64(v) => NumVec::I64(transpose(v, rows, cols)),
        NumVec::U8(v) => NumVec::U8(transpose(v, rows, cols)),
        NumVec::U16(v) => NumVec::U16(transpose(v, rows, cols)),
        NumVec::U32(v) => NumVec::U32(transpose(v, rows, cols)),
        NumVec::U64(v) => NumVec::U64(transpose(v, rows, cols)),
        NumVec::Bool(v) => NumVec::Bool(transpose(v, rows, cols)),
    })
}

// ---------------------------------------------------------------------------
// Complex reading
// ---------------------------------------------------------------------------

fn read_complex(
    ds: &Dataset,
    shape: &[u64],
    class: MatClass,
    dtype: &DType,
) -> Result<MatValue, MatError> {
    let (rows, cols, total) = shape_decomposition(shape)?;
    let Some(tag) = ComplexTag::from_class(class) else {
        return Err(MatError::Custom(format!(
            "complex {{real, imag}} compound on class {}, which has no complex form",
            class.as_str()
        )));
    };
    validate_complex_dtype(dtype, tag)?;
    let bytes = ds.read_raw().map_err(MatError::Hdf5)?;
    let pairs = parse_complex_vec(&bytes, total, tag)?;

    if total == 1 {
        // `get` is in bounds: `parse_complex_vec` returned `total` pairs.
        return Ok(MatValue::ComplexScalar(pairs.get(0).expect("one pair")));
    }
    // Preserve the MATLAB [rows, cols] shape even when one dim is 1, so
    // `Matrix<Complex*>` round-trips as a row vs column vector. The
    // deserializer flattens to a plain sequence for `Vec<Complex*>` callers.
    let row_major = if rows == 1 || cols == 1 {
        pairs
    } else {
        // Stored column-major: transposing `[cols, rows]` yields row-major.
        pairs.transposed(cols, rows)
    };
    Ok(MatValue::ComplexMatrix {
        rows,
        cols,
        pairs: row_major,
    })
}

/// Decode `count` `{real, imag}` pairs of the component class `tag` from the
/// dataset's raw element bytes.
fn parse_complex_vec(bytes: &[u8], count: usize, tag: ComplexTag) -> Result<ComplexVec, MatError> {
    macro_rules! arms {
        ($($variant:ident => $ty:ty),* $(,)?) => {
            match tag {
                $(ComplexTag::$variant => {
                    ComplexVec::$variant(parse_complex_pairs::<$ty>(bytes, count, tag)?)
                })*
            }
        };
    }
    Ok(arms! {
        F64 => f64,
        F32 => f32,
        I64 => i64,
        I32 => i32,
        I16 => i16,
        I8 => i8,
        U64 => u64,
        U32 => u32,
        U16 => u16,
        U8 => u8,
    })
}

fn parse_complex_pairs<T: ComplexComponent>(
    bytes: &[u8],
    count: usize,
    tag: ComplexTag,
) -> Result<Vec<(T, T)>, MatError> {
    let component = size_of::<T>();
    let element = component * 2;
    if bytes.len() < count * element {
        return Err(MatError::Custom(format!(
            "complex {} raw bytes too short: need {}, have {}",
            tag.class().as_str(),
            count * element,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * element;
        out.push((
            T::decode_le(&bytes[off..off + component]),
            T::decode_le(&bytes[off + component..off + element]),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A declared 2-D shape whose product wraps to a plausible element count.
    /// The dimensions come from the file, so this is reachable from
    /// `mat::from_bytes` on a malformed input, not just from a Rust call site.
    #[test]
    fn shape_decomposition_refuses_a_wrapping_two_dimensional_shape() {
        // Derived from `usize::MAX` rather than written out: each dimension has
        // to fit in a `usize` (or the narrowing errors first, for a different
        // reason) while the product wraps. A literal tuned to 64-bit would
        // narrow-error on the 32-bit job and pass for the wrong reason.
        let wraps_to_four = (usize::MAX / 4 + 2) as u64;
        let err = shape_decomposition(&[wraps_to_four, 4]).unwrap_err();
        assert!(
            matches!(&err, MatError::Custom(m) if m.contains("address space")),
            "unexpected error: {err:?}"
        );
    }

    /// The same rule for rank >= 3, where the product accumulates across a loop.
    #[test]
    fn shape_decomposition_refuses_a_wrapping_high_rank_shape() {
        let err = shape_decomposition(&[1 << 31, 1 << 31, 1 << 31, 4]).unwrap_err();
        assert!(
            matches!(&err, MatError::Custom(m) if m.contains("address space")),
            "unexpected error: {err:?}"
        );
    }

    /// The guard must not reject shapes that genuinely fit.
    #[test]
    fn shape_decomposition_accepts_ordinary_shapes() {
        assert_eq!(shape_decomposition(&[]).unwrap(), (1, 1, 1));
        assert_eq!(shape_decomposition(&[7]).unwrap(), (1, 7, 7));
        assert_eq!(shape_decomposition(&[3, 5]).unwrap(), (5, 3, 15));
        assert_eq!(shape_decomposition(&[2, 3, 4]).unwrap(), (1, 24, 24));
    }

    #[test]
    fn enum_member_names_maps_indices_to_pool() {
        let pool = vec!["enum1".to_owned(), "enum2".to_owned(), "enum3".to_owned()];
        // Repeated and out-of-order indices both resolve through the shared pool.
        let names = enum_member_names(&pool, &[0, 2, 1, 0]).unwrap();
        assert_eq!(names, ["enum1", "enum3", "enum2", "enum1"]);
    }

    #[test]
    fn enum_member_names_rejects_out_of_range_index() {
        // An index past the member pool is corruption, not a silently-empty name.
        let pool = vec!["enum1".to_owned()];
        let err = enum_member_names(&pool, &[0, 5]).unwrap_err();
        assert!(
            matches!(&err, MatError::Custom(m) if m.contains("out of range")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn enum_indices_to_usize_accepts_uint32_shapes() {
        assert_eq!(
            enum_indices_to_usize(&MatValue::Scalar(ScalarNum::U32(4))).unwrap(),
            [4]
        );
        assert_eq!(
            enum_indices_to_usize(&MatValue::Vec1D(NumVec::U32(vec![1, 2, 3]))).unwrap(),
            [1, 2, 3]
        );
        assert_eq!(
            enum_indices_to_usize(&MatValue::Matrix {
                rows: 2,
                cols: 2,
                vec: NumVec::U32(vec![0, 1, 2, 3]),
            })
            .unwrap(),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn enum_indices_to_usize_rejects_non_uint32() {
        // The dtype guard in `read_enum_indices` rejects this up front; the
        // flattener is the backstop and must not silently produce wrong indices.
        let err = enum_indices_to_usize(&MatValue::Vec1D(NumVec::F64(vec![1.0]))).unwrap_err();
        assert!(matches!(err, MatError::Custom(_)), "unexpected: {err:?}");
    }
}
