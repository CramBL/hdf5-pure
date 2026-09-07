// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Attributes in the standard variable-length string datatype (issue #383).
//!
//! `H5T_STRING` with `STRSIZE = H5T_VARIABLE` is what h5py, the reference C
//! library and this crate's own `DatasetBuilder::with_vlen_strings` write, and
//! before `AttrValue::VarLenString` and its three siblings no `AttrValue`
//! originated it: the only variable-length attribute variant was
//! `VarLenAsciiCharArray`, which writes MATLAB's `H5T_VLEN { H5T_STRING { STRSIZE
//! = 1 } }` sequence instead. (`repack` already carried a C-written one across
//! verbatim; nothing could write one from a value.) The element bytes are the
//! same either way, so every assertion here is about the datatype the reader is
//! handed — which is what decides whether the C library can convert it at all.

use hdf5::types::{TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5_pure::{AttrValue, File, FileBuilder};
use tempfile::tempdir;

/// The four values under test, one per variant, keyed by the attribute name
/// each is written under.
fn cases() -> Vec<(&'static str, AttrValue)> {
    vec![
        ("utf8_scalar", AttrValue::VarLenString("mètre".into())),
        (
            "utf8_array",
            AttrValue::VarLenStringArray(vec!["m/s".into(), "kg".into()]),
        ),
        (
            "ascii_scalar",
            AttrValue::VarLenAsciiString("double".into()),
        ),
        (
            "ascii_array",
            AttrValue::VarLenAsciiStringArray(vec!["x".into(), "yy".into()]),
        ),
    ]
}

/// A file whose root group and whose dataset `d` both carry every case.
fn write_cases(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    for (name, value) in cases() {
        b.set_attr(name, value);
    }
    {
        let d = b.create_dataset("d");
        d.with_f64_data(&[1.0]);
        for (name, value) in cases() {
            d.set_attr(name, value);
        }
    }
    b.write(path).unwrap();
}

/// What the C library makes of each attribute of `object`: the conversion the
/// issue reported failing (`read_raw::<VarLenAscii>`, `read_scalar`) and the
/// datatype descriptor it reports.
fn assert_c_reads_the_cases(object: &hdf5::Group, owner: &str) {
    let scalar_utf8 = object.attr("utf8_scalar").unwrap();
    assert_eq!(
        scalar_utf8.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::VarLenUnicode,
        "{owner}: a UTF-8 variable-length string attribute"
    );
    assert_eq!(
        scalar_utf8.ndim(),
        0,
        "{owner}: the scalar variant writes a scalar dataspace, not a 1-element array"
    );
    assert_eq!(
        scalar_utf8.read_scalar::<VarLenUnicode>().unwrap().as_str(),
        "mètre"
    );

    let array_utf8 = object.attr("utf8_array").unwrap();
    assert_eq!(
        array_utf8.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::VarLenUnicode
    );
    let got: Vec<VarLenUnicode> = array_utf8.read_raw().unwrap();
    assert_eq!(
        got.iter().map(VarLenUnicode::as_str).collect::<Vec<_>>(),
        ["m/s", "kg"],
        "{owner}: a UTF-8 variable-length string array"
    );

    let scalar_ascii = object.attr("ascii_scalar").unwrap();
    assert_eq!(
        scalar_ascii.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::VarLenAscii,
        "{owner}: the ASCII charset is on the datatype, not inferred from the bytes"
    );
    assert_eq!(
        scalar_ascii.read_scalar::<VarLenAscii>().unwrap().as_str(),
        "double"
    );

    let array_ascii = object.attr("ascii_array").unwrap();
    let got: Vec<VarLenAscii> = array_ascii.read_raw().unwrap();
    assert_eq!(
        got.iter().map(VarLenAscii::as_str).collect::<Vec<_>>(),
        ["x", "yy"],
        "{owner}: an ASCII variable-length string array"
    );
}

/// Every case reads back as the variant it was written from, through this
/// crate's own reader.
fn assert_pure_reads_the_cases(attrs: &std::collections::HashMap<String, AttrValue>, owner: &str) {
    for (name, written) in cases() {
        assert_eq!(attrs.get(name), Some(&written), "{owner}: attribute {name}");
    }
}

/// The claim the issue was filed on: the C library converts these attributes
/// into `VarLenUnicode` and `VarLenAscii`, where the MATLAB encoding it used to
/// get instead failed with "no appropriate function for conversion path".
#[test]
fn the_c_library_reads_every_variant_as_a_variable_length_string() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vl.h5");
    write_cases(&path);

    let c = hdf5::File::open(&path).unwrap();
    assert_c_reads_the_cases(&c.as_group().unwrap(), "root group");
    // A dataset's attributes travel a different sizing path than a group's, so
    // both are written and both are read.
    let d = c.dataset("d").unwrap();
    for (name, _) in cases() {
        assert!(d.attr(name).is_ok(), "dataset attribute {name}");
    }
    assert_eq!(
        d.attr("ascii_array")
            .unwrap()
            .read_raw::<VarLenAscii>()
            .unwrap()
            .iter()
            .map(VarLenAscii::as_str)
            .collect::<Vec<_>>(),
        ["x", "yy"]
    );

    let f = File::open(&path).unwrap();
    assert_pure_reads_the_cases(&f.root().attrs().unwrap(), "root group");
    assert_pure_reads_the_cases(&f.dataset("d").unwrap().attrs().unwrap(), "dataset");
}

/// MATLAB's shape and the standard one carry the same text over the same
/// element bytes, and differ in the datatype alone — which is the whole reason
/// the standard one needed a variant of its own rather than a fix to the
/// existing encoding.
#[test]
fn the_matlab_shape_keeps_its_own_datatype() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("both.h5");
    let words = vec!["alpha".to_string(), "beta".to_string()];

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.0]);
    b.set_attr("matlab", AttrValue::VarLenAsciiCharArray(words.clone()));
    b.set_attr("standard", AttrValue::VarLenAsciiStringArray(words.clone()));
    b.write(&path).unwrap();

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.attr("standard")
            .unwrap()
            .dtype()
            .unwrap()
            .to_descriptor()
            .unwrap(),
        TypeDescriptor::VarLenAscii
    );
    // Unchanged: a VLEN sequence of one-character strings, which is what MATLAB
    // and matio expect for `MATLAB_fields` and its neighbours.
    assert_eq!(
        c.attr("matlab")
            .unwrap()
            .dtype()
            .unwrap()
            .to_descriptor()
            .unwrap(),
        TypeDescriptor::VarLenArray(Box::new(TypeDescriptor::FixedAscii(1)))
    );

    let f = File::open(&path).unwrap();
    let attrs = f.root().attrs().unwrap();
    assert_eq!(
        attrs.get("matlab"),
        Some(&AttrValue::VarLenAsciiCharArray(words.clone()))
    );
    assert_eq!(
        attrs.get("standard"),
        Some(&AttrValue::VarLenAsciiStringArray(words))
    );
}

/// An empty variable-length string is a real value, not an absent one: it takes
/// a zero-length heap object at an index of its own, where a *null* element
/// would take no object at all, and it must read back as an empty string rather
/// than dropping the attribute.
#[test]
fn an_empty_variable_length_string_round_trips() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.0]);
    b.set_attr("empty", AttrValue::VarLenString(String::new()));
    b.set_attr(
        "some_empty",
        AttrValue::VarLenStringArray(vec!["a".into(), String::new(), "c".into()]),
    );
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    let attrs = f.root().attrs().unwrap();
    assert_eq!(
        attrs.get("empty"),
        Some(&AttrValue::VarLenString(String::new()))
    );
    assert_eq!(
        attrs.get("some_empty"),
        Some(&AttrValue::VarLenStringArray(vec![
            "a".into(),
            String::new(),
            "c".into()
        ]))
    );

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.attr("empty")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .as_str(),
        ""
    );
}

/// The in-place edit engine stages a variable-length attribute through its own
/// path — the value's heap collections are placed by the commit and its
/// placeholder addresses patched afterwards — so it needs its own coverage.
#[test]
fn an_edit_session_writes_the_standard_datatype() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.0]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        let mut d = s.dataset("d").unwrap();
        for (name, value) in cases() {
            d.set_attr(name, value).unwrap();
        }
        s.commit().unwrap();
        drop(d);
        drop(s);
    }

    let f = File::open(&path).unwrap();
    assert_pure_reads_the_cases(&f.dataset("d").unwrap().attrs().unwrap(), "edited dataset");
    drop(f);

    let c = hdf5::File::open(&path).unwrap();
    let d = c.dataset("d").unwrap();
    assert_eq!(
        d.attr("utf8_scalar")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .as_str(),
        "mètre"
    );
    assert_eq!(
        d.attr("ascii_array")
            .unwrap()
            .dtype()
            .unwrap()
            .to_descriptor()
            .unwrap(),
        TypeDescriptor::VarLenAscii
    );
}

/// A dense (fractal-heap) rebuild carries the attribute's raw element bytes
/// across, so a variable-length one is the case where those bytes address a
/// global heap collection the same commit placed.
#[test]
fn a_dense_rebuild_keeps_the_standard_datatype() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.0]);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        let mut d = s.dataset("d").unwrap();
        for (name, value) in cases() {
            d.set_attr(name, value).unwrap();
        }
        // Past the eight-attribute threshold, so the whole set moves into a
        // fractal heap (#102).
        for i in 0..8 {
            d.set_attr(&format!("filler_{i}"), AttrValue::I64(i))
                .unwrap();
        }
        s.commit().unwrap();
        drop(d);
        drop(s);
    }

    let f = File::open(&path).unwrap();
    assert_pure_reads_the_cases(&f.dataset("d").unwrap().attrs().unwrap(), "dense dataset");
    drop(f);

    let c = hdf5::File::open(&path).unwrap();
    let d = c.dataset("d").unwrap();
    assert_eq!(
        d.attr("utf8_array")
            .unwrap()
            .read_raw::<VarLenUnicode>()
            .unwrap()
            .iter()
            .map(VarLenUnicode::as_str)
            .collect::<Vec<_>>(),
        ["m/s", "kg"],
        "the heap-resident attribute's element references resolve"
    );
    assert_eq!(
        d.attr("ascii_scalar")
            .unwrap()
            .dtype()
            .unwrap()
            .to_descriptor()
            .unwrap(),
        TypeDescriptor::VarLenAscii
    );
}

/// Repack restages every attribute's payload into the destination's heap while
/// keeping the source datatype, which it did before this change too. What is new
/// is what comes back out: the four variants, rather than the fixed-width ones
/// the decode used to report for them.
#[test]
fn repack_keeps_the_standard_datatype() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");
    write_cases(&src);

    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::default()).unwrap();

    let f = File::open(&dst).unwrap();
    assert_pure_reads_the_cases(&f.root().attrs().unwrap(), "repacked root");
    assert_pure_reads_the_cases(
        &f.dataset("d").unwrap().attrs().unwrap(),
        "repacked dataset",
    );
    drop(f);

    let c = hdf5::File::open(&dst).unwrap();
    assert_c_reads_the_cases(&c.as_group().unwrap(), "repacked root");
}

/// An attribute whose datatype is a *committed* one still keeps its strings in
/// the global heap, and the writer must stage them.
///
/// The committed path encodes the value and then moves its datatype out of the
/// message, which used to hand the result to the writer as already-final bytes.
/// For a variable-length value those bytes are not final: every element still
/// held the placeholder heap address 0. `write` returned `Ok` on a file whose
/// attribute this crate then dropped from `attrs` entirely and the C library
/// read as an empty string. It applies to MATLAB's shape as much as to the
/// standard one, so both are pinned here.
#[test]
fn a_committed_datatype_attribute_stages_its_heap() {
    use hdf5_pure::{CharacterSet, Datatype, DatatypeByteOrder, StringPadding};

    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    let standard = Datatype::VariableLength {
        is_string: true,
        padding: Some(StringPadding::NullTerminate),
        charset: Some(CharacterSet::Utf8),
        base_type: Box::new(Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        }),
    };
    let matlab = Datatype::VariableLength {
        is_string: false,
        padding: None,
        charset: None,
        base_type: Box::new(Datatype::String {
            size: 1,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        }),
    };

    // A fixed-width string under a committed datatype is the other side of the
    // same decision: it must keep taking the verbatim path, since staging a heap
    // for it would replace its padded bytes with heap references.
    let fixed = Datatype::String {
        size: 6,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Ascii,
    };

    let mut b = FileBuilder::new();
    b.commit_datatype("vlstr", standard);
    b.commit_datatype("vlchar", matlab);
    b.commit_datatype("class", fixed);
    b.create_dataset("d").with_f64_data(&[1.0]);
    b.set_attr_committed("class", AttrValue::AsciiString("double".into()), "/class");
    b.set_attr_committed("units", AttrValue::VarLenString("m/s".into()), "/vlstr");
    b.set_attr_committed(
        "fields",
        AttrValue::VarLenAsciiCharArray(vec!["a".into(), "bb".into()]),
        "/vlchar",
    );
    b.write(&path).unwrap();

    let f = File::open(&path).unwrap();
    let attrs = f.root().attrs().unwrap();
    assert_eq!(
        attrs.get("units"),
        Some(&AttrValue::VarLenString("m/s".into())),
        "a committed-datatype attribute must not be dropped for an unresolvable heap address"
    );
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "a".into(),
            "bb".into()
        ]))
    );
    assert_eq!(
        attrs.get("class"),
        Some(&AttrValue::AsciiString("double".into())),
        "a fixed-width committed attribute keeps its own bytes"
    );
    drop(f);

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.attr("units")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .as_str(),
        "m/s",
        "the C library resolves the heap reference the committed path staged"
    );
}
