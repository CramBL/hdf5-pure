//! Reading MATLAB MCOS opaque value classes (`datetime`, `duration`,
//! `categorical`) and the lossless `Opaque` fallback for classes without a
//! dedicated decoder.
//!
//! There is no MATLAB available in CI, so these fixtures are assembled in Rust
//! to the documented `#subsystem#/FileWrapper__` byte layout (cross-validated
//! against the `matio`, `MatFileHandler`, and `foreverallama` parsers and
//! against this crate's own writer). They exercise the full read path: opaque
//! parent metadata -> object/class/property tables -> heap-cell resolution ->
//! per-class decode. The structural parser is additionally checked against the
//! production string writer's blob in `mcos_reader`'s unit tests, which is
//! real-MATLAB-derived rather than self-generated.

#![cfg(feature = "serde")]

use hdf5_pure::mat::{self, MatCategorical, MatDatetime, MatDuration};
use hdf5_pure::{AttrValue, DatasetBuilder, FileBuilder};
use serde::Deserialize;

const MCOS_MAGIC: u32 = 0xDD00_0000;
const FT_HEAP: u32 = 1;
const FT_INLINE: u32 = 2;

/// The contents of one MCOS heap cell or auxiliary `#refs#` dataset.
enum Cell {
    /// The FileWrapper metadata blob (a `uint8` array).
    Blob(Vec<u8>),
    /// A reserved/placeholder cell, never read.
    Empty,
    /// A complex `double` array (`(re, im)` pairs).
    ComplexF64(Vec<(f64, f64)>),
    /// A real `double` array.
    F64(Vec<f64>),
    /// A `uint8` integer array.
    U8(Vec<u8>),
    /// A `char` string.
    Char(String),
    /// A cell array of object references (by `#refs#` path).
    Refs(Vec<String>),
}

fn write_cell(d: &mut DatasetBuilder, cell: &Cell) {
    let ascii = |s: &str| AttrValue::AsciiString(s.to_owned());
    match cell {
        Cell::Blob(b) => {
            d.with_u8_data(b).with_shape(&[1, b.len() as u64]);
            d.set_attr("MATLAB_class", ascii("uint8"));
        }
        Cell::Empty => {
            d.with_u8_data(&[0]).with_shape(&[1, 1]);
            d.set_attr("MATLAB_class", ascii("uint8"));
        }
        Cell::ComplexF64(p) => {
            d.with_complex64_data(p).with_shape(&[1, p.len() as u64]);
            d.set_attr("MATLAB_class", ascii("double"));
        }
        Cell::F64(v) => {
            d.with_f64_data(v).with_shape(&[1, v.len() as u64]);
            d.set_attr("MATLAB_class", ascii("double"));
        }
        Cell::U8(v) => {
            d.with_u8_data(v).with_shape(&[1, v.len() as u64]);
            d.set_attr("MATLAB_class", ascii("uint8"));
        }
        Cell::Char(s) => {
            let units: Vec<u16> = s.encode_utf16().collect();
            d.with_u16_data(&units).with_shape(&[1, units.len() as u64]);
            d.set_attr("MATLAB_class", ascii("char"));
        }
        Cell::Refs(paths) => {
            let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
            d.with_path_references(&refs)
                .with_shape(&[1, refs.len() as u64]);
            d.set_attr("MATLAB_class", ascii("cell"));
        }
    }
}

/// Pack a property block: `[nprops, (name_idx, field_type, value)…]` padded to
/// an 8-byte (two-`u32`) boundary.
fn prop_block(triples: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut w = vec![triples.len() as u32];
    for &(name, ftype, value) in triples {
        w.extend_from_slice(&[name, ftype, value]);
    }
    if w.len() % 2 != 0 {
        w.push(0);
    }
    w
}

fn le32(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Build a FileWrapper metadata blob to the documented layout. The caller
/// supplies only the real (1-based) entries; the reserved leading class entry,
/// object entry, and empty property block 0 are prepended automatically.
fn filewrapper_blob(
    names: &[&str],
    classes: &[(u32, u32)],      // (namespace_idx, name_idx)
    objects: &[(u32, u32, u32)], // (class_id, saveobj_id, normalobj_id)
    seg1: &[&[(u32, u32, u32)]], // type-1 property blocks
    seg2: &[&[(u32, u32, u32)]], // type-2 property blocks
) -> Vec<u8> {
    let mut names_b = Vec::new();
    for n in names {
        names_b.extend_from_slice(n.as_bytes());
        names_b.push(0);
    }
    while names_b.len() % 8 != 0 {
        names_b.push(0);
    }

    let mut class_w = vec![0u32; 4];
    for &(ns, nm) in classes {
        class_w.extend_from_slice(&[ns, nm, 0, 0]);
    }

    let mut object_w = vec![0u32; 6];
    for &(cid, so, no) in objects {
        object_w.extend_from_slice(&[cid, 0, 0, so, no, 0]);
    }

    let seg_words = |blocks: &[&[(u32, u32, u32)]]| -> Vec<u32> {
        let mut w = prop_block(&[]); // reserved empty block 0
        for b in blocks {
            w.extend(prop_block(b));
        }
        w
    };
    let seg1_b = le32(&seg_words(seg1));
    let seg2_b = le32(&seg_words(seg2));
    let dyn_b = le32(&prop_block(&[])); // empty dynamic-property table
    let class_b = le32(&class_w);
    let object_b = le32(&object_w);

    let mut off = (40 + names_b.len()) as u32;
    let mut offsets = [0u32; 8];
    offsets[0] = off;
    off += class_b.len() as u32;
    offsets[1] = off;
    off += seg1_b.len() as u32;
    offsets[2] = off;
    off += object_b.len() as u32;
    offsets[3] = off;
    off += seg2_b.len() as u32;
    offsets[4] = off;
    off += dyn_b.len() as u32;
    offsets[5] = off;
    offsets[6] = off;
    offsets[7] = off;

    let mut blob = Vec::new();
    blob.extend_from_slice(&4u32.to_le_bytes()); // version
    blob.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for o in offsets {
        blob.extend_from_slice(&o.to_le_bytes());
    }
    blob.extend_from_slice(&names_b);
    blob.extend_from_slice(&class_b);
    blob.extend_from_slice(&seg1_b);
    blob.extend_from_slice(&object_b);
    blob.extend_from_slice(&seg2_b);
    blob.extend_from_slice(&dyn_b);
    blob
}

/// Parent-dataset opaque metadata for one scalar object:
/// `[MAGIC, ndims=2, 1, 1, object_id, class_id]`.
fn opaque_meta(object_id: u32, class_id: u32) -> Vec<u32> {
    vec![MCOS_MAGIC, 2, 1, 1, object_id, class_id]
}

/// Assemble a `.mat` file with one opaque variable. `heap` are the MCOS cells in
/// order from cell 0 (the blob); `aux` are extra `#refs#` datasets reachable by
/// reference (e.g. a categorical's category strings) but not in the MCOS array.
fn build_mat(
    var: &str,
    class_name: &str,
    meta: &[u32],
    heap: Vec<(&str, Cell)>,
    aux: Vec<(&str, Cell)>,
) -> Vec<u8> {
    let mut fb = FileBuilder::new();

    let mut refs = fb.create_group("#refs#");
    let mut mcos_paths = Vec::new();
    for (name, cell) in heap {
        let d = refs.create_dataset(name);
        write_cell(d, &cell);
        mcos_paths.push(format!("#refs#/{name}"));
    }
    for (name, cell) in aux {
        let d = refs.create_dataset(name);
        write_cell(d, &cell);
    }
    fb.add_group(refs.finish());

    let mut sub = fb.create_group("#subsystem#");
    {
        let path_refs: Vec<&str> = mcos_paths.iter().map(String::as_str).collect();
        let d = sub.create_dataset("MCOS");
        d.with_path_references(&path_refs)
            .with_shape(&[1, path_refs.len() as u64]);
        d.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString("FileWrapper__".into()),
        );
        d.set_attr("MATLAB_object_decode", AttrValue::I32(3));
    }
    fb.add_group(sub.finish());

    {
        let d = fb.create_dataset(var);
        d.with_u32_data(meta).with_shape(&[1, meta.len() as u64]);
        d.set_attr("MATLAB_class", AttrValue::AsciiString(class_name.into()));
        d.set_attr("MATLAB_object_decode", AttrValue::I32(3));
    }

    fb.finish().expect("assemble fixture .mat")
}

#[test]
fn datetime_decodes_millis_and_sub_ms() {
    // datetime with one object whose `data` property is a 1x2 complex double:
    // 1000 ms (whole) and 2000 ms + 0.5 ms sub-millisecond.
    let blob = filewrapper_blob(
        &["datetime", "data"],
        &[(0, 1)],             // class 1 = "datetime"
        &[(1, 0, 1)],          // object 1: class 1, normal-segment block 1
        &[],                   // no saveobj blocks
        &[&[(2, FT_HEAP, 0)]], // block 1: data -> heap cell 0 (= MCOS cell 2)
    );
    let meta = opaque_meta(1, 1);
    let bytes = build_mat(
        "ts",
        "datetime",
        &meta,
        vec![
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("data", Cell::ComplexF64(vec![(1000.0, 0.0), (2000.0, 0.5)])),
        ],
        vec![],
    );

    #[derive(Deserialize)]
    struct Doc {
        ts: MatDatetime,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode datetime");
    assert_eq!(doc.ts.millis_utc, vec![1000.0, 2000.0]);
    assert_eq!(doc.ts.sub_ms, vec![0.0, 0.5]);
    assert_eq!(doc.ts.nanoseconds(), vec![1_000_000_000.0, 2_000_500_000.0]);
}

#[test]
fn datetime_carries_format_string() {
    let blob = filewrapper_blob(
        &["datetime", "data", "fmt"],
        &[(0, 1)],
        &[(1, 0, 1)],
        &[],
        &[&[(2, FT_HEAP, 0), (3, FT_HEAP, 1)]], // data -> cell 2, fmt -> cell 3
    );
    let meta = opaque_meta(1, 1);
    let bytes = build_mat(
        "t",
        "datetime",
        &meta,
        vec![
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("data", Cell::ComplexF64(vec![(0.0, 0.0)])),
            ("fmt", Cell::Char("uuuu-MM-dd".to_owned())),
        ],
        vec![],
    );

    #[derive(Deserialize)]
    struct Doc {
        t: MatDatetime,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode datetime with fmt");
    assert_eq!(doc.t.millis_utc, vec![0.0]);
    assert_eq!(doc.t.fmt.as_deref(), Some("uuuu-MM-dd"));
}

#[test]
fn duration_decodes_milliseconds() {
    let blob = filewrapper_blob(
        &["duration", "millis"],
        &[(0, 1)],
        &[(1, 0, 1)],
        &[],
        &[&[(2, FT_HEAP, 0)]],
    );
    let meta = opaque_meta(1, 1);
    let bytes = build_mat(
        "elapsed",
        "duration",
        &meta,
        vec![
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("millis", Cell::F64(vec![1000.0, 2000.0, 60_000.0])),
        ],
        vec![],
    );

    #[derive(Deserialize)]
    struct Doc {
        elapsed: MatDuration,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode duration");
    assert_eq!(doc.elapsed.millis, vec![1000.0, 2000.0, 60_000.0]);
    assert_eq!(doc.elapsed.seconds(), vec![1.0, 2.0, 60.0]);
}

#[test]
fn categorical_decodes_codes_categories_and_flags() {
    // codes 1-based with a 0 = <undefined>; two categories; ordinal, unprotected.
    let blob = filewrapper_blob(
        &[
            "categorical",
            "codes",
            "categoryNames",
            "isOrdinal",
            "isProtected",
        ],
        &[(0, 1)],
        &[(1, 0, 1)],
        &[],
        &[&[
            (2, FT_HEAP, 0),   // codes -> cell 2
            (3, FT_HEAP, 1),   // categoryNames -> cell 3
            (4, FT_INLINE, 1), // isOrdinal = true
            (5, FT_INLINE, 0), // isProtected = false
        ]],
    );
    let meta = opaque_meta(1, 1);
    let bytes = build_mat(
        "grade",
        "categorical",
        &meta,
        vec![
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("codes", Cell::U8(vec![1, 2, 1, 0])),
            (
                "catnames",
                Cell::Refs(vec!["#refs#/cat0".into(), "#refs#/cat1".into()]),
            ),
        ],
        vec![
            ("cat0", Cell::Char("Low".to_owned())),
            ("cat1", Cell::Char("High".to_owned())),
        ],
    );

    #[derive(Deserialize)]
    struct Doc {
        grade: MatCategorical,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode categorical");
    assert_eq!(doc.grade.codes, vec![1, 2, 1, 0]);
    assert_eq!(
        doc.grade.categories,
        vec!["Low".to_owned(), "High".to_owned()]
    );
    assert!(doc.grade.is_ordinal);
    assert!(!doc.grade.is_protected);
    assert_eq!(
        doc.grade.labels(),
        vec![
            Some("Low".to_owned()),
            Some("High".to_owned()),
            Some("Low".to_owned()),
            None,
        ]
    );
}

#[test]
fn unknown_opaque_class_is_lossless() {
    // A class without a dedicated decoder (here a namespaced `containers.Map`)
    // surfaces its raw properties so it still deserializes as a struct.
    let blob = filewrapper_blob(
        &["containers", "Map", "value"],
        &[(1, 2)], // class 1 = namespace "containers" + name "Map"
        &[(1, 0, 1)],
        &[],
        &[&[(3, FT_HEAP, 0)]], // value -> cell 2
    );
    let meta = opaque_meta(1, 1);
    let bytes = build_mat(
        "m",
        "containers.Map",
        &meta,
        vec![
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("value", Cell::F64(vec![42.0])),
        ],
        vec![],
    );

    #[derive(Deserialize)]
    struct Inner {
        value: f64,
    }
    #[derive(Deserialize)]
    struct Doc {
        m: Inner,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode unknown opaque as struct");
    assert_eq!(doc.m.value, 42.0);
}

#[test]
fn dangling_heap_reference_is_an_error_not_a_panic() {
    // A property points at heap value 5 (MCOS cell 7), which does not exist.
    let blob = filewrapper_blob(
        &["datetime", "data"],
        &[(0, 1)],
        &[(1, 0, 1)],
        &[],
        &[&[(2, FT_HEAP, 5)]],
    );
    let meta = opaque_meta(1, 1);
    let bytes = build_mat(
        "ts",
        "datetime",
        &meta,
        vec![("blob", Cell::Blob(blob)), ("canon", Cell::Empty)],
        vec![],
    );

    #[derive(Deserialize)]
    struct Doc {
        #[allow(dead_code)]
        ts: MatDatetime,
    }
    let result: Result<Doc, _> = mat::from_bytes(&bytes);
    assert!(result.is_err(), "dangling heap reference must error");
}
