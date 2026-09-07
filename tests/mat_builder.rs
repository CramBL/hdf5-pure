//! Direct tests for the public `mat::MatBuilder` API.

use hdf5_pure::mat::{Compression, MatBuilder, MatClass, MatError, Options, StringClass};
use hdf5_pure::{AttrValue, File, LibVer};

use test_util::temp;

/// A `.mat` fixture, in a directory that goes away with the returned value.
///
/// Replaces a nanosecond-stamped name in the shared temporary directory, which
/// no two runs collided on but no run ever cleaned up (issue #334).
fn temp_path(name: &str) -> temp::TempPath {
    temp::temp_path(&format!("{name}.mat"))
}

fn read_class(file: &File, ds_path: &str) -> String {
    let ds = file.dataset(ds_path).unwrap();
    let attrs = ds.attrs().unwrap();
    match &attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        other => panic!("unexpected class: {other:?}"),
    }
}

#[test]
fn scalar_numeric_classes() {
    let mut mb = MatBuilder::new(Options::default());
    mb.write_scalar_f64("d", 1.0).unwrap();
    mb.write_scalar_f32("s", 2.0).unwrap();
    mb.write_scalar_i32("i", -3).unwrap();
    mb.write_scalar_u8("b", 7).unwrap();
    mb.write_scalar_logical("flag", true).unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("scalars");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "d"), "double");
    assert_eq!(read_class(&f, "s"), "single");
    assert_eq!(read_class(&f, "i"), "int32");
    assert_eq!(read_class(&f, "b"), "uint8");
    assert_eq!(read_class(&f, "flag"), "logical");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn vector_round_trips_with_class() {
    let mut mb = MatBuilder::new(Options::default());
    let dims = mb.vector_dims(4);
    mb.write_f64("v", &dims, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("vec");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    let ds = f.dataset("v").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_class(&f, "v"), "double");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn struct_with_fields() {
    let mut mb = MatBuilder::new(Options::default());
    mb.struct_("payload", |s| {
        s.write_scalar_u32("answer", 7)?;
        s.write_char("label", "ready")?;
        Ok(())
    })
    .unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("struct");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    let group = f.group("payload").unwrap();
    let attrs = group.attrs().unwrap();
    let class = match &attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(class, "struct");
    let fields = attrs["MATLAB_fields"]
        .as_strings()
        .expect("MATLAB_fields must hold strings");
    assert_eq!(fields, vec!["answer", "label"]);
    assert_eq!(read_class(&f, "payload/answer"), "uint32");
    assert_eq!(read_class(&f, "payload/label"), "char");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cell_with_mixed_elements() {
    let mut mb = MatBuilder::new(Options::default());
    mb.cell("c", &[3, 1], |cw| {
        cw.push_scalar_u8(1)?;
        cw.push_scalar_f64(3.14)?;
        cw.push_char("hi")?;
        Ok(())
    })
    .unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("cell");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "c"), "cell");
    assert_eq!(read_class(&f, "#refs#/ref_0000000000000000"), "uint8");
    assert_eq!(read_class(&f, "#refs#/ref_0000000000000001"), "double");
    assert_eq!(read_class(&f, "#refs#/ref_0000000000000002"), "char");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn nested_struct_and_cell() {
    let mut mb = MatBuilder::new(Options::default());
    mb.struct_("root", |s| {
        s.cell("entries", &[2, 1], |cw| {
            cw.push_struct(|inner| {
                inner.write_scalar_u32("x", 1)?;
                Ok(())
            })?;
            cw.push_scalar_u32(2)?;
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("nested");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "root/entries"), "cell");
    // First ref is the inner struct (ref_0).
    let g = f.group("#refs#/ref_0000000000000000").unwrap();
    let attrs = g.attrs().unwrap();
    let class = match &attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        _ => panic!(),
    };
    assert_eq!(class, "struct");
    assert_eq!(read_class(&f, "#refs#/ref_0000000000000000/x"), "uint32");
    // Second ref is the scalar.
    assert_eq!(read_class(&f, "#refs#/ref_0000000000000001"), "uint32");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn string_object_emits_subsystem() {
    let mut options = Options::default();
    options.string_class = StringClass::String;
    let mut mb = MatBuilder::new(options);
    mb.write_string_object("greeting", &["hello".to_owned()], &[1, 1])
        .unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("string-obj");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "greeting"), "string");
    let ds = f.dataset("greeting").unwrap();
    let attrs = ds.attrs().unwrap();
    let decode = match &attrs["MATLAB_object_decode"] {
        AttrValue::I64(v) => *v,
        AttrValue::I32(v) => *v as i64,
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(decode, 3);
    // Subsystem must be present.
    let sub = f.dataset("#subsystem#/MCOS").unwrap();
    let sub_attrs = sub.attrs().unwrap();
    let sub_class = match &sub_attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        _ => panic!(),
    };
    assert_eq!(sub_class, "FileWrapper__");
    // 5 helpers + 1 string payload = 6 entries total.
    assert_eq!(sub.shape().unwrap(), vec![1, 6]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn name_sanitization_handles_keyword() {
    let mut options = Options::default();
    options.invalid_name_policy = hdf5_pure::mat::InvalidNamePolicy::Sanitize;
    let mut mb = MatBuilder::new(options);
    mb.struct_("payload", |s| {
        // `end` is a MATLAB keyword and must be sanitized.
        s.write_scalar_u32("end", 1)?;
        Ok(())
    })
    .unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("sanitize");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    // Sanitize appends `_` to keyword.
    assert_eq!(read_class(&f, "payload/end_"), "uint32");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn invalid_name_errors_by_default() {
    let mut mb = MatBuilder::new(Options::default());
    let err = mb.struct_("payload", |s| {
        s.write_scalar_u32("end", 1)?;
        Ok(())
    });
    assert!(err.is_err());
}

/// The default empty marker is the one MATLAB itself writes: a two-element
/// `uint64` dataset whose *payload* is the dimension vector, not a zero-element
/// dataset of that shape. Every empty in `tests/data/matlab` takes this
/// form, including the `[1 1]` and `[1 0]` ones that a zero-element encoding
/// cannot express at all.
#[test]
fn empty_marker_data_as_dims_default() {
    let mut mb = MatBuilder::new(Options::default());
    mb.write_empty("empty", MatClass::Double, &[0, 0]).unwrap();
    let bytes = mb.finish().unwrap();

    let path = temp_path("empty");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    let ds = f.dataset("empty").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2]);
    assert_eq!(ds.read_u64().unwrap(), vec![0, 0]);
    let attrs = ds.attrs().unwrap();
    let empty = match &attrs["MATLAB_empty"] {
        AttrValue::U32(v) => *v as u64,
        AttrValue::U64(v) => *v,
        AttrValue::I32(v) => *v as u64,
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(empty, 1);
    std::fs::remove_file(path).unwrap();
}

/// A `.mat` file's whole purpose is to be read by MATLAB, and MATLAB reads MAT
/// v7.3 with HDF5 1.8.12 before R2021b — a library predating superblock 3.
/// So the MAT default is the 1.8 format, which is checked here against the raw
/// bytes rather than through a library that would read either version happily.
#[test]
fn the_default_mat_file_is_in_the_1_8_format() {
    let mut mb = MatBuilder::new(Options::default());
    mb.write_f64("values", &[3, 1], &[1.0, 2.0, 3.0]).unwrap();
    let bytes = mb.finish().unwrap();

    let sig = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("the file carries an HDF5 signature");
    assert_eq!(sig, 512, "the MAT userblock precedes the superblock");
    assert_eq!(
        bytes[sig + 8],
        2,
        "a version 3 superblock is a 1.10 addition MATLAB's loader cannot read"
    );

    // Opting into 1.10 is what a caller does to get compression, and it must
    // actually change the file.
    let mut options = Options::default();
    options.libver = LibVer::V110;
    let mut mb = MatBuilder::new(options);
    mb.write_f64("values", &[3, 1], &[1.0, 2.0, 3.0]).unwrap();
    let bytes = mb.finish().unwrap();
    assert_eq!(bytes[sig + 8], 3);
}

/// Compression needs chunked storage, whose chunk indices need 1.10, so it
/// cannot be combined with the 1.8 default. Refused rather than resolved either
/// way: silently dropping compression loses what was asked for, and silently
/// raising the format produces a file MATLAB cannot load.
#[test]
fn compression_under_the_1_8_default_is_refused_by_name() {
    let mut options = Options::default();
    options.compression = Compression::Deflate {
        level: 4,
        shuffle: false,
    };
    let mut mb = MatBuilder::new(options);
    mb.write_f64("values", &[3, 1], &[1.0, 2.0, 3.0]).unwrap();
    assert!(matches!(
        mb.finish().unwrap_err(),
        MatError::CompressionNeedsNewerFormat
    ));

    // Raising the format is the documented way through, and it works.
    let mut options = Options::default();
    options.compression = Compression::Deflate {
        level: 4,
        shuffle: false,
    };
    options.libver = LibVer::V110;
    let mut mb = MatBuilder::new(options);
    mb.write_f64("values", &[3, 1], &[1.0, 2.0, 3.0]).unwrap();
    assert!(mb.finish().is_ok());
}
