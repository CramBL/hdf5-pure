#![cfg(feature = "serde")]
//! Tests for the new `to_bytes_with_options` / `to_file_with_options` API.

use hdf5_pure::mat::{
    self, Compression, EmptyMarkerEncoding, EmptySequencePolicy, InvalidNamePolicy,
    OneDimensionalMode, Options, StringClass,
};
use hdf5_pure::{AttrValue, File, LibVer};
use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Doc {
    name: String,
    score: f64,
}

#[test]
fn default_options_produce_char_strings() {
    let doc = Doc {
        name: "alice".into(),
        score: 9.5,
    };
    let bytes = mat::to_bytes_with_options(&doc, &Options::default()).unwrap();
    let path = temp_path("default-char");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "name"), "char");
    assert_eq!(read_class(&f, "score"), "double");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn string_class_option_produces_string_objects() {
    let doc = Doc {
        name: "alice".into(),
        score: 9.5,
    };
    let mut opts = Options::default();
    opts.string_class = StringClass::String;
    let bytes = mat::to_bytes_with_options(&doc, &opts).unwrap();
    let path = temp_path("string-class");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "name"), "string");
    let sub = f.dataset("#subsystem#/MCOS").unwrap();
    let sub_attrs = sub.attrs().unwrap();
    let sub_class = match &sub_attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        _ => panic!(),
    };
    assert_eq!(sub_class, "FileWrapper__");
    std::fs::remove_file(path).unwrap();
}

#[derive(Serialize)]
struct WithKeyword {
    end: u32,
}

#[test]
fn sanitize_policy_rewrites_keywords() {
    let doc = WithKeyword { end: 5 };
    let mut opts = Options::default();
    opts.invalid_name_policy = InvalidNamePolicy::Sanitize;
    let bytes = mat::to_bytes_with_options(&doc, &opts).unwrap();
    let path = temp_path("sanitize");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    assert_eq!(read_class(&f, "end_"), "uint32");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn error_policy_rejects_keywords() {
    let doc = WithKeyword { end: 5 };
    let mut opts = Options::default();
    opts.invalid_name_policy = InvalidNamePolicy::Error;
    let err = mat::to_bytes_with_options(&doc, &opts).unwrap_err();
    assert!(err.to_string().contains("invalid MATLAB name"));
}

#[derive(Serialize)]
struct Big {
    payload: Vec<f64>,
}

#[test]
fn deflate_compression_shrinks_repetitive_data() {
    // 1 MB of zeros should compress dramatically.
    let doc = Big {
        payload: vec![0.0; 128 * 1024],
    };
    // Compression needs chunked storage, whose chunk indices need the HDF5 1.10
    // format, so both sides ask for it. The MAT default is the 1.8 format
    // (MATLAB linked HDF5 1.8.12 before R2021b) and refuses compression by name;
    // comparing a 1.10 file against a 1.8 one would also be measuring the
    // superblock rather than the deflate.
    let mut plain_opts = Options::default();
    plain_opts.libver = LibVer::V110;
    let plain = mat::to_bytes_with_options(&doc, &plain_opts).unwrap();
    let mut opts = plain_opts.clone();
    opts.compression = Compression::Deflate {
        level: 6,
        shuffle: true,
    };
    let compressed = mat::to_bytes_with_options(&doc, &opts).unwrap();
    assert!(
        compressed.len() < plain.len() / 2,
        "compressed {} not less than half of plain {}",
        compressed.len(),
        plain.len()
    );
}

#[test]
fn data_as_dims_empty_marker_encoding() {
    #[derive(Serialize)]
    struct OnlyEmpty {
        v: Vec<f64>,
    }
    let doc = OnlyEmpty { v: Vec::new() };
    let mut opts = Options::default();
    opts.empty_marker_encoding = EmptyMarkerEncoding::DataAsDims;
    let bytes = mat::to_bytes_with_options(&doc, &opts).unwrap();
    let path = temp_path("data-as-dims");
    std::fs::write(&path, &bytes).unwrap();
    let f = File::open(&path).unwrap();
    let ds = f.dataset("v").unwrap();
    let attrs = ds.attrs().unwrap();
    let empty = match &attrs["MATLAB_empty"] {
        AttrValue::U32(v) => *v as u64,
        AttrValue::U64(v) => *v,
        AttrValue::I32(v) => *v as u64,
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(empty, 1);
    // The data-as-dims encoding stores the dimension vector as the data. An
    // empty vector is `0x0` — MATLAB's `[]` — under either 1-D mode, because the
    // mode orients a vector and an empty one has no orientation to preserve.
    // Counted across the MATLAB-authored fixtures, `double` empties are `0x0`
    // 47 times against `0x1` once.
    let data = ds.read_u64().unwrap();
    assert_eq!(data, vec![0, 0]);
    std::fs::remove_file(path).unwrap();
}

/// The 1-D mode does not change an empty value's shape, so asking for the other
/// mode cannot resurrect the `0x1` this used to write.
#[test]
fn the_one_dimensional_mode_does_not_orient_an_empty_value() {
    #[derive(Serialize)]
    struct OnlyEmpty {
        v: Vec<f64>,
    }
    for mode in [
        OneDimensionalMode::ColumnVector,
        OneDimensionalMode::RowVector,
    ] {
        let mut opts = Options::default();
        opts.one_dimensional_mode = mode;
        let bytes = mat::to_bytes_with_options(&OnlyEmpty { v: Vec::new() }, &opts).unwrap();
        let f = File::from_bytes(bytes).unwrap();
        assert_eq!(
            f.dataset("v").unwrap().read_u64().unwrap(),
            vec![0, 0],
            "{mode:?} must still describe the empty value as 0x0"
        );
    }
}

/// A cell array is a 1-D value, so the mode orients it exactly as it orients a
/// numeric vector in the same file. It used to be `Nx1` under both modes, from a
/// shape rule of its own that the mode never reached.
#[test]
fn the_one_dimensional_mode_orients_a_cell_array() {
    #[derive(Serialize)]
    struct Doc {
        nums: Vec<f64>,
        cells: Vec<Vec<i32>>,
    }
    // Ragged on purpose: equal-length inner vectors unify to a *matrix*, whose
    // shape is 2-D and has no orientation to take, so a rectangular fixture
    // would measure the numeric path twice and never reach a cell at all. The
    // class assertion below is what holds that.
    let doc = Doc {
        nums: vec![1.0, 2.0, 3.0],
        cells: vec![vec![1], vec![2, 3], vec![4, 5, 6]],
    };

    // HDF5 storage shape is the reverse of the MATLAB shape, so a MATLAB `3x1`
    // column is stored `[1, 3]` and a `1x3` row is stored `[3, 1]`. Asserting
    // both datasets rather than only their agreement is what keeps this from
    // passing on two shapes that are equally wrong.
    for (mode, storage) in [
        (OneDimensionalMode::ColumnVector, vec![1u64, 3]),
        (OneDimensionalMode::RowVector, vec![3u64, 1]),
    ] {
        let mut opts = Options::default();
        opts.one_dimensional_mode = mode;
        let f = File::from_bytes(mat::to_bytes_with_options(&doc, &opts).unwrap()).unwrap();
        assert_eq!(
            read_class(&f, "cells"),
            "cell",
            "{mode:?}: not a cell array"
        );
        assert_eq!(
            f.dataset("nums").unwrap().shape().unwrap(),
            storage,
            "{mode:?}: numeric vector"
        );
        assert_eq!(
            f.dataset("cells").unwrap().shape().unwrap(),
            storage,
            "{mode:?}: cell array"
        );
    }

    // And an empty cell has no orientation to take, under either mode, the same
    // rule `vector_dims` applies to every other empty.
    #[derive(Serialize)]
    struct Empty {
        cells: Vec<Vec<i32>>,
    }
    for mode in [
        OneDimensionalMode::ColumnVector,
        OneDimensionalMode::RowVector,
    ] {
        let mut opts = Options::default();
        opts.one_dimensional_mode = mode;
        opts.empty_sequence_policy = EmptySequencePolicy::Cell;
        let f = File::from_bytes(
            mat::to_bytes_with_options(&Empty { cells: Vec::new() }, &opts).unwrap(),
        )
        .unwrap();
        assert_eq!(
            f.dataset("cells").unwrap().read_u64().unwrap(),
            vec![0, 0],
            "{mode:?}: empty cell is 0x0"
        );
    }
}
