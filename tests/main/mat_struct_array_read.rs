//! MATLAB struct arrays deserialized from independent HDF5 images.
//!
//! Array fields contain object references, and scalar fields contain direct
//! values. The byte builders and complete graph assertions also cover
//! column-major dimensions, nested aliases, and MATLAB attributes.
#![cfg(feature = "serde")]

use hdf5_pure::mat;
use rstest::rstest;
use serde::Deserialize;
use serde::de::IgnoredAny;
use test_util::mat_struct_array;
use test_util_hdf5::reader::ReadBackend;

#[derive(Deserialize, Debug, PartialEq)]
#[allow(non_snake_case)]
struct Data {
    fieldA: u64,
    fieldB: String,
    fieldC: Vec<i64>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct GridElem {
    id: f64,
    tag: String,
}

#[derive(Deserialize, Debug, PartialEq)]
struct Inner {
    p: f64,
}

#[derive(Deserialize, Debug, PartialEq)]
#[allow(non_snake_case)]
struct Nested {
    fieldA: f64,
    inner: Inner,
}

fn expected_data(n: u64) -> Data {
    Data {
        fieldA: n,
        fieldB: ((b'a' + (n as u8) - 1) as char).to_string(),
        fieldC: vec![-6, -5, -4, -3, -2, -1],
    }
}

// The MATLAB row has six elements.
#[test]
fn row_struct_array_reads_as_vec() {
    #[derive(Deserialize)]
    struct File {
        row: Vec<Data>,
    }
    let row = mat::from_bytes::<File>(&mat_struct_array::struct_array())
        .unwrap()
        .row;
    let expected: Vec<Data> = (1..=6).map(expected_data).collect();
    assert_eq!(row, expected);
}

// The MATLAB column has six elements.
#[test]
fn col_struct_array_reads_as_vec() {
    #[derive(Deserialize)]
    struct File {
        col: Vec<Data>,
    }
    let col = mat::from_bytes::<File>(&mat_struct_array::struct_array())
        .unwrap()
        .col;
    let expected: Vec<Data> = (1..=6).map(expected_data).collect();
    assert_eq!(col, expected);
}

// Each MATLAB row has three elements with `id = row * 10 + col`.
#[test]
fn grid_struct_array_reads_as_rows() {
    #[derive(Deserialize)]
    struct File {
        grid: Vec<Vec<GridElem>>,
    }
    let grid = mat::from_bytes::<File>(&mat_struct_array::struct_array())
        .unwrap()
        .grid;
    assert_eq!(grid.len(), 2);
    assert_eq!(
        grid[0],
        vec![
            GridElem {
                id: 0.0,
                tag: "a".into()
            },
            GridElem {
                id: 1.0,
                tag: "b".into()
            },
            GridElem {
                id: 2.0,
                tag: "c".into()
            },
        ]
    );
    assert_eq!(
        grid[1],
        vec![
            GridElem {
                id: 10.0,
                tag: "d".into()
            },
            GridElem {
                id: 11.0,
                tag: "e".into()
            },
            GridElem {
                id: 12.0,
                tag: "f".into()
            },
        ]
    );
}

#[test]
fn nested_scalar_struct_in_array_decodes() {
    #[derive(Deserialize)]
    struct File {
        nested: Vec<Nested>,
    }
    let nested = mat::from_bytes::<File>(&mat_struct_array::struct_array())
        .unwrap()
        .nested;
    assert_eq!(
        nested,
        vec![
            Nested {
                fieldA: 1.0,
                inner: Inner { p: 0.0 }
            },
            Nested {
                fieldA: 2.0,
                inner: Inner { p: 100.0 }
            },
        ]
    );
}

// Both fields contain six references but have HDF5 shapes `[3, 2]` and `[2, 3]`.
#[test]
fn a_struct_array_field_of_other_dimensions_is_rejected() {
    let bytes = mat_struct_array::mismatched_struct_array();
    let err = mat::from_bytes::<IgnoredAny>(&bytes).unwrap_err();
    let mat::MatError::StructArrayFieldShape {
        field,
        expected,
        actual,
    } = &err
    else {
        panic!("expected StructArrayFieldShape, got {err:?}");
    };
    assert_eq!(field, "b");
    assert_eq!(expected, &[3, 2]);
    assert_eq!(actual, &[2, 3]);
}

#[test]
fn scalar_struct_is_not_treated_as_array() {
    #[derive(Deserialize)]
    struct File {
        scalar: Data,
    }
    assert_eq!(
        mat::from_bytes::<File>(&mat_struct_array::struct_array())
            .unwrap()
            .scalar,
        expected_data(1)
    );
}

#[test]
fn struct_array_into_single_struct_errors() {
    #[derive(Deserialize, Debug)]
    struct File {
        #[allow(dead_code)]
        row: Data,
    }
    let bytes = mat_struct_array::struct_array();
    let err = mat::from_bytes::<File>(&bytes).unwrap_err();
    let mat::MatError::Custom(message) = &err else {
        panic!("expected Custom, got {err:?}");
    };
    assert_eq!(message, "expected map/struct, got struct array");
}

#[rstest]
#[case::buffered(ReadBackend::Buffered)]
#[case::bytes(ReadBackend::Bytes)]
#[case::source(ReadBackend::Source)]
#[case::streaming(ReadBackend::Streaming)]
fn complete_struct_array_graph(#[case] backend: ReadBackend) {
    let path = test_util::temp::temp_path("struct_array.mat");
    let bytes = mat_struct_array::struct_array();
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    test_util_hdf5::mat_struct_array::assert_contents(&file, &bytes);
    file.close().unwrap();
}

#[rstest]
#[case::buffered(ReadBackend::Buffered)]
#[case::bytes(ReadBackend::Bytes)]
#[case::source(ReadBackend::Source)]
#[case::streaming(ReadBackend::Streaming)]
fn complete_mismatched_struct_array_graph(#[case] backend: ReadBackend) {
    let path = test_util::temp::temp_path("mismatched_struct_array.mat");
    let bytes = mat_struct_array::mismatched_struct_array();
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    test_util_hdf5::mat_struct_array::assert_mismatched_contents(&file, &bytes);
    file.close().unwrap();
}
