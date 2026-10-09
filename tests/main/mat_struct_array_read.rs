//! Reading MATLAB struct *arrays* (issue #127) into `Vec<T>` / `Vec<Vec<T>>`.
//!
//! On disk a struct array is a `MATLAB_class="struct"` group whose every field
//! is a dataset of per-element object references, unlike a scalar struct whose
//! fields are direct value datasets. The fixture is synthetic, built to MATLAB's
//! documented v7.3 layout by `tests/data/h5py/mat/gen_struct_array.py`
//! (see `tests/data/h5py/mat/NOTICE.md`).
#![cfg(feature = "serde")]

use hdf5_pure::mat;
use serde::Deserialize;
use test_util_hdf5::mat_file;

const FIXTURE: &str = "tests/data/h5py/mat/struct_array_v73.mat";

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

/// The issue's exact case: a `1×6` struct array deserializes into `Vec<Data>`.
#[test]
fn row_struct_array_reads_as_vec() {
    #[derive(Deserialize)]
    struct File {
        row: Vec<Data>,
    }
    let row = mat_file::decode_fixture(FIXTURE, mat::from_bytes::<File>).row;
    let expected: Vec<Data> = (1..=6).map(expected_data).collect();
    assert_eq!(row, expected);
}

/// A `6×1` column struct array flattens into the same `Vec<Data>` as the row
/// orientation, matching how the crate flattens `1×N` / `N×1` numeric arrays.
#[test]
fn col_struct_array_reads_as_vec() {
    #[derive(Deserialize)]
    struct File {
        col: Vec<Data>,
    }
    let col = mat_file::decode_fixture(FIXTURE, mat::from_bytes::<File>).col;
    let expected: Vec<Data> = (1..=6).map(expected_data).collect();
    assert_eq!(col, expected);
}

/// A true `2×3` struct array yields a row-major `Vec<Vec<T>>`, mirroring the
/// numeric `Matrix` row split. `id = row*10 + col` pins the ordering.
#[test]
fn grid_struct_array_reads_as_rows() {
    #[derive(Deserialize)]
    struct File {
        grid: Vec<Vec<GridElem>>,
    }
    let grid = mat_file::decode_fixture(FIXTURE, mat::from_bytes::<File>).grid;
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

/// Each struct-array element may itself contain a nested *scalar* struct,
/// resolved through the element's reference like any other field value.
#[test]
fn nested_scalar_struct_in_array_decodes() {
    #[derive(Deserialize)]
    struct File {
        nested: Vec<Nested>,
    }
    let nested = mat_file::decode_fixture(FIXTURE, mat::from_bytes::<File>).nested;
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

/// Regression guard: a scalar (1×1) struct must still read as a single struct,
/// not be misdetected as a struct array.
#[test]
fn scalar_struct_is_not_treated_as_array() {
    #[derive(Deserialize)]
    struct File {
        scalar: Data,
    }
    assert_eq!(
        mat_file::decode_fixture(FIXTURE, mat::from_bytes::<File>).scalar,
        expected_data(1)
    );
}

/// Deserializing a multi-element struct array into a single struct fails with a
/// clear error rather than silently taking the first element.
#[test]
fn struct_array_into_single_struct_errors() {
    #[derive(Deserialize, Debug)]
    struct File {
        #[allow(dead_code)]
        row: Data,
    }
    let bytes = mat_file::read_fixture(FIXTURE);
    let err = mat::from_bytes::<File>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("struct array"),
        "error should mention the struct array, got: {msg}"
    );
}
