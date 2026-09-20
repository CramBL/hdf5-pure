#![cfg(feature = "serde")]
//! Read-back (deserialize) tests for MATLAB cell arrays.
//!
//! The serializer lowers any sequence that doesn't fit a numeric matrix
//! (`Vec<Struct>`, ragged `Vec<Vec<T>>`, `Vec<Option<T>>` with `None`
//! interspersed, nested cells) to a MATLAB cell array: the parent dataset holds
//! object references into the hidden `#refs#` group. These tests round-trip
//! such values through the pure-Rust reader, which resolves those references and
//! reconstructs the cell. They use only `hdf5_pure` (no reference C library), so
//! they run without cmake.

use std::collections::BTreeMap;

use hdf5_pure::mat;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Point {
    x: f64,
    y: f64,
}

fn roundtrip<T>(value: &T) -> T
where
    T: Serialize + serde::de::DeserializeOwned,
{
    let bytes = mat::to_bytes(value).expect("serialize");
    mat::from_bytes(&bytes).expect("deserialize")
}

#[test]
fn vec_of_struct_roundtrips() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        path: Vec<Point>,
    }

    let root = Root {
        path: vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 2.0 },
            Point { x: 3.0, y: 4.5 },
        ],
    };
    assert_eq!(roundtrip(&root), root);
}

#[test]
fn single_element_vec_of_struct_roundtrips() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        path: Vec<Point>,
    }

    let root = Root {
        path: vec![Point { x: -1.0, y: 7.0 }],
    };
    assert_eq!(roundtrip(&root), root);
}

#[test]
fn nested_vec_of_struct_roundtrips() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        grid: Vec<Vec<Point>>,
    }

    let root = Root {
        grid: vec![
            vec![Point { x: 0.0, y: 0.0 }, Point { x: 1.0, y: 1.0 }],
            vec![Point { x: 2.0, y: 2.0 }],
            vec![],
        ],
    };
    assert_eq!(roundtrip(&root), root);
}

#[test]
fn ragged_vec_of_vec_f64_roundtrips() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        rows: Vec<Vec<f64>>,
    }

    // Ragged inner lengths cannot form a matrix, so this lowers to a cell of
    // numeric row vectors.
    let root = Root {
        rows: vec![vec![1.0, 2.0, 3.0], vec![4.0], vec![5.0, 6.0]],
    };
    assert_eq!(roundtrip(&root), root);
}

#[test]
fn vec_of_option_struct_roundtrips_none_slots() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        items: Vec<Option<Point>>,
    }

    // `None` interspersed forces a cell array; each `None` becomes `struct([])`
    // and must read back as `None`.
    let root = Root {
        items: vec![
            Some(Point { x: 1.0, y: 1.0 }),
            None,
            Some(Point { x: 2.0, y: 2.0 }),
            None,
        ],
    };
    assert_eq!(roundtrip(&root), root);
}

#[test]
fn empty_vec_of_struct_roundtrips() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        path: Vec<Point>,
    }

    let root = Root { path: vec![] };
    assert_eq!(roundtrip(&root), root);
}

#[test]
fn reserved_refs_group_is_not_surfaced_as_a_field() {
    // Writing a cell array creates the hidden `#refs#` group at the top level.
    // The reader must skip it: a `deny_unknown_fields` target deserializes
    // cleanly only if `#refs#` never appears as a struct field.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct WriteRoot {
        path: Vec<Point>,
    }
    #[derive(Deserialize, Debug, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct StrictRoot {
        path: Vec<Point>,
    }

    let bytes = mat::to_bytes(&WriteRoot {
        path: vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }],
    })
    .unwrap();

    let back: StrictRoot = mat::from_bytes(&bytes).expect("no `#refs#` field should leak");
    assert_eq!(back.path.len(), 2);
}

#[test]
fn cell_field_alongside_scalars_roundtrips() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        name: String,
        trial: u32,
        path: Vec<Point>,
        scale: f64,
    }

    let root = Root {
        name: "run1".into(),
        trial: 7,
        path: vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }],
        scale: 2.5,
    };
    assert_eq!(roundtrip(&root), root);
}

#[test]
fn self_referential_cell_errors_instead_of_overflowing() {
    use hdf5_pure::{AttrValue, FileBuilder};

    // Hand-craft a malformed cell whose single object reference points back at
    // itself. A real `.mat` file never does this, but a hostile one could; the
    // reader must follow it to a bounded depth and return a typed error, not
    // recurse until the stack overflows.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("loop")
        .with_path_references(&["loop"])
        .with_shape(&[1, 1])
        .set_attr("MATLAB_class", AttrValue::String("cell".into()));
    let bytes = builder.finish().unwrap();

    #[derive(Deserialize, Debug)]
    struct Any {}
    let result: Result<Any, _> = mat::from_bytes(&bytes);
    let err = result.expect_err("a self-referential cell must be rejected");
    assert!(
        err.to_string().contains("depth"),
        "expected a nesting-depth error, got: {err}"
    );
}

#[test]
fn map_of_vec_struct_roundtrips() {
    // A top-level map whose value is a cell array, deserialized into a
    // `BTreeMap` — exercises that only the user variable is surfaced (no
    // `#refs#` key).
    let mut root: BTreeMap<String, Vec<Point>> = BTreeMap::new();
    root.insert(
        "trajectory".into(),
        vec![Point { x: 0.0, y: 0.0 }, Point { x: 1.0, y: 1.0 }],
    );
    let back: BTreeMap<String, Vec<Point>> = roundtrip(&root);
    assert_eq!(back, root);
}
