// Crosschecks against the reference HDF5 C library (`hdf5-metno`), gated to
// 64-bit little-endian targets; skip elsewhere so `cross test` on `i686` and
// `s390x` stays pure-Rust.
#![cfg(all(
    feature = "serde",
    not(target_pointer_width = "32"),
    target_endian = "little"
))]
//! Cell-array serialization for sequences that don't fit a numeric matrix.
//!
//! `Vec<MyStruct>`, `Vec<Option<T>>` with `None` interspersed, and
//! `Vec<Vec<MyStruct>>` all serialize as MATLAB cell arrays. Each element is
//! interned under the conventional `#refs#` group; the parent dataset stores
//! object references. `None` becomes `struct([])` (an empty struct array)
//! so every cell slot has a well-defined MATLAB type.

use hdf5_pure::mat;
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct PathRoot {
    path: Vec<Point>,
}

#[test]
fn vec_of_struct_writes_cell_array_with_refs_group() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("path.mat");

    let root = PathRoot {
        path: vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 2.0 },
            Point { x: 3.0, y: 4.0 },
        ],
    };
    mat::to_file(&root, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    // Parent dataset has MATLAB_class="cell" and shape [1, 3] (column-major
    // storage of the column vector [3, 1] MATLAB shape).
    let cell = file.dataset("path").unwrap();
    assert_eq!(cell.shape(), vec![1, 3]);
    let cls = cell
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "cell");

    // The hidden #refs# group exists and holds three children, one per
    // element. Each child is a struct group with MATLAB_class="struct" and
    // the field names from `Point`.
    let refs = file.group("#refs#").unwrap();
    let names = refs.member_names().unwrap();
    assert_eq!(names.len(), 3, "expected one ref per cell element");

    for ref_name in &names {
        let g = refs.group(ref_name).unwrap();
        let cls = g
            .attr("MATLAB_class")
            .unwrap()
            .read_scalar::<hdf5::types::FixedAscii<32>>()
            .unwrap();
        assert_eq!(cls.as_str(), "struct");
        // Two fields, "x" and "y", both length-1 doubles.
        let x = g.dataset("x").unwrap();
        assert_eq!(x.shape(), vec![1, 1]);
        let y = g.dataset("y").unwrap();
        assert_eq!(y.shape(), vec![1, 1]);
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct OptionalSeq {
    items: Vec<Option<Point>>,
}

#[test]
fn vec_of_option_with_none_uses_empty_struct_marker() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("optional.mat");

    let root = OptionalSeq {
        items: vec![
            Some(Point { x: 1.0, y: 2.0 }),
            None,
            Some(Point { x: 3.0, y: 4.0 }),
        ],
    };
    mat::to_file(&root, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    let cell = file.dataset("items").unwrap();
    assert_eq!(cell.shape(), vec![1, 3]);
    let cls = cell
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "cell");

    let refs = file.group("#refs#").unwrap();
    let names = refs.member_names().unwrap();
    assert_eq!(names.len(), 3);

    // Exactly one of the three refs is the empty-struct marker for `None`.
    let mut empty_struct_count = 0;
    for ref_name in &names {
        // Try as group first (Some(struct)); fall back to dataset (None marker).
        if let Ok(g) = refs.group(ref_name) {
            let cls = g
                .attr("MATLAB_class")
                .unwrap()
                .read_scalar::<hdf5::types::FixedAscii<32>>()
                .unwrap();
            assert_eq!(cls.as_str(), "struct");
            assert!(g.dataset("x").is_ok());
        } else {
            let ds = refs.dataset(ref_name).unwrap();
            let cls = ds
                .attr("MATLAB_class")
                .unwrap()
                .read_scalar::<hdf5::types::FixedAscii<32>>()
                .unwrap();
            assert_eq!(cls.as_str(), "struct");
            let empty = ds
                .attr("MATLAB_empty")
                .unwrap()
                .read_scalar::<u32>()
                .unwrap();
            assert_eq!(empty, 1);
            empty_struct_count += 1;
        }
    }
    assert_eq!(empty_struct_count, 1, "expected one struct([]) marker");
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct NestedSeq {
    rows: Vec<Vec<Option<Point>>>,
}

#[test]
fn nested_vec_of_vec_produces_cell_of_cells() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.mat");

    // `Vec<Vec<Option<Struct>>>`: rows of optional points. Each outer
    // entry is a cell of inner refs; missing inner elements become
    // `struct([])` markers.
    let root = NestedSeq {
        rows: vec![
            vec![Some(Point { x: 1.0, y: 1.0 }), None],
            vec![None, Some(Point { x: 2.0, y: 2.0 })],
        ],
    };
    mat::to_file(&root, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    // Outer cell array: [1, 2] storage shape for a [2, 1] MATLAB column vector.
    let outer = file.dataset("rows").unwrap();
    assert_eq!(outer.shape(), vec![1, 2]);
    let cls = outer
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "cell");

    // #refs# holds: 2 outer entries (each itself a cell) + 4 inner entries
    // (2 structs + 2 empty markers) = 6 total.
    let refs = file.group("#refs#").unwrap();
    let names = refs.member_names().unwrap();
    assert_eq!(names.len(), 6, "2 outer cells + 4 inner elements");

    let mut outer_cell_count = 0;
    let mut inner_struct_count = 0;
    let mut empty_struct_count = 0;
    for ref_name in &names {
        if let Ok(g) = refs.group(ref_name) {
            assert_eq!(
                g.attr("MATLAB_class")
                    .unwrap()
                    .read_scalar::<hdf5::types::FixedAscii<32>>()
                    .unwrap()
                    .as_str(),
                "struct"
            );
            inner_struct_count += 1;
        } else {
            let ds = refs.dataset(ref_name).unwrap();
            let cls = ds
                .attr("MATLAB_class")
                .unwrap()
                .read_scalar::<hdf5::types::FixedAscii<32>>()
                .unwrap();
            match cls.as_str() {
                "cell" => outer_cell_count += 1,
                "struct" => empty_struct_count += 1,
                other => panic!("unexpected class in #refs#: {other}"),
            }
        }
    }
    assert_eq!(outer_cell_count, 2);
    assert_eq!(inner_struct_count, 2);
    assert_eq!(empty_struct_count, 2);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct RaggedRoot {
    rows: Vec<Vec<f64>>,
}

#[test]
fn ragged_vec_of_vec_falls_back_to_cell() {
    // Same-length inner vecs unify into a numeric matrix; ragged vecs go to
    // cell so we don't lose data.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ragged.mat");

    let root = RaggedRoot {
        rows: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]],
    };
    mat::to_file(&root, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    let cell = file.dataset("rows").unwrap();
    let cls = cell
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "cell");
    assert_eq!(cell.shape(), vec![1, 2]);

    let refs = file.group("#refs#").unwrap();
    let names = refs.member_names().unwrap();
    assert_eq!(names.len(), 2);
}

#[test]
fn empty_vec_of_struct_serializes_as_empty_double() {
    // `Vec<MyStruct>` with zero elements has unknown element type at the
    // serializer level (no `Some(_)` ever fired), so it falls into the
    // "empty sequence" branch which still picks the f64 default. This is
    // existing behavior; the test pins it.
    #[derive(Serialize)]
    struct Root {
        items: Vec<Point>,
    }
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.mat");
    mat::to_file(&Root { items: Vec::new() }, &path).unwrap();
    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("items").unwrap();
    let cls = ds
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "double");
}

/// Cell-array reading resolves each element's `#refs#` object reference back
/// into a `MatValue` and reconstructs the sequence, so a `Vec<Struct>`
/// round-trips through the pure-Rust deserializer. (Pure-Rust read coverage
/// lives in `tests/mat_cell_read.rs`; this case pairs it with the C-library
/// write checks above.)
#[test]
fn from_file_on_cell_array_roundtrips() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("roundtrip_cell.mat");

    let original = PathRoot {
        path: vec![Point { x: 10.0, y: 20.0 }, Point { x: 30.0, y: 40.0 }],
    };
    mat::to_file(&original, &path).unwrap();

    let back: PathRoot = mat::from_file(&path).expect("cell array should deserialize");
    assert_eq!(back, original);
}

#[test]
fn c_library_can_open_cell_array_file() {
    // Sanity-check that the cell-array file is valid HDF5: the reference
    // C library opens it, sees the cell dataset and the #refs# group, and
    // can follow the references back to the element data.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_check.mat");

    mat::to_file(
        &PathRoot {
            path: vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }],
        },
        &path,
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    assert!(file.dataset("path").is_ok());
    let refs = file.group("#refs#").unwrap();
    let names = refs.member_names().unwrap();
    assert_eq!(names.len(), 2);
    // Each ref entry is a struct group with x and y datasets.
    for name in &names {
        let g = refs.group(name).unwrap();
        let x = g.dataset("x").unwrap().read_raw::<f64>().unwrap();
        let y = g.dataset("y").unwrap().read_raw::<f64>().unwrap();
        assert_eq!(x.len(), 1);
        assert_eq!(y.len(), 1);
    }
}

#[test]
fn cell_references_resolve_in_element_order() {
    // Pin the writer's invariant that the i-th object reference in the cell
    // dataset dereferences to the i-th input element. A bug where the
    // RefsAccumulator queued out of order, or where with_path_references
    // reordered, would surface here.
    use hdf5::{ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let path = dir.path().join("ordered.mat");

    let inputs = vec![
        Point { x: 10.0, y: -10.0 },
        Point { x: 20.0, y: -20.0 },
        Point { x: 30.0, y: -30.0 },
        Point { x: 40.0, y: -40.0 },
    ];
    mat::to_file(
        &PathRoot {
            path: inputs.clone(),
        },
        &path,
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let cell = file.dataset("path").unwrap();
    let refs: Vec<ObjectReference1> = cell.read_raw().unwrap();
    assert_eq!(refs.len(), inputs.len());

    for (i, r) in refs.iter().enumerate() {
        let target = file.dereference(r).unwrap();
        let g = match target {
            ReferencedObject::Group(g) => g,
            other => panic!("ref {i} dereferenced to non-group: {other:?}"),
        };
        let x: Vec<f64> = g.dataset("x").unwrap().read_raw().unwrap();
        let y: Vec<f64> = g.dataset("y").unwrap().read_raw().unwrap();
        assert_eq!(
            (x[0], y[0]),
            (inputs[i].x, inputs[i].y),
            "cell slot {i} should resolve to input element {i}"
        );
    }
}
