//! Tests for the `ndarray` integration: round-trips through `hdf5-pure` itself
//! and byte-level cross-validation against the reference HDF5 library
//! (`hdf5-metno`). Only built with the `ndarray` feature.
// Byte-level crosscheck links the reference HDF5 C library (`hdf5-metno`), gated
// to 64-bit little-endian targets; skip elsewhere so `cross test` on `i686` and
// `s390x` stays pure-Rust.
#![cfg(all(
    feature = "ndarray",
    not(target_pointer_width = "32"),
    target_endian = "little"
))]

use hdf5_pure::{Error, File, FileBuilder};
use ndarray::{Array1, Array2, Array3, ArrayD, ShapeBuilder, array};
#[cfg(feature = "__hdf5-1.10")]
use tempfile::tempdir;

/// Build a one-dataset file from an ndarray and reopen it in memory.
fn write_then_open(name: &str, build: impl FnOnce(&mut FileBuilder)) -> File {
    let mut fb = FileBuilder::new();
    build(&mut fb);
    let bytes = fb.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();
    // Sanity: the dataset exists.
    file.dataset(name).unwrap();
    file
}

// ---------------------------------------------------------------------------
// Round-trips through hdf5-pure across ranks and element types
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_1d_f32() {
    let a: Array1<f32> = array![1.5, 2.5, 3.5, 4.5];
    let file = write_then_open("v", |fb| {
        fb.create_dataset("v").with_ndarray(&a);
    });
    let ds = file.dataset("v").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);
    let back: Array1<f32> = ds.read_array().unwrap();
    assert_eq!(back, a);
}

#[test]
fn roundtrip_2d_f64() {
    let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let file = write_then_open("m", |fb| {
        fb.create_dataset("m").with_ndarray(&a);
    });
    let ds = file.dataset("m").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    let back: Array2<f64> = ds.read_array().unwrap();
    assert_eq!(back, a);
}

#[test]
fn roundtrip_3d_i32() {
    // 2 x 2 x 3 cube of distinct values.
    let a: Array3<i32> = Array3::from_shape_vec((2, 2, 3), (0..12).collect()).unwrap();
    let file = write_then_open("cube", |fb| {
        fb.create_dataset("cube").with_ndarray(&a);
    });
    let ds = file.dataset("cube").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 2, 3]);
    let back: Array3<i32> = ds.read_array().unwrap();
    assert_eq!(back, a);
}

#[test]
fn roundtrip_2d_i64_and_u32() {
    let a: Array2<i64> = array![[-1, -2], [-3, -4], [-5, -6]];
    let b: Array2<u32> = array![[10, 20, 30], [40, 50, 60]];
    let mut fb = FileBuilder::new();
    fb.create_dataset("a").with_ndarray(&a);
    fb.create_dataset("b").with_ndarray(&b);
    let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    assert_eq!(
        file.dataset("a").unwrap().read_array::<i64, _>().unwrap(),
        a
    );
    assert_eq!(
        file.dataset("b").unwrap().read_array::<u32, _>().unwrap(),
        b
    );
}

#[test]
fn read_array_dyn_reports_runtime_rank() {
    let a: Array3<f64> = Array3::from_shape_vec((3, 1, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let file = write_then_open("d", |fb| {
        fb.create_dataset("d").with_ndarray(&a);
    });
    let dynamic: ArrayD<f64> = file.dataset("d").unwrap().read_array_dyn().unwrap();
    assert_eq!(dynamic.ndim(), 3);
    assert_eq!(dynamic.shape(), &[3, 1, 2]);
    assert_eq!(dynamic, a.into_dyn());
}

// ---------------------------------------------------------------------------
// Non-standard memory layouts must be repacked into row-major on write
// ---------------------------------------------------------------------------

#[test]
fn transposed_view_is_written_row_major() {
    let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let transposed = a.t(); // 3x2 view, NOT standard layout
    assert!(!transposed.is_standard_layout());

    let file = write_then_open("t", |fb| {
        fb.create_dataset("t").with_ndarray(&transposed);
    });
    let ds = file.dataset("t").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![3, 2]);
    let back: Array2<f64> = ds.read_array().unwrap();
    // Logical values must match the transpose, regardless of source layout.
    assert_eq!(back, transposed);
    assert_eq!(back, array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
}

#[test]
fn fortran_order_array_is_written_row_major() {
    // Same logical 2x3 matrix, but stored column-major (Fortran order).
    let f: Array2<f64> =
        Array2::from_shape_vec((2, 3).f(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    assert!(!f.is_standard_layout());
    let expected: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(f, expected);

    let file = write_then_open("f", |fb| {
        fb.create_dataset("f").with_ndarray(&f);
    });
    let back: Array2<f64> = file.dataset("f").unwrap().read_array().unwrap();
    assert_eq!(back, expected);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn read_array_wrong_rank_errors() {
    let a: Array3<f64> =
        Array3::from_shape_vec((2, 2, 2), (0..8).map(|x| x as f64).collect()).unwrap();
    let file = write_then_open("cube", |fb| {
        fb.create_dataset("cube").with_ndarray(&a);
    });
    let ds = file.dataset("cube").unwrap();
    // Requesting a 2-D array from a 3-D dataset must fail, not panic.
    let err = ds.read_array::<f64, ndarray::Ix2>().unwrap_err();
    assert!(matches!(err, Error::Shape(_)), "got {err:?}");
    // But read_array_dyn still works.
    assert_eq!(ds.read_array_dyn::<f64>().unwrap(), a.into_dyn());
}

// ---------------------------------------------------------------------------
// Cross-validation against the reference HDF5 library (hdf5-metno)
// ---------------------------------------------------------------------------

// Written in the 1.10 format, which a 1.8 library cannot open.
#[cfg(feature = "__hdf5-1.10")]
#[test]
fn crosscheck_2d_with_reference_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nd2.h5");

    let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut fb = FileBuilder::new();
    fb.create_dataset("m").with_ndarray(&a);
    fb.write(&path).unwrap();

    // The reference library must see the same shape and row-major values.
    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("m").unwrap();
    assert_eq!(ds.shape(), vec![2, 3]);
    let read: Array2<f64> = ds.read_2d::<f64>().unwrap();
    assert_eq!(read, a);
}

#[cfg(feature = "__hdf5-1.10")]
#[test]
fn crosscheck_3d_with_reference_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nd3.h5");

    let a: Array3<i32> = Array3::from_shape_vec((2, 3, 4), (0..24).collect()).unwrap();
    let mut fb = FileBuilder::new();
    fb.create_dataset("cube").with_ndarray(&a);
    fb.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("cube").unwrap();
    assert_eq!(ds.shape(), vec![2, 3, 4]);
    let read: ArrayD<i32> = ds.read_dyn::<i32>().unwrap();
    assert_eq!(read, a.into_dyn());
}

#[cfg(feature = "__hdf5-1.10")]
#[test]
fn crosscheck_chunked_compressed_ndarray() {
    // ndarray write composes with the existing chunking/compression builders.
    let dir = tempdir().unwrap();
    let path = dir.path().join("nd_chunked.h5");

    let a: Array2<f64> = Array2::from_shape_fn((8, 8), |(i, j)| (i * 8 + j) as f64 * 0.25);
    let mut fb = FileBuilder::new();
    fb.create_dataset("m")
        .with_ndarray(&a)
        .with_chunks(&[4, 4])
        .with_deflate(6);
    fb.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let read: Array2<f64> = file.dataset("m").unwrap().read_2d::<f64>().unwrap();
    assert_eq!(read, a);

    // And it round-trips back through hdf5-pure too.
    let ours = File::open(&path).unwrap();
    let back: Array2<f64> = ours.dataset("m").unwrap().read_array().unwrap();
    assert_eq!(back, a);
}
