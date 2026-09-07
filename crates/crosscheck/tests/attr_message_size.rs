// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for the compact-attribute message-size limit
//! (issue #190).
//!
//! An attribute's object-header message must fit the header's 2-byte size field.
//! The writer used to refuse a larger one; since #195 it moves it to a fractal
//! heap instead, so the limit is now where *storage changes*, not where writing
//! stops. Either way the boundary is only worth anything if the largest attribute
//! still written compactly produces a file the reference C library reads
//! correctly — otherwise the limit would just be sitting in the wrong place. So
//! this writes right up against it and reads the values back through libhdf5.

use hdf5_pure::{AttrValue, FileBuilder, OBJECT_HEADER_MESSAGE_MAX};
use tempfile::tempdir;

use test_util::heap::has_fractal_heap;

/// Whether a root `i64` attribute of `n` elements is stored compactly.
fn stores_i64_attr_compactly(n: usize) -> bool {
    let mut builder = FileBuilder::new();
    builder.set_attr("probe", AttrValue::I64Array(vec![0; n]));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = builder.finish().expect("every size is writable");
    !has_fractal_heap(&bytes)
}

/// The largest `i64` attribute the writer keeps compact, found by probing down
/// from the size field's limit. The exact element count depends on the encoder's
/// per-message overhead (name, datatype, dataspace), and this test wants the
/// boundary itself rather than an approximation that drifts away from it.
///
/// The probe would happily settle on a much smaller attribute if the storage
/// choice ever started flipping at large `n` for an unrelated reason — leaving
/// both tests below green while testing nothing near the limit — so the element
/// it stops at is checked to be the boundary: one more must go to a heap.
fn largest_compact_i64_attr() -> Vec<i64> {
    for n in (1..=OBJECT_HEADER_MESSAGE_MAX / 8).rev() {
        if stores_i64_attr_compactly(n) {
            assert!(
                !stores_i64_attr_compactly(n + 1),
                "{n} elements stayed compact, and so did {}, so this is not the boundary",
                n + 1
            );
            return (0..n as i64).collect();
        }
    }
    panic!("no i64 attribute size was stored compactly");
}

#[test]
fn c_reads_the_largest_accepted_root_attribute() {
    let values = largest_compact_i64_attr();
    let dir = tempdir().unwrap();
    let path = dir.path().join("boundary_root.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("labels", AttrValue::I64Array(values.clone()));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let got: Vec<i64> = file.attr("labels").unwrap().read_raw().unwrap();
    assert_eq!(got, values);
    file.close().unwrap();
}

/// The other side of the same boundary, which used to be unreachable because the
/// writer refused it. One element past the compact limit the attribute moves to a
/// heap, and libhdf5 has to read it there — so the two storage choices meet
/// without a gap, rather than the compact side being verified and the dense side
/// assumed.
#[test]
fn c_reads_one_element_past_the_compact_boundary() {
    let mut values = largest_compact_i64_attr();
    values.push(values.len() as i64);

    let dir = tempdir().unwrap();
    let path = dir.path().join("past_boundary.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("labels", AttrValue::I64Array(values.clone()));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    assert!(
        has_fractal_heap(&std::fs::read(&path).unwrap()),
        "one element past the compact limit must select heap storage"
    );

    let file = hdf5::File::open(&path).unwrap();
    let got: Vec<i64> = file.attr("labels").unwrap().read_raw().unwrap();
    assert_eq!(got, values);
    file.close().unwrap();
}

#[test]
fn c_reads_the_largest_accepted_dataset_attribute() {
    let values = largest_compact_i64_attr();
    let dir = tempdir().unwrap();
    let path = dir.path().join("boundary_dataset.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("x")
        .with_f64_data(&[1.0])
        .set_attr("labels", AttrValue::I64Array(values.clone()));
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let got: Vec<i64> = file
        .dataset("x")
        .unwrap()
        .attr("labels")
        .unwrap()
        .read_raw()
        .unwrap();
    assert_eq!(got, values);
    file.close().unwrap();
}
