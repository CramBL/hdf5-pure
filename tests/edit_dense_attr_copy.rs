// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Object copy reproduces dense (fractal-heap) attribute storage (issue #87,
//! follow-up to PR #78).
//!
//! Before this, `File::copy` / `copy_from` refused any object whose
//! attributes were stored densely (a *defined* fractal-heap address in its
//! Attribute Info message). Now the copy reads the source attributes out of the
//! heap, rebuilds a fresh single-direct-block heap + B-tree v2 name index at the
//! destination, and references it from the rebuilt header — both within one file
//! (`copy`) and across two (`copy_from`). These tests build dense-attribute
//! sources, copy them, and verify every attribute (name + value) survives, in
//! this crate's reader and the reference C library.

use hdf5::file::LibraryVersion;
use hdf5_pure::{AttrValue, File, FileBuilder};
use std::collections::HashMap;
use tempfile::tempdir;

mod common;
use common::heap::{has_fractal_heap, huge_object_counts};

/// Number of attributes that forces the whole-file writer into dense storage
/// (its threshold is 8 compact attributes).
const DENSE_ATTR_COUNT: usize = 12;

/// Attribute name/value pairs that exceed the compact threshold.
fn dense_attr_set() -> Vec<(String, i64)> {
    (0..DENSE_ATTR_COUNT)
        .map(|i| (format!("attr_{i:02}"), 1000 + i as i64))
        .collect()
}

/// Build a file (with this crate's writer) holding a dataset and a group that
/// each carry [`DENSE_ATTR_COUNT`] attributes, forcing dense storage.
fn write_dense_source(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    {
        let ds = b.create_dataset("payload");
        ds.with_f64_data(&[1.5, 2.5, 3.5]);
        for (name, value) in dense_attr_set() {
            ds.set_attr(&name, AttrValue::I64(value));
        }
    }
    {
        let mut grp = b.create_group("bundle");
        grp.create_dataset("inner").with_i32_data(&[7, 8, 9]);
        for (name, value) in dense_attr_set() {
            grp.set_attr(&name, AttrValue::I64(value));
        }
        b.add_group(grp.finish());
    }
    b.write(path).unwrap();
}

/// The file must actually contain a fractal heap (the dense-storage signature);
/// otherwise the test would pass trivially against compact storage.
fn assert_file_has_fractal_heap(path: &std::path::Path) {
    assert!(
        has_fractal_heap(&std::fs::read(path).unwrap()),
        "source file does not use dense (fractal-heap) storage",
    );
}

/// Pull `name -> i64` for every attribute readable by this crate's reader.
fn i64_attrs(attrs: &HashMap<String, AttrValue>) -> HashMap<String, i64> {
    attrs
        .iter()
        .filter_map(|(k, v)| match v {
            AttrValue::I64(n) => Some((k.clone(), *n)),
            _ => None,
        })
        .collect()
}

fn expected_attr_map() -> HashMap<String, i64> {
    dense_attr_set().into_iter().collect()
}

#[test]
fn same_file_copy_reproduces_dense_dataset_and_group_attrs() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense_same.h5");
    write_dense_source(&path);
    assert_file_has_fractal_heap(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("payload", "payload_copy").unwrap();
        session.copy("bundle", "bundle_copy").unwrap();
        session.commit().unwrap();
    }

    // This crate's reader: the copy carries every dense attribute, by name+value.
    let f = File::open(&path).unwrap();
    let ds_copy = f.dataset("payload_copy").unwrap();
    assert_eq!(ds_copy.read_f64().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(i64_attrs(&ds_copy.attrs().unwrap()), expected_attr_map());

    let grp_copy = f.group("bundle_copy").unwrap();
    assert_eq!(i64_attrs(&grp_copy.attrs().unwrap()), expected_attr_map());
    assert_eq!(
        f.dataset("bundle_copy/inner").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );

    // Reference C library: the copy's dense storage is valid in the new file.
    let c = hdf5::File::open(&path).unwrap();
    let ds = c.dataset("payload_copy").unwrap();
    assert_eq!(ds.read_raw::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
    for (name, value) in dense_attr_set() {
        let got: i64 = ds.attr(&name).unwrap().read_scalar().unwrap();
        assert_eq!(got, value, "dataset attr {name} mismatch (C library)");
    }
    let g = c.group("bundle_copy").unwrap();
    for (name, value) in dense_attr_set() {
        let got: i64 = g.attr(&name).unwrap().read_scalar().unwrap();
        assert_eq!(got, value, "group attr {name} mismatch (C library)");
    }
}

/// Copy rebuilds the destination heap from scratch, so it rebuilds the name
/// index too. Twelve attributes fit one B-tree leaf and say nothing about that;
/// 600 need a root and two levels below it, so this is where the copy path
/// either carries the multi-level index or does not (issue #195).
#[test]
fn copy_reproduces_a_multi_level_dense_name_index() {
    const COUNT: usize = 600;
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense_many.h5");

    {
        let mut b = FileBuilder::new();
        {
            let ds = b.create_dataset("payload");
            ds.with_f64_data(&[1.5]);
            for i in 0..COUNT {
                ds.set_attr(&format!("attr_{i:04}"), AttrValue::I64(i as i64));
            }
        }
        b.write(&path).unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("payload", "payload_copy").unwrap();
        session.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    let attrs = f.dataset("payload_copy").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), COUNT);
    for i in 0..COUNT {
        let name = format!("attr_{i:04}");
        assert!(
            matches!(attrs.get(&name), Some(AttrValue::I64(v)) if *v == i as i64),
            "{name} missing from the copy"
        );
    }
    drop(f);

    // The C library, opening each by name: a copy whose records landed out of
    // order still iterates cleanly and still fails to be searched.
    let c = hdf5::File::open(&path).unwrap();
    let ds = c.dataset("payload_copy").unwrap();
    for i in [0, 1, COUNT / 2, COUNT - 1] {
        let name = format!("attr_{i:04}");
        let got: i64 = ds.attr(&name).unwrap().read_scalar().unwrap();
        assert_eq!(got, i as i64, "{name} mismatch (C library)");
    }
}

/// Copy rebuilds the destination heap from the source attributes, so an
/// attribute large enough to need fractal-heap *huge* storage has to be rebuilt
/// as one — including the huge-objects B-tree and the header fields that point at
/// it. The blob is placed at end-of-file rather than address zero, so this also
/// exercises the relocation of every address inside it.
#[test]
fn copy_reproduces_huge_dense_attrs() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense_huge_copy.h5");

    let big: Vec<i64> = (0..10_000).collect();
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("payload");
        ds.with_f64_data(&[1.5]);
        ds.set_attr("big", AttrValue::I64Array(big.clone()));
        for (name, value) in dense_attr_set() {
            ds.set_attr(&name, AttrValue::I64(value));
        }
        b.write(&path).unwrap();
    }
    assert_eq!(huge_object_counts(&std::fs::read(&path).unwrap()), vec![1]);

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("payload", "payload_copy").unwrap();
        session.commit().unwrap();
    }

    // Two heaps now, and the copy's own header must declare a huge object too
    // rather than having quietly fallen back to a managed object it cannot hold.
    // Reading only the first heap would report the source's choice, not the copy's.
    assert_eq!(
        huge_object_counts(&std::fs::read(&path).unwrap()),
        vec![1, 1],
        "expected a second fractal heap for the copy, huge-storing as the source does"
    );

    let f = File::open(&path).unwrap();
    let copied = f.dataset("payload_copy").unwrap().attrs().unwrap();
    assert_eq!(i64_attrs(&copied), expected_attr_map());
    match copied.get("big") {
        Some(AttrValue::I64Array(v)) => assert_eq!(*v, big),
        other => panic!("huge attribute came back as {other:?}"),
    }

    let c = hdf5::File::open(&path).unwrap();
    let ds = c.dataset("payload_copy").unwrap();
    assert_eq!(ds.attr("big").unwrap().read_raw::<i64>().unwrap(), big);
    for (name, value) in dense_attr_set() {
        let got: i64 = ds.attr(&name).unwrap().read_scalar().unwrap();
        assert_eq!(got, value, "copied attr {name} mismatch (C library)");
    }
}

#[test]
fn cross_file_copy_reproduces_dense_attrs() {
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("dense_xsrc.h5");
    let dst_path = dir.path().join("dense_xdst.h5");
    write_dense_source(&src_path);
    assert_file_has_fractal_heap(&src_path);

    // A plain destination written by this crate.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("keep").with_f64_data(&[0.0]);
        b.write(&dst_path).unwrap();
    }

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session.copy_from(&source, "payload", "payload").unwrap();
        session.copy_from(&source, "bundle", "bundle").unwrap();
        session.commit().unwrap();
    }

    let f = File::open(&dst_path).unwrap();
    assert_eq!(
        f.dataset("payload").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        i64_attrs(&f.dataset("payload").unwrap().attrs().unwrap()),
        expected_attr_map()
    );
    assert_eq!(
        i64_attrs(&f.group("bundle").unwrap().attrs().unwrap()),
        expected_attr_map()
    );

    let c = hdf5::File::open(&dst_path).unwrap();
    let ds = c.dataset("payload").unwrap();
    for (name, value) in dense_attr_set() {
        let got: i64 = ds.attr(&name).unwrap().read_scalar().unwrap();
        assert_eq!(got, value, "cross-file dataset attr {name} mismatch");
    }
    let g = c.group("bundle").unwrap();
    for (name, value) in dense_attr_set() {
        let got: i64 = g.attr(&name).unwrap().read_scalar().unwrap();
        assert_eq!(got, value, "cross-file group attr {name} mismatch");
    }
    // The destination's pre-existing object survives.
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![0.0]
    );
}

#[test]
fn c_written_dense_attr_dataset_copies_in_place() {
    // The denseness comes from the reference C library this time, so the read
    // path is exercised against genuine C-written fractal-heap attribute storage,
    // not only this crate's own emitter.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_dense.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((3,))
            .create("calibration")
            .unwrap();
        ds.write(&[0.99f64, 1.0, 1.01]).unwrap();
        for (name, value) in dense_attr_set() {
            ds.new_attr::<i64>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&value)
                .unwrap();
        }
        file.close().unwrap();
    }
    assert_file_has_fractal_heap(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("calibration", "calibration_copy").unwrap();
        session.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        i64_attrs(&f.dataset("calibration_copy").unwrap().attrs().unwrap()),
        expected_attr_map()
    );

    let c = hdf5::File::open(&path).unwrap();
    let ds = c.dataset("calibration_copy").unwrap();
    assert_eq!(ds.read_raw::<f64>().unwrap(), vec![0.99, 1.0, 1.01]);
    for (name, value) in dense_attr_set() {
        let got: i64 = ds.attr(&name).unwrap().read_scalar().unwrap();
        assert_eq!(got, value, "C-written dense attr {name} mismatch");
    }
}

#[test]
fn cross_file_copy_refuses_variable_length_dense_attrs() {
    // A dense attribute set whose values embed source-file global-heap addresses
    // (variable-length strings) cannot be copied to another file verbatim — the
    // addresses would dangle. The cross-file path must refuse it by name rather
    // than mis-encode. (Same-file copy keeps such attributes valid by sharing the
    // source heaps, so this restriction is cross-file only.)
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("vlen_dense_src.h5");
    let dst_path = dir.path().join("vlen_dense_dst.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&src_path)
            .unwrap();
        let ds = file.new_dataset::<f64>().shape((2,)).create("vds").unwrap();
        ds.write(&[1.0f64, 2.0]).unwrap();
        // Enough variable-length string attributes to force dense storage.
        for i in 0..DENSE_ATTR_COUNT {
            ds.new_attr::<hdf5::types::VarLenUnicode>()
                .shape(())
                .create(format!("note_{i:02}").as_str())
                .unwrap()
                .write_scalar(
                    &format!("value {i}")
                        .parse::<hdf5::types::VarLenUnicode>()
                        .unwrap(),
                )
                .unwrap();
        }
        file.close().unwrap();
    }
    assert_file_has_fractal_heap(&src_path);

    {
        let mut b = FileBuilder::new();
        b.create_dataset("keep").with_f64_data(&[0.0]);
        b.write(&dst_path).unwrap();
    }

    let source = File::open(&src_path).unwrap();
    let session = File::open_rw(&dst_path).unwrap();
    let err = session
        .copy_from(&source, "vds", "vds")
        .expect_err("variable-length dense attrs must be refused cross-file");
    let msg = err.to_string();
    assert!(
        msg.contains("variable-length") && msg.contains("dense"),
        "unexpected refusal message: {msg}",
    );
}
