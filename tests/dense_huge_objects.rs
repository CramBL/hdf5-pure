// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reading dense group-link and dense-attribute storage whose messages are too
//! large for the fractal heap to "manage" and are instead stored as "huge"
//! objects, resolved through the heap's huge-objects v2 B-tree.
//!
//! Regression: a fractal-heap heap ID encodes its object type in bits 4-5 of the
//! first byte, but the reader read bits 6-7 (the format version). Every huge
//! object was therefore mis-decoded as managed, producing a garbage heap offset
//! and a downstream `InvalidObjectHeaderVersion`/`UnexpectedEof`. A link or
//! attribute message larger than the heap's max managed object size (4096 bytes
//! for these heaps, reached with ~4 KiB names) is stored as a huge object and
//! exercises this path. Files are written by the reference C library and read
//! back through both the buffered and streaming readers.

use hdf5_pure::{AttrValue, File};
use std::path::Path;
use tempfile::tempdir;

/// A unique, deterministic name of about `len` bytes that starts with `d{i}_`.
fn long_name(i: usize, len: usize) -> String {
    let prefix = format!("d{i}_");
    let pad = len.saturating_sub(prefix.len());
    format!("{prefix}{}", "x".repeat(pad))
}

/// Write group `g` with one single-i32 dataset per name, value `i`, using the
/// latest format so the C library stores the links densely in a fractal heap.
fn write_group(path: &Path, names: &[String]) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_latest())
        .create(path)
        .unwrap();
    let g = file.create_group("g").unwrap();
    for (i, name) in names.iter().enumerate() {
        g.new_dataset::<i32>()
            .shape((1,))
            .create(name.as_str())
            .unwrap()
            .write(&[i as i32])
            .unwrap();
    }
    file.close().unwrap();
}

/// Assert every `g/{name}` resolves to its dataset value `i`, through both the
/// buffered and the streaming reader.
fn assert_links_resolve(path: &Path, names: &[String]) {
    let buffered = File::open(path).unwrap();
    let streaming = File::open_streaming(path).unwrap();
    for (i, name) in names.iter().enumerate() {
        let p = format!("g/{name}");
        assert_eq!(
            buffered.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "buffered link {i}"
        );
        assert_eq!(
            streaming.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "streaming link {i}"
        );
    }
}

#[test]
fn reads_dense_links_stored_as_huge_objects() {
    let dir = tempdir().unwrap();
    // 4 KiB names push every link message just past the heap's 4096-byte managed
    // limit, so each is a huge object; 40 entries give the huge-objects B-tree
    // several records to search.
    let names: Vec<String> = (0..40).map(|i| long_name(i, 4096)).collect();
    let path = dir.path().join("huge_links.h5");
    write_group(&path, &names);
    assert_links_resolve(&path, &names);
}

#[test]
fn reads_dense_links_mixing_managed_and_huge() {
    let dir = tempdir().unwrap();
    // Alternate short (managed) and very long (huge) names within one dense
    // group, so both heap-ID types are decoded from the same heap.
    let names: Vec<String> = (0..40)
        .map(|i| {
            if i % 2 == 0 {
                format!("short_{i}")
            } else {
                long_name(i, 5000)
            }
        })
        .collect();
    let path = dir.path().join("mixed_links.h5");
    write_group(&path, &names);
    assert_links_resolve(&path, &names);
}

#[test]
fn reads_dense_attributes_stored_as_huge_objects() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("huge_attrs.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_latest())
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((1,))
            .create("data")
            .unwrap();
        ds.write(&[7i32]).unwrap();
        // Enough attributes to force dense storage, each with a ~5 KiB name so
        // the whole attribute message is stored as a huge object.
        for i in 0..30 {
            let name = format!("a{i}_{}", "y".repeat(5000));
            ds.new_attr::<i64>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&(i as i64))
                .unwrap();
        }
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![7]);

    let attrs = ds.attrs().unwrap();
    assert_eq!(attrs.len(), 30, "all dense attributes resolved");
    for (name, value) in &attrs {
        // Name is "a{i}_yyyy..."; the value written was `i`.
        let i: i64 = name[1..name.find('_').unwrap()].parse().unwrap();
        assert_eq!(*value, AttrValue::I64(i), "attribute {name}");
    }
}
