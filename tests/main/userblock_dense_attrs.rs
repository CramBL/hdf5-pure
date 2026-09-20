//! Dense (fractal-heap) attributes in a file that has a userblock.
//!
//! Every address stored in an object header is relative to the superblock's base
//! address, which for a userblock file is where the userblock ends. The attribute
//! walk used to take the Attribute Info message's fractal-heap address as an
//! absolute file offset, so on such a file it looked one userblock too early and
//! returned `InvalidFractalHeapSignature` — for a file the reference C library
//! reads correctly, and that this crate had just written.
//!
//! Nothing paired the two before: dense storage needed more than eight
//! attributes, and the userblock tests all used a handful. It became much easier
//! to reach once a single oversized attribute could select dense storage on its
//! own (#195).

use hdf5_pure::{AttrValue, File, FileBuilder, RepackOptions, repack};
use tempfile::tempdir;

use test_util::heap::has_fractal_heap;

const USERBLOCK: u64 = 512;

/// Attributes numerous enough to select dense storage by count.
fn many() -> Vec<(String, i64)> {
    (0..12).map(|i| (format!("a{i:02}"), 100 + i)).collect()
}

/// One attribute too large for an object-header message, which selects dense
/// storage on size alone — the shape #195 made reachable with no filler
/// attributes at all.
fn one_big() -> AttrValue {
    AttrValue::StringArray(vec!["y".repeat(200); 400])
}

/// A userblock file whose root, one group and one dataset all carry dense
/// attributes, so every object kind is covered by one file.
fn write_source(path: &std::path::Path) {
    let mut builder = FileBuilder::new();
    builder.with_userblock(USERBLOCK);
    builder.set_attr("big", one_big());
    {
        let dataset = builder.create_dataset("payload").with_f64_data(&[1.5, 2.5]);
        for (name, value) in many() {
            dataset.set_attr(&name, AttrValue::I64(value));
        }
    }
    {
        let mut group = builder.create_group("bundle");
        for (name, value) in many() {
            group.set_attr(&name, AttrValue::I64(value));
        }
        group.create_dataset("inner").with_i32_data(&[7, 8]);
        builder.add_group(group.finish());
    }
    builder.write(path).unwrap();

    let bytes = std::fs::read(path).unwrap();
    assert!(
        has_fractal_heap(&bytes),
        "fixture must use dense storage, or it proves nothing"
    );
}

/// Every attribute of `file`'s three objects, by name and value.
#[track_caller]
fn assert_all_attrs_readable(file: &File) {
    let expected: std::collections::HashMap<String, i64> = many().into_iter().collect();

    let root = file.root().attrs().unwrap();
    match root.get("big") {
        Some(AttrValue::StringArray(v)) => assert_eq!(v.len(), 400),
        other => panic!("root's size-selected dense attribute came back as {other:?}"),
    }

    for attrs in [
        file.dataset("payload").unwrap().attrs().unwrap(),
        file.group("bundle").unwrap().attrs().unwrap(),
    ] {
        let got: std::collections::HashMap<String, i64> = attrs
            .iter()
            .filter_map(|(k, v)| match v {
                AttrValue::I64(n) => Some((k.clone(), *n)),
                _ => None,
            })
            .collect();
        assert_eq!(got, expected);
    }
}

#[test]
fn buffered_reader_reads_dense_attributes_past_a_userblock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_dense.h5");
    write_source(&path);

    assert_all_attrs_readable(&File::open(&path).unwrap());
    assert_all_attrs_readable(&File::from_bytes(std::fs::read(&path).unwrap()).unwrap());
}

/// The streaming backend reaches the heap through a `Source` rather than a slice,
/// so it needs its own framing and is worth its own test.
#[test]
fn streaming_reader_reads_dense_attributes_past_a_userblock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_dense_stream.h5");
    write_source(&path);

    assert_all_attrs_readable(&File::open_streaming(&path).unwrap());
}

/// Repack reads every attribute through the reader and refuses rather than drop
/// one it cannot reproduce, so on a userblock file it used to fail outright.
#[test]
fn repack_carries_dense_attributes_past_a_userblock() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("ub_dense_src.h5");
    let dst = dir.path().join("ub_dense_packed.h5");
    write_source(&src);

    repack(&src, &dst, &RepackOptions::new()).unwrap();
    assert_all_attrs_readable(&File::open(&dst).unwrap());
}

/// The editor's copy reads the source attributes out of its heap, and the source
/// here is the userblock file itself.
#[test]
fn same_file_copy_reproduces_dense_attributes_past_a_userblock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_dense_copy.h5");
    write_source(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("payload", "payload_copy").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_all_attrs_readable(&file);
    let expected: std::collections::HashMap<String, i64> = many().into_iter().collect();
    let copied = file.dataset("payload_copy").unwrap().attrs().unwrap();
    let got: std::collections::HashMap<String, i64> = copied
        .iter()
        .filter_map(|(k, v)| match v {
            AttrValue::I64(n) => Some((k.clone(), *n)),
            _ => None,
        })
        .collect();
    assert_eq!(got, expected);
    assert_eq!(
        file.dataset("payload_copy").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5]
    );
}

/// *Cross-file* copy refuses a userblock source before it reaches any attribute,
/// so the base address never comes up there. Pinned so that the day the refusal is
/// lifted, this fails rather than quietly starting to read the heap at the wrong
/// offset — the dense read behind it is framed for a base address it has not yet
/// been given the chance to have.
#[test]
fn cross_file_copy_still_refuses_a_userblock_source() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("ub_dense_xsrc.h5");
    let dst = dir.path().join("plain_xdst.h5");
    write_source(&src);
    {
        let mut builder = FileBuilder::new();
        builder.create_dataset("keep").with_f64_data(&[0.0]);
        builder.write(&dst).unwrap();
    }

    let source = File::open(&src).unwrap();
    let session = File::open_rw(&dst).unwrap();
    let err = session
        .copy_from(&source, "payload", "payload")
        .expect_err("a userblock source is refused cross-file");
    let msg = err.to_string();
    assert!(
        msg.contains("userblock"),
        "unexpected refusal message: {msg}"
    );
}
