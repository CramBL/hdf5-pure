//! An attribute stored compactly lives in its object header, whose per-message
//! size field is 2 bytes wide. The whole-file writer used to cast the message
//! length into that field unchecked, so an oversized attribute either vanished
//! from the file it was written to or left the header unparseable (issue #190).
//!
//! Since #195 an oversized attribute is not refused but moved to a fractal heap,
//! which has no such field, so these pin the *write*. Variable-length data
//! included: it was briefly the one exception, until the writer learned to build
//! each heap after the global-heap references it copies have been patched. What
//! still overflows the field is a message with no dense alternative, and the last
//! test here is that one.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError, OBJECT_HEADER_MESSAGE_MAX};

use test_util::heap::has_fractal_heap;

/// A `VarLenAsciiCharArray` past the message-size limit: each element contributes a
/// 16-byte global-heap reference, so ~4,100 elements cross it. This is the shape
/// from the issue report, where the attribute silently disappeared.
fn oversized_vlen() -> AttrValue {
    AttrValue::VarLenAsciiCharArray((0..4_200).map(|i| format!("s{i}")).collect())
}

/// A fixed-width string array past the message-size limit. No variable-length
/// data is involved, so this is not specific to the heap-backed shape above; in
/// the issue report the truncated length left readers parsing a message body as
/// a message header.
fn oversized_fixed() -> AttrValue {
    AttrValue::StringArray(vec!["x".repeat(200); 400])
}

/// The strings of an array-of-strings attribute, whichever variant the reader
/// chose for it.
#[track_caller]
fn strings(attrs: &std::collections::HashMap<String, AttrValue>, name: &str) -> Vec<String> {
    match attrs.get(name) {
        Some(AttrValue::VarLenAsciiCharArray(v) | AttrValue::StringArray(v)) => v.clone(),
        other => panic!("expected an array-of-strings attribute {name:?}, got {other:?}"),
    }
}

/// The shape from the issue report, which is variable-length and so was the last
/// one to be refused on size. Its element bytes hold global-heap references, and
/// the heap it now goes into copies them, so this reads back only if the writer
/// builds that heap after those references have real addresses.
#[test]
fn root_vlen_attr_past_the_limit_moves_to_heap_storage() {
    let expected = match oversized_vlen() {
        AttrValue::VarLenAsciiCharArray(v) => v,
        other => panic!("fixture changed shape: {other:?}"),
    };

    let mut builder = FileBuilder::new();
    builder.set_attr("labels", oversized_vlen());
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().expect("written, not refused");
    assert!(has_fractal_heap(&bytes));

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(strings(&file.root().attrs().unwrap(), "labels"), expected);
}

/// A fixed-width attribute past the limit is no longer refused: it selects heap
/// storage, where the 2-byte field does not apply, and comes back intact. On
/// every object that carries attributes, since each reaches the writer's storage
/// choice by a different route.
#[test]
fn fixed_width_attrs_past_the_limit_move_to_heap_storage() {
    let expected = match oversized_fixed() {
        AttrValue::StringArray(v) => v,
        other => panic!("fixture changed shape: {other:?}"),
    };

    let mut builder = FileBuilder::new();
    builder.set_attr("big", oversized_fixed());
    builder
        .create_dataset("x")
        .with_f64_data(&[1.0])
        .set_attr("big", oversized_fixed());
    let mut group = builder.create_group("g");
    group.set_attr("big", oversized_fixed());
    group.create_dataset("y").with_f64_data(&[2.0]);
    builder.add_group(group.finish());

    let bytes = builder.finish().expect("written, not refused");
    assert!(has_fractal_heap(&bytes));

    let file = File::from_bytes(bytes).unwrap();
    for attrs in [
        file.root().attrs().unwrap(),
        file.dataset("x").unwrap().attrs().unwrap(),
        file.group("g").unwrap().attrs().unwrap(),
    ] {
        match attrs.get("big") {
            Some(AttrValue::StringArray(v)) => assert_eq!(*v, expected),
            other => panic!("oversized attribute came back as {other:?}"),
        }
    }
}

/// `write` refuses on the same terms as `finish`, rather than only `finish`, and
/// the refusal costs the destination path nothing.
///
/// It used to open the file with `std::fs::File::create` before serializing —
/// which truncates — so a refusal left an empty file where the caller's own file
/// had been. Now the path is created when the writer has bytes for it, and this
/// refusal fires before any. Uses the one shape still refused, since no
/// attribute size is.
#[test]
fn write_to_disk_refuses_without_touching_the_destination() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oversized.h5");
    let previous = b"the caller's existing file";
    std::fs::write(&path, previous).unwrap();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset(&"a".repeat(70_000))
        .with_f64_data(&[1.0]);

    assert!(matches!(
        builder.write(&path).unwrap_err(),
        Error::Format(FormatError::ObjectHeaderMessageTooLarge { .. })
    ));
    assert_eq!(
        std::fs::read(&path).unwrap(),
        previous,
        "the refused build overwrote the destination"
    );
}

/// A *small* variable-length attribute swept into heap storage by an oversized
/// neighbour. It has no size reason of its own to be there, so nothing about it
/// prompts the writer to treat its references specially — the ordering has to
/// hold for whatever lands in the heap, not just for the attribute that put it
/// there.
#[test]
fn a_small_vlen_attribute_swept_into_the_heap_keeps_its_values() {
    let mut builder = FileBuilder::new();
    builder.set_attr("big", oversized_fixed());
    builder.set_attr(
        "labels",
        AttrValue::VarLenAsciiCharArray(vec!["a".to_string(), "b".to_string()]),
    );
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().expect("written, not refused");
    assert!(has_fractal_heap(&bytes));

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 2);
    assert_eq!(strings(&attrs, "labels"), ["a", "b"]);
}

/// An oversized attribute is not the only way to overflow the message-size
/// field: a long enough dataset name does it through the Link message. That has
/// no name-carrying check in front of it, so it lands on the writer's backstop —
/// which must still refuse rather than truncate.
#[test]
fn oversized_link_message_hits_the_writer_backstop() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset(&"a".repeat(70_000))
        .with_f64_data(&[1.0]);

    match builder.finish().unwrap_err() {
        Error::Format(FormatError::ObjectHeaderMessageTooLarge { size, .. }) => {
            assert!(size > OBJECT_HEADER_MESSAGE_MAX);
        }
        other => panic!("expected ObjectHeaderMessageTooLarge, got {other:?}"),
    }
}

/// The refusal must bite only past the limit: a large attribute that still fits
/// the size field round-trips, on every object that carries attributes.
#[test]
fn large_but_fitting_attrs_still_round_trip() {
    // 4,000 variable-length elements sit just under the limit; the issue reports
    // this exact count as the last one that worked.
    let labels: Vec<String> = (0..4_000).map(|i| format!("s{i}")).collect();
    let expected = labels.clone();

    let mut builder = FileBuilder::new();
    builder.set_attr("labels", AttrValue::VarLenAsciiCharArray(labels.clone()));
    builder
        .create_dataset("x")
        .with_f64_data(&[1.0])
        .set_attr("labels", AttrValue::VarLenAsciiCharArray(labels.clone()));
    let mut group = builder.create_group("g");
    group.set_attr("labels", AttrValue::VarLenAsciiCharArray(labels));
    group.create_dataset("y").with_f64_data(&[2.0]);
    builder.add_group(group.finish());

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    // Values, not just the count: a variable-length attribute whose references
    // were never resolved comes back *present* and empty in some readers, so a
    // length check alone would pass on data that had been lost.
    for attrs in [
        file.root().attrs().unwrap(),
        file.dataset("x").unwrap().attrs().unwrap(),
        file.group("g").unwrap().attrs().unwrap(),
    ] {
        assert_eq!(attrs.len(), 1);
        assert_eq!(strings(&attrs, "labels"), expected);
    }
}
