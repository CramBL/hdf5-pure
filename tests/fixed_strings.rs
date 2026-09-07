//! Pure-Rust tests for the fixed-width string writers: the dataset entry points
//! (`DatasetBuilder::with_ascii_strings` and its three siblings, issue #355) and
//! the attribute values that declare a width (`AttrValue::ascii_string_sized`
//! and its three siblings, issue #359).
//!
//! The unit tests beside the builder assert what it stages; these assert what a
//! reader gets back out of a file, which is the other half of the same claim —
//! a datatype message and a run of element bytes that agree on a width are only
//! correct if `read_string` recovers the values from them.
//!
//! C-library interop lives in `crates/crosscheck/tests/fixed_string.rs` for the datasets and
//! in `crates/crosscheck/tests/attr_width.rs` for the attributes.

use hdf5_pure::{
    AttrValue, CharacterSet, Datatype, Error, File, FileBuilder, FormatError, StringPadding,
    VlenStringReadOptions,
};
use tempfile::tempdir;

/// Every fixed-width string entry point, through a file and back.
#[test]
fn fixed_width_strings_round_trip_through_a_file() {
    let values = ["north", "s", "", "east"];
    let mut b = FileBuilder::new();
    b.create_dataset("derived_ascii")
        .with_ascii_strings(&values)
        .unwrap();
    b.create_dataset("sized_ascii")
        .with_ascii_strings_sized(&values, 16)
        .unwrap();
    b.create_dataset("derived_utf8")
        .with_strings(&values)
        .unwrap();
    b.create_dataset("sized_utf8")
        .with_strings_sized(&values, 16)
        .unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    for (name, width, charset) in [
        ("derived_ascii", 5, CharacterSet::Ascii),
        ("sized_ascii", 16, CharacterSet::Ascii),
        ("derived_utf8", 5, CharacterSet::Utf8),
        ("sized_utf8", 16, CharacterSet::Utf8),
    ] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(
            ds.datatype().unwrap(),
            Datatype::String {
                size: width,
                padding: StringPadding::NullPad,
                charset,
            },
            "{name}"
        );
        assert_eq!(ds.shape().unwrap(), vec![4], "{name}");
        assert_eq!(ds.read_string().unwrap(), values, "{name}");
        // The windowed reader decodes straight from the raw window, a separate
        // path from the whole-dataset one.
        assert_eq!(ds.read_string_rows(1, 2).unwrap(), ["s", ""], "{name}");
    }
}

/// The width counts bytes, and a value whose characters are wider than a byte
/// still reads back whole. Written through the UTF-8 entry point because the
/// charset bit is what tells a reader the bytes are not one character each.
#[test]
fn a_utf8_value_is_measured_in_bytes_and_read_back_whole() {
    let values = ["mètre", "K", "°C"];
    let mut b = FileBuilder::new();
    b.create_dataset("units").with_strings(&values).unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    let ds = file.dataset("units").unwrap();
    // "mètre" is 6 bytes over 5 characters, and it sets the width.
    assert_eq!(
        ds.datatype().unwrap(),
        Datatype::String {
            size: 6,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Utf8,
        }
    );
    assert_eq!(ds.read_string().unwrap(), values);
}

/// The reason [`DatasetBuilder::with_ascii_strings_sized`] exists: a dataset can
/// be extended, and a width derived from the values in hand leaves a later,
/// longer value with nowhere to go. Declaring 16 up front makes the append fit.
///
/// The append itself pads by hand through `append_raw`, which is what a caller
/// does today — the typed append counterpart is not written yet.
#[test]
fn a_declared_width_leaves_room_for_a_later_longer_value() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stations.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("station")
        .with_ascii_strings_sized(&["north", "s"], 16)
        .unwrap()
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    // Longer than the derived width of 5 would have been, and it fits.
    let appended = ["north-northeast", "e"];
    {
        let session = File::open_rw(&path).unwrap();
        let mut raw = Vec::new();
        for value in appended {
            let mut element = value.as_bytes().to_vec();
            element.resize(16, 0);
            raw.extend_from_slice(&element);
        }
        session
            .dataset("station")
            .unwrap()
            .append_staged(|a| {
                a.append_raw(&raw);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("station").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(
        ds.read_string().unwrap(),
        ["north", "s", "north-northeast", "e"]
    );
}

/// A refusal leaves the file writable: the builder staged nothing, so the same
/// dataset name is still free for a call that fits.
#[test]
fn a_refused_width_leaves_the_builder_usable() {
    let mut b = FileBuilder::new();
    let ds = b.create_dataset("station");
    assert!(matches!(
        ds.with_ascii_strings_sized(&["north"], 2),
        Err(FormatError::FixedStringTooLong { index: 0, .. })
    ));
    assert!(matches!(
        ds.with_ascii_strings_sized(&["north"], 0),
        Err(FormatError::ZeroFixedStringWidth)
    ));
    ds.with_ascii_strings_sized(&["north"], 8).unwrap();

    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(
        file.dataset("station").unwrap().read_string().unwrap(),
        ["north"]
    );
}

/// A fixed-width dataset is not a variable-length one, and the reader that
/// bounds variable-length allocation says so rather than decoding the padding
/// as heap references.
#[test]
fn a_fixed_width_dataset_is_refused_by_the_vlen_reader() {
    let mut b = FileBuilder::new();
    b.create_dataset("station")
        .with_ascii_strings(&["north"])
        .unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    let ds = file.dataset("station").unwrap();
    assert!(
        matches!(
            ds.read_vlen_strings(VlenStringReadOptions::default()),
            Err(Error::Format(FormatError::TypeMismatch {
                expected: "VariableLength string",
                ..
            }))
        ),
        "expected a type mismatch naming the type it wanted"
    );
    // The charset-agnostic reader still works on it, which is the point of
    // `read_string` dispatching on the datatype.
    assert_eq!(ds.read_string().unwrap(), ["north"]);
}

/// Every value empty is the case a zero-width datatype would be the natural
/// answer to, and the one the C library cannot read. One padding byte per
/// element is what goes out instead, and it reads back as the empty string.
#[test]
fn all_empty_values_take_a_one_byte_datatype() {
    let mut b = FileBuilder::new();
    b.create_dataset("blank")
        .with_ascii_strings(&["", "", ""])
        .unwrap();
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    let ds = file.dataset("blank").unwrap();
    assert_eq!(
        ds.datatype().unwrap(),
        Datatype::String {
            size: 1,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        }
    );
    assert_eq!(ds.read_string().unwrap(), ["", "", ""]);
}

// ---- Attribute widths a caller declares (issue #359) ----

/// The `STRSIZE` an attribute declares, read back off the file.
fn attr_width(file: &File, dataset: &str, attr: &str) -> u32 {
    let types = file.dataset(dataset).unwrap().attr_datatypes().unwrap();
    match types.get(attr) {
        Some(Datatype::String { size, .. }) => *size,
        other => panic!("expected a string attribute datatype, got {other:?}"),
    }
}

/// A declared attribute width survives a whole file round trip: the value comes
/// back whole, without the padding, and the slot comes back the size it was
/// asked for.
#[test]
fn a_declared_attribute_width_round_trips_through_a_file() {
    let mut b = FileBuilder::new();
    b.create_dataset("reading")
        .with_i32_data(&[1])
        .set_attr("units", AttrValue::ascii_string_sized("ok", 64).unwrap())
        .set_attr(
            "labels",
            AttrValue::string_array_sized(vec!["north".into(), "s".into()], 16).unwrap(),
        );
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    assert_eq!(attr_width(&file, "reading", "units"), 64);
    assert_eq!(attr_width(&file, "reading", "labels"), 16);

    let attrs = file.dataset("reading").unwrap().attrs().unwrap();
    assert_eq!(attrs["units"].as_str(), Some("ok"));
    assert_eq!(attrs["labels"].as_strings().unwrap(), ["north", "s"]);
}

/// The issue's own case: a slot rewritten from a shorter value keeps its width
/// when the width is declared, and shrinks to the content when it is not.
///
/// Both halves are needed. The declared one alone would pass on a writer that
/// had stopped resizing anything at all, and the plain one is what says the
/// contrast is real rather than an artefact of how this test writes.
#[test]
fn a_declared_slot_keeps_its_width_when_it_is_rewritten() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("slots.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1])
        .set_attr("sized", AttrValue::ascii_string_sized("hello", 64).unwrap())
        .set_attr("plain", AttrValue::AsciiString("hello".into()));
    b.write(&path).unwrap();

    {
        let file = File::open(&path).unwrap();
        assert_eq!(attr_width(&file, "d", "sized"), 64);
        assert_eq!(attr_width(&file, "d", "plain"), 5);
    }

    {
        let session = File::open_rw(&path).unwrap();
        let mut dataset = session.dataset("d").unwrap();
        dataset
            .set_attr("sized", AttrValue::ascii_string_sized("x", 64).unwrap())
            .unwrap();
        dataset
            .set_attr("plain", AttrValue::AsciiString("x".into()))
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        attr_width(&file, "d", "sized"),
        64,
        "a declared width must survive being rewritten from a shorter value"
    );
    assert_eq!(
        attr_width(&file, "d", "plain"),
        1,
        "a derived width follows the content, which is what a declared one is for"
    );

    let attrs = file.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs["sized"].as_str(), Some("x"));
    assert_eq!(attrs["plain"].as_str(), Some("x"));
}

/// A slot read out of a file can be written back without knowing how wide it
/// was: the value carries the width, so a read-modify-write preserves it.
#[test]
fn a_slot_read_back_can_be_rewritten_without_measuring_it() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("reuse.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1]).set_attr(
        "units",
        AttrValue::ascii_string_sized("metres", 32).unwrap(),
    );
    b.write(&path).unwrap();

    // Read the value, keep only its width, and write a different string into it.
    let rewritten = {
        let file = File::open(&path).unwrap();
        let attrs = file.dataset("d").unwrap().attrs().unwrap();
        match &attrs["units"] {
            AttrValue::AsciiStringSized { width, .. } => {
                AttrValue::ascii_string_sized("feet", width.get()).unwrap()
            }
            other => panic!("expected a sized ASCII string, got {other:?}"),
        }
    };

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .set_attr("units", rewritten)
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(attr_width(&file, "d", "units"), 32);
    let attrs = file.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs["units"].as_str(), Some("feet"));
}

/// A group attribute takes a declared width through the edit path too.
///
/// `Group::set_attr` reaches the object header by a different route than
/// `Dataset::set_attr` — `apply_compact_attr_ops` rather than the dataset
/// rebuild — so the dataset tests above say nothing about it.
#[test]
fn a_group_attribute_keeps_a_declared_width_through_an_edit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("group_slot.h5");

    let mut b = FileBuilder::new();
    let mut g = b.create_group("run");
    g.create_dataset("d").with_i32_data(&[1]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("run")
            .unwrap()
            .set_attr("units", AttrValue::ascii_string_sized("ok", 48).unwrap())
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let group = file.group("run").unwrap();
    assert!(
        matches!(
            group.attr_datatypes().unwrap().get("units"),
            Some(Datatype::String { size: 48, .. })
        ),
        "a group attribute must keep its declared width, got {:?}",
        group.attr_datatypes().unwrap().get("units")
    );
    assert_eq!(group.attrs().unwrap()["units"].as_str(), Some("ok"));
}
