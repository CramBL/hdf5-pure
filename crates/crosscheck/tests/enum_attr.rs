// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! An enumeration *attribute* written by the reference C library must reach the
//! caller, decoded through the enum's integer base type (#248).
//!
//! This is the read direction of `crates/crosscheck/tests/enum_base.rs`, and it matters because
//! the C library is what writes the files in the wild: `H5T_NATIVE_HBOOL` — what
//! h5py gives every `np.bool_` and what `hdf5-metno` gives Rust's `bool` — is an
//! `enum[FALSE, TRUE]` over an 8-bit integer, so a boolean attribute is an enum
//! attribute. `attrs()` used to drop those silently, taking with it flags like
//! DROID's per-episode `success`.
//!
//! Every call here goes through the safe `hdf5-metno` API, which serializes its
//! own C calls through an internal lock, so these tests need no extra guard.

use std::collections::{BTreeMap, HashMap};

use hdf5_pure::{AttrValue, Datatype, File};
use tempfile::tempdir;

/// Write attributes with the C library, then open the file with this crate.
fn c_writes(write: impl FnOnce(&hdf5::File)) -> File {
    let dir = tempdir().unwrap();
    let path = dir.path().join("enum_attrs.h5");

    {
        let file = hdf5::File::create(&path).expect("the C library should create the file");
        write(&file);
        file.close().expect("close");
    }

    let bytes = std::fs::read(&path).expect("read back the written file");
    File::from_bytes(bytes).expect("from_bytes")
}

/// Write attributes with the C library, then read them all back with this crate.
fn c_writes_then_pure_reads(write: impl FnOnce(&hdf5::File)) -> HashMap<String, AttrValue> {
    c_writes(write).root().attrs().expect("attrs")
}

#[test]
fn a_c_written_bool_attribute_reaches_the_caller() {
    let attrs = c_writes_then_pure_reads(|file| {
        file.new_attr::<bool>()
            .create("flag")
            .expect("create attribute")
            .write_scalar(&true)
            .expect("write attribute");
    });

    // The codes survive; `FALSE`/`TRUE` do not, since no `AttrValue` carries
    // them. The width does: the enum's base is one signed byte (#350).
    assert_eq!(attrs.get("flag"), Some(&AttrValue::I8(1)));
}

#[test]
fn a_c_written_bool_array_attribute_reaches_the_caller() {
    let attrs = c_writes_then_pure_reads(|file| {
        file.new_attr::<bool>()
            .shape([3])
            .create("flags")
            .expect("create attribute")
            .write(&[true, false, true])
            .expect("write attribute");
    });

    assert_eq!(attrs.get("flags"), Some(&AttrValue::I8Array(vec![1, 0, 1])));
}

/// A boolean attribute must not cost the object its other attributes: the decoder
/// skips what it cannot represent, so a regression here would be silent.
#[test]
fn a_bool_attribute_sits_alongside_the_rest() {
    let attrs = c_writes_then_pure_reads(|file| {
        file.new_attr::<bool>()
            .create("success")
            .expect("create attribute")
            .write_scalar(&false)
            .expect("write attribute");
        file.new_attr::<f64>()
            .create("frequency")
            .expect("create attribute")
            .write_scalar(&30.0)
            .expect("write attribute");
    });

    assert_eq!(attrs.get("success"), Some(&AttrValue::I8(0)));
    assert_eq!(attrs.get("frequency"), Some(&AttrValue::F64(30.0)));
}

/// The datatype channel is what says an attribute was a boolean.
///
/// Decoding through the base type is what makes the value readable, and it is
/// also what makes `true` and `1i8` indistinguishable in the value alone — both
/// arrive as `AttrValue::I8(1)`, which the last assertion here states rather
/// than assumes. `H5T_NATIVE_HBOOL`'s `enum[FALSE, TRUE]` over one byte is the
/// only record of which was written, and `attr_datatypes` is the only way to
/// read it. This is the shape a consumer matches on to map an h5py `np.bool_`
/// attribute onto a boolean column.
#[test]
fn a_c_written_bool_attribute_keeps_its_enumeration_in_the_datatype_channel() {
    let file = c_writes(|file| {
        file.new_attr::<bool>()
            .create("flag")
            .expect("create attribute")
            .write_scalar(&true)
            .expect("write attribute");
        file.new_attr::<i8>()
            .create("count")
            .expect("create attribute")
            .write_scalar(&1i8)
            .expect("write attribute");
    });
    let datatypes = file.root().attr_datatypes().expect("attr_datatypes");

    let Some(Datatype::Enumeration {
        size,
        base_type,
        members,
    }) = datatypes.get("flag")
    else {
        panic!(
            "expected an enumeration for the bool, got {:?}",
            datatypes.get("flag")
        );
    };
    assert_eq!(*size, 1, "the enum's own width");
    assert!(
        matches!(**base_type, Datatype::FixedPoint { size: 1, .. }),
        "the base is the 8-bit integer the codes are stored as, got {base_type:?}"
    );
    let mapping: BTreeMap<&str, &[u8]> = members
        .iter()
        .map(|m| (m.name.as_str(), m.value.as_slice()))
        .collect();
    assert_eq!(
        mapping,
        BTreeMap::from([("FALSE", &[0u8][..]), ("TRUE", &[1u8][..])]),
        "the member names and codes are what identify the boolean convention"
    );

    // The one-byte integer beside it is what a bool has to be told apart *from*,
    // and the value channel cannot do it.
    assert!(
        matches!(
            datatypes.get("count"),
            Some(Datatype::FixedPoint { size: 1, .. })
        ),
        "expected a plain 8-bit integer for the i8, got {:?}",
        datatypes.get("count")
    );
    let values = file.root().attrs().expect("attrs");
    assert_eq!(
        values.get("flag"),
        values.get("count"),
        "if these ever differ, the value channel can tell a bool from an i8 on \
         its own and this test is measuring the wrong thing"
    );
}
