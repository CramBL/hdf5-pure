// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! The reference C library must agree about an enumeration this crate wrote over
//! a base type other than `i32`/`u8` (issue #208).
//!
//! `EnumTypeBuilder::with_base` reaches base types the old constructors could
//! not, and `Datatype::serialize` writes the base type and each member's raw
//! bytes generically. That is only correct if the C library recovers the same
//! base width, signedness, and member values — `hdf5-metno`'s `EnumType`
//! descriptor is built from `H5Tget_super` and `H5Tget_member_value`, so reading
//! it back exercises exactly those calls.
//!
//! Every call here goes through the safe `hdf5-metno` API, which serializes its
//! own C calls through an internal lock, so these tests need no extra guard.

use hdf5::types::TypeDescriptor;
use hdf5_pure::{EnumTypeBuilder, FileBuilder, make_i16_type, make_u16_type, make_u64_type};
use tempfile::tempdir;

/// Read the enumeration descriptor the C library recovers for dataset `name`.
fn c_enum_descriptor(path: &std::path::Path, name: &str) -> hdf5::types::EnumType {
    let file = hdf5::File::open(path).expect("the C library should open the file");
    let ds = file.dataset(name).expect("dataset should exist");
    match ds.dtype().unwrap().to_descriptor().unwrap() {
        TypeDescriptor::Enum(e) => e,
        other => panic!("the C library read a non-enumeration datatype: {other:?}"),
    }
}

#[test]
fn c_library_agrees_on_an_unsigned_16_bit_enum() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("u16_enum.h5");

    let dt = EnumTypeBuilder::with_base(make_u16_type())
        .value("low", 1)
        .value("high", 40_000)
        .build()
        .unwrap();

    let mut b = FileBuilder::new();
    let mut raw = Vec::new();
    raw.extend_from_slice(&1u16.to_le_bytes());
    raw.extend_from_slice(&40_000u16.to_le_bytes());
    b.create_dataset("e")
        .with_raw_data(dt, raw, 2)
        .with_shape(&[2]);
    b.write(&path).unwrap();

    let e = c_enum_descriptor(&path, "e");
    assert_eq!(e.size, hdf5::types::IntSize::U2, "H5Tget_super width");
    assert!(!e.signed, "an unsigned base must not read back as signed");
    let members: Vec<(String, u64)> = e
        .members
        .iter()
        .map(|m| (m.name.clone(), m.value))
        .collect();
    assert_eq!(
        members,
        vec![("low".to_string(), 1u64), ("high".to_string(), 40_000u64)],
        "H5Tget_member_value should recover both members verbatim"
    );
}

#[test]
fn c_library_agrees_on_a_signed_16_bit_enum() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("i16_enum.h5");

    let dt = EnumTypeBuilder::with_base(make_i16_type())
        .value("neg", -2)
        .value("pos", 7)
        .build()
        .unwrap();

    let mut b = FileBuilder::new();
    let mut raw = Vec::new();
    raw.extend_from_slice(&(-2i16).to_le_bytes());
    raw.extend_from_slice(&7i16.to_le_bytes());
    b.create_dataset("e")
        .with_raw_data(dt, raw, 2)
        .with_shape(&[2]);
    b.write(&path).unwrap();

    let e = c_enum_descriptor(&path, "e");
    assert_eq!(e.size, hdf5::types::IntSize::U2);
    assert!(e.signed, "a signed base must read back as signed");
    // `EnumMember::value` is a `u64` holding the raw bit pattern, so a signed
    // member must be reinterpreted at the base type's width to compare.
    let values: Vec<i16> = e.members.iter().map(|m| m.value as u16 as i16).collect();
    assert_eq!(
        values,
        vec![-2, 7],
        "a negative member value must survive the round trip"
    );
}

/// `raw_value` is the route for values no `i64` can express; the C library must
/// still recover the exact stored bytes.
#[test]
fn c_library_agrees_on_a_64_bit_enum_built_from_raw_bytes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("u64_enum.h5");

    let dt = EnumTypeBuilder::with_base(make_u64_type())
        .raw_value("max", &u64::MAX.to_le_bytes())
        .raw_value("zero", &0u64.to_le_bytes())
        .build()
        .unwrap();

    let mut b = FileBuilder::new();
    let mut raw = Vec::new();
    raw.extend_from_slice(&u64::MAX.to_le_bytes());
    raw.extend_from_slice(&0u64.to_le_bytes());
    b.create_dataset("e")
        .with_raw_data(dt, raw, 2)
        .with_shape(&[2]);
    b.write(&path).unwrap();

    let e = c_enum_descriptor(&path, "e");
    assert_eq!(e.size, hdf5::types::IntSize::U8);
    assert!(!e.signed);
    assert_eq!(
        e.members.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
        vec!["max".to_string(), "zero".to_string()]
    );
    // `u64::MAX` is read back through the descriptor's unsigned accessor.
    assert_eq!(e.members[0].value, u64::MAX);
    assert_eq!(e.members[1].value, 0);
}
