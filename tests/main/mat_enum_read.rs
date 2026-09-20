//! Reading MATLAB enumeration arrays into the typed [`MatEnum`] view, over a
//! real MATLAB fixture (BSD-3, `foreverallama/matio`; see
//! `tests/data/matlab/NOTICE.md`). Expected values come from the generator
//! `tests/data/generators/test_enum_gen.m` and the `test_enum.py` oracle.
#![cfg(feature = "serde")]

use hdf5_pure::mat::{self, MatEnum};
use serde::Deserialize;

fn read<T: serde::de::DeserializeOwned>() -> T {
    let bytes = std::fs::read("tests/data/matlab/test_enum_v73.mat")
        .expect("read test_enum_v73.mat fixture");
    mat::from_bytes(&bytes).expect("decode fixture")
}

#[test]
fn enum_scalar_decodes_to_member_name() {
    #[derive(Deserialize)]
    struct File {
        enum_scalar: MatEnum,
    }
    let e = read::<File>().enum_scalar;
    assert_eq!(e.class_name, "TestClasses.EnumClass");
    assert_eq!(e.names, ["enum1"]);
    assert_eq!(e.len(), 1);
    assert!(!e.is_empty());
}

#[test]
fn enum_with_integer_base_class() {
    // `EnumClassWithBase` derives from `uint32`; the member name is still the
    // identity surfaced, and the class name reflects the derived class.
    #[derive(Deserialize)]
    struct File {
        enum_uint32: MatEnum,
    }
    let e = read::<File>().enum_uint32;
    assert_eq!(e.class_name, "TestClasses.EnumClassWithBase");
    assert_eq!(e.names, ["enum1"]);
}

#[test]
fn enum_array_member_names_are_row_major() {
    // `enum_array` is a 2x3 array laid out (column-major) as
    // [[enum1, enum3, enum5], [enum2, enum4, enum6]]; presented row-major it
    // reads across each row in turn.
    #[derive(Deserialize)]
    struct File {
        enum_array: MatEnum,
    }
    let e = read::<File>().enum_array;
    assert_eq!(e.class_name, "TestClasses.EnumClass");
    assert_eq!(
        e.names,
        ["enum1", "enum3", "enum5", "enum2", "enum4", "enum6"]
    );
    assert_eq!(e.len(), 6);
}

#[test]
fn enum_nested_in_object_cell_and_struct() {
    // `enum_nested` is a user `classdef` (`BasicClass`) holding an enum in three
    // positions: a direct property `a`, a `1x1` cell `b`, and a struct field
    // `c.InnerProp`. Each position must decode to a `MatEnum`, proving the
    // enumeration detection fires wherever a group is read.
    #[derive(Deserialize)]
    struct Nested {
        a: MatEnum,
        b: Vec<MatEnum>,
        c: Inner,
    }
    #[derive(Deserialize)]
    struct Inner {
        #[serde(rename = "InnerProp")]
        inner_prop: MatEnum,
    }
    #[derive(Deserialize)]
    struct File {
        enum_nested: Nested,
    }
    let n = read::<File>().enum_nested;
    assert_eq!(n.a.names, ["enum1"]);
    assert_eq!(n.a.class_name, "TestClasses.EnumClass");

    assert_eq!(n.b.len(), 1);
    assert_eq!(n.b[0].names, ["enum2"]);

    assert_eq!(n.c.inner_prop.names, ["enum3"]);
    assert_eq!(n.c.inner_prop.class_name, "TestClasses.EnumClass");
}

#[test]
fn enum_deserializes_into_a_plain_struct() {
    // The decoded enumeration exposes exactly `class_name` and `names`, so a
    // `deny_unknown_fields` struct of just those fields deserializes: no reserved
    // or metadata keys leak into ordinary targets.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Plain {
        class_name: String,
        names: Vec<String>,
    }
    #[derive(Deserialize)]
    struct File {
        enum_scalar: Plain,
    }
    let p = read::<File>().enum_scalar;
    assert_eq!(p.class_name, "TestClasses.EnumClass");
    assert_eq!(p.names, ["enum1"]);
}
