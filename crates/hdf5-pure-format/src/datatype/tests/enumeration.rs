use alloc::vec::Vec;

use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, DatatypeMemberName, EnumMember, EnumerationType, FixedPointType,
};

use super::super::header;
use super::super::message::version::DATATYPE_VERSION_THREE;
use super::super::semantic;
use super::super::semantic::DecodedDatatype;
use super::fixtures;

#[test]
fn parses_enumeration_base_names_and_values() {
    let mut buf = header::build_datatype_header(8, DATATYPE_VERSION_THREE, [3, 0, 0], 4);
    buf.extend_from_slice(&fixtures::build_fixed_point(4, false, true, 0, 32));
    // The fixture stores all names before the values.
    buf.extend_from_slice(b"RED\0");
    buf.extend_from_slice(b"GREEN\0");
    buf.extend_from_slice(b"BLUE\0");
    buf.extend_from_slice(&0i32.to_le_bytes());
    buf.extend_from_slice(&1i32.to_le_bytes());
    buf.extend_from_slice(&2i32.to_le_bytes());

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    match dt {
        Datatype::Enumeration(datatype) => {
            assert_eq!(
                datatype.base_type(),
                &Datatype::FixedPoint(FixedPointType::i32())
            );
            assert_eq!(datatype.size().get(), 4);
            let members = datatype.members();
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name.as_bytes(), b"RED");
            assert_eq!(members[0].value, 0i32.to_le_bytes().to_vec());
            assert_eq!(members[1].name.as_bytes(), b"GREEN");
            assert_eq!(members[1].value, 1i32.to_le_bytes().to_vec());
            assert_eq!(members[2].name.as_bytes(), b"BLUE");
            assert_eq!(members[2].value, 2i32.to_le_bytes().to_vec());
        }
        _ => panic!("expected Enumeration"),
    }
}

#[test]
fn serialize_parse_enum_roundtrip() {
    let dt = make_enum(
        FixedPointType::i32(),
        &[("RED", 0), ("GREEN", 1), ("BLUE", 2)],
    );
    let bytes = semantic::serialize_datatype(&dt).unwrap();
    let parsed = DecodedDatatype::from_record(&bytes)
        .unwrap()
        .into_datatype();
    assert_eq!(parsed, dt);
}

#[test]
fn serialize_parse_enum_base_type_variety() {
    for base in [
        fixed_point_enum_base(1, false, false), // u8
        fixed_point_enum_base(2, true, true),   // i16 big-endian
        fixed_point_enum_base(8, false, true),  // i64
    ] {
        let dt = make_enum(base, &[("A", 0), ("B", 1), ("NEG", -1)]);
        let bytes = semantic::serialize_datatype(&dt).unwrap();
        let (decoded, consumed) = DecodedDatatype::parse_prefix(&bytes).unwrap();
        let parsed = decoded.into_datatype();
        assert_eq!(parsed, dt, "round-trip failed for base {base:?}");
        assert_eq!(consumed, bytes.len());
    }
}

#[test]
fn serialize_parse_enum_large_member_count() {
    // The member count uses both bytes of the count field.
    let owned: Vec<(String, i64)> = (0..300).map(|i| (format!("M{i}"), i)).collect();
    let members: Vec<(&str, i64)> = owned.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let dt = make_enum(fixed_point_enum_base(4, false, true), &members);
    let bytes = semantic::serialize_datatype(&dt).unwrap();
    let parsed = DecodedDatatype::from_record(&bytes)
        .unwrap()
        .into_datatype();
    assert_eq!(parsed, dt);
    match parsed {
        Datatype::Enumeration(datatype) => {
            let members = datatype.members();
            assert_eq!(members.len(), 300);
            assert_eq!(members[299].name.as_bytes(), b"M299");
        }
        other => panic!("expected Enumeration, got {other:?}"),
    }
}

/// Builds a fixed-point base with the full element width as its precision.
///
/// # Panics
///
/// Panics if `size` is zero or the computed bit precision exceeds [`u16::MAX`].
fn fixed_point_enum_base(size: u32, be: bool, signed: bool) -> FixedPointType {
    FixedPointType::new(
        size,
        if be {
            DatatypeByteOrder::BigEndian
        } else {
            DatatypeByteOrder::LittleEndian
        },
        signed,
        0,
        u16::try_from(size * 8).unwrap(),
    )
    .unwrap()
}

/// Stores the low bytes of each value in little-endian order up to the base width.
///
/// # Panics
///
/// Panics if a member name contains a NUL byte or if `members` is nonempty and the base
/// is wider than eight bytes.
fn make_enum(base: FixedPointType, members: &[(&str, i64)]) -> Datatype {
    let width = base.size().get() as usize;
    let members = members
        .iter()
        .map(|(name, value)| {
            EnumMember::new(
                DatatypeMemberName::new(name).unwrap(),
                value.to_le_bytes()[..width].to_vec(),
            )
        })
        .collect();
    Datatype::Enumeration(EnumerationType::new(base, members).unwrap())
}
