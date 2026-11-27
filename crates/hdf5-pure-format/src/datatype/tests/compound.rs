use alloc::boxed::Box;
use alloc::vec;

use byteorder::{ByteOrder, LittleEndian};
use hdf5_pure_core::{
    CompoundMember, Datatype, DatatypeByteOrder, DatatypeMemberName, FixedPointType,
    FloatingPointLayout, FloatingPointType, FormatError,
};
use rstest::rstest;

use super::super::header::{self, ARRAY_CLASS, COMPOUND_CLASS, DATATYPE_SIZE_BYTES};
use super::super::message::compound_layout;
use super::super::message::version::{
    DATATYPE_VERSION_FIVE, DATATYPE_VERSION_ONE, DATATYPE_VERSION_THREE, DATATYPE_VERSION_TWO,
};
use super::super::semantic;
use super::super::semantic::DecodedDatatype;
use super::fixtures;

#[rstest]
#[case::minimum_size(1, 1)]
#[case::one_byte_limit(255, 1)]
#[case::two_byte_start(256, 2)]
#[case::two_byte_limit(65_535, 2)]
#[case::three_byte_start(65_536, 3)]
#[case::three_byte_interior(70_000, 3)]
#[case::three_byte_limit(16_777_215, 3)]
#[case::four_byte_start(16_777_216, 4)]
#[case::four_byte_limit(u32::MAX, 4)]
fn compound_offsets_use_the_minimum_byte_width(#[case] size: u32, #[case] width: usize) {
    let datatype = Datatype::Compound {
        size,
        members: vec![CompoundMember::new(
            DatatypeMemberName::from(c"last"),
            u64::from(size - 1),
            fixtures::byte_member_type(),
        )],
    };
    let mut expected =
        header::build_datatype_header(COMPOUND_CLASS, DATATYPE_VERSION_THREE, [1, 0, 0], size);
    expected.extend_from_slice(b"last\0");
    expected.extend_from_slice(&(size - 1).to_le_bytes()[..width]);
    expected
        .extend_from_slice(&semantic::serialize_datatype(&fixtures::byte_member_type()).unwrap());

    assert_eq!(compound_layout::offset_bytes_for_size(size), width);
    assert_eq!(semantic::serialize_datatype(&datatype).unwrap(), expected);
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&expected).unwrap();
    assert_eq!(
        (decoded.into_datatype(), bytes_read),
        (datatype, expected.len())
    );
}

#[rstest]
#[case::compound_v3(COMPOUND_CLASS, DATATYPE_VERSION_THREE)]
#[case::compound_v4(COMPOUND_CLASS, DATATYPE_VERSION_FOUR)]
#[case::compound_v5(COMPOUND_CLASS, DATATYPE_VERSION_FIVE)]
fn three_byte_member_offsets_preserve_the_next_datatype(#[case] class: u8, #[case] version: u8) {
    let member = fixtures::byte_member_type();
    let mut encoded = header::build_datatype_header(class, version, [2, 0, 0], 70_000);
    encoded.extend_from_slice(b"first\0\0\0\0");
    encoded.extend_from_slice(&semantic::serialize_datatype(&member).unwrap());
    encoded.extend_from_slice(b"last\0\x6f\x11\x01");
    encoded.extend_from_slice(&semantic::serialize_datatype(&member).unwrap());
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&encoded).unwrap();
    assert_eq!(
        (decoded.into_datatype(), bytes_read),
        (
            Datatype::Compound {
                size: 70_000,
                members: vec![
                    CompoundMember::new(DatatypeMemberName::from(c"first"), 0, member.clone()),
                    CompoundMember::new(DatatypeMemberName::from(c"last"), 69_999, member),
                ],
            },
            encoded.len(),
        )
    );
}

#[test]
fn parses_version_three_compound_member_names_offsets_and_types() {
    // Each version 3 member offset occupies one byte for this compound size.
    let mut buf = header::build_datatype_header(6, DATATYPE_VERSION_THREE, [2, 0, 0], 12);
    buf.extend_from_slice(b"x\0");
    buf.push(0);
    buf.extend_from_slice(&fixtures::build_fixed_point(4, false, false, 0, 32));
    buf.extend_from_slice(b"y\0");
    buf.push(4);
    buf.extend_from_slice(&fixtures::build_float(8, 52, 11, 0, 52, 1023));

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 12);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name.as_bytes(), b"x");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name.as_bytes(), b"y");
            assert_eq!(members[1].byte_offset, 4);
            match &members[0].datatype {
                Datatype::FixedPoint(numeric)
                    if numeric.size().get() == 4 && !numeric.layout().signed() => {}
                other => panic!("expected u32, got {other:?}"),
            }
            match &members[1].datatype {
                Datatype::FloatingPoint(numeric) if numeric.size().get() == 8 => {}
                other => panic!("expected f64, got {other:?}"),
            }
        }
        _ => panic!("expected Compound"),
    }
}

#[test]
fn parses_version_one_compound_with_real_and_imaginary_members() {
    let mut buf = header::build_datatype_header(6, DATATYPE_VERSION_ONE, [2, 0, 0], 16);
    // Each name in this fixture occupies eight bytes. The dimension block occupies 28.
    for (name, offset) in [(&b"real\0\0\0\0"[..], 0u32), (&b"imag\0\0\0\0"[..], 8)] {
        buf.extend_from_slice(name);
        let mut off = [0u8; 4];
        LittleEndian::write_u32(&mut off, offset);
        buf.extend_from_slice(&off);
        buf.extend_from_slice(&[0u8; 28]);
        buf.extend_from_slice(&fixtures::build_float(8, 52, 11, 0, 52, 1023));
    }

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 16);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name.as_bytes(), b"real");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name.as_bytes(), b"imag");
            assert_eq!(members[1].byte_offset, 8);
            for m in &members {
                assert!(
                    matches!(m.datatype, Datatype::FloatingPoint(numeric) if numeric.size().get() == 8),
                    "expected f64 member, got {:?}",
                    m.datatype
                );
            }
        }
        _ => panic!("expected Compound"),
    }
}

#[rstest]
#[case::rank_one(1, [4, 0, 0, 0], 0, 16)]
#[case::rank_two(2, [2, 3, 0, 0], 4, 24)]
#[case::largest_i32_array(1, [u32::MAX / 4, 0, 0, 0], 0, u32::MAX - 3)]
#[case::rank_three(3, [2, 3, 4, 0], 0, 96)]
#[case::rank_four(4, [2, 3, 4, 5], 0, 480)]
#[case::unused_dimensions(1, [4, u32::MAX, 7, 0], 0, 16)]
fn compound_v1_array_dimensions(
    #[case] rank: u8,
    #[case] dimensions: [u32; 4],
    #[case] offset: u32,
    #[case] footprint: u32,
) {
    let base = fixtures::build_fixed_point(4, false, true, 0, 32);
    let buf = test_util::legacy_compound::raw_member(
        footprint + offset,
        offset,
        rank,
        0,
        dimensions,
        &base,
    );
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&buf).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(consumed, buf.len());
    let Datatype::Compound { size, members } = &datatype else {
        panic!("expected compound, got {datatype:?}");
    };
    assert_eq!(*size, footprint + offset);
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name.as_bytes(), b"arr");
    assert_eq!(members[0].byte_offset, u64::from(offset));
    assert_eq!(members[0].datatype.type_size(), footprint);
    let Datatype::Array {
        base_type,
        dimensions: parsed,
    } = &members[0].datatype
    else {
        panic!("expected array, got {:?}", members[0].datatype);
    };
    assert_eq!(parsed, &dimensions[..usize::from(rank)]);
    assert_eq!(
        **base_type,
        Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap()
        )
    );
    let encoded = semantic::serialize_datatype(&datatype).unwrap();
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&encoded).unwrap();
    let reparsed = decoded.into_datatype();
    assert_eq!(consumed, encoded.len());
    assert_eq!(reparsed, datatype);
}

#[test]
fn compound_v1_scalar_ignores_dimension_slots() {
    let base = fixtures::build_fixed_point(4, false, true, 0, 32);
    let buf = test_util::legacy_compound::raw_member(8, 4, 0, 0, [9, 0, 7, 6], &base);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&buf).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(consumed, buf.len());
    assert_eq!(
        datatype,
        Datatype::Compound {
            size: 8,
            members: vec![CompoundMember::new(
                DatatypeMemberName::from(c"arr"),
                4,
                DecodedDatatype::from_record(&base).unwrap().into_datatype()
            )],
        }
    );
}

#[rstest]
#[case::unsupported_rank(5, [1; 4], 0, FormatError::InvalidCompoundV1Rank(5))]
#[case::zero_first_extent(1, [0; 4], 0, FormatError::InvalidCompoundV1Extent { dimension: 0 })]
#[case::zero_later_extent(4, [1, 1, 1, 0], 0, FormatError::InvalidCompoundV1Extent { dimension: 3 })]
#[case::overflow(2, [65_536, 65_536, 0, 0], 0, FormatError::CompoundArraySizeOverflow { size: 262_144, extent: 65_536 })]
#[case::nonzero_permutation(1, [4, 0, 0, 0], 1, FormatError::UnsupportedCompoundV1Permutation(1))]
#[case::scalar_permutation(0, [0; 4], 0x0102_0304, FormatError::UnsupportedCompoundV1Permutation(0x0102_0304))]
fn compound_v1_rejects_invalid_dimensions(
    #[case] rank: u8,
    #[case] dimensions: [u32; 4],
    #[case] permutation: u32,
    #[case] expected: FormatError,
) {
    let buf = test_util::legacy_compound::raw_member(
        u32::MAX,
        0,
        rank,
        permutation,
        dimensions,
        &fixtures::build_fixed_point(4, false, true, 0, 32),
    );
    assert_eq!(DecodedDatatype::parse_prefix(&buf).unwrap_err(), expected);
}

#[rstest]
#[case::array(1, [4, 0, 0, 0], 4, 16, 16)]
#[case::scalar(0, [0; 4], 5, 8, 4)]
#[case::offset_addition_exceeds_u32(1, [1, 0, 0, 0], u32::MAX, u32::MAX, 4)]
fn compound_v1_rejects_member_outside_declared_size(
    #[case] rank: u8,
    #[case] dimensions: [u32; 4],
    #[case] offset: u32,
    #[case] size: u32,
    #[case] footprint: u32,
) {
    let buf = test_util::legacy_compound::raw_member(
        size,
        offset,
        rank,
        0,
        dimensions,
        &fixtures::build_fixed_point(4, false, true, 0, 32),
    );
    assert_eq!(
        DecodedDatatype::parse_prefix(&buf).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: DatatypeMemberName::from(c"arr"),
            offset: u64::from(offset),
            field_size: footprint,
            compound_size: size,
        }
    );
}

#[rstest]
#[case::missing_block(0)]
#[case::missing_permutation(4)]
#[case::missing_extents(12)]
#[case::partial_last_extent(27)]
fn compound_v1_rejects_truncated_dimensions(#[case] available: usize) {
    let mut buf = test_util::legacy_compound::array_member();
    let block_start = 20;
    buf.truncate(block_start + available);
    assert_eq!(
        DecodedDatatype::parse_prefix(&buf).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: block_start + 28,
            available: buf.len(),
        }
    );
}

#[test]
fn compound_v1_retains_nested_array_shape() {
    let inner = test_util::legacy_compound::array_member();
    let buf = test_util::legacy_compound::raw_member(32, 0, 1, 0, [2, 0, 0, 0], &inner);
    let (decoded, consumed) = DecodedDatatype::parse_prefix(&buf).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(consumed, buf.len());
    let expected = Datatype::Compound {
        size: 32,
        members: vec![CompoundMember::new(
            DatatypeMemberName::from(c"arr"),
            0,
            Datatype::Array {
                base_type: Box::new(
                    DecodedDatatype::from_record(&inner)
                        .unwrap()
                        .into_datatype(),
                ),
                dimensions: vec![2],
            },
        )],
    };
    assert_eq!(datatype, expected);
    assert_eq!(
        DecodedDatatype::from_record(&semantic::serialize_datatype(&datatype).unwrap())
            .unwrap()
            .into_datatype(),
        expected
    );
}

#[test]
fn compound_v1_rejects_overflow_in_base_array() {
    let mut base =
        header::build_datatype_header(ARRAY_CLASS, DATATYPE_VERSION_THREE, [0; 3], u32::MAX);
    base.push(1);
    base.extend_from_slice(&u32::MAX.to_le_bytes());
    base.extend_from_slice(&fixtures::build_fixed_point(4, false, true, 0, 32));
    let buf = test_util::legacy_compound::raw_member(u32::MAX, 0, 1, 0, [1, 0, 0, 0], &base);
    assert_eq!(
        DecodedDatatype::parse_prefix(&buf).unwrap_err(),
        FormatError::LengthTooLarge {
            length: u64::from(u32::MAX) * 4,
            length_size: DATATYPE_SIZE_BYTES,
        }
    );
}

#[rstest]
#[case::version_two(DATATYPE_VERSION_TWO)]
#[case::version_three(DATATYPE_VERSION_THREE)]
#[case::version_five(DATATYPE_VERSION_FIVE)]
fn parses_explicit_arrays_in_compound_members(#[case] version: u8) {
    let array = Datatype::Array {
        base_type: Box::new(
            DecodedDatatype::from_record(&fixtures::build_fixed_point(4, false, true, 0, 32))
                .unwrap()
                .into_datatype(),
        ),
        dimensions: vec![2, 3],
    };
    let mut buf = header::build_datatype_header(COMPOUND_CLASS, version, [1, 0, 0], 28);
    buf.extend_from_slice(b"arr\0");
    if version == DATATYPE_VERSION_TWO {
        buf.extend_from_slice(&[0; 4]);
        buf.extend_from_slice(&4u32.to_le_bytes());
    } else {
        buf.push(4);
    }
    buf.extend_from_slice(&semantic::serialize_datatype(&array).unwrap());
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(&buf).unwrap();
    assert_eq!(
        (decoded.into_datatype(), bytes_read),
        (
            Datatype::Compound {
                size: 28,
                members: vec![CompoundMember::new(
                    DatatypeMemberName::from(c"arr"),
                    4,
                    array
                )],
            },
            buf.len()
        )
    );
}

// The version 5 fixture has unpadded member names and one-byte offsets.
#[test]
fn a_version_5_compound_parses_like_version_3() {
    let members = |version: u8| {
        let mut buf = header::build_datatype_header(6, version, [2, 0, 0], 16);
        buf.extend_from_slice(b"re\0");
        buf.push(0);
        buf.extend_from_slice(&fixtures::build_float(8, 52, 11, 0, 52, 1023));
        buf.extend_from_slice(b"im\0");
        buf.push(8);
        buf.extend_from_slice(&fixtures::build_float(8, 52, 11, 0, 52, 1023));
        buf
    };
    let (decoded, expected_len) = DecodedDatatype::parse_prefix(&members(3)).unwrap();
    let expected = decoded.into_datatype();
    let (decoded, len) = DecodedDatatype::parse_prefix(&members(5)).unwrap();
    let parsed = decoded.into_datatype();
    assert_eq!((parsed, len), (expected, expected_len));
}

#[test]
fn rejects_a_zero_sized_compound_member() {
    let member = header::build_datatype_header(3, DATATYPE_VERSION_ONE, [0x01, 0, 0], 0);
    let mut buf = header::build_datatype_header(6, DATATYPE_VERSION_THREE, [1, 0, 0], 8);
    buf.extend_from_slice(b"s\0");
    buf.push(0);
    buf.extend_from_slice(&member);

    assert_eq!(
        DecodedDatatype::parse_prefix(&buf).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 3 }
    );
}

#[test]
fn parses_an_array_of_enumerations_in_a_compound_member() {
    let mut enum_bytes = header::build_datatype_header(8, DATATYPE_VERSION_THREE, [2, 0, 0], 4);
    enum_bytes.extend_from_slice(&fixtures::build_fixed_point(4, false, true, 0, 32));
    enum_bytes.extend_from_slice(b"A\0");
    enum_bytes.extend_from_slice(b"B\0");
    enum_bytes.extend_from_slice(&0i32.to_le_bytes());
    enum_bytes.extend_from_slice(&1i32.to_le_bytes());

    let mut array_bytes = header::build_datatype_header(10, DATATYPE_VERSION_THREE, [0, 0, 0], 8);
    array_bytes.push(1);
    array_bytes.extend_from_slice(&2u32.to_le_bytes());
    array_bytes.extend_from_slice(&enum_bytes);

    let mut buf = header::build_datatype_header(6, DATATYPE_VERSION_THREE, [1, 0, 0], 8);
    buf.extend_from_slice(b"data\0");
    buf.push(0);
    buf.extend_from_slice(&array_bytes);

    let dt = DecodedDatatype::from_record(&buf).unwrap().into_datatype();
    match dt {
        Datatype::Compound { members, .. } => {
            assert_eq!(members.len(), 1);
            assert_eq!(members[0].name.as_bytes(), b"data");
            match &members[0].datatype {
                Datatype::Array {
                    dimensions,
                    base_type,
                } => {
                    assert_eq!(dimensions, &[2]);
                    match base_type.as_ref() {
                        Datatype::Enumeration(datatype) => {
                            let members = datatype.members();
                            assert_eq!(members.len(), 2);
                            assert_eq!(members[0].name.as_bytes(), b"A");
                            assert_eq!(members[1].name.as_bytes(), b"B");
                        }
                        other => panic!("expected Enum, got {other:?}"),
                    }
                }
                other => panic!("expected Array, got {other:?}"),
            }
        }
        _ => panic!("expected Compound"),
    }
}

#[test]
fn serialize_parse_compound_roundtrip() {
    let dt = Datatype::Compound {
        size: 20,
        members: vec![
            CompoundMember::new(
                DatatypeMemberName::from(c"x"),
                0,
                Datatype::FloatingPoint(
                    FloatingPointType::from_layout(
                        8,
                        DatatypeByteOrder::LittleEndian,
                        FloatingPointLayout::IEEE754_BINARY64,
                    )
                    .unwrap(),
                ),
            ),
            CompoundMember::new(
                DatatypeMemberName::from(c"y"),
                8,
                Datatype::FloatingPoint(
                    FloatingPointType::from_layout(
                        8,
                        DatatypeByteOrder::LittleEndian,
                        FloatingPointLayout::IEEE754_BINARY64,
                    )
                    .unwrap(),
                ),
            ),
            CompoundMember::new(
                DatatypeMemberName::from(c"id"),
                16,
                Datatype::FixedPoint(
                    FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
                ),
            ),
        ],
    };
    let bytes = semantic::serialize_datatype(&dt).unwrap();
    let parsed = DecodedDatatype::from_record(&bytes)
        .unwrap()
        .into_datatype();
    assert_eq!(parsed, dt);
}

const DATATYPE_VERSION_FOUR: u8 = 4;
