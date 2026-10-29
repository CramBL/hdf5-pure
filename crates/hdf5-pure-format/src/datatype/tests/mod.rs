use hdf5_pure_core::FloatingPointField;
use rstest::rstest;

use super::*;

#[cfg(feature = "std")]
mod display;

/// `class_may_hold_object_address` admits the encoded class of every datatype
/// that `datatype_holds_object_address` accepts.
///
/// A whole-file scan in `hdf5-pure` calls the class predicate to
/// decide whether to parse a datatype. The class predicate reads only the first
/// byte, so it is cheap. A datatype the class predicate rejects is never parsed,
/// so a datatype that holds an address and fails the class predicate would leave
/// that address dangling with no error reported. Nothing in either function's
/// code enforces this correspondence, so this test does.
#[test]
fn the_class_gate_admits_every_reference_holding_datatype() {
    let object_ref = || Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let i32_le = || {
        Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )
    };
    let holds_an_address = [
        ("a bare object reference", object_ref()),
        (
            "a dataset-region reference",
            Datatype::Reference {
                size: 12,
                ref_type: ReferenceType::DatasetRegion,
            },
        ),
        (
            "a compound holding one",
            Datatype::Compound {
                size: 12,
                members: vec![
                    __private::compound_member("r".into(), 0, object_ref()),
                    __private::compound_member("i".into(), 8, i32_le()),
                ],
            },
        ),
        (
            "an array of them",
            Datatype::Array {
                base_type: Box::new(object_ref()),
                dimensions: vec![2],
            },
        ),
        (
            "a variable length of them",
            Datatype::VariableLength(VariableLengthType::sequence(object_ref())),
        ),
        (
            "an enumeration over one",
            Datatype::Enumeration {
                size: 8,
                base_type: Box::new(object_ref()),
                members: vec![__private::enum_member("a".into(), vec![0; 8])],
            },
        ),
        (
            "one nested two deep",
            Datatype::Array {
                base_type: Box::new(Datatype::Compound {
                    size: 8,
                    members: vec![__private::compound_member("r".into(), 0, object_ref())],
                }),
                dimensions: vec![3],
            },
        ),
    ];
    for (what, dt) in holds_an_address {
        assert!(
            datatype_holds_object_address(&dt),
            "{what} holds an object address"
        );
        let encoded = serialize_datatype(&dt);
        assert!(
            class_may_hold_object_address(encoded[0]),
            "{what} encodes as class {}, which the gate rejects — a walk would \
             never parse it and would pass over the address inside it",
            encoded[0] & 0x0F
        );
    }
}

/// `class_may_hold_object_address` is a necessary condition for holding an
/// object address but not a sufficient one.
///
/// A compound of two integers passes the class predicate, so the caller must
/// not treat a `true` result as proof that an address is present.
#[test]
fn the_class_gate_is_necessary_and_not_sufficient() {
    let ints = Datatype::Compound {
        size: 8,
        members: vec![__private::compound_member(
            "a".into(),
            0,
            Datatype::FixedPoint(
                FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap(),
            ),
        )],
    };
    assert!(!datatype_holds_object_address(&ints));
    assert!(
        class_may_hold_object_address(serialize_datatype(&ints)[0]),
        "a compound of integers is admitted by the class gate and holds no address"
    );
}

// Helper to build a fixed-point datatype message
fn build_fixed_point(
    size: u32,
    be: bool,
    signed: bool,
    bit_offset: u16,
    bit_precision: u16,
) -> Vec<u8> {
    let bf0 = if be { 0x01 } else { 0x00 } | if signed { 0x08 } else { 0x00 };
    let mut buf = build_dt_header(0, DATATYPE_VERSION_ONE, [bf0, 0, 0], size);
    let mut props = [0u8; 4];
    LittleEndian::write_u16(&mut props[0..2], bit_offset);
    LittleEndian::write_u16(&mut props[2..4], bit_precision);
    buf.extend_from_slice(&props);
    buf
}

// Helper to build a floating-point datatype message
fn build_float(
    size: u32,
    exp_loc: u8,
    exp_size: u8,
    mant_loc: u8,
    mant_size: u8,
    exp_bias: u32,
) -> Vec<u8> {
    let bf0 = FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE << FLOAT_NORMALIZATION_SHIFT;
    let bf1 = u8::try_from(size * 8 - 1).unwrap();
    let bf2 = 0;
    let mut buf = build_dt_header(1, DATATYPE_VERSION_ONE, [bf0, bf1, bf2], size);
    let mut props = [0u8; 12];
    LittleEndian::write_u16(&mut props[0..2], 0); // `bit_offset`
    LittleEndian::write_u16(&mut props[2..4], (size * 8) as u16); // `bit_precision`
    props[4] = exp_loc;
    props[5] = exp_size;
    props[6] = mant_loc;
    props[7] = mant_size;
    LittleEndian::write_u32(&mut props[8..12], exp_bias);
    buf.extend_from_slice(&props);
    buf
}

#[rstest]
#[case(FIXED_POINT_CLASS)]
#[case(BIT_FIELD_CLASS)]
fn preserves_low_and_high_padding(
    #[case] class: u8,
    #[values(false, true)] low: bool,
    #[values(false, true)] high: bool,
    #[values(false, true)] big_endian: bool,
) {
    let mut bytes = build_fixed_point(2, big_endian, false, 3, 9);
    bytes[0] = class | (DATATYPE_VERSION_ONE << 4);
    bytes[1] |= padding_bits(low, high);
    let (datatype, consumed) = parse_datatype(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    match &datatype {
        Datatype::FixedPoint(numeric) => {
            let layout = numeric.layout();
            assert_eq!((layout.low_padding(), layout.high_padding()), (low, high));
        }
        Datatype::BitField(numeric) => {
            assert_eq!((numeric.low_padding(), numeric.high_padding()), (low, high));
        }
        other => panic!("expected an integer or bit-field datatype, got {other:?}"),
    }
    assert_eq!(serialize_datatype(&datatype), bytes);
}

#[rstest]
fn preserves_floating_point_padding(
    #[values(false, true)] low: bool,
    #[values(false, true)] high: bool,
    #[values(false, true)] internal: bool,
) {
    let mut bytes = build_float(4, 23, 8, 0, 23, 127);
    bytes[1] |= padding_bits(low, high);
    if internal {
        bytes[1] |= FLOAT_INTERNAL_PADDING_BIT;
    }
    let (datatype, consumed) = parse_datatype(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let Datatype::FloatingPoint(numeric) = &datatype else {
        panic!("expected a floating-point datatype, got {datatype:?}");
    };
    let layout = numeric.layout();
    assert_eq!(
        (
            layout.low_padding(),
            layout.high_padding(),
            layout.internal_padding()
        ),
        (low, high, internal)
    );
    assert_eq!(serialize_datatype(&datatype), bytes);
}

#[test]
fn test_fixed_point_u8() {
    let data = build_fixed_point(1, false, false, 0, 8);
    let (dt, consumed) = parse_datatype(&data).unwrap();
    assert_eq!(consumed, 12);
    assert_eq!(
        dt,
        Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap()
        )
    );
}

#[test]
fn test_fixed_point_i16_le() {
    let data = build_fixed_point(2, false, true, 0, 16);
    let (dt, _) = parse_datatype(&data).unwrap();
    assert_eq!(
        dt,
        Datatype::FixedPoint(
            FixedPointType::new(2, DatatypeByteOrder::LittleEndian, true, 0, 16).unwrap()
        )
    );
}

#[test]
fn test_fixed_point_u32_be() {
    let data = build_fixed_point(4, true, false, 0, 32);
    let (dt, _) = parse_datatype(&data).unwrap();
    match &dt {
        Datatype::FixedPoint(numeric) => {
            assert_eq!(numeric.byte_order(), DatatypeByteOrder::BigEndian);
            assert!(!numeric.layout().signed());
            assert_eq!(numeric.size().get(), 4);
        }
        _ => panic!("expected FixedPoint"),
    }
}

#[test]
fn test_fixed_point_i64_le() {
    let data = build_fixed_point(8, false, true, 0, 64);
    let (dt, _) = parse_datatype(&data).unwrap();
    assert_eq!(
        dt,
        Datatype::FixedPoint(
            FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap()
        )
    );
}

#[rstest]
#[case(MantissaNormalization::Unnormalized, FLOAT_NORMALIZATION_UNNORMALIZED)]
#[case(MantissaNormalization::LeadingOne, FLOAT_NORMALIZATION_LEADING_ONE)]
#[case(
    MantissaNormalization::ImpliedLeadingOne,
    FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE
)]
fn preserves_floating_point_class_fields(
    #[case] normalization: MantissaNormalization,
    #[case] normalization_bits: u8,
    #[values(0, 30, 31, u8::MAX)] sign_location: u8,
    #[values(
        DatatypeByteOrder::LittleEndian,
        DatatypeByteOrder::BigEndian,
        DatatypeByteOrder::Vax
    )]
    byte_order: DatatypeByteOrder,
) {
    let (size, precision, exponent_size, mantissa_location, mantissa_size) = match sign_location {
        0 => (4, 32, 8, 1, 22),
        30 => (4, 32, 7, 0, 23),
        31 => (4, 32, 8, 0, 23),
        _ => (32, 256, 8, 0, 23),
    };
    let datatype = Datatype::FloatingPoint(
        FloatingPointType::new(
            size,
            byte_order,
            normalization,
            sign_location,
            0,
            precision,
            23,
            exponent_size,
            mantissa_location,
            mantissa_size,
            127,
        )
        .unwrap(),
    );
    let bytes = serialize_datatype(&datatype);
    assert_eq!(
        (bytes[1] >> FLOAT_NORMALIZATION_SHIFT) & FLOAT_NORMALIZATION_MASK,
        normalization_bits
    );
    assert_eq!(bytes[2], sign_location);
    assert_eq!(parse_datatype(&bytes).unwrap(), (datatype, bytes.len()));
}

#[test]
fn test_float_f32_le() {
    // IEEE 754 f32: exp=8 bits at bit 23, mant=23 bits at bit 0, bias=127
    let data = build_float(4, 23, 8, 0, 23, 127);
    let (dt, consumed) = parse_datatype(&data).unwrap();
    assert_eq!(consumed, 20);
    assert_eq!(
        dt,
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                4,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY32
            )
            .unwrap()
        )
    );
}

#[test]
fn test_float_f64_le() {
    let data = build_float(8, 52, 11, 0, 52, 1023);
    let (dt, _) = parse_datatype(&data).unwrap();
    assert_eq!(
        dt,
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64
            )
            .unwrap()
        )
    );
}

#[rstest]
#[case(0x00, DatatypeByteOrder::LittleEndian)]
#[case(0x01, DatatypeByteOrder::BigEndian)]
#[case(0x40, DatatypeByteOrder::Vax)]
#[case(0x41, DatatypeByteOrder::Vax)]
fn each_pair_of_byte_order_bits_parses_to_its_order(
    #[case] bit_field: u8,
    #[case] expected: DatatypeByteOrder,
) {
    let mut data = build_float(4, 23, 8, 0, 23, 127);
    data[1] = (data[1] & !(FLOAT_BYTE_ORDER_HIGH_BIT | FLOAT_BYTE_ORDER_LOW_BIT)) | bit_field;

    let (dt, _) = parse_datatype(&data).unwrap();
    let Datatype::FloatingPoint(numeric) = dt else {
        panic!("expected a floating-point datatype, got {dt:?}");
    };
    assert_eq!(numeric.byte_order(), expected);
}

#[test]
fn test_string_null_terminated_ascii() {
    let buf = build_dt_header(3, DATATYPE_VERSION_ONE, [0x00, 0, 0], 10); // padding=0(nullterm), charset=0(ascii)
    let (dt, consumed) = parse_datatype(&buf).unwrap();
    assert_eq!(consumed, 8);
    assert_eq!(
        dt,
        Datatype::String {
            size: 10,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        }
    );
}

#[test]
fn test_string_space_padded_utf8() {
    // padding=2(space pad), charset=1(utf8) → bf0 = 0x12
    let buf = build_dt_header(3, DATATYPE_VERSION_ONE, [0x12, 0, 0], 32);
    let (dt, _) = parse_datatype(&buf).unwrap();
    assert_eq!(
        dt,
        Datatype::String {
            size: 32,
            padding: StringPadding::SpacePad,
            charset: CharacterSet::Utf8,
        }
    );
}

#[test]
fn test_opaque() {
    // `tag_len` = 4, `tag` = "BLOB"
    let mut buf = build_dt_header(5, DATATYPE_VERSION_ONE, [4, 0, 0], 64);
    buf.extend_from_slice(b"BLOB");
    // Pad to 8 bytes
    buf.extend_from_slice(&[0, 0, 0, 0]);
    let (dt, consumed) = parse_datatype(&buf).unwrap();
    assert_eq!(consumed, 16); // 8 header + 8 padded tag
    assert_eq!(
        dt,
        Datatype::Opaque {
            size: 64,
            tag: b"BLOB".to_vec(),
        }
    );
}

#[rstest]
#[case(1, 1)]
#[case(255, 1)]
#[case(256, 2)]
#[case(65_535, 2)]
#[case(65_536, 3)]
#[case(70_000, 3)]
#[case(16_777_215, 3)]
#[case(16_777_216, 4)]
#[case(u32::MAX, 4)]
fn compound_offsets_use_the_minimum_byte_width(#[case] size: u32, #[case] width: usize) {
    let datatype = Datatype::Compound {
        size,
        members: vec![__private::compound_member(
            "last".into(),
            u64::from(size - 1),
            byte_member_type(),
        )],
    };
    let mut expected = build_dt_header(COMPOUND_CLASS, DATATYPE_VERSION_THREE, [1, 0, 0], size);
    expected.extend_from_slice(b"last\0");
    expected.extend_from_slice(&(size - 1).to_le_bytes()[..width]);
    expected.extend_from_slice(&serialize_datatype(&byte_member_type()));

    assert_eq!(offset_bytes_for_size(size), width);
    assert_eq!(serialize_datatype(&datatype), expected);
    assert_eq!(
        parse_datatype(&expected).unwrap(),
        (datatype, expected.len())
    );
}

#[rstest]
#[case(COMPOUND_CLASS, DATATYPE_VERSION_THREE)]
#[case(COMPOUND_CLASS, DATATYPE_VERSION_FOUR)]
#[case(COMPOUND_CLASS, DATATYPE_VERSION_FIVE)]
#[case(COMPLEX_CLASS, DATATYPE_VERSION_FIVE)]
fn three_byte_member_offsets_preserve_the_next_datatype(#[case] class: u8, #[case] version: u8) {
    let member = byte_member_type();
    let mut encoded = build_dt_header(class, version, [2, 0, 0], 70_000);
    encoded.extend_from_slice(b"first\0\0\0\0");
    encoded.extend_from_slice(&serialize_datatype(&member));
    encoded.extend_from_slice(b"last\0\x6f\x11\x01");
    encoded.extend_from_slice(&serialize_datatype(&member));
    assert_eq!(
        parse_datatype(&encoded).unwrap(),
        (
            Datatype::Compound {
                size: 70_000,
                members: vec![
                    __private::compound_member("first".into(), 0, member.clone()),
                    __private::compound_member("last".into(), 69_999, member),
                ],
            },
            encoded.len(),
        )
    );
}

#[test]
fn test_compound_v3_two_members() {
    // Compound with size=12, 2 members: "x" u32 at offset 0, "y" f64 at offset 4
    // Size=12, so offset_bytes=1
    let mut buf = build_dt_header(6, DATATYPE_VERSION_THREE, [2, 0, 0], 12); // 2 members
    // Member "x": name "x\0", offset=0, then u32 LE datatype
    buf.extend_from_slice(b"x\0");
    buf.push(0); // `byte_offset` = 0
    buf.extend_from_slice(&build_fixed_point(4, false, false, 0, 32));
    // Member "y": name "y\0", offset=4, then f64 LE datatype
    buf.extend_from_slice(b"y\0");
    buf.push(4); // `byte_offset` = 4
    buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));

    let (dt, _) = parse_datatype(&buf).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 12);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "x");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "y");
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

/// A version-1 compound of two `f64` members named `real` and `imag`, at byte
/// offsets 0 and 8, parses as a 16-byte compound with those two members.
///
/// MATLAB writes complex values in this layout. In version 1, each member name
/// is NUL-terminated and padded to a multiple of 8 bytes, and a fixed 28-byte
/// dimension block sits between the byte offset and the member datatype
/// message. The block holds one dimensionality byte, three reserved bytes, four
/// dimension permutation bytes, four reserved bytes, and sixteen dimension size
/// bytes.

#[test]
fn test_compound_v1_complex_matlab_layout() {
    let mut buf = build_dt_header(6, DATATYPE_VERSION_ONE, [2, 0, 0], 16); // v1, 2 members, size 16
    for (name, offset) in [(&b"real\0\0\0\0"[..], 0u32), (&b"imag\0\0\0\0"[..], 8)] {
        buf.extend_from_slice(name); // NUL-terminated, padded to 8
        let mut off = [0u8; 4];
        LittleEndian::write_u32(&mut off, offset);
        buf.extend_from_slice(&off);
        buf.extend_from_slice(&[0u8; 28]); // v1 dimension block
        buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));
    }

    let (dt, _) = parse_datatype(&buf).unwrap();
    match dt {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 16);
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "real");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "imag");
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

#[test]
fn test_reference_object() {
    let buf = build_dt_header(7, DATATYPE_VERSION_ONE, [0, 0, 0], 8);
    let (dt, _) = parse_datatype(&buf).unwrap();
    assert_eq!(
        dt,
        Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        }
    );
}

#[test]
fn test_reference_region() {
    let buf = build_dt_header(7, DATATYPE_VERSION_ONE, [1, 0, 0], 12);
    let (dt, _) = parse_datatype(&buf).unwrap();
    assert_eq!(
        dt,
        Datatype::Reference {
            size: 12,
            ref_type: ReferenceType::DatasetRegion,
        }
    );
}

#[test]
fn test_enumeration() {
    // Enum with base type i32 LE, 3 members
    let mut buf = build_dt_header(8, DATATYPE_VERSION_THREE, [3, 0, 0], 4); // 3 members
    // Base type: i32 LE
    buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));
    // Names: "RED\0", "GREEN\0", "BLUE\0"
    buf.extend_from_slice(b"RED\0");
    buf.extend_from_slice(b"GREEN\0");
    buf.extend_from_slice(b"BLUE\0");
    // Values: 0, 1, 2 (as i32 LE)
    buf.extend_from_slice(&0i32.to_le_bytes());
    buf.extend_from_slice(&1i32.to_le_bytes());
    buf.extend_from_slice(&2i32.to_le_bytes());

    let (dt, _) = parse_datatype(&buf).unwrap();
    match dt {
        Datatype::Enumeration {
            size,
            base_type,
            members,
        } => {
            assert_eq!(size, 4);
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "RED");
            assert_eq!(members[0].value, 0i32.to_le_bytes().to_vec());
            assert_eq!(members[1].name, "GREEN");
            assert_eq!(members[1].value, 1i32.to_le_bytes().to_vec());
            assert_eq!(members[2].name, "BLUE");
            assert_eq!(members[2].value, 2i32.to_le_bytes().to_vec());
            match *base_type {
                Datatype::FixedPoint(numeric)
                    if numeric.size().get() == 4 && numeric.layout().signed() => {}
                other => panic!("expected i32, got {other:?}"),
            }
        }
        _ => panic!("expected Enumeration"),
    }
}

#[rstest]
#[case(2)]
#[case(3)]
#[case(4)]
#[case(5)]
#[case(6)]
#[case(7)]
#[case(8)]
#[case(9)]
#[case(10)]
#[case(11)]
#[case(12)]
#[case(13)]
#[case(14)]
#[case(15)]
fn rejects_reserved_variable_length_subtypes(#[case] subtype: u8) {
    let mut bytes = build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [subtype, 0, 0],
        16,
    );
    bytes.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));
    assert_eq!(
        parse_datatype(&bytes).unwrap_err(),
        FormatError::InvalidVariableLengthSubtype(subtype)
    );
}

#[rstest]
#[case(0x00_1000)]
#[case(0x00_2000)]
#[case(0x00_4000)]
#[case(0x00_8000)]
#[case(0x01_0000)]
#[case(0x02_0000)]
#[case(0x04_0000)]
#[case(0x08_0000)]
#[case(0x10_0000)]
#[case(0x20_0000)]
#[case(0x40_0000)]
#[case(0x80_0000)]
fn rejects_variable_length_reserved_bits(
    #[case] bits: u32,
    #[values(VARIABLE_LENGTH_SEQUENCE, VARIABLE_LENGTH_STRING)] subtype: u8,
) {
    let [bf0, bf1, bf2, _] = bits.to_le_bytes();
    let mut bytes = build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [bf0 | subtype, bf1, bf2],
        16,
    );
    bytes.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));
    assert_eq!(
        parse_datatype(&bytes).unwrap_err(),
        FormatError::InvalidVariableLengthReservedBits(bits)
    );
}

#[rstest]
#[case(1, 0)]
#[case(2, 0)]
#[case(15, 0)]
#[case(0, 1)]
#[case(0, 15)]
#[case(1, 1)]
fn rejects_string_metadata_on_variable_length_sequences(#[case] padding: u8, #[case] charset: u8) {
    let mut bytes = build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [padding << VARIABLE_LENGTH_PADDING_SHIFT, charset, 0],
        16,
    );
    bytes.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));
    assert_eq!(
        parse_datatype(&bytes).unwrap_err(),
        FormatError::InvalidVariableLengthSequenceMetadata { padding, charset }
    );
}

#[rstest]
#[case([0x31, 0, 0], FormatError::InvalidStringPadding(3))]
#[case([0xF1, 0, 0], FormatError::InvalidStringPadding(15))]
#[case([1, 2, 0], FormatError::InvalidCharacterSet(2))]
#[case([1, 15, 0], FormatError::InvalidCharacterSet(15))]
fn rejects_invalid_variable_length_string_metadata(
    #[case] flags: [u8; 3],
    #[case] expected: FormatError,
) {
    let mut bytes = build_dt_header(VARIABLE_LENGTH_CLASS, DATATYPE_VERSION_ONE, flags, 16);
    bytes.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));
    assert_eq!(parse_datatype(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case(STRING_CLASS)]
#[case(OPAQUE_CLASS)]
fn rejects_zero_sized_variable_length_string_bases(#[case] class: u8) {
    let mut bytes = build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [VARIABLE_LENGTH_STRING, 0, 0],
        16,
    );
    bytes.extend_from_slice(&build_dt_header(class, DATATYPE_VERSION_ONE, [0, 0, 0], 0));
    assert_eq!(
        parse_datatype(&bytes).unwrap_err(),
        FormatError::ZeroSizedDatatype { class }
    );
}

#[rstest]
#[case(0, StringPadding::NullTerminate)]
#[case(1, StringPadding::NullPad)]
#[case(2, StringPadding::SpacePad)]
fn variable_length_strings_preserve_metadata_and_base_byte_order(
    #[case] padding_code: u8,
    #[case] padding: StringPadding,
    #[values(false, true)] utf8: bool,
    #[values(false, true)] big_endian: bool,
) {
    let mut bytes = build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [
            VARIABLE_LENGTH_STRING | (padding_code << VARIABLE_LENGTH_PADDING_SHIFT),
            u8::from(utf8),
            0,
        ],
        16,
    );
    bytes.extend_from_slice(&build_fixed_point(1, big_endian, false, 0, 8));
    let (datatype, consumed) = parse_datatype(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(
        datatype,
        Datatype::VariableLength(VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                padding,
                if utf8 {
                    CharacterSet::Utf8
                } else {
                    CharacterSet::Ascii
                },
                Datatype::FixedPoint(
                    FixedPointType::new(
                        1,
                        if big_endian {
                            DatatypeByteOrder::BigEndian
                        } else {
                            DatatypeByteOrder::LittleEndian
                        },
                        false,
                        0,
                        8
                    )
                    .unwrap()
                )
            )
            .unwrap()
        ))
    );
    assert_eq!(serialize_datatype(&datatype), bytes);
}

#[test]
fn nested_variable_length_sequences_roundtrip() {
    let datatype = Datatype::VariableLength(VariableLengthType::sequence(
        Datatype::VariableLength(VariableLengthType::sequence(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        ))),
    ));
    let mut expected = build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [VARIABLE_LENGTH_SEQUENCE, 0, 0],
        16,
    );
    expected.extend_from_slice(&build_dt_header(
        VARIABLE_LENGTH_CLASS,
        DATATYPE_VERSION_ONE,
        [VARIABLE_LENGTH_SEQUENCE, 0, 0],
        16,
    ));
    expected.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));
    assert_eq!(serialize_datatype(&datatype), expected);
    assert_eq!(
        parse_datatype(&expected).unwrap(),
        (datatype, expected.len())
    );
}

#[test]
fn test_variable_length_string_utf8() {
    // VL string: type=1, padding=0(null term), charset=1(utf8)
    // bf0: bits 0-3 = 1 (string), bits 4-7 = 0 (null term) → 0x01
    // bf1: bits 0-3 = 1 (utf8) → 0x01
    let mut buf = build_dt_header(9, DATATYPE_VERSION_ONE, [0x01, 0x01, 0], 16);
    // Base type: u8 (class 0, unsigned, size 1)
    buf.extend_from_slice(&build_fixed_point(1, false, false, 0, 8));

    let (dt, _) = parse_datatype(&buf).unwrap();
    match dt {
        Datatype::VariableLength(VariableLengthType::String(string)) => {
            assert_eq!(string.padding(), &StringPadding::NullTerminate);
            assert_eq!(string.charset(), &CharacterSet::Utf8);
            assert_eq!(string.base_type().type_size(), 1);
        }
        _ => panic!("expected VariableLength"),
    }
}

#[test]
fn test_variable_length_sequence_f32() {
    // VL sequence: type=0
    // bf0 = 0x00
    let mut buf = build_dt_header(9, DATATYPE_VERSION_ONE, [0x00, 0x00, 0], 16);
    // Base type: f32 LE
    buf.extend_from_slice(&build_float(4, 23, 8, 0, 23, 127));

    let (dt, _) = parse_datatype(&buf).unwrap();
    match dt {
        Datatype::VariableLength(VariableLengthType::Sequence { element_type }) => {
            assert_eq!(element_type.type_size(), 4);
        }
        _ => panic!("expected VariableLength"),
    }
}

// The version 5 fixture has unpadded member names and one-byte offsets.
#[test]
fn a_version_5_compound_parses_like_version_3() {
    let members = |version: u8| {
        let mut buf = build_dt_header(6, version, [2, 0, 0], 16);
        buf.extend_from_slice(b"re\0");
        buf.push(0);
        buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));
        buf.extend_from_slice(b"im\0");
        buf.push(8);
        buf.extend_from_slice(&build_float(8, 52, 11, 0, 52, 1023));
        buf
    };
    let (expected, expected_len) = parse_datatype(&members(3)).unwrap();
    let (parsed, len) = parse_datatype(&members(5)).unwrap();
    assert_eq!((parsed, len), (expected, expected_len));
}

/// The same for an array: version 5 keeps version 3's layout.
#[test]
fn a_version_5_array_parses_like_version_3() {
    let array = |version: u8| {
        let mut buf = build_dt_header(10, version, [0, 0, 0], 48);
        buf.push(2);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));
        buf
    };
    let (expected, expected_len) = parse_datatype(&array(3)).unwrap();
    let (parsed, len) = parse_datatype(&array(5)).unwrap();
    assert_eq!((parsed, len), (expected, expected_len));
}

#[test]
fn test_array_2d() {
    // Array [3][4] of i32 LE, version 3
    let mut buf = build_dt_header(10, DATATYPE_VERSION_THREE, [0, 0, 0], 48); // 3*4*4=48
    buf.push(2); // ndims=2
    buf.extend_from_slice(&3u32.to_le_bytes()); // dim 0
    buf.extend_from_slice(&4u32.to_le_bytes()); // dim 1
    // Base type: i32 LE
    buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));

    let (dt, _) = parse_datatype(&buf).unwrap();
    match dt {
        Datatype::Array {
            base_type,
            dimensions,
        } => {
            assert_eq!(dimensions, vec![3, 4]);
            match *base_type {
                Datatype::FixedPoint(numeric)
                    if numeric.size().get() == 4 && numeric.layout().signed() => {}
                other => panic!("expected i32, got {other:?}"),
            }
        }
        _ => panic!("expected Array"),
    }
}

/// A datatype message that declares a zero element size is rejected during
/// parsing, so every reader can divide by the element size.
#[test]
fn a_zero_width_element_type_is_refused() {
    let buf = build_dt_header(3, DATATYPE_VERSION_ONE, [0x01, 0, 0], 0); // fixed-length string of 0 bytes
    assert_eq!(
        parse_datatype(&buf).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 3 }
    );
}

/// Parsing rejects an array with a zero dimension, even when the header
/// declares a 48-byte element size.
///
/// An array's size is computed from its base type and dimensions, so a zero
/// dimension makes the element zero-width while the header field stays
/// non-zero.
#[test]
fn an_array_with_a_zero_dimension_is_refused_despite_its_header_size() {
    let mut buf = build_dt_header(10, DATATYPE_VERSION_THREE, [0, 0, 0], 48);
    buf.push(2); // ndims=2
    buf.extend_from_slice(&0u32.to_le_bytes()); // dim 0 (no elements)
    buf.extend_from_slice(&4u32.to_le_bytes()); // dim 1
    buf.extend_from_slice(&build_fixed_point(4, false, true, 0, 32));

    assert_eq!(
        parse_datatype(&buf).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 10 }
    );
}

/// Parsing rejects a zero-width datatype nested inside a compound member.
///
/// A member datatype is parsed through the same entry point as a top-level
/// datatype, so the zero-size check applies at any depth.
#[test]
fn a_zero_width_compound_member_is_refused() {
    let member = build_dt_header(3, DATATYPE_VERSION_ONE, [0x01, 0, 0], 0);
    let mut buf = build_dt_header(6, DATATYPE_VERSION_THREE, [1, 0, 0], 8); // one member
    buf.extend_from_slice(b"s\0");
    buf.push(0); // byte offset, one byte for a size-8 compound
    buf.extend_from_slice(&member);

    assert_eq!(
        parse_datatype(&buf).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 3 }
    );
}

/// `element_size` returns the same value as `type_size` for a type with a
/// non-zero size.
///
/// The two differ in their return type. `element_size` returns a `NonZeroU32`,
/// so a caller can divide by it without a separate zero check.
#[test]
fn element_size_matches_type_size_for_a_type_that_has_one() {
    let dt = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    assert_eq!(dt.element_size().unwrap().get(), dt.type_size());
}

/// A caller-built literal never passes through `parse`, so `element_size` is
/// the only thing standing between a degenerate type and the writers. The
/// `Array` case is the one that matters: its width is *computed* from its
/// dimensions, so this cannot be caught by inspecting a stored size field.
#[test]
fn element_size_rejects_a_constructed_array_with_a_zero_dimensionq() {
    let dt = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )),
        dimensions: vec![0, 4],
    };
    assert_eq!(dt.type_size(), 0);
    assert_eq!(
        dt.element_size().unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 10 }
    );
    assert_eq!(
        element_size_usize(&dt).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 10 }
    );
}

/// `element_size` reports the class of the datatype it was called on.
///
/// A caller can match on the class to identify which datatype produced the
/// error, without reconstructing the path to a nested type.
#[test]
fn element_size_reports_the_refused_types_own_class() {
    let dt = Datatype::Compound {
        size: 0,
        members: vec![],
    };
    assert_eq!(
        dt.element_size().unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 6 }
    );
}

#[test]
fn test_bitfield() {
    let mut buf = build_dt_header(4, DATATYPE_VERSION_ONE, [0, 0, 0], 2); // 16-bit LE bitfield
    let mut props = [0u8; 4];
    LittleEndian::write_u16(&mut props[0..2], 0);
    LittleEndian::write_u16(&mut props[2..4], 16);
    buf.extend_from_slice(&props);

    let (dt, _) = parse_datatype(&buf).unwrap();
    assert_eq!(
        dt,
        Datatype::BitField(BitFieldType::new(2, DatatypeByteOrder::LittleEndian, 0, 16).unwrap())
    );
}

#[test]
fn test_time() {
    let mut buf = build_dt_header(2, DATATYPE_VERSION_ONE, [0, 0, 0], 8);
    let mut props = [0u8; 2];
    LittleEndian::write_u16(&mut props[0..2], 64);
    buf.extend_from_slice(&props);

    let (dt, consumed) = parse_datatype(&buf).unwrap();
    assert_eq!(consumed, 10);
    assert_eq!(
        dt,
        Datatype::Time(TimeType::new(8, DatatypeByteOrder::LittleEndian, 64).unwrap())
    );
}

#[test]
fn test_time_byte_order_roundtrips() {
    // A big-endian time type must serialize and re-parse with its byte order
    // preserved (bf0 bit 0), so repack can reproduce it faithfully.
    for (be, order) in [
        (0u8, DatatypeByteOrder::LittleEndian),
        (1u8, DatatypeByteOrder::BigEndian),
    ] {
        let mut buf = build_dt_header(2, DATATYPE_VERSION_ONE, [be, 0, 0], 4);
        buf.extend_from_slice(&32u16.to_le_bytes());
        let (dt, _) = parse_datatype(&buf).unwrap();
        assert_eq!(dt, Datatype::Time(TimeType::new(4, order, 32).unwrap()));
        // serialize -> parse must round-trip the byte order.
        let (reparsed, _) = parse_datatype(&serialize_datatype(&dt)).unwrap();
        assert_eq!(reparsed, dt);
    }
}

#[test]
fn test_nested_compound_array_enum() {
    // Compound containing a single member "data" which is an Array[2] of Enum(i32, 2 values)
    // Build the enum first
    let mut enum_bytes = build_dt_header(8, DATATYPE_VERSION_THREE, [2, 0, 0], 4); // 2 members
    enum_bytes.extend_from_slice(&build_fixed_point(4, false, true, 0, 32)); // base i32
    enum_bytes.extend_from_slice(b"A\0");
    enum_bytes.extend_from_slice(b"B\0");
    enum_bytes.extend_from_slice(&0i32.to_le_bytes());
    enum_bytes.extend_from_slice(&1i32.to_le_bytes());

    // Build array[2] of that enum, version 3
    let mut array_bytes = build_dt_header(10, DATATYPE_VERSION_THREE, [0, 0, 0], 8); // 2*4=8
    array_bytes.push(1); // ndims=1
    array_bytes.extend_from_slice(&2u32.to_le_bytes()); // dim[0]=2
    array_bytes.extend_from_slice(&enum_bytes);

    // Build compound with 1 member, size=8
    let mut buf = build_dt_header(6, DATATYPE_VERSION_THREE, [1, 0, 0], 8); // 1 member
    buf.extend_from_slice(b"data\0");
    buf.push(0); // `byte_offset` = 0 (size=8, so 1 byte offsets)
    buf.extend_from_slice(&array_bytes);

    let (dt, _) = parse_datatype(&buf).unwrap();
    match dt {
        Datatype::Compound { members, .. } => {
            assert_eq!(members.len(), 1);
            assert_eq!(members[0].name, "data");
            match &members[0].datatype {
                Datatype::Array {
                    dimensions,
                    base_type,
                } => {
                    assert_eq!(dimensions, &[2]);
                    match base_type.as_ref() {
                        Datatype::Enumeration { members, .. } => {
                            assert_eq!(members.len(), 2);
                            assert_eq!(members[0].name, "A");
                            assert_eq!(members[1].name, "B");
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
fn test_error_invalid_class() {
    let buf = build_dt_header(13, DATATYPE_VERSION_ONE, [0, 0, 0], 4);
    let err = parse_datatype(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidDatatypeClass(13));
}

#[test]
fn test_error_truncated_data() {
    let buf = [0u8; 4]; // too short for header
    let err = parse_datatype(&buf).unwrap_err();
    match err {
        FormatError::UnexpectedEof { .. } => {}
        other => panic!("expected UnexpectedEof, got {other:?}"),
    }
}

#[test]
fn test_error_invalid_string_padding() {
    let buf = build_dt_header(3, DATATYPE_VERSION_ONE, [0x03, 0, 0], 10); // padding=3 invalid
    let err = parse_datatype(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidStringPadding(3));
}

#[test]
fn test_error_invalid_charset() {
    let buf = build_dt_header(3, DATATYPE_VERSION_ONE, [0x20, 0, 0], 10); // charset=2 invalid
    let err = parse_datatype(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidCharacterSet(2));
}

#[test]
fn test_error_invalid_reference_type() {
    let buf = build_dt_header(7, DATATYPE_VERSION_ONE, [5, 0, 0], 8);
    let err = parse_datatype(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidReferenceType(5));
}

#[test]
fn serialize_parse_compound_roundtrip() {
    let dt = Datatype::Compound {
        size: 20,
        members: vec![
            __private::compound_member(
                "x".to_string(),
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
            __private::compound_member(
                "y".to_string(),
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
            __private::compound_member(
                "id".to_string(),
                16,
                Datatype::FixedPoint(
                    FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
                ),
            ),
        ],
    };
    let bytes = serialize_datatype(&dt);
    let (parsed, _) = parse_datatype(&bytes).unwrap();
    assert_eq!(parsed, dt);
}

#[test]
fn serialize_parse_enum_roundtrip() {
    let dt = Datatype::Enumeration {
        size: 4,
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )),
        members: vec![
            __private::enum_member("RED".to_string(), 0i32.to_le_bytes().to_vec()),
            __private::enum_member("GREEN".to_string(), 1i32.to_le_bytes().to_vec()),
            __private::enum_member("BLUE".to_string(), 2i32.to_le_bytes().to_vec()),
        ],
    };
    let bytes = serialize_datatype(&dt);
    let (parsed, _) = parse_datatype(&bytes).unwrap();
    assert_eq!(parsed, dt);
}

/// Builds a fixed-point base type for enumeration tests.
fn enum_base_fp(size: u32, be: bool, signed: bool) -> Datatype {
    Datatype::FixedPoint(
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
        .unwrap(),
    )
}

/// Builds an enum datatype over `base`, storing each member value truncated to
/// the base width (the value blob is opaque bytes, so any content round-trips).
fn make_enum(base: Datatype, members: &[(&str, i64)]) -> Datatype {
    let size = base.type_size();
    let width = size as usize;
    Datatype::Enumeration {
        size,
        base_type: Box::new(base),
        members: members
            .iter()
            .map(|(name, v)| {
                __private::enum_member((*name).to_string(), v.to_le_bytes()[..width].to_vec())
            })
            .collect(),
    }
}

#[test]
fn serialize_parse_enum_base_type_variety() {
    for base in [
        enum_base_fp(1, false, false), // u8
        enum_base_fp(2, true, true),   // i16 big-endian
        enum_base_fp(8, false, true),  // i64
    ] {
        let dt = make_enum(base.clone(), &[("A", 0), ("B", 1), ("NEG", -1)]);
        let bytes = serialize_datatype(&dt);
        let (parsed, consumed) = parse_datatype(&bytes).unwrap();
        assert_eq!(parsed, dt, "round-trip failed for base {base:?}");
        assert_eq!(consumed, bytes.len());
    }
}

#[test]
fn serialize_parse_enum_large_member_count() {
    // More than 256 members exercises the 2-byte member-count field, which is
    // split across bf0/bf1 in the datatype message header.
    let owned: Vec<(String, i64)> = (0..300).map(|i| (format!("M{i}"), i)).collect();
    let members: Vec<(&str, i64)> = owned.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let dt = make_enum(enum_base_fp(4, false, true), &members);
    let bytes = serialize_datatype(&dt);
    let (parsed, _) = parse_datatype(&bytes).unwrap();
    assert_eq!(parsed, dt);
    match parsed {
        Datatype::Enumeration { members, .. } => {
            assert_eq!(members.len(), 300);
            assert_eq!(members[299].name, "M299");
        }
        other => panic!("expected Enumeration, got {other:?}"),
    }
}

/// A member value wider than the base type serializes in full and truncates on
/// parse.
///
/// `EnumTypeBuilder::build` and `Datatype::Enumeration` take the element size
/// from the base type and do not check that member value blobs match it. A
/// 4-byte value on a 1-byte base therefore parses back as its low byte. This
/// test records that behavior, and the assertion fails if either side changes.
#[test]
fn enum_value_width_is_not_validated_against_base_size() {
    let dt = Datatype::Enumeration {
        size: 1,
        base_type: Box::new(enum_base_fp(1, false, false)),
        members: vec![__private::enum_member(
            "X".to_string(),
            5i32.to_le_bytes().to_vec(),
        )],
    };
    let bytes = serialize_datatype(&dt);
    let (parsed, _) = parse_datatype(&bytes).unwrap();
    assert_ne!(
        parsed, dt,
        "a value wider than the base silently truncates on parse"
    );
    match parsed {
        Datatype::Enumeration { members, .. } => assert_eq!(members[0].value, vec![5]),
        other => panic!("expected Enumeration, got {other:?}"),
    }
}

#[test]
fn serialize_parse_array_roundtrip() {
    let dt = Datatype::Array {
        base_type: Box::new(Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64,
            )
            .unwrap(),
        )),
        dimensions: vec![3],
    };
    let bytes = serialize_datatype(&dt);
    let (parsed, _) = parse_datatype(&bytes).unwrap();
    assert_eq!(parsed, dt);
}

#[test]
fn serialize_parse_time_roundtrip() {
    let dt = Datatype::Time(TimeType::new(8, DatatypeByteOrder::LittleEndian, 64).unwrap());
    let bytes = serialize_datatype(&dt);
    let (parsed, consumed) = parse_datatype(&bytes).unwrap();
    assert_eq!(parsed, dt);
    assert_eq!(consumed, bytes.len());
}

#[test]
fn serialize_parse_bitfield_roundtrip() {
    for byte_order in [
        DatatypeByteOrder::LittleEndian,
        DatatypeByteOrder::BigEndian,
    ] {
        let dt = Datatype::BitField(BitFieldType::new(4, byte_order, 3, 17).unwrap());
        let bytes = serialize_datatype(&dt);
        let (parsed, consumed) = parse_datatype(&bytes).unwrap();
        assert_eq!(parsed, dt);
        assert_eq!(consumed, bytes.len());
    }
}

#[test]
fn serialize_parse_opaque_roundtrip() {
    // Tag lengths that do and do not land on an 8-byte boundary, to exercise
    // the zero padding both ways.
    for tag in [
        b"abc".to_vec(),         // 3 bytes -> padded to 8
        b"12345678".to_vec(),    // 8 bytes -> no padding
        b"sensor-id\0".to_vec(), // 10 bytes -> padded to 16, embedded NUL preserved
    ] {
        let dt = Datatype::Opaque { size: 16, tag };
        let bytes = serialize_datatype(&dt);
        // The property section (after the 8-byte header) must be a multiple
        // of 8, matching what the reference library expects.
        assert_eq!((bytes.len() - 8) % 8, 0);
        let (parsed, consumed) = parse_datatype(&bytes).unwrap();
        assert_eq!(parsed, dt);
        assert_eq!(consumed, bytes.len());
    }
}

#[test]
fn test_type_size() {
    let dt = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    assert_eq!(dt.type_size(), 4);

    let dt = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )),
        dimensions: vec![3, 4],
    };
    assert_eq!(dt.type_size(), 48);
}

fn byte_member_type() -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap(),
    )
}

const DATATYPE_VERSION_FOUR: u8 = 4;
#[rstest]
#[case(0, 0, 8, FormatError::ZeroSizedDatatype { class: FIXED_POINT_CLASS })]
#[case(1, 0, 0, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 0, bit_precision: 0 })]
#[case(1, 8, 1, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 8, bit_precision: 1 })]
#[case(1, 4, 5, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 4, bit_precision: 5 })]
#[case(8, u16::MAX, u16::MAX, FormatError::InvalidFixedPointRange { size: 8, bit_offset: u16::MAX, bit_precision: u16::MAX })]
fn rejects_invalid_fixed_point_metadata(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[case] expected: FormatError,
) {
    assert_eq!(
        parse_datatype(&build_fixed_point(size, false, false, offset, precision)).unwrap_err(),
        expected
    );
}

#[rstest]
#[case(4, 0, FormatError::ZeroSizedDatatype { class: FLOATING_POINT_CLASS })]
#[case(8, 1, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 1, bit_precision: 32 })]
#[case(10, 0, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 0, bit_precision: 0 })]
#[case(10, 33, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 0, bit_precision: 33 })]
#[case(2, 32, FormatError::InvalidFloatingPointField { field: FloatingPointField::Sign, location: 32, size: 1, bit_offset: 0, bit_precision: 32 })]
#[case(13, 0, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 0, bit_offset: 0, bit_precision: 32 })]
#[case(13, 10, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 10, bit_offset: 0, bit_precision: 32 })]
#[case(15, 0, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 0, size: 0, bit_offset: 0, bit_precision: 32 })]
#[case(14, 250, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 250, size: 23, bit_offset: 0, bit_precision: 32 })]
fn rejects_invalid_floating_point_metadata(
    #[case] index: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
) {
    let mut bytes = build_float(4, 23, 8, 0, 23, 127);
    bytes[index] = value;
    assert_eq!(parse_datatype(&bytes).unwrap_err(), expected);
}

#[test]
fn rejects_reserved_mantissa_normalization() {
    let mut bytes = build_float(4, 23, 8, 0, 23, 127);
    bytes[1] = FLOAT_NORMALIZATION_RESERVED << FLOAT_NORMALIZATION_SHIFT;
    assert_eq!(
        parse_datatype(&bytes).unwrap_err(),
        FormatError::InvalidMantissaNormalization(FLOAT_NORMALIZATION_RESERVED)
    );
}

#[rstest]
#[case(
    2,
    23,
    FloatingPointField::Sign,
    23,
    1,
    FloatingPointField::Exponent,
    23,
    8
)]
#[case(
    2,
    0,
    FloatingPointField::Sign,
    0,
    1,
    FloatingPointField::Mantissa,
    0,
    23
)]
#[case(
    15,
    24,
    FloatingPointField::Exponent,
    23,
    8,
    FloatingPointField::Mantissa,
    0,
    24
)]
fn rejects_overlapping_floating_point_metadata(
    #[case] index: usize,
    #[case] value: u8,
    #[case] first: FloatingPointField,
    #[case] first_location: u8,
    #[case] first_size: u8,
    #[case] second: FloatingPointField,
    #[case] second_location: u8,
    #[case] second_size: u8,
) {
    let mut bytes = build_float(4, 23, 8, 0, 23, 127);
    bytes[index] = value;
    assert_eq!(
        parse_datatype(&bytes).unwrap_err(),
        FormatError::OverlappingFloatingPointFields {
            first,
            first_location,
            first_size,
            second,
            second_location,
            second_size
        }
    );
}

#[rstest]
#[case(
    FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian),
    build_float(4, 23, 8, 0, 23, 127)
)]
#[case(
    FloatingPointType::ieee754_binary64(DatatypeByteOrder::LittleEndian),
    build_float(8, 52, 11, 0, 52, 1023)
)]
fn preserves_ieee_bytes(#[case] numeric: FloatingPointType, #[case] expected: Vec<u8>) {
    let datatype = Datatype::FloatingPoint(numeric);
    let encoded = serialize_datatype(&datatype);
    assert_eq!(encoded, expected);
    assert_eq!(parse_datatype(&encoded).unwrap(), (datatype, encoded.len()));
}

#[rstest]
#[case(16)]
#[case(u32::MAX)]
fn parses_wide_numeric_metadata(#[case] size: u32) {
    let fixed = Datatype::FixedPoint(
        FixedPointType::new(size, DatatypeByteOrder::BigEndian, true, 1, 127).unwrap(),
    );
    let encoded = serialize_datatype(&fixed);
    assert_eq!(parse_datatype(&encoded).unwrap().0, fixed);
    let floating = Datatype::FloatingPoint(
        FloatingPointType::from_layout(
            size,
            DatatypeByteOrder::Vax,
            FloatingPointLayout::IEEE754_BINARY64,
        )
        .unwrap(),
    );
    let encoded = serialize_datatype(&floating);
    assert_eq!(parse_datatype(&encoded).unwrap().0, floating);
}
#[rstest]
#[case(0, 1, FormatError::ZeroSizedDatatype { class: TIME_CLASS })]
#[case(1, 0, FormatError::InvalidNumericBitRange { class: TIME_CLASS, size: 1, bit_offset: 0, bit_precision: 0 })]
#[case(1, 9, FormatError::InvalidNumericBitRange { class: TIME_CLASS, size: 1, bit_offset: 0, bit_precision: 9 })]
fn rejects_invalid_time_metadata(
    #[case] size: u32,
    #[case] precision: u16,
    #[case] expected: FormatError,
) {
    let mut bytes = build_dt_header(TIME_CLASS, DATATYPE_VERSION_ONE, [0, 0, 0], size);
    bytes.extend_from_slice(&precision.to_le_bytes());
    assert_eq!(parse_datatype(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case(0, 0, 1, FormatError::ZeroSizedDatatype { class: BIT_FIELD_CLASS })]
#[case(1, 0, 0, FormatError::InvalidNumericBitRange { class: BIT_FIELD_CLASS, size: 1, bit_offset: 0, bit_precision: 0 })]
#[case(1, 4, 5, FormatError::InvalidNumericBitRange { class: BIT_FIELD_CLASS, size: 1, bit_offset: 4, bit_precision: 5 })]
fn rejects_invalid_bit_field_metadata(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[case] expected: FormatError,
) {
    let mut bytes = build_dt_header(BIT_FIELD_CLASS, DATATYPE_VERSION_ONE, [0, 0, 0], size);
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.extend_from_slice(&precision.to_le_bytes());
    assert_eq!(parse_datatype(&bytes).unwrap_err(), expected);
}
