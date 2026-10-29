use rstest::rstest;

use super::*;
use crate::FloatingPointLayout;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::datatype::Datatype;
use crate::{CharacterSet, DatatypeByteOrder, StringPadding};

#[cfg(feature = "std")]
mod display;

// The declared `u32::MAX` width exercises bit-count formatting beyond 32 bits.
#[rstest]
#[case(Datatype::FixedPoint(FixedPointType::new(u32::MAX, DatatypeByteOrder::LittleEndian, true, 0, 1).unwrap()), "other(i34359738360(bits 0..1))")]
#[case(Datatype::FloatingPoint(FloatingPointType::from_layout(u32::MAX, DatatypeByteOrder::LittleEndian, FloatingPointLayout::IEEE754_BINARY64).unwrap()), "other(f34359738360(bits 0..64))")]
fn an_unusual_size_arrives_whole_and_writes_its_width(
    #[case] datatype: Datatype,
    #[case] expected: &str,
) {
    let classified = classify_datatype(&datatype);
    assert_eq!(classified, DType::Other(Box::new(datatype)));
    assert_eq!(classified.to_string(), expected);
}

// `Other` preserves the full datatype in both compounds and arrays.
#[test]
fn an_unclassified_type_reaches_the_caller_whole_through_either_recursion() {
    let opaque = Datatype::Opaque {
        size: 3,
        tag: b"rgb".to_vec(),
    };
    let carried = DType::Other(Box::new(opaque.clone()));

    let compound = Datatype::Compound {
        size: 3,
        members: vec![crate::datatype::__private::compound_member(
            "pixel".into(),
            0,
            opaque.clone(),
        )],
    };
    let DType::Compound(fields) = classify_datatype(&compound) else {
        panic!("a compound classifies as one");
    };
    assert_eq!(fields, vec![("pixel".to_string(), carried.clone())]);

    let array = Datatype::Array {
        base_type: Box::new(opaque),
        dimensions: vec![2, 3],
    };
    assert_eq!(
        classify_datatype(&array),
        DType::Array(Box::new(carried), vec![2, 3])
    );
}

/// Writes attributes to an in-memory file and returns their decoded values by name.
fn round_trip(cases: &[(&str, AttrValue)]) -> HashMap<std::string::String, AttrValue> {
    let mut builder = crate::writer::FileBuilder::new();
    for (name, value) in cases {
        builder.set_attr(name, value.clone());
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = builder.finish().unwrap();
    crate::File::from_bytes(bytes)
        .unwrap()
        .root()
        .attrs()
        .unwrap()
}

/// The variant a string attribute was written from is the variant it reads
/// back as. The one-element arrays are the point: charset and dataspace
/// kind are both on disk, so neither collapses into the scalar form.
#[test]
fn every_string_variant_round_trips_to_itself() {
    let cases = vec![
        ("utf8_scalar", AttrValue::String("m/s".into())),
        ("utf8_one", AttrValue::StringArray(vec!["m/s".into()])),
        (
            "utf8_two",
            AttrValue::StringArray(vec!["m/s".into(), "kg".into()]),
        ),
        ("ascii_scalar", AttrValue::AsciiString("double".into())),
        (
            "ascii_one",
            AttrValue::AsciiStringArray(vec!["double".into()]),
        ),
        (
            "ascii_two",
            AttrValue::AsciiStringArray(vec!["double".into(), "int16".into()]),
        ),
        (
            "vlen_one",
            AttrValue::VarLenAsciiCharArray(vec!["x".into()]),
        ),
        (
            "vlen_three",
            AttrValue::VarLenAsciiCharArray(vec!["x".into(), "y".into(), "velocity".into()]),
        ),
        // The standard variable-length string, whose datatype the MATLAB
        // shape above does not write (#383). Charset and arity separate the
        // four the same way they separate the fixed-width variants.
        ("vlen_utf8_scalar", AttrValue::VarLenString("mètre".into())),
        (
            "vlen_utf8_one",
            AttrValue::VarLenStringArray(vec!["mètre".into()]),
        ),
        (
            "vlen_ascii_scalar",
            AttrValue::VarLenAsciiString("double".into()),
        ),
        (
            "vlen_ascii_two",
            AttrValue::VarLenAsciiStringArray(vec!["x".into(), "yy".into()]),
        ),
        // A declared width is stored and recovered too, or a slot would
        // shrink to its content the next time it was written (#359).
        (
            "ascii_sized",
            AttrValue::ascii_string_sized("ok", 64).unwrap(),
        ),
        (
            "ascii_array_sized",
            AttrValue::ascii_string_array_sized(vec!["north".into(), "s".into()], 16).unwrap(),
        ),
        ("utf8_sized", AttrValue::string_sized("mètre", 32).unwrap()),
        (
            "utf8_array_sized",
            AttrValue::string_array_sized(vec!["m/s".into()], 12).unwrap(),
        ),
    ];
    let read = round_trip(&cases);
    for (name, written) in &cases {
        assert_eq!(read.get(*name), Some(written), "attribute {name}");
    }
}

#[test]
fn a_width_the_content_implies_reads_back_as_the_plain_variant() {
    let cases = vec![
        ("exact", AttrValue::ascii_string_sized("double", 6).unwrap()),
        ("empty", AttrValue::ascii_string_sized("", 1).unwrap()),
        (
            "longest",
            AttrValue::string_array_sized(vec!["m/s".into(), "kg".into()], 3).unwrap(),
        ),
    ];
    let read = round_trip(&cases);
    assert_eq!(
        read.get("exact"),
        Some(&AttrValue::AsciiString("double".into()))
    );
    assert_eq!(
        read.get("empty"),
        Some(&AttrValue::AsciiString(String::new()))
    );
    assert_eq!(
        read.get("longest"),
        Some(&AttrValue::StringArray(vec!["m/s".into(), "kg".into()]))
    );
}

/// Decodes an attribute from its datatype, raw bytes, and dimensions.
fn decode_raw(datatype: Datatype, raw_data: Vec<u8>, dimensions: Vec<u64>) -> Option<AttrValue> {
    use crate::dataspace::{Dataspace, DataspaceType};

    let scalar = dimensions.is_empty();
    let attr = crate::attribute::AttributeMessage {
        name: "a".into(),
        datatype,
        dataspace: Dataspace {
            space_type: if scalar {
                DataspaceType::Scalar
            } else {
                DataspaceType::Simple
            },
            #[expect(clippy::cast_possible_truncation)]
            rank: dimensions.len() as u8,
            dimensions,
            max_dimensions: None,
        },
        raw_data,
        datatype_location: crate::shared_message::DatatypeLocation::Inline,
    };
    decode_attr_value(
        &attr,
        &crate::source::BytesSource::new(Vec::new()),
        FormatWidths {
            offsets: crate::width::OffsetWidth::Eight,
            lengths: crate::width::LengthWidth::Eight,
        },
        AddressContext::from_effective_base(BaseAddress::ZERO),
    )
}

// Invalid UTF-8 expands the two-byte slot into four bytes of decoded text.
#[test]
fn a_slot_narrower_than_its_decoded_text_is_not_reported_as_padded() {
    let value = decode_raw(
        Datatype::String {
            size: 2,
            padding: crate::datatype::StringPadding::NullPad,
            charset: crate::datatype::CharacterSet::Ascii,
        },
        vec![0xB0, b'C'],
        vec![],
    )
    .expect("a fixed ASCII attribute decodes");

    assert_eq!(
        value,
        AttrValue::AsciiString("\u{FFFD}C".into()),
        "a slot its own text does not fit must not claim a width"
    );
    assert!(
        value.as_str().is_some_and(|s| s.len() > 2),
        "the case only bites because the decoded text outgrew the slot"
    );
}

#[test]
fn an_enum_attribute_decodes_as_its_base_type() {
    let h5py_bool =
        crate::type_builders::EnumTypeBuilder::with_base(crate::type_builders::make_i8_type())
            .value("FALSE", 0)
            .value("TRUE", 1)
            .build()
            .unwrap();

    // The base is one byte wide, so the code arrives as an 8-bit value (#350).
    assert_eq!(
        decode_raw(h5py_bool.clone(), vec![1], vec![]),
        Some(AttrValue::I8(1))
    );
    // The dataspace is simple with three elements.
    assert_eq!(
        decode_raw(h5py_bool, vec![1, 0, 1], vec![3]),
        Some(AttrValue::I8Array(vec![1, 0, 1]))
    );
}

/// The base type's signedness and width carry through, so an enum over an
/// unsigned 16-bit base lands in the unsigned variant of that width with its
/// value intact.
#[test]
fn an_unsigned_enum_attribute_keeps_its_base_signedness_and_width() {
    let mode =
        crate::type_builders::EnumTypeBuilder::with_base(crate::type_builders::make_u16_type())
            .value("low", 1)
            .value("high", 40_000)
            .build()
            .unwrap();

    assert_eq!(
        decode_raw(mode, 40_000_u16.to_le_bytes().to_vec(), vec![]),
        Some(AttrValue::U16(40_000))
    );
}

// Include extreme and subnormal binary32 values.
#[test]
fn every_float_variant_round_trips_to_itself() {
    let cases = vec![
        ("f64_scalar", AttrValue::F64(1.5)),
        ("f64_one", AttrValue::F64Array(vec![1.5])),
        ("f64_two", AttrValue::F64Array(vec![1.5, 2.5])),
        ("f32_scalar", AttrValue::F32(1.5)),
        ("f32_one", AttrValue::F32Array(vec![1.5])),
        ("f32_two", AttrValue::F32Array(vec![f32::MIN, f32::MAX])),
        (
            "f32_edges",
            AttrValue::F32Array(vec![
                f32::MIN_POSITIVE,
                f32::from_bits(1),
                f32::EPSILON,
                -0.0,
                f32::INFINITY,
            ]),
        ),
    ];
    let read = round_trip(&cases);
    for (name, written) in &cases {
        assert_eq!(read.get(*name), Some(written), "attribute {name}");
    }
}

#[test]
fn a_full_range_unsigned_value_survives_at_every_length() {
    let read = round_trip(&[
        ("scalar", AttrValue::U64(u64::MAX)),
        ("one", AttrValue::U64Array(vec![u64::MAX])),
        ("two", AttrValue::U64Array(vec![u64::MAX, 1])),
    ]);
    assert_eq!(read.get("scalar"), Some(&AttrValue::U64(u64::MAX)));
    assert_eq!(read.get("one"), Some(&AttrValue::U64Array(vec![u64::MAX])));
    assert_eq!(
        read.get("two"),
        Some(&AttrValue::U64Array(vec![u64::MAX, 1]))
    );
    // Full-range unsigned values exceed the signed accessor's range.
    for (name, expected) in [
        ("scalar", vec![u64::MAX]),
        ("one", vec![u64::MAX]),
        ("two", vec![u64::MAX, 1]),
    ] {
        let value = read.get(name).expect("present");
        assert_eq!(value.to_u64s(), Some(expected), "{name} must read unsigned");
        assert_eq!(
            value.to_i64s(),
            None,
            "{name} does not fit an i64 and must not wrap"
        );
    }
    assert_eq!(read["scalar"].as_u64(), Some(u64::MAX));
    assert_eq!(read["one"].as_u64(), Some(u64::MAX));
    assert_eq!(
        read["two"].as_u64(),
        None,
        "two elements are not a single value"
    );
}

#[test]
fn only_a_sequence_of_ascii_chars_claims_the_matlab_shape() {
    let char_base = Datatype::String {
        size: 1,
        padding: StringPadding::NullTerminate,
        charset: CharacterSet::Ascii,
    };
    let int_base = Datatype::FixedPoint(
        FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap(),
    );

    assert_eq!(
        vlen_string_shape(&VariableLengthType::sequence(char_base.clone())),
        VlenStringShape::AsciiCharSequence
    );
    assert_eq!(
        vlen_string_shape(&VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                StringPadding::NullTerminate,
                CharacterSet::Ascii,
                char_base.clone()
            )
            .unwrap()
        )),
        VlenStringShape::Ascii
    );
    assert_eq!(
        vlen_string_shape(&VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                StringPadding::NullTerminate,
                CharacterSet::Utf8,
                char_base
            )
            .unwrap()
        )),
        VlenStringShape::Utf8
    );
    assert_eq!(
        vlen_string_shape(&VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                StringPadding::NullTerminate,
                CharacterSet::Ascii,
                int_base.clone()
            )
            .unwrap()
        )),
        VlenStringShape::Ascii
    );
    assert_eq!(
        vlen_string_shape(&VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                StringPadding::NullTerminate,
                CharacterSet::Utf8,
                int_base
            )
            .unwrap()
        )),
        VlenStringShape::Utf8
    );
}

// The zero-size string datatype has an empty scalar value.
#[test]
fn an_empty_string_attribute_is_not_dropped() {
    let cases = vec![
        ("utf8", AttrValue::String(std::string::String::new())),
        ("ascii", AttrValue::AsciiString(std::string::String::new())),
    ];
    let read = round_trip(&cases);
    for (name, written) in &cases {
        assert_eq!(
            read.get(*name),
            Some(written),
            "attribute {name} must survive with its empty value"
        );
    }
}

// Native width conformance is checked in `crates/crosscheck/tests/main/attr_width.rs`.
#[test]
fn every_integer_width_round_trips_to_itself() {
    let cases = vec![
        ("i8", AttrValue::I8(-7)),
        ("i8_one", AttrValue::I8Array(vec![-7])),
        ("i8_two", AttrValue::I8Array(vec![i8::MIN, i8::MAX])),
        ("i16", AttrValue::I16(-7)),
        ("i16_one", AttrValue::I16Array(vec![-7])),
        ("i16_two", AttrValue::I16Array(vec![i16::MIN, i16::MAX])),
        ("i32", AttrValue::I32(-7)),
        ("i32_one", AttrValue::I32Array(vec![-7])),
        ("i32_two", AttrValue::I32Array(vec![i32::MIN, i32::MAX])),
        ("u8", AttrValue::U8(7)),
        ("u8_one", AttrValue::U8Array(vec![7])),
        ("u8_two", AttrValue::U8Array(vec![0, u8::MAX])),
        ("u16", AttrValue::U16(7)),
        ("u16_one", AttrValue::U16Array(vec![7])),
        ("u16_two", AttrValue::U16Array(vec![0, u16::MAX])),
        ("u32", AttrValue::U32(7)),
        ("u32_one", AttrValue::U32Array(vec![7])),
        ("u32_two", AttrValue::U32Array(vec![0, u32::MAX])),
        ("i64", AttrValue::I64(-7)),
        ("i64_one", AttrValue::I64Array(vec![-7])),
        ("i64_two", AttrValue::I64Array(vec![i64::MIN, i64::MAX])),
        ("u64", AttrValue::U64(7)),
        ("u64_one", AttrValue::U64Array(vec![7])),
        ("u64_two", AttrValue::U64Array(vec![0, u64::MAX])),
    ];
    let read = round_trip(&cases);
    for (name, written) in &cases {
        assert_eq!(read.get(*name), Some(written), "attribute {name}");
    }
}

// A three-byte unsigned integer uses the 64-bit attribute variant.
#[test]
fn a_width_with_no_rust_integer_widens() {
    let three_byte = Datatype::FixedPoint(
        FixedPointType::new(3, DatatypeByteOrder::LittleEndian, false, 0, 24).unwrap(),
    );
    assert_eq!(
        decode_raw(three_byte.clone(), vec![0xFF, 0xFF, 0xFF], vec![]),
        Some(AttrValue::U64(0x00FF_FFFF))
    );
    assert_eq!(
        decode_raw(three_byte, vec![1, 0, 0, 2, 0, 0], vec![2]),
        Some(AttrValue::U64Array(vec![1, 2]))
    );
}

#[test]
fn accessors_span_the_shapes_the_reader_now_distinguishes() {
    let read = round_trip(&[
        ("scalar", AttrValue::AsciiString("double".into())),
        ("one", AttrValue::StringArray(vec!["double".into()])),
        (
            "vlen",
            AttrValue::VarLenAsciiCharArray(vec!["double".into()]),
        ),
        ("vlen_std", AttrValue::VarLenAsciiString("double".into())),
        (
            "vlen_std_one",
            AttrValue::VarLenStringArray(vec!["double".into()]),
        ),
    ]);
    for name in ["scalar", "one", "vlen", "vlen_std", "vlen_std_one"] {
        assert_eq!(
            read.get(name).and_then(AttrValue::as_str),
            Some("double"),
            "attribute {name}"
        );
    }
}
