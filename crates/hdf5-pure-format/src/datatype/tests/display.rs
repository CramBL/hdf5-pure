use rstest::rstest;

use hdf5_pure_core::__private::DISPLAY_MAX_MEMBERS;

use super::super::*;

#[test]
fn ordinary_numeric_types_read_as_their_rust_names() {
    let int = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    assert_eq!(int.to_string(), "i32");

    let float = Datatype::FloatingPoint(
        FloatingPointType::from_layout(
            8,
            DatatypeByteOrder::LittleEndian,
            FloatingPointLayout::IEEE754_BINARY64,
        )
        .unwrap(),
    );
    assert_eq!(float.to_string(), "f64");
}

/// A datatype message with a size near [`u32::MAX`] writes the correct bit
/// width, where a direct `u32` multiply would overflow and panic a debug build
///
/// Each width is `size * 8` over an on-disk `u32`. [`bit_width`] widens before
/// multiplying. The test exercises each class that calls it, including the
/// classes that `classify_datatype` routes here.
#[rstest]
#[case::fixed_point(Datatype::FixedPoint(FixedPointType::new(u32::MAX, DatatypeByteOrder::LittleEndian, true, 0, 1).unwrap()), "i34359738360(bits 0..1)")]
#[case::floating_point(Datatype::FloatingPoint(FloatingPointType::from_layout(u32::MAX, DatatypeByteOrder::LittleEndian, FloatingPointLayout::IEEE754_BINARY64).unwrap()), "f34359738360(bits 0..64)")]
#[case::time(Datatype::Time(TimeType::new(u32::MAX, DatatypeByteOrder::LittleEndian, 1).unwrap()), "time34359738360(bits 0..1)")]
#[case::bit_field(Datatype::BitField(BitFieldType::new(u32::MAX, DatatypeByteOrder::LittleEndian, 0, 1).unwrap()), "bitfield34359738360(bits 0..1)")]
fn a_crafted_size_writes_its_width_instead_of_overflowing(
    #[case] datatype: Datatype,
    #[case] expected: &str,
) {
    assert_eq!(datatype.to_string(), expected);
}

/// A bit offset and bit precision of `u16::MAX` together produce a bit span
/// whose end is 131,070.
///
/// The end of the span is the sum of two `u16` values. Both widen to a larger
/// type before the add, so the result is 131,070.
#[test]
fn a_crafted_bit_span_does_not_wrap() {
    let dtype = Datatype::FixedPoint(
        FixedPointType::new(
            16_384,
            DatatypeByteOrder::LittleEndian,
            false,
            u16::MAX,
            u16::MAX,
        )
        .unwrap(),
    );
    assert_eq!(dtype.to_string(), "u131072(bits 65535..131070)");
}

/// Only what departs from the ordinary is written, since that is what the
/// reader of the message is looking for.
#[test]
fn unusual_fields_are_written_and_ordinary_ones_are_not() {
    let big_endian = Datatype::FixedPoint(
        FixedPointType::new(2, DatatypeByteOrder::BigEndian, false, 0, 16).unwrap(),
    );
    assert_eq!(big_endian.to_string(), "u16 be");

    let narrow = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 24).unwrap(),
    );
    assert_eq!(narrow.to_string(), "i32(bits 0..24)");
}

#[test]
fn nested_types_recurse_through_their_members() {
    let compound = Datatype::Compound {
        size: 12,
        members: vec![
            __private::compound_member(
                "x".into(),
                0,
                Datatype::FloatingPoint(
                    FloatingPointType::from_layout(
                        4,
                        DatatypeByteOrder::LittleEndian,
                        FloatingPointLayout::IEEE754_BINARY32,
                    )
                    .unwrap(),
                ),
            ),
            __private::compound_member(
                "n".into(),
                4,
                Datatype::FixedPoint(
                    FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap(),
                ),
            ),
        ],
    };
    assert_eq!(compound.to_string(), "compound{x: f32, n: i64}");

    let array = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap(),
        )),
        dimensions: vec![2, 3],
    };
    assert_eq!(
        array.to_string(),
        "array<u8, 2x3>",
        "the shape is spelled `2x3`, never a `Debug` slice"
    );
}

/// The leaf enums format through `Formatter::pad`, so a caller that lines these
/// up in a column receives the width it requested.
#[test]
fn a_leaf_enum_honors_the_width_it_is_given() {
    assert_eq!(format!("{:>8}", CharacterSet::Ascii), "   ascii");
    assert_eq!(format!("{:<8}|", DatatypeByteOrder::BigEndian), "be      |");
    assert_eq!(format!("{}", StringPadding::NullPad), "null-pad");
}

#[test]
fn a_string_carries_its_width_charset_and_padding() {
    let string = Datatype::String {
        size: 16,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Utf8,
    };
    assert_eq!(string.to_string(), "string[16] utf8 null-pad");
}

/// The tag is arbitrary file bytes, so it cannot reach a message unescaped.
#[test]
fn an_opaque_tag_is_quoted_and_escaped() {
    let opaque = Datatype::Opaque {
        size: 4,
        tag: b"a\"b\x00".to_vec(),
    };
    assert_eq!(opaque.to_string(), "opaque[4] \"a\\\"b\\x00\"");
}

/// A member name comes from the file by way of `from_utf8_lossy`, which
/// rejects nothing, so it is escaped for the same reason an opaque tag is.
#[test]
fn a_member_name_cannot_carry_a_control_character_into_a_message() {
    let compound = Datatype::Compound {
        size: 4,
        members: vec![__private::compound_member(
            "a\nb\u{1b}[31m".into(),
            0,
            u32_datatype(),
        )],
    };
    let shown = compound.to_string();
    assert!(!shown.chars().any(char::is_control), "{shown}");
    assert_eq!(shown, "compound{a\\nb\\u{1b}[31m: u32}");

    let enumeration = Datatype::Enumeration {
        size: 4,
        base_type: Box::new(u32_datatype()),
        members: vec![__private::enum_member("red\u{0}".into(), vec![0, 0, 0, 0])],
    };
    let shown = enumeration.to_string();
    assert!(!shown.chars().any(char::is_control), "{shown}");
    assert_eq!(shown, "enum<u32>[red\\0]");
}

/// A compound or enumeration with more members than `DISPLAY_MAX_MEMBERS`
/// elides the excess and reports the count of the remainder.
///
/// The member count is an on-disk `u16`, so a message can describe far more
/// members than the display cap allows. Both variants that hold members elide
/// the excess.
#[test]
fn a_long_member_list_is_elided_and_reports_the_remainder() {
    let over_cap = DISPLAY_MAX_MEMBERS + 3;

    let compound = Datatype::Compound {
        size: (over_cap * 4) as u32,
        members: (0..over_cap)
            .map(|i| __private::compound_member(format!("m{i}"), (i * 4) as u64, u32_datatype()))
            .collect(),
    };
    let enumeration = Datatype::Enumeration {
        size: 4,
        base_type: Box::new(u32_datatype()),
        members: (0..over_cap)
            .map(|i| __private::enum_member(format!("m{i}"), vec![0, 0, 0, 0]))
            .collect(),
    };

    for (datatype, close) in [(compound, "}"), (enumeration, "]")] {
        let shown = datatype.to_string();
        assert!(shown.ends_with(&format!(", … 3 more{close}")), "{shown}");
        assert!(shown.contains("m0"), "{shown}");
        assert!(
            !shown.contains(&format!("m{DISPLAY_MAX_MEMBERS}")),
            "{shown}"
        );
    }
}

/// The boundary: exactly the cap is written whole, with no "0 more".
#[test]
fn a_member_list_at_exactly_the_cap_is_not_elided() {
    let members: Vec<_> = (0..DISPLAY_MAX_MEMBERS)
        .map(|i| __private::enum_member(format!("m{i}"), vec![0, 0, 0, 0]))
        .collect();
    let shown = Datatype::Enumeration {
        size: 4,
        base_type: Box::new(u32_datatype()),
        members,
    }
    .to_string();

    assert!(!shown.contains('…'), "{shown}");
    assert!(
        shown.ends_with(&format!("m{}]", DISPLAY_MAX_MEMBERS - 1)),
        "{shown}"
    );
}

fn u32_datatype() -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, false, 0, 32).unwrap(),
    )
}
