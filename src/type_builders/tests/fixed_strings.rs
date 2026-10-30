use super::super::{
    AttrValue, CharacterSet, DatasetBuilder, DataspaceType, Datatype, FormatError, StringPadding,
};

/// One fixed-width string entry point as a value, so a case table can name
/// all four without repeating the body under each.
type Stage = fn(&mut DatasetBuilder) -> Result<&mut DatasetBuilder, FormatError>;

/// Returns the staged datatype and element bytes for width and stride comparisons.
fn staged(builder: &DatasetBuilder) -> (Datatype, Vec<u8>) {
    (
        builder.datatype.clone().expect("a datatype"),
        builder.data.clone().expect("element bytes").unwrap(),
    )
}

/// Returns the declared string width after checking the byte count for `count` elements.
///
/// The payload must contain `count * width` bytes.
fn declared_width(builder: &DatasetBuilder, count: usize) -> u32 {
    let (dt, raw) = staged(builder);
    let Datatype::String { size, .. } = dt else {
        panic!("expected a string datatype, got {dt:?}");
    };
    assert_eq!(
        raw.len(),
        size as usize * count,
        "{count} elements of a {size}-byte type do not account for {} bytes",
        raw.len()
    );
    size
}

#[test]
fn a_derived_width_is_the_longest_value_and_never_zero() {
    assert_eq!(
        super::super::derived_string_width(&["north", "s", "east"]).get(),
        5
    );
    assert_eq!(
        super::super::derived_string_width(&["s", "east", "north"]).get(),
        5
    );
    // A multi-byte character is measured in bytes, not characters.
    assert_eq!(super::super::derived_string_width(&["é", "ab"]).get(), 2);
    // Empty inputs use the builder's one-byte minimum.
    assert_eq!(super::super::derived_string_width::<&str>(&[]).get(), 1);
    assert_eq!(super::super::derived_string_width(&["", ""]).get(), 1);
}

#[test]
fn a_derived_width_dataset_pads_every_element_to_the_longest() {
    let mut b = DatasetBuilder::new("d");
    b.with_ascii_strings(&["north", "s", ""]).unwrap();

    assert_eq!(declared_width(&b, 3), 5);
    let mut expected = Vec::new();
    expected.extend_from_slice(b"north"); // exactly the width, unpadded
    expected.extend_from_slice(b"s\0\0\0\0");
    expected.extend_from_slice(b"\0\0\0\0\0"); // an empty value is all padding
    assert_eq!(
        staged(&b).1,
        expected,
        "each value is zero-padded on the right to the declared width"
    );
    assert_eq!(b.shape, Some(vec![3]));
}

#[test]
fn a_declared_width_is_used_verbatim_even_when_the_values_are_shorter() {
    let mut b = DatasetBuilder::new("d");
    b.with_ascii_strings_sized(&["ab", "c"], 4).unwrap();

    assert_eq!(
        declared_width(&b, 2),
        4,
        "the declared width stands; it is not shrunk to the values in hand"
    );
    assert_eq!(staged(&b).1, b"ab\0\0c\0\0\0".to_vec());
}

// The overlong value is between two valid values.
#[test]
fn a_value_longer_than_the_declared_width_is_refused_not_truncated() {
    let mut b = DatasetBuilder::new("d");
    let refused = b.with_ascii_strings_sized(&["ab", "north", "cd"], 4);

    assert!(
        matches!(
            refused,
            Err(FormatError::FixedStringTooLong {
                index: 1,
                len: 5,
                width: 4
            })
        ),
        "{:?}",
        refused.map(|_| ())
    );
    assert!(
        b.datatype.is_none() && b.data.is_none(),
        "a refused call stages nothing"
    );
}

// Empty values isolate zero-width rejection from per-value overflow.
#[test]
fn a_declared_width_of_zero_is_refused() {
    for refused in [
        DatasetBuilder::new("d").with_ascii_strings_sized(&[""], 0),
        DatasetBuilder::new("d").with_strings_sized(&[""], 0),
    ] {
        assert!(
            matches!(refused, Err(FormatError::ZeroFixedStringWidth)),
            "{:?}",
            refused.map(|_| ())
        );
    }
}

/// The charset bit is the only thing separating the two families, and it is
/// the one a reader uses to decide how to decode the bytes.
#[test]
fn each_entry_point_declares_its_own_charset_and_null_padding() {
    let cases: [(Stage, _); 4] = [
        (|b| b.with_ascii_strings(&["ab"]), CharacterSet::Ascii),
        (
            |b| b.with_ascii_strings_sized(&["ab"], 4),
            CharacterSet::Ascii,
        ),
        (|b| b.with_strings(&["ab"]), CharacterSet::Utf8),
        (|b| b.with_strings_sized(&["ab"], 4), CharacterSet::Utf8),
    ];
    for (stage, expected) in cases {
        let mut b = DatasetBuilder::new("d");
        stage(&mut b).unwrap();
        let Datatype::String {
            padding, charset, ..
        } = staged(&b).0
        else {
            panic!("expected a string datatype");
        };
        assert_eq!(charset, expected);
        assert_eq!(padding, StringPadding::NullPad);
    }
}

// Compare attribute and dataset encodings for derived and declared widths.
#[test]
fn a_fixed_string_dataset_matches_the_attribute_encoding_of_the_same_values() {
    const VALUES: [&str; 3] = ["north", "s", ""];
    let owned: Vec<String> = VALUES.iter().map(|s| (*s).to_string()).collect();

    let cases: [(AttrValue, Stage); 4] = [
        (AttrValue::AsciiStringArray(owned.clone()), |b| {
            b.with_ascii_strings(&VALUES)
        }),
        (AttrValue::StringArray(owned.clone()), |b| {
            b.with_strings(&VALUES)
        }),
        (
            AttrValue::ascii_string_array_sized(owned.clone(), 16).unwrap(),
            |b| b.with_ascii_strings_sized(&VALUES, 16),
        ),
        (AttrValue::string_array_sized(owned, 16).unwrap(), |b| {
            b.with_strings_sized(&VALUES, 16)
        }),
    ];
    for (attr, stage) in cases {
        let message = super::super::build_attr_message("a", &attr).unwrap();
        let mut b = DatasetBuilder::new("d");
        stage(&mut b).unwrap();
        let (datatype, raw) = staged(&b);

        assert_eq!(datatype, message.datatype, "{attr:?}");
        assert_eq!(raw, message.raw_data, "{attr:?}");
    }
}

#[test]
fn an_explicit_shape_survives_a_fixed_string_write() {
    let mut b = DatasetBuilder::new("d");
    b.with_shape(&[2, 2]);
    b.with_ascii_strings(&["a", "b", "c", "d"]).unwrap();

    assert_eq!(b.shape, Some(vec![2, 2]));
    assert_eq!(declared_width(&b, 4), 1);
}

// ---- Attribute widths a caller declares (issue #359) ----

/// Each sized attribute constructor declares the width it was given, pads
/// every element out to it, and keeps its own charset and dataspace kind.
///
/// The width is read off the message and the stride off the bytes, because a
/// datatype that says 64 over elements laid out at 2 is the defect this
/// family exists to make impossible.
#[test]
fn a_declared_attribute_width_reaches_the_message_and_the_bytes() {
    let cases = [
        (
            AttrValue::ascii_string_sized("ok", 64).unwrap(),
            CharacterSet::Ascii,
            DataspaceType::Scalar,
            1,
        ),
        (
            AttrValue::string_sized("ok", 64).unwrap(),
            CharacterSet::Utf8,
            DataspaceType::Scalar,
            1,
        ),
        (
            AttrValue::ascii_string_array_sized(vec!["ok".into(), String::new()], 64).unwrap(),
            CharacterSet::Ascii,
            DataspaceType::Simple,
            2,
        ),
        (
            AttrValue::string_array_sized(vec!["ok".into(), String::new()], 64).unwrap(),
            CharacterSet::Utf8,
            DataspaceType::Simple,
            2,
        ),
    ];
    for (value, expected_charset, expected_space, count) in cases {
        let message = super::super::build_attr_message("a", &value).unwrap();
        let Datatype::String {
            size,
            padding,
            charset,
        } = message.datatype.clone()
        else {
            panic!("expected a string datatype for {value:?}");
        };
        assert_eq!(size, 64, "{value:?}");
        assert_eq!(charset, expected_charset, "{value:?}");
        assert_eq!(padding, StringPadding::NullPad, "{value:?}");
        assert_eq!(message.dataspace.space_type, expected_space, "{value:?}");
        assert_eq!(message.raw_data.len(), 64 * count, "{value:?}");
        assert_eq!(&message.raw_data[..2], b"ok", "{value:?}");
        assert!(
            message.raw_data[2..].iter().all(|b| *b == 0),
            "everything past the value is padding, {value:?}"
        );
    }
}

#[test]
fn a_plain_attribute_takes_the_content_width_and_a_sized_one_keeps_its_slot() {
    let shrunk =
        super::super::build_attr_message("a", &AttrValue::AsciiString("x".into())).unwrap();
    assert!(matches!(shrunk.datatype, Datatype::String { size: 1, .. }));

    let kept =
        super::super::build_attr_message("a", &AttrValue::ascii_string_sized("x", 5).unwrap())
            .unwrap();
    assert!(matches!(kept.datatype, Datatype::String { size: 5, .. }));
}

#[rstest::rstest]
#[case::ascii_scalar(AttrValue::ascii_string_sized("north", 2), 0, 5, 2)]
#[case::utf8_scalar(AttrValue::string_sized("mètre", 5), 0, 6, 5)]
#[case::ascii_array(AttrValue::ascii_string_array_sized(vec!["ab".into(), "north".into()], 4), 1, 5, 4)]
#[case::utf8_array(AttrValue::string_array_sized(vec!["ab".into(), "north".into()], 4), 1, 5, 4)]
fn an_attribute_value_past_its_declared_width_is_rejected(
    #[case] outcome: Result<AttrValue, FormatError>,
    #[case] index: usize,
    #[case] len: usize,
    #[case] width: u32,
) {
    assert_eq!(
        outcome.unwrap_err(),
        FormatError::FixedStringTooLong { index, len, width }
    );
}

// Empty values isolate zero-width rejection from per-value overflow.
#[test]
fn a_declared_attribute_width_of_zero_is_refused() {
    let refused = [
        AttrValue::ascii_string_sized("", 0).map(|_| ()),
        AttrValue::string_sized("", 0).map(|_| ()),
        AttrValue::ascii_string_array_sized(vec![], 0).map(|_| ()),
        AttrValue::string_array_sized(vec![], 0).map(|_| ()),
    ];
    for outcome in refused {
        assert!(
            matches!(outcome, Err(FormatError::ZeroFixedStringWidth)),
            "{outcome:?}"
        );
    }
}
