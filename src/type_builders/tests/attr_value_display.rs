use super::super::{ATTR_DISPLAY_MAX_ELEMENTS, AttrValue};

/// Returns a sample of every attribute variant.
///
/// The match is exhaustive over [`AttrValue`].
fn one_of_every_variant() -> Vec<AttrValue> {
    let values = vec![
        AttrValue::F32(0.0),
        AttrValue::F32Array(vec![]),
        AttrValue::F64(0.0),
        AttrValue::F64Array(vec![]),
        AttrValue::I8(0),
        AttrValue::I8Array(vec![]),
        AttrValue::I16(0),
        AttrValue::I16Array(vec![]),
        AttrValue::I32(0),
        AttrValue::I32Array(vec![]),
        AttrValue::I64(0),
        AttrValue::I64Array(vec![]),
        AttrValue::U8(0),
        AttrValue::U8Array(vec![]),
        AttrValue::U16(0),
        AttrValue::U16Array(vec![]),
        AttrValue::U32(0),
        AttrValue::U32Array(vec![]),
        AttrValue::U64(0),
        AttrValue::U64Array(vec![]),
        AttrValue::String(String::new()),
        AttrValue::string_sized("", 4).unwrap(),
        AttrValue::StringArray(vec![]),
        AttrValue::string_array_sized(vec![], 4).unwrap(),
        AttrValue::AsciiString(String::new()),
        AttrValue::ascii_string_sized("", 4).unwrap(),
        AttrValue::AsciiStringArray(vec![]),
        AttrValue::ascii_string_array_sized(vec![], 4).unwrap(),
        AttrValue::VarLenAsciiCharArray(vec![]),
        AttrValue::VarLenString(String::new()),
        AttrValue::VarLenStringArray(vec![]),
        AttrValue::VarLenAsciiString(String::new()),
        AttrValue::VarLenAsciiStringArray(vec![]),
    ];
    for value in &values {
        match value {
            AttrValue::F32(_) | AttrValue::F32Array(_) => {}
            AttrValue::F64(_) | AttrValue::F64Array(_) => {}
            AttrValue::I8(_) | AttrValue::I8Array(_) => {}
            AttrValue::I16(_) | AttrValue::I16Array(_) => {}
            AttrValue::I32(_) | AttrValue::I32Array(_) => {}
            AttrValue::I64(_) | AttrValue::I64Array(_) => {}
            AttrValue::U8(_) | AttrValue::U8Array(_) => {}
            AttrValue::U16(_) | AttrValue::U16Array(_) => {}
            AttrValue::U32(_) | AttrValue::U32Array(_) => {}
            AttrValue::U64(_) | AttrValue::U64Array(_) => {}
            AttrValue::String(_) | AttrValue::StringArray(_) => {}
            AttrValue::StringSized { .. } | AttrValue::StringArraySized { .. } => {}
            AttrValue::AsciiString(_) | AttrValue::AsciiStringArray(_) => {}
            AttrValue::AsciiStringSized { .. } | AttrValue::AsciiStringArraySized { .. } => {}
            AttrValue::VarLenAsciiCharArray(_) => {}
            AttrValue::VarLenString(_) | AttrValue::VarLenStringArray(_) => {}
            AttrValue::VarLenAsciiString(_) | AttrValue::VarLenAsciiStringArray(_) => {}
        }
    }
    values
}

/// A caller that fell through its own `_` arm has only this name to report,
/// so no two variants may share one.
#[test]
fn type_name_is_distinct_for_every_variant() {
    let values = one_of_every_variant();

    let mut names: Vec<&str> = values.iter().map(AttrValue::type_name).collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), count, "two variants share a type name");
    assert!(!names.contains(&""));
}

#[test]
fn display_writes_something_for_every_variant() {
    for value in one_of_every_variant() {
        assert!(
            !value.to_string().is_empty(),
            "{} writes nothing",
            value.type_name()
        );
    }
}

#[test]
fn display_writes_the_value_not_the_variant() {
    assert_eq!(AttrValue::F64(1.5).to_string(), "1.5");
    assert_eq!(AttrValue::F32(1.5).to_string(), "1.5");
    assert_eq!(AttrValue::F32(1.0).to_string(), "1.0");
    assert_eq!(AttrValue::I8(-7).to_string(), "-7");
    assert_eq!(AttrValue::I16(-7).to_string(), "-7");
    assert_eq!(AttrValue::I32(-7).to_string(), "-7");
    assert_eq!(AttrValue::U8(255).to_string(), "255");
    assert_eq!(AttrValue::U16(65_535).to_string(), "65535");
    assert_eq!(AttrValue::U32(7).to_string(), "7");
    assert_eq!(AttrValue::U8Array(vec![1, 2]).to_string(), "[1, 2]");
    assert_eq!(AttrValue::I16Array(vec![-1, 2]).to_string(), "[-1, 2]");
    assert_eq!(AttrValue::U64(u64::MAX).to_string(), "18446744073709551615");
    assert_eq!(AttrValue::String("metres".into()).to_string(), "\"metres\"");
    assert_eq!(
        AttrValue::I64Array(vec![1, 2, 3]).to_string(),
        "[1, 2, 3]",
        "no `I64Array(..)` wrapper, which is what `Debug` is for"
    );
    assert_eq!(
        AttrValue::StringArray(vec!["a".into(), "b".into()]).to_string(),
        "[\"a\", \"b\"]"
    );
    assert_eq!(AttrValue::F64Array(vec![]).to_string(), "[]");
}

#[test]
fn display_keeps_the_point_on_a_whole_float() {
    assert_eq!(AttrValue::F64(1.0).to_string(), "1.0");
    assert_eq!(
        AttrValue::F64Array(vec![1.0, 2.5]).to_string(),
        "[1.0, 2.5]"
    );
}

#[test]
fn display_elides_a_long_array_and_reports_the_remainder() {
    let values: Vec<i64> = (0..ATTR_DISPLAY_MAX_ELEMENTS as i64 + 5).collect();
    let shown = AttrValue::I64Array(values).to_string();

    assert!(shown.ends_with(", … 5 more]"), "{shown}");
    assert_eq!(shown.matches(", ").count(), ATTR_DISPLAY_MAX_ELEMENTS);
}

/// The boundary: exactly the cap is written whole, with no "0 more".
#[test]
fn display_does_not_elide_at_exactly_the_cap() {
    let values: Vec<i64> = (0..ATTR_DISPLAY_MAX_ELEMENTS as i64).collect();
    let shown = AttrValue::I64Array(values).to_string();

    assert!(!shown.contains('…'), "{shown}");
    assert!(shown.ends_with("7]"), "{shown}");
}
