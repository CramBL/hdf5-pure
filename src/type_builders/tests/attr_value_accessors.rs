use super::super::AttrValue;

/// Every representation of a single string reaches `as_str`. The point of
/// the accessor is that a caller need not know which one a file yielded.
#[test]
fn as_str_spans_every_single_string_shape() {
    for value in [
        AttrValue::String("double".into()),
        AttrValue::AsciiString("double".into()),
        AttrValue::StringArray(vec!["double".into()]),
        AttrValue::AsciiStringArray(vec!["double".into()]),
        AttrValue::VarLenAsciiCharArray(vec!["double".into()]),
        AttrValue::string_sized("double", 32).unwrap(),
        AttrValue::ascii_string_sized("double", 32).unwrap(),
        AttrValue::string_array_sized(vec!["double".into()], 32).unwrap(),
        AttrValue::ascii_string_array_sized(vec!["double".into()], 32).unwrap(),
    ] {
        assert_eq!(value.as_str(), Some("double"), "{value:?}");
    }
}

/// "Exactly one" is the contract: an array of two is not a single string,
/// and neither is a numeric value.
#[test]
fn as_str_rejects_non_single_strings() {
    for value in [
        AttrValue::StringArray(vec!["a".into(), "b".into()]),
        AttrValue::AsciiStringArray(vec!["a".into(), "b".into()]),
        AttrValue::VarLenAsciiCharArray(vec![]),
        AttrValue::string_array_sized(vec!["a".into(), "b".into()], 8).unwrap(),
        AttrValue::ascii_string_array_sized(vec![], 8).unwrap(),
        AttrValue::F64(1.5),
        AttrValue::I64(3),
    ] {
        assert_eq!(value.as_str(), None, "{value:?}");
    }
}

/// A scalar is viewed as a one-element slice, and the view borrows: no
/// element is copied to produce it.
#[test]
fn as_strings_reads_a_scalar_as_one_element() {
    for value in [
        AttrValue::String("m/s".into()),
        AttrValue::AsciiString("m/s".into()),
        AttrValue::string_sized("m/s", 8).unwrap(),
        AttrValue::ascii_string_sized("m/s", 8).unwrap(),
    ] {
        let seen = value.as_strings().expect("a string value");
        assert_eq!(seen, ["m/s"], "{value:?}");
        assert_eq!(seen.len(), 1, "{value:?}");
    }
}

#[test]
fn as_strings_keeps_every_element_of_each_array_shape() {
    let fields: Vec<String> = vec!["x".into(), "y".into(), "velocity".into()];
    for value in [
        AttrValue::StringArray(fields.clone()),
        AttrValue::AsciiStringArray(fields.clone()),
        AttrValue::VarLenAsciiCharArray(fields.clone()),
        AttrValue::string_array_sized(fields.clone(), 16).unwrap(),
        AttrValue::ascii_string_array_sized(fields.clone(), 16).unwrap(),
    ] {
        assert_eq!(
            value.as_strings().expect("a string value"),
            ["x", "y", "velocity"],
            "{value:?}"
        );
    }
}

#[test]
fn as_strings_separates_empty_from_absent() {
    let empty = AttrValue::StringArray(vec![]);
    assert_eq!(empty.as_strings(), Some(&[][..]));
    assert_eq!(AttrValue::I64(1).as_strings(), None);
}

/// Returns scalar and one-element array samples for every integer variant.
///
/// Signed values are -7 to check sign preservation during widening.
fn every_integer_variant() -> Vec<(AttrValue, i64)> {
    vec![
        (AttrValue::I8(-7), -7),
        (AttrValue::I16(-7), -7),
        (AttrValue::I32(-7), -7),
        (AttrValue::I64(-7), -7),
        (AttrValue::U8(7), 7),
        (AttrValue::U16(7), 7),
        (AttrValue::U32(7), 7),
        (AttrValue::U64(7), 7),
        (AttrValue::I8Array(vec![-7]), -7),
        (AttrValue::I16Array(vec![-7]), -7),
        (AttrValue::I32Array(vec![-7]), -7),
        (AttrValue::I64Array(vec![-7]), -7),
        (AttrValue::U8Array(vec![7]), 7),
        (AttrValue::U16Array(vec![7]), 7),
        (AttrValue::U32Array(vec![7]), 7),
        (AttrValue::U64Array(vec![7]), 7),
    ]
}

#[test]
fn as_i64_widens_every_integer_variant() {
    for (value, expected) in every_integer_variant() {
        assert_eq!(value.as_i64(), Some(expected), "{}", value.type_name());
        assert_eq!(
            value.to_i64s(),
            Some(vec![expected]),
            "{} through the plural accessor",
            value.type_name()
        );
    }
}

/// The same span for `u64`, where the signed variants' `-7` has no value and
/// the unsigned ones do.
#[test]
fn as_u64_widens_every_integer_variant() {
    for (value, expected) in every_integer_variant() {
        let expected = u64::try_from(expected).ok();
        assert_eq!(value.as_u64(), expected, "{}", value.type_name());
        assert_eq!(
            value.to_u64s(),
            expected.map(|v| vec![v]),
            "{} through the plural accessor",
            value.type_name()
        );
    }
}

#[test]
fn the_widest_value_of_each_width_reads_as_itself() {
    assert_eq!(AttrValue::U8(u8::MAX).as_u64(), Some(255));
    assert_eq!(AttrValue::U16(u16::MAX).as_u64(), Some(65_535));
    assert_eq!(AttrValue::U32(u32::MAX).as_u64(), Some(4_294_967_295));
    assert_eq!(AttrValue::I8(i8::MIN).as_i64(), Some(-128));
    assert_eq!(AttrValue::I16(i16::MIN).as_i64(), Some(-32_768));
    assert_eq!(AttrValue::I32(i32::MIN).as_i64(), Some(-2_147_483_648));
    assert_eq!(
        AttrValue::U8Array(vec![u8::MAX, 0]).to_i64s(),
        Some(vec![255, 0])
    );
    assert_eq!(
        AttrValue::I16Array(vec![i16::MIN, i16::MAX]).to_i64s(),
        Some(vec![-32_768, 32_767])
    );
}

// Test both signed-to-unsigned and unsigned-to-signed range failures.
#[test]
fn scalar_accessors_refuse_a_value_that_does_not_fit() {
    let past_max = (i64::MAX as u64) + 1;
    assert_eq!(AttrValue::U64(u64::MAX).as_i64(), None);
    assert_eq!(AttrValue::U64(past_max).as_i64(), None);
    assert_eq!(AttrValue::U64Array(vec![past_max]).as_i64(), None);
    assert_eq!(
        AttrValue::U64(i64::MAX as u64).as_i64(),
        Some(i64::MAX),
        "the largest value that does fit must still be readable"
    );

    assert_eq!(AttrValue::I64(-1).as_u64(), None);
    assert_eq!(AttrValue::I32(-1).as_u64(), None);
    assert_eq!(AttrValue::I64Array(vec![-1]).as_u64(), None);
    assert_eq!(AttrValue::I64(0).as_u64(), Some(0));
}

#[test]
fn as_i64_rejects_multi_element_and_non_integer() {
    assert_eq!(AttrValue::I64Array(vec![1, 2]).as_i64(), None);
    assert_eq!(AttrValue::U64Array(vec![1, 2]).as_i64(), None);
    assert_eq!(AttrValue::F64(1.0).as_i64(), None);
    assert_eq!(AttrValue::String("1".into()).as_i64(), None);
    assert_eq!(AttrValue::F64(1.0).as_u64(), None);
}

#[test]
fn to_i64s_reads_scalars_and_arrays_alike() {
    assert_eq!(AttrValue::I64(4).to_i64s(), Some(vec![4]));
    assert_eq!(AttrValue::I32(4).to_i64s(), Some(vec![4]));
    assert_eq!(AttrValue::U32(4).to_i64s(), Some(vec![4]));
    assert_eq!(AttrValue::U64(4).to_i64s(), Some(vec![4]));
    assert_eq!(
        AttrValue::I64Array(vec![1, 2, 3]).to_i64s(),
        Some(vec![1, 2, 3])
    );
    assert_eq!(AttrValue::U64Array(vec![1, 2]).to_i64s(), Some(vec![1, 2]));
    assert_eq!(AttrValue::I64Array(vec![]).to_i64s(), Some(vec![]));
    assert_eq!(AttrValue::U64Array(vec![]).to_i64s(), Some(vec![]));
    assert_eq!(AttrValue::F64Array(vec![1.0]).to_i64s(), None);
}

#[test]
fn to_u64s_reads_scalars_and_arrays_alike() {
    assert_eq!(AttrValue::U64(4).to_u64s(), Some(vec![4]));
    assert_eq!(AttrValue::U32(4).to_u64s(), Some(vec![4]));
    assert_eq!(AttrValue::I64(4).to_u64s(), Some(vec![4]));
    assert_eq!(
        AttrValue::U64Array(vec![1, u64::MAX]).to_u64s(),
        Some(vec![1, u64::MAX])
    );
    assert_eq!(AttrValue::I64Array(vec![1, 2]).to_u64s(), Some(vec![1, 2]));
    assert_eq!(AttrValue::F64(1.0).to_u64s(), None);
}

#[test]
fn plural_accessors_apply_the_range_rule_to_every_element() {
    let past_max = (i64::MAX as u64) + 1;
    assert_eq!(AttrValue::U64Array(vec![1, past_max]).to_i64s(), None);
    assert_eq!(AttrValue::U64Array(vec![past_max, 1]).to_i64s(), None);
    assert_eq!(AttrValue::U64(u64::MAX).to_i64s(), None);
    assert_eq!(
        AttrValue::U64Array(vec![1, i64::MAX as u64]).to_i64s(),
        Some(vec![1, i64::MAX]),
        "every element fitting must still read"
    );

    assert_eq!(AttrValue::I64Array(vec![1, -1]).to_u64s(), None);
    assert_eq!(AttrValue::I64Array(vec![-1, 1]).to_u64s(), None);
    assert_eq!(AttrValue::I64(-1).to_u64s(), None);
}

#[test]
fn as_f64_reads_one_float_from_either_shape() {
    assert_eq!(AttrValue::F64(1.5).as_f64(), Some(1.5));
    assert_eq!(AttrValue::F64Array(vec![1.5]).as_f64(), Some(1.5));
    assert_eq!(AttrValue::F64Array(vec![1.5, 2.5]).as_f64(), None);
}

// Extreme binary32 values widen exactly to binary64.
#[test]
fn the_float_accessors_span_both_widths() {
    assert_eq!(AttrValue::F32(1.5).as_f64(), Some(1.5));
    assert_eq!(AttrValue::F32Array(vec![1.5]).as_f64(), Some(1.5));
    assert_eq!(AttrValue::F32Array(vec![1.5, 2.5]).as_f64(), None);
    assert_eq!(AttrValue::F32(f32::MAX).as_f64(), Some(f64::from(f32::MAX)));
    assert_eq!(AttrValue::F32(1.5).to_f64s(), Some(vec![1.5]));
    assert_eq!(
        AttrValue::F32Array(vec![f32::MIN, f32::MAX]).to_f64s(),
        Some(vec![f64::from(f32::MIN), f64::from(f32::MAX)])
    );
    assert_eq!(AttrValue::F32Array(vec![]).to_f64s(), Some(vec![]));
    // An integer is not a float, at either width.
    assert_eq!(AttrValue::F32(1.0).as_i64(), None);
    assert_eq!(AttrValue::I32(1).as_f64(), None);
}

#[test]
fn float_accessors_do_not_convert_integers() {
    assert_eq!(AttrValue::I64(1).as_f64(), None);
    assert_eq!(AttrValue::U32(1).as_f64(), None);
    assert_eq!(AttrValue::I64Array(vec![1]).to_f64s(), None);
    assert_eq!(AttrValue::U64Array(vec![1]).to_f64s(), None);
}

#[test]
fn to_f64s_reads_scalars_and_arrays_alike() {
    assert_eq!(AttrValue::F64(1.5).to_f64s(), Some(vec![1.5]));
    assert_eq!(
        AttrValue::F64Array(vec![1.5, 2.5]).to_f64s(),
        Some(vec![1.5, 2.5])
    );
    assert_eq!(AttrValue::F64Array(vec![]).to_f64s(), Some(vec![]));
    assert_eq!(AttrValue::String("1.5".into()).to_f64s(), None);
}
