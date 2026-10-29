use super::super::*;
use crate::{
    DatatypeByteOrder,
    datatype::{CharacterSet, Datatype, StringPadding, layout::FloatingPointLayout},
};

#[test]
fn an_array_shape_is_not_a_debug_slice() {
    let dtype = DType::Array(Box::new(DType::F32), vec![2, 3]);
    assert_eq!(dtype.to_string(), "array<f32, 2x3>");
    assert_eq!(
        DType::Array(Box::new(DType::U8), vec![4]).to_string(),
        "array<u8, 4>"
    );
}

/// An unclassified datatype carries the type, and writes as the summary of
/// it. The whole `Debug` record is unreadable in the message that quotes it.
#[test]
fn an_unclassified_type_carries_the_type_and_writes_a_summary() {
    let vax = Datatype::FloatingPoint(
        FloatingPointType::from_layout(
            4,
            DatatypeByteOrder::Vax,
            FloatingPointLayout::IEEE754_BINARY32,
        )
        .unwrap(),
    );
    // Floating-point classification uses storage size.
    assert_eq!(classify_datatype(&vax), DType::F32);

    let time = Datatype::Time(TimeType::new(4, DatatypeByteOrder::LittleEndian, 32).unwrap());
    let classified = classify_datatype(&time);
    assert_eq!(classified, DType::Other(Box::new(time.clone())));
    assert_eq!(classified.to_string(), "other(time32)");

    let opaque = Datatype::Opaque {
        size: 3,
        tag: b"rgb".to_vec(),
    };
    assert_eq!(
        classify_datatype(&opaque).to_string(),
        "other(opaque[3] \"rgb\")",
        "not `other(Opaque {{ size: 3, tag: [114, 103, 98] }})`"
    );
}

#[test]
fn a_curated_member_name_is_escaped_in_either_variant() {
    let compound = DType::Compound(vec![("a\nb".into(), DType::I32)]).to_string();
    assert!(!compound.chars().any(char::is_control), "{compound}");
    assert_eq!(compound, "compound{a\\nb: i32}");

    let enumeration = DType::Enum(vec!["a\u{1b}[31mb".into()]).to_string();
    assert!(!enumeration.chars().any(char::is_control), "{enumeration}");
    assert_eq!(enumeration, "enum[a\\u{1b}[31mb]");
}

/// Likewise for the cap: a file can declare far more members than a message
/// can carry, in either variant.
#[test]
fn a_curated_member_list_is_elided_in_either_variant() {
    let over_cap = DISPLAY_MAX_MEMBERS + 2;
    let names: Vec<String> = (0..over_cap).map(|i| format!("m{i}")).collect();

    let compound = DType::Compound(
        names
            .iter()
            .map(|name| (name.clone(), DType::I32))
            .collect(),
    );
    let enumeration = DType::Enum(names);

    for (dtype, close) in [(compound, "}"), (enumeration, "]")] {
        let shown = dtype.to_string();
        assert!(shown.ends_with(&format!(", … 2 more{close}")), "{shown}");
        assert!(
            !shown.contains(&format!("m{DISPLAY_MAX_MEMBERS}")),
            "{shown}"
        );
    }
}

/// The two views describe the same file, so a type that classifies to a
/// named [`DType`] is spelled the same way by both. Where they differ, the
/// `Datatype` is the longer of the two, never a different word: it carries
/// the on-disk detail `DType` drops.
#[test]
fn dtype_and_datatype_agree_on_the_names_they_share() {
    let identical = [
        Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        ),
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                4,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY32,
            )
            .unwrap(),
        ),
    ];
    for datatype in identical {
        assert_eq!(
            classify_datatype(&datatype).to_string(),
            datatype.to_string()
        );
    }

    // The full string datatype includes width, character set, and padding.
    let string = Datatype::String {
        size: 8,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Ascii,
    };
    assert_eq!(classify_datatype(&string).to_string(), "string");
    assert_eq!(string.to_string(), "string[8] ascii null-pad");
    assert!(
        string
            .to_string()
            .starts_with(&classify_datatype(&string).to_string()),
        "the longer spelling still opens with the shorter one"
    );
}
