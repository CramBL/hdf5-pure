use crate::CharacterSet;
use crate::MantissaNormalization;
use crate::StringPadding;

use crate::data_read::*;

/// Builds a fixed-point datatype with all bits significant.
fn wide_int(size: u32, signed: bool, byte_order: DatatypeByteOrder) -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(
            size,
            byte_order,
            signed,
            0,
            u16::try_from(size * 8).expect("test widths fit a u16"),
        )
        .unwrap(),
    )
}

/// Builds an 80-bit floating-point layout in 16-byte storage.
fn wide_float() -> Datatype {
    Datatype::FloatingPoint(
        FloatingPointType::new(
            16,
            DatatypeByteOrder::LittleEndian,
            MantissaNormalization::LeadingOne,
            79,
            0,
            80,
            64,
            15,
            0,
            64,
            16383,
        )
        .unwrap(),
    )
}

/// A numeric reader with its output type erased.
type Reader = (
    &'static str,
    fn(&[u8], &Datatype) -> Result<(), FormatError>,
);

/// Numeric conversion functions with their successful values discarded.
const READERS: [Reader; 8] = [
    ("i64", |raw, dt| read_as_i64(raw, dt).map(|_| ())),
    ("u64", |raw, dt| read_as_u64(raw, dt).map(|_| ())),
    ("i32", |raw, dt| read_as_i32(raw, dt).map(|_| ())),
    ("u32", |raw, dt| read_as_u32(raw, dt).map(|_| ())),
    ("i16", |raw, dt| read_as_i16(raw, dt).map(|_| ())),
    ("u16", |raw, dt| read_as_u16(raw, dt).map(|_| ())),
    ("f64", |raw, dt| read_as_f64(raw, dt).map(|_| ())),
    ("f32", |raw, dt| read_as_f32(raw, dt).map(|_| ())),
];

#[test]
fn listed_numeric_readers_reject_sixteen_byte_elements() {
    let raw = [0xFFu8; 16];
    for dt in [
        wide_int(16, false, DatatypeByteOrder::LittleEndian),
        wide_int(16, true, DatatypeByteOrder::BigEndian),
        wide_float(),
    ] {
        for (name, read) in READERS {
            assert!(
                matches!(
                    read(&raw, &dt),
                    Err(FormatError::NumericElementTooWide { size: 16 })
                ),
                "read_as_{name} accepted a {dt}"
            );
        }
    }
}

#[test]
fn eight_bytes_decodes_at_full_range_and_nine_does_not() {
    let unsigned = wide_int(8, false, DatatypeByteOrder::LittleEndian);
    assert_eq!(
        read_as_u64(&u64::MAX.to_le_bytes(), &unsigned).unwrap(),
        vec![u64::MAX]
    );
    let signed = wide_int(8, true, DatatypeByteOrder::LittleEndian);
    assert_eq!(
        read_as_i64(&i64::MIN.to_le_bytes(), &signed).unwrap(),
        vec![i64::MIN]
    );

    // A nine-byte element exceeds the supported width even when its value is zero.
    assert!(matches!(
        read_as_u64(
            &[0u8; 9],
            &wide_int(9, false, DatatypeByteOrder::LittleEndian)
        ),
        Err(FormatError::NumericElementTooWide { size: 9 })
    ));
}

#[test]
fn an_enumeration_over_a_wide_base_is_rejected() {
    let dt = Datatype::Enumeration {
        size: 16,
        base_type: Box::new(wide_int(16, false, DatatypeByteOrder::LittleEndian)),
        members: Vec::new(),
    };
    assert!(matches!(
        read_as_u64(&[0xFFu8; 16], &dt),
        Err(FormatError::NumericElementTooWide { size: 16 })
    ));
}

// A variable-length string has 16-byte elements but is not numeric.
#[test]
fn a_non_numeric_datatype_is_rejected_for_its_class_not_its_width() {
    let vlen = Datatype::VariableLength(VariableLengthType::String(
        VariableLengthStringType::with_base_type(
            StringPadding::NullTerminate,
            CharacterSet::Ascii,
            wide_int(1, false, DatatypeByteOrder::LittleEndian),
        )
        .unwrap(),
    ));
    assert_eq!(
        vlen.type_size(),
        16,
        "the premise: it looks like a wide one"
    );
    assert_eq!(
        read_as_u64(&[0u8; 16], &vlen).unwrap_err(),
        FormatError::TypeMismatch {
            expected: "numeric",
            actual: "VariableLength",
        }
    );
}

#[test]
fn a_wide_element_is_rejected_before_its_length_is_judged() {
    let dt = wide_int(16, false, DatatypeByteOrder::LittleEndian);
    // Three bytes cannot contain a whole 16-byte element.
    assert!(matches!(
        read_as_u64(&[0xFFu8; 3], &dt),
        Err(FormatError::NumericElementTooWide { size: 16 })
    ));
}

#[test]
fn reads_standard_unsigned_u8_as_f32() {
    let datatype = Datatype::FixedPoint(
        FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap(),
    );

    let values = read_as_f32(&[0xff], &datatype).unwrap();

    assert_eq!(values, vec![255.0]);
}
