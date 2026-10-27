#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec;

use rstest::rstest;

use super::*;
use crate::datatype::layout::FloatingPointLayout;
use crate::{Error, H5Element, MantissaNormalization};
use crate::{FixedPointType, FloatingPointType};

macro_rules! assert_targets {
    ($assertion:ident, $($args:expr),+) => {
        $assertion!(read_as_f64_into, f64, $($args),+);
        $assertion!(read_as_f32_into, f32, $($args),+);
        $assertion!(read_as_i64_into, i64, $($args),+);
        $assertion!(read_as_i32_into, i32, $($args),+);
        $assertion!(read_as_i16_into, i16, $($args),+);
        $assertion!(read_as_i8_into, i8, $($args),+);
        $assertion!(read_as_u64_into, u64, $($args),+);
        $assertion!(read_as_u32_into, u32, $($args),+);
        $assertion!(read_as_u16_into, u16, $($args),+);
        $assertion!(read_as_u8_into, u8, $($args),+);
    };
}

macro_rules! assert_appends {
    ($read:ident, $target:ty, $raw:expr, $datatype:expr, $first:expr) => {{
        let mut values = vec![7 as $target];
        $read($raw, $datatype, &mut values).unwrap();
        assert_eq!(
            <$target>::convert_raw($raw, $datatype).unwrap(),
            vec![$first as $target, 3 as $target]
        );
        assert_eq!(
            values,
            vec![7 as $target, $first as $target, 3 as $target],
            stringify!($read)
        );
    }};
}

macro_rules! assert_rejects {
    ($read:ident, $target:ty, $raw:expr, $datatype:expr, $error:expr) => {{
        let mut values = vec![7 as $target];
        assert_eq!(
            $read($raw, $datatype, &mut values).unwrap_err(),
            $error,
            "{}: {:?}",
            stringify!($read),
            $datatype
        );
        assert_eq!(values, vec![7 as $target], stringify!($read));
        let error = <$target>::convert_raw($raw, $datatype).unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected numeric format error, got {error:?}");
        };
        assert_eq!(error, $error);
    }};
}

#[rstest]
fn reads_unsigned_offset_nibble_as_f64(#[values(false, true)] enumeration: bool) {
    let datatype = fixed(1, false, DatatypeByteOrder::LittleEndian, 4, 4);
    let datatype = maybe_enumeration(datatype, enumeration);
    assert_eq!(read_as_f64(&[0xB0], &datatype).unwrap(), vec![11.0]);
}

#[rstest]
#[case(1, 0, 8)]
#[case(2, 0, 16)]
#[case(3, 0, 24)]
#[case(4, 0, 32)]
#[case(5, 0, 40)]
#[case(6, 0, 48)]
#[case(7, 0, 56)]
#[case(8, 0, 64)]
#[case(1, 4, 4)]
#[case(2, 4, 8)]
#[case(8, 60, 4)]
fn appends_every_fixed_point_element(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[values(false, true)] signed: bool,
    #[values(DatatypeByteOrder::LittleEndian, DatatypeByteOrder::BigEndian)]
    order: DatatypeByteOrder,
    #[values(false, true)] enumeration: bool,
) {
    let datatype = maybe_enumeration(fixed(size, signed, order, offset, precision), enumeration);
    let first = if signed && precision == 4 { 3 } else { 11 };
    let raw = [first << offset, 3u64 << offset]
        .into_iter()
        .flat_map(|word| {
            let bytes = match order {
                DatatypeByteOrder::LittleEndian => word.to_le_bytes(),
                DatatypeByteOrder::BigEndian => word.to_be_bytes(),
                DatatypeByteOrder::Vax => panic!("test case uses an unsupported byte order"),
            };
            match order {
                DatatypeByteOrder::LittleEndian => bytes[..size as usize].to_vec(),
                DatatypeByteOrder::BigEndian => bytes[8 - size as usize..].to_vec(),
                DatatypeByteOrder::Vax => panic!("test case uses an unsupported byte order"),
            }
        })
        .collect::<Vec<_>>();
    assert_targets!(assert_appends, &raw, &datatype, first);
}

#[rstest]
#[case(4, FloatingPointLayout::IEEE754_BINARY32)]
#[case(8, FloatingPointLayout::IEEE754_BINARY64)]
fn appends_every_ieee_element(
    #[case] size: u32,
    #[case] layout: FloatingPointLayout,
    #[values(DatatypeByteOrder::LittleEndian, DatatypeByteOrder::BigEndian)]
    order: DatatypeByteOrder,
) {
    let datatype =
        Datatype::FloatingPoint(FloatingPointType::from_layout(size, order, layout).unwrap());
    let raw = if size == 4 {
        [11.0f32, 3.0]
            .into_iter()
            .flat_map(|value| match order {
                DatatypeByteOrder::LittleEndian => value.to_le_bytes(),
                DatatypeByteOrder::BigEndian => value.to_be_bytes(),
                DatatypeByteOrder::Vax => panic!("test case uses an unsupported byte order"),
            })
            .collect::<Vec<_>>()
    } else {
        [11.0f64, 3.0]
            .into_iter()
            .flat_map(|value| match order {
                DatatypeByteOrder::LittleEndian => value.to_le_bytes(),
                DatatypeByteOrder::BigEndian => value.to_be_bytes(),
                DatatypeByteOrder::Vax => panic!("test case uses an unsupported byte order"),
            })
            .collect::<Vec<_>>()
    };
    assert_targets!(assert_appends, &raw, &datatype, 11);
}

#[rstest]
#[case::bias32(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 31, 0, 32, 23, 8, 0, 23, 128).unwrap())]
#[case::bias64(FloatingPointType::new(8, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 63, 0, 64, 52, 11, 0, 52, 1024).unwrap())]
#[case::offset(FloatingPointType::new(5, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 32, 1, 32, 24, 8, 1, 23, 127).unwrap())]
#[case::precision(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 30, 0, 31, 22, 8, 0, 22, 127).unwrap())]
#[case::exponent_location(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 31, 0, 32, 22, 8, 0, 22, 127).unwrap())]
#[case::exponent_size(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 31, 0, 32, 23, 7, 0, 23, 127).unwrap())]
#[case::mantissa_location(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 31, 0, 32, 23, 8, 1, 22, 127).unwrap())]
#[case::mantissa_size(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 31, 0, 32, 23, 8, 0, 22, 127).unwrap())]
#[case::sign(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::ImpliedLeadingOne, 30, 0, 32, 23, 7, 0, 23, 127).unwrap())]
#[case::unnormalized(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::Unnormalized, 31, 0, 32, 23, 8, 0, 23, 127).unwrap())]
#[case::leading_one(FloatingPointType::new(4, DatatypeByteOrder::LittleEndian, MantissaNormalization::LeadingOne, 31, 0, 32, 23, 8, 0, 23, 127).unwrap())]
#[case::size(FloatingPointType::from_layout(5, DatatypeByteOrder::LittleEndian, FloatingPointLayout::IEEE754_BINARY32).unwrap())]
fn rejects_unsupported_float_layouts(
    #[case] numeric: FloatingPointType,
    #[values(&[][..], &[0][..], &[0; 8][..])] raw: &[u8],
) {
    let datatype = Datatype::FloatingPoint(numeric);
    let error = FormatError::UnsupportedFloatingPointLayout {
        size: numeric.size().get() as usize,
        layout: numeric.layout(),
    };
    assert_targets!(assert_rejects, raw, &datatype, error);
}

#[rstest]
#[case(Datatype::FloatingPoint(FloatingPointType::from_layout(4, DatatypeByteOrder::Vax, FloatingPointLayout::IEEE754_BINARY32).unwrap()))]
fn rejects_unsupported_byte_order(
    #[case] datatype: Datatype,
    #[values(&[][..], &[0][..], &[0; 8][..])] raw: &[u8],
) {
    assert_targets!(
        assert_rejects,
        raw,
        &datatype,
        FormatError::UnsupportedVaxByteOrder
    );
}

#[rstest]
fn rejects_partial_elements(#[values(false, true)] floating: bool) {
    let datatype = if floating {
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                4,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY32,
            )
            .unwrap(),
        )
    } else {
        fixed(2, true, DatatypeByteOrder::LittleEndian, 4, 8)
    };
    let raw = [0; 5];
    assert_targets!(
        assert_rejects,
        &raw,
        &datatype,
        FormatError::DataSizeMismatch {
            expected: 4,
            actual: 5
        }
    );
}

#[rstest]
#[case(2, 1)]
#[case(1, 2)]
#[case(0, 1)]
fn rejects_enumeration_width_mismatch(
    #[case] size: u32,
    #[case] base_size: u32,
    #[values(&[][..], &[0][..], &[0; 8][..])] raw: &[u8],
    #[values(false, true)] nested: bool,
) {
    let datatype = Datatype::Enumeration {
        size,
        base_type: Box::new(fixed(
            base_size,
            false,
            DatatypeByteOrder::LittleEndian,
            0,
            8,
        )),
        members: Vec::new(),
    };
    let datatype = maybe_enumeration(datatype, nested);
    assert_targets!(
        assert_rejects,
        raw,
        &datatype,
        FormatError::EnumBaseSizeMismatch { size, base_size }
    );
}

#[rstest]
#[case(1, 4, 4, &[0xB0, 0x70])]
#[case(3, 4, 4, &[0xB0, 0, 0, 0x70, 0, 0])]
#[case(8, 60, 4, &[0, 0, 0, 0, 0, 0, 0, 0xB0, 0, 0, 0, 0, 0, 0, 0, 0x70])]
fn sign_extends_supported_ranges(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[case] raw: &[u8],
    #[values(false, true)] enumeration: bool,
) {
    let datatype = maybe_enumeration(
        fixed(
            size,
            true,
            DatatypeByteOrder::LittleEndian,
            offset,
            precision,
        ),
        enumeration,
    );
    assert_eq!(read_as_i64(raw, &datatype).unwrap(), vec![-5, 7]);
    assert_eq!(i8::convert_raw(raw, &datatype).unwrap(), [-5, 7]);
    assert_eq!(u8::convert_raw(raw, &datatype).unwrap(), [0, 7]);
    assert_eq!(read_as_f64(raw, &datatype).unwrap(), vec![-5.0, 7.0]);
    assert_eq!(read_as_f32(raw, &datatype).unwrap(), vec![-5.0, 7.0]);
}

fn fixed(
    size: u32,
    signed: bool,
    byte_order: DatatypeByteOrder,
    bit_offset: u16,
    bit_precision: u16,
) -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(size, byte_order, signed, bit_offset, bit_precision).unwrap(),
    )
}

fn maybe_enumeration(datatype: Datatype, enumeration: bool) -> Datatype {
    if enumeration {
        Datatype::Enumeration {
            size: datatype.type_size(),
            base_type: Box::new(datatype),
            members: Vec::new(),
        }
    } else {
        datatype
    }
}

#[rstest]
fn decodes_ieee_binary32_with_each_padding_value(
    #[values(false, true)] low_padding: bool,
    #[values(false, true)] high_padding: bool,
    #[values(false, true)] internal_padding: bool,
) {
    let datatype = Datatype::FloatingPoint(
        FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian).with_padding(
            low_padding,
            high_padding,
            internal_padding,
        ),
    );
    assert_eq!(
        super::read_as_f64(&1.5f32.to_le_bytes(), &datatype).unwrap(),
        [1.5],
    );
}

#[rstest]
#[case::empty(&[])]
#[case::nonempty(b"text")]
fn rejects_nonnumeric_element_conversion(#[case] raw: &[u8]) {
    let datatype = Datatype::String {
        size: 4,
        padding: crate::StringPadding::NullPad,
        charset: crate::CharacterSet::Ascii,
    };
    assert_targets!(
        assert_rejects,
        raw,
        &datatype,
        FormatError::TypeMismatch {
            expected: "numeric",
            actual: "String"
        }
    );
}
