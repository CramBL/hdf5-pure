#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec;

use rstest::rstest;

use super::*;
use crate::MantissaNormalization;
use crate::datatype::layout::FloatingPointLayout;

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
    let datatype = Datatype::FloatingPoint {
        size,
        byte_order: order,
        layout,
    };
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
#[case(1, 0, 0)]
#[case(1, 8, 1)]
#[case(1, 4, 5)]
#[case(2, 0, 17)]
#[case(8, 64, 1)]
#[case(8, u16::MAX, u16::MAX)]
fn rejects_invalid_fixed_point_ranges_before_decoding(
    #[case] size: u32,
    #[case] bit_offset: u16,
    #[case] bit_precision: u16,
    #[values(&[][..], &[0][..], &[0; 8][..])] raw: &[u8],
    #[values(false, true)] enumeration: bool,
) {
    let datatype = maybe_enumeration(
        fixed(
            size,
            false,
            DatatypeByteOrder::LittleEndian,
            bit_offset,
            bit_precision,
        ),
        enumeration,
    );
    let error = FormatError::InvalidFixedPointRange {
        size: size as usize,
        bit_offset,
        bit_precision,
    };
    assert_targets!(assert_rejects, raw, &datatype, error);
}

#[rstest]
#[case::bias32(4, FloatingPointLayout { exponent_bias: 128, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::bias64(8, FloatingPointLayout { exponent_bias: 1024, ..FloatingPointLayout::IEEE754_BINARY64 })]
#[case::offset(4, FloatingPointLayout { bit_offset: 1, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::precision(4, FloatingPointLayout { bit_precision: 31, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::exponent_location(4, FloatingPointLayout { exponent_location: 22, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::exponent_size(4, FloatingPointLayout { exponent_size: 7, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::mantissa_location(4, FloatingPointLayout { mantissa_location: 1, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::mantissa_size(4, FloatingPointLayout { mantissa_size: 22, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::sign(4, FloatingPointLayout { sign_location: 30, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::unnormalized(4, FloatingPointLayout { mantissa_normalization: MantissaNormalization::Unnormalized, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::leading_one(4, FloatingPointLayout { mantissa_normalization: MantissaNormalization::LeadingOne, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::reserved(4, FloatingPointLayout { mantissa_normalization: MantissaNormalization::Reserved, ..FloatingPointLayout::IEEE754_BINARY32 })]
#[case::size(2, FloatingPointLayout::IEEE754_BINARY32)]
fn rejects_unsupported_float_layouts(
    #[case] size: u32,
    #[case] layout: FloatingPointLayout,
    #[values(&[][..], &[0][..], &[0; 8][..])] raw: &[u8],
) {
    let datatype = Datatype::FloatingPoint {
        size,
        byte_order: DatatypeByteOrder::LittleEndian,
        layout,
    };
    let error = FormatError::UnsupportedFloatingPointLayout {
        size: size as usize,
        layout,
    };
    assert_targets!(assert_rejects, raw, &datatype, error);
}

#[rstest]
#[case(fixed(1, false, DatatypeByteOrder::Vax, 0, 8))]
#[case(Datatype::FloatingPoint { size: 4, byte_order: DatatypeByteOrder::Vax,
    layout: FloatingPointLayout::IEEE754_BINARY32 })]
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
        Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FloatingPointLayout::IEEE754_BINARY32,
        }
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
    Datatype::FixedPoint {
        size,
        byte_order,
        layout: FixedPointLayout {
            signed,
            bit_offset,
            bit_precision,
        },
    }
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
