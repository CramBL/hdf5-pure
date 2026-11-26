use rstest::rstest;

use super::{
    FLOATING_POINT_CLASS, FloatingPointField, FloatingPointLayout, FloatingPointType,
    MantissaNormalization,
};
use crate::{DatatypeByteOrder, FormatError};

#[rstest]
fn preserves_floating_point_layout_padding(
    #[values(false, true)] low: bool,
    #[values(false, true)] high: bool,
    #[values(false, true)] internal: bool,
) {
    let layout = FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian)
        .with_padding(low, high, internal)
        .layout();
    assert_eq!(
        FloatingPointType::from_layout(4, DatatypeByteOrder::BigEndian, layout).unwrap(),
        FloatingPointType::ieee754_binary32(DatatypeByteOrder::BigEndian)
            .with_padding(low, high, internal),
    );
}

#[rstest]
#[case::zero_size(0, FloatingFields::BINARY32, FormatError::ZeroSizedDatatype { class: FLOATING_POINT_CLASS })]
#[case::zero_precision(4, FloatingFields { bit_precision: 0, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 0, bit_precision: 0 })]
#[case::precision_past_offset_end(4, FloatingFields { bit_offset: 1, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 1, bit_precision: 32 })]
#[case::binary32_in_two_bytes(2, FloatingFields::BINARY32, FormatError::InvalidFloatingPointRange { size: 2, bit_offset: 0, bit_precision: 32 })]
#[case::sign_at_precision_end(4, FloatingFields { sign_location: 32, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Sign, location: 32, size: 1, bit_offset: 0, bit_precision: 32 })]
#[case::zero_exponent_size(4, FloatingFields { exponent_size: 0, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 0, bit_offset: 0, bit_precision: 32 })]
#[case::exponent_past_precision_end(4, FloatingFields { exponent_size: 10, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 10, bit_offset: 0, bit_precision: 32 })]
#[case::zero_mantissa_size(4, FloatingFields { mantissa_size: 0, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 0, size: 0, bit_offset: 0, bit_precision: 32 })]
#[case::mantissa_past_precision_end(4, FloatingFields { mantissa_location: 250, mantissa_size: 255, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 250, size: 255, bit_offset: 0, bit_precision: 32 })]
#[case::mantissa_below_bit_offset(5, FloatingFields { bit_offset: 1, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 0, size: 23, bit_offset: 1, bit_precision: 32 })]
fn rejects_floating_point_size_range_or_field(
    #[case] size: u32,
    #[case] fields: FloatingFields,
    #[case] expected: FormatError,
) {
    assert_eq!(
        fields
            .parse(size, DatatypeByteOrder::LittleEndian)
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::sign_overlaps_exponent(FloatingFields { sign_location: 23, ..FloatingFields::BINARY32 }, FloatingPointField::Sign, 23, 1, FloatingPointField::Exponent, 23, 8)]
#[case::sign_overlaps_mantissa(FloatingFields { sign_location: 0, ..FloatingFields::BINARY32 }, FloatingPointField::Sign, 0, 1, FloatingPointField::Mantissa, 0, 23)]
#[case::mantissa_overlaps_exponent_start(FloatingFields { mantissa_size: 24, ..FloatingFields::BINARY32 }, FloatingPointField::Exponent, 23, 8, FloatingPointField::Mantissa, 0, 24)]
#[case::mantissa_matches_exponent(FloatingFields { mantissa_location: 23, mantissa_size: 8, ..FloatingFields::BINARY32 }, FloatingPointField::Exponent, 23, 8, FloatingPointField::Mantissa, 23, 8)]
fn rejects_overlapping_floating_point_fields(
    #[case] fields: FloatingFields,
    #[case] first: FloatingPointField,
    #[case] first_location: u8,
    #[case] first_size: u8,
    #[case] second: FloatingPointField,
    #[case] second_location: u8,
    #[case] second_size: u8,
) {
    assert_eq!(
        fields
            .parse(4, DatatypeByteOrder::LittleEndian)
            .unwrap_err(),
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
#[case::binary32(4, FloatingFields::BINARY32)]
#[case::binary64(8, FloatingFields::BINARY64)]
#[case::binary64_in_sixteen_bytes(16, FloatingFields::BINARY64)]
#[case::binary64_in_maximum_size(u32::MAX, FloatingFields::BINARY64)]
#[case::offset_binary32(5, FloatingFields { sign_location: 32, bit_offset: 1, exponent_location: 24, mantissa_location: 1, ..FloatingFields::BINARY32 })]
#[case::reduced_mantissa_and_exponent(4, FloatingFields { mantissa_size: 20, exponent_size: 7, ..FloatingFields::BINARY32 })]
fn accepts_floating_point_fields(
    #[case] size: u32,
    #[case] fields: FloatingFields,
    #[values(
        DatatypeByteOrder::LittleEndian,
        DatatypeByteOrder::BigEndian,
        DatatypeByteOrder::Vax
    )]
    order: DatatypeByteOrder,
) {
    let numeric = fields.parse(size, order).unwrap();
    assert_eq!(numeric.size().get(), size);
    assert_eq!(numeric.byte_order(), order);
    fields.assert_layout(numeric.layout());
}

#[test]
fn rejects_floating_point_layout_with_bit_range_past_element() {
    assert_eq!(
        FloatingPointType::from_layout(
            1,
            DatatypeByteOrder::LittleEndian,
            FloatingPointLayout::IEEE754_BINARY64
        )
        .unwrap_err(),
        FormatError::InvalidFloatingPointRange {
            size: 1,
            bit_offset: 0,
            bit_precision: 64
        }
    );
}

#[derive(Clone, Copy)]
struct FloatingFields {
    mantissa_normalization: MantissaNormalization,
    sign_location: u8,
    bit_offset: u16,
    bit_precision: u16,
    exponent_location: u8,
    exponent_size: u8,
    mantissa_location: u8,
    mantissa_size: u8,
    exponent_bias: u32,
}

impl FloatingFields {
    const fn from_layout(layout: FloatingPointLayout) -> Self {
        let FloatingPointLayout {
            low_padding: _,
            high_padding: _,
            internal_padding: _,
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        } = layout;
        Self {
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        }
    }

    fn parse(self, size: u32, order: DatatypeByteOrder) -> Result<FloatingPointType, FormatError> {
        let Self {
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        } = self;
        FloatingPointType::new(
            size,
            order,
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        )
    }

    fn assert_layout(self, layout: FloatingPointLayout) {
        let Self {
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        } = self;
        assert_eq!(
            (
                layout.mantissa_normalization(),
                layout.sign_location(),
                layout.bit_offset(),
                layout.bit_precision(),
                layout.exponent_location(),
                layout.exponent_size(),
                layout.mantissa_location(),
                layout.mantissa_size(),
                layout.exponent_bias()
            ),
            (
                mantissa_normalization,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias
            )
        );
    }

    const BINARY32: Self = Self::from_layout(FloatingPointLayout::IEEE754_BINARY32);
    const BINARY64: Self = Self::from_layout(FloatingPointLayout::IEEE754_BINARY64);
}
