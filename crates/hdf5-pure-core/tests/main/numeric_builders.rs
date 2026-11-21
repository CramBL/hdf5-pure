use rstest::rstest;

use hdf5_pure_core::{
    BitFieldType, BitFieldTypeBuilder, DatatypeByteOrder, FixedPointType, FixedPointTypeBuilder,
    FloatingPointField, FloatingPointType, FloatingPointTypeBuilder, FormatError,
    MantissaNormalization, NumericDatatypeProperty,
};

#[rstest]
#[case::signed_offset_range(4, DatatypeByteOrder::LittleEndian, true, 3, 21, false, false)]
#[case::high_nibble(1, DatatypeByteOrder::BigEndian, false, 4, 4, true, false)]
fn fixed_point_builder_preserves_named_properties(
    #[case] size: u32,
    #[case] byte_order: DatatypeByteOrder,
    #[case] signed: bool,
    #[case] bit_offset: u16,
    #[case] bit_precision: u16,
    #[case] low_padding: bool,
    #[case] high_padding: bool,
) {
    let datatype = FixedPointType::builder()
        .size(size)
        .byte_order(byte_order)
        .signed(signed)
        .bit_offset(bit_offset)
        .bit_precision(bit_precision)
        .low_padding(low_padding)
        .high_padding(high_padding)
        .build()
        .unwrap();
    assert_eq!(
        (
            datatype.size().get(),
            datatype.byte_order(),
            datatype.layout().signed(),
            datatype.layout().bit_offset(),
            datatype.layout().bit_precision(),
            datatype.layout().low_padding(),
            datatype.layout().high_padding(),
        ),
        (
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
            low_padding,
            high_padding
        ),
    );
}

#[rstest]
#[case::signed(FixedPointType::builder().size(1).byte_order(DatatypeByteOrder::LittleEndian).bit_offset(0).bit_precision(8), NumericDatatypeProperty::Signed)]
#[case::bit_precision(FixedPointType::builder().size(1).byte_order(DatatypeByteOrder::LittleEndian).signed(false).bit_offset(0), NumericDatatypeProperty::BitPrecision)]
fn fixed_point_builder_rejects_missing_properties(
    #[case] builder: FixedPointTypeBuilder,
    #[case] property: NumericDatatypeProperty,
) {
    assert_eq!(
        builder.build().unwrap_err(),
        FormatError::MissingNumericDatatypeProperty {
            class: FIXED_POINT_CLASS,
            property
        },
    );
}

#[rstest]
#[case::vax(1, DatatypeByteOrder::Vax, 0, 8, FormatError::InvalidNumericByteOrder { class: FIXED_POINT_CLASS, byte_order: DatatypeByteOrder::Vax })]
#[case::range_past_element(1, DatatypeByteOrder::BigEndian, 4, 5, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 4, bit_precision: 5 })]
fn fixed_point_builder_rejects_invalid_layouts(
    #[case] size: u32,
    #[case] byte_order: DatatypeByteOrder,
    #[case] bit_offset: u16,
    #[case] bit_precision: u16,
    #[case] expected: FormatError,
) {
    assert_eq!(
        FixedPointType::builder()
            .size(size)
            .byte_order(byte_order)
            .signed(true)
            .bit_offset(bit_offset)
            .bit_precision(bit_precision)
            .build()
            .unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::offset_range(4, DatatypeByteOrder::LittleEndian, 3, 21, false, false)]
#[case::high_nibble(1, DatatypeByteOrder::BigEndian, 4, 4, true, false)]
fn bit_field_builder_preserves_named_properties(
    #[case] size: u32,
    #[case] byte_order: DatatypeByteOrder,
    #[case] bit_offset: u16,
    #[case] bit_precision: u16,
    #[case] low_padding: bool,
    #[case] high_padding: bool,
) {
    let datatype = BitFieldType::builder()
        .size(size)
        .byte_order(byte_order)
        .bit_offset(bit_offset)
        .bit_precision(bit_precision)
        .low_padding(low_padding)
        .high_padding(high_padding)
        .build()
        .unwrap();
    assert_eq!(
        (
            datatype.size().get(),
            datatype.byte_order(),
            datatype.bit_offset(),
            datatype.bit_precision().get(),
            datatype.low_padding(),
            datatype.high_padding(),
        ),
        (
            size,
            byte_order,
            bit_offset,
            bit_precision,
            low_padding,
            high_padding
        ),
    );
}

#[rstest]
#[case::byte_order(BitFieldType::builder().size(1).bit_offset(0).bit_precision(8), NumericDatatypeProperty::ByteOrder)]
#[case::bit_precision(BitFieldType::builder().size(1).byte_order(DatatypeByteOrder::LittleEndian).bit_offset(0), NumericDatatypeProperty::BitPrecision)]
fn bit_field_builder_rejects_missing_properties(
    #[case] builder: BitFieldTypeBuilder,
    #[case] property: NumericDatatypeProperty,
) {
    assert_eq!(
        builder.build().unwrap_err(),
        FormatError::MissingNumericDatatypeProperty {
            class: BIT_FIELD_CLASS,
            property
        },
    );
}

#[rstest]
#[case::vax(1, DatatypeByteOrder::Vax, 0, 8, FormatError::InvalidNumericByteOrder { class: BIT_FIELD_CLASS, byte_order: DatatypeByteOrder::Vax })]
#[case::range_past_element(1, DatatypeByteOrder::BigEndian, 4, 5, FormatError::InvalidNumericBitRange { class: BIT_FIELD_CLASS, size: 1, bit_offset: 4, bit_precision: 5 })]
fn bit_field_builder_rejects_invalid_layouts(
    #[case] size: u32,
    #[case] byte_order: DatatypeByteOrder,
    #[case] bit_offset: u16,
    #[case] bit_precision: u16,
    #[case] expected: FormatError,
) {
    assert_eq!(
        BitFieldType::builder()
            .size(size)
            .byte_order(byte_order)
            .bit_offset(bit_offset)
            .bit_precision(bit_precision)
            .build()
            .unwrap_err(),
        expected,
    );
}

#[test]
fn builders_default_to_zero_padding_and_allow_replacing_properties() {
    const INTEGER: FixedPointTypeBuilder = FixedPointTypeBuilder::new()
        .size(0)
        .size(4)
        .byte_order(DatatypeByteOrder::Vax)
        .byte_order(DatatypeByteOrder::LittleEndian)
        .signed(true)
        .signed(false)
        .bit_offset(8)
        .bit_offset(0)
        .bit_precision(0)
        .bit_precision(32)
        .low_padding(true)
        .low_padding(false)
        .high_padding(true)
        .high_padding(false);
    const BITS: BitFieldTypeBuilder = BitFieldTypeBuilder::new()
        .size(0)
        .size(2)
        .byte_order(DatatypeByteOrder::Vax)
        .byte_order(DatatypeByteOrder::BigEndian)
        .bit_offset(8)
        .bit_offset(0)
        .bit_precision(0)
        .bit_precision(16);
    assert_eq!(INTEGER.build().unwrap(), FixedPointType::u32());
    assert_eq!(
        BITS.build().unwrap(),
        BitFieldType::new(2, DatatypeByteOrder::BigEndian, 0, 16).unwrap(),
    );
    assert_eq!(FixedPointTypeBuilder::default(), FixedPointType::builder());
    assert_eq!(BitFieldTypeBuilder::default(), BitFieldType::builder());
}

#[rstest]
#[case::offset_and_internal_padding(
    4,
    DatatypeByteOrder::BigEndian,
    MantissaNormalization::LeadingOne,
    25,
    3,
    23,
    17,
    5,
    3,
    11,
    15,
    true,
    true,
    true
)]
#[case::vax(
    4,
    DatatypeByteOrder::Vax,
    MantissaNormalization::Unnormalized,
    31,
    0,
    32,
    23,
    8,
    0,
    23,
    127,
    true,
    false,
    true
)]
fn floating_point_builder_preserves_named_properties(
    #[case] size: u32,
    #[case] byte_order: DatatypeByteOrder,
    #[case] mantissa_normalization: MantissaNormalization,
    #[case] sign_location: u8,
    #[case] bit_offset: u16,
    #[case] bit_precision: u16,
    #[case] exponent_location: u8,
    #[case] exponent_size: u8,
    #[case] mantissa_location: u8,
    #[case] mantissa_size: u8,
    #[case] exponent_bias: u32,
    #[case] low_padding: bool,
    #[case] high_padding: bool,
    #[case] internal_padding: bool,
) {
    let datatype = FloatingPointType::builder()
        .size(size)
        .byte_order(byte_order)
        .mantissa_normalization(mantissa_normalization)
        .sign_location(sign_location)
        .bit_offset(bit_offset)
        .bit_precision(bit_precision)
        .exponent_location(exponent_location)
        .exponent_size(exponent_size)
        .mantissa_location(mantissa_location)
        .mantissa_size(mantissa_size)
        .exponent_bias(exponent_bias)
        .low_padding(low_padding)
        .high_padding(high_padding)
        .internal_padding(internal_padding)
        .build()
        .unwrap();
    assert_eq!(datatype.size().get(), size);
    assert_eq!(datatype.byte_order(), byte_order);
    assert_eq!(
        (
            datatype.layout().mantissa_normalization(),
            datatype.layout().sign_location(),
            datatype.layout().bit_offset(),
            datatype.layout().bit_precision(),
            datatype.layout().exponent_location(),
            datatype.layout().exponent_size(),
            datatype.layout().mantissa_location(),
            datatype.layout().mantissa_size(),
            datatype.layout().exponent_bias(),
            datatype.layout().low_padding(),
            datatype.layout().high_padding(),
            datatype.layout().internal_padding(),
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
            exponent_bias,
            low_padding,
            high_padding,
            internal_padding,
        ),
    );
}

#[rstest]
#[case::mantissa_normalization(FloatingPointType::builder().size(4).byte_order(DatatypeByteOrder::LittleEndian).sign_location(31).bit_offset(0).bit_precision(32).exponent_location(23).exponent_size(8).mantissa_location(0).mantissa_size(23).exponent_bias(127), NumericDatatypeProperty::MantissaNormalization)]
#[case::exponent_bias(FloatingPointType::builder().size(4).byte_order(DatatypeByteOrder::LittleEndian).mantissa_normalization(MantissaNormalization::ImpliedLeadingOne).sign_location(31).bit_offset(0).bit_precision(32).exponent_location(23).exponent_size(8).mantissa_location(0).mantissa_size(23), NumericDatatypeProperty::ExponentBias)]
fn floating_point_builder_rejects_missing_properties(
    #[case] builder: FloatingPointTypeBuilder,
    #[case] property: NumericDatatypeProperty,
) {
    assert_eq!(
        builder.build().unwrap_err(),
        FormatError::MissingNumericDatatypeProperty {
            class: FLOATING_POINT_CLASS,
            property,
        },
    );
}

#[rstest]
#[case::range_past_element(binary32_builder().bit_offset(1), FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 1, bit_precision: 32 })]
#[case::zero_exponent_size(binary32_builder().exponent_size(0), FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 0, bit_offset: 0, bit_precision: 32 })]
#[case::sign_overlaps_exponent(binary32_builder().sign_location(23), FormatError::OverlappingFloatingPointFields { first: FloatingPointField::Sign, second: FloatingPointField::Exponent, first_location: 23, first_size: 1, second_location: 23, second_size: 8 })]
fn floating_point_builder_rejects_invalid_layouts(
    #[case] builder: FloatingPointTypeBuilder,
    #[case] expected: FormatError,
) {
    assert_eq!(builder.build().unwrap_err(), expected);
}

#[test]
fn floating_point_builder_defaults_to_zero_padding_and_allows_replacing_properties() {
    const BUILDER: FloatingPointTypeBuilder = FloatingPointTypeBuilder::new()
        .size(0)
        .size(4)
        .byte_order(DatatypeByteOrder::Vax)
        .byte_order(DatatypeByteOrder::BigEndian)
        .mantissa_normalization(MantissaNormalization::LeadingOne)
        .mantissa_normalization(MantissaNormalization::ImpliedLeadingOne)
        .sign_location(0)
        .sign_location(31)
        .bit_offset(8)
        .bit_offset(0)
        .bit_precision(0)
        .bit_precision(32)
        .exponent_location(0)
        .exponent_location(23)
        .exponent_size(0)
        .exponent_size(8)
        .mantissa_location(1)
        .mantissa_location(0)
        .mantissa_size(0)
        .mantissa_size(23)
        .exponent_bias(0)
        .exponent_bias(127)
        .low_padding(true)
        .low_padding(false)
        .high_padding(true)
        .high_padding(false)
        .internal_padding(true)
        .internal_padding(false);
    assert_eq!(
        BUILDER.build().unwrap(),
        FloatingPointType::ieee754_binary32(DatatypeByteOrder::BigEndian),
    );
    assert_eq!(
        binary32_builder().build().unwrap(),
        FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian),
    );
    assert_eq!(
        FloatingPointTypeBuilder::default(),
        FloatingPointType::builder(),
    );
}

const fn binary32_builder() -> FloatingPointTypeBuilder {
    FloatingPointType::builder()
        .size(4)
        .byte_order(DatatypeByteOrder::LittleEndian)
        .mantissa_normalization(MantissaNormalization::ImpliedLeadingOne)
        .sign_location(31)
        .bit_offset(0)
        .bit_precision(32)
        .exponent_location(23)
        .exponent_size(8)
        .mantissa_location(0)
        .mantissa_size(23)
        .exponent_bias(127)
}

/// The floating-point class code in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOATING_POINT_CLASS: u8 = 1;

/// The fixed-point class code in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FIXED_POINT_CLASS: u8 = 0;
/// The bit-field class code in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const BIT_FIELD_CLASS: u8 = 4;
