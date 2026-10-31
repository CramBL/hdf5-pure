use rstest::rstest;

use hdf5_pure_core::{
    BitFieldType, BitFieldTypeBuilder, DatatypeByteOrder, FixedPointType, FixedPointTypeBuilder,
    FormatError, NumericDatatypeProperty,
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
#[case::size(NumericDatatypeProperty::Size, "size")]
#[case::byte_order(NumericDatatypeProperty::ByteOrder, "byte_order")]
#[case::signed(NumericDatatypeProperty::Signed, "signed")]
#[case::bit_offset(NumericDatatypeProperty::BitOffset, "bit_offset")]
#[case::bit_precision(NumericDatatypeProperty::BitPrecision, "bit_precision")]
fn missing_property_diagnostics_identify_the_property(
    #[case] property: NumericDatatypeProperty,
    #[case] expected: &str,
) {
    assert_eq!(property.to_string(), expected);
    assert_eq!(
        FormatError::MissingNumericDatatypeProperty {
            class: FIXED_POINT_CLASS,
            property
        }
        .to_string(),
        format!("datatype class 0 requires the {expected} property"),
    );
}

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
