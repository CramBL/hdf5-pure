use rstest::rstest;

use super::{FIXED_POINT_CLASS, FixedPointType};
use crate::{DatatypeByteOrder, FormatError};

#[rstest]
#[case::zero_size(0, 0, 8, FormatError::ZeroSizedDatatype { class: FIXED_POINT_CLASS })]
#[case::zero_precision(1, 0, 0, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 0, bit_precision: 0 })]
#[case::offset_at_storage_end(1, 8, 1, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 8, bit_precision: 1 })]
#[case::range_past_one_byte(1, 4, 5, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 4, bit_precision: 5 })]
#[case::precision_past_two_bytes(2, 0, 17, FormatError::InvalidFixedPointRange { size: 2, bit_offset: 0, bit_precision: 17 })]
#[case::offset_at_eight_byte_end(8, 64, 1, FormatError::InvalidFixedPointRange { size: 8, bit_offset: 64, bit_precision: 1 })]
#[case::maximum_offset_and_precision(8, u16::MAX, u16::MAX, FormatError::InvalidFixedPointRange { size: 8, bit_offset: u16::MAX, bit_precision: u16::MAX })]
fn rejects_fixed_point_size_or_bit_range(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[case] expected: FormatError,
) {
    assert_eq!(
        FixedPointType::new(
            size,
            DatatypeByteOrder::LittleEndian,
            false,
            offset,
            precision
        )
        .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::one_byte_high_nibble(1, 4, 4)]
#[case::three_byte_offset_range(3, 7, 13)]
#[case::sixteen_byte_offset_range(16, 1, 127)]
#[case::maximum_size_and_fields(u32::MAX, u16::MAX, u16::MAX)]
fn accepts_fixed_point_range(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[values(false, true)] signed: bool,
    #[values(DatatypeByteOrder::LittleEndian, DatatypeByteOrder::BigEndian)]
    order: DatatypeByteOrder,
) {
    let numeric = FixedPointType::new(size, order, signed, offset, precision).unwrap();
    assert_eq!(numeric.size().get(), size);
    assert_eq!(numeric.byte_order(), order);
    assert_eq!(
        (
            numeric.layout().signed(),
            numeric.layout().bit_offset(),
            numeric.layout().bit_precision()
        ),
        (signed, offset, precision)
    );
}

#[rstest]
fn preserves_fixed_point_layout_padding(
    #[values(false, true)] low: bool,
    #[values(false, true)] high: bool,
) {
    let layout = FixedPointType::u16().with_padding(low, high).layout();
    assert_eq!(
        FixedPointType::from_layout(2, DatatypeByteOrder::BigEndian, layout).unwrap(),
        FixedPointType::new(2, DatatypeByteOrder::BigEndian, false, 0, 16)
            .unwrap()
            .with_padding(low, high),
    );
}

#[test]
fn rejects_fixed_point_layout_with_bit_range_past_element() {
    let layout = FixedPointType::u64().layout();
    assert_eq!(
        FixedPointType::from_layout(1, DatatypeByteOrder::LittleEndian, layout).unwrap_err(),
        FormatError::InvalidFixedPointRange {
            size: 1,
            bit_offset: 0,
            bit_precision: 64
        }
    );
}

#[test]
fn rejects_vax_fixed_point_byte_order() {
    assert_eq!(
        FixedPointType::new(1, DatatypeByteOrder::Vax, false, 0, 8).unwrap_err(),
        FormatError::InvalidNumericByteOrder {
            class: FIXED_POINT_CLASS,
            byte_order: DatatypeByteOrder::Vax,
        }
    );
}
