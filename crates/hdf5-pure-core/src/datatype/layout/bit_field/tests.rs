use rstest::rstest;

use super::{BIT_FIELD_CLASS, BitFieldType};
use crate::{DatatypeByteOrder, FormatError};

#[rstest]
#[case::zero_size(0, 0, 1)]
#[case::zero_precision(1, 0, 0)]
#[case::range_past_one_byte(1, 4, 5)]
#[case::maximum_offset_and_precision(8, u16::MAX, u16::MAX)]
fn rejects_bit_field_size_or_bit_range(
    #[case] size: u32,
    #[case] bit_offset: u16,
    #[case] bit_precision: u16,
) {
    let expected = if size == 0 {
        FormatError::ZeroSizedDatatype {
            class: BIT_FIELD_CLASS,
        }
    } else {
        FormatError::InvalidNumericBitRange {
            class: BIT_FIELD_CLASS,
            size,
            bit_offset,
            bit_precision,
        }
    };
    assert_eq!(
        BitFieldType::new(
            size,
            DatatypeByteOrder::LittleEndian,
            bit_offset,
            bit_precision
        )
        .unwrap_err(),
        expected
    );
}

#[test]
fn rejects_vax_bit_field_byte_order() {
    assert_eq!(
        BitFieldType::new(1, DatatypeByteOrder::Vax, 0, 8).unwrap_err(),
        FormatError::InvalidNumericByteOrder {
            class: BIT_FIELD_CLASS,
            byte_order: DatatypeByteOrder::Vax,
        }
    );
}
