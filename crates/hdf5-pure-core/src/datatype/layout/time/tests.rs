use rstest::rstest;

use super::{TIME_CLASS, TimeType};
use crate::{DatatypeByteOrder, FormatError};

#[rstest]
#[case::zero_size(0, 1)]
#[case::zero_precision(1, 0)]
#[case::precision_past_one_byte(1, 9)]
fn rejects_time_size_or_precision(#[case] size: u32, #[case] bit_precision: u16) {
    let expected = if size == 0 {
        FormatError::ZeroSizedDatatype { class: TIME_CLASS }
    } else {
        FormatError::InvalidNumericBitRange {
            class: TIME_CLASS,
            size,
            bit_offset: 0,
            bit_precision,
        }
    };
    assert_eq!(
        TimeType::new(size, DatatypeByteOrder::LittleEndian, bit_precision).unwrap_err(),
        expected
    );
}

#[test]
fn rejects_vax_time_byte_order() {
    assert_eq!(
        TimeType::new(1, DatatypeByteOrder::Vax, 8).unwrap_err(),
        FormatError::InvalidNumericByteOrder {
            class: TIME_CLASS,
            byte_order: DatatypeByteOrder::Vax,
        }
    );
}
