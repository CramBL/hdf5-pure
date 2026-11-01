//! Numeric formatting for snapshot reports.

/// Formats an unsigned integer with comma-separated decimal groups.
pub(crate) fn grouped_decimal(value: u64) -> String {
    grouped_unsigned(u128::from(value))
}

/// Formats a signed integer with comma-separated decimal groups.
///
/// Negative values include a leading minus sign.
pub(crate) fn grouped_signed(value: i128) -> String {
    let grouped = grouped_unsigned(value.unsigned_abs());
    if value.is_negative() {
        format!("-{grouped}")
    } else {
        grouped
    }
}

/// Formats an unsigned integer as lowercase hexadecimal with a `0x` prefix.
///
/// `digits` is the minimum digit count, excluding the prefix. Shorter values are padded with zeros.
pub(crate) fn hexadecimal(value: u64, digits: usize) -> String {
    format!("0x{value:0digits$x}")
}

fn grouped_unsigned(value: u128) -> String {
    let digits = value.to_string();
    let mut grouped = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    #[rstest]
    #[case::zero(0, "0")]
    #[case::negative(-1_234_567, "-1,234,567")]
    #[case::positive(1_234_567, "1,234,567")]
    #[case::minimum(i128::MIN, "-170,141,183,460,469,231,731,687,303,715,884,105,728")]
    #[case::maximum(i128::MAX, "170,141,183,460,469,231,731,687,303,715,884,105,727")]
    fn signed_decimals_preserve_sign_and_group_magnitude(
        #[case] value: i128,
        #[case] expected: &str,
    ) {
        assert_eq!(super::grouped_signed(value), expected);
    }

    #[rstest]
    #[case::zero(0, 8, "0x00000000")]
    #[case::unpadded(0, 0, "0x0")]
    #[case::padded(16, 8, "0x00000010")]
    #[case::wider_than_minimum(0x1_0000, 4, "0x10000")]
    #[case::maximum(u64::MAX, 8, "0xffffffffffffffff")]
    fn hexadecimal_preserves_value_and_minimum_digits(
        #[case] value: u64,
        #[case] digits: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(super::hexadecimal(value, digits), expected);
    }

    #[rstest]
    #[case::zero(0, "0")]
    #[case::below_thousand(999, "999")]
    #[case::thousand(1_000, "1,000")]
    #[case::below_million(999_999, "999,999")]
    #[case::million(1_000_000, "1,000,000")]
    #[case::readable_example(1_300_233, "1,300,233")]
    #[case::maximum(u64::MAX, "18,446,744,073,709,551,615")]
    fn decimal_numbers_have_thousands_separators(#[case] value: u64, #[case] expected: &str) {
        assert_eq!(super::grouped_decimal(value), expected);
    }
}
