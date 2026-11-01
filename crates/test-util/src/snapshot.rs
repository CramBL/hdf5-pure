pub(crate) fn grouped_decimal(value: u64) -> String {
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
