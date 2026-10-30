use rstest::rstest;
use test_util::filtered_chunks;

#[rstest]
#[case::empty(&[], &[])]
#[case::partial_element(&[1, 2], &[1, 2])]
#[case::complete_elements(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 5, 2, 6, 3, 7, 4, 8])]
#[case::trailing_bytes(&[1, 2, 3, 4, 5, 6, 7, 8, 99, 98], &[1, 5, 2, 6, 3, 7, 4, 8, 99, 98])]
fn shuffle_control_preserves_elements_and_trailing_bytes(
    #[case] raw: &[u8],
    #[case] expected: &[u8],
) {
    assert_eq!(filtered_chunks::shuffled_pipeline_data(raw), expected);
}

#[test]
fn pipeline_values_have_four_little_endian_bytes_per_element() {
    assert_eq!(
        filtered_chunks::pipeline_data(3),
        [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
    );
}

#[rstest]
#[case::cycling(filtered_chunks::PipelinePattern::Cycling, &[0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0])]
#[case::incompressible(filtered_chunks::PipelinePattern::Incompressible, &[0xa5, 0x5a, 0x98, 0x87, 0xa3, 0x24, 0x5b, 0x15, 0xc4, 0xf4, 0x20, 0x48])]
fn pipeline_patterns_produce_independent_expected_words(
    #[case] pattern: filtered_chunks::PipelinePattern,
    #[case] expected: &[u8],
) {
    assert_eq!(pattern.data(3), expected);
}
