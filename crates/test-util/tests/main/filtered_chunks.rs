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
