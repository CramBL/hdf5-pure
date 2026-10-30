use rstest::rstest;
use test_util::filtered_chunks::{self, Direction, PipelineCase, PipelineFilter};

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
#[case::deflate(PipelineCase::Deflate, "deflate", &[PipelineFilter::Deflate], None)]
#[case::deflate_checksum(PipelineCase::DeflateFletcher32, "deflate_fletcher32", &[PipelineFilter::Deflate, PipelineFilter::Fletcher32], None)]
#[case::checksum(PipelineCase::Fletcher32, "fletcher32", &[PipelineFilter::Fletcher32], None)]
#[case::lzf(PipelineCase::Lzf, "lzf", &[PipelineFilter::Lzf], None)]
#[case::shuffle(PipelineCase::Shuffle, "shuffle", &[PipelineFilter::Shuffle], None)]
#[case::shuffle_deflate(PipelineCase::ShuffleDeflate, "shuffle_deflate", &[PipelineFilter::Shuffle, PipelineFilter::Deflate], Some(PipelineCase::Deflate))]
#[case::shuffle_deflate_checksum(PipelineCase::ShuffleDeflateFletcher32, "shuffle_deflate_fletcher32", &[PipelineFilter::Shuffle, PipelineFilter::Deflate, PipelineFilter::Fletcher32], Some(PipelineCase::DeflateFletcher32))]
#[case::shuffle_checksum(PipelineCase::ShuffleFletcher32, "shuffle_fletcher32", &[PipelineFilter::Shuffle, PipelineFilter::Fletcher32], None)]
#[case::shuffle_lzf(PipelineCase::ShuffleLzf, "shuffle_lzf", &[PipelineFilter::Shuffle, PipelineFilter::Lzf], Some(PipelineCase::Lzf))]
fn pipeline_cases_describe_filters_and_compressor_controls(
    #[case] case: PipelineCase,
    #[case] label: &str,
    #[case] filters: &[PipelineFilter],
    #[case] control: Option<PipelineCase>,
) {
    assert_eq!(case.name(), label);
    assert_eq!(case.filters(), filters);
    assert_eq!(case.codec_control(), control);
}

#[rstest]
#[case::decode(Direction::Decode, "decode")]
#[case::encode(Direction::Encode, "encode")]
fn directions_have_measurement_labels(#[case] direction: Direction, #[case] label: &str) {
    assert_eq!(direction.name(), label);
}

#[test]
fn measured_pipelines_retain_the_complete_workload_selection() {
    assert_eq!(
        filtered_chunks::PIPELINES,
        [
            PipelineCase::Shuffle,
            PipelineCase::Fletcher32,
            PipelineCase::ShuffleFletcher32,
            PipelineCase::ShuffleLzf,
            PipelineCase::ShuffleDeflate,
            PipelineCase::ShuffleDeflateFletcher32,
        ]
    );
}
