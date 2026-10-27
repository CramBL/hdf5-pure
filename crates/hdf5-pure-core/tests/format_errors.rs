use hdf5_pure_core::FormatError;

#[rstest::rstest]
#[case::decoded_size_limit(
    FormatError::DecodedSizeLimitExceeded {
        expected_decoded_size: 4_294_967_600,
        filter_id: 32_001,
        output_limit: 4_294_967_604,
        minimum_output_size: 4_294_967_605,
    },
    "filter 32001 output needs at least 4294967605 bytes, above limit 4294967604 for a payload expected to decode to 4294967600 bytes"
)]
#[case::resource_limit(
    FormatError::ResourceLimit { limit: 4_294_967_600, required: 4_294_967_616 },
    "resource requires 4294967616, above limit 4294967600"
)]
#[case::decoded_size(
    FormatError::DecodedSizeMismatch { expected: 4_294_967_600, actual: 4_294_967_616 },
    "decoded payload has 4294967616 bytes, expected 4294967600"
)]
#[case::wrong_layout(FormatError::WrongLayout, "reference belongs to another layout")]
#[case::reserved_heap_bits(
    FormatError::InvalidFractalHeapIdFlags(0x01),
    "reserved bits in fractal heap ID: 0x01"
)]
#[case::heap_id_version(
    FormatError::UnsupportedFractalHeapIdVersion(1),
    "unsupported fractal heap ID version: 1"
)]
fn huge_object_diagnostics_preserve_their_fields(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}
