use hdf5_pure_format::__private::{FilterDescription, FilterPipeline, FilterPipelineError};
use hdf5_pure_format::FormatError;
use rstest::rstest;
use test_util::filter_pipeline::{Pipeline, Version};

#[rstest]
#[case::version_one(Version::One)]
#[case::version_two(Version::Two)]
fn decoded_filters_preserve_names_flags_values_and_order(#[case] version: Version) {
    let mut fixture = Pipeline::new(version);
    fixture
        .push(
            255,
            (version == Version::One).then_some("predefined"),
            0,
            &[7],
        )
        .unwrap();
    fixture.push(256, Some(""), 0x03, &[u32::MAX, 14]).unwrap();
    fixture
        .push(40_500, Some("custom"), 0xfe, &[0, 1, 0x1234_5678])
        .unwrap();
    fixture.push(u16::MAX, None, 1, &[]).unwrap();

    assert_eq!(
        FilterPipeline::parse(&fixture.build()).unwrap(),
        expected_pipeline(&fixture)
    );
}

#[rstest]
#[case::version_one(Version::One)]
#[case::version_two(Version::Two)]
fn parser_ignores_bytes_after_an_empty_pipeline(#[case] version: Version) {
    let mut bytes = Pipeline::new(version).build();
    bytes.extend_from_slice(&[0xa5; 3]);
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap(),
        FilterPipeline {
            version: version.byte(),
            filters: Vec::new(),
        }
    );
}

#[rstest]
#[case::version_one(Version::One)]
#[case::version_two(Version::Two)]
fn parser_accepts_thirty_two_filters(#[case] version: Version) {
    let mut fixture = Pipeline::new(version);
    for id in 1..=32 {
        fixture.push(id, None, 0, &[]).unwrap();
    }
    assert_eq!(
        FilterPipeline::parse(&fixture.build()).unwrap(),
        expected_pipeline(&fixture)
    );
}

#[rstest]
#[case::aligned_name(Version::One, 65_527)]
#[case::unpadded_name(Version::Two, 65_534)]
fn parser_accepts_the_largest_name_and_client_data_fields(
    #[case] version: Version,
    #[case] name_len: usize,
) {
    let name = "n".repeat(name_len);
    let mut client_data = vec![0x1234_5678; 65_535];
    client_data[0] = 0;
    client_data[65_534] = u32::MAX;
    let mut fixture = Pipeline::new(version);
    fixture
        .push(FILTER_ID, Some(&name), 0, &client_data)
        .unwrap();

    assert_eq!(
        FilterPipeline::parse(&fixture.build()).unwrap(),
        expected_pipeline(&fixture)
    );
}

#[rstest]
#[case::missing_prefix(&[], 2)]
#[case::version_without_count(&[3], 2)]
#[case::version_one_reserved_bytes(&[1, 33, 1], 8)]
#[case::version_one_filter_header(&[1, 1, 0, 0, 0, 0, 0, 0, 44, 1, 8, 0, 0, 1, 1], 16)]
#[case::version_two_filter_id(&[2, 1, 44], 4)]
#[case::version_two_predefined_fields(&[2, 1, 1, 0, 0, 1, 1], 8)]
#[case::version_two_custom_fields(&[2, 1, 44, 1, 3, 0, 0, 1, 1], 10)]
fn parser_reports_the_required_end_of_an_incomplete_pipeline_or_filter_header(
    #[case] bytes: &[u8],
    #[case] expected: usize,
) {
    assert_eq!(
        FilterPipeline::parse(bytes).unwrap_err(),
        FilterPipelineError::Format(FormatError::UnexpectedEof {
            expected,
            available: bytes.len(),
        })
    );
}

#[test]
fn an_unknown_version_precedes_an_out_of_range_filter_count() {
    assert_eq!(
        FilterPipeline::parse(&[3, 33]).unwrap_err(),
        FilterPipelineError::Format(FormatError::InvalidFilterPipelineVersion(3))
    );
}

#[rstest]
#[case::version_one(&[1, 33, 1, 0, 0, 0, 0, 0])]
#[case::version_two(&[2, 33])]
fn parser_rejects_an_out_of_range_filter_count_after_header_acquisition(#[case] bytes: &[u8]) {
    assert_eq!(
        FilterPipeline::parse(bytes).unwrap_err(),
        FilterPipelineError::FieldTooLarge {
            field: "filter count",
            value: 33,
            maximum: 32,
        }
    );
}

#[test]
fn parser_rejects_nonzero_reserved_header_bytes_in_version_one() {
    let mut bytes = Pipeline::new(Version::One).build();
    bytes[3] = 0x80;
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::Format(FormatError::InvalidFilterPipelineReservedBytes([
            0, 0x80, 0, 0, 0, 0,
        ]))
    );
}

#[rstest]
#[case::version_one(Version::One, 13)]
#[case::version_two(Version::Two, 7)]
fn invalid_filter_flags_precede_a_truncated_filter_body(
    #[case] version: Version,
    #[case] high_flags_at: usize,
) {
    let mut bytes = one_filter(version).build();
    bytes[high_flags_at] = 0x80;
    bytes.truncate(high_flags_at + 3);
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::Format(FormatError::InvalidFilterFlags {
            filter_id: FILTER_ID,
            flags: 0x80fe,
        })
    );
}

#[rstest]
#[case::version_one_name(Version::One, 16, 23, 32)]
#[case::version_one_client_values(Version::One, 16, 27, 32)]
#[case::version_one_padding(Version::One, 16, 28, 32)]
#[case::version_two_name(Version::Two, 10, 12, 17)]
#[case::version_two_client_values(Version::Two, 10, 16, 17)]
fn a_truncated_filter_body_precedes_invalid_name_bytes(
    #[case] version: Version,
    #[case] name_at: usize,
    #[case] available: usize,
    #[case] expected: usize,
) {
    let mut bytes = one_filter(version).build();
    bytes[name_at] = 0xff;
    bytes.truncate(available);
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::Format(FormatError::UnexpectedEof {
            expected,
            available
        })
    );
}

#[rstest]
#[case::valid_flags(0x00fe)]
#[case::invalid_flags(0x80fe)]
fn invalid_name_length_precedes_invalid_flags_and_body_truncation_in_version_one(
    #[case] flags: u16,
) {
    let mut bytes = one_filter(Version::One).build();
    test_util::bytes::set_u16_at(&mut bytes, V1_NAME_LENGTH_AT, 7);
    test_util::bytes::set_u16_at(&mut bytes, V1_FLAGS_AT, flags);
    bytes.truncate(V1_FILTER_HEADER_END);
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::InvalidName {
            filter_id: FILTER_ID,
            reason: "name length is not a multiple of eight",
        }
    );
}

#[test]
fn a_truncated_filter_header_precedes_an_invalid_name_length_in_version_one() {
    let mut bytes = one_filter(Version::One).build();
    test_util::bytes::set_u16_at(&mut bytes, V1_NAME_LENGTH_AT, 7);
    let available = V1_FILTER_HEADER_END - 1;
    bytes.truncate(available);
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::Format(FormatError::UnexpectedEof {
            expected: V1_FILTER_HEADER_END,
            available,
        })
    );
}

#[test]
fn parser_rejects_nonzero_client_data_padding_in_version_one() {
    let mut bytes = one_filter(Version::One).build();
    bytes[30] = 0x42;
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::Format(FormatError::InvalidFilterClientDataPadding {
            filter_id: FILTER_ID,
            padding: [0, 0, 0x42, 0],
        })
    );
}

#[rstest]
#[case::invalid_utf8_v1(Version::One, 16, [0xff, b'b', 0], "invalid UTF-8")]
#[case::invalid_utf8_v2(Version::Two, 10, [0xff, b'b', 0], "invalid UTF-8")]
#[case::non_ascii(Version::Two, 10, [0xc2, 0xa3, 0], "non-ASCII name")]
#[case::missing_terminator(Version::Two, 10, *b"abc", "missing null terminator")]
#[case::nonzero_v1_padding(Version::One, 16, [b'a', 0, b'b'], "nonzero byte after null terminator")]
#[case::nonfinal_v2_terminator(Version::Two, 10, [b'a', 0, 0], "bytes after null terminator")]
fn parser_reports_an_invalid_name_in_a_complete_filter_body(
    #[case] version: Version,
    #[case] name_at: usize,
    #[case] name: [u8; 3],
    #[case] reason: &'static str,
) {
    let mut bytes = one_filter(version).build();
    bytes[name_at..name_at + name.len()].copy_from_slice(&name);
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::InvalidName {
            filter_id: FILTER_ID,
            reason
        }
    );
}

#[test]
fn parser_rejects_an_unaligned_name_length_in_version_one() {
    let mut bytes = one_filter(Version::One).build();
    bytes[10] = 7;
    bytes.remove(23);
    assert_eq!(
        FilterPipeline::parse(&bytes).unwrap_err(),
        FilterPipelineError::InvalidName {
            filter_id: FILTER_ID,
            reason: "name length is not a multiple of eight",
        }
    );
}

fn one_filter(version: Version) -> Pipeline<'static> {
    let mut fixture = Pipeline::new(version);
    fixture
        .push(FILTER_ID, Some("ab"), 0xfe, &[0x1234_5678])
        .unwrap();
    fixture
}

fn expected_pipeline(fixture: &Pipeline<'_>) -> FilterPipeline {
    FilterPipeline {
        version: fixture.version().byte(),
        filters: fixture
            .filters()
            .iter()
            .map(|filter| FilterDescription {
                filter_id: filter.id(),
                name: filter.name().map(str::to_owned),
                flags: u16::from(filter.flags()),
                client_data: filter.client_data().to_vec(),
            })
            .collect(),
    }
}

const FILTER_ID: u16 = 300;
const V1_NAME_LENGTH_AT: usize = 10;
const V1_FLAGS_AT: usize = 12;
const V1_FILTER_HEADER_END: usize = 16;
