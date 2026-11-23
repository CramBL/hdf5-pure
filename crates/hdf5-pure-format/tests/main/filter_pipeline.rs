use hdf5_pure_format::__private::MessageFlags;
use hdf5_pure_format::__private::{
    FilterDescription, FilterPipeline, FilterPipelineError, FilterPipelineOutputTooSmall,
    MessageRecordLayout, PreparedFilterPipeline,
};
use hdf5_pure_format::FormatError;
use rstest::rstest;
use test_util::filter_pipeline::{Pipeline, Version};
use test_util::object_header::MessageType as FixtureMessageType;

#[rstest]
#[case::version_one(Version::One)]
#[case::version_two(Version::Two)]
fn decoded_filters_preserve_names_flags_values_and_order(#[case] version: Version) {
    let fixture = representative_pipeline(version);
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
        FilterPipeline::new(version.byte(), Vec::new()).unwrap()
    );
}

#[rstest]
#[case::version_one(Version::One)]
#[case::version_two(Version::Two)]
fn parser_and_writer_accept_thirty_two_filters(#[case] version: Version) {
    let mut fixture = Pipeline::new(version);
    for id in 1..=32 {
        fixture.push(id, None, 0, &[]).unwrap();
    }
    let bytes = fixture.build();
    let expected = expected_pipeline(&fixture);
    assert_eq!(FilterPipeline::parse(&bytes).unwrap(), expected);
    assert_eq!(expected.to_bytes(), bytes);
}

#[rstest]
#[case::aligned_name(Version::One, 65_527)]
#[case::unpadded_name(Version::Two, 65_534)]
fn parser_and_writer_accept_the_largest_name_and_client_data_fields(
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

    let bytes = fixture.build();
    let expected = expected_pipeline(&fixture);
    assert_eq!(FilterPipeline::parse(&bytes).unwrap(), expected);
    assert_eq!(expected.to_bytes(), bytes);
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

#[rstest]
#[case::version_one(Version::One)]
#[case::version_two(Version::Two)]
fn writer_encodes_fixture_bytes_from_borrowed_owned_and_parsed_pipelines(#[case] version: Version) {
    let fixture = representative_pipeline(version);
    let expected = fixture.build();
    let filters = descriptions(&fixture);
    let borrowed = PreparedFilterPipeline::new(version.byte(), &filters).unwrap();
    let owned = FilterPipeline::new(version.byte(), filters.clone()).unwrap();
    let decoded = FilterPipeline::parse(&expected).unwrap();

    assert_eq!(borrowed.encoded_size(), expected.len());
    assert_eq!(borrowed.to_bytes(), expected);
    assert_eq!(owned.to_bytes(), expected);
    assert_eq!(decoded.to_bytes(), expected);
    assert_eq!(owned, decoded);
}

#[rstest]
#[case::version_one(Version::One)]
#[case::version_two(Version::Two)]
fn writer_preserves_the_destination_on_capacity_error_and_guards_on_retry(
    #[case] version: Version,
) {
    let fixture = representative_pipeline(version);
    let expected = fixture.build();
    let filters = descriptions(&fixture);
    let prepared = PreparedFilterPipeline::new(version.byte(), &filters).unwrap();
    let required = prepared.encoded_size();
    assert_eq!(required, expected.len());
    let mut output = vec![0xa5; required + 3];
    for available in [0, required - 1] {
        assert_eq!(
            prepared
                .write_into(&mut output[1..1 + available])
                .unwrap_err(),
            FilterPipelineOutputTooSmall {
                required,
                available
            }
        );
        assert_eq!(output, vec![0xa5; required + 3]);
    }
    assert_eq!(prepared.write_into(&mut output[1..]).unwrap(), required);
    let mut guarded = vec![0xa5];
    guarded.extend_from_slice(&expected);
    guarded.extend_from_slice(&[0xa5; 2]);
    assert_eq!(output, guarded);
}

#[rstest]
#[case::unknown_version(
    9, vec![description(300, None, 0x8000, 0); 33],
    FilterPipelineError::Format(FormatError::InvalidFilterPipelineVersion(9))
)]
#[case::too_many_filters(
    Version::Two.byte(), vec![description(300, None, 0x8000, 0); 33],
    FilterPipelineError::FieldTooLarge { field: "filter count", value: 33, maximum: 32 }
)]
#[case::reserved_flags_before_invalid_name(
    Version::One.byte(), vec![description(300, Some("a\0b".into()), 0x0100, 0)],
    FilterPipelineError::Format(FormatError::InvalidFilterFlags { filter_id: 300, flags: 0x0100 })
)]
#[case::reserved_flags_version_two(
    Version::Two.byte(), vec![description(300, None, 0x8000, 0)],
    FilterPipelineError::Format(FormatError::InvalidFilterFlags { filter_id: 300, flags: 0x8000 })
)]
#[case::predefined_name(
    Version::Two.byte(), vec![description(255, Some(String::new()), 0, 0)],
    FilterPipelineError::InvalidName { filter_id: 255, reason: "predefined filters have no name field in version 2" }
)]
#[case::embedded_null(
    Version::One.byte(), vec![description(300, Some("a\0b".into()), 0, 0)],
    FilterPipelineError::InvalidName { filter_id: 300, reason: "name must be ASCII without null bytes" }
)]
#[case::non_ascii(
    Version::Two.byte(), vec![description(300, Some("été".into()), 0, 0)],
    FilterPipelineError::InvalidName { filter_id: 300, reason: "name must be ASCII without null bytes" }
)]
#[case::overlong_aligned_name(
    Version::One.byte(), vec![description(300, Some("n".repeat(65_528)), 0, 65_536)],
    FilterPipelineError::FieldTooLarge { field: "filter name length", value: 65_536, maximum: 65_535 }
)]
#[case::overlong_unpadded_name(
    Version::Two.byte(), vec![description(300, Some("n".repeat(65_535)), 0, 0)],
    FilterPipelineError::FieldTooLarge { field: "filter name length", value: 65_536, maximum: 65_535 }
)]
#[case::too_many_client_values(
    Version::Two.byte(), vec![description(300, None, 0, 65_536)],
    FilterPipelineError::FieldTooLarge { field: "client data count", value: 65_536, maximum: 65_535 }
)]
fn construction_rejects_unencodable_pipelines(
    #[case] version: u8,
    #[case] filters: Vec<FilterDescription>,
    #[case] error: FilterPipelineError,
) {
    assert_eq!(
        PreparedFilterPipeline::new(version, &filters).unwrap_err(),
        error
    );
    assert_eq!(FilterPipeline::new(version, filters).unwrap_err(), error);
}

#[test]
fn writer_uses_the_shortest_aligned_name_field_when_reencoding_version_one() {
    let mut fixture = Pipeline::new(Version::One);
    fixture.push(FILTER_ID, Some("x"), 0, &[]).unwrap();
    let expected = fixture.build();
    let mut bytes = expected.clone();
    bytes[10] = 16;
    bytes.extend_from_slice(&[0; 8]);

    let pipeline = FilterPipeline::parse(&bytes).unwrap();
    assert_eq!(pipeline.prepare().encoded_size(), expected.len());
    assert_eq!(pipeline.to_bytes(), expected);
}

#[test]
fn record_writer_preserves_the_destination_when_the_pipeline_body_exceeds_its_limit() {
    let values = vec![7; 65_535];
    let mut fixture = Pipeline::new(Version::Two);
    fixture.push(FILTER_ID, None, 0, &values).unwrap();
    let pipeline = expected_pipeline(&fixture);
    let mut destination = vec![0x37; 17];

    assert_eq!(
        MessageRecordLayout::PLAIN
            .prepare_record(MessageFlags::NONE, 0, pipeline.prepare())
            .and_then(|record| record.append_to(&mut destination))
            .unwrap_err(),
        FormatError::ObjectHeaderMessageTooLarge {
            message_type: FixtureMessageType::FILTER_PIPELINE.0,
            size: fixture.build().len(),
        }
    );
    assert_eq!(destination, vec![0x37; 17]);
}

fn description(id: u16, name: Option<String>, flags: u16, count: usize) -> FilterDescription {
    FilterDescription {
        filter_id: id,
        name,
        flags,
        client_data: vec![0; count],
    }
}

fn one_filter(version: Version) -> Pipeline<'static> {
    let mut fixture = Pipeline::new(version);
    fixture
        .push(FILTER_ID, Some("ab"), 0xfe, &[0x1234_5678])
        .unwrap();
    fixture
}

fn representative_pipeline(version: Version) -> Pipeline<'static> {
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

    fixture
}

fn expected_pipeline(fixture: &Pipeline<'_>) -> FilterPipeline {
    FilterPipeline::new(fixture.version().byte(), descriptions(fixture)).unwrap()
}

fn descriptions(fixture: &Pipeline<'_>) -> Vec<FilterDescription> {
    fixture
        .filters()
        .iter()
        .map(|filter| FilterDescription {
            filter_id: filter.id(),
            name: filter.name().map(str::to_owned),
            flags: u16::from(filter.flags()),
            client_data: filter.client_data().to_vec(),
        })
        .collect()
}

const FILTER_ID: u16 = 300;
const V1_NAME_LENGTH_AT: usize = 10;
const V1_FLAGS_AT: usize = 12;
const V1_FILTER_HEADER_END: usize = 16;
