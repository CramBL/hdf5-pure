#[cfg(not(feature = "std"))]
use alloc::format;

use hdf5_pure_format::FilterPipelineError;

use crate::error::FormatError;

pub use hdf5_pure_filter::FILTER_DEFLATE;
pub use hdf5_pure_filter::FILTER_FLETCHER32;
pub use hdf5_pure_filter::FILTER_LZF;
pub use hdf5_pure_filter::FILTER_SCALEOFFSET;
pub use hdf5_pure_filter::FILTER_SHUFFLE;
#[cfg(feature = "zfp")]
pub use hdf5_pure_filter::FILTER_ZFP;
pub use hdf5_pure_filter::H5Z_FLAG_OPTIONAL;
pub use hdf5_pure_format::FilterDescription;
pub use hdf5_pure_format::FilterPipeline;

pub(crate) fn parse_filter_pipeline(data: &[u8]) -> Result<FilterPipeline, FormatError> {
    FilterPipeline::parse(data).map_err(map_filter_pipeline_error)
}

pub(crate) fn map_filter_pipeline_error(error: FilterPipelineError) -> FormatError {
    match error {
        FilterPipelineError::Format(error) => error,
        FilterPipelineError::InvalidName { filter_id, reason } => {
            FormatError::FilterError(format!("invalid name for filter {filter_id}: {reason}"))
        }
        FilterPipelineError::FieldTooLarge {
            field,
            value,
            maximum,
        } => FormatError::FilterError(format!(
            "filter pipeline {field} is {value}, above maximum {maximum}"
        )),
    }
}

/// Presents a pipeline's stored filter descriptions to the filter routines.
pub(crate) struct FilterStepsRef<'a>(&'a [FilterDescription]);

impl<'a> FilterStepsRef<'a> {
    pub(crate) fn new(pipeline: &'a FilterPipeline) -> Self {
        Self(&pipeline.filters)
    }
}

impl hdf5_pure_filter::FilterSteps for FilterStepsRef<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn id(&self, index: usize) -> u16 {
        self.0[index].filter_id
    }

    fn flags(&self, index: usize) -> u16 {
        self.0[index].flags
    }

    fn client_data(&self, index: usize) -> &[u32] {
        &self.0[index].client_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case(
        FilterPipelineError::InvalidName { filter_id: 300, reason: "invalid UTF-8" },
        "invalid name for filter 300: invalid UTF-8"
    )]
    #[case(
        FilterPipelineError::FieldTooLarge { field: "filter count", value: 33, maximum: 32 },
        "filter pipeline filter count is 33, above maximum 32"
    )]
    fn filter_pipeline_validation_errors_map_to_filter_error(
        #[case] error: FilterPipelineError,
        #[case] message: &str,
    ) {
        assert_eq!(
            map_filter_pipeline_error(error),
            FormatError::FilterError(message.into())
        );
    }

    #[test]
    fn parse_v1_single_deflate() {
        // v1 pipeline with 1 filter: deflate (id=1) level 6
        let mut buf = vec![1u8, 1]; // version=1, nfilters=1
        buf.extend_from_slice(&[0u8; 6]); // reserved
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes()); // filter_id=1
        buf.extend_from_slice(&0u16.to_le_bytes()); // name_length
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&1u16.to_le_bytes()); // nclient
        buf.extend_from_slice(&6u32.to_le_bytes()); // level=6
        // odd client data count => 4 bytes padding
        buf.extend_from_slice(&[0u8; 4]);

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.version, 1);
        assert_eq!(fp.filters.len(), 1);
        assert_eq!(fp.filters[0].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[0].client_data, vec![6]);
        assert_eq!(fp.filters[0].name, None);
    }

    #[test]
    fn parse_v1_shuffle_and_deflate() {
        let mut buf = vec![1u8, 2];
        buf.extend_from_slice(&[0u8; 6]);
        // shuffle (id=2)
        buf.extend_from_slice(&FILTER_SHUFFLE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        // deflate (id=1)
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // padding for odd client data

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.filters.len(), 2);
        assert_eq!(fp.filters[0].filter_id, FILTER_SHUFFLE);
        assert_eq!(fp.filters[1].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[1].client_data, vec![6]);
    }

    #[test]
    fn parse_v2_deflate() {
        let mut buf = vec![2u8, 1]; // version=2, nfilters=1
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes()); // id=1
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&1u16.to_le_bytes()); // nclient
        buf.extend_from_slice(&6u32.to_le_bytes()); // level=6

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.version, 2);
        assert_eq!(fp.filters.len(), 1);
        assert_eq!(fp.filters[0].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[0].client_data, vec![6]);
    }

    #[test]
    fn parse_v2_three_filters() {
        let mut buf = vec![2u8, 3]; // version=2, nfilters=3
        // shuffle (id=2)
        buf.extend_from_slice(&FILTER_SHUFFLE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        // deflate (id=1)
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes());
        // fletcher32 (id=3)
        buf.extend_from_slice(&FILTER_FLETCHER32.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.filters.len(), 3);
        assert_eq!(fp.filters[0].filter_id, FILTER_SHUFFLE);
        assert_eq!(fp.filters[1].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[2].filter_id, FILTER_FLETCHER32);
    }

    #[test]
    fn serialize_parse_roundtrip_v1() {
        let pipeline = FilterPipeline {
            version: 1,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let serialized = pipeline.serialize().unwrap();
        let parsed = FilterPipeline::parse(&serialized).unwrap();
        assert_eq!(parsed, pipeline);
    }

    #[test]
    fn serialize_parse_roundtrip_v2() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![9],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let serialized = pipeline.serialize().unwrap();
        let parsed = FilterPipeline::parse(&serialized).unwrap();
        assert_eq!(parsed, pipeline);
    }

    #[test]
    fn custom_filter_with_name_v1() {
        let mut buf = vec![1u8, 1];
        buf.extend_from_slice(&[0u8; 6]);
        // custom filter: id=300 (>=256), name_length=16 ("myfilter\0" padded to 8)
        buf.extend_from_slice(&300u16.to_le_bytes());
        let name = b"myfilter\0"; // 9 bytes, pad to 16
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&2u16.to_le_bytes()); // nclient=2
        // name padded to 8-byte boundary: 9 bytes -> 16 bytes
        buf.extend_from_slice(name);
        buf.extend_from_slice(&[0u8; 7]); // pad 9->16
        // client data: 2 values (even, no padding needed)
        buf.extend_from_slice(&42u32.to_le_bytes());
        buf.extend_from_slice(&99u32.to_le_bytes());

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.filters[0].filter_id, 300);
        assert_eq!(fp.filters[0].name, Some("myfilter".to_string()));
        assert_eq!(fp.filters[0].client_data, vec![42, 99]);
    }

    #[test]
    fn custom_filter_with_name_v2() {
        let mut buf = vec![2u8, 1];
        // id=300 (>=256), so name_length field present
        buf.extend_from_slice(&300u16.to_le_bytes());
        let name = b"custom\0";
        buf.extend_from_slice(&7u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // nclient=0
        buf.extend_from_slice(name);
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: 300,
                name: Some("custom".to_string()),
                flags: 0,
                client_data: Vec::new(),
            }],
        };
        assert_eq!(FilterPipeline::parse(&buf), Ok(pipeline.clone()));
        assert_eq!(pipeline.serialize(), Ok(buf));
    }

    #[test]
    fn invalid_version() {
        let buf = vec![3u8, 0];
        let err = FilterPipeline::parse(&buf).unwrap_err();
        assert_eq!(
            err,
            FilterPipelineError::Format(FormatError::InvalidFilterPipelineVersion(3))
        );
    }

    #[test]
    fn encoding_rejects_an_unknown_version() {
        let pipeline = FilterPipeline {
            version: 3,
            filters: Vec::new(),
        };
        assert_eq!(
            pipeline.serialize().unwrap_err(),
            FilterPipelineError::Format(FormatError::InvalidFilterPipelineVersion(3))
        );
    }

    #[test]
    fn version_two_predefined_filter_rejects_a_name_it_cannot_store() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: Some("deflate".to_string()),
                flags: 0,
                client_data: Vec::new(),
            }],
        };
        assert_eq!(
            pipeline.serialize().unwrap_err(),
            FilterPipelineError::InvalidName {
                filter_id: FILTER_DEFLATE,
                reason: "predefined filters have no name field in version 2",
            }
        );
    }

    #[test]
    fn version_two_custom_filter_without_name_round_trips() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: 300,
                name: None,
                flags: 0,
                client_data: Vec::new(),
            }],
        };
        assert_eq!(
            FilterPipeline::parse(&pipeline.serialize().unwrap()),
            Ok(pipeline)
        );
    }

    #[rstest::rstest]
    #[case(1)]
    #[case(2)]
    fn encoding_rejects_too_many_filters(#[case] version: u8) {
        let pipeline = FilterPipeline {
            version,
            filters: vec![filter_description(); 33],
        };
        assert_eq!(
            pipeline.serialize().unwrap_err(),
            FilterPipelineError::FieldTooLarge {
                field: "filter count",
                value: 33,
                maximum: 32,
            }
        );
    }

    #[test]
    fn parsing_rejects_a_count_above_the_format_limit() {
        assert_eq!(
            FilterPipeline::parse(&[2, 33]).unwrap_err(),
            FilterPipelineError::FieldTooLarge {
                field: "filter count",
                value: 33,
                maximum: 32,
            }
        );
    }

    #[rstest::rstest]
    #[case(1)]
    #[case(2)]
    fn encoding_rejects_a_name_longer_than_its_field(#[case] version: u8) {
        let mut filter = filter_description();
        filter.name = Some("x".repeat(65_535));
        let pipeline = FilterPipeline {
            version,
            filters: vec![filter],
        };
        assert_eq!(
            pipeline.serialize().unwrap_err(),
            FilterPipelineError::FieldTooLarge {
                field: "filter name length",
                value: 65_536,
                maximum: 65_535,
            }
        );
    }

    #[rstest::rstest]
    #[case(1)]
    #[case(2)]
    fn encoding_rejects_too_many_client_values(#[case] version: u8) {
        let mut filter = filter_description();
        filter.client_data = vec![0; 65_536];
        let pipeline = FilterPipeline {
            version,
            filters: vec![filter],
        };
        assert_eq!(
            pipeline.serialize().unwrap_err(),
            FilterPipelineError::FieldTooLarge {
                field: "client data count",
                value: 65_536,
                maximum: 65_535,
            }
        );
    }

    #[rstest::rstest]
    #[case(1)]
    #[case(2)]
    fn parsing_rejects_invalid_utf8_in_a_filter_name(#[case] version: u8) {
        let mut bytes = vec![version, 1];
        if version == 1 {
            bytes.extend_from_slice(&[0; 6]);
        }
        bytes.extend_from_slice(&300u16.to_le_bytes());
        let name_length = if version == 1 { 8u16 } else { 2u16 };
        bytes.extend_from_slice(&name_length.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0]);
        if version == 1 {
            bytes.extend_from_slice(&[0; 6]);
        }
        assert_eq!(
            FilterPipeline::parse(&bytes).unwrap_err(),
            FilterPipelineError::InvalidName {
                filter_id: 300,
                reason: "invalid UTF-8",
            }
        );
    }

    fn filter_description() -> FilterDescription {
        FilterDescription {
            filter_id: 300,
            name: Some("custom".to_string()),
            flags: 0,
            client_data: Vec::new(),
        }
    }
}
