#[cfg(feature = "deflate")]
use core::num::NonZeroU32;

#[cfg(feature = "deflate")]
use hdf5_pure::DatasetBuilder;
#[cfg(feature = "deflate")]
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FilterDisposition;
use hdf5_pure::FormatError;
use hdf5_pure::{CustomFilterId, Dataset, Error, FilterId, FilterParameters};
use rstest::rstest;
use test_util::filter_pipeline::{Pipeline, Version};
use test_util::object_header::v2;

#[rstest]
#[cfg(feature = "deflate")]
#[case::level_zero(DeflateLevel::Zero)]
#[case::level_six(DeflateLevel::Six)]
#[case::level_nine(DeflateLevel::Nine)]
fn deflate_preserves_typed_level(
    #[case] level: DeflateLevel,
    #[values(false, true)] shuffle: bool,
    #[values(false, true)] staged: bool,
) {
    let configure = |builder: &mut DatasetBuilder| {
        builder.with_i32_data(&[1, 2, 3, 4]).with_deflate(level);
        if shuffle {
            builder.with_shuffle();
        }
    };
    let file = if staged {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("staged.h5");
        FileBuilder::new().write(&path).unwrap();
        {
            let file = File::open_rw(&path).unwrap();
            file.root().create_dataset("data", configure).unwrap();
            file.commit().unwrap();
        }
        File::from_bytes(std::fs::read(path).unwrap()).unwrap()
    } else {
        let mut builder = FileBuilder::new();
        configure(builder.create_dataset("data"));
        File::from_bytes(builder.finish().unwrap()).unwrap()
    };
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [1, 2, 3, 4]);
    assert_eq!(
        dataset
            .filter_pipeline()
            .unwrap()
            .last()
            .unwrap()
            .parameters(),
        &FilterParameters::Deflate(level)
    );
    assert_eq!(
        dataset
            .filter_pipeline()
            .unwrap()
            .last()
            .unwrap()
            .client_data(),
        [u32::from(level.get())]
    );
}

#[rstest]
#[case::chunks(FilterAccessor::Chunks)]
#[case::filters(FilterAccessor::Filters)]
#[case::pipeline(FilterAccessor::Pipeline)]
fn filter_accessors_report_malformed_metadata(
    #[case] accessor: FilterAccessor,
    #[values(
        MalformedFilter::MissingDeflateParameter,
        MalformedFilter::InvalidDeflateLevel,
        MalformedFilter::ZeroShuffleSize,
        MalformedFilter::ReservedFlags,
        MalformedFilter::ReservedId,
        MalformedFilter::UnknownVersion
    )]
    malformed: MalformedFilter,
) {
    let (id, values, expected) = match malformed {
        MalformedFilter::InvalidDeflateLevel => {
            (1, vec![10], FormatError::InvalidDeflateLevel { level: 10 })
        }
        MalformedFilter::MissingDeflateParameter => (
            1,
            vec![],
            FormatError::InvalidFilterParameterCount {
                filter_id: 1,
                expected: 1,
                actual: 0,
            },
        ),
        MalformedFilter::ReservedFlags => (
            2,
            vec![4],
            FormatError::InvalidFilterFlags {
                filter_id: 2,
                flags: 0x0100,
            },
        ),
        MalformedFilter::ReservedId => (2, vec![4], FormatError::InvalidFilterId { id: 0 }),
        MalformedFilter::UnknownVersion => {
            (2, vec![4], FormatError::InvalidFilterPipelineVersion(99))
        }
        MalformedFilter::ZeroShuffleSize => (
            2,
            vec![0],
            FormatError::InvalidShuffleElementSize { size: 0 },
        ),
    };
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shuffle();
    let mut bytes = builder.finish().unwrap();
    let mut fixture = Pipeline::new(Version::Two);
    fixture.push(id, None, 0, &values).unwrap();
    let mut body = fixture.build();
    match malformed {
        MalformedFilter::ReservedFlags => body[5] = 1,
        MalformedFilter::ReservedId => body[2] = 0,
        MalformedFilter::UnknownVersion => body[0] = 99,
        _ => {}
    }
    replace_pipeline(&mut bytes, &body);
    let dataset = File::from_bytes(bytes).unwrap().dataset("data").unwrap();
    assert_eq!(accessor.format_error(&dataset), expected);
}

#[rstest]
#[case::version_one_count(
    pipeline_with_excessive_count(Version::One),
    excessive_filter_count_error()
)]
#[case::version_two_count(
    pipeline_with_excessive_count(Version::Two),
    excessive_filter_count_error()
)]
#[case::version_one_name(pipeline_with_invalid_name(Version::One), invalid_filter_name_error())]
#[case::version_two_name(pipeline_with_invalid_name(Version::Two), invalid_filter_name_error())]
#[case::version_one_incomplete_reserved_bytes(
    pipeline_with_excessive_count(Version::One)[..V1_PIPELINE_PREFIX - 1].to_vec(),
    FormatError::UnexpectedEof { expected: V1_PIPELINE_PREFIX, available: V1_PIPELINE_PREFIX - 1 }
)]
#[case::missing_count(
    vec![UNSUPPORTED_VERSION],
    FormatError::UnexpectedEof { expected: V2_PIPELINE_PREFIX, available: FILTER_COUNT_AT }
)]
#[case::unknown_version(
    vec![UNSUPPORTED_VERSION, MAX_FILTER_COUNT + 1],
    FormatError::InvalidFilterPipelineVersion(UNSUPPORTED_VERSION)
)]
fn filter_accessors_preserve_pipeline_parse_errors(
    #[case] body: Vec<u8>,
    #[case] expected: FormatError,
    #[values(
        FilterAccessor::Chunks,
        FilterAccessor::Filters,
        FilterAccessor::Pipeline
    )]
    accessor: FilterAccessor,
) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[1, 2, 3, 4])
        .with_lzf()
        .with_shuffle()
        .with_fletcher32();
    let mut bytes = builder.finish().unwrap();
    replace_pipeline(&mut bytes, &body);
    let dataset = File::from_bytes(bytes).unwrap().dataset("data").unwrap();

    assert_eq!(accessor.format_error(&dataset), expected);
}

#[rstest]
#[case::lzf(32000, FilterId::Lzf)]
#[case::zfp(32013, FilterId::Zfp)]
#[case::custom(40000, FilterId::Custom(CustomFilterId::try_from(40000).unwrap()))]
fn opaque_filter_metadata_is_lossless(
    #[case] raw: u16,
    #[case] id: FilterId,
    #[values(0x80, 0x81)] flags: u8,
) {
    let values = vec![u32::MAX, 0, 7];
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[1, 2, 3, 4])
        .with_lzf();
    let mut bytes = builder.finish().unwrap();
    let mut fixture = Pipeline::new(Version::Two);
    fixture.push(raw, Some("xyz"), flags, &values).unwrap();
    replace_pipeline(&mut bytes, &fixture.build());
    let dataset = File::from_bytes(bytes).unwrap().dataset("data").unwrap();
    assert_eq!(dataset.filters().unwrap(), [id]);
    let pipeline = dataset.filter_pipeline().unwrap();
    assert_eq!(pipeline.len(), 1);
    assert_eq!(pipeline[0].id(), id);
    assert_eq!(pipeline[0].parameters(), &FilterParameters::Opaque);
    assert_eq!(pipeline[0].name(), Some("xyz"));
    assert_eq!(pipeline[0].flags(), u16::from(flags));
    assert_eq!(
        pipeline[0].is_optional(),
        u16::from(flags) & hdf5_pure_filter::__private::H5Z_FLAG_OPTIONAL != 0
    );
    assert_eq!(pipeline[0].client_data(), values);
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 1);
    let (filter, disposition) = chunks[0].filters().filter(0).unwrap();
    assert_eq!(filter, &pipeline[0]);
    assert_eq!(disposition, FilterDisposition::Applied);
}

#[rstest]
#[cfg(feature = "deflate")]
#[case::mandatory(false)]
#[case::optional(true)]
fn staged_filter_parameters_are_pending_until_commit(#[case] optional: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pending.h5");
    FileBuilder::new().write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let dataset = file
        .root()
        .create_dataset("data", |builder| {
            builder.with_i32_data(&[1, 2, 3, 4]).with_shuffle();
            if optional {
                builder.with_lzf();
            } else {
                builder.with_deflate(DeflateLevel::Six);
            }
        })
        .unwrap();
    let id = if optional {
        FilterId::Lzf
    } else {
        FilterId::Deflate
    };
    assert_eq!(dataset.filters().unwrap(), [FilterId::Shuffle, id]);
    let pending = dataset.filter_pipeline().unwrap();
    assert_eq!(
        pending
            .iter()
            .map(|filter| filter.parameters())
            .collect::<Vec<_>>(),
        [&FilterParameters::Pending, &FilterParameters::Pending]
    );
    assert_eq!(pending[0].id(), FilterId::Shuffle);
    assert_eq!(pending[1].id(), id);
    assert_eq!(pending[0].name(), None);
    assert_eq!(pending[1].name(), None);
    assert!(pending[0].client_data().is_empty());
    assert!(pending[1].client_data().is_empty());
    assert_eq!(pending[0].flags(), 0);
    assert_eq!(
        pending[1].flags(),
        if optional {
            hdf5_pure_filter::__private::H5Z_FLAG_OPTIONAL
        } else {
            0
        }
    );
    assert!(!pending[0].is_optional());
    assert_eq!(pending[1].is_optional(), optional);
    assert_eq!(pending[0].to_string(), "shuffle [pending]");
    file.commit().unwrap();
    let parsed = dataset.filter_pipeline().unwrap();
    assert_eq!(parsed[0].id(), FilterId::Shuffle);
    assert_eq!(parsed[1].id(), id);
    assert_eq!(
        parsed[0].parameters(),
        &FilterParameters::Shuffle(NonZeroU32::new(4).unwrap())
    );
    assert_eq!(parsed[0].client_data(), [4]);
    if optional {
        assert_eq!(parsed[1].parameters(), &FilterParameters::Opaque);
        assert_eq!(parsed[1].client_data(), [4, 0x0105, 16]);
        assert_eq!(parsed[1].name(), Some("lzf"));
    } else {
        assert_eq!(
            parsed[1].parameters(),
            &FilterParameters::Deflate(DeflateLevel::Six)
        );
        assert_eq!(parsed[1].client_data(), [6]);
    }
    assert_eq!(parsed[0].flags(), pending[0].flags());
    assert_eq!(parsed[1].flags(), pending[1].flags());
    assert!(!parsed[0].is_optional());
    assert_eq!(parsed[1].is_optional(), optional);
    assert_eq!(dataset.read_i32().unwrap(), [1, 2, 3, 4]);
    assert_eq!(pending[1].parameters(), &FilterParameters::Pending);
}

#[derive(Clone, Copy)]
enum FilterAccessor {
    Chunks,
    Filters,
    Pipeline,
}

impl FilterAccessor {
    fn format_error(self, dataset: &Dataset) -> FormatError {
        let error = match self {
            Self::Chunks => dataset.chunks().unwrap_err(),
            Self::Filters => dataset.filters().unwrap_err(),
            Self::Pipeline => dataset.filter_pipeline().unwrap_err(),
        };
        let Error::Format(error) = error else {
            panic!("expected format error, got {error:?}")
        };
        error
    }
}

#[derive(Clone, Copy)]
enum MalformedFilter {
    InvalidDeflateLevel,
    MissingDeflateParameter,
    ReservedFlags,
    ReservedId,
    UnknownVersion,
    ZeroShuffleSize,
}

fn pipeline_with_excessive_count(version: Version) -> Vec<u8> {
    let mut bytes = Pipeline::new(version).build();
    bytes[FILTER_COUNT_AT] = MAX_FILTER_COUNT + 1;
    bytes
}

fn excessive_filter_count_error() -> FormatError {
    FormatError::InvalidFilterPipelineField {
        field: "filter count",
        value: usize::from(MAX_FILTER_COUNT) + 1,
        maximum: usize::from(MAX_FILTER_COUNT),
    }
}

fn pipeline_with_invalid_name(version: Version) -> Vec<u8> {
    let mut fixture = Pipeline::new(version);
    fixture.push(NAMED_FILTER_ID, Some("odd"), 0, &[]).unwrap();
    let mut bytes = fixture.build();
    let name_at = bytes.windows(3).position(|bytes| bytes == b"odd").unwrap();
    bytes[name_at] = 0xff;
    bytes
}

fn invalid_filter_name_error() -> FormatError {
    FormatError::InvalidFilterName {
        filter_id: NAMED_FILTER_ID,
        reason: "invalid UTF-8",
    }
}

// "The Data Storage - Filter Pipeline Message" in the format specification, version 4.0,
// defines the version 1 and 2 prefix fields and limits the filter count to 32.
const FILTER_COUNT_AT: usize = 1;
const MAX_FILTER_COUNT: u8 = 32;
const V1_PIPELINE_PREFIX: usize = 8;
const V2_PIPELINE_PREFIX: usize = 2;
const NAMED_FILTER_ID: u16 = 300;
const UNSUPPORTED_VERSION: u8 = 99;

fn replace_pipeline(bytes: &mut [u8], body: &[u8]) {
    let header = bytes
        .windows(v2::SIGNATURE.len())
        .rposition(|window| window == v2::SIGNATURE)
        .unwrap();
    test_util::object_header::v2::replace_message(
        bytes,
        header,
        test_util::object_header::MessageType::FILTER_PIPELINE,
        body,
    );
}
