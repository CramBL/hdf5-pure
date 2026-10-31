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
use hdf5_pure::{CustomFilterId, Error, FilterId, FilterParameters};
use hdf5_pure_format::__private::FILTER_PIPELINE_VERSION_TWO;
use rstest::rstest;
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
        MalformedFilter::ReservedId,
        MalformedFilter::UnknownVersion
    )]
    malformed: MalformedFilter,
) {
    let (version, id, values, expected) = match malformed {
        MalformedFilter::InvalidDeflateLevel => (
            2,
            1,
            vec![10],
            FormatError::InvalidDeflateLevel { level: 10 },
        ),
        MalformedFilter::MissingDeflateParameter => (
            2,
            1,
            vec![],
            FormatError::InvalidFilterParameterCount {
                filter_id: 1,
                expected: 1,
                actual: 0,
            },
        ),
        MalformedFilter::ReservedId => (2, 0, vec![4], FormatError::InvalidFilterId { id: 0 }),
        MalformedFilter::UnknownVersion => (
            99,
            2,
            vec![4],
            FormatError::InvalidFilterPipelineVersion(99),
        ),
        MalformedFilter::ZeroShuffleSize => (
            2,
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
    let mut body = hdf5_pure_format::__private::FilterPipeline {
        version: FILTER_PIPELINE_VERSION_TWO,
        filters: vec![hdf5_pure_format::__private::FilterDescription {
            filter_id: id,
            name: None,
            flags: 0,
            client_data: values,
        }],
    }
    .serialize()
    .unwrap();
    body[0] = version;
    replace_pipeline(&mut bytes, &body);
    let dataset = File::from_bytes(bytes).unwrap().dataset("data").unwrap();
    let error = match accessor {
        FilterAccessor::Chunks => dataset.chunks().unwrap_err(),
        FilterAccessor::Filters => dataset.filters().unwrap_err(),
        FilterAccessor::Pipeline => dataset.filter_pipeline().unwrap_err(),
    };
    let Error::Format(actual) = error else {
        panic!("expected format error, got {error:?}")
    };
    assert_eq!(actual, expected);
}

#[rstest]
#[case::lzf(32000, FilterId::Lzf)]
#[case::zfp(32013, FilterId::Zfp)]
#[case::custom(40000, FilterId::Custom(CustomFilterId::try_from(40000).unwrap()))]
fn opaque_filter_metadata_is_lossless(
    #[case] raw: u16,
    #[case] id: FilterId,
    #[values(0x8000, 0x8001)] flags: u16,
) {
    let values = vec![u32::MAX, 0, 7];
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[1, 2, 3, 4])
        .with_lzf();
    let mut bytes = builder.finish().unwrap();
    let body = hdf5_pure_format::__private::FilterPipeline {
        version: FILTER_PIPELINE_VERSION_TWO,
        filters: vec![hdf5_pure_format::__private::FilterDescription {
            filter_id: raw,
            name: Some("xyz".into()),
            flags,
            client_data: values.clone(),
        }],
    }
    .serialize()
    .unwrap();
    replace_pipeline(&mut bytes, &body);
    let dataset = File::from_bytes(bytes).unwrap().dataset("data").unwrap();
    assert_eq!(dataset.filters().unwrap(), [id]);
    let pipeline = dataset.filter_pipeline().unwrap();
    assert_eq!(pipeline.len(), 1);
    assert_eq!(pipeline[0].id(), id);
    assert_eq!(pipeline[0].parameters(), &FilterParameters::Opaque);
    assert_eq!(pipeline[0].name(), Some("xyz"));
    assert_eq!(pipeline[0].flags(), flags);
    assert_eq!(
        pipeline[0].is_optional(),
        flags & hdf5_pure_filter::__private::H5Z_FLAG_OPTIONAL != 0
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

#[derive(Clone, Copy)]
enum MalformedFilter {
    InvalidDeflateLevel,
    MissingDeflateParameter,
    ReservedId,
    UnknownVersion,
    ZeroShuffleSize,
}

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
