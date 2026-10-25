use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DatasetCreateProperties;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FillTime;
use hdf5_pure::FilterId;
use hdf5_pure::FormatError;
use hdf5_pure::Layout;
use hdf5_pure::MaxExtent;
use hdf5_pure::PartialEdgeFiltering;
use hdf5_pure::ScaleOffset;
use rstest::rstest;
use test_util_hdf5::dataset_writer::DatasetWriter;

const VALUES: [i32; 8] = [-3, 0, 1, 2, 7, 8, 9, 100];
const OVERWRITTEN: [i32; 8] = [5, 4, 3, 2, 1, 0, -1, -2];

#[rstest]
#[case::empty(|p: DatasetCreateProperties| p, |_: &mut DatasetBuilder| {})]
#[case::chunks(|p: DatasetCreateProperties| p.with_chunks(&[4]), |b: &mut DatasetBuilder| { b.with_chunks(&[4]); })]
#[case::deflate(|p: DatasetCreateProperties| p.with_deflate(DeflateLevel::Six), |b: &mut DatasetBuilder| { b.with_deflate(DeflateLevel::Six); })]
#[case::shuffle_deflate(
    |p: DatasetCreateProperties| p.with_chunks(&[4]).with_shuffle().with_deflate(DeflateLevel::Six),
    |b: &mut DatasetBuilder| { b.with_chunks(&[4]).with_shuffle().with_deflate(DeflateLevel::Six); },
)]
#[case::lzf_fletcher32(
    |p: DatasetCreateProperties| p.with_chunks(&[2]).with_lzf().with_fletcher32(),
    |b: &mut DatasetBuilder| { b.with_chunks(&[2]).with_lzf().with_fletcher32(); },
)]
#[case::scale_offset_deflate(
    |p: DatasetCreateProperties| p.with_scale_offset(ScaleOffset::Integer(0)).with_deflate(DeflateLevel::One),
    |b: &mut DatasetBuilder| { b.with_scale_offset(ScaleOffset::Integer(0)).with_deflate(DeflateLevel::One); },
)]
#[case::fill_contiguous(
    |p: DatasetCreateProperties| p.with_fill_value(-1_i32).with_fill_time(FillTime::Alloc),
    |b: &mut DatasetBuilder| { b.with_fill_value(-1_i32).with_fill_time(FillTime::Alloc); },
)]
#[case::fill_time_without_value(
    |p: DatasetCreateProperties| p.with_fill_time(FillTime::Never),
    |b: &mut DatasetBuilder| { b.with_fill_time(FillTime::Never); },
)]
#[case::default_partial_edge_filtering(
    |p: DatasetCreateProperties| p
        .with_chunks(&[3])
        .with_deflate(DeflateLevel::Six)
        .with_partial_edge_filtering(PartialEdgeFiltering::default()),
    |b: &mut DatasetBuilder| { b.with_chunks(&[3]).with_deflate(DeflateLevel::Six); },
)]
#[case::partial_edge_filtering_applied(
    |p: DatasetCreateProperties| p
        .with_chunks(&[3])
        .with_deflate(DeflateLevel::Six)
        .with_partial_edge_filtering(PartialEdgeFiltering::Apply),
    |b: &mut DatasetBuilder| { b.with_chunks(&[3]).with_deflate(DeflateLevel::Six); },
)]
#[case::all(
    |p: DatasetCreateProperties| p
        .with_chunks(&[4])
        .with_shuffle()
        .with_deflate(DeflateLevel::Six)
        .with_fill_value(-1_i32)
        .with_fill_time(FillTime::IfSet),
    |b: &mut DatasetBuilder| {
        b.with_chunks(&[4])
            .with_shuffle()
            .with_deflate(DeflateLevel::Six)
            .with_fill_value(-1_i32)
            .with_fill_time(FillTime::IfSet);
    },
)]
fn properties_write_the_bytes_of_the_builder_setters(
    #[case] properties: fn(DatasetCreateProperties) -> DatasetCreateProperties,
    #[case] setters: fn(&mut DatasetBuilder),
    #[values(DatasetWriter::Buffered, DatasetWriter::Staged)] writer: DatasetWriter,
) {
    let with_properties = write(writer, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_create_properties(properties(DatasetCreateProperties::new()));
    });
    let with_setters = write(writer, |builder| {
        setters(builder.with_i32_data(&VALUES));
    });
    assert_eq!(with_properties, with_setters);
}

#[rstest]
#[case::chunked(|b: &mut DatasetBuilder| { b.with_chunks(&[4]); })]
#[case::filtered(|b: &mut DatasetBuilder| { b.with_chunks(&[4]).with_shuffle().with_deflate(DeflateLevel::Six); })]
#[case::scale_offset(|b: &mut DatasetBuilder| { b.with_scale_offset(ScaleOffset::Integer(0)); })]
#[case::fill(|b: &mut DatasetBuilder| { b.with_fill_value(5_i32).with_fill_time(FillTime::Alloc); })]
fn default_properties_keep_the_builder_defaults(#[case] setters: fn(&mut DatasetBuilder)) {
    let with_properties = write(DatasetWriter::Buffered, |builder| {
        setters(
            builder
                .with_i32_data(&VALUES)
                .with_create_properties(DatasetCreateProperties::default()),
        );
    });
    let with_setters = write(DatasetWriter::Buffered, |builder| {
        setters(builder.with_i32_data(&VALUES));
    });
    assert_eq!(with_properties, with_setters);
}

#[test]
fn cloned_properties_write_the_bytes_of_the_builder_setters_on_each_dataset() {
    let properties = DatasetCreateProperties::new()
        .with_chunks(&[4])
        .with_shuffle()
        .with_deflate(DeflateLevel::Six)
        .with_fill_value(-1_i32)
        .with_fill_time(FillTime::IfSet);

    let mut builder = FileBuilder::new();
    for name in ["first", "second"] {
        builder
            .create_dataset(name)
            .with_i32_data(&VALUES)
            .with_create_properties(properties.clone());
    }
    let with_properties = builder.finish().unwrap();

    let mut builder = FileBuilder::new();
    for name in ["first", "second"] {
        builder
            .create_dataset(name)
            .with_i32_data(&VALUES)
            .with_chunks(&[4])
            .with_shuffle()
            .with_deflate(DeflateLevel::Six)
            .with_fill_value(-1_i32)
            .with_fill_time(FillTime::IfSet);
    }
    assert_eq!(with_properties, builder.finish().unwrap());
}

#[rstest]
#[case::with_empty(|p: DatasetCreateProperties| p, |_: &mut DatasetBuilder| {})]
#[case::with_shuffle(|p: DatasetCreateProperties| p.with_shuffle(), |b: &mut DatasetBuilder| { b.with_shuffle(); })]
#[case::with_fill_time(
    |p: DatasetCreateProperties| p.with_fill_time(FillTime::Never),
    |b: &mut DatasetBuilder| { b.with_fill_time(FillTime::Never); },
)]
#[case::with_lzf(|p: DatasetCreateProperties| p.with_chunks(&[2]).with_lzf(), |b: &mut DatasetBuilder| { b.with_chunks(&[2]).with_lzf(); })]
fn properties_replace_earlier_builder_settings(
    #[case] properties: fn(DatasetCreateProperties) -> DatasetCreateProperties,
    #[case] setters: fn(&mut DatasetBuilder),
) {
    let replaced = write(DatasetWriter::Buffered, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_chunks(&[4])
            .with_deflate(DeflateLevel::Six)
            .with_fletcher32()
            .with_fill_value(-1_i32)
            .with_fill_time(FillTime::Alloc)
            .with_create_properties(properties(DatasetCreateProperties::new()));
    });
    let expected = write(DatasetWriter::Buffered, |builder| {
        setters(builder.with_i32_data(&VALUES));
    });
    assert_eq!(replaced, expected);
}

#[test]
fn properties_replace_earlier_properties() {
    let replaced = write(DatasetWriter::Buffered, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_create_properties(
                DatasetCreateProperties::new()
                    .with_chunks(&[4])
                    .with_deflate(DeflateLevel::Six)
                    .with_fill_value(-1_i32),
            )
            .with_create_properties(DatasetCreateProperties::new().with_fletcher32());
    });
    let expected = write(DatasetWriter::Buffered, |builder| {
        builder.with_i32_data(&VALUES).with_fletcher32();
    });
    assert_eq!(replaced, expected);
}

#[rstest]
#[case::chunks(|b: &mut DatasetBuilder| { b.with_chunks(&[2]); })]
#[case::deflate_level(|b: &mut DatasetBuilder| { b.with_deflate(DeflateLevel::One); })]
#[case::added_filter(|b: &mut DatasetBuilder| { b.with_fletcher32(); })]
#[case::fill_value(|b: &mut DatasetBuilder| { b.with_fill_value(7_i32); })]
#[case::fill_time(|b: &mut DatasetBuilder| { b.with_fill_time(FillTime::Never); })]
fn builder_setters_override_applied_properties(#[case] setters: fn(&mut DatasetBuilder)) {
    let overridden = write(DatasetWriter::Buffered, |builder| {
        setters(
            builder.with_i32_data(&VALUES).with_create_properties(
                DatasetCreateProperties::new()
                    .with_chunks(&[4])
                    .with_shuffle()
                    .with_deflate(DeflateLevel::Six)
                    .with_fill_value(-1_i32)
                    .with_fill_time(FillTime::Alloc),
            ),
        );
    });
    let expected = write(DatasetWriter::Buffered, |builder| {
        setters(
            builder
                .with_i32_data(&VALUES)
                .with_chunks(&[4])
                .with_shuffle()
                .with_deflate(DeflateLevel::Six)
                .with_fill_value(-1_i32)
                .with_fill_time(FillTime::Alloc),
        );
    });
    assert_eq!(overridden, expected);
}

#[rstest]
#[case::chunks(
    |p: DatasetCreateProperties| p.with_chunks(&[5]),
    |b: &mut DatasetBuilder| { b.with_chunks(&[5]); },
)]
#[case::deflate_level(
    |p: DatasetCreateProperties| p.with_deflate(DeflateLevel::One),
    |b: &mut DatasetBuilder| { b.with_deflate(DeflateLevel::One); },
)]
#[case::added_filter(
    |p: DatasetCreateProperties| p.with_fletcher32(),
    |b: &mut DatasetBuilder| { b.with_fletcher32(); },
)]
#[case::fill_value(
    |p: DatasetCreateProperties| p.with_fill_value(7_i32),
    |b: &mut DatasetBuilder| { b.with_fill_value(7_i32); },
)]
#[case::fill_time(
    |p: DatasetCreateProperties| p.with_fill_time(FillTime::Never),
    |b: &mut DatasetBuilder| { b.with_fill_time(FillTime::Never); },
)]
fn builder_setters_keep_the_partial_edge_filtering_of_applied_properties(
    #[case] property: fn(DatasetCreateProperties) -> DatasetCreateProperties,
    #[case] setter: fn(&mut DatasetBuilder),
    #[values(DatasetWriter::Buffered, DatasetWriter::Staged)] writer: DatasetWriter,
) {
    let skipping = || {
        DatasetCreateProperties::new()
            .with_chunks(&[3])
            .with_deflate(DeflateLevel::Six)
            .with_partial_edge_filtering(PartialEdgeFiltering::Skip)
    };
    let overridden = write(writer, |builder| {
        setter(
            builder
                .with_i32_data(&VALUES)
                .with_create_properties(skipping()),
        );
    });
    let expected = write(writer, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_create_properties(property(skipping()));
    });
    assert_eq!(overridden, expected);
    let bypassed = pipeline_bypass_by_chunk(overridden);
    let mut last_chunk_bypassed = vec![false; bypassed.len() - 1];
    last_chunk_bypassed.push(true);
    assert_eq!(bypassed, last_chunk_bypassed);
}

#[test]
fn properties_replace_earlier_partial_edge_filtering() {
    let replaced = write(DatasetWriter::Buffered, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_create_properties(
                DatasetCreateProperties::new()
                    .with_chunks(&[3])
                    .with_deflate(DeflateLevel::Six)
                    .with_partial_edge_filtering(PartialEdgeFiltering::Skip),
            )
            .with_create_properties(
                DatasetCreateProperties::new()
                    .with_chunks(&[3])
                    .with_deflate(DeflateLevel::Six),
            );
    });
    let expected = write(DatasetWriter::Buffered, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_chunks(&[3])
            .with_deflate(DeflateLevel::Six);
    });
    assert_eq!(replaced, expected);
}

#[rstest]
#[case::apply(PartialEdgeFiltering::Apply, [false, false, false])]
#[case::skip(PartialEdgeFiltering::Skip, [false, false, true])]
fn only_a_partial_edge_chunk_under_skip_bypasses_the_pipeline(
    #[case] filtering: PartialEdgeFiltering,
    #[case] bypassed: [bool; 3],
    #[values(DatasetWriter::Buffered, DatasetWriter::Staged)] writer: DatasetWriter,
) {
    let bytes = write(writer, |builder| {
        builder.with_i32_data(&VALUES).with_create_properties(
            DatasetCreateProperties::new()
                .with_chunks(&[3])
                .with_shuffle()
                .with_deflate(DeflateLevel::Six)
                .with_partial_edge_filtering(filtering),
        );
    });
    let file = File::from_bytes(bytes).unwrap();
    let dataset = file.dataset("data").unwrap();
    let mut chunks = dataset.chunks().unwrap();
    chunks.sort_by_key(|chunk| chunk.offset.clone());
    let stored: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            (
                chunk.offset.clone(),
                chunk.filter_mask().raw_bits(),
                chunk.filters().bypasses_pipeline(),
            )
        })
        .collect();
    assert_eq!(
        stored,
        [
            (vec![0], 0, bypassed[0]),
            (vec![3], 0, bypassed[1]),
            (vec![6], 0, bypassed[2]),
        ]
    );
    assert_eq!(dataset.read_i32().unwrap(), VALUES);
}

#[rstest]
#[case::filter_without_chunks(
    |b: &mut DatasetBuilder| {
        b.with_create_properties(skipping_with(|p| p.with_deflate(DeflateLevel::Six)));
    },
    ChunkIndex::SingleChunk,
)]
#[case::maxshape_without_chunks(
    |b: &mut DatasetBuilder| {
        b.with_maxshape(&[MaxExtent::Unlimited])
            .with_create_properties(skipping_with(|p| p));
    },
    ChunkIndex::ExtensibleArray,
)]
fn skipping_partial_edge_filters_is_accepted_on_a_dataset_chunked_without_chunk_dimensions(
    #[case] configure: fn(&mut DatasetBuilder),
    #[case] index: ChunkIndex,
    #[values(DatasetWriter::Buffered, DatasetWriter::Staged)] writer: DatasetWriter,
) {
    let bytes = write(writer, |builder| {
        configure(builder.with_i32_data(&VALUES));
    });
    let file = File::from_bytes(bytes).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(
        dataset.layout().unwrap(),
        Layout::Chunked {
            chunk_shape: vec![8],
            index,
        }
    );
    assert_eq!(dataset.read_i32().unwrap(), VALUES);
}

#[rstest]
#[case::rank_1(
    |b: &mut DatasetBuilder| { b.with_i32_data(&VALUES); },
    |p: DatasetCreateProperties| p,
)]
#[case::scalar(
    |b: &mut DatasetBuilder| { b.with_i32_data(&[1]).with_shape(&[]); },
    |p: DatasetCreateProperties| p,
)]
#[case::fill_value(
    |b: &mut DatasetBuilder| { b.with_i32_data(&VALUES); },
    |p: DatasetCreateProperties| p.with_fill_value(-1_i32).with_fill_time(FillTime::Alloc),
)]
fn skipping_partial_edge_filters_on_a_contiguous_dataset_is_rejected(
    #[case] data: fn(&mut DatasetBuilder),
    #[case] properties: fn(DatasetCreateProperties) -> DatasetCreateProperties,
    #[values(DatasetWriter::Buffered, DatasetWriter::Staged)] writer: DatasetWriter,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rejected.h5");
    FileBuilder::new().write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let skip = |builder: &mut DatasetBuilder| {
        data(builder);
        builder.with_create_properties(skipping_with(properties));
    };

    let err = if matches!(writer, DatasetWriter::Staged) {
        let file = File::open_rw(&path).unwrap();
        let err = file.root().create_dataset("data", skip).unwrap_err();
        file.commit().unwrap();
        err
    } else {
        let mut builder = FileBuilder::new();
        skip(builder.create_dataset("data"));
        builder.finish().unwrap_err()
    };
    assert!(
        matches!(
            err,
            Error::Format(FormatError::PartialEdgeSkipRequiresChunking)
        ),
        "{err:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[rstest]
#[case::contiguous(|b: &mut DatasetBuilder| { b.with_i32_data(&VALUES); })]
#[case::chunked(|b: &mut DatasetBuilder| {
    b.with_i32_data(&VALUES).with_chunks(&[3]).with_deflate(DeflateLevel::Six);
})]
#[case::chunked_skipping_partial_edge_filters(|b: &mut DatasetBuilder| {
    b.with_i32_data(&VALUES)
        .with_create_properties(skipping_with(|p| p.with_chunks(&[3]).with_deflate(DeflateLevel::Six)));
})]
fn an_overwrite_that_sets_partial_edge_filtering_is_rejected(
    #[case] existing: fn(&mut DatasetBuilder),
    #[values(PartialEdgeFiltering::Apply, PartialEdgeFiltering::Skip)]
    filtering: PartialEdgeFiltering,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("overwritten.h5");
    DatasetWriter::Streaming.create(&path, "data", existing);
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let err = file
            .dataset("data")
            .unwrap()
            .write_staged(|builder| {
                builder.with_i32_data(&VALUES).with_create_properties(
                    DatasetCreateProperties::new().with_partial_edge_filtering(filtering),
                );
            })
            .unwrap_err();
        let Error::EditUnsupported(reason) = err else {
            panic!("expected EditUnsupported, got {err:?}");
        };
        assert_eq!(
            reason,
            "write_dataset overwrites values only; it cannot set whether partial edge \
             chunks are filtered (set it when the dataset is created)"
        );
        file.commit().unwrap();
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[rstest]
#[case::without_properties(|b: &mut DatasetBuilder| { b.with_i32_data(&OVERWRITTEN); })]
#[case::default_properties(|b: &mut DatasetBuilder| {
    b.with_i32_data(&OVERWRITTEN).with_create_properties(DatasetCreateProperties::new());
})]
fn an_overwrite_that_leaves_partial_edge_filtering_unset_keeps_skipping_partial_edge_filters(
    #[case] overwrite: fn(&mut DatasetBuilder),
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("overwritten.h5");
    DatasetWriter::Streaming.create(&path, "data", |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_create_properties(skipping_with(|p| {
                p.with_chunks(&[3]).with_deflate(DeflateLevel::Six)
            }));
    });
    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("data")
            .unwrap()
            .write_staged(overwrite)
            .unwrap();
        file.commit().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        pipeline_bypass_by_chunk(bytes.clone()),
        [false, false, true]
    );
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(
        file.dataset("data").unwrap().read_i32().unwrap(),
        OVERWRITTEN
    );
}

#[rstest]
fn applying_partial_edge_filters_writes_the_bytes_of_the_default_policy(
    #[values(
        DatasetWriter::Buffered,
        DatasetWriter::Staged,
        DatasetWriter::Streaming
    )]
    writer: DatasetWriter,
) {
    let write_with = |properties: DatasetCreateProperties| {
        write(writer, |builder| {
            builder
                .with_i32_data(&VALUES)
                .with_create_properties(properties);
        })
    };
    let properties = DatasetCreateProperties::new()
        .with_chunks(&[3])
        .with_deflate(DeflateLevel::Six);
    let applied = write_with(
        properties
            .clone()
            .with_partial_edge_filtering(PartialEdgeFiltering::Apply),
    );
    assert_eq!(applied, write_with(properties));
    assert_eq!(pipeline_bypass_by_chunk(applied), [false, false, false]);
}

#[rstest]
#[case::alloc(FillTime::Alloc)]
#[case::never(FillTime::Never)]
#[case::if_set(FillTime::IfSet)]
fn fill_value_and_fill_time_read_back(#[case] fill_time: FillTime) {
    let bytes = write(DatasetWriter::Buffered, |builder| {
        builder.with_i32_data(&VALUES).with_create_properties(
            DatasetCreateProperties::new()
                .with_chunks(&[4])
                .with_fill_value(-1_i32)
                .with_fill_time(fill_time),
        );
    });
    let file = File::from_bytes(bytes).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.fill_value::<i32>().unwrap(), Some(-1));
    assert_eq!(dataset.fill_time().unwrap(), fill_time);
    assert_eq!(dataset.read_i32().unwrap(), VALUES);
}

#[rstest]
#[case::data_after_properties(true)]
#[case::data_before_properties(false)]
fn fill_value_of_another_width_is_rejected(#[case] data_after_properties: bool) {
    let properties = DatasetCreateProperties::new().with_fill_value(1_u8);
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("data");
    if data_after_properties {
        dataset
            .with_create_properties(properties)
            .with_i32_data(&VALUES);
    } else {
        dataset
            .with_i32_data(&VALUES)
            .with_create_properties(properties);
    }
    let err = builder.finish().unwrap_err();
    assert!(
        matches!(
            err,
            Error::Format(FormatError::FillValueSizeMismatch {
                expected: 4,
                actual: 1
            })
        ),
        "{err:?}"
    );
}

#[test]
fn properties_place_filters_in_canonical_order_with_lzf_optional() {
    let bytes = write(DatasetWriter::Buffered, |builder| {
        builder.with_i32_data(&VALUES).with_create_properties(
            DatasetCreateProperties::new()
                .with_chunks(&[4])
                .with_fletcher32()
                .with_lzf()
                .with_shuffle(),
        );
    });
    let file = File::from_bytes(bytes).unwrap();
    let pipeline: Vec<(FilterId, bool)> = file
        .dataset("data")
        .unwrap()
        .filter_pipeline()
        .unwrap()
        .into_iter()
        .map(|filter| (filter.id, filter.is_optional))
        .collect();
    assert_eq!(
        pipeline,
        [
            (FilterId::Shuffle, false),
            (FilterId::Lzf, true),
            (FilterId::Fletcher32, false),
        ]
    );
}

#[rstest]
#[case::lzf_deflate(
    |p: DatasetCreateProperties| p.with_lzf().with_deflate(DeflateLevel::Six),
    "lzf and deflate cannot be combined on one dataset",
)]
#[case::shuffle_scale_offset(
    |p: DatasetCreateProperties| p.with_scale_offset(ScaleOffset::Integer(0)).with_shuffle(),
    "shuffle and scale-offset cannot be combined on one dataset",
)]
fn conflicting_filters_are_rejected(
    #[case] properties: fn(DatasetCreateProperties) -> DatasetCreateProperties,
    #[case] reason: &str,
) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&VALUES)
        .with_create_properties(properties(DatasetCreateProperties::new().with_chunks(&[4])));
    let err = builder.finish().unwrap_err();
    let Error::Format(FormatError::FilterError(message)) = &err else {
        panic!("expected FilterError, got {err:?}");
    };
    assert_eq!(message, reason);
}

#[cfg(feature = "zfp")]
#[rstest]
fn zfp_properties_write_the_bytes_of_the_builder_setter(
    #[values(DatasetWriter::Buffered, DatasetWriter::Staged)] writer: DatasetWriter,
) {
    use hdf5_pure::ZfpRate;

    let rate = ZfpRate::try_from(16.0).unwrap();
    let with_properties = write(writer, |builder| {
        builder.with_i32_data(&VALUES).with_create_properties(
            DatasetCreateProperties::new()
                .with_chunks(&[4])
                .with_zfp(rate),
        );
    });
    let with_setters = write(writer, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_chunks(&[4])
            .with_zfp(rate);
    });
    assert_eq!(with_properties, with_setters);
}

fn skipping_with(
    properties: impl FnOnce(DatasetCreateProperties) -> DatasetCreateProperties,
) -> DatasetCreateProperties {
    properties(
        DatasetCreateProperties::new().with_partial_edge_filtering(PartialEdgeFiltering::Skip),
    )
}

fn pipeline_bypass_by_chunk(bytes: Vec<u8>) -> Vec<bool> {
    let file = File::from_bytes(bytes).unwrap();
    let mut chunks = file.dataset("data").unwrap().chunks().unwrap();
    chunks.sort_by_key(|chunk| chunk.offset.clone());
    chunks
        .iter()
        .map(|chunk| chunk.filters().bypasses_pipeline())
        .collect()
}

/// Writes a file with one dataset, `data`, that `configure` sets up, through `writer`.
fn write(writer: DatasetWriter, configure: impl FnOnce(&mut DatasetBuilder)) -> Vec<u8> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("written.h5");
    writer.create(&path, "data", configure);
    std::fs::read(path).unwrap()
}
