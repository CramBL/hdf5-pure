use hdf5_pure::DatasetBuilder;
use hdf5_pure::DatasetCreateProperties;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FillTime;
use hdf5_pure::FilterId;
use hdf5_pure::FormatError;
use hdf5_pure::ScaleOffset;
use rstest::rstest;

const VALUES: [i32; 8] = [-3, 0, 1, 2, 7, 8, 9, 100];

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
    #[values(false, true)] staged: bool,
) {
    let with_properties = write(staged, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_create_properties(properties(DatasetCreateProperties::new()));
    });
    let with_setters = write(staged, |builder| {
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
    let with_properties = write(false, |builder| {
        setters(
            builder
                .with_i32_data(&VALUES)
                .with_create_properties(DatasetCreateProperties::default()),
        );
    });
    let with_setters = write(false, |builder| {
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
    let replaced = write(false, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_chunks(&[4])
            .with_deflate(DeflateLevel::Six)
            .with_fletcher32()
            .with_fill_value(-1_i32)
            .with_fill_time(FillTime::Alloc)
            .with_create_properties(properties(DatasetCreateProperties::new()));
    });
    let expected = write(false, |builder| {
        setters(builder.with_i32_data(&VALUES));
    });
    assert_eq!(replaced, expected);
}

#[test]
fn properties_replace_earlier_properties() {
    let replaced = write(false, |builder| {
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
    let expected = write(false, |builder| {
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
    let overridden = write(false, |builder| {
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
    let expected = write(false, |builder| {
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
#[case::alloc(FillTime::Alloc)]
#[case::never(FillTime::Never)]
#[case::if_set(FillTime::IfSet)]
fn fill_value_and_fill_time_read_back(#[case] fill_time: FillTime) {
    let bytes = write(false, |builder| {
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
    let bytes = write(false, |builder| {
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
fn zfp_properties_write_the_bytes_of_the_builder_setter(#[values(false, true)] staged: bool) {
    use hdf5_pure::ZfpRate;

    let rate = ZfpRate::try_from(16.0).unwrap();
    let with_properties = write(staged, |builder| {
        builder.with_i32_data(&VALUES).with_create_properties(
            DatasetCreateProperties::new()
                .with_chunks(&[4])
                .with_zfp(rate),
        );
    });
    let with_setters = write(staged, |builder| {
        builder
            .with_i32_data(&VALUES)
            .with_chunks(&[4])
            .with_zfp(rate);
    });
    assert_eq!(with_properties, with_setters);
}

/// Writes a file with one dataset, `data`, that `configure` sets up, through a [`FileBuilder`],
/// or through a read-write session that stages and commits the dataset where `staged` is true.
fn write(staged: bool, configure: impl FnOnce(&mut DatasetBuilder)) -> Vec<u8> {
    if !staged {
        let mut builder = FileBuilder::new();
        configure(builder.create_dataset("data"));
        return builder.finish().unwrap();
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged.h5");
    FileBuilder::new().write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.root().create_dataset("data", configure).unwrap();
        file.commit().unwrap();
    }
    std::fs::read(path).unwrap()
}
