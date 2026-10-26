use hdf5_pure::AttrValue;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FillTime;
use hdf5_pure::FormatError;
use hdf5_pure::Layout;
use hdf5_pure::MaxExtent;
use rstest::rstest;

#[rstest]
#[case::contiguous(&[3, 4], false, false, -7)]
#[case::chunked(&[3, 4], true, false, -7)]
#[case::extensible(&[3, 4], true, true, -7)]
#[case::empty(&[0, 4], true, true, -7)]
#[case::scalar(&[], false, false, -7)]
#[case::default_fill(&[3, 4], false, false, 0)]
fn schemas_read_as_fill_without_element_storage(
    #[case] shape: &[u64],
    #[case] chunked: bool,
    #[case] extensible: bool,
    #[case] fill: i32,
    #[values(false, true)] streaming: bool,
) {
    let mut builder = FileBuilder::new();
    let dataset = builder
        .create_dataset("schema")
        .with_unallocated_storage(hdf5_pure::make_i32_type(), shape);
    if fill != 0 {
        dataset.with_fill_value(fill);
    }
    if chunked {
        dataset.with_chunks(&[2, 2]).with_fletcher32();
    }
    if extensible {
        dataset.with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Fixed(4)]);
    }
    let bytes = if streaming {
        let mut bytes = Vec::new();
        builder.finish_to(&mut bytes).unwrap();
        bytes
    } else {
        builder.finish().unwrap()
    };
    let file = File::from_bytes(bytes).unwrap();
    let dataset = file.dataset("schema").unwrap();
    assert_eq!(dataset.shape().unwrap(), shape);
    assert_eq!(dataset.datatype().unwrap(), hdf5_pure::make_i32_type());
    assert_eq!(
        dataset.read_i32().unwrap(),
        vec![fill; shape.iter().product::<u64>() as usize]
    );
    if chunked {
        assert_eq!(dataset.chunks().unwrap().len(), 0);
    } else {
        assert_eq!(
            dataset.layout().unwrap(),
            Layout::Contiguous {
                address: None,
                size: shape.iter().product::<u64>() * 4,
            }
        );
    }
}

#[rstest]
#[case::before_data(true, &[1, 2, 3])]
#[case::after_data(false, &[1, 2, 3])]
#[case::before_empty_data(true, &[])]
#[case::after_empty_data(false, &[])]
fn unallocated_storage_rejects_element_payloads(#[case] schema_first: bool, #[case] data: &[i32]) {
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("schema");
    if schema_first {
        dataset.with_unallocated_storage(hdf5_pure::make_i32_type(), &[3]);
    }
    dataset.with_i32_data(data);
    if !schema_first {
        dataset.with_unallocated_storage(hdf5_pure::make_i32_type(), &[3]);
    }
    let error = builder.finish().unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::ConflictingDatasetStorage)),
        "{error:?}"
    );
}

#[rstest]
#[case::missing_datatype(false, true, FormatError::DatasetMissingDatatype)]
#[case::missing_shape(true, false, FormatError::DatasetMissingShape)]
#[case::missing_data(true, true, FormatError::DatasetMissingData)]
fn ordinary_builders_require_data(
    #[case] datatype: bool,
    #[case] shape: bool,
    #[case] expected: FormatError,
) {
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("incomplete");
    if datatype {
        dataset.with_dtype(hdf5_pure::make_i32_type());
    }
    if shape {
        dataset.with_shape(&[3]);
    }
    let error = builder.finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::element_count(hdf5_pure::make_u8_type(), &[u64::MAX - 1, 2], FormatError::DatasetSizeOverflow { size: u64::MAX - 1, dimension: 2 })]
#[case::byte_size(hdf5_pure::make_i32_type(), &[u64::MAX / 4 + 1], FormatError::DatasetSizeOverflow { size: 4, dimension: u64::MAX / 4 + 1 })]
#[case::rank(hdf5_pure::make_i32_type(), &[1; 33], FormatError::DatasetRankUnsupported { rank: 33, maximum: 32 })]
#[case::unlimited_dimension(hdf5_pure::make_u8_type(), &[u64::MAX], FormatError::DatasetDimensionUnlimited { dimension: 0 })]
#[case::empty_unlimited_dimension(hdf5_pure::make_u8_type(), &[0, u64::MAX], FormatError::DatasetDimensionUnlimited { dimension: 1 })]
fn schemas_reject_unrepresentable_shapes(
    #[case] datatype: Datatype,
    #[case] shape: &[u64],
    #[case] expected: FormatError,
) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("schema")
        .with_unallocated_storage(datatype, shape);
    let error = builder.finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::fill_width(|dataset: &mut DatasetBuilder| { dataset.with_fill_value(1_u8); }, FormatError::FillValueSizeMismatch { expected: 4, actual: 1 })]
#[case::chunk_rank(|dataset: &mut DatasetBuilder| { dataset.with_chunks(&[2]); }, FormatError::InvalidChunkGeometry("chunk dimensions must have the same rank as the dataset shape"))]
#[case::wide_chunk(|dataset: &mut DatasetBuilder| { dataset.with_chunks(&[4_294_967_297, 1]); }, FormatError::InvalidChunkGeometry("chunk logical byte size exceeds the reader's single-chunk allocation limit"))]
#[case::large_chunk(|dataset: &mut DatasetBuilder| { dataset.with_dtype(hdf5_pure::make_u8_type()).with_chunks(&[65_536, 65_536]); }, FormatError::InvalidChunkGeometry("chunk logical byte size exceeds the reader's single-chunk allocation limit"))]
#[case::maximum_shape(|dataset: &mut DatasetBuilder| { dataset.with_maxshape(&[MaxExtent::Fixed(2), MaxExtent::Fixed(4)]); }, FormatError::InvalidChunkGeometry("maxshape must be at least the current shape in every dimension"))]
fn schemas_reject_incompatible_settings(
    #[case] configure: fn(&mut DatasetBuilder),
    #[case] expected: FormatError,
    #[values(false, true)] streaming: bool,
) {
    let mut builder = FileBuilder::new();
    configure(
        builder
            .create_dataset("schema")
            .with_unallocated_storage(hdf5_pure::make_i32_type(), &[3, 4]),
    );
    let error = if streaming {
        builder.finish_to(&mut Vec::new()).unwrap_err()
    } else {
        builder.finish().unwrap_err()
    };
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
}

#[cfg(feature = "provenance")]
#[rstest]
#[case::contiguous(false)]
#[case::chunked(true)]
fn nonempty_schemas_require_element_data_for_provenance(#[case] chunked: bool) {
    let mut builder = FileBuilder::new();
    let schema = builder
        .create_dataset("schema")
        .with_unallocated_storage(hdf5_pure::make_i32_type(), &[3])
        .with_provenance("acquisition", "2026-10-21T00:00:00Z", None);
    if chunked {
        schema.with_chunks(&[2]);
    }
    let error = builder.finish().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::DatasetProvenanceRequiresData)
        ),
        "{error:?}"
    );
}

#[cfg(feature = "provenance")]
#[test]
fn empty_schemas_verify_their_provenance() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("schema")
        .with_unallocated_storage(hdf5_pure::make_i32_type(), &[0])
        .with_provenance("acquisition", "2026-10-21T00:00:00Z", None);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(
        file.dataset("schema").unwrap().verify_provenance().unwrap(),
        hdf5_pure::VerifyResult::Ok
    );
}

#[rstest]
#[case::contiguous(&[3, 4], false, false)]
#[case::chunked(&[3, 4], true, false)]
#[case::extensible(&[3, 4], true, true)]
#[case::empty(&[0, 4], true, true)]
#[case::scalar(&[], false, false)]
fn edited_schemas_read_fill_without_element_storage(
    #[case] shape: &[u64],
    #[case] chunked: bool,
    #[case] extensible: bool,
    #[values(None, Some(-7_i32))] fill: Option<i32>,
    #[values(FillTime::IfSet, FillTime::Never)] fill_time: FillTime,
) {
    let bytes = edit_schema(|schema| {
        schema
            .with_unallocated_storage(hdf5_pure::make_i32_type(), shape)
            .with_fill_time(fill_time)
            .set_attr("units", AttrValue::String("counts".into()));
        if let Some(fill) = fill {
            schema.with_fill_value(fill);
        }
        if chunked {
            schema.with_chunks(&[2, 2]).with_fletcher32();
        }
        if extensible {
            schema.with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Fixed(4)]);
        }
    })
    .unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let dataset = file.dataset("schema").unwrap();
    assert_eq!(dataset.shape().unwrap(), shape);
    assert_eq!(dataset.datatype().unwrap(), hdf5_pure::make_i32_type());
    let expected_fill = if fill_time == FillTime::Never {
        0
    } else {
        fill.unwrap_or(0)
    };
    assert_eq!(
        dataset.read_i32().unwrap(),
        vec![expected_fill; shape.iter().product::<u64>() as usize]
    );
    assert_eq!(
        dataset.attrs().unwrap()["units"],
        AttrValue::String("counts".into())
    );
    if chunked {
        assert_eq!(dataset.chunks().unwrap().len(), 0);
    } else {
        assert_eq!(
            dataset.layout().unwrap(),
            Layout::Contiguous {
                address: None,
                size: shape.iter().product::<u64>() * 4,
            }
        );
    }
}

#[rstest]
#[case::rank(&[1; 33], FormatError::DatasetRankUnsupported { rank: 33, maximum: 32 })]
#[case::unlimited(&[u64::MAX], FormatError::DatasetDimensionUnlimited { dimension: 0 })]
#[case::empty_unlimited(&[0, u64::MAX], FormatError::DatasetDimensionUnlimited { dimension: 1 })]
#[case::overflow(&[u64::MAX / 4 + 1], FormatError::DatasetSizeOverflow { size: 4, dimension: u64::MAX / 4 + 1 })]
fn edited_schemas_reject_unrepresentable_shapes(
    #[case] shape: &[u64],
    #[case] expected: FormatError,
) {
    let error = edit_schema(|schema| {
        schema.with_unallocated_storage(hdf5_pure::make_i32_type(), shape);
    })
    .unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::fill_width(|schema: &mut DatasetBuilder| { schema.with_fill_value(1_u8); }, Error::Format(FormatError::FillValueSizeMismatch { expected: 4, actual: 1 }))]
#[case::chunk_rank(|schema: &mut DatasetBuilder| { schema.with_chunks(&[2]); }, Error::EditUnsupported("chunk dimensions must have the same rank as the dataset shape"))]
#[case::zero_chunk(|schema: &mut DatasetBuilder| { schema.with_chunks(&[0, 2]); }, Error::EditUnsupported("chunk dimensions must all be non-zero"))]
#[case::wide_chunk(|schema: &mut DatasetBuilder| { schema.with_chunks(&[4_294_967_297, 1]); }, Error::Format(FormatError::InvalidChunkGeometry("chunk logical byte size exceeds the reader's single-chunk allocation limit")))]
#[case::large_chunk(|schema: &mut DatasetBuilder| { schema.with_dtype(hdf5_pure::make_u8_type()).with_chunks(&[65_536, 65_536]); }, Error::Format(FormatError::InvalidChunkGeometry("chunk logical byte size exceeds the reader's single-chunk allocation limit")))]
#[case::maximum_shape(|schema: &mut DatasetBuilder| { schema.with_maxshape(&[MaxExtent::Fixed(2), MaxExtent::Fixed(4)]); }, Error::EditUnsupported("maxshape must be at least the current shape in every dimension"))]
#[case::scalar_chunk(|schema: &mut DatasetBuilder| { schema.with_shape(&[]).with_chunks(&[]); }, Error::EditUnsupported("a scalar dataset cannot be chunked, filtered, or extensible"))]
#[case::zero_width_type(|schema: &mut DatasetBuilder| { schema.with_dtype(Datatype::String { size: 0, padding: hdf5_pure::StringPadding::NullPad, charset: hdf5_pure::CharacterSet::Ascii }); }, Error::Format(FormatError::ZeroSizedDatatype { class: 3 }))]
#[case::filter_type(|schema: &mut DatasetBuilder| { schema.with_dtype(Datatype::String { size: 4, padding: hdf5_pure::StringPadding::NullPad, charset: hdf5_pure::CharacterSet::Ascii }).with_scale_offset(hdf5_pure::ScaleOffset::Integer(0)); }, Error::Format(FormatError::FilterError("scale-offset requires an integer or floating-point scalar datatype with a definite (little/big endian) byte order".into())))]
#[case::payload(|schema: &mut DatasetBuilder| { schema.with_i32_data(&[1, 2, 3]); }, Error::Format(FormatError::ConflictingDatasetStorage))]
#[case::empty_payload(|schema: &mut DatasetBuilder| { schema.with_i32_data(&[]); }, Error::Format(FormatError::ConflictingDatasetStorage))]
fn edited_schemas_reject_incompatible_settings(
    #[case] configure: fn(&mut DatasetBuilder),
    #[case] expected: Error,
) {
    let error = edit_schema(|schema| {
        schema.with_unallocated_storage(hdf5_pure::make_i32_type(), &[3, 4]);
        configure(schema);
    })
    .unwrap_err();
    match (error, expected) {
        (Error::Format(actual), Error::Format(expected)) => assert_eq!(actual, expected),
        (Error::EditUnsupported(actual), Error::EditUnsupported(expected)) => {
            assert_eq!(actual, expected);
        }
        (actual, expected) => panic!("expected {expected:?}, got {actual:?}"),
    }
}

#[rstest]
#[case::before_commit(false)]
#[case::after_commit(true)]
fn edited_logging_schema_can_be_appended(#[case] commit_before_append: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("schema.h5");
    FileBuilder::new().write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file
            .root()
            .create_dataset("temperature", |schema| {
                schema
                    .with_unallocated_storage(hdf5_pure::make_f64_type(), &[0])
                    .with_chunks(&[512])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            })
            .unwrap();
        if commit_before_append {
            file.commit().unwrap();
        }
        dataset
            .append_staged(|append| {
                append.append_f64(&[18.5, 19.0]);
            })
            .unwrap();
        file.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("temperature").unwrap();
    assert_eq!(dataset.shape().unwrap(), [2]);
    assert_eq!(dataset.read_f64().unwrap(), [18.5, 19.0]);
}

#[rstest]
#[case::contiguous(false)]
#[case::chunked(true)]
fn appending_to_a_staged_schema_preserves_fill(#[case] chunked: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("schema.h5");
    FileBuilder::new().write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file
            .root()
            .create_dataset("schema", |schema| {
                schema
                    .with_unallocated_storage(hdf5_pure::make_i32_type(), &[3])
                    .with_fill_value(-7_i32);
                if chunked {
                    schema.with_chunks(&[2]);
                }
            })
            .unwrap();
        dataset
            .append_staged(|append| {
                append.append_i32(&[8, 9]);
            })
            .unwrap();
        file.commit().unwrap();
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("schema")
            .unwrap()
            .read_i32()
            .unwrap(),
        [-7, -7, -7, 8, 9]
    );
}

#[test]
fn unallocated_storage_cannot_replace_dataset_values() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("schema.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("schema").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let error = file
        .dataset("schema")
        .unwrap()
        .write_staged(|schema| {
            schema.with_unallocated_storage(hdf5_pure::make_i32_type(), &[3]);
        })
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("write_dataset requires element data, not unallocated storage")
        ),
        "{error:?}"
    );
    assert!(!file.has_staged_edits());
    file.commit().unwrap();
    assert_eq!(
        file.dataset("schema").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
}

#[cfg(feature = "provenance")]
#[test]
fn edited_nonempty_schemas_require_element_data_for_provenance() {
    let error = edit_schema(|schema| {
        schema
            .with_unallocated_storage(hdf5_pure::make_i32_type(), &[3])
            .with_provenance("acquisition", "2026-10-21T00:00:00Z", None);
    })
    .unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::DatasetProvenanceRequiresData)
        ),
        "{error:?}"
    );
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "rejected schemas must leave pending edits and committed bytes unchanged"
)]
fn edit_schema(configure: impl FnOnce(&mut DatasetBuilder)) -> Result<Vec<u8>, Error> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("schema.h5");
    FileBuilder::new().write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let result = file.root().create_dataset("schema", configure);
        if let Err(error) = result {
            assert!(!file.has_staged_edits());
            file.commit().unwrap();
            drop(file);
            assert_eq!(std::fs::read(&path).unwrap(), before);
            return Err(error);
        }
        file.commit()?;
    }
    Ok(std::fs::read(path).unwrap())
}
