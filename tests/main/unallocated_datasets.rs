use hdf5_pure::DatasetBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
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
#[case::maximum_shape(|dataset: &mut DatasetBuilder| { dataset.with_maxshape(&[MaxExtent::Fixed(2), MaxExtent::Fixed(4)]); }, FormatError::InvalidChunkGeometry("maxshape must be at least the current shape in every dimension"))]
fn schemas_reject_incompatible_settings(
    #[case] configure: fn(&mut DatasetBuilder),
    #[case] expected: FormatError,
) {
    let mut builder = FileBuilder::new();
    configure(
        builder
            .create_dataset("schema")
            .with_unallocated_storage(hdf5_pure::make_i32_type(), &[3, 4]),
    );
    let error = builder.finish().unwrap_err();
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
