//! Dataset request errors, overwrite compatibility, and append behavior.

use core::num::{NonZeroU32, NonZeroUsize};

use hdf5_pure_format::__private::ChunkedGeometry;
use hdf5_pure_format::__private::FILTER_PIPELINE_VERSION_TWO;
use rstest::rstest;

use super::common;

use crate::AttrValue;
use crate::DatatypeByteOrder;
use crate::DeflateLevel;
use crate::FixedPointType;
use crate::FloatingPointType;
use crate::VariableLengthStringType;
use crate::VariableLengthType;
#[cfg(feature = "zfp")]
use crate::ZfpRate;
use crate::chunked_write;
use crate::chunked_write::{ChunkProvider, StorageAllocation};
use crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS;
use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags, DataLayout};
use crate::dataspace::{Dataspace, DataspaceType, MaxExtent};
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
use crate::datatype::StringPadding;
use crate::datatype::layout::FloatingPointLayout;
use crate::edit;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::edit::dataset::{ChunkGeometry, ChunkedOverwrite};
use crate::error::{Error, FormatError};
use crate::file_lock::FileLocking;
use crate::fill_value::PaddingFill;
use crate::filter_pipeline::FILTER_DEFLATE;
use crate::filter_pipeline::FilterDescription;
use crate::filter_pipeline::FilterPipeline;
use crate::filters;
use crate::filters::ChunkContext;
use crate::object_path::{LinkName, ObjectPathBuf};
use crate::type_builders;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

#[rstest]
#[case::missing_datatype(|b: &mut DatasetBuilder| b.datatype = None, FormatError::DatasetMissingDatatype)]
#[case::missing_shape(|b: &mut DatasetBuilder| b.shape = None, FormatError::DatasetMissingShape)]
#[case::missing_data(|b: &mut DatasetBuilder| b.data = None, FormatError::DatasetMissingData)]
#[case::shape_payload(|b: &mut DatasetBuilder| { b.with_shape(&[2]); }, FormatError::LogicalShapeDataMismatch {
    expected: 8, actual: 4, element_size: NonZeroU32::new(4).unwrap().into(),
})]
#[case::zero_extent_payload(|b: &mut DatasetBuilder| { b.with_shape(&[0]); }, FormatError::LogicalShapeDataMismatch {
    expected: 0, actual: 4, element_size: NonZeroU32::new(4).unwrap().into(),
})]
#[case::fill_width(|b: &mut DatasetBuilder| { b.with_fill_value(1_u8); }, FormatError::FillValueSizeMismatch { expected: 4, actual: 1 })]
#[case::rank(|b: &mut DatasetBuilder| { b.with_shape(&[1; 33]); }, FormatError::DatasetRankUnsupported { rank: 33, maximum: 32 })]
#[case::unlimited_current(|b: &mut DatasetBuilder| { b.with_shape(&[u64::MAX]); }, FormatError::DatasetDimensionUnlimited { dimension: 0 })]
#[case::chunk_rank(|b: &mut DatasetBuilder| { b.with_chunks(&[1, 1]); }, FormatError::InvalidChunkGeometry("chunk dimensions must have the same rank as the dataset shape"))]
#[case::chunk_zero(|b: &mut DatasetBuilder| { b.with_chunks(&[0]); }, FormatError::InvalidChunkGeometry("chunk dimensions must all be non-zero"))]
#[case::scalar_chunks(|b: &mut DatasetBuilder| { b.with_shape(&[]).with_chunks(&[1]); }, FormatError::InvalidChunkGeometry("a scalar dataset cannot be chunked, filtered, or extensible"))]
#[case::maximum_rank(|b: &mut DatasetBuilder| { b.with_maxshape(&[MaxExtent::Fixed(1), MaxExtent::Fixed(1)]); }, FormatError::InvalidChunkGeometry("maxshape must have the same rank as the dataset shape"))]
#[case::maximum_below_current(|b: &mut DatasetBuilder| { b.with_maxshape(&[MaxExtent::Fixed(0)]); }, FormatError::InvalidChunkGeometry("maxshape must be at least the current shape in every dimension"))]
#[case::byte_extent_overflow(|b: &mut DatasetBuilder| { b.with_shape(&[u64::MAX / 4 + 1]); }, FormatError::DatasetSizeOverflow { size: 4, dimension: u64::MAX / 4 + 1 })]
#[case::zero_element_width(|b: &mut DatasetBuilder| { b.with_dtype(Datatype::String { size: 0, padding: StringPadding::NullPad, charset: CharacterSet::Ascii }); }, FormatError::ZeroSizedDatatype { class: STRING_DATATYPE_CLASS })]
#[case::unallocated_payload(|b: &mut DatasetBuilder| b.allocation = StorageAllocation::Unallocated, FormatError::ConflictingDatasetStorage)]
#[case::producer_extent(|b: &mut DatasetBuilder| { b.data = None; b.with_produced_data(type_builders::make_i32_type(), &[1], 8, 4, Box::new(UnreadDatasetProvider)); }, FormatError::LogicalShapeDataMismatch {
    expected: 4, actual: 8, element_size: NonZeroU32::new(4).unwrap().into(),
})]
#[case::empty_attribute(|b: &mut DatasetBuilder| { b.set_attr("", AttrValue::I32(1)); }, FormatError::InvalidAttributeName { name: String::new() })]
#[case::null_attribute(|b: &mut DatasetBuilder| { b.set_attr("a\0b", AttrValue::I32(1)); }, FormatError::InvalidAttributeName { name: "a\0b".into() })]
fn creation_backends_preserve_common_request_causes(
    #[case] configure: fn(&mut DatasetBuilder),
    #[case] expected: FormatError,
) {
    let mut file = FileBuilder::new();
    let builder = file.create_dataset("values");
    builder.with_i32_data(&[1]);
    configure(builder);
    let error = file.finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);

    let mut builder = DatasetBuilder::new("values");
    builder.with_i32_data(&[1]);
    configure(&mut builder);
    let result = edit::dataset::flatten_dataset(builder, LinkName::new("values").unwrap().into());
    let Err(Error::Format(error)) = result else {
        panic!("expected the same format error from in-place creation");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::unsigned_byte(1, false, 0, 8, false, 0, 8, true)]
#[case::signed_byte(1, true, 0, 8, true, 0, 8, true)]
#[case::narrow_byte(1, false, 1, 7, false, 1, 7, true)]
#[case::two_bytes(2, false, 0, 16, false, 0, 16, false)]
#[case::four_bytes(4, false, 0, 32, false, 0, 32, false)]
#[case::signedness(1, false, 0, 8, true, 0, 8, false)]
#[case::offset(1, false, 0, 7, false, 1, 7, false)]
#[case::precision(1, false, 0, 8, false, 0, 7, false)]
fn overwrite_ignores_one_byte_order_and_requires_matching_layout(
    #[case] size: u32,
    #[case] disk_signed: bool,
    #[case] disk_offset: u16,
    #[case] disk_precision: u16,
    #[case] staged_signed: bool,
    #[case] staged_offset: u16,
    #[case] staged_precision: u16,
    #[case] compatible: bool,
) {
    let disk = Datatype::FixedPoint(
        FixedPointType::new(
            size,
            DatatypeByteOrder::BigEndian,
            disk_signed,
            disk_offset,
            disk_precision,
        )
        .unwrap(),
    );
    let staged = Datatype::FixedPoint(
        FixedPointType::new(
            size,
            DatatypeByteOrder::LittleEndian,
            staged_signed,
            staged_offset,
            staged_precision,
        )
        .unwrap(),
    );
    assert_ne!(disk, staged);
    assert_eq!(
        edit::dataset::overwrite_datatypes_match(&disk, &staged),
        compatible
    );
    assert_eq!(
        edit::dataset::overwrite_datatypes_match(&staged, &disk),
        compatible
    );
}

#[rstest]
#[case::array(array_type(big_endian_u8_type(), vec![2]), array_type(type_builders::make_u8_type(), vec![2]), true)]
#[case::array_shape(array_type(big_endian_u8_type(), vec![2]), array_type(type_builders::make_u8_type(), vec![1, 2]), false)]
#[case::compound(
    single_member_compound_type(big_endian_u8_type(), 1, "x", 0),
    single_member_compound_type(type_builders::make_u8_type(), 1, "x", 0),
    true
)]
#[case::compound_size(
    single_member_compound_type(big_endian_u8_type(), 2, "x", 0),
    single_member_compound_type(type_builders::make_u8_type(), 1, "x", 0),
    false
)]
#[case::compound_name(
    single_member_compound_type(big_endian_u8_type(), 1, "x", 0),
    single_member_compound_type(type_builders::make_u8_type(), 1, "y", 0),
    false
)]
#[case::compound_offset(
    single_member_compound_type(big_endian_u8_type(), 2, "x", 1),
    single_member_compound_type(type_builders::make_u8_type(), 2, "x", 0),
    false
)]
#[case::enum_base(
    single_member_enumeration_type(big_endian_u8_type(), 1),
    single_member_enumeration_type(type_builders::make_u8_type(), 1),
    true
)]
#[case::enum_value(
    single_member_enumeration_type(big_endian_u8_type(), 1),
    single_member_enumeration_type(type_builders::make_u8_type(), 2),
    false
)]
#[case::vlen_string(
    variable_length_string_type(
        big_endian_u8_type(),
        CharacterSet::Utf8,
        StringPadding::NullTerminate
    ),
    variable_length_string_type(
        type_builders::make_u8_type(),
        CharacterSet::Utf8,
        StringPadding::NullTerminate
    ),
    true
)]
#[case::vlen_sequence(
    Datatype::VariableLength(VariableLengthType::sequence(big_endian_u8_type())),
    Datatype::VariableLength(VariableLengthType::sequence(type_builders::make_u8_type())),
    true
)]
#[case::vlen_kind(
    Datatype::VariableLength(VariableLengthType::sequence(big_endian_u8_type())),
    variable_length_string_type(
        type_builders::make_u8_type(),
        CharacterSet::Utf8,
        StringPadding::NullTerminate
    ),
    false
)]
#[case::vlen_charset(
    variable_length_string_type(
        big_endian_u8_type(),
        CharacterSet::Ascii,
        StringPadding::NullTerminate
    ),
    variable_length_string_type(
        type_builders::make_u8_type(),
        CharacterSet::Utf8,
        StringPadding::NullTerminate
    ),
    false
)]
#[case::vlen_padding(
    variable_length_string_type(big_endian_u8_type(), CharacterSet::Utf8, StringPadding::SpacePad),
    variable_length_string_type(
        type_builders::make_u8_type(),
        CharacterSet::Utf8,
        StringPadding::NullTerminate
    ),
    false
)]
#[case::nested(array_type(single_member_compound_type(variable_length_string_type(big_endian_u8_type(), CharacterSet::Utf8, StringPadding::NullTerminate), 16, "x", 0), vec![2]), array_type(single_member_compound_type(variable_length_string_type(type_builders::make_u8_type(), CharacterSet::Utf8, StringPadding::NullTerminate), 16, "x", 0), vec![2]), true)]
fn overwrite_requires_matching_nested_fields_except_one_byte_order(
    #[case] disk: Datatype,
    #[case] staged: Datatype,
    #[case] compatible: bool,
) {
    assert_ne!(disk, staged);
    assert_eq!(
        edit::dataset::overwrite_datatypes_match(&disk, &staged),
        compatible
    );
    assert_eq!(
        edit::dataset::overwrite_datatypes_match(&staged, &disk),
        compatible
    );
}

#[rstest]
#[case::rank_1_partial_tail(&[10], &[4], &[false, false, true])]
#[case::partial_along_both_axes(&[5, 5], &[2, 3], &[false, true, false, true, true, true])]
#[case::partial_along_the_inner_axis(&[4, 5], &[2, 3], &[false, true, false, true])]
#[case::partial_along_the_outer_axis(
    &[5, 6],
    &[2, 3],
    &[false, false, false, false, true, true],
)]
#[case::partial_along_the_first_and_last_of_three_axes(
    &[3, 4, 5],
    &[2, 4, 3],
    &[false, true, true, true],
)]
fn a_chunked_overwrite_stores_its_partial_edge_chunks_raw_under_the_layout_flag(
    #[case] shape: &[u64],
    #[case] chunk_dims: &[u64],
    #[case] partial: &[bool],
    #[values(false, true)] stored_raw: bool,
) {
    let element_size = crate::convert::nz(4);
    let pipeline = FilterPipeline {
        version: FILTER_PIPELINE_VERSION_TWO,
        filters: vec![FilterDescription {
            filter_id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: vec![6],
        }],
    };
    let padding = PaddingFill::Value(17_i32.to_le_bytes().to_vec());
    let values: Vec<u8> = (0..shape.iter().product::<u64>())
        .flat_map(|value| i32::try_from(value).unwrap().to_le_bytes())
        .collect();
    let logical = chunked_write::split_into_chunks(
        &values,
        shape,
        chunk_dims,
        element_size,
        padding.pattern(element_size),
    )
    .unwrap();
    let dt = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    let expected: Vec<(Vec<u8>, u32)> = logical
        .iter()
        .zip(partial)
        .map(|(logical, &partial)| {
            let bytes = if stored_raw && partial {
                logical.clone()
            } else {
                filters::compress_chunk(
                    logical,
                    &pipeline,
                    ChunkContext::from_datatype(chunk_dims, &dt).unwrap(),
                )
                .unwrap()
            };
            (bytes, 0)
        })
        .collect();
    let flags = if stored_raw {
        ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS)
    } else {
        ChunkedLayoutFlags::NONE
    };
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: shape.len().try_into().unwrap(),
        dimensions: shape.to_vec(),
        max_dimensions: None,
    };
    let layout = DataLayout::Chunked {
        flags,
        chunk_dimensions: chunk_dims
            .iter()
            .copied()
            .chain([element_size.get() as u64])
            .collect(),
        index: ChunkIndexLayout::FixedArray { address: None },
    };
    let overwrite = ChunkedOverwrite {
        geometry: ChunkGeometry::new(
            &dataspace,
            ChunkedGeometry::new(&dt, &dataspace, &layout).unwrap(),
        ),
        flags,
        pipeline: Some(pipeline),
        padding,
        dt,
    };

    let chunks = overwrite.prepare_chunks(&values).unwrap();

    let stored: Vec<(Vec<u8>, u32)> = chunks
        .iter()
        .map(|chunk| (chunk.bytes().to_vec(), chunk.filter_mask()))
        .collect();
    assert_eq!(stored, expected);
}

#[rstest]
#[case::scalar(false, false)]
#[case::chunked_scalar(false, true)]
#[case::vlen(true, false)]
#[case::chunked_vlen(true, true)]
fn overwrite_preserves_the_on_disk_one_byte_order(#[case] vlen: bool, #[case] chunked: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("byte_order.h5");
    let datatype = if vlen {
        variable_length_string_type(
            big_endian_u8_type(),
            CharacterSet::Utf8,
            StringPadding::NullTerminate,
        )
    } else {
        big_endian_u8_type()
    };
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("data");
    if vlen {
        dataset.with_vlen_strings(&["before", "original"]);
    } else {
        dataset.with_u8_data(&[1, 2]);
    }
    dataset.with_dtype(datatype.clone());
    if chunked {
        dataset.with_chunks(&[1]).with_deflate(DeflateLevel::Six);
    }
    builder.write(&path).unwrap();
    {
        let file = crate::File::open_rw(&path).unwrap();
        file.dataset("data")
            .unwrap()
            .write_staged(|dataset| {
                if vlen {
                    dataset.with_vlen_strings(&["after", "replacement"]);
                } else {
                    dataset.with_u8_data(&[3, 4]);
                }
            })
            .unwrap();
        file.commit().unwrap();
    }
    let file = crate::File::open(&path).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.datatype().unwrap(), datatype);
    if vlen {
        assert_eq!(dataset.read_string().unwrap(), ["after", "replacement"]);
    } else {
        assert_eq!(dataset.read_u8().unwrap(), [3, 4]);
    }
}

#[cfg(feature = "zfp")]
#[test]
fn append_onto_a_lossy_partial_tail_is_rejected() {
    use crate::reader::File as PureFile;
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let committed: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let build = |path: &std::path::Path| {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_f64_data(&committed)
            .with_shape(&[6])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_zfp(ZfpRate::try_from(8.0).unwrap());
        b.write(path).unwrap();
    };
    let p = dir.path().join("inplace.h5");
    build(&p);
    let before = std::fs::read(&p).unwrap();
    {
        let f = crate::reader::File::open_rw(&p).unwrap();
        let err = f
            .dataset("d")
            .unwrap()
            .append(&[99.0f64, 98.0, 97.0])
            .expect_err("a lossy pipeline must not have its trailing chunk re-encoded");
        assert!(
            matches!(&err, Error::AppendInPlaceUnsupported(m) if m.contains("lossy")),
            "got: {err:?}"
        );
    }
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "the refusal wrote bytes"
    );
    assert_eq!(
        PureFile::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_f64()
            .unwrap(),
        committed
    );
    let p = dir.path().join("aligned.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_f64_data(&committed[..4])
            .with_shape(&[4])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_zfp(ZfpRate::try_from(8.0).unwrap());
        b.write(&p).unwrap();
    }
    {
        let f = crate::reader::File::open_rw(&p).unwrap();
        f.dataset("d").unwrap().append(&[9.5f64, 8.5, 7.5]).unwrap();
    }
    {
        let file = PureFile::open(&p).unwrap();
        let dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.shape().unwrap(), [7]);
        let values = dataset.read_f64().unwrap();
        assert_eq!(values.len(), 7);
        assert_eq!(values[..4], committed[..4], "an untouched chunk changed");
        assert!(
            values[4..]
                .iter()
                .zip([9.5, 8.5, 7.5])
                .all(|(actual, expected)| (actual - expected).abs() <= 1.5),
            "{values:?}"
        );
    }
    let p = dir.path().join("staged.h5");
    build(&p);
    let before = std::fs::read(&p).unwrap();
    {
        let f = crate::reader::File::open_rw(&p).unwrap();
        let err = f
            .dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_f64(&[99.0, 98.0, 97.0]);
            })
            .expect_err("a staged append must not re-encode the committed tail either");
        assert!(
            matches!(&err, Error::AppendUnsupported(m) if m.contains("lossy")),
            "got: {err:?}"
        );
        f.close().unwrap();
    }
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "the refusal wrote bytes"
    );
    let p = dir.path().join("buffered.h5");
    build(&p);
    {
        let f = crate::reader::File::open_rw(&p).unwrap();
        let mut ds = f.dataset("d").unwrap();
        let err = ds
            .buffered_appender()
            .expect_err("its first write would re-encode the committed tail");
        assert!(
            matches!(&err, Error::AppendInPlaceUnsupported(m) if m.contains("lossy")),
            "got: {err:?}"
        );
    }
}

#[test]
fn a_lossy_scale_offset_mode_is_rejected_and_a_lossless_one_is_not() {
    use crate::reader::File as PureFile;
    use crate::scaleoffset::ScaleOffset;
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();

    let p = dir.path().join("dscale.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_f64_data(&[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
            .with_shape(&[6])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_scale_offset(ScaleOffset::FloatDScale(1));
        b.write(&p).unwrap();
    }
    {
        let f = crate::reader::File::open_rw(&p).unwrap();
        let err = f
            .dataset("d")
            .unwrap()
            .append(&[99.0f64, 98.0, 97.0])
            .expect_err("float D-scale is lossy");
        assert!(
            matches!(&err, Error::AppendInPlaceUnsupported(m) if m.contains("lossy")),
            "got: {err:?}"
        );
    }

    let p = dir.path().join("int.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..6).collect::<Vec<_>>())
            .with_shape(&[6])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_scale_offset(ScaleOffset::Integer(0));
        b.write(&p).unwrap();
    }
    {
        let f = crate::reader::File::open_rw(&p).unwrap();
        let mut ds = f.dataset("d").unwrap();
        ds.append(&[6i32, 7, 8]).unwrap();
        ds.append(&[9i32]).unwrap();
    }
    assert_eq!(
        PureFile::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..10).collect::<Vec<_>>()
    );
}

#[test]
fn raw_appendable_recurses_into_aggregates() {
    let f64_with = |byte_order| {
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(8, byte_order, FloatingPointLayout::IEEE754_BINARY64)
                .unwrap(),
        )
    };
    let le_f64 = f64_with(DatatypeByteOrder::LittleEndian);
    let be_f64 = f64_with(DatatypeByteOrder::BigEndian);
    assert!(edit::datatype_is_raw_appendable(&le_f64));
    assert!(!edit::datatype_is_raw_appendable(&be_f64));
    assert!(edit::datatype_is_raw_appendable(&big_endian_u8_type()));
    let be_member = Datatype::Compound {
        size: 8,
        members: vec![crate::datatype::__private::compound_member(
            "x".into(),
            0,
            be_f64.clone(),
        )],
    };
    assert!(!edit::datatype_is_raw_appendable(&be_member));
    let le_member = Datatype::Compound {
        size: 8,
        members: vec![crate::datatype::__private::compound_member(
            "x".into(),
            0,
            le_f64.clone(),
        )],
    };
    assert!(edit::datatype_is_raw_appendable(&le_member));
    assert!(!edit::datatype_is_raw_appendable(&Datatype::Array {
        base_type: Box::new(be_f64.clone()),
        dimensions: vec![4],
    }));
    assert!(!edit::datatype_is_raw_appendable(
        &Datatype::VariableLength(VariableLengthType::sequence(le_f64.clone()))
    ));
    assert!(!edit::datatype_is_raw_appendable(&Datatype::Reference {
        size: 8,
        ref_type: crate::datatype::ReferenceType::Object,
    }));
}

#[test]
fn add_vlen_string_dataset_with_null_elements_via_edit_session() {
    use crate::type_builders::VlStringElement;
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen_null.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0]);
    b.write(&path).unwrap();

    let datatype = crate::type_builders::make_vlen_string_type(crate::datatype::CharacterSet::Utf8);
    let elements = vec![
        VlStringElement::Bytes(b"alpha".to_vec()),
        VlStringElement::Null,
        VlStringElement::Bytes(b"gamma".to_vec()),
    ];

    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let mut b = DatasetBuilder::new("labels");
        b.with_vlen_string_elements(datatype, &elements).unwrap();
        s.stage_created_dataset(&ObjectPathBuf::parse("labels"), b)
            .unwrap();
        s.commit().unwrap();
    }

    let file = crate::reader::File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    assert_eq!(
        ds.read_string().unwrap(),
        vec!["alpha".to_string(), String::new(), "gamma".to_string()]
    );
}

#[test]
fn bounded_multi_batch_append_commits_every_batch() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("multibatch.h5");
    common::build_appendable(&p, 5, 512);
    let total = 700_000i32;
    {
        let mut engine = common::open_bounded_session(&p);
        let addr = common::dataset_addr(&engine);
        let mut b = AppendBuilder::new();
        b.append_i32(&(5..total).collect::<Vec<_>>());
        engine
            .append_inplace_gathered(AppendTarget::Header(addr), &b, 4)
            .unwrap();
    }
    let got = crate::File::open(&p)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap();
    assert_eq!(got.len(), total as usize);
    assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32));
}

#[test]
fn only_a_bounded_session_batches_a_large_append() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("batching.h5");
    common::build_appendable(&p, 8, 4);

    let bounded_batch = {
        let mut engine = common::open_bounded_session(&p);
        engine
            .append_geometry(AppendTarget::Path(&ObjectPathBuf::parse("d")))
            .unwrap()
            .full_batch_elems
    };
    let mirror_batch = {
        let mut engine = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
        engine
            .append_geometry(AppendTarget::Path(&ObjectPathBuf::parse("d")))
            .unwrap()
            .full_batch_elems
    };

    assert_eq!(
        mirror_batch,
        u64::MAX,
        "a mirror session must take the whole append as one crash-atomic batch"
    );
    assert!(
        bounded_batch < u64::MAX,
        "a bounded session must cap a batch, got {bounded_batch}"
    );
    assert_eq!(
        bounded_batch % 4,
        0,
        "a batch must be a whole number of chunks"
    );
}

fn big_endian_u8_type() -> Datatype {
    Datatype::FixedPoint(FixedPointType::new(1, DatatypeByteOrder::BigEndian, false, 0, 8).unwrap())
}

fn array_type(base: Datatype, dimensions: Vec<u32>) -> Datatype {
    Datatype::Array {
        base_type: Box::new(base),
        dimensions,
    }
}

fn single_member_compound_type(base: Datatype, size: u32, name: &str, offset: u64) -> Datatype {
    Datatype::Compound {
        size,
        members: vec![crate::datatype::__private::compound_member(
            name.into(),
            offset,
            base,
        )],
    }
}

fn single_member_enumeration_type(base: Datatype, value: u8) -> Datatype {
    crate::EnumTypeBuilder::with_base(base)
        .u8_value("x", value)
        .build()
        .unwrap()
}

fn variable_length_string_type(
    base: Datatype,
    charset: CharacterSet,
    padding: StringPadding,
) -> Datatype {
    Datatype::VariableLength(VariableLengthType::String(
        VariableLengthStringType::with_base_type(padding, charset, base).unwrap(),
    ))
}

#[rstest]
#[case::encoded_chunks(true, "encoded chunk providers cannot be added in place yet")]
#[case::producer(false, "data producers cannot be added in place yet")]
fn in_place_creation_rejects_provider_sources(
    #[case] chunks: bool,
    #[case] expected: &str,
    #[values(false, true)] empty: bool,
) {
    let shape = [u64::from(!empty)];
    let mut builder = DatasetBuilder::new("provider");
    if chunks {
        builder.with_raw_chunks_lazy(
            type_builders::make_i32_type(),
            &shape,
            None,
            &[1],
            NonZeroUsize::new(4).unwrap(),
            ChunkedLayoutFlags::new(0),
            None,
            Vec::new(),
            Box::new(UnreadDatasetProvider),
        );
    } else {
        builder.with_produced_data(
            type_builders::make_i32_type(),
            &shape,
            shape[0] * 4,
            4,
            Box::new(UnreadDatasetProvider),
        );
    }
    let result = edit::dataset::flatten_dataset(builder, LinkName::new("provider").unwrap().into());
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("expected an unsupported provider source"),
    };
    let Error::EditUnsupported(reason) = error else {
        panic!("expected an unsupported provider source, got {error:?}");
    };
    assert_eq!(reason, expected);
}

struct UnreadDatasetProvider;

impl ChunkProvider for UnreadDatasetProvider {
    fn chunk_bytes(&self, _: usize, _: &mut Vec<u8>) -> Result<(), FormatError> {
        Err(FormatError::Internal(
            "in-place creation read the provider".into(),
        ))
    }
}

#[test]
fn in_place_creation_preserves_the_missing_data_cause() {
    let mut builder = DatasetBuilder::new("missing");
    builder
        .with_dtype(type_builders::make_i32_type())
        .with_shape(&[1]);
    let result = edit::dataset::flatten_dataset(builder, LinkName::new("missing").unwrap().into());
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("expected a missing data error"),
    };
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, FormatError::DatasetMissingData);
}

/// The string class code defined in "Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const STRING_DATATYPE_CLASS: u8 = 3;
