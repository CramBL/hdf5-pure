use hdf5_pure::{
    CharacterSet, DatasetBuilder, DatasetCreateProperties, Datatype, DatatypeByteOrder, Error,
    File, FileBuilder, FixedAscii, FixedPointType, FormatError, H5Element, StringPadding,
};
use rstest::rstest;
use test_util::datatype::Class;

#[derive(Clone, Copy)]
enum Fill {
    Array,
    Ascii,
    Byte,
    Float,
    Signed,
    Unsigned,
    ZeroArray,
}

impl Fill {
    fn set(self, builder: &mut DatasetBuilder, properties: bool) {
        match self {
            Self::Array => set_fill(builder, properties, [255_u8, 0, 1, 2]),
            Self::Ascii => set_fill(builder, properties, FixedAscii::<4>::new(b"a\0b").unwrap()),
            Self::Signed => set_fill(builder, properties, -7_i32),
            Self::Unsigned => set_fill(builder, properties, 7_u32),
            Self::Float => set_fill(builder, properties, 7_f32),
            Self::Byte => set_fill(builder, properties, 7_u8),
            Self::ZeroArray => set_fill(builder, properties, [0_u8; 0]),
        }
    }
}

#[rstest]
#[case::array_on_float(Fill::Array, f32::datatype().unwrap(), FormatError::FillValueDatatypeMismatch)]
#[case::ascii_on_array(Fill::Ascii, <[u8; 4]>::datatype().unwrap(), FormatError::FillValueDatatypeMismatch)]
#[case::array_rank(Fill::Array, array(vec![2, 2], u8::datatype().unwrap()), FormatError::FillValueDatatypeMismatch)]
#[case::array_base(Fill::Array, array(vec![4], i8::datatype().unwrap()), FormatError::FillValueDatatypeMismatch)]
#[case::signed_on_unsigned(Fill::Signed, u32::datatype().unwrap(), FormatError::FillValueDatatypeMismatch)]
#[case::unsigned_on_signed(Fill::Unsigned, i32::datatype().unwrap(), FormatError::FillValueDatatypeMismatch)]
#[case::float_on_integer(Fill::Float, i32::datatype().unwrap(), FormatError::FillValueDatatypeMismatch)]
#[case::numeric_byte_order(Fill::Signed, Datatype::FixedPoint(FixedPointType::new(4, DatatypeByteOrder::BigEndian, true, 0, 32).unwrap()), FormatError::FillValueDatatypeMismatch)]
#[case::numeric_precision(Fill::Signed, Datatype::FixedPoint(FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 24).unwrap()), FormatError::FillValueDatatypeMismatch)]
#[case::numeric_offset(Fill::Signed, Datatype::FixedPoint(FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 8, 24).unwrap()), FormatError::FillValueDatatypeMismatch)]
#[case::array_precision(Fill::Array, array(vec![4], Datatype::FixedPoint(FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 7).unwrap())), FormatError::FillValueDatatypeMismatch)]
#[case::array_dimensions(Fill::Array, array(vec![2], u16::datatype().unwrap()), FormatError::FillValueDatatypeMismatch)]
#[case::space_padding(
    Fill::Ascii,
    string(StringPadding::SpacePad, CharacterSet::Ascii),
    FormatError::FillValueDatatypeMismatch
)]
#[case::terminated_padding(
    Fill::Ascii,
    string(StringPadding::NullTerminate, CharacterSet::Ascii),
    FormatError::FillValueDatatypeMismatch
)]
#[case::utf8(
    Fill::Ascii,
    string(StringPadding::NullPad, CharacterSet::Utf8),
    FormatError::FillValueDatatypeMismatch
)]
#[case::different_width(Fill::Byte, i32::datatype().unwrap(), FormatError::FillValueSizeMismatch { expected: 4, actual: 1 })]
#[case::zero_array(Fill::ZeroArray, u8::datatype().unwrap(), FormatError::ZeroSizedDatatype { class: Class::ARRAY.0 })]
fn incompatible_fill_is_rejected_before_output_or_staging(
    #[case] fill: Fill,
    #[case] datatype: Datatype,
    #[case] expected: FormatError,
    #[values(false, true)] properties: bool,
    #[values(false, true)] fill_first: bool,
    #[values(false, true)] unallocated: bool,
    #[values(&[][..], &[0][..], &[2][..])] dims: &[u64],
) {
    let configure = |builder: &mut DatasetBuilder| {
        if fill_first {
            fill.set(builder, properties);
        }
        if unallocated {
            builder.with_unallocated_storage(datatype.clone(), dims);
        } else {
            let count = dims.iter().product::<u64>();
            builder
                .with_raw_data(
                    datatype.clone(),
                    vec![0; datatype.type_size() as usize * count as usize],
                    count,
                )
                .with_shape(dims);
        }
        if !fill_first {
            fill.set(builder, properties);
        }
    };
    let mut builder = FileBuilder::new();
    configure(builder.create_dataset("rejected"));
    let mut sink = Vec::new();
    let error = builder.finish_to(&mut sink).unwrap_err();
    assert_eq!(error_format(error), expected);
    assert_eq!(sink, Vec::<u8>::new());

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("seed").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let error = file
        .root()
        .create_dataset("rejected", configure)
        .unwrap_err();
    assert_eq!(error_format(error), expected);
    assert_eq!(file.root().datasets().unwrap(), ["seed"]);
    assert!(!file.has_staged_edits());
    file.commit().unwrap();
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[derive(Clone, Copy)]
enum Replacement {
    Direct,
    Properties,
    Unset,
}

#[rstest]
#[case::properties_replace_bad_fill(Replacement::Properties)]
#[case::direct_replaces_properties(Replacement::Direct)]
#[case::properties_remove_fill(Replacement::Unset)]
fn replacement_uses_only_the_final_fill(#[case] replacement: Replacement) {
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("values");
    dataset.with_unallocated_storage_for::<i32>(&[2]);
    match replacement {
        Replacement::Properties => {
            dataset
                .with_fill_value([1_u8; 4])
                .with_create_properties(DatasetCreateProperties::new().with_fill_value(-7_i32));
        }
        Replacement::Direct => {
            dataset
                .with_create_properties(DatasetCreateProperties::new().with_fill_value([1_u8; 4]))
                .with_fill_value(-7_i32);
        }
        Replacement::Unset => {
            dataset
                .with_fill_value([1_u8; 4])
                .with_create_properties(DatasetCreateProperties::new());
        }
    }
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("values").unwrap();
    let fill = (!matches!(replacement, Replacement::Unset)).then_some(-7_i32);
    assert_eq!(dataset.fill_value::<i32>().unwrap(), fill);
    assert_eq!(dataset.read_i32().unwrap(), [fill.unwrap_or(0); 2]);
}

#[test]
fn byte_array_fill_accepts_one_byte_base_order() {
    let datatype = array(
        vec![4],
        Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::BigEndian, false, 0, 8).unwrap(),
        ),
    );
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("bytes")
        .with_unallocated_storage(datatype.clone(), &[2])
        .with_fill_value([255_u8, 0, 1, 2]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("bytes").unwrap();
    assert_eq!(dataset.datatype().unwrap(), datatype);
    assert_eq!(dataset.read::<[u8; 4]>().unwrap(), [[255, 0, 1, 2]; 2]);
}

fn set_fill<T: H5Element>(builder: &mut DatasetBuilder, properties: bool, value: T) {
    if properties {
        builder.with_create_properties(DatasetCreateProperties::new().with_fill_value(value));
    } else {
        builder.with_fill_value(value);
    }
}

fn array(dimensions: Vec<u32>, base_type: Datatype) -> Datatype {
    Datatype::Array {
        dimensions,
        base_type: Box::new(base_type),
    }
}

fn string(padding: StringPadding, charset: CharacterSet) -> Datatype {
    Datatype::String {
        size: 4,
        padding,
        charset,
    }
}

fn error_format(error: Error) -> FormatError {
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    error
}
