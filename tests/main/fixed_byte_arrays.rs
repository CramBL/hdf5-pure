use hdf5_pure::{
    CharacterSet, Datatype, DatatypeByteOrder, Error, File, FileBuilder, FixedPointType,
    FormatError, H5Element, StringPadding,
};
use rstest::rstest;

use test_util_hdf5::dataset::AppendMode;

#[rstest]
#[case::contiguous_width_one(vec![[0], [255], [128]], false)]
#[case::chunked_width_one(vec![[0], [255], [128]], true)]
#[case::contiguous_width_three(vec![[0, 255, 128], [17, 23, 42], [1, 2, 3], [4, 5, 6]], false)]
#[case::chunked_width_three(vec![[0, 255, 128], [17, 23, 42], [1, 2, 3], [4, 5, 6]], true)]
#[case::contiguous_empty(Vec::<[u8; 7]>::new(), false)]
#[case::chunked_empty(Vec::<[u8; 7]>::new(), true)]
fn typed_byte_arrays_preserve_elements_and_datatype<const N: usize>(
    #[case] values: Vec<[u8; N]>,
    #[case] chunked: bool,
) {
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("values");
    dataset.with_data(&values).with_fill_value([42; N]);
    if chunked {
        dataset.with_chunks(&[3]);
    }
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape().unwrap(), [values.len() as u64]);
    assert_eq!(dataset.datatype().unwrap(), array_type(N as u32));
    assert_eq!(dataset.read::<[u8; N]>().unwrap(), values);
    assert_eq!(dataset.fill_value::<[u8; N]>().unwrap(), Some([42; N]));
    assert_eq!(dataset.read_raw().unwrap(), values.as_flattened());
}

#[rstest]
#[case::scalar(hdf5_pure::make_u8_type())]
#[case::wrong_width(array_type(3))]
#[case::ascii(Datatype::String { size: 2, padding: StringPadding::NullPad, charset: CharacterSet::Ascii })]
#[case::opaque(Datatype::Opaque { size: 2, tag: Vec::new() })]
#[case::wrong_rank(Datatype::Array {
    base_type: Box::new(hdf5_pure::make_u8_type()), dimensions: vec![1, 2]
})]
#[case::signed(Datatype::Array {
    base_type: Box::new(hdf5_pure::make_i8_type()), dimensions: vec![2]
})]
#[case::wide_base(Datatype::Array {
    base_type: Box::new(hdf5_pure::make_u16_type()), dimensions: vec![2]
})]
#[case::narrow_precision(Datatype::Array {
    base_type: Box::new(Datatype::FixedPoint(
        FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 7).unwrap()
    )), dimensions: vec![2]
})]
fn typed_byte_arrays_reject_incompatible_datatypes(
    #[case] datatype: Datatype,
    #[values(0, 2)] count: usize,
) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("values").with_raw_data(
        datatype.clone(),
        vec![0; datatype.type_size() as usize * count],
        count as u64,
    );
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_type_mismatch(
        file.dataset("values")
            .unwrap()
            .read::<[u8; 2]>()
            .unwrap_err(),
        &datatype,
    );
    assert_type_mismatch(
        <[u8; 2]>::convert_raw(&[], &datatype).unwrap_err(),
        &datatype,
    );
}

#[test]
fn typed_byte_arrays_preserve_explicit_shape_and_clear_previous_staging() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_shape(&[2, 1])
        .with_vlen_strings(&["old", "data"])
        .with_data(&[[1, 2], [3, 4]]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape().unwrap(), [2, 1]);
    assert_eq!(dataset.datatype().unwrap(), array_type(2));
    assert_eq!(dataset.read::<[u8; 2]>().unwrap(), [[1, 2], [3, 4]]);
}

#[test]
fn typed_byte_arrays_reject_incomplete_raw_elements() {
    let error = <[u8; 2]>::convert_raw(&[1, 2, 3], &array_type(2)).unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::DataSizeMismatch {
                expected: 2,
                actual: 3
            })
        ),
        "{error:?}"
    );
}

#[test]
fn typed_byte_arrays_reject_zero_width() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("values").with_data(&[[]]);
    let error = builder.finish().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::ZeroSizedDatatype { class: ARRAY_CLASS })
        ),
        "{error:?}"
    );
    let error = <[u8; 0]>::convert_raw(&[], &array_type(1)).unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::ZeroSizedDatatype { class: ARRAY_CLASS })
        ),
        "{error:?}"
    );
}

fn array_type(width: u32) -> Datatype {
    Datatype::Array {
        base_type: Box::new(hdf5_pure::make_u8_type()),
        dimensions: vec![width],
    }
}

fn assert_type_mismatch(error: Error, datatype: &Datatype) {
    let Error::Format(FormatError::TypeMismatch { expected, actual }) = error else {
        panic!("expected TypeMismatch, got {error:?}");
    };
    assert_eq!(
        expected,
        "one-dimensional array of full-width unsigned bytes with matching width"
    );
    assert_eq!(
        actual,
        match datatype {
            Datatype::Array { .. } => "Array",
            Datatype::String { .. } => "String",
            Datatype::Opaque { .. } => "Opaque",
            _ => "FixedPoint",
        }
    );
}

#[test]
fn typed_byte_arrays_read_big_endian_byte_elements_without_conversion() {
    let datatype = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::BigEndian, false, 0, 8).unwrap(),
        )),
        dimensions: vec![2],
    };
    assert_eq!(
        <[u8; 2]>::convert_raw(&[128, 255, 0, 42], &datatype).unwrap(),
        [[128, 255], [0, 42]]
    );
}

const ARRAY_CLASS: u8 = 10;

#[rstest]
#[case::zero_width(&[[]], FormatError::ZeroSizedDatatype { class: ARRAY_CLASS })]
#[cfg_attr(target_pointer_width = "64", case::width_overflow(
    OVERFLOW_ELEMENTS,
    FormatError::ValueTooLargeForRepresentation { value: OVERFLOW_WIDTH as u64, target: "u32" }
))]
fn invalid_byte_array_appends_preserve_target_state<const N: usize>(
    #[case] values: &[[u8; N]],
    #[case] expected: FormatError,
    #[values(
        AppendMode::Immediate,
        AppendMode::Staged,
        AppendMode::Buffered,
        AppendMode::OwnedBuffered
    )]
    mode: AppendMode,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_u8_data(&[11, 22])
        .with_chunks(&[4])
        .with_maxshape(&[hdf5_pure::MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("values").unwrap();
        let error = match mode {
            AppendMode::Immediate => dataset.append(values).unwrap_err(),
            AppendMode::Staged => dataset
                .append_staged(|append| {
                    append.append(values).append_u8(&[33]);
                })
                .unwrap_err(),
            AppendMode::Buffered => {
                let mut appender = dataset.buffered_appender().unwrap();
                let error = appender.append(values).unwrap_err();
                assert_eq!(appender.buffered_elements(), 0);
                assert!(appender.unwritten().is_empty());
                appender.finish().unwrap();
                error
            }
            AppendMode::OwnedBuffered => {
                let mut appender = dataset.clone().into_buffered_appender().unwrap();
                let error = appender.append(values).unwrap_err();
                assert_eq!(appender.buffered_elements(), 0);
                assert!(appender.unwritten().is_empty());
                appender.finish().unwrap();
                error
            }
        };
        assert_format_error(error, expected);
        assert!(!file.has_staged_edits());
        assert_eq!(dataset.shape().unwrap(), [2]);
        assert_eq!(dataset.read::<u8>().unwrap(), [11, 22]);
        file.commit().unwrap();
    }
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read::<u8>().unwrap(),
        [11, 22]
    );
}

#[test]
fn zero_width_appends_preserve_a_staged_creation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    FileBuilder::new().write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file
            .root()
            .create_dataset("values", |builder| {
                builder.with_u8_data(&[11, 22]);
            })
            .unwrap();
        let error = dataset
            .append_staged(|append| {
                append.append(&[[]]).append_u8(&[33]);
            })
            .unwrap_err();
        assert_format_error(error, FormatError::ZeroSizedDatatype { class: ARRAY_CLASS });
        file.commit().unwrap();
        assert_eq!(dataset.shape().unwrap(), [2]);
        assert_eq!(dataset.read::<u8>().unwrap(), [11, 22]);
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read::<u8>().unwrap(),
        [11, 22]
    );
}

#[test]
fn zero_width_appends_preserve_buffered_input_and_allow_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_u8_data(&[11, 22])
        .with_chunks(&[4])
        .with_maxshape(&[hdf5_pure::MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("values").unwrap();
    let mut appender = dataset.buffered_appender().unwrap();
    appender.append(&[33u8]).unwrap();
    let error = appender.append(&[[]]).unwrap_err();
    assert_format_error(error, FormatError::ZeroSizedDatatype { class: ARRAY_CLASS });
    assert_eq!(appender.buffered_elements(), 1);
    assert_eq!(appender.unwritten(), [33]);
    appender.append(&[44u8]).unwrap();
    appender.finish().unwrap();
    assert_eq!(dataset.read::<u8>().unwrap(), [11, 22, 33, 44]);
}

fn assert_format_error(error: Error, expected: FormatError) {
    let Error::Format(actual) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(actual, expected);
}

#[cfg(target_pointer_width = "64")]
const OVERFLOW_WIDTH: usize = u32::MAX as usize + 1;

#[cfg(target_pointer_width = "64")]
const OVERFLOW_ELEMENTS: &[[u8; OVERFLOW_WIDTH]] = &[];
