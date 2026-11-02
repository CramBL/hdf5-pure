use std::fmt::Debug;

use hdf5_pure::AttrValue;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FillTime;
use hdf5_pure::FilterId;
use hdf5_pure::FixedAscii;
use hdf5_pure::FormatError;
use hdf5_pure::H5Element;
use hdf5_pure::Layout;
use hdf5_pure::LibVer;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use rstest::rstest;

use super::SchemaCreation;

#[rstest]
#[case::numeric(-7_i32)]
#[case::byte_array([0_u8, 128, 255])]
#[case::ascii(FixedAscii::<4>::new("abc").unwrap())]
fn generic_typed_declarations_preserve_datatypes_shapes_and_fill_values<T>(#[case] fill: T)
where
    T: H5Element + PartialEq + Debug,
{
    let mut builder = FileBuilder::new();
    declare::<T>(builder.create_dataset("root")).with_fill_value(fill);
    let mut group = builder.create_group("nested");
    group
        .create_dataset("scalar")
        .with_unallocated_storage_for::<T>(&[])
        .with_fill_value(fill);
    builder.add_group(group.finish());
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let root = file.dataset("root").unwrap();
    assert_eq!(root.datatype().unwrap(), T::datatype().unwrap());
    assert_eq!(root.shape().unwrap(), [3]);
    assert_eq!(root.read::<T>().unwrap(), [fill; 3]);
    let scalar = file.dataset("nested/scalar").unwrap();
    assert_eq!(scalar.datatype().unwrap(), T::datatype().unwrap());
    assert_eq!(scalar.shape().unwrap(), Vec::<u64>::new());
    assert_eq!(scalar.read::<T>().unwrap(), [fill]);
}

#[rstest]
fn typed_declarations_preserve_independently_configured_settings(
    #[values(false, true)] declaration_first: bool,
    #[values(FillTime::IfSet, FillTime::Never)] fill_time: FillTime,
    #[values(
        SchemaCreation::Buffered,
        SchemaCreation::Streaming,
        SchemaCreation::Edit
    )]
    creation: SchemaCreation,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("typed.h5");
    creation.write(&path, FileSpaceSettings::default(), |schema| {
        if declaration_first {
            schema.with_unallocated_storage_for::<i32>(&[0, 4]);
        }
        schema
            .with_chunks(&[2, 2])
            .with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Fixed(4)])
            .with_fletcher32()
            .with_fill_value(-7_i32)
            .with_fill_time(fill_time)
            .set_attr("units", AttrValue::String("counts".into()));
        if !declaration_first {
            schema.with_unallocated_storage_for::<i32>(&[0, 4]);
        }
    });
    let file = File::open(&path).unwrap();
    let schema = file.dataset("schema").unwrap();
    assert_eq!(schema.datatype().unwrap(), i32::datatype().unwrap());
    assert_eq!(schema.shape().unwrap(), [0, 4]);
    assert_eq!(
        schema.maxshape().unwrap(),
        Some(vec![MaxExtent::Unlimited, MaxExtent::Fixed(4)])
    );
    assert_eq!(schema.chunk_shape().unwrap(), Some(vec![2, 2]));
    assert_eq!(schema.filters().unwrap(), [FilterId::Fletcher32]);
    assert_eq!(schema.fill_value::<i32>().unwrap(), Some(-7));
    assert_eq!(schema.fill_time().unwrap(), fill_time);
    assert_eq!(schema.attrs().unwrap().len(), 1);
    assert_eq!(
        schema.attrs().unwrap()["units"],
        AttrValue::String("counts".into())
    );
    assert_eq!(schema.read::<i32>().unwrap(), Vec::<i32>::new());
    assert_eq!(schema.chunks().unwrap(), []);
    drop(schema);
    file.close().unwrap();
}

#[rstest]
#[case::zero_array(declare::<[u8; 0]>, FormatError::ZeroSizedDatatype { class: ARRAY_CLASS })]
#[case::zero_ascii(declare::<FixedAscii<0>>, FormatError::ZeroFixedStringWidth)]
#[cfg_attr(target_pointer_width = "64", case::wide_array(
    declare::<[u8; u32::MAX as usize + 1]>, width_error()
))]
#[cfg_attr(target_pointer_width = "64", case::wide_ascii(
    declare::<FixedAscii<{ u32::MAX as usize + 1 }>>, width_error()
))]
fn invalid_typed_declarations_fail_before_whole_file_output(
    #[case] configure: fn(&mut DatasetBuilder) -> &mut DatasetBuilder,
    #[case] expected: FormatError,
    #[values(false, true)] previous_metadata: bool,
    #[values(Output::Buffered, Output::Sink, Output::ExistingPath, Output::NewPath)] output: Output,
) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("preserved")
        .with_data(&[1_u32, 2, 3]);
    let schema = builder.create_dataset("schema");
    if previous_metadata {
        schema
            .with_dtype(hdf5_pure::make_u8_type())
            .with_shape(&[1]);
    }
    configure(schema);
    let error = match output {
        Output::Buffered => builder.finish().unwrap_err(),
        Output::Sink => {
            let mut bytes = b"previous bytes".to_vec();
            let error = builder.finish_to(&mut bytes).unwrap_err();
            assert_eq!(bytes, b"previous bytes");
            error
        }
        Output::ExistingPath | Output::NewPath => {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("schema.h5");
            if matches!(output, Output::ExistingPath) {
                std::fs::write(&path, b"previous file").unwrap();
            }
            let error = builder.write(&path).unwrap_err();
            if matches!(output, Output::ExistingPath) {
                assert_eq!(std::fs::read(&path).unwrap(), b"previous file");
            } else {
                assert!(!path.exists());
            }
            error
        }
    };
    assert_format_error(error, expected);
}

#[rstest]
#[case::zero_array(declare::<[u8; 0]>, FormatError::ZeroSizedDatatype { class: ARRAY_CLASS })]
#[case::zero_ascii(declare::<FixedAscii<0>>, FormatError::ZeroFixedStringWidth)]
#[cfg_attr(target_pointer_width = "64", case::wide_array(
    declare::<[u8; u32::MAX as usize + 1]>, width_error()
))]
#[cfg_attr(target_pointer_width = "64", case::wide_ascii(
    declare::<FixedAscii<{ u32::MAX as usize + 1 }>>, width_error()
))]
fn invalid_typed_declarations_preserve_edit_state(
    #[case] configure: fn(&mut DatasetBuilder) -> &mut DatasetBuilder,
    #[case] expected: FormatError,
    #[values(false, true)] nested: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("typed.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("preserved")
        .with_data(&[1_u32, 2, 3]);
    let group = builder.create_group("nested");
    builder.add_group(group.finish());
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    let parent = if nested {
        file.group("nested").unwrap()
    } else {
        file.root()
    };
    let error = parent
        .create_dataset("schema", |schema| {
            configure(schema);
        })
        .unwrap_err();
    assert_format_error(error, expected);
    assert!(!file.has_staged_edits());
    assert_eq!(
        parent.datasets().unwrap(),
        if nested { vec![] } else { vec!["preserved"] }
    );
    drop(parent);
    file.commit().unwrap();
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("preserved").unwrap().read::<u32>().unwrap(),
        [1, 2, 3]
    );
    file.close().unwrap();
}

#[rstest]
#[case::before_data(true, &[1_i32, 2, 3])]
#[case::after_data(false, &[1_i32, 2, 3])]
#[case::before_empty_data(true, &[])]
#[case::after_empty_data(false, &[])]
fn typed_declarations_preserve_storage_conflicts(
    #[case] declaration_first: bool,
    #[case] data: &[i32],
    #[values(
        SchemaCreation::Buffered,
        SchemaCreation::Streaming,
        SchemaCreation::Edit
    )]
    creation: SchemaCreation,
) {
    let configure = |schema: &mut DatasetBuilder| {
        if declaration_first {
            schema.with_unallocated_storage_for::<i32>(&[3]);
        }
        schema.with_data(data);
        if !declaration_first {
            schema.with_unallocated_storage_for::<i32>(&[3]);
        }
    };
    let error = match creation {
        SchemaCreation::Edit => super::edit_schema(configure).unwrap_err(),
        SchemaCreation::Buffered | SchemaCreation::Streaming => {
            let mut builder = FileBuilder::new();
            configure(builder.create_dataset("schema"));
            if matches!(creation, SchemaCreation::Buffered) {
                builder.finish().unwrap_err()
            } else {
                builder.finish_to(&mut Vec::new()).unwrap_err()
            }
        }
    };
    assert_format_error(error, FormatError::ConflictingDatasetStorage);
}

#[test]
fn replacing_element_data_clears_a_staged_type_error() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_unallocated_storage_for::<[u8; 0]>(&[3])
        .with_data(&[1_u32, 2, 3]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let values = file.dataset("values").unwrap();
    assert_eq!(values.datatype().unwrap(), u32::datatype().unwrap());
    assert_eq!(values.read::<u32>().unwrap(), [1, 2, 3]);
}

#[rstest]
#[case::numeric(-7_i32)]
#[case::byte_array([0_u8, 128, 255])]
#[case::ascii(FixedAscii::<4>::new("abc").unwrap())]
fn typed_unallocated_storage_preserves_supported_file_versions<T>(
    #[case] fill: T,
    #[values(LibVer::V18, LibVer::V110)] version: LibVer,
    #[values(false, true)] streaming: bool,
) where
    T: H5Element + PartialEq + Debug,
{
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, version);
    builder
        .create_dataset("schema")
        .with_unallocated_storage_for::<T>(&[3])
        .with_fill_value(fill);
    let bytes = if streaming {
        let mut bytes = Vec::new();
        builder.finish_to(&mut bytes).unwrap();
        bytes
    } else {
        builder.finish().unwrap()
    };
    let file = File::from_bytes(bytes).unwrap();
    let schema = file.dataset("schema").unwrap();
    assert_eq!(file.libver_bound(), version);
    assert_eq!(schema.datatype().unwrap(), T::datatype().unwrap());
    assert_eq!(schema.shape().unwrap(), [3]);
    assert_eq!(schema.read::<T>().unwrap(), [fill; 3]);
    assert_eq!(
        schema.layout().unwrap(),
        Layout::Contiguous {
            address: None,
            size: 3 * u64::from(T::datatype().unwrap().type_size()),
        }
    );
}

fn declare<T: H5Element>(schema: &mut DatasetBuilder) -> &mut DatasetBuilder {
    schema.with_unallocated_storage_for::<T>(&[3])
}

fn assert_format_error(error: Error, expected: FormatError) {
    let Error::Format(actual) = error else {
        panic!("expected {expected:?}, got {error:?}");
    };
    assert_eq!(actual, expected);
}

#[cfg(target_pointer_width = "64")]
fn width_error() -> FormatError {
    FormatError::ValueTooLargeForRepresentation {
        value: u64::from(u32::MAX) + 1,
        target: "u32",
    }
}

#[derive(Clone, Copy)]
enum Output {
    Buffered,
    ExistingPath,
    NewPath,
    Sink,
}

const ARRAY_CLASS: u8 = 10;
