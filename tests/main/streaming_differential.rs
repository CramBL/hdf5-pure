use std::collections::HashMap;

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError, SuperblockStatusFlags};
use rstest::rstest;
use test_util::attribute;
use test_util::image::Image as FileImage;
use test_util::object_header::{Message, MessageType, v2};
use test_util::superblock;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;

use self::oracle::{Comparison, Operation};

#[path = "../../fuzz/streaming_differential.rs"]
mod oracle;

#[rstest]
#[case::legacy_dataset(Image::Dataset, 1, 2)]
#[case::legacy_groups(Image::Groups, 3, 4)]
#[case::depth_limit(Image::DeepGroups, 3, 0)]
#[case::children_limit(Image::ManyChildren, 9, 16)]
#[case::nan_attributes(Image::NanAttributes, 1, 2)]
fn matching_files_execute_bounded_metadata_operations(
    #[case] image: Image,
    #[case] groups: usize,
    #[case] datasets: usize,
) {
    assert_eq!(
        oracle::compare(&image.build()),
        Comparison::Opened { groups, datasets }
    );
}

#[test]
fn invalid_signatures_are_rejected_by_both_path_opens() {
    assert_eq!(oracle::compare(b"not an HDF5 file"), Comparison::Rejected);
}

#[test]
fn path_opens_apply_the_same_writer_mark_policy() {
    let header = Superblock::new(Widths::EIGHT)
        .version(superblock::VERSION_3)
        .consistency_flags(u8::try_from(SuperblockStatusFlags::write_marked().bits()).unwrap());
    let image = header.eof_address(header.build().len() as u64).build();
    let snapshot = File::from_bytes(image.clone()).unwrap();
    assert_eq!(
        snapshot.superblock().consistency_flags(),
        SuperblockStatusFlags::write_marked()
    );
    assert_eq!(oracle::compare(&image), Comparison::Rejected);
}

#[rstest]
#[case::equal_successes(Ok(7), Ok(7), Some((7, 7)))]
#[case::matching_rejections(
    Err(Error::NotADataset("data".into())),
    Err(Error::NotADataset("data".into())),
    None,
)]
#[case::buffer_and_source_eof(
    Err(FormatError::UnexpectedEof { expected: 8, available: 4 }.into()),
    Err(FormatError::UnexpectedSourceEof { expected: 24, available: 20 }.into()),
    None,
)]
#[case::buffer_eof_and_source_range_overflow(
    Err(FormatError::UnexpectedEof { expected: usize::MAX - 1, available: 48 }.into()),
    Err(FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 4 }.into()),
    None,
)]
fn outcome_pairs_preserve_successes_or_match_failures(
    #[case] left: Result<u8, Error>,
    #[case] right: Result<u8, Error>,
    #[case] expected: Option<(u8, u8)>,
) {
    assert_eq!(oracle::matched(Operation::Shape, left, right), expected);
}

#[rstest]
#[case::buffered_rejection(Disagreement::Buffered)]
#[case::streaming_rejection(Disagreement::Streaming)]
#[case::different_error_classes(Disagreement::Class)]
#[case::different_error_fields(Disagreement::Fields)]
#[case::different_error_paths(Disagreement::Paths)]
#[case::insufficient_region_and_checksum_error(Disagreement::RegionClass)]
#[case::arithmetic_error_and_truncation(Disagreement::Arithmetic)]
#[should_panic(expected = "DatasetNames:")]
fn different_outcomes_fail_the_comparison(#[case] disagreement: Disagreement) {
    let (left, right) = disagreement.results();
    oracle::matched(Operation::DatasetNames, left, right);
}

#[rstest]
#[case::different_group_names(Operation::GroupNames)]
#[case::different_dataset_shapes(Operation::Shape)]
#[case::different_dataset_types(Operation::Datatype)]
#[should_panic(expected = "different values")]
fn different_success_values_fail_the_comparison(#[case] operation: Operation) {
    oracle::compare_value(operation, Ok(vec![1_u8]), Ok(vec![1, 2]));
}

#[rstest]
#[case::missing_attribute(AttributeDifference::Missing)]
#[case::changed_value(AttributeDifference::Value)]
#[case::extra_array_element(AttributeDifference::Length)]
#[case::changed_scalar_type(AttributeDifference::Type)]
#[should_panic(expected = "GroupAttributes: different values")]
fn different_attributes_fail_the_comparison(#[case] difference: AttributeDifference) {
    let left = HashMap::from([("values".into(), AttrValue::F64Array(vec![f64::NAN, 1.0]))]);
    oracle::compare_attrs(
        Operation::GroupAttributes,
        Ok(left),
        Ok(difference.values()),
    );
}

#[test]
fn nan_attributes_compare_independently_of_map_order() {
    let left = HashMap::from([
        ("nan".into(), AttrValue::F64(f64::NAN)),
        (
            "values".into(),
            AttrValue::F32Array(vec![f32::NAN, -0.0, f32::INFINITY]),
        ),
    ]);
    let right = HashMap::from([
        (
            "values".into(),
            AttrValue::F32Array(vec![f32::NAN, -0.0, f32::INFINITY]),
        ),
        ("nan".into(), AttrValue::F64(f64::NAN)),
    ]);
    oracle::compare_attrs(Operation::GroupAttributes, Ok(left), Ok(right));
}

#[rstest]
#[case::filesystem(InfrastructureFailure::Io)]
#[case::source(InfrastructureFailure::Source)]
#[should_panic(expected = "GroupNames: infrastructure failure")]
fn infrastructure_errors_are_not_matching_parser_rejections(
    #[case] failure: InfrastructureFailure,
) {
    oracle::matched::<()>(
        Operation::GroupNames,
        Err(failure.error()),
        Err(failure.error()),
    );
}

#[test]
#[should_panic(expected = "Shape: internal failure")]
fn matching_internal_errors_fail_the_comparison() {
    oracle::matched::<()>(
        Operation::Shape,
        Err(FormatError::Internal("invariant violated".into()).into()),
        Err(FormatError::Internal("invariant violated".into()).into()),
    );
}

#[rstest]
#[case::past_extent(1024, 1028, FormatError::UnexpectedSourceEof { expected: 1028, available: 48 })]
#[cfg_attr(target_pointer_width = "64", case::overflowing_range(u64::MAX - 1, usize::MAX, FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 4 }))]
fn root_header_range_rejections_preserve_backend_diagnostics(
    #[case] address: u64,
    #[case] buffered_expected: usize,
    #[case] streaming_error: FormatError,
) {
    let header = Superblock::new(Widths::EIGHT).root_header_address(address);
    let image = header.eof_address(header.build().len() as u64).build();
    let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    std::fs::write(&path, &image).unwrap();
    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    let left = buffered.root().attrs().unwrap_err();
    let right = streaming.root().attrs().unwrap_err();
    assert!(
        matches!(&left, Error::Format(FormatError::UnexpectedEof { expected, available })
            if *expected == buffered_expected && *available == image.len()),
        "{left:?}",
    );
    let Error::Format(actual) = &right else {
        panic!("expected a format error, got {right:?}");
    };
    assert_eq!(actual, &streaming_error);
    assert_eq!(
        oracle::matched::<()>(Operation::GroupAttributes, Err(left), Err(right)),
        None,
    );
    drop(streaming);
    drop(buffered);
    assert_eq!(
        oracle::compare(&image),
        Comparison::Opened {
            groups: 1,
            datasets: 0
        },
    );
}

#[rstest]
#[case::truncated_prefix(2072)]
#[case::at_extent(2080)]
#[case::ci_heap_address(281_474_976_710_656)]
fn group_attribute_heap_range_rejections_preserve_backend_diagnostics(#[case] heap_address: u64) {
    let superblock = Superblock::new(Widths::EIGHT);
    let header_at = superblock.build().len();
    let mut image = FileImage::starting_with(
        &superblock
            .root_header_address(header_at as u64)
            .eof_address(2080)
            .build(),
    );
    image.place(
        header_at,
        &v2::Header::new()
            .message(Message::new(
                MessageType::ATTRIBUTE_INFO,
                &attribute::info(Some(heap_address), None, Widths::EIGHT),
            ))
            .build(),
    );
    image.place(2079, &[0]);
    let image = image.build();
    let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    std::fs::write(&path, &image).unwrap();
    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    let left = buffered.root().attrs().unwrap_err();
    let right = streaming.root().attrs().unwrap_err();
    let Error::Format(actual) = &left else {
        panic!("expected a format error, got {left:?}");
    };
    assert_eq!(
        actual,
        &FormatError::InsufficientRegion {
            required_start: heap_address,
            required_length: 9,
            provided_start: 0,
            provided_length: 2080,
        },
    );
    let Error::Format(actual) = &right else {
        panic!("expected a format error, got {right:?}");
    };
    assert_eq!(
        actual,
        &FormatError::UnexpectedSourceEof {
            expected: heap_address + 9,
            available: 2080,
        },
    );
    oracle::compare_attrs(Operation::GroupAttributes, Err(left), Err(right));
    drop(streaming);
    drop(buffered);
    assert_eq!(
        oracle::compare(&image),
        Comparison::Opened {
            groups: 1,
            datasets: 0
        },
    );
}

enum Image {
    Dataset,
    DeepGroups,
    Groups,
    ManyChildren,
    NanAttributes,
}

impl Image {
    fn build(self) -> Vec<u8> {
        match self {
            Self::Dataset => test_util::legacy::simple_dataset(),
            Self::DeepGroups => test_util::legacy::nested_groups(),
            Self::Groups => test_util::legacy::two_groups(),
            Self::ManyChildren => {
                let mut builder = FileBuilder::new();
                for index in 0..9 {
                    builder
                        .create_dataset(&format!("data_{index}"))
                        .with_i32_data(&[index]);
                    let group = builder.create_group(&format!("group_{index}"));
                    builder.add_group(group.finish());
                }
                builder.finish().unwrap()
            }
            Self::NanAttributes => {
                let mut builder = FileBuilder::new();
                builder.set_attr("nan", AttrValue::F64(f64::NAN));
                builder.set_attr(
                    "values",
                    AttrValue::F32Array(vec![f32::NAN, -0.0, f32::INFINITY]),
                );
                builder
                    .create_dataset("data")
                    .with_i32_data(&[1, 2, 3])
                    .set_attr("nan", AttrValue::F64Array(vec![f64::NAN, 1.0]));
                builder.finish().unwrap()
            }
        }
    }
}

enum Disagreement {
    Arithmetic,
    Buffered,
    Class,
    Fields,
    Paths,
    RegionClass,
    Streaming,
}

impl Disagreement {
    fn results(self) -> (Result<u8, Error>, Result<u8, Error>) {
        match self {
            Self::Arithmetic => (
                Err(FormatError::UnexpectedEof {
                    expected: 8,
                    available: 4,
                }
                .into()),
                Err(FormatError::OffsetOverflow {
                    offset: 8,
                    length: 4,
                }
                .into()),
            ),
            Self::Buffered => (Err(Error::NotADataset("data".into())), Ok(7)),
            Self::Streaming => (Ok(7), Err(Error::NotADataset("data".into()))),
            Self::Class => (
                Err(FormatError::UnexpectedEof {
                    expected: 8,
                    available: 4,
                }
                .into()),
                Err(FormatError::ChecksumMismatch {
                    expected: 1,
                    computed: 2,
                }
                .into()),
            ),
            Self::Fields => (
                Err(FormatError::ChecksumMismatch {
                    expected: 1,
                    computed: 2,
                }
                .into()),
                Err(FormatError::ChecksumMismatch {
                    expected: 1,
                    computed: 3,
                }
                .into()),
            ),
            Self::Paths => (
                Err(Error::NotADataset("first".into())),
                Err(Error::NotADataset("second".into())),
            ),
            Self::RegionClass => (
                Err(FormatError::InsufficientRegion {
                    required_start: 281_474_976_710_656,
                    required_length: 9,
                    provided_start: 0,
                    provided_length: 2080,
                }
                .into()),
                Err(FormatError::ChecksumMismatch {
                    expected: 1,
                    computed: 2,
                }
                .into()),
            ),
        }
    }
}

enum AttributeDifference {
    Length,
    Missing,
    Type,
    Value,
}

impl AttributeDifference {
    fn values(self) -> HashMap<String, AttrValue> {
        match self {
            Self::Missing => HashMap::new(),
            Self::Length => HashMap::from([(
                "values".into(),
                AttrValue::F64Array(vec![f64::NAN, 1.0, 2.0]),
            )]),
            Self::Type => {
                HashMap::from([("values".into(), AttrValue::F32Array(vec![f32::NAN, 1.0]))])
            }
            Self::Value => {
                HashMap::from([("values".into(), AttrValue::F64Array(vec![f64::NAN, 2.0]))])
            }
        }
    }
}

enum InfrastructureFailure {
    Io,
    Source,
}

impl InfrastructureFailure {
    fn error(&self) -> Error {
        match self {
            Self::Io => Error::Io(std::io::Error::other("fixture I/O failed")),
            Self::Source => FormatError::Source("fixture I/O failed".into()).into(),
        }
    }
}
