use std::collections::HashMap;

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError, SuperblockStatusFlags};
use rstest::rstest;
use test_util::attribute;
use test_util::dataspace;
use test_util::datatype;
use test_util::dense_attributes::ChecksumUpdate;
use test_util::dense_attributes::NodeCorruption;
use test_util::image::Image as FileImage;
use test_util::object_header::{Message, MessageType, v2};
use test_util::superblock;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::reader::ReadBackend;

use self::oracle::{Comparison, Operation};

#[path = "../../fuzz/streaming_differential.rs"]
mod oracle;

#[rstest]
#[case::legacy_dataset(Image::Dataset, 1, 2)]
#[case::dense_attributes(Image::DenseAttributes, 1, 2)]
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

#[rstest]
#[case::version_one_truncated(AttributeCase {
    version: AttributeVersion::One,
    fields: AttributeFields::TruncatedDataspace,
})]
#[case::version_one_invalid_type(AttributeCase {
    version: AttributeVersion::One,
    fields: AttributeFields::InvalidDatatype,
})]
#[case::version_one_valid(AttributeCase {
    version: AttributeVersion::One,
    fields: AttributeFields::Valid,
})]
#[case::version_two_truncated(AttributeCase {
    version: AttributeVersion::Two,
    fields: AttributeFields::TruncatedDataspace,
})]
#[case::version_two_invalid_type(AttributeCase {
    version: AttributeVersion::Two,
    fields: AttributeFields::InvalidDatatype,
})]
#[case::version_two_valid(AttributeCase {
    version: AttributeVersion::Two,
    fields: AttributeFields::Valid,
})]
#[case::version_three_truncated(AttributeCase {
    version: AttributeVersion::Three,
    fields: AttributeFields::TruncatedDataspace,
})]
#[case::version_three_invalid_type(AttributeCase {
    version: AttributeVersion::Three,
    fields: AttributeFields::InvalidDatatype,
})]
#[case::version_three_valid(AttributeCase {
    version: AttributeVersion::Three,
    fields: AttributeFields::Valid,
})]
fn attribute_field_truncation_precedes_datatype_decoding(
    #[case] case: AttributeCase,
    #[values(AttributeStorage::Compact, AttributeStorage::Dense)] storage: AttributeStorage,
    #[values(
        Check::Read(ReadBackend::Buffered),
        Check::Read(ReadBackend::Bytes),
        Check::Read(ReadBackend::Source),
        Check::Read(ReadBackend::Streaming),
        Check::ComparePathOpens
    )]
    check: Check,
) {
    let body = case.body();
    let image = storage.image(&body);
    match check {
        Check::ComparePathOpens => assert_eq!(
            oracle::compare(&image),
            Comparison::Opened {
                groups: 1,
                datasets: 0
            },
        ),
        Check::Read(backend) => {
            let file = backend.open_image(image).unwrap();
            let actual = file.root().attrs().map_err(|error| match error {
                Error::Format(error) => error,
                error => panic!("expected an attribute format error, got {error:?}"),
            });
            assert_eq!(actual, case.expected(body.len()));
            file.close().unwrap();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct AttributeCase {
    version: AttributeVersion,
    fields: AttributeFields,
}

impl AttributeCase {
    fn body(self) -> Vec<u8> {
        let Self { version, fields } = self;
        let mut datatype = datatype::header(datatype::Class::FIXED_POINT, 1, [0; 3], 8);
        datatype.extend_from_slice(&0_u16.to_le_bytes());
        datatype.extend_from_slice(&fields.precision().to_le_bytes());
        let scalar = dataspace::scalar();
        let value = 42_u64.to_le_bytes();
        let body = attribute::Attribute::new("value", &datatype, &scalar, &value);
        let mut body = match version {
            AttributeVersion::One => body,
            AttributeVersion::Two => body.flags(attribute::Flags::NONE),
            AttributeVersion::Three => body.character_set(0),
        }
        .build();
        let dataspace_at = version.dataspace_offset();
        assert_eq!(&body[dataspace_at..dataspace_at + scalar.len()], scalar);
        if matches!(fields, AttributeFields::TruncatedDataspace) {
            body[6..8].copy_from_slice(&4096_u16.to_le_bytes());
        }
        body
    }

    fn expected(self, body_len: usize) -> Result<HashMap<String, AttrValue>, FormatError> {
        let Self { version, fields } = self;
        match fields {
            AttributeFields::InvalidDatatype => Err(FormatError::InvalidFixedPointRange {
                size: 8,
                bit_offset: 0,
                bit_precision: fields.precision(),
            }),
            AttributeFields::TruncatedDataspace => Err(FormatError::UnexpectedEof {
                expected: version.dataspace_offset() + 4096,
                available: body_len,
            }),
            AttributeFields::Valid => Ok(HashMap::from([("value".into(), AttrValue::U64(42))])),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AttributeFields {
    InvalidDatatype,
    TruncatedDataspace,
    Valid,
}

impl AttributeFields {
    fn precision(self) -> u16 {
        match self {
            Self::InvalidDatatype | Self::TruncatedDataspace => 68,
            Self::Valid => 64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AttributeVersion {
    One,
    Two,
    Three,
}

impl AttributeVersion {
    fn dataspace_offset(self) -> usize {
        match self {
            Self::One => 32,
            Self::Two => 26,
            Self::Three => 27,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AttributeStorage {
    Compact,
    Dense,
}

impl AttributeStorage {
    fn image(self, body: &[u8]) -> Vec<u8> {
        match self {
            Self::Compact => {
                let header = Superblock::new(Widths::EIGHT);
                let root_at = header.build().len();
                let mut image = FileImage::starting_with(&header.build());
                image.place(
                    root_at,
                    &v2::Header::new()
                        .message(Message::new(MessageType::ATTRIBUTE, body))
                        .build(),
                );
                image.place(
                    0,
                    &header
                        .root_header_address(root_at as u64)
                        .eof_address(image.as_bytes().len() as u64)
                        .build(),
                );
                image.build()
            }
            Self::Dense => test_util::dense_attributes::one_message(body, Widths::EIGHT),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Check {
    ComparePathOpens,
    Read(ReadBackend),
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

#[test]
fn invalid_object_header_version_precedes_missing_alignment_padding() {
    let image = std::fs::read(test_util::data(
        "fuzz/invalid_v1_version_without_padding.h5",
    ))
    .unwrap();
    let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    std::fs::write(&path, &image).unwrap();
    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    let left = buffered.root().attrs().unwrap_err();
    let right = streaming.root().attrs().unwrap_err();
    let Error::Format(left) = left else {
        panic!("expected a format error, got {left:?}");
    };
    let Error::Format(right) = right else {
        panic!("expected a format error, got {right:?}");
    };
    assert_eq!(left, FormatError::InvalidObjectHeaderVersion(2));
    assert_eq!(right, FormatError::InvalidObjectHeaderVersion(2));
    drop(streaming);
    drop(buffered);
    assert_eq!(
        oracle::compare(&image),
        Comparison::Opened {
            groups: 1,
            datasets: 0
        }
    );
}

#[rstest]
#[case::valid(false)]
#[case::shared_leaf(true)]
fn group_listings_reject_repeated_btree_metadata_nodes(
    #[case] repeated: bool,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let image = if repeated {
        test_util::fuzz::repeated_group_btree_leaf()
    } else {
        test_util::legacy::simple_dataset()
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("group-tree.h5");
    std::fs::write(&path, image).unwrap();
    let file = backend.open(&path).unwrap();
    let root = file.root();
    if repeated {
        let error = root.groups().unwrap_err();
        let Error::Format(groups) = error else {
            panic!("expected a group tree format error, got {error:?}");
        };
        let error = root.datasets().unwrap_err();
        let Error::Format(datasets) = error else {
            panic!("expected a dataset tree format error, got {error:?}");
        };
        assert_eq!(groups, FormatError::InvalidBTreeSignature);
        assert_eq!(datasets, FormatError::InvalidBTreeSignature);
    } else {
        assert_eq!(root.groups().unwrap(), Vec::<String>::new());
        assert_eq!(root.datasets().unwrap(), ["data"]);
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap(),
            [1.0, 2.0, 3.0]
        );
    }
    drop(root);
    file.close().unwrap();
}

#[test]
fn legacy_group_btree_header_truncation_precedes_an_invalid_signature() {
    let image = test_util::fuzz::truncated_group_btree_header();
    assert_eq!(image.len(), 2202);
    assert_eq!(
        u64::from_le_bytes(image[0x78..0x80].try_into().unwrap()),
        0x888
    );
    assert_eq!(&image[0x88..0x8c], test_util::btree::v1::SIGNATURE);
    assert_eq!(&image[0x888..0x88c], &[0xf0, 0x3f, 0, 0]);
    assert_eq!(image.len() - 0x888, 18);
    let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    std::fs::write(&path, &image).unwrap();
    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    assert_eq!(buffered.superblock().offset_size.get(), 8);
    let left = buffered.root().datasets().unwrap_err();
    let right = streaming.root().datasets().unwrap_err();
    let Error::Format(left_error) = &left else {
        panic!("expected a format error, got {left:?}");
    };
    let Error::Format(right_error) = &right else {
        panic!("expected a format error, got {right:?}");
    };
    assert_eq!(
        left_error,
        &FormatError::UnexpectedEof {
            expected: 2208,
            available: 2202
        }
    );
    assert_eq!(
        right_error,
        &FormatError::UnexpectedSourceEof {
            expected: 2208,
            available: 2202
        }
    );
    assert_eq!(
        oracle::matched::<Vec<String>>(Operation::DatasetNames, Err(left), Err(right)),
        None
    );
    drop(streaming);
    drop(buffered);
    assert_eq!(
        oracle::compare(&image),
        Comparison::Opened {
            groups: 1,
            datasets: 0
        }
    );
}

#[rstest]
#[case::owner_type_with_restamped_checksum(NodeCorruption::ClientType(ChecksumUpdate::Recompute))]
#[case::owner_type_with_original_checksum(NodeCorruption::ClientType(ChecksumUpdate::Keep))]
#[case::version_with_restamped_checksum(NodeCorruption::Version)]
#[case::signature_with_restamped_checksum(NodeCorruption::Signature)]
#[case::checksum_mismatch(NodeCorruption::Checksum)]
fn dense_attribute_node_corruption_preserves_reader_outcome_parity(
    #[case] corruption: NodeCorruption,
) {
    assert_eq!(
        oracle::compare(&corruption.build()),
        Comparison::Opened {
            groups: 1,
            datasets: 2
        },
    );
}

enum Image {
    Dataset,
    DeepGroups,
    DenseAttributes,
    Groups,
    ManyChildren,
    NanAttributes,
}

impl Image {
    fn build(self) -> Vec<u8> {
        match self {
            Self::Dataset => test_util::legacy::simple_dataset(),
            Self::DenseAttributes => test_util::dense_attributes::dense_attrs(),
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
