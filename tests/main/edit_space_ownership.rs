use std::panic;
use std::panic::AssertUnwindSafe;

use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure::SpaceOwnership;
use hdf5_pure::StoredAddress;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::OffsetWidth;
use rstest::rstest;
use test_util::fractal_heap::Header as HeapHeader;
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v1::Header as HeaderV1;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::file_space;

#[test]
fn a_contiguous_i32_dataset_has_complete_ownership() {
    let path = temp::temp_path("supported_ownership.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("d").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let ownership: SpaceOwnership = file.__space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    assert!(ownership.unclassified_space.is_empty());
    let summary = file_space::assert_exact_space_ownership(&file);
    assert_eq!(summary.live_raw_bytes, 12);
    file.close().unwrap();
}

#[rstest]
#[case::dataset(true, 0)]
#[case::compact_attribute(false, 0)]
#[case::dense_attribute(false, 12)]
fn variable_length_storage_is_incomplete(#[case] dataset: bool, #[case] attributes: usize) {
    let path = temp::temp_path("variable_length_ownership.h5");
    let mut builder = FileBuilder::new();
    if dataset {
        builder
            .create_dataset("d")
            .with_vlen_strings(&["alpha", "beta"]);
    } else {
        builder.set_attr(
            "labels",
            AttrValue::VarLenAsciiCharArray(vec!["alpha".into(), "beta".into()]),
        );
        for index in 0..attributes {
            builder.set_attr(&format!("a{index}"), AttrValue::I64(1));
        }
    }
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let ownership = file.__space_ownership().unwrap();
    assert!(!ownership.analysis_complete);
    assert!(!ownership.unclassified_space.is_empty());
    file.close().unwrap();
}

#[test]
fn unknown_reachable_metadata_is_incomplete_even_without_unclassified_bytes() {
    let path = temp::temp_path("unknown_ownership.h5");
    std::fs::write(
        &path,
        ownership_chain(2, &[Message::new(MessageType(0x7f), &[])]),
    )
    .unwrap();
    let file = File::open_rw(&path).unwrap();
    let ownership = file.__space_ownership().unwrap();
    assert!(!ownership.analysis_complete);
    assert!(ownership.unclassified_space.is_empty());
    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
        file_space::assert_exact_space_ownership(&file)
    }))
    .unwrap_err();
    assert_eq!(
        *panic.downcast_ref::<&str>().unwrap(),
        "ownership analysis must be complete"
    );
    file.close().unwrap();
}

#[rstest]
#[case::at_depth_limit(1_001, false)]
#[case::below_depth_limit(1_000, true)]
fn ownership_is_complete_only_below_the_traversal_depth_limit(
    #[case] depth: usize,
    #[case] complete: bool,
) {
    let path = temp::temp_path("depth_ownership.h5");
    std::fs::write(&path, ownership_chain(depth, &[])).unwrap();
    let file = File::open_rw(&path).unwrap();
    let ownership = file.__space_ownership().unwrap();
    assert_eq!(ownership.analysis_complete, complete);
    assert_eq!(ownership.unclassified_space.is_empty(), complete);
    file.close().unwrap();
}

#[rstest]
#[case::invalid_layout(MessageType::DATA_LAYOUT, vec![0xff, 0], FormatError::InvalidLayoutVersion(0xff))]
#[case::invalid_attribute_info(MessageType::ATTRIBUTE_INFO, vec![0xff, 0], FormatError::InvalidAttributeInfoVersion(0xff))]
#[case::invalid_attribute_heap(
    MessageType::ATTRIBUTE_INFO,
    attribute_heap_at_superblock(),
    FormatError::InvalidFractalHeapSignature
)]
#[case::truncated_attribute(MessageType::ATTRIBUTE, vec![], FormatError::UnexpectedEof { expected: 2, available: 0 })]
#[case::truncated_link(MessageType::LINK, vec![], FormatError::UnexpectedEof { expected: 2, available: 0 })]
#[case::overflowing_raw_address(MessageType::DATA_LAYOUT, contiguous_layout(u64::MAX - 1, 8), FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 8 })]
fn reachable_metadata_failures_are_returned(
    #[case] message: MessageType,
    #[case] body: Vec<u8>,
    #[case] expected: FormatError,
) {
    let path = temp::temp_path("malformed_ownership.h5");
    std::fs::write(
        &path,
        ownership_chain(2, &[Message::new(message, &body), Message::nil(128)]),
    )
    .unwrap();
    let file = File::open_rw(&path).unwrap();
    let error = file.__space_ownership().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}")
    };
    assert_eq!(error, expected);
    file.close().unwrap();
}

#[rstest]
#[case::overlap(1, 1)]
#[case::beyond_eof(100_000, 1)]
fn invalid_owned_extents_report_their_geometry(#[case] address: u64, #[case] length: u64) {
    let path = temp::temp_path("owned_extent_geometry.h5");
    let bytes = ownership_chain(
        2,
        &[Message::new(
            MessageType::DATA_LAYOUT,
            &contiguous_layout(address, length),
        )],
    );
    let logical_size = bytes.len() as u64;
    std::fs::write(&path, bytes).unwrap();
    let file = File::open_rw(&path).unwrap();
    let error = file.__space_ownership().unwrap_err();
    let Error::InvalidSpaceOwnershipExtent {
        address: actual_address,
        length: actual_length,
        previous_end,
        logical_size: actual_size,
    } = error
    else {
        panic!("expected InvalidSpaceOwnershipExtent, got {error:?}")
    };
    assert_eq!(
        (actual_address, actual_length, actual_size),
        (address, length, logical_size)
    );
    assert_eq!(
        previous_end,
        if address == 1 {
            Superblock::new(Widths::EIGHT).build().len() as u64
        } else {
            logical_size
        }
    );
    file.close().unwrap();
}

#[test]
fn inconsistent_heap_ownership_is_returned_as_a_typed_error() {
    let path = temp::temp_path("inconsistent_heap_ownership.h5");
    let heap = HeapHeader::new(u64::MAX)
        .managed_object_count(1)
        .build(Widths::EIGHT);
    let mut image = Image::new();
    let superblock = Superblock::new(Widths::EIGHT).version(SUPERBLOCK_VERSION);
    let heap_address = superblock.build().len() as u64;
    image.place(heap_address as usize, &heap);
    let info = AttributeInfoMessage {
        max_creation_index: None,
        indexes_creation_order: false,
        fractal_heap_address: Some(StoredAddress::new(heap_address)),
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    let root = image.append(
        &Header::new()
            .message(Message::new(
                MessageType::ATTRIBUTE_INFO,
                &info.serialize(OffsetWidth::Eight),
            ))
            .build(),
    );
    image.place(
        0,
        &superblock
            .root_header_address(root as u64)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    std::fs::write(&path, image.build()).unwrap();
    let file = File::open_rw(&path).unwrap();
    let error = file.__space_ownership().unwrap_err();
    let Error::InvalidFractalHeapStorage { address } = error else {
        panic!("expected InvalidFractalHeapStorage, got {error:?}")
    };
    assert_eq!(address, heap_address);
    file.close().unwrap();
}

#[test]
fn a_version_one_header_has_incomplete_ownership() {
    let path = temp::temp_path("version_one_ownership.h5");
    let mut image = Image::new();
    let superblock = Superblock::new(Widths::EIGHT).version(SUPERBLOCK_VERSION);
    let header_address = superblock.build().len();
    image.place(
        header_address,
        &HeaderV1::new().message(Message::nil(0)).build(),
    );
    image.place(
        0,
        &superblock
            .root_header_address(header_address as u64)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    std::fs::write(&path, image.build()).unwrap();
    let file = File::open_rw(&path).unwrap();
    let ownership = file.__space_ownership().unwrap();
    assert!(!ownership.analysis_complete);
    assert!(!ownership.unclassified_space.is_empty());
    file.close().unwrap();
}

#[test]
fn a_userblock_has_incomplete_ownership() {
    let path = temp::temp_path("userblock_ownership.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(USERBLOCK_SIZE);
    builder.create_dataset("d").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let ownership = file.__space_ownership().unwrap();
    assert!(!ownership.analysis_complete);
    assert_eq!(ownership.unclassified_space, [(0, USERBLOCK_SIZE)]);
    file.close().unwrap();
}

#[rstest]
#[case::unknown_message(Message::new(MessageType(0x7f), &[]))]
#[case::shared_dataspace(Message::new(MessageType::DATASPACE, &format::encode_sohm_ref(&[0; format::FHEAP_ID_LEN])).with_flags(MessageFlags::SHARED))]
#[case::shared_attribute(Message::new(MessageType::ATTRIBUTE, &format::encode_sohm_ref(&[0; format::FHEAP_ID_LEN])).with_flags(MessageFlags::SHARED))]
fn incomplete_coverage_still_returns_later_parser_errors(#[case] unsupported: Message) {
    let path = temp::temp_path("incomplete_malformed_ownership.h5");
    let bytes = ownership_chain(
        2,
        &[
            unsupported,
            Message::new(MessageType::DATA_LAYOUT, &[0xff, 0]),
        ],
    );
    std::fs::write(&path, bytes).unwrap();
    let file = File::open_rw(&path).unwrap();
    let error = file.__space_ownership().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::InvalidLayoutVersion(0xff))
        ),
        "{error:?}"
    );
    file.close().unwrap();
}

fn ownership_chain(depth: usize, leaf: &[Message]) -> Vec<u8> {
    let mut image = Image::new();
    let superblock = Superblock::new(Widths::EIGHT).version(SUPERBLOCK_VERSION);
    let leaf_address = superblock.build().len();
    image.place(
        leaf_address,
        &Header::new().messages(leaf.iter().cloned()).build(),
    );
    let mut child = leaf_address;
    for _ in 1..depth {
        child = image.append(
            &Header::new()
                .message(Message::new(
                    MessageType::LINK,
                    &HardLink::new("child", child as u64).build(Widths::EIGHT),
                ))
                .build(),
        );
    }
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .version(SUPERBLOCK_VERSION)
            .root_header_address(child as u64)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    image.build()
}

fn contiguous_layout(address: u64, size: u64) -> Vec<u8> {
    let mut layout = vec![LAYOUT_VERSION, CONTIGUOUS_LAYOUT_CLASS];
    layout.extend_from_slice(&address.to_le_bytes());
    layout.extend_from_slice(&size.to_le_bytes());
    layout
}

fn attribute_heap_at_superblock() -> Vec<u8> {
    AttributeInfoMessage {
        max_creation_index: None,
        indexes_creation_order: false,
        fractal_heap_address: Some(StoredAddress::new(0)),
        btree_name_index_address: None,
        btree_creation_order_address: None,
    }
    .serialize(OffsetWidth::Eight)
}

const LAYOUT_VERSION: u8 = 3;
const CONTIGUOUS_LAYOUT_CLASS: u8 = 1;

const SUPERBLOCK_VERSION: u8 = 3;

const USERBLOCK_SIZE: u64 = 512;
