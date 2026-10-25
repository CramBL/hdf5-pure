use std::ops::Range;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::path::Path;

use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::OffsetWidth;
use hdf5_pure_space::__private::AGGREGATE_BLOCK_SIZE;
use rstest::rstest;
use test_util::attribute::Attribute;
use test_util::data_layout;
use test_util::dataspace;
use test_util::fractal_heap::Header as HeapHeader;
use test_util::global_heap;
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
use test_util_hdf5::global_heap_fixture;

use crate::AttrValue;
use crate::Error;
use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::Layout;
use crate::StoredAddress;
use crate::UserblockSize;
use crate::address::StoredAddressExt;
use crate::edit::{MemoryStrategy, SpaceOwnership, WriteEngine};
use crate::source::MetadataCacheConfig;

#[rstest]
#[case::plain(0, &[])]
#[case::userblock(USERBLOCK_SIZE, &[(0, USERBLOCK_SIZE)])]
fn a_contiguous_i32_dataset_has_complete_ownership(
    #[case] userblock: u64,
    #[case] userblock_space: &[(u64, u64)],
) {
    let path = temp::temp_path("supported_ownership.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.create_dataset("d").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let engine = open_engine(&path);
    let ownership: SpaceOwnership = engine.space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    assert!(ownership.unclassified_space.is_empty());
    assert_eq!(ownership.userblock_space, userblock_space);
    ownership.assert_exact();
    assert_eq!(
        ownership
            .live_raw_space
            .iter()
            .map(|&(_, len)| len)
            .sum::<u64>(),
        12
    );
    drop(engine);
}

#[rstest]
fn default_creation_leaves_only_unused_metadata_spare_unclassified(
    #[values(0, USERBLOCK_SIZE)] userblock: u64,
) {
    let path = temp::temp_path("default_ownership.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.create_dataset("d").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let data = contiguous_extent(&path, "d");
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.dataset("d").unwrap().read_i32().unwrap(), [1, 2, 3]);
    reader.close().unwrap();
    let engine = open_engine(&path);
    let ownership = engine.space_ownership().unwrap();
    let metadata_end = ownership
        .live_metadata_space
        .iter()
        .map(|&(start, len)| start + len)
        .max()
        .unwrap();
    assert!(ownership.analysis_complete);
    assert_eq!(ownership.live_raw_space, [(data.start, 12)]);
    assert_eq!(ownership.free_space_manager_space, []);
    assert_eq!(ownership.reusable_free_space, []);
    assert_eq!(ownership.trailing_slack_space, []);
    assert_eq!(
        ownership.unclassified_space,
        [(metadata_end, data.start - metadata_end)]
    );
    let superblock_len = format::serialize_superblock(engine.superblock())
        .unwrap()
        .len() as u64;
    assert_eq!(
        data.start,
        userblock + superblock_len + AGGREGATE_BLOCK_SIZE
    );
    ownership.assert_classified(&[(metadata_end, data.start - metadata_end)]);
    assert_eq!(ownership.logical_size, data.end);
    assert_eq!(ownership.physical_size, data.end);
    drop(engine);
}

#[rstest]
#[case::dataset(true, 0)]
#[case::compact_attribute(false, 0)]
#[case::dense_attribute(false, 12)]
fn variable_length_collections_are_live_raw_space(
    #[case] dataset: bool,
    #[case] attributes: usize,
) {
    let path = temp::temp_path("variable_length_ownership.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
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
    let mut expected: Vec<(u64, u64)> =
        global_heap::collection_ranges(&std::fs::read(&path).unwrap(), Widths::EIGHT)
            .into_iter()
            .map(|collection| (collection.start, collection.end - collection.start))
            .collect();
    if dataset {
        let data = contiguous_extent(&path, "d");
        expected.push((data.start, data.end - data.start));
    }
    expected.sort_unstable();
    let engine = open_engine(&path);

    engine.space_ownership().unwrap().assert_exact();
    let mut live_raw_space = engine.space_ownership().unwrap().live_raw_space;
    live_raw_space.sort_unstable();
    assert_eq!(live_raw_space, expected);
    drop(engine);
}

#[test]
fn a_malformed_live_collection_is_an_error() {
    let path = temp::temp_path("malformed_collection_ownership.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_vlen_strings(&["alpha", "beta"]);
    builder.write(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let [collection] = global_heap::collection_starts(&bytes)[..] else {
        panic!("expected one collection");
    };
    bytes[collection as usize + global_heap::VERSION_OFFSET] = 2;
    std::fs::write(&path, bytes).unwrap();
    let engine = open_engine(&path);

    let error = engine.space_ownership().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::InvalidGlobalHeapVersion(2))
        ),
        "{error:?}"
    );
    drop(engine);
}

#[rstest]
#[case::unknown_message(vec![Message::new(MessageType(0x7f), &[])])]
#[case::shared_dataspace(vec![Message::new(MessageType::DATASPACE, &format::encode_sohm_ref(&[0; format::FHEAP_ID_LEN])).with_flags(MessageFlags::SHARED)])]
#[case::shared_attribute(vec![Message::new(MessageType::ATTRIBUTE, &format::encode_sohm_ref(&[0; format::FHEAP_ID_LEN])).with_flags(MessageFlags::SHARED)])]
#[case::object_reference_dataset(object_reference_dataset())]
#[case::object_reference_attribute(vec![Message::new(
    MessageType::ATTRIBUTE,
    &Attribute::new(
        "a",
        &format::serialize_datatype(&global_heap_fixture::object_reference()),
        &dataspace::scalar(),
        &vec![0; global_heap_fixture::object_reference().type_size() as usize],
    )
    .build(),
)])]
fn unaccounted_reachable_metadata_is_incomplete_even_without_unclassified_bytes(
    #[case] leaf: Vec<Message>,
) {
    let path = temp::temp_path("unknown_ownership.h5");
    std::fs::write(&path, ownership_chain(2, &leaf)).unwrap();
    let engine = open_engine(&path);
    let ownership = engine.space_ownership().unwrap();
    assert!(!ownership.analysis_complete);
    assert!(ownership.unclassified_space.is_empty());
    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
        engine.space_ownership().unwrap().assert_exact()
    }))
    .unwrap_err();
    assert_eq!(
        *panic.downcast_ref::<&str>().unwrap(),
        "ownership analysis must be complete"
    );
    drop(engine);
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
    let engine = open_engine(&path);
    let ownership = engine.space_ownership().unwrap();
    assert_eq!(ownership.analysis_complete, complete);
    assert_eq!(ownership.unclassified_space.is_empty(), complete);
    drop(engine);
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
#[case::overflowing_raw_address(MessageType::DATA_LAYOUT, data_layout::contiguous(u64::MAX - 1, 8), FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 8 })]
fn reachable_metadata_failures_are_returned(
    #[case] message: MessageType,
    #[case] body: Vec<u8>,
    #[case] expected: FormatError,
) {
    let path = temp::temp_path("malformed_ownership.h5");
    std::fs::write(
        &path,
        ownership_chain(
            2,
            &[
                datatype_message(),
                dataspace_message(),
                Message::new(message, &body),
                Message::nil(128),
            ],
        ),
    )
    .unwrap();
    let engine = open_engine(&path);
    let error = engine.space_ownership().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}")
    };
    assert_eq!(error, expected);
    drop(engine);
}

#[rstest]
#[case::overlap(1, 1)]
#[case::beyond_eof(100_000, 1)]
fn invalid_owned_extents_report_their_geometry(#[case] address: u64, #[case] length: u64) {
    let path = temp::temp_path("owned_extent_geometry.h5");
    let bytes = ownership_chain(
        2,
        &[
            datatype_message(),
            dataspace_message(),
            Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::contiguous(address, length),
            ),
        ],
    );
    let logical_size = bytes.len() as u64;
    std::fs::write(&path, bytes).unwrap();
    let engine = open_engine(&path);
    let error = engine.space_ownership().unwrap_err();
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
    drop(engine);
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
    let engine = open_engine(&path);
    let error = engine.space_ownership().unwrap_err();
    let Error::InvalidFractalHeapStorage { address } = error else {
        panic!("expected InvalidFractalHeapStorage, got {error:?}")
    };
    assert_eq!(address, heap_address);
    drop(engine);
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
    let engine = open_engine(&path);
    let ownership = engine.space_ownership().unwrap();
    assert!(!ownership.analysis_complete);
    assert!(!ownership.unclassified_space.is_empty());
    drop(engine);
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
    let engine = open_engine(&path);
    let error = engine.space_ownership().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::InvalidLayoutVersion(0xff))
        ),
        "{error:?}"
    );
    drop(engine);
}

impl SpaceOwnership {
    /// Asserts complete classification of every byte in the file.
    ///
    /// # Panics
    ///
    /// Panics if the analysis is incomplete, any byte is unclassified, the physical and logical
    /// file lengths differ, or the extents do not partition the file into nonempty ranges.
    pub(crate) fn assert_exact(&self) {
        self.assert_classified(&[]);
    }

    /// Asserts that the ownership categories partition the file with the expected unclassified
    /// extents.
    ///
    /// # Panics
    ///
    /// Panics if the analysis is incomplete, the unclassified extents differ, the physical and
    /// logical file lengths differ, or the extents do not partition the file into nonempty ranges.
    pub(crate) fn assert_classified(&self, expected_unclassified: &[(u64, u64)]) {
        let Self {
            physical_size,
            logical_size,
            userblock_space,
            live_raw_space,
            live_metadata_space,
            free_space_manager_space,
            reusable_free_space,
            unclassified_space,
            analysis_complete,
            trailing_slack_space,
        } = self;
        assert!(*analysis_complete, "ownership analysis must be complete");
        assert_eq!(unclassified_space, expected_unclassified);
        assert_eq!(trailing_slack_space, &[]);
        assert_eq!(physical_size, logical_size);
        let mut extents = userblock_space
            .iter()
            .chain(live_raw_space)
            .chain(live_metadata_space)
            .chain(free_space_manager_space)
            .chain(reusable_free_space)
            .chain(unclassified_space)
            .chain(trailing_slack_space)
            .copied()
            .collect::<Vec<_>>();
        extents.sort_unstable();
        let mut end = 0;
        for (start, len) in extents {
            assert!(len > 0, "empty ownership extent at {start}");
            assert_eq!(start, end, "ownership partition has a gap or overlap");
            end = start.checked_add(len).unwrap();
            assert!(
                end <= *physical_size,
                "ownership extent exceeds physical EOF"
            );
        }
        assert_eq!(
            end, *physical_size,
            "ownership partition must cover the file"
        );
    }
}

/// Opens a writable fixture with metadata caching disabled.
///
/// # Panics
///
/// Panics if the fixture cannot be opened for editing.
pub(super) fn open_engine(path: &Path) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Auto,
    )
    .unwrap()
}

fn contiguous_extent(path: &Path, dataset: &str) -> Range<u64> {
    let file = File::from_bytes(std::fs::read(path).unwrap()).unwrap();
    let layout = file.dataset(dataset).unwrap().layout().unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = layout
    else {
        panic!("expected allocated contiguous storage, got {layout:?}");
    };
    address..address + size
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

fn datatype_message() -> Message {
    Message::new(
        MessageType::DATATYPE,
        &format::serialize_datatype(&crate::make_u8_type()),
    )
}

fn dataspace_message() -> Message {
    Message::new(MessageType::DATASPACE, &dataspace::scalar())
}

fn object_reference_dataset() -> Vec<Message> {
    let mut messages = global_heap_fixture::dataset_messages(
        &global_heap_fixture::object_reference(),
        &dataspace::scalar(),
    );
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::contiguous(global_heap_fixture::UNDEFINED_ADDRESS, 0),
    ));
    messages
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

const SUPERBLOCK_VERSION: u8 = 3;

const USERBLOCK_SIZE: u64 = 512;
