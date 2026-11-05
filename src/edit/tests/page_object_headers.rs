use hdf5_pure_space::__private::{AllocationClass, Extent, PageTransition, PageType, SessionSpace};
use rstest::rstest;
use test_util::object_header::v2::{Header, HeaderFlags};
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::userblock;

use crate::PhysicalAddress;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::edit::test_util as edit_test_util;
use crate::source::Source;
use crate::{FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy};

#[rstest]
#[case::small(99)]
#[case::page_minus_one(4_095)]
#[case::page(4_096)]
#[case::page_plus_one(4_097)]
fn complete_object_headers_use_checked_block_placement(
    #[case] length: usize,
    #[values(false, true)] reused: bool,
    #[values(0, 512)] base: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("headers.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.write(&path).unwrap();
    if base != 0 {
        userblock::prepend(&path, &vec![0xA5; usize::try_from(base).unwrap()]);
    }
    let mut engine = edit_test_util::open_engine(&path);
    engine.space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    engine
        .space
        .record_transition(PageType::Meta, PageTransition::Unchanged);
    let physical = engine.image.len();
    let stored = physical - base;
    assert_eq!(stored % PAGE, 0);
    if reused {
        engine.append(&[13; 12_288]).unwrap();
        engine.space.return_allocation(
            Extent::new(StoredAddress::new(stored + 4_000), 8_192).unwrap(),
            AllocationClass::Metadata,
        );
    } else {
        engine.append(&[13; 4_000]).unwrap();
    }
    let body = vec![0; length - 16];
    let header = Header::new()
        .flags(TWO_BYTE_CHUNK_SIZE)
        .message(Message::new(MessageType::NIL, &body))
        .build();
    assert_eq!(header.len(), length);
    let address = engine.place_object_header(&header).unwrap();
    assert_eq!(address, physical + PAGE);
    assert_eq!(
        engine
            .oh_chunk_spans(PhysicalAddress::new(address))
            .unwrap()[0]
            .length(),
        length as u64
    );
    assert_eq!(
        engine.image().read_exact_at(address, length).unwrap(),
        header
    );
    if reused {
        assert_eq!(engine.image.len(), physical + 12_288);
        assert_eq!(
            engine.image().read_exact_at(physical, 4_096).unwrap(),
            [13; 4_096]
        );
    } else {
        assert_eq!(engine.image.len(), address + length as u64);
        assert_eq!(
            engine.image().read_exact_at(physical + 4_000, 96).unwrap(),
            [0; 96]
        );
    }
    let next = Header::new().build();
    let next_address = engine.place_object_header(&next).unwrap();
    assert_eq!(
        next_address,
        if reused {
            physical + 4_000
        } else if length == 4_095 {
            address + PAGE
        } else {
            address + length as u64
        }
    );
}

#[test]
fn flat_object_header_placement_keeps_the_current_tail() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("flat.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.append(&[13; 7]).unwrap();
    let address = engine.image.len();
    let header = Header::new().build();
    assert_eq!(engine.place_object_header(&header).unwrap(), address);
    assert_eq!(engine.image.len(), address + header.len() as u64);
}

const PAGE: u64 = 4_096;

// Section `subsubsec_fmt4_dataobject_hdr_prefix_two`, version 4.0, assigns 1 to a two-byte
// chunk-size field.
const TWO_BYTE_CHUNK_SIZE: HeaderFlags = HeaderFlags(0x01);
