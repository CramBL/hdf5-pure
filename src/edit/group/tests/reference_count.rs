use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::MessageType as WireType;
use test_util::object_header::v1;
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::access_mode::AccessMode;
use crate::edit::group;
use crate::edit::test_util as edit_test_util;
use crate::file_writer;
use crate::message_type::MessageType;
use crate::object_graph::ObjectId;
use crate::object_header::ObjectHeader;
use crate::width::OffsetWidth;

#[rstest]
#[case::one(1)]
#[case::multiple(3)]
#[case::zero(0)]
#[case::maximum(u32::MAX)]
fn converted_group_records_preserve_supported_prefix_counts(#[case] count: u32) {
    let bytes = v1::Header::new()
        .reference_count(count)
        .message(Message::nil(8))
        .build();
    let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(header.reference_count, Some(count));
    let mut region = group::fresh_group_region(file_writer::WIDTHS).unwrap();
    let original = region.clone();
    region.preserve_v1_count(&header).unwrap();
    assert_eq!(region.reference_count_mut().unwrap().count(), count);
    let mut expected = original;
    if count != 1 {
        expected.push(
            MessageType::OBJECT_REFERENCE_COUNT,
            &Message::reference_count(count).data,
        );
    }
    assert_eq!(&region[..], &expected[..]);
    assert_eq!(region.props(), expected.props());
}

#[test]
fn modern_link_storage_keeps_the_version_1_prefix_count() {
    let path = test_util::temp::temp_path("modern-prefix-count.h5");
    let mut fixture = Fixture::new();
    let mut link_info = Message::new(
        WireType::LINK_INFO,
        &file_writer::COMPACT_LINK_INFO
            .serialize(OffsetWidth::Eight)
            .unwrap(),
    );
    link_info.data.resize(24, 0);
    let header = v1::Header::new()
        .reference_count(3)
        .message(link_info)
        .build();
    let address = u64::try_from(fixture.image.append(&header)).unwrap();
    fixture.link("g", address);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let group::GroupInfo { mut region, links } = engine
        .reconstruct_v1_group(ObjectId::new(PhysicalAddress::new(address)))
        .unwrap();
    assert!(links.links().is_empty());
    assert_eq!(region.reference_count_mut().unwrap().count(), 3);
}

#[rstest]
#[case::missing(false)]
#[case::ambiguous(true)]
fn missing_or_conflicting_prefix_counts_leave_the_region_unchanged(#[case] ambiguous: bool) {
    let mut message = Message::reference_count(3);
    message.data.resize(8, 0);
    let bytes = v1::Header::new()
        .reference_count(7)
        .message(message)
        .build();
    let mut header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    if !ambiguous {
        header.reference_count = None;
        header.messages.clear();
    }
    let mut region = group::fresh_group_region(file_writer::WIDTHS).unwrap();
    let original = region.clone();
    region.preserve_v1_count(&header).unwrap();
    assert_eq!(&region[..], &original[..]);
    assert_eq!(region.props(), original.props());
}
