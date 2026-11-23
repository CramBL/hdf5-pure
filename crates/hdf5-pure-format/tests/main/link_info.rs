use std::cell::Cell;

use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FormatError, OffsetWidth, StoredAddress};
use hdf5_pure_format::__private::{
    AttributeInfoMessage, GroupEstimates, GroupInfo, GroupPhaseChange, LinkCreationOrderIndex,
    LinkInfoMessage, MessageFlags, MessageRecordLayout, ObjectHeaderPrefix, ObjectHeaderWriter,
    PreparedLink,
};
use hdf5_pure_format::CharacterSet;
use hdf5_pure_format::MessageType;
use rstest::rstest;
use test_util::attribute_info::{AttributeInfo, CreationOrderIndex, Width as AttributeWidth};
use test_util::group_info::{Estimates, GroupInfo as GroupInfoBytes, PhaseChange};
use test_util::link_info::{Body, CreationIndex, Width};
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags as FixtureFlags;
use test_util::object_header::MessageType as FixtureMessageType;
use test_util::object_header::v2::{Header, HeaderFlags};
use test_util::widths::Widths;

#[rstest]
#[case::compact(&[0, 0x80, 0xff, 0xff, 0xff, 0xff], 0x80)]
#[case::tracked_without_body(&[0, 0x81], 0x81)]
#[case::indexed_without_body(&[0, 0x06], 0x06)]
fn reserved_link_info_flags_are_rejected(#[case] bytes: &[u8], #[case] flags: u8) {
    assert_eq!(
        LinkInfoMessage::parse(bytes, 2).unwrap_err(),
        FormatError::InvalidLinkInfoFlags(flags),
    );
}

#[rstest]
#[case::two(Width::Two, 0xe102, 0x3210)]
#[case::four(Width::Four, 0x8765_3210, 0x0123_acef)]
#[case::eight(Width::Eight, 0x8765_4321_0123_4567, 0x9abc_cdef_0fed_8810)]
fn link_info_fields_agree_with_independent_bytes(
    #[case] width: Width,
    #[case] name_index_address: u64,
    #[case] creation_index_address: u64,
    #[values(None, Some(0), Some(u64::MAX))] maximum: Option<u64>,
    #[values(
        CreationIndex::Absent,
        CreationIndex::Undefined,
        CreationIndex::Defined(creation_index_address)
    )]
    creation: CreationIndex,
    #[values(None, Some(0), Some(0x1045))] heap: Option<u64>,
) {
    let expected = Body::new(width, maximum, heap, Some(name_index_address), creation)
        .unwrap()
        .build();
    let offset_width = OffsetWidth::try_from(width.bytes()).unwrap();
    let creation_order_index = match creation {
        CreationIndex::Absent => LinkCreationOrderIndex::Absent,
        CreationIndex::Undefined => LinkCreationOrderIndex::Undefined,
        CreationIndex::Defined(value) => {
            assert_eq!(value, creation_index_address);
            LinkCreationOrderIndex::Defined(StoredAddress::new(value))
        }
    };
    let message = LinkInfoMessage::new(
        offset_width,
        maximum,
        heap.map(StoredAddress::new),
        Some(StoredAddress::new(name_index_address)),
        creation_order_index,
    )
    .unwrap();
    let decoded = LinkInfoMessage::parse_with_offset_width(&expected, offset_width).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(decoded.offset_width(), offset_width);
    assert_eq!(decoded.max_creation_order(), maximum);
    assert_eq!(decoded.fractal_heap_address(), heap.map(StoredAddress::new));
    assert_eq!(
        decoded.btree_name_index_address(),
        Some(StoredAddress::new(name_index_address))
    );
    assert_eq!(decoded.creation_order_index(), creation_order_index);
    assert_eq!(message.to_bytes(), expected);
}

#[rstest]
#[case::two(Width::Two, OffsetWidth::Two)]
#[case::four(Width::Four, OffsetWidth::Four)]
#[case::eight(Width::Eight, OffsetWidth::Eight)]
fn compact_link_info_agrees_with_independent_bytes(
    #[case] width: Width,
    #[case] offset_width: OffsetWidth,
) {
    let expected = Body::new(width, None, None, None, CreationIndex::Absent)
        .unwrap()
        .build();
    let message = LinkInfoMessage::compact(offset_width, None, false);
    let decoded = LinkInfoMessage::parse_with_offset_width(&expected, offset_width).unwrap();

    assert_eq!(decoded, message);
    assert_eq!(decoded.offset_width(), offset_width);
    assert_eq!(decoded.max_creation_order(), None);
    assert_eq!(decoded.fractal_heap_address(), None);
    assert_eq!(decoded.btree_name_index_address(), None);
    assert_eq!(
        decoded.creation_order_index(),
        LinkCreationOrderIndex::Absent
    );
    assert_eq!(message.to_bytes(), expected);
}

#[rstest]
#[case::empty(&[], 2, 0)]
#[case::version_without_flags(&[1], 2, 1)]
fn truncated_link_info_prefix_precedes_version(
    #[case] bytes: &[u8],
    #[case] expected: usize,
    #[case] available: usize,
) {
    assert_eq!(
        LinkInfoMessage::parse(bytes, 2).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        },
    );
}

#[test]
fn invalid_link_info_version_precedes_flags_and_body() {
    assert_eq!(
        LinkInfoMessage::parse(&[1, 0xff], 2).unwrap_err(),
        FormatError::InvalidLinkInfoVersion(1),
    );
}

#[test]
fn invalid_link_info_offset_width_precedes_framing() {
    assert_eq!(
        LinkInfoMessage::parse(&[1, 0xff], 3).unwrap_err(),
        FormatError::InvalidOffsetSize(3),
    );
}

#[rstest]
#[case::maximum(9)]
#[case::heap(13)]
#[case::name_index(17)]
#[case::creation_index(21)]
fn truncated_link_info_body_reports_the_complete_extent(#[case] available: usize) {
    let expected = Body::new(
        Width::Four,
        Some(7),
        Some(0),
        Some(512),
        CreationIndex::Defined(1024),
    )
    .unwrap()
    .build();
    assert_eq!(expected.len(), 22);
    assert_eq!(
        LinkInfoMessage::parse(&expected[..available], 4).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 22,
            available
        },
    );
}

#[test]
fn link_info_decoding_leaves_trailing_bytes_outside_the_body() {
    let expected = Body::new(Width::Two, Some(42), None, None, CreationIndex::Undefined)
        .unwrap()
        .build();
    let mut padded = expected.clone();
    padded.extend_from_slice(&[0x80, 0xff, 0x55]);
    let info = LinkInfoMessage::parse(&padded, 2).unwrap();
    assert_eq!(info.encoded_size(), expected.len());
    assert_eq!(info.to_bytes(), expected);
}

#[rstest]
#[case::two(Width::Two, OffsetWidth::Two)]
#[case::four(Width::Four, OffsetWidth::Four)]
#[case::eight(Width::Eight, OffsetWidth::Eight)]
fn link_info_bounded_output_preserves_guards_and_supports_reuse(
    #[case] width: Width,
    #[case] offset_width: OffsetWidth,
) {
    let expected = Body::new(width, Some(42), None, None, CreationIndex::Undefined)
        .unwrap()
        .build();
    let info = LinkInfoMessage::compact(offset_width, Some(42), true);
    let required = expected.len();
    assert_eq!(info.encoded_size(), required);
    let mut output = vec![0xa5; required + 3];
    for available in [0, required - 1] {
        assert_eq!(
            info.write_into(&mut output[1..1 + available]).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: required,
                available
            },
        );
        assert_eq!(output, vec![0xa5; required + 3]);
    }
    assert_eq!(info.write_into(&mut output[1..]).unwrap(), required);
    let mut guarded = vec![0xa5; output.len()];
    guarded[1..1 + required].copy_from_slice(&expected);
    assert_eq!(output, guarded);
}

#[rstest]
#[case::one_byte_chunk_size(HeaderFlags::default(), 16)]
#[case::two_byte_chunk_size(HeaderFlags(1) | HeaderFlags::TRACKS_CREATION_ORDER, 225)]
fn group_header_records_match_independent_bytes_in_wire_order(
    #[case] flags: HeaderFlags,
    #[case] padding: usize,
) {
    let link_info_bytes = Body::new(Width::Four, Some(42), None, None, CreationIndex::Undefined)
        .unwrap()
        .build();
    let phase_change = PhaseChange::new(12, 8).unwrap();
    let (max_compact, min_dense) = phase_change.thresholds();
    let estimates = Estimates {
        entries: 21,
        name_length: 34,
    };
    let group_info_bytes = GroupInfoBytes {
        phase_change: Some(phase_change),
        estimates: Some(estimates),
    }
    .build();
    let link_bytes = HardLink::new("child", 0x4321)
        .creation_order(13)
        .build(Widths::new(4, 8));
    let attr_maximum = flags.tracks_creation_order().then_some(5);
    let attribute_info_bytes = AttributeInfo::new(
        AttributeWidth::Four,
        attr_maximum,
        None,
        None,
        CreationOrderIndex::Absent,
    )
    .build();
    let padding = vec![0; padding];
    let expected = Header::new()
        .flags(flags)
        .message(Message::new(
            FixtureMessageType::LINK_INFO,
            &link_info_bytes,
        ))
        .message(Message::new(
            FixtureMessageType::GROUP_INFO,
            &group_info_bytes,
        ))
        .message(Message::new(FixtureMessageType::LINK, &link_bytes))
        .message(
            Message::new(FixtureMessageType::ATTRIBUTE_INFO, &attribute_info_bytes)
                .with_flags(FixtureFlags::CONSTANT)
                .with_creation_order(23),
        )
        .message(Message::new(FixtureMessageType::NIL, &padding).with_creation_order(17))
        .build();
    let link_info = LinkInfoMessage::compact(OffsetWidth::Four, Some(42), true);
    let group_info = GroupInfo::new(
        Some(GroupPhaseChange::new(max_compact, min_dense).unwrap()),
        Some(GroupEstimates {
            entries: estimates.entries,
            name_length: estimates.name_length,
        }),
    );
    let mut writer = ObjectHeaderWriter::for_group(
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
        &link_info,
        &group_info,
        core::iter::once(PreparedLink::hard(
            OffsetWidth::Four,
            "child",
            StoredAddress::new(0x4321),
            Some(13),
            CharacterSet::Ascii,
        )),
    )
    .unwrap();
    writer
        .append_message(
            AttributeInfoMessage::compact(OffsetWidth::Four, attr_maximum, false),
            MessageFlags::CONSTANT,
            23,
        )
        .unwrap();
    writer
        .append_raw_record(MessageType::NIL, &padding, MessageFlags::NONE, 17)
        .unwrap();

    assert_eq!(writer.serialize().unwrap(), expected);
}

#[test]
fn a_group_header_rejects_a_hard_link_at_a_different_offset_width() {
    let info = LinkInfoMessage::compact(OffsetWidth::Four, None, false);
    let link = hard_link(OffsetWidth::Two);

    assert_eq!(
        ObjectHeaderWriter::for_group(
            ObjectHeaderPrefix::PLAIN,
            &info,
            &GroupInfo::DEFAULT,
            core::iter::once(Ok(link))
        )
        .unwrap_err(),
        FormatError::LinkOffsetWidthMismatch {
            group_width: 4,
            link_width: 2,
        }
    );
}

#[test]
fn a_group_header_rejects_inline_links_with_dense_storage() {
    let info = LinkInfoMessage::new(
        OffsetWidth::Four,
        None,
        Some(StoredAddress::new(0x1234)),
        Some(StoredAddress::new(0x5678)),
        LinkCreationOrderIndex::Absent,
    )
    .unwrap();
    let link = hard_link(OffsetWidth::Four);

    assert_eq!(
        ObjectHeaderWriter::for_group(
            ObjectHeaderPrefix::PLAIN,
            &info,
            &GroupInfo::DEFAULT,
            core::iter::once(Ok(link))
        )
        .unwrap_err(),
        FormatError::InlineLinksWithDenseStorage {
            heap_address: 0x1234,
        }
    );
}

#[rstest]
#[case::record(
    Ok(hard_link(OffsetWidth::Two)),
    FormatError::LinkOffsetWidthMismatch {
        group_width: 4,
        link_width: 2,
    },
)]
#[case::iterator(
    Err(FormatError::InvalidLinkName("broken".into())),
    FormatError::InvalidLinkName("broken".into()),
)]
fn group_writer_stops_consuming_links_at_the_first_error(
    #[case] bad_link: Result<PreparedLink<'static>, FormatError>,
    #[case] expected: FormatError,
) {
    let info = LinkInfoMessage::compact(OffsetWidth::Four, None, false);
    let read = Cell::new(0);
    let links = [
        Ok(hard_link(OffsetWidth::Four)),
        bad_link,
        Ok(hard_link(OffsetWidth::Four)),
    ]
    .into_iter()
    .inspect(|_| read.set(read.get() + 1));

    let error =
        ObjectHeaderWriter::for_group(ObjectHeaderPrefix::PLAIN, &info, &GroupInfo::DEFAULT, links)
            .unwrap_err();
    assert_eq!(error, expected);
    assert_eq!(read.get(), 2);
}

#[test]
fn group_writer_rejects_supplemental_typed_group_info() {
    let mut writer = ObjectHeaderWriter::for_group(
        ObjectHeaderPrefix::PLAIN,
        &LinkInfoMessage::compact(OffsetWidth::Four, None, false),
        &GroupInfo::DEFAULT,
        core::iter::empty(),
    )
    .unwrap();
    assert_eq!(
        writer.append_message(GroupInfo::DEFAULT, MessageFlags::NONE, 0),
        Err(FormatError::QueuedGroupLinkMessage(
            FixtureMessageType::GROUP_INFO.0
        ))
    );
}

#[rstest]
#[case::link(
    MessageType::LINK,
    FixtureMessageType::LINK,
    HardLink::new("child", 0x1234).build(Widths::new(2, 8)),
)]
#[case::link_info(
    MessageType::LINK_INFO,
    FixtureMessageType::LINK_INFO,
    Body::new(Width::Four, None, None, None, CreationIndex::Absent).unwrap().build(),
)]
#[case::group_info(MessageType::GROUP_INFO, FixtureMessageType::GROUP_INFO, vec![0, 0])]
#[case::symbol_table(
    MessageType::SYMBOL_TABLE,
    FixtureMessageType::SYMBOL_TABLE,
    vec![0x12, 0, 0, 0, 0x34, 0, 0, 0],
)]
fn ordinary_writer_accepts_raw_records_rejected_as_group_supplemental_records(
    #[case] message_type: MessageType,
    #[case] fixture_type: FixtureMessageType,
    #[case] body: Vec<u8>,
) {
    let expected = Header::new()
        .message(Message::new(fixture_type, &body))
        .build();
    let info = LinkInfoMessage::compact(OffsetWidth::Four, None, false);
    let mut group = ObjectHeaderWriter::for_group(
        ObjectHeaderPrefix::PLAIN,
        &info,
        &GroupInfo::DEFAULT,
        core::iter::empty(),
    )
    .unwrap();
    assert_eq!(
        group.append_raw_message(message_type, &body),
        Err(FormatError::QueuedGroupLinkMessage(fixture_type.0)),
    );

    let mut ordinary = ObjectHeaderWriter::new();
    ordinary.append_raw_message(message_type, &body).unwrap();
    assert_eq!(ordinary.serialize().unwrap(), expected);
}

fn hard_link(offset_width: OffsetWidth) -> PreparedLink<'static> {
    PreparedLink::hard(
        offset_width,
        "child",
        StoredAddress::new(0x4321),
        None,
        CharacterSet::Ascii,
    )
    .unwrap()
}

#[rstest]
#[case::heap(0)]
#[case::name_index(1)]
#[case::creation_index(2)]
fn link_info_construction_rejects_the_undefined_marker_as_a_defined_address(
    #[case] field: usize,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let sentinel = StoredAddress::undefined(width);
    assert_eq!(
        construction_at(width, field, sentinel).unwrap_err(),
        FormatError::DefinedAddressIsUndefined {
            address: sentinel.get(),
            offset_size: width.get()
        },
    );
}

#[rstest]
#[case::heap(0)]
#[case::name_index(1)]
#[case::creation_index(2)]
fn link_info_construction_rejects_an_address_outside_its_width(
    #[case] field: usize,
    #[values(OffsetWidth::Two, OffsetWidth::Four)] width: OffsetWidth,
) {
    let address = StoredAddress::new(StoredAddress::undefined(width).get() + 1);
    assert_eq!(
        construction_at(width, field, address).unwrap_err(),
        FormatError::AddressTooLarge {
            address: address.get(),
            offset_size: width.get()
        },
    );
}

#[rstest]
#[case::heap(Some(0xffff), Some(0x1_0000), 0xffff, FormatError::DefinedAddressIsUndefined { address: 0xffff, offset_size: 2 })]
#[case::name_index(None, Some(0x1_0000), 0xffff, FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 })]
fn link_info_construction_checks_addresses_in_wire_order(
    #[case] heap: Option<u64>,
    #[case] name_index_address: Option<u64>,
    #[case] creation_index_address: u64,
    #[case] expected: FormatError,
) {
    assert_eq!(
        LinkInfoMessage::new(
            OffsetWidth::Two,
            None,
            heap.map(StoredAddress::new),
            name_index_address.map(StoredAddress::new),
            LinkCreationOrderIndex::Defined(StoredAddress::new(creation_index_address))
        )
        .unwrap_err(),
        expected,
    );
}

#[test]
fn removing_a_link_info_maximum_preserves_index_field_presence() {
    let mut info = LinkInfoMessage::compact(OffsetWidth::Two, Some(42), true);
    info.set_max_creation_order(None);
    let expected = Body::new(Width::Two, None, None, None, CreationIndex::Undefined)
        .unwrap()
        .build();
    assert_eq!(info.to_bytes(), expected);
    assert_eq!(info, LinkInfoMessage::parse(&expected, 2).unwrap());
}

fn construction_at(
    width: OffsetWidth,
    field: usize,
    address: StoredAddress,
) -> Result<LinkInfoMessage, FormatError> {
    LinkInfoMessage::new(
        width,
        None,
        (field == 0).then_some(address),
        (field == 1).then_some(address),
        if field == 2 {
            LinkCreationOrderIndex::Defined(address)
        } else {
            LinkCreationOrderIndex::Absent
        },
    )
}
