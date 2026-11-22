use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{
    AttributeInfoField, DecodedStoredAddress, FormatError, OffsetWidth, StoredAddress,
};
use hdf5_pure_format::__private::{
    AttributeCreationOrderIndex, AttributeInfoMessage, AttributeInfoOutputTooSmall,
};
use rstest::rstest;
use test_util::attribute_info::{AttributeInfo, CreationOrderIndex, Width};
use test_util::object_header::v1::MESSAGE_ALIGNMENT;

#[rstest]
#[case::untracked(None, false)]
#[case::tracked(Some(0), false)]
#[case::indexed(None, true)]
#[case::tracked_and_indexed(Some(u16::MAX), true)]
fn attribute_info_serializes_tracking_and_indexing_combinations(
    #[case] maximum: Option<u16>,
    #[case] indexed: bool,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
) {
    let fixture = defined_addresses(offsets, maximum, indexed);
    assert_eq!(message(fixture).to_bytes(), fixture.build());
}

#[rstest]
#[case::untracked(None, false)]
#[case::tracked(Some(0), false)]
#[case::indexed(None, true)]
#[case::tracked_and_indexed(Some(u16::MAX), true)]
fn attribute_info_parses_tracking_and_indexing_combinations(
    #[case] maximum: Option<u16>,
    #[case] indexed: bool,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
) {
    let fixture = defined_addresses(offsets, maximum, indexed);
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&fixture.build(), offset_width(fixture))
            .unwrap(),
        message(fixture)
    );
}

#[rstest]
#[case::untracked(CreationOrderIndex::Absent)]
#[case::indexed(CreationOrderIndex::Undefined)]
fn attribute_info_serializes_undefined_addresses(
    #[case] creation: CreationOrderIndex,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
) {
    let fixture = AttributeInfo::new(offsets, None, None, None, creation);
    assert_eq!(message(fixture).to_bytes(), fixture.build());
}

#[rstest]
#[case::untracked(CreationOrderIndex::Absent)]
#[case::indexed(CreationOrderIndex::Undefined)]
fn attribute_info_parses_undefined_addresses(
    #[case] creation: CreationOrderIndex,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
) {
    let fixture = AttributeInfo::new(offsets, None, None, None, creation);
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&fixture.build(), offset_width(fixture))
            .unwrap(),
        message(fixture)
    );
}

#[rstest]
#[case::empty_prefix(
    &[],
    FormatError::IncompleteAttributeInfo {
        expected: 2,
        available: 0,
        field: AttributeInfoField::Version,
        field_offset: 0,
        field_size: 1,
        field_available: 0,
    }
)]
#[case::unsupported_version_without_flags(
    &[1],
    FormatError::IncompleteAttributeInfo {
        expected: 2,
        available: 1,
        field: AttributeInfoField::Flags,
        field_offset: 1,
        field_size: 1,
        field_available: 0,
    }
)]
#[case::unsupported_version_with_reserved_flags(&[1, 0xff], FormatError::InvalidAttributeInfoVersion(1))]
#[case::reserved_flags_without_body(&[0, 0xff], FormatError::InvalidAttributeInfoFlags(0xff))]
#[case::valid_prefix_without_body(
    &[0, 0x03],
    FormatError::IncompleteAttributeInfo {
        expected: 16,
        available: 2,
        field: AttributeInfoField::MaximumCreationIndex,
        field_offset: 2,
        field_size: 2,
        field_available: 0,
    }
)]
fn attribute_info_reports_framing_errors_in_acquisition_order(
    #[case] input: &[u8],
    #[case] expected: FormatError,
) {
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(input, OffsetWidth::Four).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::missing(2)]
#[case::partial(3)]
fn attribute_info_reports_an_incomplete_maximum_creation_index(#[case] available: usize) {
    let fixture = defined_addresses(Width::Two, Some(0x1234), false);
    assert_incomplete_field(
        fixture,
        available,
        AttributeInfoField::MaximumCreationIndex,
        2,
        2,
    );
}

#[rstest]
#[case::heap(0, AttributeInfoField::FractalHeapAddress, false)]
#[case::name_index(1, AttributeInfoField::NameIndexAddress, false)]
#[case::indexed_heap(0, AttributeInfoField::FractalHeapAddress, true)]
#[case::indexed_name_index(1, AttributeInfoField::NameIndexAddress, true)]
#[case::creation_order_index(2, AttributeInfoField::CreationOrderIndexAddress, true)]
fn attribute_info_reports_an_incomplete_address(
    #[case] address: usize,
    #[case] field: AttributeInfoField,
    #[case] indexed: bool,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
    #[values(None, Some(0x1234))] maximum: Option<u16>,
) {
    let fixture = defined_addresses(offsets, maximum, indexed);
    let width = usize::from(offsets.get());
    let start = maximum.map_or(2, |_| 4) + address * width;
    for available in start..start + width {
        assert_incomplete_field(fixture, available, field, start, width);
    }
}

#[test]
fn attribute_info_truncation_offsets_are_relative_to_the_supplied_body() {
    let fixture = defined_addresses(Width::Four, Some(0x1234), true);
    let mut body = fixture.build();
    body.truncate(14);
    let mut buffer = vec![0xa5; 3];
    buffer.extend(body);
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&buffer[3..], OffsetWidth::Four).unwrap_err(),
        FormatError::IncompleteAttributeInfo {
            expected: 16,
            available: 14,
            field: AttributeInfoField::CreationOrderIndexAddress,
            field_offset: 12,
            field_size: 4,
            field_available: 2,
        }
    );
}

#[test]
fn attribute_info_rejects_an_unsupported_version() {
    let fixture = defined_addresses(Width::Two, None, false);
    let mut body = fixture.build();
    body[0] = 1;
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&body, offset_width(fixture)).unwrap_err(),
        FormatError::InvalidAttributeInfoVersion(1)
    );
}

#[rstest]
#[case::bit_2(0x04)]
#[case::bit_3(0x08)]
#[case::bit_4(0x10)]
#[case::bit_5(0x20)]
#[case::bit_6(0x40)]
#[case::bit_7(0x80)]
#[case::all_reserved_bits(0xfc)]
fn attribute_info_rejects_reserved_flags(#[case] reserved: u8) {
    let fixture = defined_addresses(Width::Two, Some(17), true);
    let mut body = fixture.build();
    body[1] |= reserved;
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&body, offset_width(fixture)).unwrap_err(),
        FormatError::InvalidAttributeInfoFlags(body[1])
    );
}

#[test]
fn attribute_info_accepts_version_one_object_header_padding() {
    let fixture = defined_addresses(Width::Eight, Some(17), true);
    let mut body = fixture.build();
    body.resize(body.len().next_multiple_of(MESSAGE_ALIGNMENT), 0);
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&body, offset_width(fixture)).unwrap(),
        message(fixture)
    );
}

#[test]
fn attribute_info_round_trips_a_present_undefined_creation_order_index() {
    let fixture = AttributeInfo::new(
        Width::Two,
        Some(17),
        None,
        None,
        CreationOrderIndex::Undefined,
    );
    let info = message(fixture);
    let width = offset_width(fixture);
    let body = info.to_bytes();
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&body, width).unwrap(),
        info
    );
}

#[rstest]
#[case::two(Width::Two, 0x1_0000)]
#[case::four(Width::Four, 0x1_0000_0000)]
fn attribute_info_rejects_an_address_outside_the_selected_width(
    #[case] offsets: Width,
    #[case] address: u64,
    #[values(
        AddressField::Heap,
        AddressField::NameIndex,
        AddressField::CreationOrderIndex
    )]
    field: AddressField,
) {
    assert_eq!(
        field
            .construct(OffsetWidth::try_from(offsets.get()).unwrap(), address)
            .unwrap_err(),
        FormatError::AddressTooLarge {
            address,
            offset_size: offsets.get(),
        }
    );
}

#[rstest]
#[case::two(Width::Two, 0xffff)]
#[case::four(Width::Four, 0xffff_ffff)]
#[case::eight(Width::Eight, u64::MAX)]
fn attribute_info_rejects_a_defined_address_equal_to_the_undefined_marker(
    #[case] offsets: Width,
    #[case] address: u64,
    #[values(
        AddressField::Heap,
        AddressField::NameIndex,
        AddressField::CreationOrderIndex
    )]
    field: AddressField,
) {
    assert_eq!(
        field
            .construct(OffsetWidth::try_from(offsets.get()).unwrap(), address)
            .unwrap_err(),
        FormatError::DefinedAddressIsUndefined {
            address,
            offset_size: offsets.get(),
        }
    );
}

#[rstest]
#[case::heap_before_indexes(
    0xffff, 0x1_0001, 0x1_0002,
    FormatError::DefinedAddressIsUndefined { address: 0xffff, offset_size: 2 }
)]
#[case::name_before_creation_index(
    0, 0x1_0001, 0xffff,
    FormatError::AddressTooLarge { address: 0x1_0001, offset_size: 2 }
)]
fn attribute_info_checks_address_fits_in_wire_order(
    #[case] heap: u64,
    #[case] name: u64,
    #[case] creation: u64,
    #[case] expected: FormatError,
) {
    assert_eq!(
        AttributeInfoMessage::new(
            OffsetWidth::Two,
            None,
            DecodedStoredAddress::Defined(StoredAddress::new(heap)),
            DecodedStoredAddress::Defined(StoredAddress::new(name)),
            AttributeCreationOrderIndex::Defined(StoredAddress::new(creation)),
        )
        .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::two_byte_marker(Width::Four, 0xffff)]
#[case::four_byte_marker(Width::Eight, 0xffff_ffff)]
fn narrower_undefined_markers_are_defined_at_the_selected_width(
    #[case] offsets: Width,
    #[case] address: u64,
) {
    let fixture = AttributeInfo::new(
        offsets,
        None,
        Some(address),
        None,
        CreationOrderIndex::Absent,
    );
    let body = fixture.build();
    let decoded =
        AttributeInfoMessage::parse_with_offset_width(&body, offset_width(fixture)).unwrap();
    assert_eq!(
        decoded.fractal_heap_address(),
        Some(StoredAddress::new(address))
    );
    assert_eq!(message(fixture).to_bytes(), body);
}

#[rstest]
#[case::absent(CreationOrderIndex::Absent)]
#[case::undefined(CreationOrderIndex::Undefined)]
#[case::defined_at_zero(CreationOrderIndex::Defined(0))]
fn attribute_info_updates_creation_order_index_presence_and_address(
    #[case] index: CreationOrderIndex,
) {
    let fixture = AttributeInfo::new(Width::Two, Some(7), None, None, index);
    let mut info = message(AttributeInfo::new(
        Width::Two,
        Some(7),
        None,
        None,
        CreationOrderIndex::Defined(37),
    ));
    info.set_creation_order_index(creation_order_index(index))
        .unwrap();
    assert_eq!(info.creation_order_index(), creation_order_index(index));
    assert_eq!(info.to_bytes(), fixture.build());
}

#[rstest]
#[case::overflow(0x1_0000, FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 })]
#[case::undefined_marker(0xffff, FormatError::DefinedAddressIsUndefined { address: 0xffff, offset_size: 2 })]
fn attribute_info_keeps_the_index_after_a_failed_address_update(
    #[case] address: u64,
    #[case] expected: FormatError,
) {
    let mut info = message(defined_addresses(Width::Two, Some(7), true));
    let original = info;
    assert_eq!(
        info.set_creation_order_index(AttributeCreationOrderIndex::Defined(StoredAddress::new(
            address
        )))
        .unwrap_err(),
        expected
    );
    assert_eq!(info, original);
}

#[rstest]
#[case::untracked(None, CreationOrderIndex::Absent)]
#[case::tracked(Some(0x1234), CreationOrderIndex::Absent)]
#[case::indexed(None, CreationOrderIndex::Defined(0x789a))]
#[case::tracked_and_indexed(Some(u16::MAX), CreationOrderIndex::Defined(0x789a))]
#[case::undefined_creation_order_index(Some(0), CreationOrderIndex::Undefined)]
fn attribute_info_writes_only_its_encoding_to_an_unaligned_destination(
    #[case] maximum: Option<u16>,
    #[case] index: CreationOrderIndex,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
) {
    #[repr(C, align(8))]
    struct Aligned([u8; 32]);

    let fixture = AttributeInfo::new(offsets, maximum, Some(0x2345), None, index);
    let info = message(fixture);
    let expected = fixture.build();
    let len = expected.len();
    assert_eq!(info.encoded_size(), len);

    let mut output = Aligned([0xa5; 32]);
    assert_eq!(info.write_into(&mut output.0[1..]).unwrap(), len);
    assert_eq!(output.0[..1], [0xa5]);
    assert_eq!(output.0[1..1 + len], expected);
    assert_eq!(output.0[1 + len..], [0xa5; 32][1 + len..]);
}

#[test]
fn attribute_info_preserves_the_destination_and_can_be_reused_after_capacity_failure() {
    let fixture = defined_addresses(Width::Eight, Some(17), true);
    let info = message(fixture);
    let expected = fixture.build();
    let len = expected.len();

    for available in 0..len {
        let mut output = [0xa5; 32];
        assert_eq!(
            info.write_into(&mut output[1..1 + available]).unwrap_err(),
            AttributeInfoOutputTooSmall {
                required: len,
                available
            }
        );
        assert_eq!(output, [0xa5; 32]);

        let mut retry = vec![0x5a; len];
        assert_eq!(info.write_into(&mut retry).unwrap(), len);
        assert_eq!(retry, expected);
    }
}

fn defined_addresses(offsets: Width, maximum: Option<u16>, indexed: bool) -> AttributeInfo {
    let [heap, name, creation] = match offsets {
        Width::Two => [0x1234, 0x5678, 0xfffe],
        Width::Four => [0x1234_5678, 0x90ab_cdef, 0xffff_fffe],
        Width::Eight => [0x1234_5678_9abc_def0, 0, 0xffff_ffff_ffff_fffe],
    };
    AttributeInfo::new(
        offsets,
        maximum,
        Some(heap),
        Some(name),
        if indexed {
            CreationOrderIndex::Defined(creation)
        } else {
            CreationOrderIndex::Absent
        },
    )
}

fn message(fixture: AttributeInfo) -> AttributeInfoMessage {
    AttributeInfoMessage::new(
        offset_width(fixture),
        fixture.max_creation_index(),
        decoded_address(fixture.fractal_heap_address()),
        decoded_address(fixture.name_index_address()),
        creation_order_index(fixture.creation_order_index()),
    )
    .unwrap()
}

fn decoded_address(address: Option<u64>) -> DecodedStoredAddress {
    address.map_or(DecodedStoredAddress::Undefined, |bits| {
        DecodedStoredAddress::Defined(StoredAddress::new(bits))
    })
}

fn creation_order_index(index: CreationOrderIndex) -> AttributeCreationOrderIndex {
    match index {
        CreationOrderIndex::Absent => AttributeCreationOrderIndex::Absent,
        CreationOrderIndex::Defined(bits) => {
            AttributeCreationOrderIndex::Defined(StoredAddress::new(bits))
        }
        CreationOrderIndex::Undefined => AttributeCreationOrderIndex::Undefined,
    }
}

fn offset_width(fixture: AttributeInfo) -> OffsetWidth {
    OffsetWidth::try_from(fixture.offsets().get()).unwrap()
}

fn assert_incomplete_field(
    fixture: AttributeInfo,
    available: usize,
    field: AttributeInfoField,
    field_offset: usize,
    field_size: usize,
) {
    let mut body = fixture.build();
    let expected = body.len();
    body.truncate(available);
    assert_eq!(
        AttributeInfoMessage::parse_with_offset_width(&body, offset_width(fixture)).unwrap_err(),
        FormatError::IncompleteAttributeInfo {
            expected,
            available,
            field,
            field_offset,
            field_size,
            field_available: available - field_offset,
        }
    );
}

#[derive(Clone, Copy, Debug)]
enum AddressField {
    CreationOrderIndex,
    Heap,
    NameIndex,
}

impl AddressField {
    fn construct(self, width: OffsetWidth, bits: u64) -> Result<AttributeInfoMessage, FormatError> {
        let address = StoredAddress::new(bits);
        let (heap, name, index) = match self {
            Self::Heap => (Some(bits), None, AttributeCreationOrderIndex::Undefined),
            Self::NameIndex => (None, Some(bits), AttributeCreationOrderIndex::Undefined),
            Self::CreationOrderIndex => (None, None, AttributeCreationOrderIndex::Defined(address)),
        };
        AttributeInfoMessage::new(
            width,
            None,
            decoded_address(heap),
            decoded_address(name),
            index,
        )
    }
}
