use hdf5_pure_core::__private::FileRegion;
use rstest::rstest;
use test_util::free_space;

use super::*;
use crate::address::StoredAddressExt;
use crate::free_space_manager::{
    self, CHECKSUM_LEN, FILE_FSM_MAX_SECTION_SIZE, FreeSection, SECTION_CLASS_SIMPLE, section_list,
    test_support,
};

#[rstest]
#[case::two(
    test_support::widths(2, 2),
    free_space::two_byte_widths_header(),
    free_space::two_byte_widths_section_info(),
    0x01E5,
    0x0207
)]
#[case::four(
    test_support::widths(4, 4),
    free_space::four_byte_widths_header(),
    free_space::four_byte_widths_section_info(),
    0x0217,
    0x0249
)]
#[case::mixed(
    test_support::widths(8, 4),
    free_space::four_byte_lengths_header(),
    free_space::four_byte_lengths_section_info(),
    0x0273,
    0x02A9
)]
#[case::eight(
    test_support::widths(8, 8),
    free_space::single_section_header(),
    free_space::single_section_info(),
    619,
    701
)]
fn region_sequence_parses_c_library_manager(
    #[case] widths: FormatWidths,
    #[case] header_bytes: Vec<u8>,
    #[case] list_bytes: Vec<u8>,
    #[case] header_at: u64,
    #[case] list_at: u64,
) {
    use hdf5_pure_core::__private::BaseAddressExt;
    use hdf5_pure_core::BaseAddress;

    use crate::__private::FreeSpaceManagerHeaderRef;

    let base = BaseAddress::new(512);
    let header_addr = StoredAddress::new(header_at);
    let header_ref =
        FreeSpaceManagerHeaderRef::new(widths, base.resolve(header_addr).unwrap()).unwrap();
    assert_eq!(header_ref.range(), header_ref.parse_range());
    let frame = header_ref
        .parse(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
        .unwrap();
    let max_section_size = if widths.lengths == LengthWidth::Eight {
        FILE_FSM_MAX_SECTION_SIZE
    } else {
        width::length_width_max(widths.lengths)
    };
    assert_eq!(
        FreeSpaceManagerHeader::from_frame(&frame).unwrap(),
        FreeSpaceManagerHeader {
            total_space: 1600,
            total_sections: 1,
            addr_space_bits: AddressSpaceBits::new(63).unwrap(),
            max_section_size,
            section_list_addr: StoredAddress::new(list_at),
            section_list_used: list_bytes.len() as u64,
            section_list_allocated: list_bytes.len() as u64,
        }
    );
    assert_eq!(frame.client_id(), FILE_FSM_CLIENT_ID);
    assert_eq!(frame.total_space(), 1600);
    assert_eq!(frame.total_sections(), 1);
    assert_eq!(frame.serialized_sections(), 1);
    assert_eq!(frame.unserialized_sections(), 0);
    assert_eq!(frame.section_classes(), FILE_FSM_NUM_CLASSES);
    assert_eq!(frame.shrink_percent(), FILE_FSM_SHRINK_PCT);
    assert_eq!(frame.expand_percent(), FILE_FSM_EXPAND_PCT);
    assert_eq!(frame.addr_space_bits(), 63);
    assert_eq!(frame.max_section_size(), max_section_size);
    let DecodedStoredAddress::Defined(list_addr) = frame.section_list_addr() else {
        panic!("expected a defined section list")
    };
    assert_eq!(list_addr.get(), list_at);
    assert_eq!(frame.section_list_used(), list_bytes.len() as u64);
    assert_eq!(frame.section_list_allocated(), list_bytes.len() as u64);
    let list_ref = frame
        .section_list_ref(header_addr, base.resolve(list_addr).unwrap())
        .unwrap();
    let view = list_ref
        .parse(FileRegion::new(list_ref.parse_range(), &list_bytes).unwrap())
        .unwrap();
    assert_eq!(list_ref.range(), list_ref.parse_range());
    assert_eq!(
        view.records(),
        &list_bytes[section_list::section_list_prefix_len(widths.offsets)
            ..list_bytes.len() - CHECKSUM_LEN]
    );
    assert_eq!(
        view.records().as_ptr(),
        list_bytes[section_list::section_list_prefix_len(widths.offsets)..].as_ptr()
    );
}

#[rstest]
#[case::heap(test_support::FRACTAL_HEAP_FSM_CLIENT_ID)]
#[case::reserved(2)]
fn file_semantics_reject_another_client_after_framing(#[case] client: u8) {
    let mut bytes = free_space::single_section_header();
    bytes[5] = client;
    let len = bytes.len();
    test_util::checksum::restamp(&mut bytes, 0, len);
    let frame = FreeSpaceManagerHeaderFrame::parse(test_support::widths(8, 8), &bytes).unwrap();
    assert_eq!(frame.client_id(), client);
    assert_eq!(
        FreeSpaceManagerHeader::from_frame(&frame).unwrap_err(),
        FormatError::InvalidFreeSpaceManager(format!(
            "the client ID is {client}, not 1, the ID of the file client"
        ))
    );
}

#[rstest]
#[case::two_byte_widths(
    test_support::widths(2, 2),
    free_space::two_byte_widths_header(),
    free_space::two_byte_widths_section_info(),
    0x01E5,
    0x0207,
    29
)]
#[case::four_byte_widths(
    test_support::widths(4, 4),
    free_space::four_byte_widths_header(),
    free_space::four_byte_widths_section_info(),
    0x0217,
    0x0249,
    31
)]
#[case::four_byte_lengths(
    test_support::widths(8, 4),
    free_space::four_byte_lengths_header(),
    free_space::four_byte_lengths_section_info(),
    0x0273,
    0x02A9,
    35
)]
fn a_c_library_manager_parses_at_the_widths_of_its_file(
    #[case] widths: FormatWidths,
    #[case] header_bytes: Vec<u8>,
    #[case] section_list: Vec<u8>,
    #[case] header_addr: u64,
    #[case] section_list_addr: u64,
    #[case] section_list_used: u64,
) {
    let header = FreeSpaceManagerHeader::parse(widths, &header_bytes).unwrap();

    assert_eq!(
        header,
        FreeSpaceManagerHeader {
            total_space: 1600,
            total_sections: 1,
            addr_space_bits: AddressSpaceBits::new(63).unwrap(),
            max_section_size: width::length_width_max(widths.lengths),
            section_list_addr: StoredAddress::new(section_list_addr),
            section_list_used,
            section_list_allocated: section_list_used,
        }
    );
    assert_eq!(
        free_space_manager::free_space_manager_header_len(widths),
        header_bytes.len() as u64
    );
    assert_eq!(
        free_space_manager::parse_section_info(
            widths,
            &section_list,
            StoredAddress::new(header_addr),
            &header
        ),
        Ok(vec![test_support::section(3648, 1600)])
    );
}

#[rstest]
#[case::all_ones(0xFFFF_FFFF, 63, free_space::four_byte_widths_section_info())]
#[case::all_ones_less_one(0xFFFF_FFFE, 63, free_space::four_byte_widths_section_info())]
#[case::four_byte_address_space(
    0xFFFF_FFFE,
    32,
    test_support::section_list(test_support::widths(4, 4), 0x0217, &[0x01, 0x40, 0x06, 0x00, 0x00, 0x40, 0x0E, 0x00, 0x00, 0x00])
)]
fn section_sizes_under_a_truncated_driver_address_are_as_wide_as_section_offsets(
    #[case] max_section_size: u64,
    #[case] addr_space_bits: u8,
    #[case] list: Vec<u8>,
) {
    let widths = test_support::widths(4, 4);
    let header = FreeSpaceManagerHeader {
        max_section_size,
        addr_space_bits: AddressSpaceBits::new(u16::from(addr_space_bits)).unwrap(),
        section_list_used: list.len() as u64,
        section_list_allocated: list.len() as u64,
        ..FreeSpaceManagerHeader::parse(widths, &free_space::four_byte_widths_header()).unwrap()
    };

    assert_eq!(
        free_space_manager::parse_section_info(widths, &list, StoredAddress::new(0x0217), &header),
        Ok(vec![test_support::section(3648, 1600)])
    );
}

#[rstest]
#[case::zero(&[0, 0, 0])]
#[case::nonzero(&[0, 7, 0])]
fn unused_bytes_between_the_sections_and_the_checksum_are_accepted(#[case] padding: &[u8]) {
    // One set of one section: a 1-byte count, an 8-byte size, an 8-byte offset and a class.
    let payload = [
        &[1][..],
        &100u64.to_le_bytes(),
        &0x1000u64.to_le_bytes(),
        &[SECTION_CLASS_SIMPLE],
        padding,
    ]
    .concat();
    let list = test_support::section_list(test_support::widths(8, 8), 0x80, &payload);

    assert_eq!(
        free_space_manager::parse_section_info(
            test_support::widths(8, 8),
            &list,
            StoredAddress::new(0x80),
            &FreeSpaceManagerHeader {
                total_space: 100,
                ..header_of(&list, 1)
            }
        ),
        Ok(vec![test_support::section(0x1000, 100)])
    );
}

#[rstest]
#[case::missing_group_prefix(
    2,
    &[0x01, 0x64, 0x00, 0x10, 0x00, 0x00],
    16,
    13
)]
#[case::partial_group_prefix(1, &[0x01, 0x64], 10, 9)]
#[case::partial_group_prefix_with_zero_count(1, &[0x00, 0x64], 10, 9)]
#[case::missing_record(
    2,
    &[0x02, 0x64, 0x00, 0x10, 0x00, 0x00],
    16,
    13
)]
#[case::missing_class(1, &[0x01, 0x64, 0x00, 0x10, 0x00], 13, 12)]
fn truncated_section_groups_report_the_required_end(
    #[case] total_sections: u64,
    #[case] payload: &[u8],
    #[case] expected: usize,
    #[case] available: usize,
) {
    // A set is a 1-byte count and a 2-byte size, and a section a 2-byte offset and a class.
    let widths = test_support::widths(2, 2);
    let list = test_support::section_list(widths, 0x80, payload);
    let header = FreeSpaceManagerHeader {
        max_section_size: 0xFFFD,
        addr_space_bits: AddressSpaceBits::new(15).unwrap(),
        ..header_of(&list, total_sections)
    };

    assert_eq!(
        free_space_manager::parse_section_info(widths, &list, StoredAddress::new(0x80), &header),
        Err(FormatError::UnexpectedEof {
            expected,
            available
        })
    );
}

#[rstest]
#[case::invalid_second_class(
    &[0x02, 0x64, 0x00, 0x10, 0x00, 0x00, 0x20, 0x00, 0x03], 2,
    FormatError::InvalidFreeSpaceSectionClass { offset: 13, class_id: 3 },
    &[test_support::section(16, 100)]
)]
#[case::excess_count(
    &[0x02, 0x64, 0x00, 0x10, 0x00, 0x00, 0x20, 0x00, 0x00], 1,
    FormatError::InvalidFreeSpaceSectionGroupCount { offset: 7, count: 2, remaining: 1 },
    &[]
)]
fn section_visits_stop_at_an_invalid_count_or_class(
    #[case] payload: &[u8],
    #[case] total: u64,
    #[case] expected: FormatError,
    #[case] expected_visits: &[FreeSection],
) {
    let widths = test_support::widths(2, 2);
    let list = test_support::section_list(widths, 0x80, payload);
    let header = FreeSpaceManagerHeader {
        max_section_size: 0xFFFD,
        addr_space_bits: AddressSpaceBits::new(15).unwrap(),
        ..header_of(&list, total)
    };
    let mut visited = Vec::new();
    assert_eq!(
        free_space_manager::visit_section_info(
            widths,
            &list,
            StoredAddress::new(0x80),
            &header,
            |section| visited.push(section)
        ),
        Err(expected.clone())
    );
    assert_eq!(visited, expected_visits);
    assert_eq!(
        free_space_manager::parse_section_info(widths, &list, StoredAddress::new(0x80), &header),
        Err(expected)
    );
}

#[rstest]
#[case::fractal_heap_client(
    vec![(5, vec![0])],
    "the client ID is 0, not 1, the ID of the file client"
)]
#[case::unserialized_sections(
    vec![(30, vec![1, 0, 0, 0, 0, 0, 0, 0])],
    "the header's section count is 1, its serialized section count 1 and its unserialized \
     section count 1, and the file client serializes every section"
)]
#[case::more_section_classes_than_the_file_client(
    vec![(38, vec![4, 0])],
    "the header registers 4 section classes, more than the 3 of the file client"
)]
#[case::empty_address_space(vec![(44, vec![0, 0])], "an address space of 0 bits is not 1 to 64 bits wide")]
#[case::address_space_wider_than_eight_bytes(
    vec![(44, vec![72, 0])],
    "an address space of 72 bits is not 1 to 64 bits wide"
)]
#[case::section_list_used_past_its_allocation(
    vec![(70, vec![34, 0, 0, 0, 0, 0, 0, 0])],
    "the section list uses 35 bytes, more than the 34 bytes allocated to it"
)]
#[case::empty_manager_used_past_its_allocation(
    vec![(6, vec![0; 8]), (14, vec![0; 8]), (22, vec![0; 8]), (70, vec![34, 0, 0, 0, 0, 0, 0, 0])],
    "the section list uses 35 bytes, more than the 34 bytes allocated to it"
)]
#[case::empty_manager_with_free_bytes(
    vec![(14, vec![0; 8]), (22, vec![0; 8]), (54, vec![0xFF; 8])],
    "the empty manager tracks 1600 bytes"
)]
#[case::no_section_list(
    vec![(54, vec![0xFF; 8])],
    "the header's section count is 1, and it stores no section list"
)]
#[case::more_sections_than_the_section_list_holds(
    vec![(14, vec![2, 0, 0, 0, 0, 0, 0, 0]), (22, vec![2, 0, 0, 0, 0, 0, 0, 0])],
    "the header's section count is 2, and a section list of 35 bytes holds at most 1"
)]
#[case::the_largest_section_count(
    vec![(14, vec![0xFF; 8]), (22, vec![0xFF; 8])],
    "the header's section count is 18446744073709551615, and a section list of 35 bytes holds \
     at most 0"
)]
fn an_invalid_header_fails_to_parse(#[case] fields: Vec<(usize, Vec<u8>)>, #[case] reason: &str) {
    let mut fshd = free_space::single_section_header();
    for (at, value) in fields {
        test_util::bytes::set_slice_at(&mut fshd, at, &value);
    }
    let len = fshd.len();
    test_util::checksum::restamp(&mut fshd, 0, len);

    assert_eq!(
        FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd),
        Err(FormatError::InvalidFreeSpaceManager(reason.to_owned()))
    );
}

#[test]
fn an_empty_manager_header_retains_its_section_list_geometry() {
    let mut fshd = free_space::single_section_header();
    for (at, value) in [
        (6, [0; 8]),
        (14, [0; 8]),
        (22, [0; 8]),
        (54, [0xFF; 8]),
        (62, 17u64.to_le_bytes()),
        (70, 17u64.to_le_bytes()),
    ] {
        test_util::bytes::set_slice_at(&mut fshd, at, &value);
    }
    let len = fshd.len();
    test_util::checksum::restamp(&mut fshd, 0, len);

    assert_eq!(
        FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd),
        Ok(FreeSpaceManagerHeader {
            total_space: 0,
            total_sections: 0,
            addr_space_bits: AddressSpaceBits::new(63).unwrap(),
            max_section_size: FILE_FSM_MAX_SECTION_SIZE,
            section_list_addr: StoredAddress::new(u64::MAX),
            section_list_used: 17,
            section_list_allocated: 17,
        })
    );
}

/// Returns a header of `total_sections` sections whose section list is `list`, with the
/// address space and the maximum section size of the sec2 driver.
fn header_of(list: &[u8], total_sections: u64) -> FreeSpaceManagerHeader {
    FreeSpaceManagerHeader {
        total_space: 0,
        total_sections,
        addr_space_bits: AddressSpaceBits::new(63).unwrap(),
        max_section_size: FILE_FSM_MAX_SECTION_SIZE,
        section_list_addr: StoredAddress::new(0x100),
        section_list_used: list.len() as u64,
        section_list_allocated: list.len() as u64,
    }
}
