use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{FormatWidths, FreeSection, FreeSpaceManagerHeader};
use hdf5_pure_format::FormatError;
use rstest::rstest;
use test_util::free_space;

#[rstest]
#[case::incomplete_records_with_invalid_class(
    &[1, 100, 0, 16, 0, 0, 2, 101, 0, 32, 0, 3, 48, 0],
    &[section(16, 100)]
)]
#[case::incomplete_prefix_with_zero_count(
    &[1, 100, 0, 16, 0, 0, 1, 101, 0, 32, 0, 1, 0, 102],
    &[section(16, 100), section(32, 101)]
)]
fn file_sections_require_a_complete_group(
    #[case] records: &[u8],
    #[case] visited_prefix: &[FreeSection],
) {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let (header, list) = free_space::raw_manager_two(128, 256, 3, 302, records);
    let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
    let owner = StoredAddress::new(128);
    let error = FormatError::UnexpectedEof {
        expected: 22,
        available: 21,
    };
    assert_eq!(
        format::parse_section_info(widths, &list, owner, &header),
        Err(error.clone())
    );

    let mut visited = Vec::new();
    assert_eq!(
        format::visit_section_info(widths, &list, owner, &header, |section| visited
            .push(section)),
        Err(error)
    );
    assert_eq!(visited, visited_prefix);
}

#[rstest]
#[case::simple(0)]
#[case::small(1)]
#[case::large(2)]
fn parses_file_section_classes_with_unequal_widths(#[case] class: u8) {
    let widths = FormatWidths::from_sizes(8, 4).unwrap();
    let header =
        FreeSpaceManagerHeader::parse(widths, &free_space::four_byte_lengths_header()).unwrap();
    let mut list = free_space::four_byte_lengths_section_info();
    list[30] = class;
    let len = list.len();
    test_util::checksum::restamp(&mut list, 0, len);

    assert_eq!(
        format::parse_section_info(widths, &list, StoredAddress::new(627), &header),
        Ok(vec![section(3648, 1600)])
    );
}

#[rstest]
#[case::last_one_byte_count(255)]
#[case::first_two_byte_count(256)]
#[case::last_two_byte_count(65_535)]
#[case::first_three_byte_count(65_536)]
fn section_group_count_width_depends_on_total_section_count(#[case] count: u64) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let records = (0..count)
        .map(|index| (2000 + 13 * index, if index < count / 2 { 7 } else { 8 }))
        .collect::<Vec<_>>();
    let (header, list) = free_space::raw_manager_eight(619, 701, &records);
    let sections = records
        .iter()
        .map(|&(addr, size)| section(addr, size))
        .collect::<Vec<_>>();
    let (written_header, written_list) = format::serialize_free_space_manager(
        widths,
        &sections,
        StoredAddress::new(619),
        StoredAddress::new(701),
        format::SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    assert_eq!(written_header, header);
    assert_eq!(written_list, list);

    let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
    assert_eq!(
        format::parse_section_info(widths, &list, StoredAddress::new(619), &header),
        Ok(sections)
    );
}

#[test]
fn a_section_size_sum_above_u64_retains_the_complete_sum() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let (header, mut list) = free_space::raw_manager_eight(619, 701, &[(16, 7), (32, 7), (48, 7)]);
    let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
    let size = u64::try_from(i64::MAX).unwrap();
    test_util::bytes::set_slice_at(&mut list, 14, &size.to_le_bytes());
    let len = list.len();
    test_util::checksum::restamp(&mut list, 0, len);

    let mut visited = Vec::new();
    assert_eq!(
        format::visit_section_info(widths, &list, StoredAddress::new(619), &header, |section| {
            visited.push(section)
        }),
        Err(FormatError::InvalidFreeSpaceSectionSizeSum {
            expected: 21,
            actual: Box::new(3 * u128::from(size)),
        })
    );
    assert_eq!(
        visited,
        [section(16, size), section(32, size), section(48, size)]
    );
}

fn section(addr: u64, size: u64) -> FreeSection {
    FreeSection {
        addr: StoredAddress::new(addr),
        size,
    }
}

#[rstest]
#[case::simple(0, 0)]
#[case::small(1, 5)]
#[case::large(2, 9)]
fn prepared_manager_blocks_match_independent_bytes(#[case] class: u8, #[case] padding: usize) {
    let widths = FormatWidths::from_sizes(4, 2).unwrap();
    let plan = format::PreparedFreeSpaceManager::new(
        widths,
        vec![section(0x10200, 8), section(0x20000, 3), section(0xFF00, 8)],
        class,
    )
    .unwrap();
    assert_eq!(
        plan.sections().collect::<Vec<_>>(),
        [section(0x20000, 3), section(0xFF00, 8), section(0x10200, 8)]
    );

    assert_eq!(plan.section_count(), 3);

    let mut expected_list = vec![
        0x46, 0x53, 0x53, 0x45, 0, 0x20, 0x03, 0, 0, 1, 3, 0, 0, 0, 2, 0, class, 2, 8, 0, 0, 0xFF,
        0, 0, class, 0, 2, 1, 0, class,
    ];
    let required = expected_list.len() + 4;
    expected_list.resize(expected_list.len() + padding, 0);
    test_util::checksum::append(&mut expected_list);
    let allocation = u16::try_from(expected_list.len()).unwrap();
    let mut expected_header = vec![
        0x46, 0x53, 0x48, 0x44, 0, 1, 19, 0, 3, 0, 3, 0, 0, 0, 3, 0, 80, 0, 120, 0, 31, 0, 0xFD,
        0xFF, 0x80, 0, 1, 0,
    ];
    expected_header.extend_from_slice(&allocation.to_le_bytes());
    expected_header.extend_from_slice(&allocation.to_le_bytes());
    test_util::checksum::append(&mut expected_header);
    assert_eq!(plan.header_len(), expected_header.len());
    assert_eq!(plan.section_list_len(), required);

    let header_address = StoredAddress::new(0x320);
    let section_list_address = StoredAddress::new(0x1_0080);
    let placed = plan
        .place(
            header_address,
            section_list_address,
            Some(u64::from(allocation)),
        )
        .unwrap();
    let header = placed.header();
    let list = placed.section_list();

    let mut header_output = vec![0xA5; header.encoded_size() + 7];
    assert_eq!(
        header.write_into(&mut header_output).unwrap(),
        expected_header.len()
    );
    assert_eq!(&header_output[..header.encoded_size()], &expected_header);
    assert_eq!(&header_output[header.encoded_size()..], &[0xA5; 7]);

    let mut list_output = vec![0x5A; list.encoded_size() + 5];
    assert_eq!(
        list.write_into(&mut list_output).unwrap(),
        expected_list.len()
    );
    assert_eq!(&list_output[..list.encoded_size()], &expected_list);
    assert_eq!(&list_output[list.encoded_size()..], &[0x5A; 5]);
}

#[test]
fn short_manager_block_outputs_stay_unchanged() {
    let widths = FormatWidths::from_sizes(2, 4).unwrap();
    let plan = format::PreparedFreeSpaceManager::new(widths, vec![section(0x200, 5)], 0).unwrap();
    let placed = plan
        .place(
            StoredAddress::new(0x100),
            StoredAddress::new(0x500),
            Some(32),
        )
        .unwrap();
    let header = placed.header();
    let list = placed.section_list();
    let check = |length, write: &dyn Fn(&mut [u8]) -> Result<usize, FormatError>| {
        for available in 0..length {
            let mut output = vec![0xA5; length];
            assert_eq!(
                write(&mut output[..available]),
                Err(FormatError::UnexpectedEof {
                    expected: length,
                    available,
                })
            );
            assert_eq!(output, vec![0xA5; length]);
        }
    };
    check(header.encoded_size(), &|output| header.write_into(output));
    check(list.encoded_size(), &|output| list.write_into(output));
}

#[rstest]
#[case::zero(
    section(0x280, 0), 0,
    FormatError::InvalidFreeSpaceSectionSize { size: 0, maximum: 0x7FFF }
)]
#[case::past_size_bin_limit(
    section(0x280, 0x8000), 0,
    FormatError::InvalidFreeSpaceSectionSize { size: 0x8000, maximum: 0x7FFF }
)]
#[case::unsupported_class(
    section(0x280, 64), 3,
    FormatError::UnsupportedSerializedSectionClass { client_id: 1, class_id: 3 }
)]
#[case::address_outside_width(
    section(0x1_0000, 64), 0,
    FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 }
)]
#[case::undefined_address(
    section(0xFFFF, 64), 0,
    FormatError::DefinedAddressIsUndefined { address: 0xFFFF, offset_size: 2 }
)]
fn manager_preparation_rejects_invalid_fields(
    #[case] second: FreeSection,
    #[case] class: u8,
    #[case] expected: FormatError,
) {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    assert_eq!(
        format::PreparedFreeSpaceManager::new(widths, vec![section(0x200, 80), second], class)
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::undefined_header_address(
    0xFFFF, 0x100,
    FormatError::DefinedAddressIsUndefined { address: 0xFFFF, offset_size: 2 }
)]
#[case::undefined_section_list_address(
    0x100, 0xFFFF,
    FormatError::DefinedAddressIsUndefined { address: 0xFFFF, offset_size: 2 }
)]
#[case::section_list_address_outside_width(
    0x100, 0x1_0000,
    FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 }
)]
fn nonempty_manager_placement_rejects_invalid_block_addresses(
    #[case] header: u64,
    #[case] list: u64,
    #[case] expected: FormatError,
) {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let plan = format::PreparedFreeSpaceManager::new(widths, vec![section(0x200, 5)], 0).unwrap();
    assert_eq!(
        plan.place(StoredAddress::new(header), StoredAddress::new(list), None)
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::larger_than_a_slice(
    isize::MAX as u64 + 1,
    FormatError::ValueTooLargeForPlatform {
        value: isize::MAX as u64 + 1,
        target: "slice length",
    }
)]
#[cfg(target_pointer_width = "32")]
#[case::larger_than_usize(
    u64::from(u32::MAX) + 1,
    FormatError::ValueTooLargeForPlatform {
        value: u64::from(u32::MAX) + 1,
        target: "usize",
    }
)]
fn manager_placement_rejects_unrepresentable_allocations(
    #[case] allocated: u64,
    #[case] expected: FormatError,
) {
    let widths = FormatWidths::from_sizes(4, 8).unwrap();
    let plan = format::PreparedFreeSpaceManager::new(widths, vec![section(0x200, 5)], 0).unwrap();
    assert_eq!(
        plan.place(
            StoredAddress::new(0x100),
            StoredAddress::new(0x500),
            Some(allocated),
        )
        .unwrap_err(),
        expected
    );
}
