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
    let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
    let expected = records
        .iter()
        .map(|&(addr, size)| section(addr, size))
        .collect::<Vec<_>>();

    assert_eq!(
        format::parse_section_info(widths, &list, StoredAddress::new(619), &header),
        Ok(expected)
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
