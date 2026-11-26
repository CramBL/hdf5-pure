use alloc::collections::BTreeSet;

use rstest::rstest;
use test_util::free_space;

use super::*;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::free_space_manager::header::CHECKSUM_LEN;
use crate::free_space_manager::{
    self, FreeSpaceManagerHeader, SECTION_CLASS_LARGE, SECTION_CLASS_SIMPLE, SectionInfoCounts,
    test_support,
};

#[test]
fn serialize_matches_c_library_single_section() {
    // Byte-for-byte reproduction of the FSHD@619 / FSSE@701 fixtures,
    // including the Jenkins checksums the C library verifies on read.
    let fshd_fixture = free_space::single_section_header();
    let fsse_fixture = free_space::single_section_info();
    assert_eq!(
        free_space::raw_manager_eight(619, 701, &[(2848, 1600)]),
        (fshd_fixture.clone(), fsse_fixture.clone())
    );
    let (fshd, fsse) = serialize_free_space_manager(
        test_support::widths(8, 8),
        &[test_support::section(2848, 1600)],
        StoredAddress::new(619),
        StoredAddress::new(701),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
    assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
    // The header length helper agrees with the produced bytes.
    assert_eq!(
        free_space_manager::free_space_manager_header_len(test_support::widths(8, 8)),
        fshd.len() as u64
    );
}

#[test]
fn serialize_matches_c_library_two_sections() {
    // FSHD@736 / FSSE@818: two differently-sized sections (16 @871, 893 @1155)
    // emitted as two ascending size-groups.
    let fshd_fixture = free_space::two_section_header();
    let fsse_fixture = free_space::two_section_info();
    assert_eq!(
        free_space::raw_manager_eight(736, 818, &[(1155, 893), (871, 16)]),
        (fshd_fixture.clone(), fsse_fixture.clone())
    );
    let (fshd, fsse) = serialize_free_space_manager(
        test_support::widths(8, 8),
        &[
            test_support::section(1155, 893),
            test_support::section(871, 16),
        ],
        StoredAddress::new(736),
        StoredAddress::new(818),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
    assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
}

#[rstest]
fn a_manager_round_trips_at_the_widths_of_its_file(
    #[values(2, 4, 8)] offset_size: u8,
    #[values(2, 4, 8)] length_size: u8,
    #[values(&[][..], &[100], &[100, 100, 100], &[10, 20, 30], &[16384, 512, 512])] sizes: &[u64],
) {
    let widths = test_support::widths(offset_size, length_size);
    let sections: Vec<FreeSection> = sizes
        .iter()
        .zip(0u64..)
        .map(|(&size, i)| test_support::section(0x1000 + i * 0x4000, size))
        .collect();

    let size_groups = sizes.iter().collect::<BTreeSet<_>>().len().to_u64();
    let measured = SectionInfoCounts::new(sizes.len().to_u64(), size_groups)
        .unwrap()
        .encoded_len(widths)
        .unwrap();
    let (fshd, fsse) = serialize_free_space_manager(
        widths,
        &sections,
        StoredAddress::new(1000),
        StoredAddress::new(1100),
        SECTION_CLASS_LARGE,
    )
    .unwrap();

    assert_eq!(fsse.len().to_u64(), measured);
    let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
    assert_eq!(
        (fshd.len() as u64, fsse.len() as u64),
        (
            free_space_manager::free_space_manager_header_len(widths),
            free_space_manager::section_info_len(widths, sizes)
        )
    );
    assert_eq!(header.section_list_used(), fsse.len() as u64);
    let mut parsed =
        free_space_manager::parse_section_info(widths, &fsse, StoredAddress::new(1000), &header)
            .unwrap();
    parsed.sort_by_key(|s| s.addr);
    assert_eq!(parsed, sections);
}

// The manager of `four_byte_lengths_header`, with the serializer's maximum section size and
// each section size at the length width.
#[test]
fn a_mixed_width_header_holds_each_field_at_its_width() {
    let (fshd, fsse) = serialize_free_space_manager(
        test_support::widths(8, 4),
        &[test_support::section(0x0E40, 1600)],
        StoredAddress::new(0x0273),
        StoredAddress::new(0x02A9),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();

    assert_eq!(
        fshd[..fshd.len() - 4],
        [
            0x46, 0x53, 0x48, 0x44, 0x00, 0x01, 0x40, 0x06, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x50, 0x00, 0x78, 0x00,
            0x3F, 0x00, 0xFD, 0xFF, 0xFF, 0xFF, 0xA9, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x1F, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00,
        ]
    );
    assert_eq!(
        fsse[..fsse.len() - 4],
        [
            0x46, 0x53, 0x53, 0x45, 0x00, 0x73, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x40, 0x06, 0x00, 0x00, 0x40, 0x0E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
    );
}

#[rstest]
#[case::total_space(
    test_support::widths(8, 2),
    vec![test_support::section(0x1000, 0x7000), test_support::section(0x9000, 0x7000), test_support::section(0x11000, 0x7000)],
    StoredAddress::new(0x100),
    FormatError::LengthTooLarge { length: 0x1_5000, length_size: 2 },
)]
fn a_length_wider_than_its_field_returns_an_error(
    #[case] widths: FormatWidths,
    #[case] sections: Vec<FreeSection>,
    #[case] fsse_addr: StoredAddress,
    #[case] expected: FormatError,
) {
    assert_eq!(
        serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(0x80),
            fsse_addr,
            SECTION_CLASS_SIMPLE,
        ),
        Err(expected)
    );
}

#[rstest]
#[case::two_byte_lengths(test_support::widths(8, 2), 0x8000)]
#[case::four_byte_lengths(test_support::widths(8, 4), 0x8000_0000)]
#[case::eight_byte_lengths(test_support::widths(8, 8), 0x4000_0000_0000_0000)]
fn a_section_without_a_bin_returns_an_error(#[case] widths: FormatWidths, #[case] limit: u64) {
    let serialize = |size| {
        serialize_free_space_manager(
            widths,
            &[test_support::section(0x100, size)],
            StoredAddress::new(0x80),
            StoredAddress::new(0xC0),
            SECTION_CLASS_SIMPLE,
        )
    };

    let (fshd, _) = serialize(limit - 1).unwrap();
    let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
    assert_eq!(1 << header.max_section_size().ilog2(), limit);
    assert_eq!(
        serialize(limit),
        Err(FormatError::InvalidFreeSpaceSectionSize {
            size: limit,
            maximum: limit - 1,
        })
    );
}

#[rstest]
#[case::two_byte_lengths(test_support::widths(8, 2), 0xFFFD)]
#[case::four_byte_lengths(test_support::widths(8, 4), 0xFFFF_FFFD)]
fn the_written_maximum_section_size_is_below_the_truncated_driver_addresses(
    #[case] widths: FormatWidths,
    #[case] expected: u64,
) {
    let (fshd, _) = serialize_free_space_manager(
        widths,
        &[test_support::section(0x100, 0x10)],
        StoredAddress::new(0x80),
        StoredAddress::new(0xC0),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();

    assert_eq!(
        FreeSpaceManagerHeader::parse(widths, &fshd)
            .unwrap()
            .max_section_size(),
        expected
    );
}

// A 63-bit address space is sec2's, and a 32-bit one the core driver's on a 32-bit host.

#[rstest]
#[case::default_allocation(None, 0)]
#[case::exact_allocation(Some(0), 0)]
#[case::spare_allocation(Some(9), 9)]
fn serialization_records_the_requested_allocation(
    #[case] extra: Option<u64>,
    #[case] expected_extra: u64,
) {
    let widths = test_support::widths(8, 8);
    let sections = [FreeSection {
        addr: StoredAddress::new(4096),
        size: 512,
    }];
    let (_, prefix) = serialize_free_space_manager(
        widths,
        &sections,
        StoredAddress::new(1000),
        StoredAddress::new(1100),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    let used = prefix.len() as u64;
    let (header, list) = serialize_free_space_manager_with_allocation(
        widths,
        &sections,
        StoredAddress::new(1000),
        StoredAddress::new(1100),
        SECTION_CLASS_SIMPLE,
        extra.map(|extra| used + extra),
    )
    .unwrap();
    let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
    assert_eq!(header.section_list_used(), used + expected_extra);
    assert_eq!(header.section_list_allocated(), used + expected_extra);
    assert_eq!(list.len() as u64, used + expected_extra);
    let records_end = prefix.len() - CHECKSUM_LEN;
    assert_eq!(&list[..records_end], &prefix[..records_end]);
    assert!(
        list[records_end..list.len() - CHECKSUM_LEN]
            .iter()
            .all(|&byte| byte == 0)
    );
    assert_eq!(
        free_space_manager::parse_section_info(widths, &list, StoredAddress::new(1000), &header)
            .unwrap(),
        sections
    );
}

#[test]
fn serialization_rejects_an_allocation_shorter_than_the_section_list() {
    let widths = test_support::widths(8, 8);
    let sections = [FreeSection {
        addr: StoredAddress::new(4096),
        size: 512,
    }];
    let (_, list) = serialize_free_space_manager(
        widths,
        &sections,
        StoredAddress::new(1000),
        StoredAddress::new(1100),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    let used = list.len() as u64;
    let allocated = used - 1;
    assert_eq!(
        serialize_free_space_manager_with_allocation(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
            Some(allocated),
        )
        .unwrap_err(),
        FormatError::FreeSpaceSectionListAllocationTooSmall {
            required: used,
            allocated,
        }
    );
}

#[rstest]
#[case::two_byte_lengths(2, 1 << 16)]
#[case::four_byte_lengths(4, 1 << 32)]
fn serialization_rejects_an_allocation_outside_the_length_width(
    #[case] length_size: u8,
    #[case] allocated: u64,
) {
    assert_eq!(
        serialize_free_space_manager_with_allocation(
            test_support::widths(8, length_size),
            &[FreeSection {
                addr: StoredAddress::new(4096),
                size: 512,
            }],
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
            Some(allocated),
        )
        .unwrap_err(),
        FormatError::LengthTooLarge {
            length: allocated,
            length_size,
        }
    );
}

#[rstest]
#[case::section_list_address(
    test_support::widths(4, 4),
    vec![test_support::section(0x1000, 0x100)],
    StoredAddress::new(0x1_0000_0000),
    FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 },
)]
#[case::section_address(
    test_support::widths(2, 4),
    vec![test_support::section(0x1_0000, 0x100)],
    StoredAddress::new(0x100),
    FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 },
)]
fn an_address_wider_than_its_field_returns_an_error(
    #[case] widths: FormatWidths,
    #[case] sections: Vec<FreeSection>,
    #[case] fsse_addr: StoredAddress,
    #[case] expected: FormatError,
) {
    assert_eq!(
        serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(0x80),
            fsse_addr,
            SECTION_CLASS_SIMPLE,
        ),
        Err(expected)
    );
}

#[test]
fn serializing_sections_reports_the_addition_that_overflows_the_total() {
    let sections: Vec<FreeSection> = (0..5)
        .map(|i| test_support::section(0x1000 * (i + 1), (1 << 62) - 1))
        .collect();

    assert_eq!(
        serialize_free_space_manager(
            test_support::widths(8, 8),
            &sections,
            StoredAddress::new(0x80),
            StoredAddress::new(0x100),
            SECTION_CLASS_SIMPLE,
        ),
        Err(FormatError::FreeSpaceSectionSizeOverflow {
            total: u64::MAX - 3,
            size: (1 << 62) - 1,
        })
    );
}
