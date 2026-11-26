use rstest::rstest;
use test_util::free_space;

use super::*;
#[cfg(feature = "checksum")]
use crate::checksum;
use crate::free_space_manager::{self, SECTION_CLASS_SIMPLE, test_support};

#[test]
fn parses_c_library_single_section() {
    // manager @619: one 1600-byte free section at offset 2848, FSSE @701.
    let fshd = free_space::single_section_header();
    let header = FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd).unwrap();
    assert_eq!(header.total_space(), 1600);
    assert_eq!(header.total_sections(), 1);
    assert_eq!(header.addr_space_bits(), 63);
    assert_eq!(header.section_list_addr(), StoredAddress::new(701));
    assert_eq!(header.section_list_used(), 35);

    let fsse = free_space::single_section_info();
    let sections = parse_section_info(
        test_support::widths(8, 8),
        &fsse,
        StoredAddress::new(619),
        &header,
    )
    .unwrap();
    assert_eq!(sections, vec![test_support::section(2848, 1600)]);
}

#[test]
fn parses_c_library_two_sections() {
    // manager @736: 16 bytes @871 and 893 bytes @1155, FSSE @818.
    let fshd = free_space::two_section_header();
    let header = FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd).unwrap();
    assert_eq!(header.total_space(), 909);
    assert_eq!(header.total_sections(), 2);
    assert_eq!(header.section_list_addr(), StoredAddress::new(818));
    assert_eq!(header.section_list_used(), 53);

    let fsse = free_space::two_section_info();
    let mut sections = parse_section_info(
        test_support::widths(8, 8),
        &fsse,
        StoredAddress::new(736),
        &header,
    )
    .unwrap();
    sections.sort_by_key(|s| s.addr);
    assert_eq!(
        sections,
        vec![
            test_support::section(871, 16),
            test_support::section(1155, 893),
        ]
    );
    // The section sizes sum to the header's tracked total.
    let total: u64 = sections.iter().map(|s| s.size).sum();
    assert_eq!(header.total_space(), total);
}

#[rstest]
#[case::two_byte_widths(test_support::widths(2, 2))]
#[case::four_byte_widths(test_support::widths(4, 4))]
#[case::mixed_widths(test_support::widths(4, 8))]
#[case::eight_byte_widths(test_support::widths(8, 8))]
fn section_list_capacity_is_independent_of_its_parse_length(#[case] widths: FormatWidths) {
    let sections = vec![test_support::section(0x1000, 100)];
    let (mut fshd, fsse) = free_space_manager::serialize_free_space_manager(
        widths,
        &sections,
        StoredAddress::new(0x80),
        StoredAddress::new(0x100),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
    assert_eq!(header.section_list_allocated(), fsse.len().to_u64());
    let allocated = header.section_list_allocated() + 16;
    let checksum_start = fshd.len() - CHECKSUM_LEN;
    let allocated_start = checksum_start - usize::from(widths.lengths.get());
    let field = &mut fshd[allocated_start..checksum_start];
    field.copy_from_slice(&allocated.to_le_bytes()[..field.len()]);
    fshd.truncate(checksum_start);
    test_util::checksum::append(&mut fshd);
    let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
    assert_eq!(header.section_list_used(), fsse.len().to_u64());
    assert_eq!(header.section_list_allocated(), fsse.len().to_u64() + 16);
    assert_eq!(
        parse_section_info(widths, &fsse, StoredAddress::new(0x80), &header).unwrap(),
        sections
    );
    assert_eq!(
        parse_section_info(
            widths,
            &fsse[..fsse.len() - 1],
            StoredAddress::new(0x80),
            &header
        )
        .unwrap_err(),
        FormatError::UnexpectedEof {
            expected: fsse.len(),
            available: fsse.len() - 1,
        }
    );
}

#[rstest]
#[case::zero(0)]
#[case::nonzero(0xd0)]
fn an_oversized_c_library_section_list_parses_its_declared_sections(#[case] unused: u8) {
    let header = FreeSpaceManagerHeader::parse(
        test_support::widths(8, 8),
        &free_space::oversized_section_list_header(),
    )
    .unwrap();

    let mut list = free_space::oversized_section_list();
    list[185] = unused;
    let len = list.len();
    test_util::checksum::restamp(&mut list, 0, len);

    let expected = vec![
        test_support::section(842, 10),
        test_support::section(5375, 10),
        test_support::section(474, 24),
        test_support::section(2444, 27),
        test_support::section(1595, 47),
        test_support::section(130, 74),
        test_support::section(2618, 194),
        test_support::section(3793, 254),
        test_support::section(1122, 326),
        test_support::section(4385, 393),
    ];
    assert_eq!(
        parse_section_info(
            test_support::widths(8, 8),
            &list,
            StoredAddress::new(48),
            &header
        ),
        Ok(expected.clone())
    );
    let mut visited = Vec::new();
    visit_section_info(
        test_support::widths(8, 8),
        &list,
        StoredAddress::new(48),
        &header,
        |section| visited.push(section),
    )
    .unwrap();
    assert_eq!(visited, expected);
}

#[rstest]
#[case::byte_184(184)]
#[case::byte_185(185)]
#[case::byte_186(186)]
#[case::byte_187(187)]
#[case::byte_188(188)]
#[case::byte_189(189)]
#[case::byte_190(190)]
#[case::byte_191(191)]
#[case::byte_192(192)]
fn each_unused_section_list_byte_accepts_every_checksummed_value(#[case] at: usize) {
    let widths = test_support::widths(8, 8);
    let header =
        FreeSpaceManagerHeader::parse(widths, &free_space::oversized_section_list_header())
            .unwrap();
    let expected = [
        test_support::section(842, 10),
        test_support::section(5375, 10),
        test_support::section(474, 24),
        test_support::section(2444, 27),
        test_support::section(1595, 47),
        test_support::section(130, 74),
        test_support::section(2618, 194),
        test_support::section(3793, 254),
        test_support::section(1122, 326),
        test_support::section(4385, 393),
    ];
    for value in 0..=u8::MAX {
        let mut list = free_space::oversized_section_list();
        list[at] = value;
        let len = list.len();
        test_util::checksum::restamp(&mut list, 0, len);

        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(48), &header).as_deref(),
            Ok(expected.as_slice()),
            "byte {value:#04x} at offset {at}",
        );
    }
}

#[cfg(feature = "checksum")]
#[test]
fn unused_section_list_bytes_remain_covered_by_the_checksum() {
    let widths = test_support::widths(8, 8);
    let header =
        FreeSpaceManagerHeader::parse(widths, &free_space::oversized_section_list_header())
            .unwrap();
    let mut list = free_space::oversized_section_list();
    list[185] = 0xd0;
    let at = list.len() - 4;
    let expected = u32::from_le_bytes(list[at..].try_into().unwrap());
    let computed = checksum::jenkins_lookup3(&list[..at]);
    assert_ne!(expected, computed);

    assert_eq!(
        parse_section_info(widths, &list, StoredAddress::new(48), &header),
        Err(FormatError::ChecksumMismatch { expected, computed }),
    );
}

#[rstest]
#[case::a_section_list_of_another_header(
    5,
    &736u64.to_le_bytes(),
    FormatError::InvalidFreeSpaceManager(
        "the section list belongs to the header at 736, not to the header at 619".into()
    ),
    &[]
)]
#[case::a_set_of_no_sections(
    13,
    &[0],
    FormatError::InvalidFreeSpaceSectionGroupCount { offset: 13, count: 0, remaining: 1 },
    &[]
)]
#[case::a_set_of_empty_sections(
    14,
    &[0; 8],
    FormatError::InvalidFreeSpaceSectionGroupSize { offset: 13 },
    &[]
)]
#[case::a_class_the_file_client_does_not_register(
    30,
    &[3],
    FormatError::InvalidFreeSpaceSectionClass { offset: 22, class_id: 3 },
    &[]
)]
#[case::sections_that_sum_past_the_total_space(
    14,
    &1601u64.to_le_bytes(),
    FormatError::InvalidFreeSpaceSectionSizeSum { expected: 1600, actual: Box::new(1601) },
    &[test_support::section(2848, 1601)]
)]
fn an_invalid_section_list_fails_to_parse(
    #[case] at: usize,
    #[case] value: &[u8],
    #[case] expected: FormatError,
    #[case] expected_visits: &[FreeSection],
) {
    let header = FreeSpaceManagerHeader::parse(
        test_support::widths(8, 8),
        &free_space::single_section_header(),
    )
    .unwrap();
    let mut fsse = free_space::single_section_info();
    test_util::bytes::set_slice_at(&mut fsse, at, value);
    let len = fsse.len();
    test_util::checksum::restamp(&mut fsse, 0, len);

    assert_eq!(
        parse_section_info(
            test_support::widths(8, 8),
            &fsse,
            StoredAddress::new(619),
            &header
        ),
        Err(expected.clone())
    );
    let mut visited = Vec::new();
    let error = visit_section_info(
        test_support::widths(8, 8),
        &fsse,
        StoredAddress::new(619),
        &header,
        |section| visited.push(section),
    )
    .unwrap_err();
    assert_eq!(error, expected);
    assert_eq!(visited, expected_visits);
}

#[test]
fn a_section_list_shorter_than_its_used_length_fails_to_parse() {
    let list = free_space::single_section_info();
    let header = FreeSpaceManagerHeader::parse(
        test_support::widths(8, 8),
        &free_space::single_section_header(),
    )
    .unwrap();

    assert_eq!(
        parse_section_info(
            test_support::widths(8, 8),
            &list[..list.len() - 1],
            StoredAddress::new(619),
            &header
        ),
        Err(FormatError::UnexpectedEof {
            expected: 35,
            available: 34
        })
    );
}

#[cfg(feature = "checksum")]
#[test]
fn the_checksum_at_the_end_of_the_used_length_is_verified() {
    // The error reports the flipped checksum, and not the zeros appended past the used length.
    let mut list = free_space::oversized_section_list();
    let header = FreeSpaceManagerHeader::parse(
        test_support::widths(8, 8),
        &free_space::oversized_section_list_header(),
    )
    .unwrap();
    let stored = list.len() - 4;
    list[stored] ^= 0xFF;
    list.extend_from_slice(&[0; 8]);

    assert_eq!(
        parse_section_info(
            test_support::widths(8, 8),
            &list,
            StoredAddress::new(48),
            &header
        ),
        Err(FormatError::ChecksumMismatch {
            expected: 0xAE5C_5CC0,
            computed: 0xAE5C_5C3F,
        })
    );
}

#[test]
fn rejects_bad_signature() {
    let mut fshd = free_space::single_section_header();
    let header = FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd).unwrap();
    let mut fsse = free_space::single_section_info();
    fshd[0] = 0x80;
    fsse[0] = b'X';

    assert_eq!(
        FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &fshd),
        Err(FormatError::InvalidFreeSpaceManager(
            r#"the header signature is b"\x80SHD", not b"FSHD""#.to_owned()
        ))
    );
    assert_eq!(
        parse_section_info(
            test_support::widths(8, 8),
            &fsse,
            StoredAddress::new(619),
            &header
        ),
        Err(FormatError::InvalidFreeSpaceManager(
            r#"the section list signature is b"XSSE", not b"FSSE""#.to_owned()
        ))
    );
}
