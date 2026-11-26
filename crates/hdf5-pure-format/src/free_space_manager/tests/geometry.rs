use alloc::vec::Vec;

use rstest::rstest;
use test_util::free_space;

use super::*;
use crate::free_space_manager::test_support;

#[rstest]
#[case::empty(test_support::widths(8, 8), 0, 0, 17)]
#[case::one(test_support::widths(8, 8), 1, 1, 35)]
#[case::one_group(test_support::widths(8, 8), 3, 1, 53)]
#[case::distinct_sizes(test_support::widths(8, 8), 3, 3, 71)]
#[case::two_byte_fields(test_support::widths(2, 2), 3, 2, 26)]
#[case::four_byte_fields(test_support::widths(4, 4), 3, 2, 38)]
#[case::four_byte_lengths(test_support::widths(8, 4), 3, 2, 54)]
#[case::four_byte_offsets(test_support::widths(4, 8), 3, 2, 46)]
#[case::two_byte_lengths(test_support::widths(8, 2), 3, 2, 50)]
#[case::two_byte_offsets(test_support::widths(2, 8), 3, 2, 38)]
#[case::last_one_byte_count(test_support::widths(8, 8), 255, 1, 2_321)]
#[case::first_two_byte_count(test_support::widths(8, 8), 256, 1, 2_331)]
#[case::last_two_byte_count(test_support::widths(8, 8), 65_535, 1, 589_842)]
#[case::first_three_byte_count(test_support::widths(8, 8), 65_536, 1, 589_852)]
#[case::last_four_byte_count(test_support::widths(8, 8), (1 << 32) - 1, 1, 38_654_705_684)]
#[case::first_five_byte_count(test_support::widths(8, 8), 1 << 32, 1, 38_654_705_694)]
#[case::eight_byte_count(test_support::widths(8, 8), 1 << 56, 1, 648_518_346_341_351_457)]
fn section_counts_determine_the_unpadded_list_length(
    #[case] widths: FormatWidths,
    #[case] sections: u64,
    #[case] size_groups: u64,
    #[case] expected: u64,
) {
    assert_eq!(
        SectionInfoCounts::new(sections, size_groups)
            .unwrap()
            .encoded_len(widths)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::empty(0, 1)]
#[case::last_one_byte(255, 1)]
#[case::first_two_bytes(256, 2)]
#[case::first_three_bytes(65_536, 3)]
#[case::first_four_bytes(1 << 24, 4)]
#[case::first_five_bytes(1 << 32, 5)]
#[case::first_six_bytes(1 << 40, 6)]
#[case::first_seven_bytes(1 << 48, 7)]
#[case::first_eight_bytes(1 << 56, 8)]
fn section_counts_expose_the_total_count_field_width(
    #[case] sections: u64,
    #[case] expected: usize,
) {
    assert_eq!(
        SectionInfoCounts::new(sections, u64::from(sections != 0))
            .unwrap()
            .count_width(),
        expected
    );
}

#[rstest]
#[case::one_section(1, 1, free_space::single_section_info(), 0)]
#[case::two_sizes(2, 2, free_space::two_section_info(), 0)]
#[case::allocated_padding(10, 9, free_space::oversized_section_list(), 9)]
fn section_counts_match_known_native_lengths(
    #[case] sections: u64,
    #[case] size_groups: u64,
    #[case] observed: Vec<u8>,
    #[case] padding: u64,
) {
    assert_eq!(
        SectionInfoCounts::new(sections, size_groups)
            .unwrap()
            .encoded_len(test_support::widths(8, 8))
            .unwrap()
            + padding,
        observed.len().to_u64()
    );
}

#[rstest]
#[case::groups_without_sections(0, 1)]
#[case::sections_without_groups(1, 0)]
#[case::too_many_groups(3, 4)]
fn section_counts_reject_impossible_groups(#[case] sections: u64, #[case] size_groups: u64) {
    assert_eq!(
        SectionInfoCounts::new(sections, size_groups).unwrap_err(),
        FormatError::InvalidFreeSpaceSectionCounts {
            sections,
            size_groups
        }
    );
}

#[rstest]
#[case::records(u64::MAX, 1)]
#[case::groups(u64::MAX, u64::MAX)]
#[case::sum((u64::MAX - 33) / 9 + 1, 1)]
fn section_counts_reject_unrepresentable_lengths(#[case] sections: u64, #[case] size_groups: u64) {
    assert_eq!(
        SectionInfoCounts::new(sections, size_groups)
            .unwrap()
            .encoded_len(test_support::widths(8, 8))
            .unwrap_err(),
        FormatError::OffsetOverflow {
            offset: 17,
            length: sections
        }
    );
}

#[test]
fn section_counts_measure_the_largest_length_without_materializing_sections() {
    let sections = (u64::MAX - 33) / 9;
    assert_eq!(
        SectionInfoCounts::new(sections, 1)
            .unwrap()
            .encoded_len(test_support::widths(8, 8))
            .unwrap(),
        u64::MAX
    );
}

#[test]
fn section_capacity_preserves_platform_count_limits() {
    let result = section_info_capacity(test_support::widths(8, 8), usize::MAX);
    if usize::BITS == 64 {
        assert_eq!(
            result.unwrap_err(),
            FormatError::OffsetOverflow {
                offset: 17,
                length: u64::MAX
            }
        );
    } else {
        assert_eq!(result.unwrap(), 90_194_313_212);
    }
}

#[rstest]
#[case::two(2, 2, 15)]
#[case::mixed(8, 4, 256)]
#[case::eight(8, 8, 256)]
fn section_capacity_bounds_distinct_sizes_and_count_width_changes(
    #[case] offsets: u8,
    #[case] lengths: u8,
    #[case] count: usize,
) {
    let widths = test_support::widths(offsets, lengths);
    let sizes: Vec<_> = (1..=count as u64).collect();
    let capacity = section_info_capacity(widths, count).unwrap();
    assert_eq!(section_info_len(widths, &sizes), capacity);
    let fewer: Vec<_> = sizes.into_iter().take(count - 1).collect();
    assert!(section_info_len(widths, &fewer) <= capacity);
    assert!(section_info_len(widths, &alloc::vec![1; count]) <= capacity);
}

#[test]
fn integer_width_uses_the_fewest_bytes_and_at_least_one() {
    assert_eq!(enc_size(0), 1);
    assert_eq!(enc_size(255), 1);
    assert_eq!(enc_size(256), 2);
    assert_eq!(enc_size((1 << 63) - 1), 8);
}
