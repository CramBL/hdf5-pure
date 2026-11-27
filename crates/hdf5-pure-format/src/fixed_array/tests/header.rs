use rstest::rstest;
use test_util::fixed_array as fixture;

use super::support::Fixture;
use super::support::WIDTHS;
use crate::error::FormatError;
use crate::fixed_array::header;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[test]
fn an_undefined_data_block_address_leaves_the_array_without_a_data_block() {
    let mut fixture = Fixture::new(5, false);
    let at = fixture.header_at() + 8 + 8;
    fixture.bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    fixture.restamp_header();
    let header = fixture.header().unwrap();
    assert_eq!(header.data_block(), None);
    assert_eq!(header.range().length(), header::header_len(WIDTHS));
}

#[rstest]
#[case::signature(
    OffsetWidth::Eight,
    true,
    0,
    b'X',
    FormatError::InvalidFixedArraySignature
)]
#[case::version(
    OffsetWidth::Eight,
    true,
    4,
    1,
    FormatError::InvalidFixedArrayVersion(1)
)]
#[case::client(
    OffsetWidth::Eight,
    true,
    5,
    2,
    FormatError::InvalidFixedArrayClient(2)
)]
#[case::page_bits(
    OffsetWidth::Eight,
    true,
    7,
    64,
    FormatError::InvalidFixedArrayGeometry
)]
#[case::filtered_entry_too_small(OffsetWidth::Eight, true, fixture::HEADER_ENTRY_SIZE_OFFSET, 8 + 4 - 1, FormatError::InvalidFixedArrayGeometry)]
#[case::filtered_entry_too_large(OffsetWidth::Eight, true, fixture::HEADER_ENTRY_SIZE_OFFSET, 8 + 4 + 9, FormatError::InvalidFixedArrayGeometry)]
#[case::two_byte_entry_too_small(
    OffsetWidth::Two,
    false,
    fixture::HEADER_ENTRY_SIZE_OFFSET,
    1,
    FormatError::InvalidFixedArrayGeometry
)]
#[case::two_byte_entry_too_large(
    OffsetWidth::Two,
    false,
    fixture::HEADER_ENTRY_SIZE_OFFSET,
    3,
    FormatError::InvalidFixedArrayGeometry
)]
#[case::four_byte_entry_too_small(
    OffsetWidth::Four,
    false,
    fixture::HEADER_ENTRY_SIZE_OFFSET,
    3,
    FormatError::InvalidFixedArrayGeometry
)]
#[case::four_byte_entry_too_large(
    OffsetWidth::Four,
    false,
    fixture::HEADER_ENTRY_SIZE_OFFSET,
    5,
    FormatError::InvalidFixedArrayGeometry
)]
#[case::eight_byte_entry_too_small(
    OffsetWidth::Eight,
    false,
    fixture::HEADER_ENTRY_SIZE_OFFSET,
    7,
    FormatError::InvalidFixedArrayGeometry
)]
#[case::eight_byte_entry_too_large(
    OffsetWidth::Eight,
    false,
    fixture::HEADER_ENTRY_SIZE_OFFSET,
    9,
    FormatError::InvalidFixedArrayGeometry
)]
fn a_malformed_header_is_rejected(
    #[case] offsets: OffsetWidth,
    #[case] filtered: bool,
    #[case] field: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
) {
    let mut fixture = Fixture::at(
        0,
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        8,
        5,
        filtered,
    );
    let at = fixture.header_at() + field;
    fixture.bytes[at] = value;
    fixture.restamp_header();
    assert_eq!(fixture.header().unwrap_err(), expected);
}

#[test]
fn an_entry_count_whose_data_block_overflows_is_rejected() {
    let mut fixture = Fixture::new(5, false);
    let at = fixture.header_at() + 8;
    fixture.bytes[at..at + 8].copy_from_slice(&(u64::MAX / 4).to_le_bytes());
    fixture.restamp_header();
    assert_eq!(
        fixture.header().unwrap_err(),
        FormatError::InvalidFixedArrayGeometry
    );
}
