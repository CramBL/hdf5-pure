use alloc::vec;

use rstest::rstest;

use super::support::ARRAY_AT;
use super::support::Fixture;
use super::support::WIDTHS;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record;
use crate::chunk_record::IndexSlots;
use crate::error::FormatError;
use crate::fixed_array;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
fn the_header_and_data_block_ranges_tile_the_array(
    #[values(0, 1, 5, 1_024, 1_025, 3_000)] n: u64,
    #[values(false, true)] has_filters: bool,
) {
    let fixture = Fixture::new(n, has_filters);
    let header = fixture.header().unwrap();
    let block = header.data_block().unwrap();
    assert_eq!(header.range().start().get(), ARRAY_AT);
    assert_eq!(header.range().end(), block.range().start());
    assert_eq!(block.range().end().get(), fixture.bytes.len() as u64);
}

#[test]
fn a_slot_that_stores_the_undefined_address_has_no_record() {
    let chunks = chunk_record::chunks(8, 3, true);
    let slots = IndexSlots::new(&chunks, &[0, 2, 4], 5).unwrap();
    let array = fixed_array::build_fixed_array_at(
        &slots,
        8,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        true,
        StoredAddress::new(ARRAY_AT),
    )
    .unwrap();
    let mut fixture = Fixture::new(0, false);
    fixture.bytes.truncate(fixture.header_at());
    fixture.bytes.extend_from_slice(&array);
    assert_eq!(
        fixture.records(8).unwrap(),
        vec![(0, chunks[0]), (2, chunks[1]), (4, chunks[2])]
    );
}

#[rstest]
#[case::signature(0, b"X", FormatError::InvalidFixedArraySignature)]
#[case::version(4, &[1], FormatError::InvalidFixedArrayVersion(1))]
#[case::client(5, &[1], FormatError::InvalidFixedArrayBlockClient { expected: 0, actual: 1 })]
#[case::header_address(6, &[0x81], FormatError::InvalidFixedArrayBlockOwner { expected: ARRAY_AT, actual: 0x81 })]
fn a_malformed_data_block_prefix_is_rejected(
    #[case] field: usize,
    #[case] value: &[u8],
    #[case] expected: FormatError,
    #[values(5, 3_000)] n: u64,
) {
    let mut fixture = Fixture::new(n, false);
    let at = fixture.data_block_at() + field;
    fixture.bytes[at..at + value.len()].copy_from_slice(value);
    fixture.restamp_prefix();
    let block = fixture.data_block();
    let region = fixture
        .image()
        .subregion(block.parse_range().unwrap())
        .unwrap();
    assert_eq!(block.parse(region).unwrap_err(), expected);
}

#[test]
fn a_header_address_at_the_physical_header_position_is_rejected() {
    let base = 1 << 40;
    let mut fixture = Fixture::at(base, WIDTHS, 8, 5, false);
    let at = fixture.data_block_at() + 6;
    fixture.bytes[at..at + 8].copy_from_slice(&(base + ARRAY_AT).to_le_bytes());
    fixture.restamp_prefix();
    let block = fixture.data_block();
    let region = fixture
        .image()
        .subregion(block.parse_range().unwrap())
        .unwrap();
    assert_eq!(
        block.parse(region).unwrap_err(),
        FormatError::InvalidFixedArrayBlockOwner {
            expected: base + ARRAY_AT,
            actual: 2 * base + ARRAY_AT,
        }
    );
}
