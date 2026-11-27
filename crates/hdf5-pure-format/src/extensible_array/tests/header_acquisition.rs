use rstest::rstest;

use super::support::{ARRAY_AT, FILTER_MASK_BYTES, Fixture, HEADER_MINIMUM_POINTERS};
use crate::address::{StoredAddress, StoredAddressExt};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::extensible_array::block::CHECKSUM_LEN;
use crate::extensible_array::header::HEADER_PREFIX_LEN;
use crate::extensible_array::statistics::STATISTICS;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[rstest]
#[case::unfiltered(false)]
#[case::filtered(true)]
fn the_header_reports_whether_its_chunks_are_filtered(#[case] has_filters: bool) {
    assert_eq!(
        Fixture::new(5, has_filters).header().unwrap().is_filtered(),
        has_filters
    );
}

#[rstest]
#[case::two_byte_fields(OffsetWidth::Two, LengthWidth::Two)]
#[case::four_byte_fields(OffsetWidth::Four, LengthWidth::Four)]
#[case::eight_byte_fields(OffsetWidth::Eight, LengthWidth::Eight)]
#[case::short_offsets_long_lengths(OffsetWidth::Four, LengthWidth::Eight)]
#[case::long_offsets_short_lengths(OffsetWidth::Eight, LengthWidth::Four)]
fn the_region_parser_preserves_append_metadata_and_geometry(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(0, 1 << 40)] base: u64,
    #[values(false, true)] filtered: bool,
    #[values(
        (2, &[16, 32][..], 2, 27),
        (4, &[16, 32, 32, 32, 64, 64][..], 4, 25),
        (
            8,
            &[16, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128][..],
            6,
            23
        ),
        (
            16,
            &[
                16, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128,
                128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256,
                256, 256, 256, 256
            ][..],
            8,
            21
        )
    )]
    expected: (u8, &[u64], usize, usize),
) {
    let (minimum_pointers, direct_blocks, first_secondary, secondary_pointers) = expected;
    let widths = FormatWidths { offsets, lengths };
    let mut fixture = Fixture::at(base, widths, 8, 20, filtered);
    let range = fixture.locator().parse_range().unwrap();
    let local = fixture.local(range.start());
    fixture.bytes[local + HEADER_MINIMUM_POINTERS] = minimum_pointers;
    fixture.restamp(range);
    let header_len = HEADER_PREFIX_LEN.to_u64()
        + u64::from(STATISTICS) * u64::from(lengths.get())
        + u64::from(offsets.get())
        + CHECKSUM_LEN.to_u64();
    let array = fixture.header().unwrap();
    let header = array.header();
    assert_eq!(header.client_id(), u8::from(filtered));
    assert_eq!(
        header.element_size(),
        offsets.get() + if filtered { 2 + FILTER_MASK_BYTES } else { 0 },
    );
    assert_eq!(
        header.creation_parameters().encode(),
        [32, 4, minimum_pointers, 16, 10]
    );
    assert_eq!(header.stored_statistics().max_idx_set, 20);
    assert_eq!(
        header.index_block_address(),
        StoredAddress::new(ARRAY_AT + header_len)
    );
    assert_eq!(array.range().start().get(), base + ARRAY_AT);
    assert_eq!(array.range().length(), header_len);
    assert_eq!(
        array.index_block().unwrap().parse_range().start().get(),
        base + ARRAY_AT + header_len,
    );
    let actual = array.geometry().unwrap();
    assert_eq!(
        actual.sblks(),
        &[
            (1, 16),
            (1, 32),
            (2, 32),
            (2, 64),
            (4, 64),
            (4, 128),
            (8, 128),
            (8, 256),
            (16, 256),
            (16, 512),
            (32, 512),
            (32, 1_024),
            (64, 1_024),
            (64, 2_048),
            (128, 2_048),
            (128, 4_096),
            (256, 4_096),
            (256, 8_192),
            (512, 8_192),
            (512, 16_384),
            (1_024, 16_384),
            (1_024, 32_768),
            (2_048, 32_768),
            (2_048, 65_536),
            (4_096, 65_536),
            (4_096, 131_072),
            (8_192, 131_072),
            (8_192, 262_144),
            (16_384, 262_144),
        ]
    );
    assert_eq!(actual.direct_dblk_nelmts(), direct_blocks);
    assert_eq!(actual.nsblk_addrs(), secondary_pointers);
    assert_eq!(actual.first_indirect_sblk(), first_secondary);
}

#[rstest]
#[case::signature(&[(0, b'X')], FormatError::InvalidExtensibleArraySignature)]
#[case::version(&[(4, 1)], FormatError::InvalidExtensibleArrayVersion(1))]
#[case::client(&[(5, 2)], FormatError::InvalidExtensibleArrayClient(2))]
#[case::filtered_element_too_small(&[(6, 8 + 4 - 1)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::filtered_element_too_large(&[(6, 8 + 4 + 9)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::no_index_bits(&[(7, 0)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::index_bits_past_64(&[(7, 65)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::no_min_elements(&[(9, 0)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::min_elements_not_a_power_of_two(&[(9, 24)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::min_elements_past_the_index_bits(&[(7, 2), (11, 2)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::zero_min_data_blocks(&[(10, 0)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::one_min_data_block(&[(10, 1)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::min_data_blocks_not_a_power_of_two(&[(10, 6)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::direct_data_blocks_past_the_super_blocks(&[(7, 8), (10, 16), (11, 8)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::paged_direct_data_block(&[(11, 5)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::page_bits_past_the_index_bits(&[(7, 9), (11, 10)], FormatError::InvalidExtensibleArrayGeometry)]
#[case::page_bits_64(&[(7, 64), (11, 64)], FormatError::InvalidExtensibleArrayGeometry)]
fn a_malformed_header_is_rejected(#[case] fields: &[(usize, u8)], #[case] expected: FormatError) {
    let mut fixture = Fixture::new(5, true);
    let header = fixture.header().unwrap().range();
    for &(field, value) in fields {
        let at = fixture.local(header.start()) + field;
        fixture.bytes[at] = value;
    }
    fixture.restamp(header);
    assert_eq!(fixture.header().unwrap_err(), expected);
}
