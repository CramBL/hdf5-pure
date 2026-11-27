use rstest::rstest;

use super::*;

#[test]
fn bit_geometry_stores_two_bytes_with_byte_alignment() {
    assert_eq!(core::mem::size_of::<MaxIndexBits>(), 1);
    assert_eq!(core::mem::align_of::<MaxIndexBits>(), 1);
    assert_eq!(core::mem::size_of::<PageElementCount>(), 1);
    assert_eq!(core::mem::align_of::<PageElementCount>(), 1);
    assert_eq!(core::mem::size_of::<ExtensibleArrayBitGeometry>(), 2);
    assert_eq!(core::mem::align_of::<ExtensibleArrayBitGeometry>(), 1);
    assert_eq!(core::mem::size_of::<BlockOffsetWidth>(), 1);
    assert_eq!(core::mem::align_of::<BlockOffsetWidth>(), 1);
}

#[rstest]
#[case::one_byte(BlockOffsetWidth::One, &[0x01], 0x01)]
#[case::two_bytes(BlockOffsetWidth::Two, &[0x01, 0x02], 0x0201)]
#[case::three_bytes(BlockOffsetWidth::Three, &[0x01, 0x02, 0x03], 0x03_0201)]
#[case::four_bytes(BlockOffsetWidth::Four, &[0x01, 0x02, 0x03, 0x04], 0x0403_0201)]
#[case::five_bytes(BlockOffsetWidth::Five, &[0x01, 0x02, 0x03, 0x04, 0x05], 0x0005_0403_0201)]
#[case::six_bytes(BlockOffsetWidth::Six, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 0x0605_0403_0201)]
#[case::seven_bytes(BlockOffsetWidth::Seven, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07], 0x0007_0605_0403_0201)]
#[case::eight_bytes(BlockOffsetWidth::Eight, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08], 0x0807_0605_0403_0201)]
fn block_offset_width_reads_the_value_and_reports_missing_bytes(
    #[case] width: BlockOffsetWidth,
    #[case] bytes: &[u8],
    #[case] value: u64,
) {
    assert_eq!(width.read(bytes).unwrap(), value);
    let err = width.read(&bytes[..bytes.len() - 1]).unwrap_err();
    let FormatError::UnexpectedEof {
        expected,
        available,
    } = err
    else {
        panic!("expected UnexpectedEof, got {err:?}");
    };
    assert_eq!(expected, width.bytes());
    assert_eq!(available, bytes.len() - 1);
}

#[rstest]
#[case::zero_maximum(0, 10, InvalidBitGeometry::MaximumIndexBits)]
#[case::maximum_65(65, 10, InvalidBitGeometry::MaximumIndexBits)]
#[case::maximum_255(255, 10, InvalidBitGeometry::MaximumIndexBits)]
#[case::invalid_maximum_precedes_page_overflow(0, 64, InvalidBitGeometry::MaximumIndexBits)]
#[case::page_64(64, 64, InvalidBitGeometry::PageElementCount)]
#[case::page_overflow_precedes_exponent_above_maximum(10, 64, InvalidBitGeometry::PageElementCount)]
#[case::page_255(64, 255, InvalidBitGeometry::PageElementCount)]
#[case::page_exceeds_maximum(10, 11, InvalidBitGeometry::PageExponentExceedsIndexBits)]
fn malformed_bit_geometry_is_rejected(
    #[case] max_bits: u8,
    #[case] page_bits: u8,
    #[case] expected: InvalidBitGeometry,
) {
    assert_eq!(
        ExtensibleArrayBitGeometry::from_fields(max_bits, page_bits).unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::one_byte(1, 0, 1, 1)]
#[case::byte_boundary(8, 8, 256, 1)]
#[case::two_bytes(9, 9, 512, 2)]
#[case::eight_bytes(64, 10, 1024, 8)]
#[case::largest_page(64, 63, 1u64 << 63, 8)]
fn parsed_bit_geometry_supplies_page_count_and_offset_width(
    #[case] max_bits: u8,
    #[case] page_bits: u8,
    #[case] page_nelmts: u64,
    #[case] offset_size: usize,
) {
    let geometry = ExtensibleArrayBitGeometry::from_fields(max_bits, page_bits).unwrap();
    assert_eq!(geometry.max_index_bits().get(), max_bits);
    let page_count = geometry.page_nelmts();
    assert_eq!(page_count.exponent(), page_bits);
    assert_eq!(page_count.get(), page_nelmts);
    assert_eq!(geometry.block_offset_width().bytes(), offset_size);

    let offset = u64::MAX >> (8 * (8 - offset_size));
    assert_eq!(
        geometry
            .block_offset_width()
            .read(&offset.to_le_bytes())
            .unwrap(),
        offset
    );
}
