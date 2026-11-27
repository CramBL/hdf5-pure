use rstest::rstest;

use super::super::creation::{
    EA_IDX_BLK_ELMTS, EA_MAX_DBLK_NELMTS_BITS, EA_MAX_NELMTS_BITS, EA_MIN_DBLK_NELMTS,
};
use super::super::{EA_CLIENT_ID_UNFILTERED, measurement_sizes};
use super::*;
use crate::chunk_record::ArrayClient;
use crate::width::OffsetWidth;

// The oracle is `H5EA__dblock_alloc`, which pages a block of more elements than a page.
#[test]
fn a_data_block_is_paged_only_past_a_full_page() {
    let page_nelmts = ExtensibleArrayCreationParameters::parse(&[32, 0, 2, 1, 4])
        .unwrap()
        .bit_geometry()
        .page_nelmts();
    let at = |dblk_nelmts| DataBlockGeometry {
        dblk_nelmts,
        page_nelmts,
    };
    assert!(!at(15).is_paged(), "under a page");
    assert!(!at(16).is_paged(), "exactly a page is not paged");
    assert!(at(17).is_paged(), "past a page");
    assert_eq!(at(16).npages(), 0, "a block that is not paged has no pages");
    assert_eq!(at(32).npages(), 2);
}

#[test]
fn the_page_init_bitmap_is_a_byte_per_eight_pages_per_block() {
    let page_nelmts = ExtensibleArrayCreationParameters::parse(&[32, 0, 2, 1, 4])
        .unwrap()
        .bit_geometry()
        .page_nelmts();
    let sb = |ndblks, dblk_nelmts| SuperBlockGeometry {
        ndblks,
        blocks: DataBlockGeometry {
            dblk_nelmts,
            page_nelmts,
        },
    };
    // Not paged: no bitmap.
    assert_eq!(sb(4, 16).bitmap_size(), 0);
    // 2 blocks x 8 pages each -> one byte per block.
    assert_eq!(sb(2, 8 * 16).bitmap_size(), 2);
    // 2 blocks x 9 pages each -> the round-up costs a second byte per block.
    assert_eq!(sb(2, 9 * 16).bitmap_size(), 4);
    // One block, two pages: a single byte, not two bits.
    assert_eq!(sb(1, 2 * 16).bitmap_size(), 1);
}

#[rstest]
fn shared_geometry_agrees_on_block_page_and_bitmap_lengths(
    #[values(2, 4, 8, 16)] minimum_pointers: u8,
    #[values(OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(false, true)] filtered: bool,
) {
    let element_size = offsets.get() + if filtered { 6 } else { 0 };
    let client_id = if filtered {
        ArrayClient::FilteredChunks { chunk_size_len: 2 }.id()
    } else {
        EA_CLIENT_ID_UNFILTERED
    };
    let creation = ExtensibleArrayCreationParameters::parse(&[
        EA_MAX_NELMTS_BITS,
        EA_IDX_BLK_ELMTS,
        minimum_pointers,
        EA_MIN_DBLK_NELMTS,
        EA_MAX_DBLK_NELMTS_BITS,
    ])
    .unwrap();
    let encoding = ElementEncoding::new(offsets, client_id, element_size).unwrap();
    let layout = creation.layout(encoding);
    let geometry = ExtensibleArrayGeometry::from_layout(layout).unwrap();
    assert_eq!(
        layout.index_block_len().unwrap(),
        u64::try_from(measurement_sizes::index_block_len(
            offsets,
            usize::from(EA_IDX_BLK_ELMTS),
            usize::from(element_size),
            geometry.direct_dblk_nelmts().len(),
            geometry.nsblk_addrs(),
        ))
        .unwrap()
    );
    for ordinal in 0..geometry.nsblk_addrs() {
        let writer = geometry.super_block_at(ordinal, layout.page_nelmts());
        let reader = layout
            .super_block(u32::try_from(geometry.first_indirect_sblk() + ordinal).unwrap())
            .unwrap();
        assert_eq!(layout.bitmap_len(reader).unwrap(), writer.bitmap_size());
        assert_eq!(
            layout.super_block_len(reader).unwrap(),
            measurement_sizes::super_block_len(
                writer,
                offsets,
                creation.bit_geometry().block_offset_width()
            )
        );
        assert_eq!(
            layout
                .data_block_len(reader.data_block(0).unwrap())
                .unwrap(),
            measurement_sizes::data_block_len(
                writer.blocks,
                usize::from(element_size),
                offsets,
                creation.bit_geometry().block_offset_width()
            )
        );
        assert_eq!(
            layout.pages(reader.block_elements).unwrap_or(0),
            writer.blocks.npages()
        );
    }
}

#[rstest]
#[case::two_pointer_minimum(2, 2, 2, 282)]
#[case::four_pointer_minimum(4, 6, 4, 298)]
#[case::eight_pointer_minimum(8, 14, 6, 346)]
#[case::sixteen_pointer_minimum(16, 30, 8, 458)]
fn reader_and_writer_geometry_agree_on_pointer_counts(
    #[case] min_data_pointers: u8,
    #[case] direct_blocks: usize,
    #[case] first_secondary: usize,
    #[case] index_bytes: usize,
) {
    let layout = layout_from_fields(
        ElementEncoding::new(OffsetWidth::Eight, 0, 8).unwrap(),
        [
            EA_MAX_NELMTS_BITS,
            EA_IDX_BLK_ELMTS,
            min_data_pointers,
            EA_MIN_DBLK_NELMTS,
            EA_MAX_DBLK_NELMTS_BITS,
        ],
    );
    let geometry = ExtensibleArrayGeometry::from_layout(layout).unwrap();
    assert_eq!(geometry.direct_dblk_nelmts().len(), direct_blocks);
    assert_eq!(geometry.first_indirect_sblk(), first_secondary);
    assert_eq!(
        measurement_sizes::index_block_len(
            OffsetWidth::Eight,
            usize::from(EA_IDX_BLK_ELMTS),
            8,
            geometry.direct_dblk_nelmts().len(),
            geometry.nsblk_addrs()
        ),
        index_bytes
    );
    assert_eq!(
        layout
            .direct_blocks()
            .map(|block| block.map(|block| block.elements))
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        geometry.direct_dblk_nelmts()
    );
    assert_eq!(
        usize::try_from(layout.first_super_block()).unwrap(),
        geometry.first_indirect_sblk()
    );
    assert_eq!(
        usize::try_from(layout.super_blocks() - layout.first_super_block()).unwrap(),
        geometry.nsblk_addrs()
    );
    for (index, &(blocks, elements)) in geometry.sblks().iter().enumerate() {
        let super_block = layout.super_block(u32::try_from(index).unwrap()).unwrap();
        assert_eq!(
            (super_block.blocks, super_block.block_elements),
            (blocks, elements)
        );
    }
}

#[rstest]
#[case::two_min_data_blocks(2, &[(0, 16), (16, 32)], 2)]
#[case::four_min_data_blocks(4, &[(0, 16), (16, 32), (48, 32), (80, 32), (112, 64), (176, 64)], 4)]
#[case::eight_min_data_blocks(
    8,
    &[
        (0, 16), (16, 32), (48, 32), (80, 32), (112, 64), (176, 64),
        (240, 64), (304, 64), (368, 64), (432, 64),
        (496, 128), (624, 128), (752, 128), (880, 128),
    ],
    6
)]
fn the_index_block_holds_the_data_blocks_before_the_first_super_block(
    #[case] min_data_blocks: u8,
    #[case] direct_blocks: &[(u64, u64)],
    #[case] first_super_block: u32,
) {
    let encoding = ElementEncoding::new(OffsetWidth::Eight, 0, 8).unwrap();
    let layout = layout_from_fields(encoding, [32, 4, min_data_blocks, 16, 10]);
    assert_eq!(layout.first_super_block(), first_super_block);
    assert_eq!(
        layout
            .direct_blocks()
            .map(|block| block.map(|block| (block.offset, block.elements)))
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        direct_blocks
    );
    assert_eq!(layout.direct_data_blocks(), direct_blocks.len() as u64);
    assert_eq!(
        layout.index_block_len().unwrap(),
        4 + 2
            + 8
            + 4 * 8
            + (direct_blocks.len() as u64 + u64::from(29 - first_super_block)) * 8
            + 4
    );
}

// An unpaged data block is the prefix, the block offset, the elements, and the checksum, and a
// paged one the prefix, the block offset, the checksum, and the pages with a checksum each.
#[rstest]
#[case::no_elements(OffsetWidth::Four, 0, 4, 10, 0, 4 + 2 + 4 + 2 + 4)]
#[case::four_byte_addresses(OffsetWidth::Four, 0, 4, 16, 255, 4 + 2 + 4 + 2 + 255 * 4 + 4)]
#[case::filtered_four_byte_addresses(OffsetWidth::Four, 1, 10, 32, 16, 4 + 2 + 4 + 4 + 16 * 10 + 4)]
#[case::one_page(OffsetWidth::Eight, 0, 8, 32, 1_024, 4 + 2 + 8 + 4 + 1_024 * 8 + 4)]
#[case::two_pages(OffsetWidth::Eight, 1, 14, 32, 2_048, 4 + 2 + 8 + 4 + 4 + 2 * (1_024 * 14 + 4))]
#[case::four_pages_four_byte_addresses(OffsetWidth::Four, 0, 4, 16, 4_096, 4 + 2 + 4 + 2 + 4 + 4 * (1_024 * 4 + 4))]
fn a_data_block_has_the_length_of_its_fields(
    #[case] offsets: OffsetWidth,
    #[case] client_id: u8,
    #[case] element_size: u8,
    #[case] max_index_bits: u8,
    #[case] elements: u64,
    #[case] expected: u64,
) {
    let encoding = ElementEncoding::new(offsets, client_id, element_size).unwrap();
    let layout = layout_from_fields(encoding, [max_index_bits, 4, 2, 4, 10]);
    assert_eq!(
        layout
            .data_block_len(DataBlock {
                offset: 0,
                elements,
                libhdf5_offset: None,
            })
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::one_byte(1, 1)]
#[case::byte_boundary(8, 1)]
#[case::two_bytes(9, 2)]
#[case::four_bytes(32, 4)]
#[case::eight_bytes(64, 8)]
fn the_block_offset_is_the_index_bits_in_whole_bytes(#[case] bits: u8, #[case] len: u8) {
    let encoding = ElementEncoding::new(OffsetWidth::Eight, 0, 8).unwrap();
    assert_eq!(
        layout_from_fields(encoding, [bits, 0, 2, 1, 1]).block_offset_len(),
        len
    );
}

fn layout_from_fields(encoding: ElementEncoding, fields: [u8; 5]) -> Layout {
    ExtensibleArrayCreationParameters::parse(&fields)
        .unwrap()
        .layout(encoding)
}
