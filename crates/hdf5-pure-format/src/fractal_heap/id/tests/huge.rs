use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::*;
use crate::address::StoredAddressExt;
use crate::fractal_heap::{
    FractalHeapIdKind, FractalHeapIdLayout, FractalHeapIdView, test_support,
};

#[rstest]
#[case::filtered(true, FormatError::InvalidFractalHeapGeometry)]
#[case::missing_tree(false, FormatError::InvalidFractalHeapGeometry)]
fn huge_ids_require_a_defined_tracker(
    #[case] filtered: bool,
    #[case] error: FormatError,
    #[values(8, 29)] heap_id_length: u16,
) {
    let header = FractalHeapHeader {
        heap_id_length,
        filtering: if filtered {
            test_support::filtered_heap()
        } else {
            FractalHeapFiltering::Unfiltered
        },
        btree_huge_objects_address: StoredAddress::new(u64::MAX),
        ..test_support::attribute_heap_header()
    };
    let mut id = vec![0; usize::from(heap_id_length)];
    id[..2].copy_from_slice(&[FractalHeapIdType::Huge.first_byte(), 5]);
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).parse(&id),
        Err(error)
    );
}

#[rstest]
fn filtered_direct_huge_ids_decode_at_the_exact_capacity(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[values(0, 3)] padding: u16,
) {
    let capacity = 1
        + u16::from(offsets.get())
        + 2 * u16::from(lengths.get())
        + u16::try_from(size_of::<u32>()).unwrap();
    let header = FractalHeapHeader {
        heap_id_length: capacity + padding,
        filtering: test_support::filtered_heap(),
        root_block_address: StoredAddress::undefined(offsets),
        ..test_support::attribute_heap_header()
    };
    let mut id = vec![FractalHeapIdType::Huge.first_byte()];
    id.extend(filtered_huge_location(
        offsets, lengths, 0x1234, 17, 0, 1024,
    ));
    id.resize(usize::from(header.heap_id_length), 0);
    let parsed = FractalHeapIdLayout::new(&header, offsets, lengths)
        .parse(&id)
        .unwrap()
        .kind();
    let FractalHeapIdKind::Huge(HugeObjectReference::FilteredInline(location)) = parsed else {
        panic!("expected filtered direct huge ID, got {parsed:?}")
    };
    assert_eq!(location.address(), StoredAddress::new(0x1234));
    assert_eq!(location.encoded_size(), 17);
    assert_eq!(location.filter_mask(), 0);
    assert_eq!(location.decoded_size(), 1024);
    let indirect_header = FractalHeapHeader {
        heap_id_length: capacity - 1,
        ..header
    };
    let mut indirect_id = vec![0; usize::from(indirect_header.heap_id_length)];
    indirect_id[..2].copy_from_slice(&[FractalHeapIdType::Huge.first_byte(), 5]);
    let parsed = FractalHeapIdLayout::new(&indirect_header, offsets, lengths)
        .parse(&indirect_id)
        .unwrap()
        .kind();
    let FractalHeapIdKind::Huge(HugeObjectReference::FilteredIndexed(key)) = parsed else {
        panic!("expected filtered indirect huge ID, got {parsed:?}")
    };
    assert_eq!(key.value(), 5);
    assert_eq!(key.encoding(), HugeIndirectKeyEncoding::Libhdf5);
}

#[rstest]
fn filtered_indirect_huge_keys_retain_the_selected_dialect(
    #[values(
        HugeIndirectKeyEncoding::PublishedSpec,
        HugeIndirectKeyEncoding::Libhdf5
    )]
    encoding: HugeIndirectKeyEncoding,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
) {
    let header = FractalHeapHeader {
        heap_id_length: 9,
        filtering: test_support::filtered_heap(),
        ..test_support::attribute_heap_header()
    };
    let id = [FractalHeapIdType::Huge.first_byte(), 5, 1, 0, 0, 0, 0, 0, 0];
    let parsed =
        FractalHeapIdLayout::with_huge_key_encoding(&header, OffsetWidth::Eight, lengths, encoding)
            .parse(&id)
            .unwrap()
            .kind();
    let FractalHeapIdKind::Huge(HugeObjectReference::FilteredIndexed(key)) = parsed else {
        panic!("expected filtered indirect huge ID, got {parsed:?}")
    };
    assert_eq!(key.value(), 261);
    assert_eq!(key.encoding(), encoding);
}

#[test]
fn a_published_spec_huge_key_requires_its_full_length_field() {
    let header = FractalHeapHeader {
        heap_id_length: 3,
        filtering: test_support::filtered_heap(),
        ..test_support::attribute_heap_header()
    };
    assert_eq!(
        FractalHeapIdLayout::with_huge_key_encoding(
            &header,
            OffsetWidth::Eight,
            LengthWidth::Four,
            HugeIndirectKeyEncoding::PublishedSpec,
        )
        .parse(&[FractalHeapIdType::Huge.first_byte(), 5, 0])
        .unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[rstest]
#[case::spec(HugeIndirectKeyEncoding::PublishedSpec, Ok(5))]
#[case::libhdf5(HugeIndirectKeyEncoding::Libhdf5, Err(FormatError::ValueTooLargeForRepresentation { value: 0x1_0005, target: "huge-object tracker key" }))]
fn indirect_huge_keys_never_truncate_to_the_tracker_width(
    #[case] encoding: HugeIndirectKeyEncoding,
    #[case] expected: Result<u64, FormatError>,
) {
    let header = FractalHeapHeader {
        filtering: test_support::filtered_heap(),
        ..test_support::attribute_heap_header()
    };
    let parsed = FractalHeapIdLayout::with_huge_key_encoding(
        &header,
        OffsetWidth::Eight,
        LengthWidth::Two,
        encoding,
    )
    .parse(&[FractalHeapIdType::Huge.first_byte(), 5, 0, 1, 0, 0, 0, 0])
    .map(|view| match view.kind() {
        FractalHeapIdKind::Huge(HugeObjectReference::FilteredIndexed(key)) => key.value(),
        other => panic!("expected filtered indirect huge ID, got {other:?}"),
    });
    assert_eq!(parsed, expected);
}

#[rstest]
#[case::undefined(u64::MAX, 17, 0, 1024, FormatError::InvalidFractalHeapGeometry)]
#[case::empty_encoded(0x1234, 0, 0, 1024, FormatError::InvalidFractalHeapGeometry)]
#[case::empty_decoded(0x1234, 17, 0, 0, FormatError::InvalidFractalHeapGeometry)]
#[case::invalid_mask(0x1234, 17, 2, 1024, FormatError::InvalidFractalHeapGeometry)]
#[case::skipped_sizes(0x1234, 17, 1, 1024, FormatError::InvalidFractalHeapGeometry)]
#[case::overflow(u64::MAX - 16, 17, 0, 1024, FormatError::OffsetOverflow { offset: u64::MAX - 16, length: 17 })]
fn filtered_direct_huge_ids_reject_invalid_locations(
    #[case] address: u64,
    #[case] encoded_size: u64,
    #[case] mask: u32,
    #[case] decoded_size: u64,
    #[case] expected: FormatError,
) {
    let header = FractalHeapHeader {
        heap_id_length: 29,
        filtering: test_support::filtered_heap(),
        ..test_support::attribute_heap_header()
    };
    let mut id = vec![FractalHeapIdType::Huge.first_byte()];
    id.extend(filtered_huge_location(
        OffsetWidth::Eight,
        LengthWidth::Eight,
        address,
        encoded_size,
        mask,
        decoded_size,
    ));
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap_err(),
        expected
    );
}

#[rstest]
fn filtered_huge_locations_reject_undefined_addresses_at_the_encoded_width(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
) {
    let data = filtered_huge_location(
        offsets,
        LengthWidth::Eight,
        StoredAddress::undefined(offsets).get(),
        17,
        0,
        1024,
    );
    assert_eq!(
        FilteredHugeObjectLocation::parse(&data, offsets, LengthWidth::Eight, 1).unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[rstest]
#[case::empty_pipeline(0, 0)]
#[case::one_filter(1, 1)]
#[case::thirty_two_filters(32, u32::MAX)]
fn filtered_huge_locations_accept_skipped_filters(#[case] filter_count: usize, #[case] mask: u32) {
    let data = filtered_huge_location(OffsetWidth::Eight, LengthWidth::Eight, 0x1234, 17, mask, 17);
    let location = FilteredHugeObjectLocation::parse(
        &data,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        filter_count,
    )
    .unwrap();
    assert_eq!(location.filter_mask(), mask);
    assert_eq!(location.encoded_size(), location.decoded_size());
}

#[rstest]
fn filtered_huge_ids_accept_an_empty_pipeline(#[values(false, true)] direct: bool) {
    let encoded = fractal_heap::Header::new(u64::MAX)
        .filtering(0, 0, vec![test_support::PIPELINE_VERSION_TWO, 0])
        .build(Widths::EIGHT);
    let mut header =
        FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    header.heap_id_length = if direct { 29 } else { 8 };
    header.btree_huge_objects_address = StoredAddress::new(0x800);
    let id = if direct {
        let mut id = vec![FractalHeapIdType::Huge.first_byte()];
        id.extend(filtered_huge_location(
            OffsetWidth::Eight,
            LengthWidth::Eight,
            0x1234,
            17,
            0,
            17,
        ));
        id
    } else {
        vec![FractalHeapIdType::Huge.first_byte(), 5, 0, 0, 0, 0, 0, 0]
    };
    let parsed = FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
        .parse(&id)
        .unwrap()
        .kind();
    match (direct, parsed) {
        (true, FractalHeapIdKind::Huge(HugeObjectReference::FilteredInline(location))) => {
            assert_eq!(location.address(), StoredAddress::new(0x1234));
            assert_eq!(location.encoded_size(), 17);
            assert_eq!(location.filter_mask(), 0);
            assert_eq!(location.decoded_size(), 17);
        }
        (false, FractalHeapIdKind::Huge(HugeObjectReference::FilteredIndexed(key))) => {
            assert_eq!(key.value(), 5)
        }
        other => panic!("unexpected huge-ID representation: {other:?}"),
    }
}

fn filtered_huge_location(
    offsets: OffsetWidth,
    lengths: LengthWidth,
    address: u64,
    encoded_size: u64,
    mask: u32,
    decoded_size: u64,
) -> Vec<u8> {
    let mut data = Vec::new();
    bytes::write_offset(&mut data, address, offsets);
    bytes::write_length(&mut data, encoded_size, lengths);
    data.extend_from_slice(&mask.to_le_bytes());
    bytes::write_length(&mut data, decoded_size, lengths);
    data
}

#[test]
fn a_direct_huge_id_requires_room_for_the_full_location() {
    // Unfiltered: direct when `(id_len - 1) >= offset_size + length_size`.
    let mut h = test_support::dtable_header(512, 65536, 4);
    h.heap_id_length = 7; // 6 payload bytes < 8 + 8 -> indirect (B-tree)
    assert!(!h.huge_ids_direct(8, 8));
    h.heap_id_length = 17; // 16 payload bytes == 8 + 8 -> direct
    assert!(h.huge_ids_direct(8, 8));
    // A filtered heap needs room for the address, the length, the filter mask (4), and the
    // filtered length, so 17 bytes are too few.
    h.filtering = test_support::filtered_heap();
    assert!(!h.huge_ids_direct(8, 8));
}

#[test]
fn a_huge_id_decodes_to_an_index_key_or_an_inline_location() {
    let indexed = FractalHeapHeader {
        btree_huge_objects_address: StoredAddress::new(0x800),
        ..test_support::dtable_header(512, 65536, 4)
    };
    let wide = FractalHeapHeader {
        heap_id_length: 17,
        btree_huge_objects_address: StoredAddress::new(0x800),
        ..test_support::dtable_header(512, 65536, 4)
    };
    let mut inline_huge = vec![0x10];
    inline_huge.extend_from_slice(&0x900u64.to_le_bytes());
    inline_huge.extend_from_slice(&40u64.to_le_bytes());

    assert_eq!(
        FractalHeapIdLayout::new(&indexed, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&[0x10, 5, 0, 0, 0, 0, 0])
            .map(FractalHeapIdView::kind),
        Ok(FractalHeapIdKind::Huge(HugeObjectReference::Indexed(
            HugeIndirectKey {
                value: 5,
                encoding: HugeIndirectKeyEncoding::Libhdf5
            }
        )))
    );
    assert_eq!(
        FractalHeapIdLayout::new(&wide, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&inline_huge)
            .map(FractalHeapIdView::kind),
        Ok(FractalHeapIdKind::Huge(HugeObjectReference::Inline {
            addr: StoredAddress::new(0x900),
            len: 40,
        }))
    );
}

#[test]
fn an_encoded_huge_id_decodes_to_its_key() {
    let header = test_support::attribute_heap_header();
    let id = header
        .encode_huge_id(5, OffsetWidth::Eight, LengthWidth::Eight)
        .unwrap();
    assert_eq!(id, [0x10, 5, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .map(FractalHeapIdView::kind),
        Ok(FractalHeapIdKind::Huge(HugeObjectReference::Indexed(
            HugeIndirectKey {
                value: 5,
                encoding: HugeIndirectKeyEncoding::Libhdf5
            }
        )))
    );
}

#[rstest]
#[case::a_key_past_the_id(
    test_support::attribute_heap_header(),
    1 << 56,
    FormatError::Internal("a huge object key does not fit a 8-byte heap ID".into())
)]
#[case::an_id_that_holds_the_address(
    FractalHeapHeader { heap_id_length: 17, ..test_support::attribute_heap_header() },
    1,
    FormatError::Internal("a key was encoded into a heap ID that holds its object's address".into())
)]
fn a_huge_id_the_heap_cannot_hold_is_an_error(
    #[case] header: FractalHeapHeader,
    #[case] huge_id: u64,
    #[case] expected: FormatError,
) {
    assert_eq!(
        header.encode_huge_id(huge_id, OffsetWidth::Eight, LengthWidth::Eight),
        Err(expected)
    );
}
