use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::*;
use crate::fractal_heap::test_support;

#[test]
fn a_managed_id_decodes_to_its_offset_and_length() {
    let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
    let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

    // Byte 0 holds version 0 in bits 6-7 and type 0 in bits 4-5. Bytes 1-6 hold the offset in
    // `max_heap_size` (16) bits, then the length: offset 0 and length 13 pack to 0x000D_0000.
    let offset: u64 = 0;
    let length: u64 = 13;
    let payload = offset | (length << hdr.max_heap_size);
    let mut id = vec![0u8; 7];
    id[0] = 0x00; // type=0
    for i in 0..6 {
        id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
    }

    assert_eq!(
        FractalHeapIdLayout::new(&hdr, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset: 0,
            object_length: 13
        },
    );
}

#[rstest]
#[case::too_short(9, vec![0; 8])]
#[case::too_long(7, vec![0; 8])]
#[case::empty(8, vec![])]
#[case::short_with_unsupported_version(2, vec![0x60])]
#[case::long_with_reserved_flags(7, vec![0x01, 0, 0, 0, 0, 0, 0, 0])]
fn an_id_must_have_the_declared_length(#[case] declared: u16, #[case] bytes: Vec<u8>) {
    let header = FractalHeapHeader {
        heap_id_length: declared,
        ..attribute_heap_header()
    };
    let layout = FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight);
    assert_eq!(layout.encoded_len(), usize::from(declared));
    assert_eq!(
        layout.parse(&bytes),
        Err(FormatError::InvalidFractalHeapIdLength {
            expected: usize::from(declared),
            actual: bytes.len(),
        })
    );
}

#[rstest]
#[case::zero_bits(0, 9)]
#[case::offset_above_integer(65, 11)]
#[case::missing_length(64, 9)]
fn invalid_managed_id_geometry_is_rejected(#[case] bits: u16, #[case] id_length: u16) {
    let header = FractalHeapHeader {
        heap_id_length: id_length,
        max_heap_size: bits,
        ..attribute_heap_header()
    };
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&vec![0; usize::from(id_length)]),
        Err(FormatError::InvalidFractalHeapGeometry)
    );
    assert_eq!(
        header.encode_managed_id(0, 1),
        Err(FormatError::InvalidFractalHeapGeometry)
    );
}

#[rstest]
#[case::group(32, 7, 0x1234_5678, 0x9ABC, vec![0, 0x78, 0x56, 0x34, 0x12, 0xBC, 0x9A])]
#[case::unaligned(33, 8, 0x1_2345_6789, 0xABCD, vec![0, 0x89, 0x67, 0x45, 0x23, 1, 0xCD, 0xAB])]
#[case::wide_offset(64, 11, u64::MAX, 0xAB, vec![0, 255, 255, 255, 255, 255, 255, 255, 255, 0xAB, 0])]
#[case::padded(8, 12, 0xAB, 0xCDEF, vec![0, 0xAB, 0xEF, 0xCD, 0, 0, 0, 0, 0, 0, 0, 0])]
fn managed_id_fields_are_byte_aligned_and_independent(
    #[case] bits: u16,
    #[case] id_length: u16,
    #[case] heap_offset: u64,
    #[case] object_length: u64,
    #[case] expected: Vec<u8>,
) {
    let header = FractalHeapHeader {
        heap_id_length: id_length,
        max_heap_size: bits,
        ..attribute_heap_header()
    };
    assert_eq!(
        header.encode_managed_id(heap_offset, object_length),
        Ok(expected.clone())
    );
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&expected)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset,
            object_length
        }
    );
}

#[test]
fn managed_id_padding_is_not_part_of_the_length() {
    let header = FractalHeapHeader {
        heap_id_length: 10,
        ..attribute_heap_header()
    };
    let id = [0, 42, 0, 0, 0, 0, 13, 0, 0xAA, 0xBB];
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset: 42,
            object_length: 13
        }
    );
}

#[rstest]
#[case::unused_offset_bits(33, vec![0, 0, 0, 0, 0, 2, 1, 0])]

fn managed_ids_reject_nonzero_unused_offset_bits(#[case] bits: u16, #[case] bytes: Vec<u8>) {
    let header = FractalHeapHeader {
        heap_id_length: u16::try_from(bytes.len()).unwrap(),
        max_heap_size: bits,
        ..attribute_heap_header()
    };
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).parse(&bytes),
        Err(FormatError::InvalidFractalHeapGeometry)
    );
}

#[rstest]
#[case::reserved_type(vec![0x30, 0, 0, 0, 0, 0, 0, 0], FormatError::InvalidHeapIdType(3))]
#[case::tiny_past_id(vec![0x2F, 0, 0, 0, 0, 0, 0, 0], FormatError::UnexpectedEof { expected: 17, available: 8 })]
fn a_complete_id_with_invalid_fields_is_an_error(
    #[case] bytes: Vec<u8>,
    #[case] error: FormatError,
) {
    assert_eq!(
        FractalHeapIdLayout::new(
            &attribute_heap_header(),
            OffsetWidth::Eight,
            LengthWidth::Eight
        )
        .parse(&bytes),
        Err(error)
    );
}

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
        ..attribute_heap_header()
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
        ..attribute_heap_header()
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
        ..attribute_heap_header()
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
        ..attribute_heap_header()
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
        ..attribute_heap_header()
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
        ..attribute_heap_header()
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

#[rstest]
#[case::managed(MANAGED_ID, false)]
#[case::tiny_normal(&[0x20, 0xAB], false)]
#[case::tiny_extended(b"\x20\x01ab...............", false)]
#[case::huge_indirect(INDIRECT_HUGE_ID, false)]
#[case::huge_direct(DIRECT_HUGE_ID, false)]
#[case::filtered_huge_indirect(INDIRECT_HUGE_ID, true)]
#[case::filtered_huge_direct(FILTERED_DIRECT_HUGE_ID, true)]
#[case::reserved_type(&[0x3F], false)]
fn heap_id_parser_checks_the_version_before_the_type(
    #[case] id: &[u8],
    #[case] filtered: bool,
    #[values((0x40, 1), (0x80, 2), (0xC0, 3))] version: (u8, u8),
) {
    let layout = heap_id_fixture_layout(id.len(), filtered);
    let (version_bits, version) = version;
    let mut id = id.to_vec();
    id[0] |= version_bits;
    assert_eq!(
        layout.parse(&id),
        Err(FormatError::UnsupportedFractalHeapIdVersion(version))
    );
}

#[rstest]
#[case::managed(MANAGED_ID, false)]
#[case::huge_indirect(INDIRECT_HUGE_ID, false)]
#[case::huge_direct(DIRECT_HUGE_ID, false)]
#[case::filtered_huge_indirect(INDIRECT_HUGE_ID, true)]
#[case::filtered_huge_direct(FILTERED_DIRECT_HUGE_ID, true)]
fn heap_id_parser_rejects_reserved_bits_in_managed_and_huge_ids(
    #[case] id: &[u8],
    #[case] filtered: bool,
    #[values(0x01, 0x0F)] flags: u8,
) {
    let layout = heap_id_fixture_layout(id.len(), filtered);
    let mut id = id.to_vec();
    id[0] |= flags;
    assert_eq!(
        layout.parse(&id),
        Err(FormatError::InvalidFractalHeapIdFlags(flags))
    );
}

fn heap_id_fixture_layout(length: usize, filtered: bool) -> FractalHeapIdLayout {
    let header = FractalHeapHeader {
        heap_id_length: u16::try_from(length).unwrap(),
        filtering: if filtered {
            test_support::filtered_heap()
        } else {
            FractalHeapFiltering::Unfiltered
        },
        ..attribute_heap_header()
    };
    FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
}

const MANAGED_ID: &[u8] = &[0x00, 42, 0, 0, 0, 0, 13, 0];
const INDIRECT_HUGE_ID: &[u8] = &[0x10, 5, 0, 0, 0, 0, 0, 0];
const DIRECT_HUGE_ID: &[u8] = &[0x10, 0x34, 0x12, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0];
const FILTERED_DIRECT_HUGE_ID: &[u8] = &[
    0x10, 0x34, 0x12, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
];

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

#[rstest]
#[case::the_first_object(0, 13, [0, 0, 0, 0, 0, 0, 13, 0])]
#[case::an_offset_and_a_length(100, 42, [0, 100, 0, 0, 0, 0, 42, 0])]
#[case::the_widest_offset_and_length(
    (1 << 40) - 1,
    u64::from(u16::MAX),
    [0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
)]
fn an_encoded_managed_id_holds_its_offset_then_its_length(
    #[case] heap_offset: u64,
    #[case] length: u64,
    #[case] expected: [u8; 8],
) {
    let header = attribute_heap_header();
    let id = header.encode_managed_id(heap_offset, length).unwrap();
    assert_eq!(id, expected);
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset,
            object_length: length
        },
    );
}

#[rstest]
#[case::an_offset_past_the_offset_bits(
    1 << 40,
    1,
    "heap offset 1099511627776 does not fit the heap's 40-bit offsets"
)]
#[case::a_length_past_the_id(0, 1 << 16, "a managed object length does not fit a 8-byte heap ID")]
fn a_managed_id_that_does_not_fit_is_an_error(
    #[case] heap_offset: u64,
    #[case] length: u64,
    #[case] message: &str,
) {
    assert_eq!(
        attribute_heap_header().encode_managed_id(heap_offset, length),
        Err(FormatError::Internal(message.into()))
    );
}

#[test]
fn an_encoded_huge_id_decodes_to_its_key() {
    let header = attribute_heap_header();
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
    attribute_heap_header(),
    1 << 56,
    FormatError::Internal("a huge object key does not fit a 8-byte heap ID".into())
)]
#[case::an_id_that_holds_the_address(
    FractalHeapHeader { heap_id_length: 17, ..attribute_heap_header() },
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

/// Returns a header with the heap ID layout of the attribute heaps `hdf5-pure` writes: 8-byte
/// IDs over a 40-bit heap, with a huge-object B-tree.
fn attribute_heap_header() -> FractalHeapHeader {
    FractalHeapHeader {
        heap_id_length: 8,
        max_heap_size: 40,
        max_managed_object_size: 65_535,
        btree_huge_objects_address: StoredAddress::new(0x800),
        ..test_support::dtable_header(1024, 65536, 4)
    }
}
