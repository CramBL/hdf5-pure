use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{
    CharacterSet, FormatError, LengthWidth, MaxExtent, OffsetWidth, StoredAddress,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    BlockOffsetWidth, ChunkIndexInfo, ChunkedLayoutFlags, DataLayout, Dataspace, DataspaceType,
    DatatypeLocation, EA_CLIENT_ID_UNFILTERED, FilteredSingleChunk, FormatWidths,
    FractalHeapHeader, LayoutVersion, LinkMessage, LinkTarget,
};
use rstest::rstest;
use test_util::bytes::push_uint;
use test_util::checksum;
use test_util::data_layout;
use test_util::dataspace;
use test_util::fixed_array;
use test_util::fractal_heap;
use test_util::widths::Widths;

#[rstest]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn metadata_fields_use_independent_file_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(1, 8, 64, 1_025)] entries: u64,
    #[values(false, true)] mixed: bool,
) {
    let widths = FormatWidths { offsets, lengths };
    let wire = Widths::new(usize::from(offsets.get()), usize::from(lengths.get()));
    let space = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 2,
        dimensions: vec![3, 5],
        max_dimensions: Some(vec![MaxExtent::Fixed(7), MaxExtent::Fixed(9)]),
    };
    assert_eq!(
        space.serialize(lengths.get()).unwrap(),
        dataspace::v2_with_widths(
            2,
            dataspace::Flags::MAX_DIMS,
            dataspace::Kind::SIMPLE,
            &[3, 5],
            Some(&[7, 9]),
            wire,
        )
    );
    assert_fixed_array_fields(widths, wire, entries);
    assert_manager_fields(widths, wire, entries, mixed);
}

fn assert_fixed_array_fields(widths: FormatWidths, wire: Widths, entries: u64) {
    const HEADER: u64 = 4096;
    const PAGE_BITS: u8 = 10;
    let records: Vec<_> = (0..entries)
        .map(|slot| format::ChunkRecord {
            address: StoredAddress::new(1024 + 4 * slot),
            stored_size: 4,
            filter_mask: 0,
        })
        .collect();
    let fixtures: Vec<_> = records
        .iter()
        .map(|record| {
            Some(fixed_array::Entry {
                address: record.address.get(),
                stored_size: 4,
                filter_mask: 0,
            })
        })
        .collect();
    let header_len = 12 + (wire.offset + wire.length) as u64;
    let block = HEADER + header_len;
    let plan = format::FixedArrayPlan::new(entries, 4, widths, false).unwrap();
    let placement = plan
        .place(StoredAddress::new(HEADER), StoredAddress::new(block))
        .unwrap();
    let expected = [
        (
            StoredAddress::new(HEADER),
            fixed_array::header(
                wire,
                fixed_array::Client::Plain,
                PAGE_BITS,
                entries,
                Some(block),
            ),
        ),
        (
            StoredAddress::new(block),
            fixed_array::data_block(
                wire,
                fixed_array::Client::Plain,
                PAGE_BITS,
                &fixtures,
                HEADER,
            ),
        ),
    ];
    assert_eq!(
        placement
            .serialize_blocks(&format::IndexSlots::dense(&records))
            .unwrap(),
        expected
    );
    assert_eq!(
        placement.blocks().collect::<Vec<_>>(),
        expected.map(|(at, bytes)| (at, bytes.len() as u64))
    );
}

fn assert_manager_fields(widths: FormatWidths, wire: Widths, entries: u64, mixed: bool) {
    const HEADER: u64 = 256;
    const LIST: u64 = 512;
    let (first, stride) = match (wire.offset, entries) {
        (2, 1_025) => (4096, 24),
        (_, 1_025) => (16_384, 32),
        _ => (8192, 32),
    };
    let sections: Vec<_> = (0..entries)
        .map(|slot| format::FreeSection {
            addr: StoredAddress::new(first + stride * slot),
            size: if mixed && slot % 2 == 1 { 24 } else { 16 },
        })
        .collect();
    let mut expected_list = b"FSSE\0".to_vec();
    push_uint(&mut expected_list, HEADER, wire.offset);
    let count_width = if entries <= u64::from(u8::MAX) { 1 } else { 2 };
    for size in [16, 24] {
        let same_size = sections.iter().filter(|section| section.size == size);
        let count = same_size.clone().count();
        if count != 0 {
            push_uint(&mut expected_list, count as u64, count_width);
            push_uint(&mut expected_list, size, wire.length);
            for section in same_size {
                push_uint(&mut expected_list, section.addr.get(), wire.offset);
                expected_list.push(0);
            }
        }
    }
    checksum::append(&mut expected_list);

    let mut expected_header = b"FSHD\0\x01".to_vec();
    let total = sections.iter().map(|section| section.size).sum();
    for value in [total, entries, entries, 0] {
        push_uint(&mut expected_header, value, wire.length);
    }
    for field in [3, 80, 120, u16::from(widths.offsets.get()) * 8 - 1] {
        expected_header.extend_from_slice(&field.to_le_bytes());
    }
    let max_section = match widths.lengths {
        LengthWidth::Two => 0xfffd,
        LengthWidth::Four => 0xffff_fffd,
        LengthWidth::Eight => i64::MAX as u64,
    };
    push_uint(&mut expected_header, max_section, wire.length);
    push_uint(&mut expected_header, LIST, wire.offset);
    for _ in 0..2 {
        push_uint(
            &mut expected_header,
            expected_list.len() as u64,
            wire.length,
        );
    }
    checksum::append(&mut expected_header);

    let (header, list) = format::serialize_free_space_manager(
        widths,
        &sections,
        StoredAddress::new(HEADER),
        StoredAddress::new(LIST),
        format::SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    assert_eq!(header, expected_header);
    assert_eq!(list, expected_list);
    let header = format::FreeSpaceManagerHeader::parse(widths, &header).unwrap();
    let mut parsed =
        format::parse_section_info(widths, &list, StoredAddress::new(HEADER), &header).unwrap();
    parsed.sort_unstable_by_key(|section| section.addr);
    assert_eq!(parsed, sections);
}

#[rstest]
#[case::two(LengthWidth::Two, 0xffff)]
#[case::four(LengthWidth::Four, 0xffff_ffff)]
#[case::eight(LengthWidth::Eight, u64::MAX - 1)]
fn dataspace_dimensions_preserve_the_selected_length_width(
    #[case] width: LengthWidth,
    #[case] value: u64,
) {
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![value],
        max_dimensions: Some(vec![MaxExtent::Fixed(value)]),
    };
    let expected = dataspace::VersionTwo::simple(
        Widths::new(8, usize::from(width.get())),
        &[value],
        Some(&[value]),
    )
    .build();
    assert_eq!(dataspace.serialize(width.get()).unwrap(), expected);
    assert_eq!(dataspace.prepare(width).unwrap().to_bytes(), expected);
    assert_eq!(Dataspace::parse(&expected, width.get()).unwrap(), dataspace);
}

#[rstest]
#[case::scalar(DataspaceType::Scalar, dataspace::VersionTwo::scalar())]
#[case::null(DataspaceType::Null, dataspace::VersionTwo::null())]
fn prepared_scalar_and_null_dataspaces_encode_their_type_codes(
    #[case] space_type: DataspaceType,
    #[case] fixture: dataspace::VersionTwo<'_>,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
) {
    let dataspace = Dataspace {
        space_type,
        rank: 0,
        dimensions: vec![],
        max_dimensions: None,
    };
    assert_eq!(
        dataspace.prepare(width).unwrap().to_bytes(),
        fixture.build()
    );
}

#[rstest]
fn bounded_dataspace_encoding_checks_capacity_before_writing(
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
    #[values(false, true)] store_maxima: bool,
) {
    let dimensions = [0x1234, 0];
    let maxima = [0x2345, 3];
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 2,
        dimensions: dimensions.to_vec(),
        max_dimensions: store_maxima.then(|| maxima.map(MaxExtent::Fixed).to_vec()),
    };
    let expected = dataspace::VersionTwo::simple(
        Widths::new(8, usize::from(width.get())),
        &dimensions,
        store_maxima.then_some(maxima.as_slice()),
    )
    .build();
    let prepared = dataspace.prepare(width).unwrap();
    assert_eq!(prepared.encoded_size(), expected.len());

    const MAX_BODY_WITH_GUARDS: usize = dataspace::V2_HEADER_LEN + 4 * 8 + 3;
    #[repr(C, align(8))]
    struct Aligned([u8; MAX_BODY_WITH_GUARDS]);

    let guard = 0xa5;
    let mut storage = Aligned([guard; MAX_BODY_WITH_GUARDS]);
    let output = &mut storage.0[..expected.len() + 3];
    for available in [0, expected.len() - 1] {
        assert_eq!(
            prepared
                .write_into(&mut output[1..1 + available])
                .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: expected.len(),
                available,
            },
        );
        assert_eq!(output, &[guard; MAX_BODY_WITH_GUARDS][..expected.len() + 3]);
    }
    assert_eq!(
        prepared.write_into(&mut output[1..]).unwrap(),
        expected.len()
    );
    assert_eq!(output[0], guard);
    assert_eq!(&output[1..1 + expected.len()], expected);
    assert_eq!(output[1 + expected.len()..], [guard; 2]);
}

#[rstest]
#[case::simple(DataspaceType::Simple, dataspace::Kind::SIMPLE, 0)]
#[case::scalar(DataspaceType::Scalar, dataspace::Kind::SCALAR, 1)]
#[case::null(DataspaceType::Null, dataspace::Kind::NULL, 1)]
fn preparation_rejects_rank_inconsistent_with_dataspace_type(
    #[case] space_type: DataspaceType,
    #[case] kind: dataspace::Kind,
    #[case] rank: u8,
) {
    let dataspace = Dataspace {
        space_type,
        rank,
        dimensions: vec![0; usize::from(rank)],
        max_dimensions: None,
    };
    assert_eq!(
        dataspace.prepare(LengthWidth::Two).unwrap_err(),
        FormatError::InvalidDataspaceRank {
            space_type: kind.0,
            rank
        },
    );
}

#[rstest]
#[case::missing_current(&[], None)]
#[case::extra_current(&[0, 1], None)]
#[case::missing_maximum(&[0], Some([].as_slice()))]
#[case::extra_maximum(&[0], Some([MaxExtent::Fixed(1), MaxExtent::Fixed(2)].as_slice()))]
fn preparation_rejects_dimension_counts_that_differ_from_rank(
    #[case] dimensions: &[u64],
    #[case] maxima: Option<&[MaxExtent]>,
) {
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: dimensions.to_vec(),
        max_dimensions: maxima.map(<[MaxExtent]>::to_vec),
    };
    assert_eq!(
        dataspace.prepare(LengthWidth::Two).unwrap_err(),
        FormatError::InvalidDataspaceDimensionCount {
            rank: 1,
            dimensions: dimensions.len(),
            maximum_dimensions: maxima.map(<[MaxExtent]>::len),
        },
    );
}

#[rstest]
#[case::unlimited(MaxExtent::Unlimited)]
#[case::fixed_all_ones(MaxExtent::Fixed(u64::MAX))]
fn unlimited_and_fixed_u64_max_encode_the_same_eight_byte_maximum(#[case] maximum: MaxExtent) {
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![0],
        max_dimensions: Some(vec![maximum]),
    };
    let expected = dataspace::VersionTwo::simple(Widths::EIGHT, &[0], Some(&[u64::MAX])).build();
    assert_eq!(
        dataspace.prepare(LengthWidth::Eight).unwrap().to_bytes(),
        expected
    );
    assert_eq!(
        Dataspace::parse_with_length_width(&expected, LengthWidth::Eight)
            .unwrap()
            .max_dimensions,
        Some(vec![MaxExtent::Unlimited]),
    );
}

#[rstest]
#[case::dimension_two(LengthWidth::Two, 0x1_0000, None, 0x1_0000)]
#[case::dimension_four(LengthWidth::Four, 0x1_0000_0000, None, 0x1_0000_0000)]
#[case::maximum_two(LengthWidth::Two, 0, Some(MaxExtent::Fixed(0x1_0000)), 0x1_0000)]
#[case::maximum_four(
    LengthWidth::Four,
    0,
    Some(MaxExtent::Fixed(0x1_0000_0000)),
    0x1_0000_0000
)]
#[case::unlimited_two(LengthWidth::Two, 0, Some(MaxExtent::Unlimited), u64::MAX)]
#[case::unlimited_four(LengthWidth::Four, 0, Some(MaxExtent::Unlimited), u64::MAX)]
#[case::current_before_maximum(LengthWidth::Two, 0x1_0000, Some(MaxExtent::Unlimited), 0x1_0000)]
fn dataspace_fields_outside_the_length_width_are_rejected(
    #[case] width: LengthWidth,
    #[case] dimension: u64,
    #[case] maximum: Option<MaxExtent>,
    #[case] length: u64,
) {
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![dimension],
        max_dimensions: maximum.map(|value| vec![value]),
    };
    assert_eq!(
        dataspace.prepare(width).unwrap_err(),
        FormatError::LengthTooLarge {
            length,
            length_size: width.get()
        },
    );
}

#[rstest]
#[case::two(LengthWidth::Two, MaxExtent::Fixed(0xffff))]
#[case::four(LengthWidth::Four, MaxExtent::Fixed(0xffff_ffff))]
#[case::eight(LengthWidth::Eight, MaxExtent::Unlimited)]
fn unlimited_maxima_are_stored_as_the_value_an_all_ones_field_reads(
    #[case] width: LengthWidth,
    #[case] stored: MaxExtent,
) {
    assert_eq!(
        format::stored_max_extent(width, MaxExtent::Unlimited),
        stored
    );
    let mut all_ones = dataspace::v2(
        1,
        dataspace::Flags::MAX_DIMS,
        dataspace::Kind::SIMPLE,
        &[],
        None,
    );
    all_ones.extend_from_slice(&vec![0; usize::from(width.get())]);
    all_ones.extend_from_slice(&vec![0xff; usize::from(width.get())]);
    assert_eq!(
        Dataspace::parse(&all_ones, width.get()).unwrap(),
        Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![0],
            max_dimensions: Some(vec![stored]),
        }
    );
}

#[rstest]
#[case::two(LengthWidth::Two, 0x1_0000)]
#[case::four(LengthWidth::Four, 0x1_0000_0000)]
fn stored_maxima_reject_a_dimension_outside_the_length_width(
    #[case] width: LengthWidth,
    #[case] dimension: u64,
) {
    assert_eq!(
        format::stored_max_dimensions(width, &[1, dimension], vec![MaxExtent::Unlimited; 2])
            .unwrap_err(),
        FormatError::LengthTooLarge {
            length: dimension,
            length_size: width.get()
        },
    );
}

#[rstest]
#[case::zero(0)]
#[case::one(1)]
#[case::sixteen(16)]
fn dataspace_serialization_rejects_unsupported_length_sizes(#[case] size: u8) {
    let dataspace = Dataspace {
        space_type: DataspaceType::Scalar,
        rank: 0,
        dimensions: vec![],
        max_dimensions: None,
    };
    assert_eq!(
        dataspace.serialize(size).unwrap_err(),
        FormatError::InvalidLengthSize(size)
    );
}

#[rstest]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight, 0xfffe, 0x1_0000_0000)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, 0x1_0000_0000, 0xffff)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four, 0xffff_fffe, 0xffff_ffff)]
fn contiguous_layout_fields_use_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] address: u64,
    #[case] size: u64,
) {
    let mut expected = vec![data_layout::VERSION_FOUR, data_layout::CONTIGUOUS_CLASS];
    expected.extend_from_slice(&address.to_le_bytes()[..usize::from(offsets.get())]);
    expected.extend_from_slice(&size.to_le_bytes()[..usize::from(lengths.get())]);
    assert_eq!(
        DataLayout::encode_contiguous(
            LayoutVersion::Four,
            StoredAddress::new(address),
            size,
            offsets,
            lengths
        )
        .unwrap(),
        expected,
    );
    assert_eq!(
        DataLayout::parse_with_widths(&expected, FormatWidths { offsets, lengths }).unwrap(),
        DataLayout::Contiguous {
            address: Some(StoredAddress::new(address)),
            size
        },
    );
}

#[rstest]
#[case::address_two(OffsetWidth::Two, LengthWidth::Eight, 0x1_0000, 1, FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 })]
#[case::address_four(OffsetWidth::Four, LengthWidth::Two, 0x1_0000_0000, 1, FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 })]
#[case::length_two(OffsetWidth::Eight, LengthWidth::Two, 1, 0x1_0000, FormatError::LengthTooLarge { length: 0x1_0000, length_size: 2 })]
#[case::length_four(OffsetWidth::Two, LengthWidth::Four, 1, 0x1_0000_0000, FormatError::LengthTooLarge { length: 0x1_0000_0000, length_size: 4 })]
fn contiguous_layout_rejects_unrepresentable_fields(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] address: u64,
    #[case] size: u64,
    #[case] expected: FormatError,
) {
    assert_eq!(
        DataLayout::encode_contiguous(
            LayoutVersion::Four,
            StoredAddress::new(address),
            size,
            offsets,
            lengths
        )
        .unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::two(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four(OffsetWidth::Four, LengthWidth::Two)]
#[case::eight(OffsetWidth::Eight, LengthWidth::Four)]
fn unallocated_contiguous_layouts_encode_the_selected_undefined_address(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    let mut expected = vec![data_layout::VERSION_FOUR, data_layout::CONTIGUOUS_CLASS];
    expected.resize(2 + usize::from(offsets.get()), 0xff);
    expected.resize(expected.len() + usize::from(lengths.get()), 0);
    assert_eq!(
        DataLayout::encode_contiguous(
            LayoutVersion::Four,
            StoredAddress::new(u64::MAX),
            0,
            offsets,
            lengths
        )
        .unwrap(),
        expected,
    );
    assert_eq!(
        DataLayout::parse_with_widths(&expected, FormatWidths { offsets, lengths }).unwrap(),
        DataLayout::Contiguous {
            address: None,
            size: 0
        }
    );
}

#[rstest]
#[case::two(LengthWidth::Two, 0x1_0000)]
#[case::four(LengthWidth::Four, 0x1_0000_0000)]
fn filtered_single_chunk_layouts_reject_unrepresentable_sizes(
    #[case] lengths: LengthWidth,
    #[case] filtered_size: u64,
) {
    assert_eq!(
        DataLayout::encode_chunked(
            ChunkedLayoutFlags::NONE,
            &[4],
            4,
            ChunkIndexInfo::SingleChunk {
                filtered: Some(FilteredSingleChunk {
                    filtered_size,
                    filter_mask: 0
                })
            },
            StoredAddress::new(32),
            OffsetWidth::Eight,
            lengths,
        )
        .unwrap_err(),
        FormatError::LengthTooLarge {
            length: filtered_size,
            length_size: lengths.get()
        },
    );
}

#[rstest]
#[case::two(OffsetWidth::Two, 0xffff)]
#[case::four(OffsetWidth::Four, 0xffff_ffff)]
#[case::eight(OffsetWidth::Eight, u64::MAX)]
fn committed_references_and_hard_links_reject_undefined_targets(
    #[case] width: OffsetWidth,
    #[case] address: u64,
) {
    let expected = FormatError::DefinedAddressIsUndefined {
        address,
        offset_size: width.get(),
    };
    assert_eq!(
        format::encode_committed_ref(StoredAddress::new(address), width).unwrap_err(),
        expected
    );
    assert_eq!(
        DatatypeLocation::Committed(StoredAddress::new(address))
            .reference_bytes(width)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        LinkMessage {
            name: "link".into(),
            link_target: LinkTarget::Hard {
                object_header_address: StoredAddress::new(address)
            },
            creation_order: None,
            charset: CharacterSet::Ascii
        }
        .serialize(width)
        .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::two(OffsetWidth::Two, 0x1_0000)]
#[case::four(OffsetWidth::Four, 0x1_0000_0000)]
fn metadata_addresses_outside_the_selected_width_are_rejected(
    #[case] width: OffsetWidth,
    #[case] address: u64,
) {
    let expected = FormatError::AddressTooLarge {
        address,
        offset_size: width.get(),
    };
    let address = StoredAddress::new(address);
    assert_eq!(
        format::encode_committed_ref(address, width).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::next_id(|header: &mut FractalHeapHeader| header.next_huge_object_id = 0x1_0000)]
#[case::free_space(|header: &mut FractalHeapHeader| header.free_space_in_managed_blocks = 0x1_0000)]
#[case::managed_space(|header: &mut FractalHeapHeader| header.managed_space = 0x1_0000)]
#[case::allocated_space(|header: &mut FractalHeapHeader| header.allocated_managed_space = 0x1_0000)]
#[case::iterator_offset(|header: &mut FractalHeapHeader| header.direct_block_allocation_iterator_offset = 0x1_0000)]
#[case::managed_count(|header: &mut FractalHeapHeader| header.managed_objects_count = 0x1_0000)]
#[case::huge_size(|header: &mut FractalHeapHeader| header.huge_objects_size = 0x1_0000)]
#[case::huge_count(|header: &mut FractalHeapHeader| header.huge_objects_count = 0x1_0000)]
#[case::tiny_size(|header: &mut FractalHeapHeader| header.tiny_objects_size = 0x1_0000)]
#[case::tiny_count(|header: &mut FractalHeapHeader| header.tiny_objects_count = 0x1_0000)]
#[case::starting_size(|header: &mut FractalHeapHeader| header.starting_block_size = 0x1_0000)]
#[case::maximum_size(|header: &mut FractalHeapHeader| header.max_direct_block_size = 0x1_0000)]
fn fractal_heap_length_fields_reject_unrepresentable_values(
    #[case] change: fn(&mut FractalHeapHeader),
) {
    let source = fractal_heap::Header::new(128).build(Widths::EIGHT);
    let mut header = FractalHeapHeader::parse(&source, 0, 8, 8).unwrap();
    header.max_direct_block_size = 512;
    change(&mut header);
    assert_eq!(
        header
            .serialize(OffsetWidth::Eight, LengthWidth::Two)
            .unwrap_err(),
        FormatError::LengthTooLarge {
            length: 0x1_0000,
            length_size: 2
        }
    );
}

#[rstest]
#[case::heap_index(|header: &mut FractalHeapHeader| header.btree_huge_objects_address = StoredAddress::new(0x1_0000))]
#[case::free_space_manager(|header: &mut FractalHeapHeader| header.managed_block_free_space_manager_address = StoredAddress::new(0x1_0000))]
#[case::root_block(|header: &mut FractalHeapHeader| header.root_block_address = StoredAddress::new(0x1_0000))]
fn fractal_heap_addresses_outside_the_selected_width_are_rejected(
    #[case] change: fn(&mut FractalHeapHeader),
) {
    let source = fractal_heap::Header::new(128).build(Widths::EIGHT);
    let mut header = FractalHeapHeader::parse(&source, 0, 8, 8).unwrap();
    change(&mut header);
    assert_eq!(
        header
            .serialize(OffsetWidth::Two, LengthWidth::Eight)
            .unwrap_err(),
        FormatError::AddressTooLarge {
            address: 0x1_0000,
            offset_size: 2
        }
    );
}

#[rstest]
#[case::owner_address(StoredAddress::new(0x1_0000), 0, vec![], FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 })]
#[case::child_address(StoredAddress::new(0), 0, vec![StoredAddress::new(0x1_0000)], FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 })]
#[case::block_offset(StoredAddress::new(0), 256, vec![], FormatError::ValueTooLargeForRepresentation { value: 256, target: "Extensible Array block offset" })]
fn extensible_array_super_blocks_reject_unrepresentable_fields(
    #[case] owner: StoredAddress,
    #[case] block_offset: u64,
    #[case] children: Vec<StoredAddress>,
    #[case] expected: FormatError,
) {
    assert_eq!(
        format::encode_super_block(
            owner,
            block_offset,
            &[],
            &children,
            OffsetWidth::Two,
            EA_CLIENT_ID_UNFILTERED,
            BlockOffsetWidth::One
        )
        .unwrap_err(),
        expected
    );
}
