use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{
    CharacterSet, FormatError, LengthWidth, MaxExtent, OffsetWidth, StoredAddress,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    AttributeInfoMessage, BlockOffsetWidth, ChunkIndexInfo, ChunkedLayoutFlags, DataLayout,
    Dataspace, DataspaceType, DatatypeLocation, EA_CLIENT_ID_UNFILTERED, FilteredSingleChunk,
    FormatWidths, FractalHeapHeader, LayoutVersion, LinkInfoMessage, LinkMessage, LinkTarget,
};
use rstest::rstest;
use test_util::data_layout;
use test_util::dataspace;
use test_util::fractal_heap;
use test_util::widths::Widths;

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
    let mut expected = dataspace::v2(
        1,
        dataspace::Flags::MAX_DIMS,
        dataspace::Kind::SIMPLE,
        &[],
        None,
    );
    expected.extend_from_slice(&value.to_le_bytes()[..usize::from(width.get())]);
    expected.extend_from_slice(&value.to_le_bytes()[..usize::from(width.get())]);
    assert_eq!(dataspace.serialize(width.get()).unwrap(), expected);
    assert_eq!(Dataspace::parse(&expected, width.get()).unwrap(), dataspace);
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
        dataspace.serialize(width.get()).unwrap_err(),
        FormatError::LengthTooLarge {
            length,
            length_size: width.get()
        },
    );
}

#[test]
fn unlimited_maxima_remain_unlimited_at_eight_bytes() {
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![0],
        max_dimensions: Some(vec![MaxExtent::Unlimited]),
    };
    let mut expected = dataspace::v2(
        1,
        dataspace::Flags::MAX_DIMS,
        dataspace::Kind::SIMPLE,
        &[],
        None,
    );
    expected.extend_from_slice(&[0; 8]);
    expected.extend_from_slice(&[0xff; 8]);
    assert_eq!(dataspace.serialize(8).unwrap(), expected);
    assert_eq!(Dataspace::parse(&expected, 8).unwrap(), dataspace);
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
    assert_eq!(
        LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: Some(address),
            btree_name_index_address: None,
            btree_creation_order_address: None
        }
        .serialize(width)
        .unwrap_err(),
        expected
    );
    assert_eq!(
        AttributeInfoMessage {
            max_creation_index: None,
            indexes_creation_order: false,
            fractal_heap_address: Some(address),
            btree_name_index_address: None,
            btree_creation_order_address: None
        }
        .serialize(width)
        .unwrap_err(),
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
#[case::heap(|info: &mut LinkInfoMessage| info.fractal_heap_address = Some(StoredAddress::new(0xffff)))]
#[case::name_index(|info: &mut LinkInfoMessage| info.btree_name_index_address = Some(StoredAddress::new(0xffff)))]
#[case::creation_index(|info: &mut LinkInfoMessage| info.btree_creation_order_address = Some(Some(StoredAddress::new(0xffff))))]
fn link_info_rejects_defined_addresses_equal_to_the_undefined_marker(
    #[case] change: fn(&mut LinkInfoMessage),
) {
    let mut info = LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    change(&mut info);
    assert_eq!(
        info.serialize(OffsetWidth::Two).unwrap_err(),
        FormatError::DefinedAddressIsUndefined {
            address: 0xffff,
            offset_size: 2
        }
    );
}

#[rstest]
#[case::heap(|info: &mut AttributeInfoMessage| info.fractal_heap_address = Some(StoredAddress::new(0xffff)))]
#[case::name_index(|info: &mut AttributeInfoMessage| info.btree_name_index_address = Some(StoredAddress::new(0xffff)))]
#[case::creation_index(|info: &mut AttributeInfoMessage| info.btree_creation_order_address = Some(StoredAddress::new(0xffff)))]
fn attribute_info_rejects_defined_addresses_equal_to_the_undefined_marker(
    #[case] change: fn(&mut AttributeInfoMessage),
) {
    let mut info = AttributeInfoMessage {
        max_creation_index: None,
        indexes_creation_order: true,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    change(&mut info);
    assert_eq!(
        info.serialize(OffsetWidth::Two).unwrap_err(),
        FormatError::DefinedAddressIsUndefined {
            address: 0xffff,
            offset_size: 2
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
