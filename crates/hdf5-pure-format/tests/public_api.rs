use core::num::NonZeroU16;

use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_format::AttributeMessage;
use hdf5_pure_format::BTREE_V2_NODE_SIZE;
use hdf5_pure_format::BTreeV1Node;
use hdf5_pure_format::BTreeV2Header;
use hdf5_pure_format::BTreeV2NodeInfo;
use hdf5_pure_format::BTreeV2Plan;
use hdf5_pure_format::BTreeV2Record;
use hdf5_pure_format::BaseAddress;
use hdf5_pure_format::Dataspace;
use hdf5_pure_format::DataspaceType;
use hdf5_pure_format::Datatype;
use hdf5_pure_format::DatatypeByteOrder;
use hdf5_pure_format::DatatypeLocation;
use hdf5_pure_format::FillValueError;
use hdf5_pure_format::FilterDescription;
use hdf5_pure_format::FilterPipeline;
use hdf5_pure_format::FilterPipelineError;
use hdf5_pure_format::FixedPointLayout;
use hdf5_pure_format::FormatError;
use hdf5_pure_format::FractalHeapChild;
use hdf5_pure_format::FractalHeapHeader;
use hdf5_pure_format::FractalHeapIdType;
use hdf5_pure_format::GLOBAL_HEAP_MAX_OBJECTS;
use hdf5_pure_format::GlobalHeapCollectionError;
use hdf5_pure_format::GlobalHeapIndex;
use hdf5_pure_format::GlobalHeapObjectInfo;
use hdf5_pure_format::HugeObjectReference;
use hdf5_pure_format::LengthWidth;
use hdf5_pure_format::LocalHeap;
use hdf5_pure_format::MaxExtent;
use hdf5_pure_format::MessageType;
use hdf5_pure_format::OffsetWidth;
use hdf5_pure_format::SharedResolver;
use hdf5_pure_format::StoredAddress;
use hdf5_pure_format::SymbolTableEntry;
use hdf5_pure_format::SymbolTableMessage;
use hdf5_pure_format::SymbolTableNode;
use hdf5_pure_format::V3_FLAGS_DEFAULT;
use rstest::rstest;
use test_util::attribute;
use test_util::btree_v1;
use test_util::btree_v2;
use test_util::dataspace;
use test_util::datatype;
use test_util::fractal_heap;
use test_util::image::Image;
use test_util::local_heap;
use test_util::symbol_table;
use test_util::widths::Widths;

#[test]
fn on_disk_messages_round_trip_through_the_public_api() {
    let datatype = Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        layout: FixedPointLayout {
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        },
    };
    let datatype_bytes = hdf5_pure_format::serialize_datatype(&datatype);
    assert_eq!(
        hdf5_pure_format::element_size_usize(&datatype)
            .unwrap()
            .get(),
        4
    );
    assert_eq!(
        hdf5_pure_format::parse_datatype(&datatype_bytes),
        Ok((datatype, datatype_bytes.len()))
    );

    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![3],
        max_dimensions: Some(vec![MaxExtent::Unlimited]),
    };
    let dataspace_bytes = dataspace.serialize(8);
    assert_eq!(Dataspace::parse(&dataspace_bytes, 8), Ok(dataspace));

    let pipeline = FilterPipeline {
        version: 2,
        filters: vec![FilterDescription {
            filter_id: 1,
            name: None,
            flags: 1,
            client_data: vec![6],
        }],
    };
    let pipeline_bytes = pipeline.serialize().unwrap();
    let parsed_pipeline: Result<FilterPipeline, FilterPipelineError> =
        FilterPipeline::parse(&pipeline_bytes);
    assert_eq!(parsed_pipeline, Ok(pipeline));

    let fill_message: Result<Vec<u8>, FillValueError> =
        hdf5_pure_format::fill_value_message_v3(None);
    assert_eq!(fill_message, Ok(vec![3, V3_FLAGS_DEFAULT]));
}

#[test]
fn an_object_header_round_trips_through_the_public_api() {
    let mut writer = hdf5_pure_format::ObjectHeaderWriter::new();
    writer.add_message_with_flags(
        hdf5_pure_format::MessageType::DATASPACE,
        vec![1, 2, 3],
        hdf5_pure_format::MessageFlags::CONSTANT,
    );
    let bytes = writer.serialize().unwrap();

    let buffered = hdf5_pure_format::ObjectHeader::parse(
        &bytes,
        hdf5_pure_format::AccessMode::ReadOnly,
        0,
        8,
        8,
    )
    .unwrap();
    let streamed = hdf5_pure_format::ObjectHeader::parse_from_source(
        bytes.as_slice(),
        hdf5_pure_format::AccessMode::ReadOnly,
        0,
        8,
        8,
        hdf5_pure_format::BaseAddress::ZERO,
    )
    .unwrap();

    for header in [buffered, streamed] {
        let [message] = header.messages.as_slice() else {
            panic!("expected one message, got {:?}", header.messages);
        };
        assert_eq!(message.msg_type, hdf5_pure_format::MessageType::DATASPACE);
        assert_eq!(message.flags, hdf5_pure_format::MessageFlags::CONSTANT);
        assert_eq!(message.data, vec![1, 2, 3]);
    }
}

#[test]
fn group_and_attribute_storage_messages_round_trip_through_the_public_api() {
    let link = hdf5_pure_format::LinkMessage {
        name: "data".into(),
        link_target: hdf5_pure_format::LinkTarget::Hard {
            object_header_address: hdf5_pure_format::StoredAddress::new(96),
        },
        creation_order: None,
        charset: hdf5_pure_format::CharacterSet::Ascii,
    };
    let link_bytes = link.serialize(hdf5_pure_format::OffsetWidth::Eight);
    assert!(hdf5_pure_format::link_is_named(&link_bytes, "data"));
    assert_eq!(
        hdf5_pure_format::LinkMessage::parse(&link_bytes, 8),
        Ok(link)
    );

    let attribute_info = hdf5_pure_format::AttributeInfoMessage {
        max_creation_index: Some(2),
        indexes_creation_order: true,
        fractal_heap_address: Some(hdf5_pure_format::StoredAddress::new(4096)),
        btree_name_index_address: Some(hdf5_pure_format::StoredAddress::new(8192)),
        btree_creation_order_address: None,
    };
    let attribute_info_bytes = attribute_info.serialize(hdf5_pure_format::OffsetWidth::Eight);
    assert_eq!(
        hdf5_pure_format::AttributeInfoMessage::parse(&attribute_info_bytes, 8),
        Ok(attribute_info)
    );

    assert_eq!(
        hdf5_pure_format::LinkInfoMessage::parse(&[0, 0, 0xFF, 0xFF, 0xFF, 0xFF], 2),
        Ok(hdf5_pure_format::LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: None,
            btree_name_index_address: None,
            btree_creation_order_address: None,
        })
    );
}

#[test]
fn shared_message_references_round_trip_through_the_public_api() {
    let address = StoredAddress::new(800);
    let committed = hdf5_pure_format::encode_committed_ref(address, OffsetWidth::Eight);
    assert_eq!(
        hdf5_pure_format::parse_shared_ref(&committed, 8, 8).map(|reference| reference.location),
        Ok(hdf5_pure_format::SharedLocation::ObjectHeader(address))
    );
    assert_eq!(
        hdf5_pure_format::committed_address_in(&committed, 8, 8),
        Ok(Some(address))
    );

    let heap_id = [7; hdf5_pure_format::FHEAP_ID_LEN];
    let heap = hdf5_pure_format::encode_sohm_ref(&heap_id);
    assert_eq!(
        hdf5_pure_format::committed_address_in(&heap, 8, 8),
        Ok(None)
    );
    assert_eq!(
        hdf5_pure_format::Unresolvable.resolve(&heap, MessageType::DATASPACE),
        Err(FormatError::UnresolvedSharedMessage(
            MessageType::DATASPACE.to_u16()
        ))
    );
}

#[test]
fn a_superblock_round_trips_through_the_public_api() {
    let superblock = hdf5_pure_core::__private::SuperblockFields {
        version: 2,
        offset_size: 8,
        length_size: 8,
        base_address: hdf5_pure_format::BaseAddress::ZERO,
        eof_address: 4096,
        root_group_address: 48,
        group_leaf_node_k: None,
        group_internal_node_k: None,
        indexed_storage_internal_node_k: None,
        free_space_address: None,
        driver_info_address: None,
        consistency_flags: 0,
        superblock_extension_address: Some(u64::MAX),
        checksum: None,
    }
    .build();
    let bytes = hdf5_pure_format::serialize_superblock(&superblock).unwrap();

    let parsed = hdf5_pure_format::parse_superblock(&bytes, 0).unwrap();
    let streamed = hdf5_pure_format::parse_superblock_from_source(bytes.as_slice(), 0).unwrap();

    assert_eq!(parsed, streamed);
    assert_eq!(parsed.eof_address, 4096);
    assert_eq!(parsed.root_group_address, 48);
    assert_eq!(parsed.superblock_extension_address, Some(u64::MAX));
    assert_eq!(hdf5_pure_format::serialize_superblock(&parsed), Ok(bytes));
}

#[test]
fn a_file_space_info_message_round_trips_through_the_public_api() {
    let info = hdf5_pure_core::__private::FileSpaceInfoFields {
        strategy: hdf5_pure_format::FileSpaceStrategy::Page,
        persist: false,
        threshold: 1,
        page_size: 4096,
        page_end_meta_threshold: 0,
        eoa_pre_fsm: u64::MAX,
        manager_addrs: Vec::new(),
    }
    .build();
    let bytes = hdf5_pure_format::serialize_file_space_info(&info);

    assert_eq!(
        hdf5_pure_format::parse_file_space_info(&bytes, 8, 8),
        Ok(info)
    );
}

#[test]
fn a_local_heap_reads_its_names_through_the_public_api() {
    let segment_at = local_heap::header_len(Widths::EIGHT);
    let segment = local_heap::Segment::of_names(segment_at as u64, &["alpha", "beta"]);
    let mut image = Image::starting_with(&segment.header(Widths::EIGHT));
    image.place(segment_at, &segment.bytes);
    let file = image.build();

    let expected = LocalHeap {
        data_segment_size: segment.bytes.len() as u64,
        free_list_head_offset: segment.free_list_head,
        data_segment_address: StoredAddress::new(segment_at as u64),
    };

    assert_eq!(LocalHeap::parse(&file, 0, 8, 8), Ok(expected.clone()));
    assert_eq!(
        LocalHeap::parse_from_source(file.as_slice(), 0, 8, 8),
        Ok(expected.clone())
    );
    assert_eq!(
        expected.read_string(&file, BaseAddress::ZERO, segment.offset_of(1)),
        Ok("beta".into())
    );
    assert_eq!(
        expected.read_string_in_segment(&segment.bytes, segment.offset_of(0)),
        Ok("alpha".into())
    );
}

#[test]
fn symbol_table_structures_parse_through_the_public_api() {
    let mut message = 0x100u64.to_le_bytes().to_vec();
    message.extend_from_slice(&0x200u64.to_le_bytes());
    assert_eq!(
        SymbolTableMessage::parse(&message, 8),
        Ok(SymbolTableMessage {
            btree_address: StoredAddress::new(0x100),
            local_heap_address: StoredAddress::new(0x200),
        })
    );

    let entry = symbol_table::Entry {
        cache_type: 1,
        scratch_pad: [7; symbol_table::SCRATCH_PAD],
        ..symbol_table::Entry::new(8, 0x300)
    };
    let node = symbol_table::node(&[entry], Widths::EIGHT);
    let expected = SymbolTableNode {
        entries: vec![SymbolTableEntry {
            link_name_offset: 8,
            object_header_address: StoredAddress::new(0x300),
            cache_type: 1,
            scratch_pad: [7; 16],
        }],
    };
    assert_eq!(SymbolTableNode::parse(&node, 0, 8, 8), Ok(expected.clone()));
    assert_eq!(
        SymbolTableNode::parse_from_source(node.as_slice(), 0, 8, 8),
        Ok(expected)
    );
}

#[test]
fn a_version_1_group_btree_node_parses_through_the_public_api() {
    let widths = Widths::new(4, 8);
    let keys = [
        btree_v1::group_key(0, widths),
        btree_v1::group_key(6, widths),
    ];
    let node = btree_v1::node(btree_v1::NodeType::GROUP, 0, &keys, &[0x300], widths);
    let expected = BTreeV1Node {
        node_type: 0,
        node_level: 0,
        entries_used: 1,
        left_sibling: None,
        right_sibling: None,
        keys: vec![0, 6],
        children: vec![StoredAddress::new(0x300)],
    };

    assert_eq!(hdf5_pure_format::btree_v1_node_header_size(4), 16);
    assert_eq!(BTreeV1Node::parse(&node, 0, 4, 8), Ok(expected.clone()));
    assert_eq!(
        BTreeV1Node::parse_from_source(node.as_slice(), 0, 4, 8),
        Ok(expected)
    );
}

#[test]
fn version_2_btree_nodes_parse_through_the_public_api() {
    let widths = Widths::new(4, 8);
    let header = btree_v2::Header::new(5, 11, 0x200, 1)
        .depth(1)
        .total_records(3)
        .build(widths);
    let leaf = btree_v2::leaf(5, &[vec![1; 11], vec![2; 11]]);
    let child = |address| btree_v2::Child {
        address,
        records: 1,
        records_width: 1,
        subtree: None,
    };
    let internal = btree_v2::internal(5, &[vec![3; 11]], &[child(0x300), child(0x400)], widths);
    let expected = BTreeV2Header {
        tree_type: 5,
        node_size: 512,
        record_size: 11,
        depth: 1,
        root_node_address: StoredAddress::new(0x200),
        num_records_in_root: 1,
        total_records: 3,
    };

    assert_eq!(BTreeV2Header::parse(&header, 0, 4, 8), Ok(expected.clone()));
    assert_eq!(
        BTreeV2Header::parse_from_source(header.as_slice(), 0, 4, 8),
        Ok(expected)
    );
    assert_eq!(
        hdf5_pure_format::parse_btree_v2_leaf_records(&leaf, 0, 2, 11),
        Ok(vec![
            BTreeV2Record { data: vec![1; 11] },
            BTreeV2Record { data: vec![2; 11] },
        ])
    );
    assert_eq!(
        hdf5_pure_format::parse_btree_v2_internal_child_pointers(
            &internal,
            1,
            NonZeroU16::MIN,
            11,
            4,
            &BTreeV2NodeInfo::compute(512, 11, 4, 1),
        ),
        Ok(vec![
            (StoredAddress::new(0x300), 1),
            (StoredAddress::new(0x400), 1),
        ])
    );
}

#[test]
fn a_planned_version_2_btree_parses_back_through_the_public_api() {
    let records: Vec<u8> = (0..34).collect();
    let nodes_address = StoredAddress::new(hdf5_pure_format::btree_v2_header_size(
        OffsetWidth::Eight,
        LengthWidth::Eight,
    ) as u64);
    let plan = BTreeV2Plan::new(8, 2, 17, BTREE_V2_NODE_SIZE, OffsetWidth::Eight).unwrap();
    let image = plan.serialize(
        &records,
        nodes_address,
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    let (node_info, depth) =
        BTreeV2NodeInfo::for_record_count(BTREE_V2_NODE_SIZE, 17, 8, 30).unwrap();

    assert_eq!(depth, 1);
    assert_eq!(node_info.max_nrec(0), 29);
    assert!(BTreeV2Plan::new(8, 100_000, 1, 1_000_000, OffsetWidth::Eight).is_none());
    assert_eq!(plan.nodes_size(), u64::from(BTREE_V2_NODE_SIZE));

    assert_eq!(
        BTreeV2Header::parse(&image.header, 0, 8, 8),
        Ok(BTreeV2Header {
            tree_type: 8,
            node_size: BTREE_V2_NODE_SIZE,
            record_size: 17,
            depth: 0,
            root_node_address: nodes_address,
            num_records_in_root: 2,
            total_records: 2,
        })
    );
    assert_eq!(image.nodes.len(), BTREE_V2_NODE_SIZE as usize);
    assert_eq!(
        hdf5_pure_format::parse_btree_v2_leaf_records(&image.nodes, 0, 2, 17),
        Ok(vec![
            BTreeV2Record {
                data: records[..17].to_vec(),
            },
            BTreeV2Record {
                data: records[17..].to_vec(),
            },
        ])
    );
}

#[test]
fn a_fractal_heap_header_and_its_heap_ids_decode_through_the_public_api() {
    let bytes = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
    let expected = FractalHeapHeader {
        heap_id_length: 7,
        io_filter_encoded_length: 0,
        max_managed_object_size: 64,
        btree_huge_objects_address: StoredAddress::new(u64::MAX),
        table_width: 4,
        starting_block_size: 128,
        max_direct_block_size: 1024,
        max_heap_size: 16,
        start_root_rows: 2,
        root_block_address: StoredAddress::new(256),
        current_rows_in_root_indirect_block: 0,
        managed_objects_count: 1,
    };
    let managed = [0x00, 15, 0, 13, 0, 0, 0];
    let huge = [0x10, 5, 0, 0, 0, 0, 0];
    let mut inline_huge = vec![0x10];
    inline_huge.extend_from_slice(&0x900u64.to_le_bytes());
    inline_huge.extend_from_slice(&40u64.to_le_bytes());
    let indexed = FractalHeapHeader {
        btree_huge_objects_address: StoredAddress::new(0x800),
        ..expected.clone()
    };
    let wide = FractalHeapHeader {
        heap_id_length: 17,
        ..expected.clone()
    };

    assert_eq!(
        FractalHeapHeader::parse(&bytes, 0, 8, 8),
        Ok(expected.clone())
    );
    assert_eq!(
        FractalHeapHeader::parse_from_source(bytes.as_slice(), 0, 8, 8),
        Ok(expected.clone())
    );
    assert_eq!(
        FractalHeapIdType::from_heap_id(&managed),
        Ok(FractalHeapIdType::Managed)
    );
    assert_eq!(expected.decode_managed_id(&managed), Ok((15, 13)));
    assert_eq!(
        expected.decode_tiny_id(&[0x22, b'a', b'b', b'c', 0, 0, 0]),
        Ok(b"abc".to_vec())
    );
    assert_eq!(
        indexed.decode_huge_id(&huge, 8, 8),
        Ok(HugeObjectReference::Indexed(5))
    );
    assert_eq!(
        wide.decode_huge_id(&inline_huge, 8, 8),
        Ok(HugeObjectReference::Inline {
            addr: StoredAddress::new(0x900),
            len: 40,
        })
    );
    assert_eq!(
        expected.decode_tiny_id(&[]),
        Err(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })
    );
    assert_eq!(
        indexed.decode_huge_id(&[], 8, 8),
        Err(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })
    );
}

#[test]
fn an_indirect_block_locates_its_children_through_the_public_api() {
    let header = FractalHeapHeader::parse(
        &fractal_heap::Header::new(0x100).build(Widths::EIGHT),
        0,
        8,
        8,
    )
    .unwrap();
    let filtered = FractalHeapHeader {
        io_filter_encoded_length: 8,
        ..header.clone()
    };
    let mut block = fractal_heap::INDIRECT_BLOCK_SIGNATURE.to_vec();
    block.push(0);
    block.extend_from_slice(&0u64.to_le_bytes());
    block.extend_from_slice(&[0, 0]);
    for address in [0x1000u64, u64::MAX, 0x2000, u64::MAX] {
        block.extend_from_slice(&address.to_le_bytes());
    }

    assert_eq!(
        header.indirect_block_entries_len(1, 8),
        Ok(block.len() as u64)
    );
    assert_eq!(
        header.find_child_for_offset(&block, 1, 0, 300, 8),
        Ok(Some(FractalHeapChild::Direct {
            addr: StoredAddress::new(0x2000),
            block_size: 128,
            heap_offset: 256,
        }))
    );
    assert_eq!(header.find_child_for_offset(&block, 1, 0, 200, 8), Ok(None));
    assert_eq!(
        filtered.indirect_block_entries_len(1, 8),
        Err(FormatError::UnsupportedFilteredHeapObject)
    );
    assert_eq!(
        filtered.find_child_for_offset(&block, 1, 0, 300, 8),
        Err(FormatError::UnsupportedFilteredHeapObject)
    );
}

#[test]
fn a_global_heap_collection_round_trips_through_the_public_api() {
    let objects: [&[u8]; 2] = [b"alpha", b"beta"];
    let bytes =
        hdf5_pure_format::encode_global_heap_collection(LengthWidth::Eight, &objects).unwrap();

    let collection = GlobalHeapIndex::parse(bytes.as_slice(), 0, 8).unwrap();
    let filtered =
        GlobalHeapIndex::parse_filtered(bytes.as_slice(), 0, 8, |index| index == 2).unwrap();

    let beta = GlobalHeapObjectInfo {
        index: 2,
        data_address: 56,
        size: 4,
    };
    assert_eq!(
        collection.objects,
        vec![
            GlobalHeapObjectInfo {
                index: 1,
                data_address: 32,
                size: 5,
            },
            beta.clone(),
        ]
    );
    assert_eq!(filtered.objects, vec![beta.clone()]);
    assert_eq!(collection.object(2), Some(&beta));
    assert_eq!(&bytes[56..60], b"beta");
    assert_eq!(GLOBAL_HEAP_MAX_OBJECTS, usize::from(u16::MAX));
}

// 16 + 16 + 65,480 + 16 bytes fit the 65,535 of a 2-byte length, and 65,481 bytes of data pad
// to 65,488 and a collection of 65,536.
#[rstest]
#[case::fits(65_480, Ok(65_528))]
#[case::overflows(
    65_481,
    Err(GlobalHeapCollectionError::TooLarge {
        size: 65_536,
        length_width: LengthWidth::Two,
    })
)]
fn a_two_byte_length_collection_holds_at_most_its_field_maximum(
    #[case] object_len: usize,
    #[case] expected: Result<usize, GlobalHeapCollectionError>,
) {
    let object = vec![0u8; object_len];

    assert_eq!(
        hdf5_pure_format::encode_global_heap_collection(LengthWidth::Two, &[&object])
            .map(|bytes| bytes.len()),
        expected
    );
}

#[test]
fn the_encoder_rejects_more_than_global_heap_max_objects() {
    let objects = vec![b"".as_slice(); GLOBAL_HEAP_MAX_OBJECTS + 1];

    assert_eq!(
        hdf5_pure_format::encode_global_heap_collection(LengthWidth::Eight, &objects),
        Err(GlobalHeapCollectionError::TooManyObjects { count: 65_536 })
    );
}

#[rstest]
#[case::inline(
    DatatypeLocation::Inline,
    hdf5_pure_format::serialize_datatype(&i32_datatype()),
    attribute::Flags::NONE
)]
#[case::committed(
    DatatypeLocation::Committed(StoredAddress::new(800)),
    datatype::committed_reference(800),
    attribute::Flags::SHARED_DATATYPE
)]
fn an_attribute_message_round_trips_through_the_public_api(
    #[case] datatype_location: DatatypeLocation,
    #[case] datatype_field: Vec<u8>,
    #[case] flags: attribute::Flags,
) {
    let attribute = AttributeMessage {
        name: "scale".into(),
        datatype: i32_datatype(),
        dataspace: Dataspace {
            space_type: DataspaceType::Scalar,
            rank: 0,
            dimensions: Vec::new(),
            max_dimensions: None,
        },
        raw_data: 7i32.to_le_bytes().to_vec(),
        datatype_location,
    };
    let bytes = attribute.serialize_v3(OffsetWidth::Eight, 8).unwrap();

    assert_eq!(
        bytes,
        attribute::Attribute::new(
            "scale",
            &datatype_field,
            &dataspace::scalar(),
            &7i32.to_le_bytes(),
        )
        .flags(flags)
        .character_set(0)
        .build()
    );
    assert_eq!(AttributeMessage::parse_name(&bytes), Ok("scale".into()));
    assert_eq!(
        AttributeMessage::shares_a_field(&bytes),
        flags != attribute::Flags::NONE
    );
    assert_eq!(
        AttributeMessage::parse_resolving_at(&bytes, 8, &CommittedI32),
        Ok((attribute, bytes.len() - 4))
    );
}

/// A resolver that returns the `i32` datatype for every reference, as if it were committed at the
/// address the reference holds.
struct CommittedI32;

impl SharedResolver for CommittedI32 {
    fn resolve(&self, _reference: &[u8], _target: MessageType) -> Result<Vec<u8>, FormatError> {
        Ok(hdf5_pure_format::serialize_datatype(&i32_datatype()))
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        hdf5_pure_format::committed_address_in(reference, 8, 8)
    }
}

fn i32_datatype() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        layout: FixedPointLayout {
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        },
    }
}
