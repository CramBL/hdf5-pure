use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_format::Dataspace;
use hdf5_pure_format::DataspaceType;
use hdf5_pure_format::Datatype;
use hdf5_pure_format::DatatypeByteOrder;
use hdf5_pure_format::FillValueError;
use hdf5_pure_format::FilterDescription;
use hdf5_pure_format::FilterPipeline;
use hdf5_pure_format::FilterPipelineError;
use hdf5_pure_format::FixedPointLayout;
use hdf5_pure_format::MaxExtent;
use hdf5_pure_format::V3_FLAGS_DEFAULT;

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
    use hdf5_pure_format::SharedResolver;

    let address = hdf5_pure_format::StoredAddress::new(800);
    let committed =
        hdf5_pure_format::encode_committed_ref(address, hdf5_pure_format::OffsetWidth::Eight);
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
        hdf5_pure_format::Unresolvable.resolve(&heap, hdf5_pure_format::MessageType::DATASPACE),
        Err(hdf5_pure_format::FormatError::UnresolvedSharedMessage(
            hdf5_pure_format::MessageType::DATASPACE.to_u16()
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
