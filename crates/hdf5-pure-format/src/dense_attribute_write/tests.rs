use hdf5_pure_core::{
    CharacterSet, DatatypeByteOrder, FixedPointType, ReferenceType, StringPadding,
    VariableLengthType,
};
use rstest::rstest;

use crate::dataspace::Dataspace;
use crate::dataspace::DataspaceType;
use crate::datatype::Datatype;
use crate::shared_message::DatatypeLocation;

use super::*;

#[rstest]
#[case::missing_index(vec![1], 9, false,
    FormatError::Internal("a tracked attribute set needs one creation index per attribute".into()))]
#[case::extra_index(vec![1, 2, 3], 9, true,
    FormatError::Internal("a tracked attribute set needs one creation index per attribute".into()))]
#[case::duplicate_index(vec![3, 3], 9, true,
    FormatError::DuplicateAttributeCreationIndex(3))]
#[case::duplicate_unindexed(vec![3, 3], 9, false,
    FormatError::DuplicateAttributeCreationIndex(3))]
#[case::counter_equals_index(vec![1, 3], 3, true,
    FormatError::InvalidAttributeCreationCounter { index: 3, next: 3 })]
#[case::counter_below_index(vec![1, 3], 2, false,
    FormatError::InvalidAttributeCreationCounter { index: 3, next: 2 })]
fn dense_attribute_preparation_rejects_incoherent_creation_metadata(
    #[case] indices: Vec<u16>,
    #[case] max: u16,
    #[case] indexed: bool,
    #[case] expected: FormatError,
) {
    let attrs = [metadata_attribute("a", 1), metadata_attribute("b", 2)];
    assert_eq!(
        EncodedDenseAttrs::new(
            &attrs,
            DenseAttrCreationOrder::Tracked {
                indices,
                max,
                indexed
            }
        )
        .unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::tracked_and_indexed(true)]
#[case::tracked_without_index(false)]
fn dense_attribute_preparation_preserves_index_gaps_and_counter(#[case] indexed: bool) {
    let attrs = [metadata_attribute("a", 1), metadata_attribute("b", 2)];
    let prepared = EncodedDenseAttrs::new(
        &attrs,
        DenseAttrCreationOrder::Tracked {
            indices: vec![7, 2],
            max: u16::MAX,
            indexed,
        },
    )
    .unwrap();
    assert_eq!(prepared.next_index, Some(u16::MAX));
    assert_eq!(prepared.indexed, indexed);
    assert_eq!(
        prepared
            .attributes
            .iter()
            .map(|a| a.creation_index)
            .collect::<Vec<_>>(),
        vec![7, 2]
    );
    assert_eq!(
        prepared
            .attributes
            .iter()
            .map(|a| a.bytes.clone())
            .collect::<Vec<_>>(),
        attrs
            .iter()
            .map(|a| a
                .serialize_v3(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_SIZE)
                .unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        prepared.corder_order,
        if indexed { vec![1, 0] } else { vec![] }
    );
}

#[test]
fn dense_attribute_preparation_preserves_untracked_positions() {
    let attrs = [metadata_attribute("b", 1), metadata_attribute("a", 2)];
    let prepared = EncodedDenseAttrs::new(&attrs, DenseAttrCreationOrder::Untracked).unwrap();
    assert_eq!(prepared.next_index, None);
    assert!(!prepared.indexed);
    assert!(prepared.corder_order.is_empty());
    assert_eq!(
        prepared
            .attributes
            .iter()
            .map(|a| a.creation_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn dense_attribute_preparation_rejects_duplicate_names() {
    let attrs = [
        metadata_attribute("duplicate", 1),
        metadata_attribute("duplicate", 2),
    ];
    assert_eq!(
        EncodedDenseAttrs::new(&attrs, DenseAttrCreationOrder::Untracked).unwrap_err(),
        FormatError::DuplicateAttributeName("duplicate".into())
    );
}

#[rstest]
#[case::untracked(DenseAttrCreationOrder::Untracked, None, false)]
#[case::tracked(DenseAttrCreationOrder::Tracked { indices: vec![], max: 17, indexed: true }, Some(17), true)]
fn dense_attribute_preparation_preserves_empty_set_metadata(
    #[case] creation: DenseAttrCreationOrder,
    #[case] next: Option<u16>,
    #[case] indexed: bool,
) {
    let prepared = EncodedDenseAttrs::new(&[], creation).unwrap();
    assert!(prepared.attributes.is_empty());
    assert!(prepared.order.is_empty());
    assert!(prepared.corder_order.is_empty());
    assert_eq!(prepared.next_index, next);
    assert_eq!(prepared.indexed, indexed);
}

#[test]
fn a_fixed_node_size_keeps_the_derived_count_width_at_one_byte() {
    for record_size in [DENSE_ATTR_BTREE_RECORD, DENSE_ATTR_HUGE_BTREE_RECORD] {
        let (info, depth) = format_boundary::BTreeV2NodeInfo::for_record_count(
            BTREE_V2_NODE_SIZE,
            record_size,
            format_boundary::OffsetWidth::Eight,
            0,
        )
        .expect("the emitted geometry is plannable");
        assert_eq!(depth, 0);
        let capacity = info.max_nrec(0);
        assert!(
            capacity >= 20,
            "a {record_size}-byte record should leave room for 20 per node, got {capacity}"
        );
        assert_eq!(
            encoded_size_width(capacity),
            1,
            "a {record_size}-byte record's leaf capacity must stay inside one byte"
        );
    }
}

#[rstest]
#[case::undefined(u64::MAX, true)]
#[case::overflow(u64::MAX - 1, false)]
fn dense_attribute_plan_rejects_invalid_extent(#[case] base: u64, #[case] undefined: bool) {
    let attrs = [metadata_attribute("a", 1)];
    let plan = dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked).unwrap();
    let expected = if undefined {
        FormatError::DefinedAddressIsUndefined {
            address: base,
            offset_size: DENSE_ATTR_OFFSET_SIZE,
        }
    } else {
        FormatError::OffsetOverflow {
            offset: base,
            length: plan.blob_len(),
        }
    };
    assert_eq!(
        plan.attr_info_message(StoredAddress::new(base))
            .unwrap_err(),
        expected
    );
    assert_eq!(
        plan.build(StoredAddress::new(base)).err().unwrap(),
        expected
    );
}

#[rstest]
#[case::empty(vec![])]
#[case::managed(vec![metadata_attribute("a", 1)])]
#[case::huge(vec![canonical_attribute("huge", 70_000)])]
fn dense_attribute_plan_fills_a_valid_extent_at_the_address_limit(
    #[case] attrs: Vec<AttributeMessage>,
) {
    let plan = dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked).unwrap();
    let base = StoredAddress::new(u64::MAX - plan.blob_len());
    let built = plan.build(base).unwrap();
    assert_eq!(built.blob.len() as u64, plan.blob_len());
    assert_eq!(
        built.attr_info_message,
        plan.attr_info_message(base).unwrap()
    );
    let info = AttributeInfoMessage::parse(&built.attr_info_message, 8).unwrap();
    assert_eq!(info.fractal_heap_address, Some(base));
    assert!(info.btree_name_index_address.unwrap().get() < u64::MAX);
}

#[rstest]
#[case::empty_variable_value(0)]
#[case::managed(2)]
#[case::huge(5_000)]
#[case::multiple_collections(format_boundary::GLOBAL_HEAP_MAX_OBJECTS + 1)]
fn finalization_retains_geometry_and_encodes_resolved_references(#[case] references: usize) {
    let mut attrs = vec![
        variable_attribute("vl", references),
        metadata_attribute("digest", 0),
    ];
    attrs[0].datatype_location = DatatypeLocation::Committed(StoredAddress::ZERO);
    let payload = 123i64.to_le_bytes();
    let prepared = PreparedDenseAttrs::new(
        &attrs,
        DenseAttrCreationOrder::Tracked {
            indices: vec![7, 2],
            max: 19,
            indexed: true,
        },
        [
            DenseAttrFixup::GlobalHeap { attribute: 0 },
            DenseAttrFixup::CommittedDatatype { attribute: 0 },
            DenseAttrFixup::Payload { attribute: 1 },
        ],
    )
    .unwrap();
    let retained = geometry(&prepared.plan);
    let reserved = prepared.blob_len();
    let info = prepared
        .attr_info_message(StoredAddress::new(0x8000))
        .unwrap();
    let mut observed = Vec::new();
    let ready = prepared
        .finalize(
            |reference| {
                observed.push(reference);
                Ok(match reference {
                    DenseAttrReference::CommittedDatatype { attribute: 0 } => {
                        StoredAddress::new(0x3456)
                    }
                    DenseAttrReference::GlobalHeap {
                        attribute: 0,
                        collection,
                    } => StoredAddress::new(0x1000 + collection as u64 * 0x1000),
                    _ => panic!("unexpected reference {reference:?}"),
                })
            },
            |attribute| {
                assert_eq!(attribute, 1);
                Ok(payload.as_slice())
            },
        )
        .unwrap();
    let collections = references.div_ceil(format_boundary::GLOBAL_HEAP_MAX_OBJECTS);
    let mut expected_references = (0..collections)
        .map(|collection| DenseAttrReference::GlobalHeap {
            attribute: 0,
            collection,
        })
        .collect::<Vec<_>>();
    expected_references.push(DenseAttrReference::CommittedDatatype { attribute: 0 });
    assert_eq!(observed, expected_references);
    assert_eq!(geometry(&ready), retained);
    assert_eq!(ready.blob_len(), reserved);
    assert_eq!(
        ready.attr_info_message(StoredAddress::new(0x8000)).unwrap(),
        info
    );
    attrs[0].datatype_location = DatatypeLocation::Committed(StoredAddress::new(0x3456));
    for (index, reference) in attrs[0]
        .raw_data
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .enumerate()
    {
        let address = 0x1000 + (index / format_boundary::GLOBAL_HEAP_MAX_OBJECTS) as u64 * 0x1000;
        reference[4..12].copy_from_slice(&address.to_le_bytes());
    }
    attrs[1].raw_data.copy_from_slice(&payload);
    assert_eq!(
        ready
            .attributes
            .iter()
            .map(|a| a.bytes.clone())
            .collect::<Vec<_>>(),
        attrs
            .iter()
            .map(|a| a.serialize_v3(OffsetWidth::Eight, 8).unwrap())
            .collect::<Vec<_>>()
    );
    let direct = dense_attrs_plan(
        &attrs,
        DenseAttrCreationOrder::Tracked {
            indices: vec![7, 2],
            max: 19,
            indexed: true,
        },
    )
    .unwrap();
    let built = ready.build(StoredAddress::new(0x8000)).unwrap();
    assert_eq!(built.blob.len() as u64, reserved);
    assert_eq!(
        built.blob,
        direct.build(StoredAddress::new(0x8000)).unwrap().blob
    );
    assert_eq!(ready.geometry.huge_count, usize::from(references >= 5_000));
}

#[rstest]
#[case::absent(DenseAttrFixup::GlobalHeap { attribute: 1 },
    "a dense attribute fixup refers to an absent attribute")]
#[case::inline_datatype(DenseAttrFixup::CommittedDatatype { attribute: 0 },
    "a committed datatype fixup requires a committed attribute datatype")]
#[case::fixed_data(DenseAttrFixup::GlobalHeap { attribute: 0 },
    "a global heap fixup requires whole variable-length attribute references")]
fn preparation_rejects_fixups_for_other_attribute_shapes(
    #[case] fixup: DenseAttrFixup,
    #[case] expected: &str,
) {
    let attrs = [metadata_attribute("a", 0)];
    assert_eq!(
        PreparedDenseAttrs::new(&attrs, DenseAttrCreationOrder::Untracked, [fixup])
            .err()
            .unwrap(),
        FormatError::Internal(expected.into())
    );
}

#[rstest]
#[case::variable_length(variable_attribute("a", 1))]
#[case::object_reference(reference_attribute())]
fn preparation_rejects_payload_fixups_with_file_addresses(#[case] attr: AttributeMessage) {
    assert_eq!(
        PreparedDenseAttrs::new(
            &[attr],
            DenseAttrCreationOrder::Untracked,
            [DenseAttrFixup::Payload { attribute: 0 }]
        )
        .err()
        .unwrap(),
        FormatError::Internal(
            "a dense attribute payload fixup cannot replace file references".into()
        )
    );
}

#[rstest]
#[case::duplicate_payload([DenseAttrFixup::Payload { attribute: 0 }, DenseAttrFixup::Payload { attribute: 0 }])]
#[case::overlapping_payload([DenseAttrFixup::Payload { attribute: 0 }, DenseAttrFixup::GlobalHeap { attribute: 0 }])]
fn preparation_rejects_overlapping_fixups(#[case] fixups: [DenseAttrFixup; 2]) {
    assert_eq!(
        PreparedDenseAttrs::new(
            &[metadata_attribute("a", 0)],
            DenseAttrCreationOrder::Untracked,
            fixups
        )
        .err()
        .unwrap(),
        FormatError::Internal("dense attribute fixups overlap".into())
    );
}

#[rstest]
#[case::partial_reference(true)]
#[case::wrong_object_index(false)]
fn preparation_rejects_incoherent_global_heap_references(#[case] partial: bool) {
    let mut attr = variable_attribute("a", 1);
    let expected = if partial {
        attr.raw_data.pop().unwrap();
        "a global heap fixup requires whole variable-length attribute references"
    } else {
        attr.raw_data[12..16].copy_from_slice(&0u32.to_le_bytes());
        "a global heap fixup requires consecutive collection object indexes"
    };
    assert_eq!(
        PreparedDenseAttrs::new(
            &[attr],
            DenseAttrCreationOrder::Untracked,
            [DenseAttrFixup::GlobalHeap { attribute: 0 }]
        )
        .err()
        .unwrap(),
        FormatError::Internal(expected.into())
    );
}

#[rstest]
#[case::short(7)]
#[case::long(9)]
fn finalization_rejects_payload_length_changes(#[case] len: usize) {
    let prepared = PreparedDenseAttrs::new(
        &[metadata_attribute("a", 0)],
        DenseAttrCreationOrder::Untracked,
        [DenseAttrFixup::Payload { attribute: 0 }],
    )
    .unwrap();
    let data = vec![0; len];
    assert_eq!(
        prepared
            .finalize(
                |reference| panic!("unexpected reference {reference:?}"),
                |_| Ok(&data)
            )
            .err()
            .unwrap(),
        FormatError::Internal("a dense attribute payload fixup changes its prepared length".into())
    );
}

#[rstest]
#[case::committed(DenseAttrFixup::CommittedDatatype { attribute: 0 })]
#[case::global_heap(DenseAttrFixup::GlobalHeap { attribute: 0 })]
fn finalization_rejects_undefined_reference_addresses(#[case] fixup: DenseAttrFixup) {
    let mut attr = variable_attribute("a", 1);
    attr.datatype_location = DatatypeLocation::Committed(StoredAddress::ZERO);
    let prepared =
        PreparedDenseAttrs::new(&[attr], DenseAttrCreationOrder::Untracked, [fixup]).unwrap();
    assert_eq!(
        prepared
            .finalize(
                |_| Ok(StoredAddress::undefined(OffsetWidth::Eight)),
                |attribute| panic!("unexpected payload {attribute}")
            )
            .err()
            .unwrap(),
        FormatError::DefinedAddressIsUndefined {
            address: u64::MAX,
            offset_size: 8
        }
    );
}

#[rstest]
#[case::reference(DenseAttrFixup::CommittedDatatype { attribute: 0 }, FormatError::UnknownCommittedDatatype("missing".into()))]
#[case::payload(DenseAttrFixup::Payload { attribute: 0 }, FormatError::Internal("pending digest".into()))]
fn finalization_propagates_resolution_errors(
    #[case] fixup: DenseAttrFixup,
    #[case] expected: FormatError,
) {
    let mut attr = metadata_attribute("a", 0);
    attr.datatype_location = DatatypeLocation::Committed(StoredAddress::ZERO);
    let prepared =
        PreparedDenseAttrs::new(&[attr], DenseAttrCreationOrder::Untracked, [fixup]).unwrap();
    assert_eq!(
        prepared
            .finalize(|_| Err(expected.clone()), |_| Err(expected.clone()))
            .err()
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::managed(
    CanonicalShape::Managed,
    11_054,
    0xc3e7_19d6_79bd_1897,
    0x9c7d_22d3_63fa_dd12
)]
#[case::tracked(
    CanonicalShape::Tracked,
    11_604,
    0xf339_b08a_cd6a_9d44,
    0x6660_3de0_96e5_effe
)]
#[case::mixed(
    CanonicalShape::Mixed,
    73_374,
    0x2dd0_53a6_3871_24e4,
    0xf5b8_cc1e_4fed_c90d
)]
// Expected lengths and hashes were produced at revision 534712d2 (see the test data notice).
fn prepared_dense_output_matches_baseline_bytes(
    #[case] shape: CanonicalShape,
    #[case] len: usize,
    #[case] blob_hash: u64,
    #[case] info_hash: u64,
) {
    let attrs = match shape {
        CanonicalShape::Managed | CanonicalShape::Tracked => (0..10)
            .map(|i| canonical_attribute(&format!("a{i}"), 512))
            .collect::<Vec<_>>(),
        CanonicalShape::Mixed => vec![
            canonical_attribute("small", 512),
            canonical_attribute("huge", 70_000),
            canonical_attribute("after", 512),
        ],
    };
    let creation = match shape {
        CanonicalShape::Tracked => DenseAttrCreationOrder::Tracked {
            indices: vec![19, 4, 8, 1, 10, 3, 9, 6, 12, 7],
            max: 30,
            indexed: true,
        },
        CanonicalShape::Managed | CanonicalShape::Mixed => DenseAttrCreationOrder::Untracked,
    };
    let prepared = PreparedDenseAttrs::new(&attrs, creation, []).unwrap();
    let ready = prepared
        .finalize(
            |reference| panic!("unexpected reference {reference:?}"),
            |attribute| panic!("unexpected payload {attribute}"),
        )
        .unwrap();
    let built = ready.build(StoredAddress::new(0x8000)).unwrap();
    assert_eq!(
        ready.geometry.huge_count,
        usize::from(matches!(shape, CanonicalShape::Mixed))
    );
    assert_eq!(built.blob.len(), len);
    assert_eq!(canonical_hash(&built.blob), blob_hash);
    assert_eq!(canonical_hash(&built.attr_info_message), info_hash);
}

#[derive(Clone, Copy, Debug)]
enum CanonicalShape {
    Managed,
    Mixed,
    Tracked,
}

fn canonical_attribute(name: &str, len: u32) -> AttributeMessage {
    AttributeMessage {
        name: name.into(),
        datatype: Datatype::String {
            size: len,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        },
        dataspace: Dataspace {
            space_type: DataspaceType::Scalar,
            rank: 0,
            dimensions: vec![],
            max_dimensions: None,
        },
        raw_data: vec![b'y'; len as usize],
        datatype_location: DatatypeLocation::Inline,
    }
}

fn canonical_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[derive(Debug, Eq, PartialEq)]
struct RetainedGeometry {
    huge_ids: Vec<Option<u64>>,
    name_order: Vec<(u32, u32)>,
    creation_order: Vec<u32>,
    sizes_and_offsets: [u64; 12],
}

fn geometry(plan: &DenseAttrPlan) -> RetainedGeometry {
    let g = &plan.geometry;
    RetainedGeometry {
        huge_ids: g.huge_id_of.clone(),
        name_order: g.order.clone(),
        creation_order: g.corder_order.clone(),
        sizes_and_offsets: [
            g.huge_total,
            g.managed_plan.region_size(),
            g.managed_off,
            g.btree_off,
            g.name_nodes_off,
            g.corder_bthd_off,
            g.corder_nodes_off,
            g.huge_bthd_off,
            g.huge_nodes_off,
            g.huge_data_off,
            g.total_len,
            g.managed_plan.managed_space(),
        ],
    }
}

fn variable_attribute(name: &str, count: usize) -> AttributeMessage {
    let mut attr = metadata_attribute(name, 0);
    attr.datatype = Datatype::VariableLength(VariableLengthType::string(
        StringPadding::NullTerminate,
        CharacterSet::Utf8,
    ));
    attr.dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![count as u64],
        max_dimensions: None,
    };
    attr.raw_data = (0..count)
        .flat_map(|index| {
            let mut reference = Vec::new();
            reference.extend_from_slice(&1u32.to_le_bytes());
            reference.extend_from_slice(&0u64.to_le_bytes());
            reference.extend_from_slice(
                &((index % format_boundary::GLOBAL_HEAP_MAX_OBJECTS + 1) as u32).to_le_bytes(),
            );
            reference
        })
        .collect();
    attr
}

fn reference_attribute() -> AttributeMessage {
    let mut attr = metadata_attribute("a", 0);
    attr.datatype = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    attr
}

fn metadata_attribute(name: &str, value: i64) -> AttributeMessage {
    AttributeMessage {
        name: name.into(),
        datatype: Datatype::FixedPoint(
            FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap(),
        ),
        dataspace: Dataspace {
            space_type: DataspaceType::Scalar,
            rank: 0,
            dimensions: Vec::new(),
            max_dimensions: None,
        },
        raw_data: value.to_le_bytes().to_vec(),
        datatype_location: DatatypeLocation::Inline,
    }
}
