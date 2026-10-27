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

/// Both indexes used to carry a count limit, and it came from the reference C
/// library deriving a record-count width from a leaf's *capacity*: a node
/// grown large enough to hold every record in one leaf pushed that width past
/// the 2 bytes `H5B2__hdr_init` asserts on. A fixed node size takes the
/// derivation out of the record count's hands entirely — this is the property
/// that replaced the limits, so it is the one worth pinning.
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

fn metadata_attribute(name: &str, value: i64) -> AttributeMessage {
    AttributeMessage {
        name: name.into(),
        datatype: Datatype::FixedPoint(
            hdf5_pure_core::FixedPointType::new(
                8,
                hdf5_pure_core::DatatypeByteOrder::LittleEndian,
                true,
                0,
                64,
            )
            .unwrap(),
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
