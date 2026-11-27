use alloc::vec;

use hdf5_pure_core::{CharacterSet, ReferenceType, StringPadding, VariableLengthType};
use rstest::rstest;
use test_util::attribute::{UnpaddedAttribute, UnpaddedVersion};
use test_util::{dataspace, datatype};

use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::shared_message::DatatypeLocation;

use crate::width::{LengthWidth, OffsetWidth};

use super::*;
use crate::dense_attribute_write::{dense_attrs_plan, test_support};

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn dense_preparation_retains_both_file_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    let widths = FormatWidths { offsets, lengths };
    let attribute = test_support::canonical_attribute("label", 3);
    let attrs = [attribute];
    let dense =
        PreparedDenseAttrs::with_widths(widths, &attrs, DenseAttrCreationOrder::Untracked, [])
            .unwrap();
    assert_eq!(dense.widths(), widths);
}

#[rstest]
#[case::ascii_small("label", 1)]
#[case::unicode_small("étiquette", 1)]
#[case::ascii_managed("label", 128)]
#[case::unicode_managed("étiquette", 128)]
#[case::ascii_huge("label", 70_000)]
#[case::unicode_huge("étiquette", 70_000)]
fn dense_storage_emits_the_version_three_attribute_body_once(#[case] name: &str, #[case] len: u32) {
    let attribute = test_support::canonical_attribute(name, len);
    let expected = UnpaddedAttribute::inline(
        UnpaddedVersion::Three,
        name,
        &datatype::fixed_string(len),
        &dataspace::VersionTwo::scalar().build(),
        &attribute.raw_data,
    )
    .build();
    let prepared =
        PreparedDenseAttrs::new(&[attribute], DenseAttrCreationOrder::Untracked, []).unwrap();
    let body = prepared.plan.attributes()[0].bytes();
    assert_eq!(body, &expected);
    let actual = prepared
        .finalize(
            |reference| panic!("unexpected reference {reference:?}"),
            |_| panic!("unexpected payload"),
        )
        .unwrap()
        .build(StoredAddress::new(0x8000))
        .unwrap();
    assert_eq!(
        actual
            .blob
            .windows(expected.len())
            .filter(|body| *body == expected)
            .count(),
        1
    );
}

#[rstest]
#[case::managed(8)]
#[case::huge(70_000)]
fn prepared_clones_retain_shared_geometry_through_finalization(#[case] len: u32) {
    let attr = test_support::canonical_attribute("label", len);
    let replacement = vec![b'b'; usize::try_from(len).unwrap()];
    let prepared = PreparedDenseAttrs::new(
        &[attr],
        DenseAttrCreationOrder::Untracked,
        [DenseAttrFixup::Payload { attribute: 0 }],
    )
    .unwrap();
    let cloned = prepared.clone();
    assert!(core::ptr::eq(
        prepared.plan.geometry(),
        cloned.plan.geometry()
    ));
    let ready = cloned
        .finalize(
            |reference| panic!("unexpected reference {reference:?}"),
            |_| Ok(replacement.as_slice()),
        )
        .unwrap();
    assert!(core::ptr::eq(prepared.plan.geometry(), ready.geometry()));
    let blob = ready.build(StoredAddress::new(0x8000)).unwrap();
    assert_eq!(u64::try_from(blob.blob.len()).unwrap(), prepared.blob_len());
}

#[rstest]
#[case::empty_variable_value(0)]
#[case::managed(2)]
#[case::huge(5_000)]
#[case::multiple_collections(4)]
fn finalization_retains_geometry_and_encodes_resolved_references(#[case] references: usize) {
    let mut attrs = vec![
        variable_attribute("vl", references),
        test_support::metadata_attribute("digest", 0),
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
        .attribute_info_for_heap(StoredAddress::new(0x8000))
        .unwrap();
    // Collections shared with other attributes hold the strings at any object index.
    let mut resolved = attrs[0].raw_data.clone();
    for (index, reference) in resolved.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let address = 0x1000 + (index / 3) as u64 * 0x1000;
        reference[4..12].copy_from_slice(&address.to_le_bytes());
        reference[12..].copy_from_slice(&(index as u32 % 3 + 5).to_le_bytes());
    }
    let mut observed = Vec::new();
    let ready = prepared
        .finalize(
            |attribute| {
                observed.push(attribute);
                Ok(match attribute {
                    0 => StoredAddress::new(0x3456),
                    _ => panic!("unexpected committed datatype of attribute {attribute}"),
                })
            },
            |attribute| {
                Ok(match attribute {
                    0 => resolved.as_slice(),
                    1 => payload.as_slice(),
                    _ => panic!("unexpected payload {attribute}"),
                })
            },
        )
        .unwrap();
    assert_eq!(observed, [0]);
    assert_eq!(geometry(&ready), retained);
    assert_eq!(ready.blob_len(), reserved);
    assert_eq!(
        ready
            .attribute_info_for_heap(StoredAddress::new(0x8000))
            .unwrap(),
        info
    );
    attrs[0].datatype_location = DatatypeLocation::Committed(StoredAddress::new(0x3456));
    attrs[0].raw_data = resolved;
    attrs[1].raw_data.copy_from_slice(&payload);
    assert_eq!(
        ready
            .attributes()
            .iter()
            .map(|a| a.bytes().to_vec())
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
    assert_eq!(
        ready.geometry().huge_count(),
        usize::from(references >= 5_000)
    );
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
    let attrs = [test_support::metadata_attribute("a", 0)];
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
            &[test_support::metadata_attribute("a", 0)],
            DenseAttrCreationOrder::Untracked,
            fixups
        )
        .err()
        .unwrap(),
        FormatError::Internal("dense attribute fixups overlap".into())
    );
}

#[test]
fn preparation_rejects_partial_global_heap_references() {
    let mut attr = variable_attribute("a", 1);
    attr.raw_data.pop().unwrap();
    assert_eq!(
        PreparedDenseAttrs::new(
            &[attr],
            DenseAttrCreationOrder::Untracked,
            [DenseAttrFixup::GlobalHeap { attribute: 0 }]
        )
        .err()
        .unwrap(),
        FormatError::Internal(
            "a global heap fixup requires whole variable-length attribute references".into()
        )
    );
}

#[test]
fn finalization_rejects_global_heap_length_changes() {
    let attr = variable_attribute("a", 1);
    let mut longer = attr.raw_data.clone();
    longer.extend_from_slice(&attr.raw_data);
    let prepared = PreparedDenseAttrs::new(
        &[attr],
        DenseAttrCreationOrder::Untracked,
        [DenseAttrFixup::GlobalHeap { attribute: 0 }],
    )
    .unwrap();
    assert_eq!(
        prepared
            .finalize(
                |reference| panic!("unexpected reference {reference:?}"),
                |_| Ok(longer.as_slice())
            )
            .err()
            .unwrap(),
        FormatError::Internal("a dense attribute fixup changes its prepared length".into())
    );
}

#[rstest]
#[case::short(7)]
#[case::long(9)]
fn finalization_rejects_payload_length_changes(#[case] len: usize) {
    let prepared = PreparedDenseAttrs::new(
        &[test_support::metadata_attribute("a", 0)],
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
        FormatError::Internal("a dense attribute fixup changes its prepared length".into())
    );
}

#[rstest]
#[case::committed(DenseAttrFixup::CommittedDatatype { attribute: 0 })]
#[case::global_heap(DenseAttrFixup::GlobalHeap { attribute: 0 })]
fn finalization_rejects_undefined_reference_addresses(#[case] fixup: DenseAttrFixup) {
    let mut attr = variable_attribute("a", 1);
    attr.datatype_location = DatatypeLocation::Committed(StoredAddress::ZERO);
    let mut undefined = attr.raw_data.clone();
    undefined[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
    let prepared =
        PreparedDenseAttrs::new(&[attr], DenseAttrCreationOrder::Untracked, [fixup]).unwrap();
    assert_eq!(
        prepared
            .finalize(
                |_| Ok(StoredAddress::undefined(OffsetWidth::Eight)),
                |_| Ok(undefined.as_slice())
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
    let mut attr = test_support::metadata_attribute("a", 0);
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

#[derive(Debug, Eq, PartialEq)]
struct RetainedGeometry {
    huge_ids: Vec<Option<u64>>,
    name_order: Vec<(u32, u32)>,
    creation_order: Vec<u32>,
    sizes_and_offsets: [u64; 12],
}

fn geometry(plan: &DenseAttrPlan) -> RetainedGeometry {
    let g = plan.geometry();
    RetainedGeometry {
        huge_ids: g.huge_id_of().to_vec(),
        name_order: g.indexes().order().to_vec(),
        creation_order: g.indexes().corder_order().to_vec(),
        sizes_and_offsets: [
            g.huge_total(),
            g.managed_plan().region_size(),
            g.offsets().managed_off,
            g.offsets().btree_off,
            g.offsets().name_nodes_off,
            g.offsets().corder_bthd_off,
            g.offsets().corder_nodes_off,
            g.offsets().huge_bthd_off,
            g.offsets().huge_nodes_off,
            g.offsets().huge_data_off,
            g.offsets().total_len,
            g.managed_plan().managed_space(),
        ],
    }
}

fn variable_attribute(name: &str, count: usize) -> AttributeMessage {
    let mut attr = test_support::metadata_attribute(name, 0);
    attr.datatype = Datatype::VariableLength(VariableLengthType::string(
        StringPadding::NullTerminate,
        CharacterSet::Utf8,
    ));
    attr.dataspace = Dataspace::simple(vec![count as u64], None).unwrap();
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
    let mut attr = test_support::metadata_attribute("a", 0);
    attr.datatype = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    attr
}
