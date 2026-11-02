use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{MessageType, StoredAddress};
use hdf5_pure_format::__private::{
    AccessMode, AttributePhaseChange, FormatWidths, MessageFlags, MessageRecordLayout,
    ObjectHeader, ObjectHeaderAllocationGeometry, ObjectHeaderPrefix, ObjectTimes,
};
use rstest::rstest;
use test_util::object_header::v2::HeaderFlags;

#[rstest]
#[case::plain(MessageRecordLayout::PLAIN)]
#[case::tracked(MessageRecordLayout::from_header_flags(
    (HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER).0
))]
fn chosen_page_chunks_preserve_prefix_properties(#[case] layout: MessageRecordLayout) {
    let prefix = ObjectHeaderPrefix {
        times: Some(ObjectTimes {
            access: 1,
            modification: 2,
            change: 3,
            birth: 4,
        }),
        attribute_phase_change: Some(AttributePhaseChange {
            max_compact: 8,
            min_dense: 6,
        }),
        layout,
    };
    let records = layout.record_with_creation_index(
        MessageType::DATATYPE,
        MessageFlags::CONSTANT,
        17,
        &[5, 6],
    );
    let geometry = ObjectHeaderAllocationGeometry::new(
        prefix,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &records,
    )
    .unwrap();
    let plan = geometry
        .into_chunked_allocation(StoredAddress::new(128), 12_288, &[3_968, 4_096, 4_096, 128])
        .unwrap();
    assert_eq!(
        plan.chunks().collect::<Vec<_>>(),
        [
            (StoredAddress::new(128), 3_968),
            (StoredAddress::new(4_096), 4_096),
            (StoredAddress::new(8_192), 4_096),
            (StoredAddress::new(12_288), 128),
        ]
    );
    let bytes = plan.encode();
    assert_eq!(ObjectHeaderPrefix::parse(&bytes).unwrap().prefix, prefix);
    let mut image = vec![0; 128];
    image.extend(bytes);
    let parsed = ObjectHeader::parse(&image, AccessMode::ReadOnly, 128, 8, 8).unwrap();
    assert_eq!(parsed.messages.len(), 1);
    assert_eq!(parsed.messages[0].data, [5, 6]);
    assert_eq!(parsed.messages[0].flags, MessageFlags::CONSTANT);
}
