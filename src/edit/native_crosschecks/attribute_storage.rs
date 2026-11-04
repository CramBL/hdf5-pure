use std::collections::BTreeMap;

use test_util_hdf5::creation_order;
use test_util_hdf5::file;

use crate::access_mode::AccessMode;
use crate::attribute;
use crate::attribute::AttributeMessage;
use crate::edit::WriteEngine;
use crate::edit::attributes;
use crate::edit::attributes::AttrOp;
use crate::edit::object_header::OhRegion;
use crate::edit::test_util;
use crate::message_type::MessageType;
use crate::object_path::ObjectPath;
use crate::source::PhysicalSource;

#[test]
fn dense_reversion_preserves_encoded_records_and_prefix() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("records.h5");
    let file = file::libhdf5_create_v18(&path);
    let group = file
        .create_group_builder()
        .with_gcpl(|p| {
            p.attr_phase_change(16, 12)
                .attr_creation_order(creation_order::Indexing::Yes.attr_order())
        })
        .create("g")
        .unwrap();
    group
        .new_dataset::<i32>()
        .shape([2])
        .create("data")
        .unwrap()
        .write(&[7, 13])
        .unwrap();
    for index in 0..17 {
        group
            .new_attr::<i64>()
            .shape(())
            .create(format!("a{index:02}").as_str())
            .unwrap()
            .write_scalar(&i64::from(index))
            .unwrap();
    }
    drop(group);
    file.close().unwrap();
    let mut engine = test_util::open_engine(&path);
    let address = crate::group::v2::resolve_path_any_from_source(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        engine.superblock(),
        engine.widths(),
        &ObjectPath::parse("g"),
    )
    .unwrap();
    let region = WriteEngine::gather_oh_messages(
        engine.image(),
        address,
        engine.context.addresses(),
        engine.context.widths(),
    )
    .unwrap();
    assert!(region.props().times.is_some());
    let (info_start, info_body, info) = attributes::find_attribute_info(&region).unwrap().unwrap();
    let mut bodies = BTreeMap::new();
    attribute::visit_dense_attribute_messages_from_source(
        &PhysicalSource::new(&engine.opened_source()).stored(),
        AccessMode::ReadWrite,
        &info,
        info.fractal_heap_address.unwrap(),
        crate::file_writer::WIDTHS,
        None,
        |_| true,
        |body, index| {
            bodies.insert(AttributeMessage::parse_name(body)?, (body.to_vec(), index));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(bodies.len(), 17);
    assert!(
        bodies
            .values()
            .all(|(body, _)| body[0] == ATTRIBUTE_MESSAGE_VERSION_THREE)
    );
    let operations: Vec<_> = (0..6)
        .map(|index| AttrOp::Remove {
            name: format!("a{index:02}"),
        })
        .collect();
    let operations: Vec<_> = operations.iter().collect();
    let mut prepared =
        attributes::plan_attr_ops(&engine.opened_source(), &region, &operations).unwrap();
    assert!(prepared.replaces_dense_storage);
    assert_eq!(prepared.region.props(), region.props());
    engine
        .place_edited_attrs(&mut prepared.region, prepared.attrs)
        .unwrap();
    assert_eq!(
        unrelated_records(&prepared.region),
        unrelated_records(&region)
    );
    let (start, body, compact) = attributes::find_attribute_info(&prepared.region)
        .unwrap()
        .unwrap();
    assert_eq!(
        &prepared.region[start..body.start],
        &region[info_start..info_body.start]
    );
    assert_eq!(compact.max_creation_index, Some(17));
    assert!(compact.indexes_creation_order);
    assert_eq!(compact.fractal_heap_address, None);
    assert_eq!(compact.btree_name_index_address, None);
    assert_eq!(compact.btree_creation_order_address, None);
    let mut position = 0;
    let mut count = 0;
    while let Some((kind, body, end)) = prepared.region.next_message(position).unwrap() {
        if kind == MessageType::ATTRIBUTE {
            let name = AttributeMessage::parse_name(&prepared.region[body..end]).unwrap();
            let (expected, index) = &bodies[&name];
            assert_eq!(&prepared.region[body..end], expected);
            assert_eq!(prepared.region.creation_index(position), *index);
            count += 1;
        }
        position = end;
    }
    assert_eq!(count, 11);
}

fn unrelated_records(region: &OhRegion) -> Vec<Vec<u8>> {
    let mut records = Vec::new();
    let mut position = 0;
    while let Some((kind, _, end)) = region.next_message(position).unwrap() {
        if kind != MessageType::ATTRIBUTE && kind != MessageType::ATTRIBUTE_INFO {
            records.push(region[position..end].to_vec());
        }
        position = end;
    }
    records
}

// "The Attribute Message" of the format specification, version 4.0, defines version 3.
const ATTRIBUTE_MESSAGE_VERSION_THREE: u8 = 3;
