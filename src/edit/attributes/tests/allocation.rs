use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
use test_util::allocation;

use super::super::AttrOp;
use super::super::EditedAttrs;
use crate::attribute::AttributeMessage;
use crate::edit::object_header::OhRegion;
use crate::edit::test_util as edit_test_util;
use crate::message_type::MessageType;
use crate::type_builders::AttrValue;
use crate::writer::FileBuilder;

#[rstest]
#[case::one("compact_attribute_large_header_one", 1)]
#[case::many("compact_attribute_large_header_many", 128)]
fn compact_batches_copy_unrelated_records_a_bounded_number_of_times(
    #[case] region_name: &str,
    #[case] operations: usize,
) {
    let path = test_util::temp::temp_path("compact-attribute-batch-allocation.h5");
    FileBuilder::new().write(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    for _ in 0..16 {
        region.push(MessageType::from_u16(0xff), &vec![0x5a; 48_000]);
    }
    let unrelated = region.len();
    region.push(
        MessageType::ATTRIBUTE,
        &super::super::encode_attr_body("changed", &AttrValue::I64(-1)).unwrap(),
    );
    let ops: Vec<_> = (0..operations)
        .map(|index| AttrOp::Set {
            name: "changed".into(),
            value: AttrValue::I64(i64::try_from(index).unwrap()),
        })
        .collect();
    let ops: Vec<_> = ops.iter().collect();
    let (result, measured) = allocation::measure(region_name, || {
        super::super::plan_attr_ops(&engine.opened_source(), &region, &ops).unwrap()
    });
    eprintln!("{region_name}: {measured}");
    assert!(
        measured.bytes >= u64::try_from(unrelated).unwrap(),
        "{measured}"
    );
    assert!(
        measured.bytes < u64::try_from(4 * unrelated + operations * 4_096).unwrap(),
        "{measured}"
    );
    assert_eq!(&result.region[..unrelated], &region[..unrelated]);
    let EditedAttrs::RebuildCompact(compact) = result.attrs else {
        panic!("expected compact storage");
    };
    assert!(compact.fixups.is_empty());
    let (kind, body, end) = region
        .layout()
        .next_message(&compact.records, 0)
        .unwrap()
        .map(|record| {
            (
                record.msg_type,
                record.body_range.start,
                record.body_range.end,
            )
        })
        .unwrap();
    assert_eq!(kind, MessageType::ATTRIBUTE);
    assert_eq!(end, compact.records.len());
    let attribute = AttributeMessage::parse(&compact.records[body..end], 8).unwrap();
    assert_eq!(attribute.name, "changed");
    assert_eq!(
        attribute.raw_data,
        i64::try_from(operations - 1).unwrap().to_le_bytes()
    );
}

#[rstest]
#[case::medium("attribute_batch_dense_256", 256)]
#[case::large("attribute_batch_dense_512", 512)]
fn compact_to_dense_batches_allocate_in_proportion_to_final_attributes(
    #[case] region_name: &str,
    #[case] count: usize,
) {
    let path = test_util::temp::temp_path("attribute-batch-dense-allocation.h5");
    FileBuilder::new().write(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    let ops: Vec<_> = (0..count)
        .map(|index| AttrOp::Set {
            name: format!("a{index:04}"),
            value: AttrValue::I64(i64::try_from(index).unwrap()),
        })
        .collect();
    let ops: Vec<_> = ops.iter().collect();
    let (result, measured) = allocation::measure(region_name, || {
        super::super::plan_attr_ops(&engine.opened_source(), &region, &ops).unwrap()
    });
    eprintln!("{region_name}: {measured}");
    assert!(
        measured.bytes >= u64::try_from(count * 8).unwrap(),
        "{measured}"
    );
    assert!(
        measured.bytes < u64::try_from(count * 6_144).unwrap(),
        "{measured}"
    );
    assert!(result.region.is_empty());
    let EditedAttrs::Dense(dense) = result.attrs else {
        panic!("expected dense storage");
    };
    assert_eq!(dense.set.attrs.len(), count);
    assert!(dense.vl.is_empty());
    for (index, attribute) in dense.set.attrs.iter().enumerate() {
        assert_eq!(attribute.name, format!("a{index:04}"));
        assert_eq!(
            attribute.raw_data,
            i64::try_from(index).unwrap().to_le_bytes()
        );
    }
}
