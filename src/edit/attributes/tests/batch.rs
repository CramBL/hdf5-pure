use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
use test_util::object_header::v2::HeaderFlags;

use super::super::AttrOp;
use crate::attribute::AttributeMessage;
use crate::edit::object_header::OhRegion;
use crate::error::Error;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::type_builders::AttrValue;

#[rstest]
#[case::fixed(false)]
#[case::variable(true)]
fn compact_batches_assign_creation_indices_in_operation_order(#[case] variable: bool) {
    let layout = MessageRecordLayout::from_header_flags(
        (HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER).0,
    );
    let mut region = OhRegion::empty(
        crate::file_writer::WIDTHS,
        ObjectHeaderPrefix::with_layout(layout),
    );
    let info = AttributeInfoMessage {
        max_creation_index: Some(4),
        indexes_creation_order: true,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    region.push(
        MessageType::ATTRIBUTE_INFO,
        &info.serialize(crate::file_writer::OFFSET_WIDTH).unwrap(),
    );
    let value = |number| {
        if variable {
            AttrValue::VarLenAsciiString(format!("value{number}"))
        } else {
            AttrValue::I64(number)
        }
    };
    let operations = [
        AttrOp::Set {
            name: "first".into(),
            value: value(1),
        },
        AttrOp::Set {
            name: "second".into(),
            value: value(2),
        },
        AttrOp::Set {
            name: "first".into(),
            value: value(3),
        },
        AttrOp::Remove {
            name: "second".into(),
        },
        AttrOp::Set {
            name: "second".into(),
            value: value(4),
        },
    ];
    let mut batch = super::super::CompactBatch::parse(&region).unwrap();
    for (index, operation) in operations.iter().enumerate() {
        batch.apply(operation, index).unwrap();
    }
    let result = batch
        .into_compact(super::super::AttributePredecessors::default())
        .unwrap();
    let super::super::EditedAttrs::RebuildCompact(compact) = result.attrs else {
        panic!("expected compact storage");
    };
    let (_, _, info) = super::super::find_attribute_info(&result.region)
        .unwrap()
        .unwrap();
    assert_eq!(info.max_creation_index, Some(7));
    let mut actual = Vec::new();
    let mut position = 0;
    while let Some(record) = layout.next_message(&compact.records, position).unwrap() {
        let attribute = AttributeMessage::parse(record.body, 8).unwrap();
        actual.push((attribute.name, record.creation_index, attribute.raw_data));
        position = record.body_range.end;
    }
    assert_eq!(actual.len(), 2);
    assert_eq!((&*actual[0].0, actual[0].1), ("first", Some(4)));
    assert_eq!((&*actual[1].0, actual[1].1), ("second", Some(6)));
    if variable {
        assert_eq!(compact.fixups.len(), 2);
        for (fixup, attribute) in compact.fixups.iter().zip(&actual) {
            assert_eq!(&compact.records[fixup.range.clone()], attribute.2);
            assert_eq!(fixup.collections.len(), 1);
        }
    } else {
        assert!(compact.fixups.is_empty());
        assert_eq!(actual[0].2, 3i64.to_le_bytes());
        assert_eq!(actual[1].2, 4i64.to_le_bytes());
    }
}

#[rstest]
#[case::remove_before_invalid(ErrorSequence::RemoveBeforeInvalid)]
#[case::invalid_before_remove(ErrorSequence::InvalidBeforeRemove)]
#[case::remove_before_set(ErrorSequence::RemoveBeforeSet)]
fn compact_batch_reports_operation_errors_in_source_order(#[case] sequence: ErrorSequence) {
    let region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    let invalid = AttrOp::Set {
        name: String::new(),
        value: AttrValue::I64(1),
    };
    let remove = AttrOp::Remove {
        name: "missing".into(),
    };
    let set = AttrOp::Set {
        name: "missing".into(),
        value: AttrValue::I64(1),
    };
    let operations = match sequence {
        ErrorSequence::InvalidBeforeRemove => [&invalid, &remove],
        ErrorSequence::RemoveBeforeInvalid => [&remove, &invalid],
        ErrorSequence::RemoveBeforeSet => [&remove, &set],
    };
    let path = test_util::temp::temp_path("ordered-attribute-errors.h5");
    crate::FileBuilder::new().write(&path).unwrap();
    let engine = crate::edit::test_util::open_engine(&path);
    let Err(error) = super::super::plan_attr_ops(&engine.opened_source(), &region, &operations)
    else {
        panic!("invalid operations were accepted");
    };
    if matches!(sequence, ErrorSequence::InvalidBeforeRemove) {
        assert!(
            matches!(&error, Error::Format(FormatError::InvalidAttributeName { name }) if name.is_empty()),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(
                error,
                Error::EditUnsupported("attribute to remove was not found")
            ),
            "{error:?}"
        );
    }
}

#[rstest]
#[case::overwrite(false)]
#[case::new_attribute(true)]
fn compact_batch_checks_creation_exhaustion_only_for_new_names(#[case] new_name: bool) {
    let layout = MessageRecordLayout::from_header_flags(HeaderFlags::TRACKS_CREATION_ORDER.0);
    let mut region = OhRegion::empty(
        crate::file_writer::WIDTHS,
        ObjectHeaderPrefix::with_layout(layout),
    );
    region.push_bytes(&layout.record_with_creation_index(
        MessageType::ATTRIBUTE,
        MessageFlags::NONE,
        u16::MAX,
        &super::super::encode_attr_body("existing", &AttrValue::I64(1)).unwrap(),
    ));
    let operation = AttrOp::Set {
        name: if new_name { "new" } else { "existing" }.into(),
        value: AttrValue::I64(2),
    };
    let mut batch = super::super::CompactBatch::parse(&region).unwrap();
    if new_name {
        let error = batch.apply(&operation, 0).unwrap_err();
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "an object has assigned every attribute creation index its header can record"
                )
            ),
            "{error:?}"
        );
    } else {
        batch.apply(&operation, 0).unwrap();
        let result = batch
            .into_compact(super::super::AttributePredecessors::default())
            .unwrap();
        let super::super::EditedAttrs::RebuildCompact(compact) = result.attrs else {
            panic!("expected compact storage");
        };
        let record = layout.next_message(&compact.records, 0).unwrap().unwrap();
        assert_eq!(record.creation_index, Some(u16::MAX));
        assert_eq!(
            AttributeMessage::parse(record.body, 8).unwrap().raw_data,
            2i64.to_le_bytes()
        );
    }
}

#[rstest]
#[case::small(false)]
#[case::oversized(true)]
fn superseded_oversized_values_preserve_the_full_attribute_parsing_boundary(
    #[case] oversized: bool,
) {
    let path = test_util::temp::temp_path("superseded-attribute-boundary.h5");
    crate::FileBuilder::new().write(&path).unwrap();
    let engine = crate::edit::test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    let datatype = test_util::datatype::f64_le();
    let dataspace = test_util::dataspace::scalar();
    let mut body =
        test_util::attribute::Attribute::new("kept", &datatype, &dataspace, &1f64.to_le_bytes())
            .flags(test_util::attribute::Flags::NONE)
            .build();
    body.truncate(13);
    region.push(MessageType::ATTRIBUTE, &body);
    let set = AttrOp::Set {
        name: "temporary".into(),
        value: if oversized {
            AttrValue::I64Array(vec![0; 10_000])
        } else {
            AttrValue::I64(1)
        },
    };
    let remove = AttrOp::Remove {
        name: "temporary".into(),
    };
    let result = super::super::plan_attr_ops(&engine.opened_source(), &region, &[&set, &remove]);
    if oversized {
        let Err(error) = result else {
            panic!("a truncated datatype was accepted");
        };
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::UnexpectedEof {
                expected: 33,
                available: 13
            }
        );
    } else {
        let result = result.unwrap();
        assert_eq!(&*result.region, &*region);
        let super::super::EditedAttrs::RebuildCompact(compact) = result.attrs else {
            panic!("expected compact storage");
        };
        assert!(compact.records.is_empty());
        assert!(compact.fixups.is_empty());
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[derive(Clone, Copy, Debug)]
enum ErrorSequence {
    InvalidBeforeRemove,
    RemoveBeforeInvalid,
    RemoveBeforeSet,
}
